//! GPU backend on [wgpu]: WebGPU in the browser and Vulkan, Metal or DX12
//! natively (desktop and Android).
//!
//! The model code records a list of kernel launches on a [`Recorder`]
//! (convolutions, AdaIN, element-wise ops) over [`View`]s of pooled GPU
//! buffers, then [`Recorder::run`] encodes them into one command buffer,
//! submits it and reads the requested tensors back. Kernels are WGSL compute
//! shaders kept in [`shaders`]; everything else is Rust.

mod shaders;

pub use wgpu;

use crate::tensor::{Mat, Matrix, Storage};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};

pub type Result<T> = std::result::Result<T, String>;

const PARAM_SLOT: u64 = 256;
const PARAM_BYTES: u64 = 128;
const ACT_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

/// Blocks on a future (native targets, where wgpu resolves its futures
/// during `Device::poll`).
#[cfg(not(target_arch = "wasm32"))]
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    let mut cx = std::task::Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kernel {
    Conv(usize),
    Adain,
    AdainInPlace,
    DwUp2,
    Eltwise,
}

const KERNELS: [Kernel; 7] = [
    Kernel::Conv(0),
    Kernel::Conv(1),
    Kernel::Conv(2),
    Kernel::Adain,
    Kernel::AdainInPlace,
    Kernel::DwUp2,
    Kernel::Eltwise,
];

/// A GPU device with the compiled Inhouse TTS kernels.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Human-readable adapter description.
    pub name: String,
    /// True for software adapters (SwiftShader, lavapipe, WARP).
    pub software: bool,
    conv: [wgpu::ComputePipeline; 3],
    adain: wgpu::ComputePipeline,
    adain_in_place: wgpu::ComputePipeline,
    dwup2: wgpu::ComputePipeline,
    eltwise: wgpu::ComputePipeline,
    layouts: Vec<wgpu::BindGroupLayout>,
    pool: Mutex<Pool>,
}

/// Adapter selection.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuOptions {
    /// Accept software adapters (useful for tests; slow).
    pub allow_software: bool,
}

impl Gpu {
    /// Opens the default high-performance adapter and compiles the kernels.
    pub async fn new(opts: GpuOptions) -> Result<Gpu> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        if cfg!(target_arch = "wasm32") {
            desc.backends = wgpu::Backends::BROWSER_WEBGPU;
        } else if std::env::var_os("WGPU_BACKEND").is_none() {
            desc.backends = wgpu::Backends::PRIMARY;
        }
        let instance = wgpu::Instance::new(desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        let lower = info.name.to_lowercase();
        let software = info.device_type == wgpu::DeviceType::Cpu
            || [
                "swiftshader",
                "llvmpipe",
                "lavapipe",
                "basic render",
                "warp",
            ]
            .iter()
            .any(|n| lower.contains(n));
        if software && !opts.allow_software {
            return Err(
                format!("only a software adapter is available {}", info.name)
                    .trim_end()
                    .to_string(),
            );
        }
        let have = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: have.max_storage_buffer_binding_size.min(1 << 30),
            max_buffer_size: have.max_buffer_size.min(1 << 30),
            ..wgpu::Limits::downlevel_defaults()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("inhouse-tts"),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("cannot open the GPU: {e}"))?;
        let name = match (info.name.is_empty(), info.backend) {
            (true, wgpu::Backend::BrowserWebGpu) => "WebGPU".to_string(),
            (true, b) => format!("{b:?}"),
            (false, wgpu::Backend::BrowserWebGpu) => info.name.clone(),
            (false, b) => format!("{} ({b:?})", info.name),
        };
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let compile = |src: &str, label: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let mut gpu = Gpu {
            conv: [
                compile(&shaders::conv(0), "conv-f32"),
                compile(&shaders::conv(1), "conv-f16"),
                compile(&shaders::conv(2), "conv-q8"),
            ],
            adain: compile(&shaders::adain(false), "adain"),
            adain_in_place: compile(&shaders::adain(true), "adain-inplace"),
            dwup2: compile(shaders::DWUP2, "dwup2"),
            eltwise: compile(shaders::ELTWISE, "eltwise"),
            layouts: Vec::new(),
            device,
            queue,
            name,
            software,
            pool: Mutex::new(Pool::default()),
        };
        gpu.layouts = KERNELS
            .iter()
            .map(|k| gpu.pipeline(*k).get_bind_group_layout(0))
            .collect();
        if let Some(e) = scope.pop().await {
            return Err(format!("GPU kernels failed to compile: {e}"));
        }
        Ok(gpu)
    }

    fn pipeline(&self, k: Kernel) -> &wgpu::ComputePipeline {
        match k {
            Kernel::Conv(f) => &self.conv[f],
            Kernel::Adain => &self.adain,
            Kernel::AdainInPlace => &self.adain_in_place,
            Kernel::DwUp2 => &self.dwup2,
            Kernel::Eltwise => &self.eltwise,
        }
    }

    fn layout(&self, k: Kernel) -> &wgpu::BindGroupLayout {
        &self.layouts[KERNELS.iter().position(|x| *x == k).unwrap()]
    }

    fn buffer(&self, label: &str, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        // Bindings must not be empty.
        let pad = [0u8; 4];
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if bytes.is_empty() { &pad } else { bytes },
                usage,
            })
    }

    /// Uploads a read-only f32 vector.
    pub fn upload_f32(&self, label: &str, v: &[f32]) -> wgpu::Buffer {
        self.buffer(label, &f32_bytes(v), wgpu::BufferUsages::STORAGE)
    }

    /// Uploads a convolution weight (one matrix per phase, all `rows × cols`)
    /// with its bias.
    pub fn upload_conv(
        &self,
        label: &str,
        phases: &[&Matrix],
        bias: Option<&[f32]>,
        g: ConvGeom,
    ) -> Result<GpuConv> {
        let m = phases[0].rows;
        let k = phases[0].cols;
        let kpad = k.div_ceil(4) * 4;
        let mut words: Vec<u8> = Vec::new();
        let mut scales: Vec<f32> = Vec::new();
        let mut fmt = None;
        for mat in phases {
            assert_eq!((mat.rows, mat.cols), (m, k), "phase shapes differ");
            let st = mat.storage().ok_or(
                "weights were pre-packed for the CPU; load the model with LoadOptions::gpu",
            )?;
            let f = match st {
                Storage::F32(_) => 0,
                Storage::F16(_) => 1,
                Storage::Q8 { .. } => 2,
            };
            if *fmt.get_or_insert(f) != f {
                return Err(format!("{label}: mixed weight formats"));
            }
            for r in 0..m {
                match st {
                    Storage::F32(v) => {
                        words.extend(f32_bytes(&v[r * k..(r + 1) * k]));
                        words.extend(std::iter::repeat_n(0u8, (kpad - k) * 4));
                    }
                    Storage::F16(v) => {
                        for h in &v[r * k..(r + 1) * k] {
                            words.extend(h.to_le_bytes());
                        }
                        words.extend(std::iter::repeat_n(0u8, (kpad - k) * 2));
                    }
                    Storage::Q8 { q, .. } => {
                        words.extend(q[r * k..(r + 1) * k].iter().map(|&v| v as u8));
                        words.extend(std::iter::repeat_n(0u8, kpad - k));
                    }
                }
            }
            if let Storage::Q8 { scales: s, .. } = st {
                scales.extend_from_slice(s);
            }
        }
        let fmt = fmt.unwrap_or(0);
        let wphase = match fmt {
            0 => m * kpad,
            1 => m * kpad / 2,
            _ => m * kpad / 4,
        };
        let boff = scales.len();
        let mut aux = scales;
        if let Some(b) = bias {
            aux.extend_from_slice(b);
        }
        let max = self.device.limits().max_storage_buffer_binding_size as usize;
        if words.len() > max {
            return Err(format!("{label}: weights exceed the GPU binding limit"));
        }
        Ok(GpuConv {
            w: self.buffer(label, &words, wgpu::BufferUsages::STORAGE),
            aux: self.upload_f32(label, &aux),
            fmt,
            m,
            k,
            kpad,
            phases: phases.len(),
            wphase,
            boff,
            has_bias: bias.is_some(),
            geom: g,
        })
    }

    /// Starts recording a batch of kernel launches.
    pub fn recorder(&self) -> Recorder<'_> {
        Recorder {
            gpu: self,
            ops: Vec::new(),
            params: Vec::new(),
            taken: HashMap::new(),
            next_id: 0,
        }
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Geometry of a convolution (`taps` along time).
#[derive(Clone, Copy, Debug)]
pub struct ConvGeom {
    pub cin: usize,
    pub taps: usize,
    pub stride: usize,
    pub pad: usize,
    pub dil: usize,
}

/// Convolution weights resident on the GPU.
pub struct GpuConv {
    w: wgpu::Buffer,
    aux: wgpu::Buffer,
    fmt: usize,
    /// Output channels.
    pub m: usize,
    k: usize,
    kpad: usize,
    phases: usize,
    wphase: usize,
    boff: usize,
    has_bias: bool,
    pub geom: ConvGeom,
}

impl GpuConv {
    pub fn out_len(&self, t: usize) -> usize {
        let g = &self.geom;
        let span = g.dil * (g.taps - 1) + 1;
        let padded = t + 2 * g.pad;
        if padded < span {
            0
        } else {
            (padded - span) / g.stride + 1
        }
    }
}

/// Options of a convolution launch.
#[derive(Clone, Copy, Debug)]
pub struct ConvRun {
    /// Add the result to the existing output.
    pub accumulate: bool,
    /// Scale applied after accumulation.
    pub scale: f32,
    /// LeakyReLU slope applied to the input while loading it.
    pub leaky_in: Option<f32>,
    /// Nearest-neighbour upsampling of the input.
    pub up: usize,
}

impl Default for ConvRun {
    fn default() -> Self {
        ConvRun {
            accumulate: false,
            scale: 1.0,
            leaky_in: None,
            up: 1,
        }
    }
}

/// Transposed convolution launch (polyphase weights, one per stride phase).
#[derive(Clone, Copy, Debug)]
pub struct ConvTRun {
    pub kernel: usize,
    pub stride: usize,
    pub pad: usize,
    pub accumulate: bool,
    pub leaky_in: Option<f32>,
    /// Write column `o` to `o + 1` and duplicate column 1 into column 0
    /// (`ReflectionPad1d((1, 0))`).
    pub reflect_left: bool,
}

/// Activation fused into the AdaIN kernel.
#[derive(Clone, Copy, Debug)]
pub enum Act {
    None,
    Leaky(f32),
    /// Snake with per-channel α taken from the style buffer.
    Snake,
}

/// A `[rows, cols]` region of a GPU buffer with row stride `ld` (elements).
#[derive(Clone)]
pub struct View {
    buf: wgpu::Buffer,
    id: u64,
    pub off: usize,
    pub ld: usize,
    pub rows: usize,
    pub cols: usize,
}

impl View {
    /// Rows `r0 .. r0 + n` of this view.
    pub fn rows(&self, r0: usize, n: usize) -> View {
        assert!(r0 + n <= self.rows, "row range out of bounds");
        View {
            off: self.off + r0 * self.ld,
            rows: n,
            ..self.clone()
        }
    }

    fn contiguous(&self) -> bool {
        self.ld == self.cols || self.rows <= 1
    }
}

#[derive(Default)]
struct Pool {
    free: HashMap<u64, Vec<(wgpu::Buffer, u64)>>,
    run: u64,
}

fn size_class(bytes: u64) -> u64 {
    let b = bytes.max(4096);
    let step = 1u64 << (63 - b.leading_zeros()).saturating_sub(2);
    b.div_ceil(step) * step
}

enum Op {
    Dispatch {
        kernel: Kernel,
        bufs: Vec<wgpu::Buffer>,
        slot: u64,
        groups: [u32; 3],
    },
    Write {
        buf: wgpu::Buffer,
        offset: u64,
        data: Vec<u8>,
    },
}

/// Records kernel launches; see the module docs.
pub struct Recorder<'g> {
    gpu: &'g Gpu,
    ops: Vec<Op>,
    params: Vec<u8>,
    taken: HashMap<u64, (wgpu::Buffer, u64)>,
    next_id: u64,
}

impl Recorder<'_> {
    /// Allocates an uninitialised `[rows, cols]` tensor from the pool.
    pub fn alloc(&mut self, rows: usize, cols: usize) -> View {
        let bytes = (rows * cols * 4) as u64;
        let class = size_class(bytes);
        let buf = {
            let mut pool = self.gpu.pool.lock().unwrap();
            pool.free.get_mut(&class).and_then(Vec::pop).map(|(b, _)| b)
        }
        .unwrap_or_else(|| {
            self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("activation"),
                size: class,
                usage: ACT_USAGE,
                mapped_at_creation: false,
            })
        });
        self.next_id += 1;
        self.taken.insert(self.next_id, (buf.clone(), class));
        View {
            buf,
            id: self.next_id,
            off: 0,
            ld: cols,
            rows,
            cols,
        }
    }

    /// Returns a tensor's buffer to the pool for later launches of this batch.
    pub fn free(&mut self, v: View) {
        if let Some((buf, class)) = self.taken.remove(&v.id) {
            let mut pool = self.gpu.pool.lock().unwrap();
            let run = pool.run;
            pool.free.entry(class).or_default().push((buf, run));
        }
    }

    /// Uploads host data into a view (executed before the batch's kernels).
    pub fn write(&mut self, v: &View, data: &[f32]) {
        assert!(
            v.contiguous() && data.len() == v.rows * v.cols,
            "write shape"
        );
        if data.is_empty() {
            return;
        }
        self.ops.push(Op::Write {
            buf: v.buf.clone(),
            offset: (v.off * 4) as u64,
            data: f32_bytes(data),
        });
    }

    /// Allocates and uploads a matrix.
    pub fn upload(&mut self, m: &Mat) -> View {
        let v = self.alloc(m.rows, m.cols);
        self.write(&v, &m.data);
        v
    }

    fn push(&mut self, kernel: Kernel, p: [u32; 32], bufs: Vec<wgpu::Buffer>, groups: [u32; 3]) {
        if groups.contains(&0) {
            return;
        }
        let slot = self.params.len() as u64;
        self.params.extend(p.iter().flat_map(|v| v.to_le_bytes()));
        self.params.resize((slot + PARAM_SLOT) as usize, 0);
        self.ops.push(Op::Dispatch {
            kernel,
            bufs,
            slot,
            groups,
        });
    }

    /// `y (+)= conv(x)`; output length is `conv.out_len(x.cols · up)`.
    pub fn conv(&mut self, c: &GpuConv, x: &View, y: &View, r: ConvRun) {
        let g = c.geom;
        assert_eq!(x.rows, g.cin, "conv input channels");
        assert_eq!(c.phases, 1);
        let n = c.out_len(x.cols * r.up);
        assert!(y.rows == c.m && y.cols >= n, "conv output shape");
        self.launch(
            c,
            x,
            y,
            n,
            r.up,
            g.stride,
            g.dil,
            g.pad as i32,
            1,
            0,
            n,
            0,
            r.accumulate,
            r.scale,
            r.leaky_in,
            false,
        );
    }

    /// `y (+)= conv_transpose(x)` from polyphase weights.
    pub fn conv_t(&mut self, c: &GpuConv, x: &View, y: &View, r: ConvTRun) {
        let taps = c.geom.taps;
        assert_eq!(c.phases, r.stride, "one weight matrix per phase");
        let t = x.cols;
        let tout = (t - 1) * r.stride + r.kernel - 2 * r.pad;
        let shift = usize::from(r.reflect_left);
        assert!(
            y.rows == c.m && y.cols >= tout + shift,
            "conv_t output shape"
        );
        let n = t + taps - 1;
        self.launch(
            c,
            x,
            y,
            n,
            1,
            1,
            1,
            taps as i32 - 1,
            r.stride,
            -(r.pad as i32),
            tout,
            shift,
            r.accumulate,
            1.0,
            r.leaky_in,
            r.reflect_left,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &mut self,
        c: &GpuConv,
        x: &View,
        y: &View,
        n: usize,
        up: usize,
        stride: usize,
        dil: usize,
        pad: i32,
        ostride: usize,
        oshift: i32,
        tvalid: usize,
        dshift: usize,
        accumulate: bool,
        scale: f32,
        leaky_in: Option<f32>,
        reflect: bool,
    ) {
        let mut flags = 0;
        if accumulate {
            flags |= 1;
        }
        if c.has_bias {
            flags |= 2;
        }
        if leaky_in.is_some() {
            flags |= 4;
        }
        if reflect {
            flags |= 8;
        }
        if c.fmt == 2 {
            flags |= 16;
        }
        let mut p = [0u32; 32];
        let vals: [u32; 24] = [
            c.m as u32,
            n as u32,
            (c.geom.cin * c.geom.taps) as u32,
            c.kpad as u32,
            c.geom.taps as u32,
            stride as u32,
            dil as u32,
            pad as u32,
            x.cols as u32,
            up as u32,
            x.ld as u32,
            x.off as u32,
            y.ld as u32,
            y.off as u32,
            ostride as u32,
            oshift as u32,
            tvalid as u32,
            dshift as u32,
            flags,
            c.wphase as u32,
            c.m as u32,
            c.boff as u32,
            leaky_in.unwrap_or(0.0).to_bits(),
            scale.to_bits(),
        ];
        debug_assert_eq!(c.k, c.geom.cin * c.geom.taps);
        p[..24].copy_from_slice(&vals);
        let groups = [
            n.div_ceil(64) as u32,
            c.m.div_ceil(64) as u32,
            c.phases as u32,
        ];
        let bufs = vec![c.w.clone(), c.aux.clone(), x.buf.clone(), y.buf.clone()];
        self.push(Kernel::Conv(c.fmt), p, bufs, groups);
    }

    /// `y = act(AdaIN(x))`, or in place when `y` is `None`. `soff` points at
    /// `[γ | β | α]` in the style buffer `st`.
    pub fn adain(&mut self, x: &View, y: Option<&View>, st: &wgpu::Buffer, soff: usize, act: Act) {
        let out = y.unwrap_or(x);
        assert_eq!((x.rows, x.cols), (out.rows, out.cols), "adain shapes");
        let (a, slope) = match act {
            Act::None => (0, 0.0),
            Act::Leaky(s) => (1, s),
            Act::Snake => (2, 0.0),
        };
        let mut p = [0u32; 32];
        p[..10].copy_from_slice(&[
            x.rows as u32,
            x.cols as u32,
            x.ld as u32,
            x.off as u32,
            out.ld as u32,
            out.off as u32,
            soff as u32,
            a,
            f32::to_bits(slope),
            f32::to_bits(1e-5),
        ]);
        let groups = [x.rows as u32, 1, 1];
        match y {
            Some(y) => self.push(
                Kernel::Adain,
                p,
                vec![st.clone(), x.buf.clone(), y.buf.clone()],
                groups,
            ),
            None => self.push(
                Kernel::AdainInPlace,
                p,
                vec![st.clone(), x.buf.clone()],
                groups,
            ),
        }
    }

    /// Depthwise ×2 transposed convolution (`wts` = `[w0 w1 w2]·c ‖ bias·c`).
    pub fn dwup2(&mut self, x: &View, y: &View, wts: &wgpu::Buffer) {
        assert!(y.rows == x.rows && y.cols == 2 * x.cols, "dwup2 shapes");
        let mut p = [0u32; 32];
        p[..6].copy_from_slice(&[
            x.rows as u32,
            x.cols as u32,
            x.ld as u32,
            x.off as u32,
            y.ld as u32,
            y.off as u32,
        ]);
        let groups = [x.cols.div_ceil(256) as u32, x.rows as u32, 1];
        self.push(
            Kernel::DwUp2,
            p,
            vec![wts.clone(), x.buf.clone(), y.buf.clone()],
            groups,
        );
    }

    /// `y = (x (+ y)) · scale` for contiguous tensors of equal size.
    pub fn eltwise(&mut self, x: &View, y: &View, accumulate: bool, scale: f32) {
        assert!(x.contiguous() && y.contiguous() && x.rows * x.cols == y.rows * y.cols);
        let n = x.rows * x.cols;
        let mut p = [0u32; 32];
        p[..5].copy_from_slice(&[
            n as u32,
            x.off as u32,
            y.off as u32,
            u32::from(accumulate),
            scale.to_bits(),
        ]);
        let groups = [n.div_ceil(256).min(4096) as u32, 1, 1];
        self.push(
            Kernel::Eltwise,
            p,
            vec![x.buf.clone(), y.buf.clone()],
            groups,
        );
    }

    /// Encodes and submits the batch, then reads back `outputs`.
    pub async fn run(self, outputs: &[&View]) -> Result<Vec<Vec<f32>>> {
        let gpu = self.gpu;
        let dev = &gpu.device;
        let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
        let oom = dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let params = gpu.buffer(
            "params",
            &self.params,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("inhouse-tts"),
        });
        let mut i = 0;
        while i < self.ops.len() {
            if let Op::Write { buf, offset, data } = &self.ops[i] {
                // Uploads are ordered with the kernels through a copy.
                let src = gpu.buffer("upload", data, wgpu::BufferUsages::COPY_SRC);
                enc.copy_buffer_to_buffer(&src, 0, buf, *offset, data.len() as u64);
                i += 1;
                continue;
            }
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inhouse-tts"),
                timestamp_writes: None,
            });
            while let Some(Op::Dispatch {
                kernel,
                bufs,
                slot,
                groups,
            }) = self.ops.get(i)
            {
                let mut entries = vec![wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: *slot,
                        size: wgpu::BufferSize::new(PARAM_BYTES),
                    }),
                }];
                for (j, b) in bufs.iter().enumerate() {
                    entries.push(wgpu::BindGroupEntry {
                        binding: j as u32 + 1,
                        resource: b.as_entire_binding(),
                    });
                }
                let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: gpu.layout(*kernel),
                    entries: &entries,
                });
                pass.set_pipeline(gpu.pipeline(*kernel));
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                i += 1;
            }
        }
        let mut staging = Vec::new();
        for v in outputs {
            assert!(v.contiguous(), "read-back views must be contiguous");
            let bytes = (v.rows * v.cols * 4) as u64;
            let s = dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("read-back"),
                size: bytes.max(4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if bytes > 0 {
                enc.copy_buffer_to_buffer(&v.buf, (v.off * 4) as u64, &s, 0, bytes);
            }
            staging.push((s, bytes));
        }
        gpu.queue.submit([enc.finish()]);
        // Every buffer of this batch can serve the next one (queue order).
        {
            let mut pool = gpu.pool.lock().unwrap();
            pool.run += 1;
            let run = pool.run;
            for (_, (buf, class)) in self.taken {
                pool.free.entry(class).or_default().push((buf, run));
            }
            // Drop buffers unused for two batches.
            for list in pool.free.values_mut() {
                list.retain(|(_, r)| *r + 2 > run);
            }
        }
        let mut out = Vec::with_capacity(staging.len());
        for (s, bytes) in &staging {
            if *bytes == 0 {
                out.push(Vec::new());
                continue;
            }
            map_read(dev, s).await?;
            let data = s
                .slice(..)
                .get_mapped_range()
                .map_err(|e| e.to_string())?
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            s.unmap();
            out.push(data);
        }
        if let Some(e) = oom.pop().await {
            return Err(format!("GPU out of memory: {e}"));
        }
        if let Some(e) = scope.pop().await {
            return Err(format!("GPU error: {e}"));
        }
        Ok(out)
    }
}

async fn map_read(dev: &wgpu::Device, buf: &wgpu::Buffer) -> Result<()> {
    type Slot = (
        Option<std::result::Result<(), wgpu::BufferAsyncError>>,
        Option<Waker>,
    );
    let slot: Arc<Mutex<Slot>> = Arc::new(Mutex::new((None, None)));
    let s2 = slot.clone();
    buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let mut g = s2.lock().unwrap();
        g.0 = Some(r);
        if let Some(w) = g.1.take() {
            w.wake();
        }
    });
    #[cfg(not(target_arch = "wasm32"))]
    dev.poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| e.to_string())?;
    #[cfg(target_arch = "wasm32")]
    let _ = dev;
    std::future::poll_fn(|cx| {
        let mut g = slot.lock().unwrap();
        match g.0.take() {
            Some(r) => Poll::Ready(r.map_err(|e| format!("GPU read-back failed: {e}"))),
            None => {
                g.1 = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{Conv1d, ConvTranspose1d};

    fn gpu() -> Option<Gpu> {
        match block_on(Gpu::new(GpuOptions {
            allow_software: true,
        })) {
            Ok(g) => {
                eprintln!("GPU: {}", g.name);
                Some(g)
            }
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    fn rand(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(747_796_405).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    fn max_err(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn conv_formats_match_cpu() {
        let Some(g) = gpu() else { return };
        for (fmt, (cin, cout, k, stride, pad, dil, t)) in [
            (0, (22, 70, 7, 1, 3, 1, 333)),
            (1, (64, 130, 3, 1, 3, 3, 100)),
            (2, (22, 256, 12, 6, 3, 1, 601)),
            (2, (1090, 64, 1, 1, 0, 1, 77)),
        ] {
            let w = rand(cout * cin * k, 3);
            let storage = match fmt {
                0 => Storage::F32(w.clone()),
                1 => Storage::F16(w.iter().map(|&v| crate::gemm::f32_to_f16(v)).collect()),
                _ => {
                    let (q, scales) = crate::weights::quantize_rows(&w, cout, cin * k);
                    Storage::Q8 { q, scales }
                }
            };
            let conv = Conv1d {
                cin,
                cout,
                k,
                stride,
                pad,
                dil,
                w: Matrix::new(cout, cin * k, storage, false),
                bias: Some(rand(cout, 5)),
            };
            let x = Mat::from_vec(cin, t, rand(cin * t, 7));
            let want = conv.forward(&x);
            let gc = g
                .upload_conv(
                    "test",
                    &[&conv.w],
                    conv.bias.as_deref(),
                    ConvGeom {
                        cin,
                        taps: k,
                        stride,
                        pad,
                        dil,
                    },
                )
                .unwrap();
            let mut r = g.recorder();
            let xv = r.upload(&x);
            let yv = r.alloc(cout, want.cols);
            r.conv(&gc, &xv, &yv, ConvRun::default());
            let got = block_on(r.run(&[&yv])).unwrap();
            let e = max_err(&got[0], &want.data);
            assert!(e < 2e-4, "format {fmt}: max error {e}");
        }
    }

    #[test]
    fn conv_transpose_matches_cpu() {
        let Some(g) = gpu() else { return };
        let (cin, cout, k, s, pad, t) = (40, 24, 12, 6, 3, 37);
        let w = rand(cin * cout * k, 11);
        let phases = crate::ops::polyphase_weights(&w, cin, cout, k, s);
        let ct = ConvTranspose1d {
            cin,
            cout,
            k,
            stride: s,
            pad,
            out_pad: 0,
            phases: phases
                .iter()
                .map(|(a, taps)| (Matrix::from_f32(cout, cin * taps, a.clone(), false), *taps))
                .collect(),
            bias: rand(cout, 13),
        };
        let x = Mat::from_vec(cin, t, rand(cin * t, 17));
        let mut xl = x.clone();
        crate::math::leaky_relu_inplace(&mut xl.data, 0.1);
        let y = ct.forward(&xl);
        // Expected: reflection-padded transposed conv added to a residual.
        let res = Mat::from_vec(cout, y.cols + 1, rand(cout * (y.cols + 1), 19));
        let mut want = res.clone();
        for c in 0..cout {
            let row = want.row_mut(c);
            row[0] += y.at(c, 1);
            for o in 0..y.cols {
                row[o + 1] += y.at(c, o);
            }
        }
        let mats: Vec<&Matrix> = ct.phases.iter().map(|(m, _)| m).collect();
        let gc = g
            .upload_conv(
                "ct",
                &mats,
                Some(&ct.bias),
                ConvGeom {
                    cin,
                    taps: ct.phases[0].1,
                    stride: 1,
                    pad: 0,
                    dil: 1,
                },
            )
            .unwrap();
        let mut r = g.recorder();
        let xv = r.upload(&x);
        let yv = r.upload(&res);
        r.conv_t(
            &gc,
            &xv,
            &yv,
            ConvTRun {
                kernel: k,
                stride: s,
                pad,
                accumulate: true,
                leaky_in: Some(0.1),
                reflect_left: true,
            },
        );
        let got = block_on(r.run(&[&yv])).unwrap();
        let e = max_err(&got[0], &want.data);
        assert!(e < 1e-4, "max error {e}");
    }

    #[test]
    fn adain_snake_matches_cpu() {
        let Some(g) = gpu() else { return };
        let (c, t) = (37, 1000);
        let x = Mat::from_vec(
            c,
            t,
            rand(c * t, 23).iter().map(|v| v * 3.0 + 0.7).collect(),
        );
        let gamma = rand(c, 29);
        let beta = rand(c, 31);
        let alpha: Vec<f32> = rand(c, 37).iter().map(|v| v + 1.0).collect();
        let mut want = x.clone();
        for r in 0..c {
            let row = want.row_mut(r);
            let n = t as f64;
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            let inv = 1.0 / ((var as f32) + 1e-5).sqrt();
            for v in row.iter_mut() {
                *v = (*v - mean as f32) * inv * (1.0 + gamma[r]) + beta[r];
            }
            crate::math::snake_inplace(row, alpha[r]);
        }
        let st: Vec<f32> = [gamma, beta, alpha].concat();
        let sb = g.upload_f32("st", &st);
        for in_place in [false, true] {
            let mut r = g.recorder();
            let xv = r.upload(&x);
            let yv = if in_place {
                xv.clone()
            } else {
                let y = r.alloc(c, t);
                r.adain(&xv, Some(&y), &sb, 0, Act::Snake);
                y
            };
            if in_place {
                r.adain(&xv, None, &sb, 0, Act::Snake);
            }
            let got = block_on(r.run(&[&yv])).unwrap();
            let e = max_err(&got[0], &want.data);
            assert!(e < 5e-5, "in_place={in_place}: max error {e}");
        }
    }
}
