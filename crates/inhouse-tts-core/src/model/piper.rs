//! Piper voices (rhasspy/piper-voices, MIT): VITS inference on our engine.
//!
//! [`convert`] reads a Piper `.onnx` export plus its `.onnx.json` config and
//! writes an `.ihw` container (model `"piper"`) with clean tensor names, the
//! phoneme/speaker tables and the hyper-parameters inferred from the graph.
//! [`Piper`] runs the network: relative-position transformer text encoder,
//! stochastic duration predictor (reverse flows with a rational-quadratic
//! spline), mean-only residual coupling flow and the HiFi-GAN generator.
//! No ONNX runtime is involved; the graph is re-implemented with the
//! engine's GEMM and convolution kernels.

use crate::dsp::Rng;
use crate::gemm::{self, AData, AMat, BMat, Im2col};
use crate::math;
use crate::onnx;
use crate::ops::{self, Conv1d};
use crate::par;
use crate::profile::Profiler;
use crate::tensor::{Mat, Matrix};
use crate::weights::{err, DType, Error, Result, WeightFile, Writer};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Hyper-parameters of a Piper network (inferred from tensor shapes and
/// convolution attributes at conversion time).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PiperConfig {
    pub n_vocab: usize,
    pub hidden: usize,
    pub inter: usize,
    pub filter: usize,
    pub heads: usize,
    pub d_k: usize,
    /// Relative attention window (offsets `-window..=window`).
    pub window: usize,
    pub enc_layers: usize,
    pub ffn_kernel: usize,
    pub dp_filter: usize,
    pub dp_kernel: usize,
    pub dp_layers: usize,
    /// Conv-flow indices of the duration predictor, in reverse order.
    pub dp_flows: Vec<usize>,
    pub dp_flow_filter: usize,
    pub dp_flow_layers: usize,
    pub spline_bins: usize,
    pub spline_bound: f64,
    /// Coupling-layer indices of the flow, in reverse order.
    pub flows: Vec<usize>,
    pub flow_hidden: usize,
    pub flow_kernel: usize,
    pub flow_dilations: Vec<usize>,
    /// Speaker embedding size (0 for single-speaker voices).
    pub gin: usize,
    /// HiFi-GAN residual block type, `"1"` or `"2"`.
    pub resblock: String,
    pub resblock_kernels: Vec<usize>,
    pub resblock_dilations: Vec<Vec<usize>>,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernels: Vec<usize>,
    pub upsample_pads: Vec<usize>,
    pub upsample_initial: usize,
    pub pre_kernel: usize,
    pub post_kernel: usize,
    pub post_bias: bool,
}

/// Synthesis options (Piper's `scales` input plus a noise seed).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PiperOptions {
    pub noise_scale: f32,
    pub length_scale: f32,
    pub noise_w: f32,
    pub seed: u64,
}

impl Default for PiperOptions {
    fn default() -> Self {
        PiperOptions {
            noise_scale: 0.667,
            length_scale: 1.0,
            noise_w: 0.8,
            seed: 1,
        }
    }
}

pub struct PiperOutput {
    pub audio: Vec<f32>,
    /// Frames per input id (one frame = product of the upsample rates).
    pub durations: Vec<usize>,
}

fn json_err(what: &str, e: serde_json::Error) -> Error {
    Error(format!("{what}: {e}"))
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

/// Prefixes of the network's own tensors (graph constants are dropped).
const PREFIXES: [&str; 5] = ["enc_p.", "dp.", "flow.", "dec.", "emb_g."];

/// Matrices stored as int8 with `DType::Q8`; the rest of the network stays
/// in f16. Waveform SNR against f32 (davefx-medium): all f16 43 dB, plus
/// int8 generator 29 dB, int8 flow 21 dB, int8 text encoder changes the
/// durations. The generator is only ~1.4 M parameters, so f16 is the
/// recommended storage and `Q8` saves little on top of it.
fn q8_tensor(name: &str) -> bool {
    name.starts_with("dec.resblocks.") || name.starts_with("dec.ups.")
}

/// Convolution attributes recorded by module name.
#[derive(Clone, Debug, Default)]
struct ConvAttrs {
    dilation: usize,
    stride: usize,
    pad: usize,
}

/// `"/flow/flows.0/enc/in_layers.0/Conv"` → `"flow.flows.0.enc.in_layers.0"`.
fn module_name(node: &str) -> String {
    let m = node.rsplit_once('/').map_or(node, |(m, _)| m);
    m.trim_start_matches('/').replace('/', ".")
}

struct Tensors {
    map: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    attrs: HashMap<String, ConvAttrs>,
}

impl Tensors {
    fn shape(&self, name: &str) -> Result<&[usize]> {
        self.map
            .get(name)
            .map(|(s, _)| s.as_slice())
            .ok_or_else(|| Error(format!("piper: missing tensor {name}")))
    }

    fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// Number of consecutive indices `i` for which `pattern(i)` exists.
    fn count(&self, pattern: impl Fn(usize) -> String) -> usize {
        (0..).take_while(|&i| self.has(&pattern(i))).count()
    }

    /// Existing indices `i < 32` of `pattern(i)`, highest first.
    fn indices_desc(&self, pattern: impl Fn(usize) -> String) -> Vec<usize> {
        (0..32).rev().filter(|&i| self.has(&pattern(i))).collect()
    }

    fn attrs(&self, module: &str) -> Result<&ConvAttrs> {
        self.attrs
            .get(module)
            .ok_or_else(|| Error(format!("piper: no convolution node for {module}")))
    }

    fn infer_config(&self) -> Result<PiperConfig> {
        let emb = self.shape("enc_p.emb.weight")?;
        let (n_vocab, hidden) = (emb[0], emb[1]);
        let rel = self.shape("enc_p.encoder.attn_layers.0.emb_rel_k")?;
        if rel.len() != 3 || rel[0] != 1 {
            return err(format!("piper: unsupported relative embedding {rel:?}"));
        }
        let (window, d_k) = ((rel[1] - 1) / 2, rel[2]);
        let ffn = self.shape("enc_p.encoder.ffn_layers.0.conv_1.weight")?;
        let dp_flows = self.indices_desc(|i| format!("dp.flows.{i}.pre.weight"));
        let Some(&f0) = dp_flows.first() else {
            return err("piper: duration predictor has no conv flows");
        };
        let flows = self.indices_desc(|i| format!("flow.flows.{i}.pre.weight"));
        let Some(&fl0) = flows.first() else {
            return err("piper: no coupling layers");
        };
        let flow_layers = self.count(|i| format!("flow.flows.{fl0}.enc.in_layers.{i}.weight"));
        let flow_dilations = (0..flow_layers)
            .map(|i| {
                Ok(self
                    .attrs(&format!("flow.flows.{fl0}.enc.in_layers.{i}"))?
                    .dilation)
            })
            .collect::<Result<Vec<_>>>()?;
        let nups = self.count(|i| format!("dec.ups.{i}.weight"));
        let nres = self.count(|i| format!("dec.resblocks.{i}.convs.0.weight"))
            + self.count(|i| format!("dec.resblocks.{i}.convs1.0.weight"));
        if nups == 0 || nres == 0 || !nres.is_multiple_of(nups) {
            return err(format!("piper: {nups} upsamplers, {nres} resblocks"));
        }
        let resblock = if self.has("dec.resblocks.0.convs1.0.weight") {
            "1"
        } else {
            "2"
        };
        let conv = if resblock == "1" { "convs1" } else { "convs" };
        let nk = nres / nups;
        let mut resblock_kernels = Vec::new();
        let mut resblock_dilations = Vec::new();
        for j in 0..nk {
            resblock_kernels.push(self.shape(&format!("dec.resblocks.{j}.{conv}.0.weight"))?[2]);
            let n = self.count(|c| format!("dec.resblocks.{j}.{conv}.{c}.weight"));
            resblock_dilations.push(
                (0..n)
                    .map(|c| {
                        Ok(self
                            .attrs(&format!("dec.resblocks.{j}.{conv}.{c}"))?
                            .dilation)
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        let mut upsample_rates = Vec::new();
        let mut upsample_kernels = Vec::new();
        let mut upsample_pads = Vec::new();
        for i in 0..nups {
            let a = self.attrs(&format!("dec.ups.{i}"))?;
            upsample_rates.push(a.stride);
            upsample_pads.push(a.pad);
            upsample_kernels.push(self.shape(&format!("dec.ups.{i}.weight"))?[2]);
        }
        let pre = self.shape("dec.conv_pre.weight")?;
        let proj = self.shape(&format!("dp.flows.{f0}.proj.weight"))?;
        Ok(PiperConfig {
            n_vocab,
            hidden,
            inter: self.shape("enc_p.proj.weight")?[0] / 2,
            filter: ffn[0],
            heads: hidden / d_k,
            d_k,
            window,
            enc_layers: self.count(|i| format!("enc_p.encoder.attn_layers.{i}.conv_q.weight")),
            ffn_kernel: ffn[2],
            dp_filter: self.shape("dp.pre.weight")?[0],
            dp_kernel: self.shape("dp.convs.convs_sep.0.weight")?[2],
            dp_layers: self.count(|i| format!("dp.convs.convs_sep.{i}.weight")),
            dp_flow_filter: self.shape(&format!("dp.flows.{f0}.pre.weight"))?[0],
            dp_flow_layers: self.count(|i| format!("dp.flows.{f0}.convs.convs_sep.{i}.weight")),
            dp_flows,
            spline_bins: (proj[0] + 1) / 3,
            spline_bound: 5.0,
            flow_hidden: self.shape(&format!("flow.flows.{fl0}.pre.weight"))?[0],
            flow_kernel: self.shape(&format!("flow.flows.{fl0}.enc.in_layers.0.weight"))?[2],
            flow_dilations,
            flows,
            gin: self.map.get("emb_g.weight").map_or(0, |(s, _)| s[1]),
            resblock: resblock.to_string(),
            resblock_kernels,
            resblock_dilations,
            upsample_rates,
            upsample_kernels,
            upsample_pads,
            upsample_initial: pre[0],
            pre_kernel: pre[2],
            post_kernel: self.shape("dec.conv_post.weight")?[2],
            post_bias: self.has("dec.conv_post.bias"),
        })
    }
}

/// Converts a Piper ONNX voice (`.onnx` + `.onnx.json`) into an `.ihw`
/// container. `dtype` is the storage of the large weight matrices: `F32`
/// keeps everything exact, `F16` halves the size, `Q8` additionally stores
/// the generator's residual blocks and upsamplers as int8.
pub fn convert(onnx_bytes: &[u8], config_json: &[u8], dtype: DType) -> Result<Vec<u8>> {
    let model = onnx::Model::parse(onnx_bytes)?;
    let cfg: serde_json::Value =
        serde_json::from_slice(config_json).map_err(|e| json_err("piper config", e))?;
    let init: HashMap<&str, &onnx::Tensor> = model
        .initializers
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();
    let float = |name: &str| -> Option<(Vec<usize>, Vec<f32>)> {
        let t = init.get(name)?;
        Some((t.dims.clone(), t.floats().ok()?.to_vec()))
    };
    let mut t = Tensors {
        map: BTreeMap::new(),
        attrs: HashMap::new(),
    };
    for i in &model.initializers {
        if PREFIXES.iter().any(|p| i.name.starts_with(p)) {
            if let Ok(v) = i.floats() {
                t.map.insert(i.name.clone(), (i.dims.clone(), v.to_vec()));
            }
        }
    }
    // Anonymous (weight-norm folded) convolution weights take the name of
    // their Conv node; the embeddings take the name of their Gather.
    for n in &model.nodes {
        match n.op_type.as_str() {
            "Conv" | "ConvTranspose" => {
                let m = module_name(&n.name);
                let first = |a: &str| {
                    n.ints(a)
                        .and_then(|v| v.first().copied())
                        .unwrap_or(if a == "pads" { 0 } else { 1 }) as usize
                };
                t.attrs.insert(
                    m.clone(),
                    ConvAttrs {
                        dilation: first("dilations"),
                        stride: first("strides"),
                        pad: first("pads"),
                    },
                );
                for (slot, suffix) in [(1, "weight"), (2, "bias")] {
                    let Some(input) = n.inputs.get(slot).filter(|s| !s.is_empty()) else {
                        continue;
                    };
                    let key = format!("{m}.{suffix}");
                    if !t.has(&key) {
                        if let Some(v) = float(input) {
                            t.map.insert(key, v);
                        }
                    }
                }
            }
            "Gather" if n.name == "/enc_p/emb/Gather" || n.name == "/emb_g/Gather" => {
                let key = if n.name == "/emb_g/Gather" {
                    "emb_g.weight"
                } else {
                    "enc_p.emb.weight"
                };
                if let Some(v) = n.inputs.first().and_then(|i| float(i)) {
                    t.map.insert(key.to_string(), v);
                }
            }
            _ => {}
        }
    }
    // ElementwiseAffine reverse multiplies by exp(-logs); exports fold it.
    let neg = if let Some(v) = float("/dp/flows.0/Exp_output_0") {
        v
    } else if let Some((s, logs)) = t.map.get("dp.flows.0.logs") {
        (s.clone(), logs.iter().map(|l| (-l).exp()).collect())
    } else {
        return err("piper: dp.flows.0 logs not found");
    };
    t.map.insert("dp.flows.0.exp_neg_logs".into(), neg);
    t.map.remove("dp.flows.0.logs");

    let c = t.infer_config()?;
    // Transposed convolutions → stacked polyphase matrices.
    for i in 0..c.upsample_rates.len() {
        let name = format!("dec.ups.{i}.weight");
        let (shape, w) = t.map.remove(&name).unwrap();
        let (cin, cout, k) = (shape[0], shape[1], shape[2]);
        let s = c.upsample_rates[i];
        let (a, taps) = stacked_polyphase(&w, cin, cout, k, s);
        t.map.insert(
            format!("dec.ups.{i}.polyphase"),
            (vec![s * cout, cin * taps], a),
        );
    }

    let audio = &cfg["audio"];
    let inference = &cfg["inference"];
    let num_speakers = cfg["num_speakers"].as_u64().unwrap_or(1);
    let emb_g_rows = t.map.get("emb_g.weight").map_or(1, |(s, _)| s[0] as u64);
    if c.gin > 0 && num_speakers > emb_g_rows {
        return err(format!(
            "piper: config lists {num_speakers} speakers, model has {emb_g_rows}"
        ));
    }
    let meta = serde_json::json!({
        "name": format!(
            "Piper {} {}",
            cfg["dataset"].as_str().unwrap_or("voice"),
            audio["quality"].as_str().unwrap_or("")
        )
        .trim_end(),
        "compat": "Piper VITS (rhasspy/piper-voices, MIT)",
        "sample_rate": audio["sample_rate"].as_u64().unwrap_or(22_050),
        "quality": audio["quality"],
        "dataset": cfg["dataset"],
        "language": cfg["language"]["code"],
        "language_info": cfg["language"],
        "espeak_voice": cfg["espeak"]["voice"],
        "phoneme_type": cfg["phoneme_type"],
        "phoneme_id_map": cfg["phoneme_id_map"],
        "num_speakers": num_speakers,
        "speaker_id_map": cfg["speaker_id_map"],
        "inference": {
            "noise_scale": inference["noise_scale"].as_f64().unwrap_or(0.667),
            "length_scale": inference["length_scale"].as_f64().unwrap_or(1.0),
            "noise_w": inference["noise_w"].as_f64().unwrap_or(0.8),
        },
        "matrix_dtype": dtype,
        "config": c,
    });
    let mut w = Writer::new("piper", meta);
    for (name, (shape, v)) in &t.map {
        let rows = shape.first().copied().unwrap_or(1).max(1);
        let cols = v.len() / rows;
        let matrix = shape.len() >= 2 && rows >= 16 && cols >= 16 && !name.contains("emb_rel");
        let d = match dtype {
            _ if !matrix => DType::F32,
            DType::Q8 if !q8_tensor(name) => DType::F16,
            d => d,
        };
        w.add(name, shape, v, d);
    }
    Ok(w.finish())
}

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

struct Norm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl Norm {
    fn apply(&self, x: &mut Mat) {
        ops::layer_norm_channels(x, Some(&self.gamma), Some(&self.beta), 1e-5);
    }
}

struct EncLayer {
    q: Conv1d,
    k: Conv1d,
    v: Conv1d,
    o: Conv1d,
    /// `[2w+1, d_k]` relative key / value embeddings (shared by the heads).
    rel_k: Vec<f32>,
    rel_v: Vec<f32>,
    n1: Norm,
    ff1: Conv1d,
    ff2: Conv1d,
    n2: Norm,
}

/// Dilated depth-separable convolutions (DDSConv) with exact GELU.
struct Dds {
    kernel: usize,
    /// Depthwise weights `[C, k]` and biases per layer.
    sep: Vec<(Vec<f32>, Vec<f32>)>,
    n1: Vec<Norm>,
    pw: Vec<Conv1d>,
    n2: Vec<Norm>,
}

struct ConvFlow {
    pre: Conv1d,
    dds: Dds,
    proj: Conv1d,
}

struct WaveNet {
    ins: Vec<Conv1d>,
    res_skip: Vec<Conv1d>,
    cond: Option<Conv1d>,
    hidden: usize,
}

struct Coupling {
    pre: Conv1d,
    wn: WaveNet,
    post: Conv1d,
}

/// HiFi-GAN residual block: type 2 has one conv per step, type 1 a dilated
/// conv followed by a dilation-1 conv.
struct ResBlock {
    steps: Vec<(Conv1d, Option<Conv1d>)>,
    /// Input samples needed on each side of an output range.
    halo: usize,
}

pub struct Piper {
    pub config: PiperConfig,
    pub sample_rate: u32,
    /// Raw `.ihw` metadata (tables and config).
    pub meta: serde_json::Value,
    phoneme_ids: HashMap<char, Vec<i64>>,
    pad: Vec<i64>,
    bos: Vec<i64>,
    eos: Vec<i64>,
    speakers: BTreeMap<String, usize>,
    num_speakers: usize,
    defaults: PiperOptions,
    emb: Mat,
    layers: Vec<EncLayer>,
    enc_proj: Conv1d,
    emb_g: Option<Mat>,
    dp_pre: Conv1d,
    dp_cond: Option<Conv1d>,
    dp_dds: Dds,
    dp_proj: Conv1d,
    dp_flows: Vec<ConvFlow>,
    dp_m: [f32; 2],
    dp_exp_neg_logs: [f32; 2],
    flows: Vec<Coupling>,
    dec_pre: Conv1d,
    dec_cond: Option<Conv1d>,
    ups: Vec<Upsample>,
    res: Vec<ResBlock>,
    dec_post: PostConv,
}

struct L<'a> {
    f: &'a mut WeightFile,
    prepack: bool,
}

impl L<'_> {
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.f.get(name)?.shape.clone())
    }

    fn vec(&mut self, name: &str) -> Result<Vec<f32>> {
        let n = self.f.get(name)?.numel();
        self.f.f32(name, n)
    }

    /// Conv1d from `{name}.weight` `[cout, cin, k]` (+ optional bias),
    /// "same" padding for dilation `dil`.
    fn conv(&mut self, name: &str, dil: usize) -> Result<Conv1d> {
        let wname = format!("{name}.weight");
        let shape = self.shape(&wname)?;
        if shape.len() != 3 {
            return err(format!("{wname}: expected 3 dims, found {shape:?}"));
        }
        let (cout, cin, k) = (shape[0], shape[1], shape[2]);
        let w = self.f.matrix(&wname, cout, cin * k, self.prepack)?;
        let bname = format!("{name}.bias");
        let bias = if self.f.get(&bname).is_ok() {
            Some(self.f.f32(&bname, cout)?)
        } else {
            None
        };
        Ok(Conv1d {
            cin,
            cout,
            k,
            stride: 1,
            pad: dil * (k - 1) / 2,
            dil,
            w,
            bias,
        })
    }

    fn norm(&mut self, name: &str, c: usize) -> Result<Norm> {
        Ok(Norm {
            gamma: self.f.f32(&format!("{name}.gamma"), c)?,
            beta: self.f.f32(&format!("{name}.beta"), c)?,
        })
    }

    fn mat(&mut self, name: &str) -> Result<Mat> {
        let shape = self.shape(name)?;
        let v = self.vec(name)?;
        Ok(Mat::from_vec(shape[0], v.len() / shape[0].max(1), v))
    }

    fn dds(&mut self, name: &str, c: usize, layers: usize, kernel: usize) -> Result<Dds> {
        let mut d = Dds {
            kernel,
            sep: Vec::new(),
            n1: Vec::new(),
            pw: Vec::new(),
            n2: Vec::new(),
        };
        for i in 0..layers {
            d.sep.push((
                self.f
                    .f32(&format!("{name}.convs_sep.{i}.weight"), c * kernel)?,
                self.f.f32(&format!("{name}.convs_sep.{i}.bias"), c)?,
            ));
            d.n1.push(self.norm(&format!("{name}.norms_1.{i}"), c)?);
            d.pw.push(self.conv(&format!("{name}.convs_1x1.{i}"), 1)?);
            d.n2.push(self.norm(&format!("{name}.norms_2.{i}"), c)?);
        }
        Ok(d)
    }

    fn upsample(&mut self, i: usize, cin: usize, c: &PiperConfig) -> Result<Upsample> {
        let (k, stride) = (c.upsample_kernels[i], c.upsample_rates[i]);
        let bias = self.vec(&format!("dec.ups.{i}.bias"))?;
        let cout = bias.len();
        let taps = k.div_ceil(stride);
        Ok(Upsample {
            w: self.f.matrix(
                &format!("dec.ups.{i}.polyphase"),
                stride * cout,
                cin * taps,
                self.prepack,
            )?,
            cin,
            cout,
            k,
            stride,
            pad: c.upsample_pads[i],
            taps,
            bias,
        })
    }

    fn post(&mut self, c: &PiperConfig) -> Result<PostConv> {
        let w = self.vec("dec.conv_post.weight")?;
        let bias = if c.post_bias {
            self.vec("dec.conv_post.bias")?[0]
        } else {
            0.0
        };
        Ok(PostConv {
            cin: w.len() / c.post_kernel,
            k: c.post_kernel,
            w,
            bias,
        })
    }
}

/// Stacks the polyphase filters of a ConvTranspose1d weight `[cin][cout][k]`
/// into one `(stride·cout) × (cin·taps)` matrix: row `r·cout + co`, column
/// `ci·taps + kt` is `W[ci][co][r + (taps-1-kt)·stride]` (zero past `k`).
fn stacked_polyphase(
    w: &[f32],
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
) -> (Vec<f32>, usize) {
    let taps = k.div_ceil(stride);
    let mut a = vec![0f32; stride * cout * cin * taps];
    for r in 0..stride {
        for co in 0..cout {
            for ci in 0..cin {
                for kt in 0..taps {
                    let tap = r + (taps - 1 - kt) * stride;
                    if tap < k {
                        a[((r * cout + co) * cin + ci) * taps + kt] = w[(ci * cout + co) * k + tap];
                    }
                }
            }
        }
    }
    (a, taps)
}

/// Transposed convolution evaluated as a single GEMM over the stacked
/// polyphase filters, in parallel over time chunks.
struct Upsample {
    w: Matrix,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
    taps: usize,
    bias: Vec<f32>,
}

impl Upsample {
    fn forward(&self, x: &Mat) -> Mat {
        let (l, s, taps, cout) = (x.cols, self.stride, self.taps, self.cout);
        let tout = (l - 1) * s + self.k - 2 * self.pad;
        // Phase output q (before interleaving) reads x[q - taps + 1 ..= q].
        let qn = l + taps - 1;
        let threads = par::num_threads();
        let chunk = ((1usize << 16) / (s * cout))
            .max(32)
            .min(qn.div_ceil(2 * threads).max(32));
        let tasks = qn.div_ceil(chunk);
        let mut y = Mat::zeros(cout, tout);
        let yp = par::SendPtr(y.data.as_mut_ptr());
        par::for_each(tasks, |i| {
            let q0 = i * chunk;
            let n = chunk.min(qn - q0);
            let win = window(x, q0 as isize - (taps - 1) as isize, n + taps - 1);
            let mut tmp = vec![0f32; s * cout * n];
            let src = Im2col {
                x: &win,
                ld: n + taps - 1,
                t_in: n + taps - 1,
                channels: self.cin,
                taps,
                stride: 1,
                dil: 1,
                pad: 0,
                n,
            };
            self.w.gemm_serial(&src.into(), &mut tmp, n, false);
            for r in 0..s {
                for co in 0..cout {
                    let b = self.bias[co];
                    let row = &tmp[(r * cout + co) * n..(r * cout + co + 1) * n];
                    for (j, &v) in row.iter().enumerate() {
                        let o = (q0 + j) * s + r;
                        if o < self.pad {
                            continue;
                        }
                        let o = o - self.pad;
                        if o >= tout {
                            break;
                        }
                        // SAFETY: chunks cover disjoint output ranges.
                        unsafe { *yp.get().add(co * tout + o) = v + b };
                    }
                }
            }
        });
        y
    }
}

/// Final single-channel convolution followed by tanh.
struct PostConv {
    cin: usize,
    k: usize,
    w: Vec<f32>,
    bias: f32,
}

impl PostConv {
    fn forward(&self, x: &Mat) -> Vec<f32> {
        let (t, k) = (x.cols, self.k);
        let pad = (k - 1) / 2;
        let mut y = vec![0f32; t];
        let chunk = 4096;
        let yp = par::SendPtr(y.as_mut_ptr());
        par::for_each(t.div_ceil(chunk), |i| {
            let t0 = i * chunk;
            let n = chunk.min(t - t0);
            // SAFETY: chunks are disjoint.
            let out = unsafe { std::slice::from_raw_parts_mut(yp.get().add(t0), n) };
            out.fill(self.bias);
            for ci in 0..self.cin {
                let src = x.row(ci);
                for kt in 0..k {
                    let w = self.w[ci * k + kt];
                    // out[j] += w · src[t0 + j + kt - pad] where in range.
                    let off = t0 as isize + kt as isize - pad as isize;
                    let lo = (-off).clamp(0, n as isize) as usize;
                    let hi = (t as isize - off).clamp(lo as isize, n as isize) as usize;
                    let s = &src[(off + lo as isize) as usize..(off + hi as isize) as usize];
                    for (o, v) in out[lo..hi].iter_mut().zip(s) {
                        *o += w * *v;
                    }
                }
            }
            for v in out.iter_mut() {
                *v = math::tanh(*v);
            }
        });
        y
    }
}

/// Columns `start..start+len` of `x` (zero outside the signal), `[C, len]`.
fn window(x: &Mat, start: isize, len: usize) -> Vec<f32> {
    let (c, t) = (x.rows, x.cols);
    let mut y = vec![0f32; c * len];
    let lo = start.clamp(0, t as isize) as usize;
    let hi = (start + len as isize).clamp(lo as isize, t as isize) as usize;
    let off = (lo as isize - start) as usize;
    for r in 0..c {
        y[r * len + off..r * len + off + hi - lo].copy_from_slice(&x.row(r)[lo..hi]);
    }
    y
}

fn add_col(x: &mut Mat, v: &[f32]) {
    for (r, &b) in v[..x.rows].iter().enumerate() {
        for e in x.row_mut(r) {
            *e += b;
        }
    }
}

fn conv_vec(c: &Conv1d, v: &[f32]) -> Vec<f32> {
    c.forward(&Mat::from_vec(v.len(), 1, v.to_vec())).data
}

/// `conv.forward(x)` for "same" convolutions, split into time chunks that
/// run in parallel with serial GEMMs. With a few hundred time steps the
/// GEMM's own split falls back to row panels, which re-packs the whole
/// input in every task; small convolutions (below ~16 M multiply-adds)
/// stay on the calling thread, where they finish faster than a fork.
fn tconv(conv: &Conv1d, x: &Mat) -> Mat {
    let t = x.cols;
    let span = conv.dil * (conv.k - 1);
    if conv.stride != 1 || 2 * conv.pad != span {
        return conv.forward(x);
    }
    let threads = par::num_threads();
    let work = conv.cout * conv.cin * conv.k * t;
    if threads <= 1 || work < 1 << 24 || t < 64 {
        return conv_serial(conv, x);
    }
    let tasks = if t >= 256 * threads {
        2 * threads
    } else {
        threads
    };
    let chunk = t.div_ceil(tasks).div_ceil(8) * 8;
    let cout = conv.cout;
    let mut y = Mat::zeros(cout, t);
    let yp = par::SendPtr(y.data.as_mut_ptr());
    par::for_each(t.div_ceil(chunk), |i| {
        let t0 = i * chunk;
        let n = chunk.min(t - t0);
        let len = n + span;
        let win = window(x, t0 as isize - conv.pad as isize, len);
        let mut out = vec![0f32; cout * n];
        let src = Im2col {
            x: &win,
            ld: len,
            t_in: len,
            channels: conv.cin,
            taps: conv.k,
            stride: 1,
            dil: conv.dil,
            pad: 0,
            n,
        };
        conv.w.gemm_serial(&src.into(), &mut out, n, false);
        for r in 0..cout {
            let b = conv.bias.as_ref().map_or(0.0, |b| b[r]);
            // SAFETY: chunks write disjoint column ranges of every row.
            let dst = unsafe { std::slice::from_raw_parts_mut(yp.get().add(r * t + t0), n) };
            for (d, v) in dst.iter_mut().zip(&out[r * n..(r + 1) * n]) {
                *d = *v + b;
            }
        }
    });
    y
}

fn rows(x: &Mat, r0: usize, n: usize) -> Mat {
    Mat::from_vec(n, x.cols, x.data[r0 * x.cols..(r0 + n) * x.cols].to_vec())
}

/// `conv.forward(x)` on the calling thread only.
fn conv_serial(conv: &Conv1d, x: &Mat) -> Mat {
    let t = x.cols;
    let n = conv.out_len(t);
    let mut y = Mat::zeros(conv.cout, n);
    if n == 0 {
        return y;
    }
    if conv.k == 1 && conv.stride == 1 && conv.pad == 0 {
        conv.w.gemm_serial(&x.as_b(), &mut y.data, n, false);
    } else {
        let src = Im2col {
            x: &x.data,
            ld: t,
            t_in: t,
            channels: conv.cin,
            taps: conv.k,
            stride: conv.stride,
            dil: conv.dil,
            pad: conv.pad,
            n,
        };
        conv.w.gemm_serial(&src.into(), &mut y.data, n, false);
    }
    if let Some(b) = &conv.bias {
        ops::add_bias(&mut y, b);
    }
    y
}

/// Zeroes the columns of a window (first column = sample `start`) that lie
/// outside a signal of length `t`.
fn zero_outside(x: &mut Mat, start: isize, t: usize) {
    let len = x.cols as isize;
    let lo = (-start).clamp(0, len) as usize;
    let hi = (t as isize - start).clamp(lo as isize, len) as usize;
    if lo == 0 && hi == x.cols {
        return;
    }
    for r in 0..x.rows {
        let row = x.row_mut(r);
        row[..lo].fill(0.0);
        row[hi..].fill(0.0);
    }
}

/// Evaluates a time-local computation in parallel chunks: `f(start, len)`
/// returns `[rows, len]` for the window of columns `start..start+len`
/// (which extends `halo` columns past the chunk on each side); the chunk
/// centres are assembled into `[rows, t]`.
fn par_time<F>(t: usize, rows: usize, halo: usize, f: F) -> Mat
where
    F: Fn(isize, usize) -> Mat + Sync,
{
    let tasks = par::num_threads().min(t.div_ceil(32)).max(1);
    let chunk = t.div_ceil(tasks);
    let mut y = Mat::zeros(rows, t);
    let yp = par::SendPtr(y.data.as_mut_ptr());
    par::for_each(t.div_ceil(chunk), |i| {
        let t0 = i * chunk;
        let n = chunk.min(t - t0);
        let w = f(t0 as isize - halo as isize, n + 2 * halo);
        debug_assert_eq!((w.rows, w.cols), (rows, n + 2 * halo));
        for r in 0..rows {
            // SAFETY: chunks write disjoint column ranges of every row.
            let dst = unsafe { std::slice::from_raw_parts_mut(yp.get().add(r * t + t0), n) };
            dst.copy_from_slice(&w.row(r)[halo..halo + n]);
        }
    });
    y
}

/// Reverses the channel order (VITS `Flip`).
fn flip(x: &Mat) -> Mat {
    let mut f = Mat::zeros(x.rows, x.cols);
    for r in 0..x.rows {
        f.row_mut(r).copy_from_slice(x.row(x.rows - 1 - r));
    }
    f
}

#[inline(always)]
fn lrelu(v: f32, slope: f32) -> f32 {
    if v >= 0.0 {
        v
    } else {
        v * slope
    }
}

/// Rational-quadratic spline with linear tails (Durkan et al.) as used by
/// the VITS duration predictor, evaluated in f64.
struct Spline {
    bound: f64,
    /// Knot positions (`bins + 1`) along x and y.
    cw: Vec<f64>,
    ch: Vec<f64>,
    /// Derivatives at the knots (`bins + 1`).
    d: Vec<f64>,
}

impl Spline {
    const MIN_BIN: f64 = 1e-3;
    const MIN_DERIV: f64 = 1e-3;

    /// `uw`, `uh` have `bins` unnormalised entries, `ud` has `bins - 1`.
    fn new(uw: &[f64], uh: &[f64], ud: &[f64], bound: f64) -> Spline {
        let bins = uw.len();
        let knots = |u: &[f64]| -> Vec<f64> {
            let mx = u.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
            let e: Vec<f64> = u.iter().map(|v| (v - mx).exp()).collect();
            let s: f64 = e.iter().sum();
            let mut cum = Vec::with_capacity(bins + 1);
            cum.push(0.0);
            let mut acc = 0.0;
            for v in &e {
                acc += Self::MIN_BIN + (1.0 - Self::MIN_BIN * bins as f64) * (v / s);
                cum.push(acc);
            }
            for c in cum.iter_mut() {
                *c = 2.0 * bound * *c - bound;
            }
            cum[0] = -bound;
            cum[bins] = bound;
            cum
        };
        // Boundary derivatives are 1: softplus(edge) + MIN_DERIV = 1.
        let edge = ((1.0 - Self::MIN_DERIV).exp() - 1.0).ln();
        let d = (0..=bins)
            .map(|i| {
                let u = if i == 0 || i == bins { edge } else { ud[i - 1] };
                Self::MIN_DERIV + u.exp().ln_1p()
            })
            .collect();
        Spline {
            bound,
            cw: knots(uw),
            ch: knots(uh),
            d,
        }
    }

    fn inverse(&self, y: f64) -> f64 {
        let bins = self.cw.len() - 1;
        if !(-self.bound..=self.bound).contains(&y) {
            return y;
        }
        let mut bin = 0;
        for (i, &c) in self.ch.iter().enumerate() {
            let c = if i == bins { c + 1e-6 } else { c };
            if y >= c {
                bin = i;
            }
        }
        let bin = bin.min(bins - 1);
        let (x0, w) = (self.cw[bin], self.cw[bin + 1] - self.cw[bin]);
        let (y0, h) = (self.ch[bin], self.ch[bin + 1] - self.ch[bin]);
        let delta = h / w;
        let (d0, d1) = (self.d[bin], self.d[bin + 1]);
        let dy = y - y0;
        let s = d0 + d1 - 2.0 * delta;
        let a = dy * s + h * (delta - d0);
        let b = h * d0 - dy * s;
        let c = -delta * dy;
        let disc = b * b - 4.0 * a * c;
        let root = (2.0 * c) / (-b - disc.sqrt());
        root * w + x0
    }
}

impl ResBlock {
    /// Adds the block output for samples `t0..t0+n` of `x` to `acc`
    /// (`[C, n]`). Works on a window with `halo` extra samples per side and
    /// "valid" convolutions; samples outside the signal are kept at zero,
    /// which reproduces the zero padding of every convolution.
    fn forward_chunk(&self, x: &Mat, t0: usize, n: usize, acc: &mut [f32]) {
        let (c, t) = (x.rows, x.cols);
        let mut start = t0 as isize - self.halo as isize;
        let mut len = n + 2 * self.halo;
        let mut y = window(x, start, len);
        let mut a: Vec<f32> = Vec::new();
        for (c1, c2) in &self.steps {
            a.clear();
            a.extend(y.iter().map(|&v| lrelu(v, 0.1)));
            let (mut b, mut blen, mut bstart) = conv_valid(c1, &a, len, start, t);
            if let Some(c2) = c2 {
                for v in &mut b {
                    *v = lrelu(*v, 0.1);
                }
                (b, blen, bstart) = conv_valid(c2, &b, blen, bstart, t);
            }
            let shift = (bstart - start) as usize;
            for r in 0..c {
                let src = &y[r * len + shift..r * len + shift + blen];
                for (o, s) in b[r * blen..(r + 1) * blen].iter_mut().zip(src) {
                    *o += *s;
                }
            }
            (y, len, start) = (b, blen, bstart);
        }
        debug_assert!(len == n && start == t0 as isize);
        for (o, v) in acc.iter_mut().zip(&y) {
            *o += *v;
        }
    }
}

/// "Valid" convolution of a window `[cin, len]` whose first column is
/// sample `start` of a signal of length `t`; returns the output window,
/// its length and start. Outputs outside `0..t` are set to zero.
fn conv_valid(
    conv: &Conv1d,
    x: &[f32],
    len: usize,
    start: isize,
    t: usize,
) -> (Vec<f32>, usize, isize) {
    let span = conv.dil * (conv.k - 1);
    let n = len - span;
    let mut out = vec![0f32; conv.cout * n];
    let src = Im2col {
        x,
        ld: len,
        t_in: len,
        channels: conv.cin,
        taps: conv.k,
        stride: 1,
        dil: conv.dil,
        pad: 0,
        n,
    };
    conv.w.gemm_serial(&src.into(), &mut out, n, false);
    let ostart = start + (span / 2) as isize;
    let lo = (-ostart).clamp(0, n as isize) as usize;
    let hi = (t as isize - ostart).clamp(lo as isize, n as isize) as usize;
    for r in 0..conv.cout {
        let row = &mut out[r * n..(r + 1) * n];
        let b = conv.bias.as_ref().map_or(0.0, |b| b[r]);
        row[..lo].fill(0.0);
        for v in &mut row[lo..hi] {
            *v += b;
        }
        row[hi..].fill(0.0);
    }
    (out, n, ostart)
}

impl Piper {
    pub fn load(mut f: WeightFile, prepack: bool) -> Result<Piper> {
        if f.header.model != "piper" {
            return err(format!(
                "expected a piper model, found {:?}",
                f.header.model
            ));
        }
        let meta = f.header.meta.clone();
        let c: PiperConfig = serde_json::from_value(meta["config"].clone())
            .map_err(|e| json_err("bad piper config", e))?;
        let sample_rate = meta["sample_rate"].as_u64().unwrap_or(22_050) as u32;
        let id_map: BTreeMap<String, Vec<i64>> =
            serde_json::from_value(meta["phoneme_id_map"].clone())
                .map_err(|e| json_err("bad phoneme_id_map", e))?;
        let mut phoneme_ids = HashMap::new();
        for (sym, ids) in &id_map {
            let mut chars = sym.chars();
            if let (Some(ch), None) = (chars.next(), chars.next()) {
                phoneme_ids.insert(ch, ids.clone());
            }
        }
        let table = |s: &str, d: i64| id_map.get(s).cloned().unwrap_or_else(|| vec![d]);
        let speakers: BTreeMap<String, usize> = if meta["speaker_id_map"].is_object() {
            serde_json::from_value(meta["speaker_id_map"].clone())
                .map_err(|e| json_err("bad speaker_id_map", e))?
        } else {
            BTreeMap::new()
        };
        let inf = &meta["inference"];
        let defaults = PiperOptions {
            noise_scale: inf["noise_scale"].as_f64().unwrap_or(0.667) as f32,
            length_scale: inf["length_scale"].as_f64().unwrap_or(1.0) as f32,
            noise_w: inf["noise_w"].as_f64().unwrap_or(0.8) as f32,
            seed: 1,
        };

        let mut l = L { f: &mut f, prepack };
        let layers = (0..c.enc_layers)
            .map(|i| -> Result<EncLayer> {
                let a = format!("enc_p.encoder.attn_layers.{i}");
                let ff = format!("enc_p.encoder.ffn_layers.{i}");
                Ok(EncLayer {
                    q: l.conv(&format!("{a}.conv_q"), 1)?,
                    k: l.conv(&format!("{a}.conv_k"), 1)?,
                    v: l.conv(&format!("{a}.conv_v"), 1)?,
                    o: l.conv(&format!("{a}.conv_o"), 1)?,
                    rel_k: l.vec(&format!("{a}.emb_rel_k"))?,
                    rel_v: l.vec(&format!("{a}.emb_rel_v"))?,
                    n1: l.norm(&format!("enc_p.encoder.norm_layers_1.{i}"), c.hidden)?,
                    ff1: l.conv(&format!("{ff}.conv_1"), 1)?,
                    ff2: l.conv(&format!("{ff}.conv_2"), 1)?,
                    n2: l.norm(&format!("enc_p.encoder.norm_layers_2.{i}"), c.hidden)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let dp_flows = c
            .dp_flows
            .iter()
            .map(|&i| -> Result<ConvFlow> {
                let p = format!("dp.flows.{i}");
                Ok(ConvFlow {
                    pre: l.conv(&format!("{p}.pre"), 1)?,
                    dds: l.dds(
                        &format!("{p}.convs"),
                        c.dp_flow_filter,
                        c.dp_flow_layers,
                        c.dp_kernel,
                    )?,
                    proj: l.conv(&format!("{p}.proj"), 1)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let flows = c
            .flows
            .iter()
            .map(|&i| -> Result<Coupling> {
                let p = format!("flow.flows.{i}");
                let wn = WaveNet {
                    ins: c
                        .flow_dilations
                        .iter()
                        .enumerate()
                        .map(|(j, &d)| l.conv(&format!("{p}.enc.in_layers.{j}"), d))
                        .collect::<Result<_>>()?,
                    res_skip: (0..c.flow_dilations.len())
                        .map(|j| l.conv(&format!("{p}.enc.res_skip_layers.{j}"), 1))
                        .collect::<Result<_>>()?,
                    cond: if c.gin > 0 {
                        Some(l.conv(&format!("{p}.enc.cond_layer"), 1)?)
                    } else {
                        None
                    },
                    hidden: c.flow_hidden,
                };
                Ok(Coupling {
                    pre: l.conv(&format!("{p}.pre"), 1)?,
                    wn,
                    post: l.conv(&format!("{p}.post"), 1)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut ups = Vec::new();
        let mut ch = c.upsample_initial;
        for i in 0..c.upsample_rates.len() {
            let u = l.upsample(i, ch, &c)?;
            ch = u.cout;
            ups.push(u);
        }
        let conv_name = if c.resblock == "1" { "convs1" } else { "convs" };
        let nk = c.resblock_kernels.len();
        let mut res = Vec::new();
        for i in 0..ups.len() * nk {
            let (k, dils) = (c.resblock_kernels[i % nk], &c.resblock_dilations[i % nk]);
            let mut steps = Vec::new();
            let mut halo = 0;
            for (j, &d) in dils.iter().enumerate() {
                let c1 = l.conv(&format!("dec.resblocks.{i}.{conv_name}.{j}"), d)?;
                halo += d * (k - 1) / 2;
                let c2 = if c.resblock == "1" {
                    halo += (k - 1) / 2;
                    Some(l.conv(&format!("dec.resblocks.{i}.convs2.{j}"), 1)?)
                } else {
                    None
                };
                steps.push((c1, c2));
            }
            res.push(ResBlock { steps, halo });
        }
        let dp_m = l.vec("dp.flows.0.m")?;
        let dp_e = l.vec("dp.flows.0.exp_neg_logs")?;
        if dp_m.len() != 2 || dp_e.len() != 2 {
            return err("piper: dp.flows.0 must have 2 channels");
        }
        let gin = c.gin > 0;
        Ok(Piper {
            emb: l.mat("enc_p.emb.weight")?,
            layers,
            enc_proj: l.conv("enc_p.proj", 1)?,
            emb_g: if gin {
                Some(l.mat("emb_g.weight")?)
            } else {
                None
            },
            dp_pre: l.conv("dp.pre", 1)?,
            dp_cond: if gin {
                Some(l.conv("dp.cond", 1)?)
            } else {
                None
            },
            dp_dds: l.dds("dp.convs", c.dp_filter, c.dp_layers, c.dp_kernel)?,
            dp_proj: l.conv("dp.proj", 1)?,
            dp_flows,
            dp_m: [dp_m[0], dp_m[1]],
            dp_exp_neg_logs: [dp_e[0], dp_e[1]],
            flows,
            dec_pre: l.conv("dec.conv_pre", 1)?,
            dec_cond: if gin {
                Some(l.conv("dec.cond", 1)?)
            } else {
                None
            },
            ups,
            res,
            dec_post: l.post(&c)?,
            num_speakers: if gin {
                meta["num_speakers"].as_u64().unwrap_or(1) as usize
            } else {
                1
            },
            pad: table("_", 0),
            bos: table("^", 1),
            eos: table("$", 2),
            phoneme_ids,
            speakers,
            defaults,
            config: c,
            sample_rate,
            meta,
        })
    }

    /// Maps espeak-ng IPA (NFD-normalised by the caller) to Piper ids:
    /// `^ _ (symbol _)* $`. Characters missing from the voice are skipped.
    pub fn encode(&self, phonemes: &str) -> Vec<i64> {
        let mut ids = self.bos.clone();
        ids.extend(&self.pad);
        for ch in phonemes.chars() {
            if let Some(v) = self.phoneme_ids.get(&ch) {
                ids.extend(v);
                ids.extend(&self.pad);
            }
        }
        ids.extend(&self.eos);
        ids
    }

    /// Speaker names → ids (empty for single-speaker voices).
    pub fn speakers(&self) -> &BTreeMap<String, usize> {
        &self.speakers
    }

    pub fn num_speakers(&self) -> usize {
        self.num_speakers
    }

    /// Language code of the voice, e.g. `es_ES`.
    pub fn language(&self) -> &str {
        self.meta["language"].as_str().unwrap_or("")
    }

    /// espeak-ng voice the phonemes must come from, e.g. `es`.
    pub fn espeak_voice(&self) -> &str {
        self.meta["espeak_voice"].as_str().unwrap_or("")
    }

    /// Inference defaults of the voice config.
    pub fn defaults(&self) -> PiperOptions {
        self.defaults
    }

    /// Audio samples per duration frame.
    pub fn hop(&self) -> usize {
        self.config.upsample_rates.iter().product()
    }

    /// Multi-head self-attention with relative positions (window `w`, key
    /// and value embeddings shared by the heads; the scaled query is also
    /// used for the relative keys).
    fn attention(&self, l: &EncLayer, x: &Mat) -> Mat {
        let c = &self.config;
        let (t, dk, w, hd) = (x.cols, c.d_k, c.window, c.hidden);
        let scale = (dk as f32).sqrt();
        // The projections and heads are small: run them side by side, each
        // on one thread, instead of splitting every GEMM.
        let (q, (k, v)) = par::join(
            || tconv(&l.q, x),
            || par::join(|| tconv(&l.k, x), || tconv(&l.v, x)),
        );
        // [T, H] layouts: head h of step i is row i, columns h·dk..(h+1)·dk.
        let mut q = q.transpose();
        for v in &mut q.data {
            *v /= scale;
        }
        let (k, v) = (k.transpose(), v.transpose());
        let mut out = Mat::zeros(t, hd);
        let op = par::SendPtr(out.data.as_mut_ptr());
        par::for_each(c.heads, |h| {
            // Scores Q_h K_hᵀ.
            let mut p = vec![0f32; t * t];
            let qa = AMat {
                m: t,
                k: dk,
                data: AData::F32 {
                    data: &q.data[h * dk..],
                    rs: hd,
                    cs: 1,
                },
            };
            let kt = BMat {
                k: dk,
                n: t,
                data: &k.data[h * dk..],
                rs: 1,
                cs: hd,
            };
            gemm::gemm_packed_serial(&gemm::pack_a(&qa), &kt.into(), &mut p, t, false);
            // Relative keys and softmax.
            for (i, s) in p.chunks_exact_mut(t).enumerate() {
                let qi = &q.row(i)[h * dk..(h + 1) * dk];
                let (j0, j1) = (i.saturating_sub(w), (i + w + 1).min(t));
                for (j, sj) in s.iter_mut().enumerate().take(j1).skip(j0) {
                    let e = j + w - i;
                    *sj += gemm::dot(qi, &l.rel_k[e * dk..(e + 1) * dk]);
                }
                let mx = s.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                let mut sum = 0f32;
                for v in s.iter_mut() {
                    *v = (*v - mx).exp();
                    sum += *v;
                }
                for v in s.iter_mut() {
                    *v /= sum;
                }
            }
            // P V_h into columns h·dk.. of the output.
            let pa = AMat {
                m: t,
                k: t,
                data: AData::F32 {
                    data: &p,
                    rs: t,
                    cs: 1,
                },
            };
            let vb = BMat {
                k: t,
                n: dk,
                data: &v.data[h * dk..],
                rs: hd,
                cs: 1,
            };
            let mut o = vec![0f32; t * dk];
            gemm::gemm_packed_serial(&gemm::pack_a(&pa), &vb.into(), &mut o, dk, false);
            // Relative values.
            for (i, oi) in o.chunks_exact_mut(dk).enumerate() {
                for j in i.saturating_sub(w)..(i + w + 1).min(t) {
                    let (e, pij) = (j + w - i, p[i * t + j]);
                    for (oe, re) in oi.iter_mut().zip(&l.rel_v[e * dk..(e + 1) * dk]) {
                        *oe += pij * *re;
                    }
                }
                // SAFETY: heads write disjoint columns of every row.
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(op.get().add(i * hd + h * dk), dk) };
                dst.copy_from_slice(oi);
            }
        });
        tconv(&l.o, &out.transpose())
    }

    /// Text encoder: hidden states `x` and the prior statistics.
    fn encode_text(&self, ids: &[i64]) -> (Mat, Mat, Mat) {
        let c = &self.config;
        let t = ids.len();
        let sq = (c.hidden as f32).sqrt();
        let mut x = Mat::zeros(c.hidden, t);
        for (p, &id) in ids.iter().enumerate() {
            let e = self.emb.row(id as usize);
            for (ch, &v) in e.iter().enumerate() {
                x.data[ch * t + p] = v * sq;
            }
        }
        for l in &self.layers {
            let y = self.attention(l, &x);
            x.add_inplace(&y);
            l.n1.apply(&mut x);
            // conv → ReLU → conv, both "same" with zero padding.
            let halo = l.ff1.pad + l.ff2.pad;
            let f = par_time(t, c.hidden, halo, |start, len| {
                let w = Mat::from_vec(x.rows, len, window(&x, start, len));
                let mut f = conv_serial(&l.ff1, &w);
                for v in &mut f.data {
                    *v = v.max(0.0);
                }
                zero_outside(&mut f, start, t);
                conv_serial(&l.ff2, &f)
            });
            x.add_inplace(&f);
            l.n2.apply(&mut x);
        }
        let stats = tconv(&self.enc_proj, &x);
        let m_p = rows(&stats, 0, c.inter);
        let logs_p = rows(&stats, c.inter, c.inter);
        (x, m_p, logs_p)
    }

    /// DDSConv on a window (first column = sample `start` of a signal of
    /// length `t`); samples outside the signal are kept at zero.
    fn dds(&self, d: &Dds, mut x: Mat, g: Option<&Mat>, start: isize, t: usize) -> Mat {
        if let Some(g) = g {
            x.add_inplace(g);
        }
        zero_outside(&mut x, start, t);
        let (ch, len, k) = (x.rows, x.cols, d.kernel);
        for i in 0..d.sep.len() {
            let dil = k.pow(i as u32);
            let pad = dil * (k - 1) / 2;
            let (w, b) = &d.sep[i];
            let mut y = Mat::zeros(ch, len);
            for r in 0..ch {
                let (src, dst) = (x.row(r), y.row_mut(r));
                dst.fill(b[r]);
                for kk in 0..k {
                    // dst[o] += w · src[o + kk·dil - pad] where in range.
                    let off = (kk * dil) as isize - pad as isize;
                    let lo = (-off).clamp(0, len as isize) as usize;
                    let hi = (len as isize - off).clamp(lo as isize, len as isize) as usize;
                    let wk = w[r * k + kk];
                    let s = &src[(lo as isize + off) as usize..(hi as isize + off) as usize];
                    for (o, v) in dst[lo..hi].iter_mut().zip(s) {
                        *o += wk * *v;
                    }
                }
            }
            d.n1[i].apply(&mut y);
            for v in &mut y.data {
                *v = math::gelu_erf(*v);
            }
            let mut y = conv_serial(&d.pw[i], &y);
            d.n2[i].apply(&mut y);
            for v in &mut y.data {
                *v = math::gelu_erf(*v);
            }
            x.add_inplace(&y);
            zero_outside(&mut x, start, t);
        }
        x
    }

    /// Receptive radius of a DDSConv stack.
    fn dds_halo(d: &Dds) -> usize {
        (0..d.sep.len())
            .map(|i| d.kernel.pow(i as u32) * (d.kernel - 1) / 2)
            .sum()
    }

    /// Conv flow (reverse): `z` `[2, T]` with condition `cond` `[F, T]`;
    /// channel 1 goes through the inverse spline.
    fn conv_flow_reverse(&self, f: &ConvFlow, z: &Mat, cond: &Mat) -> Mat {
        let c = &self.config;
        let t = z.cols;
        let x0 = rows(z, 0, 1);
        let bins = c.spline_bins;
        let inv = 1.0 / (c.dp_flow_filter as f32).sqrt();
        let halo = Self::dds_halo(&f.dds);
        let x1 = par_time(t, 1, halo, |start, len| {
            let w = Mat::from_vec(1, len, window(&x0, start, len));
            let g = Mat::from_vec(cond.rows, len, window(cond, start, len));
            let h = conv_serial(&f.pre, &w);
            let h = self.dds(&f.dds, h, Some(&g), start, t);
            let h = conv_serial(&f.proj, &h);
            let (mut uw, mut uh, mut ud) =
                (vec![0f64; bins], vec![0f64; bins], vec![0f64; bins - 1]);
            let mut out = Mat::zeros(1, len);
            for p in 0..len {
                let s = start + p as isize;
                if s < 0 || s as usize >= t {
                    continue;
                }
                for b in 0..bins {
                    uw[b] = (h.at(b, p) * inv) as f64;
                    uh[b] = (h.at(bins + b, p) * inv) as f64;
                }
                for (b, u) in ud.iter_mut().enumerate() {
                    *u = h.at(2 * bins + b, p) as f64;
                }
                let y = z.at(1, s as usize) as f64;
                out.data[p] = Spline::new(&uw, &uh, &ud, c.spline_bound).inverse(y) as f32;
            }
            out
        });
        Mat::vcat(&[&x0, &x1])
    }

    /// Stochastic duration predictor (reverse): log-durations per id.
    fn log_durations(
        &self,
        x: &Mat,
        g: Option<&[f32]>,
        noise_w: f32,
        rng: &mut Rng,
        prof: &mut Profiler,
    ) -> Vec<f32> {
        let t = x.cols;
        let cond = match (&self.dp_cond, g) {
            (Some(c), Some(g)) => Some(conv_vec(c, g)),
            _ => None,
        };
        let h = par_time(
            t,
            self.dp_proj.cout,
            Self::dds_halo(&self.dp_dds),
            |start, len| {
                let w = Mat::from_vec(x.rows, len, window(x, start, len));
                let mut h = conv_serial(&self.dp_pre, &w);
                if let Some(cond) = &cond {
                    add_col(&mut h, cond);
                }
                let h = self.dds(&self.dp_dds, h, None, start, t);
                conv_serial(&self.dp_proj, &h)
            },
        );
        prof.mark("dp.convs");
        let mut z = Mat::zeros(2, t);
        if noise_w != 0.0 {
            for v in &mut z.data {
                *v = rng.gaussian() * noise_w;
            }
        }
        for f in &self.dp_flows {
            z = flip(&z);
            z = self.conv_flow_reverse(f, &z, &h);
        }
        prof.mark("dp.flows");
        let z = flip(&z);
        z.row(0)
            .iter()
            .map(|v| (v - self.dp_m[0]) * self.dp_exp_neg_logs[0])
            .collect()
    }

    /// Shift `m` of one coupling layer (reverse) on the window `x0`
    /// `[half, len]` whose first column is sample `start` of a signal of
    /// length `t`.
    fn coupling_shift(
        &self,
        cpl: &Coupling,
        x0: &Mat,
        start: isize,
        t: usize,
        cond: Option<&[f32]>,
    ) -> Mat {
        let wn = &cpl.wn;
        let h = wn.hidden;
        let mut x = conv_serial(&cpl.pre, x0);
        zero_outside(&mut x, start, t);
        let n = wn.ins.len();
        let mut out = Mat::zeros(h, x.cols);
        for i in 0..n {
            let mut a = conv_serial(&wn.ins[i], &x);
            if let Some(cond) = cond {
                add_col(&mut a, &cond[2 * h * i..2 * h * (i + 1)]);
            }
            let mut acts = Mat::zeros(h, x.cols);
            for r in 0..h {
                let (ta, sa) = (a.row(r), a.row(h + r));
                for ((o, t), s) in acts.row_mut(r).iter_mut().zip(ta).zip(sa) {
                    *o = math::tanh(*t) * math::sigmoid(*s);
                }
            }
            let rs = conv_serial(&wn.res_skip[i], &acts);
            let len = x.cols;
            if i + 1 < n {
                for (v, r) in x.data.iter_mut().zip(&rs.data[..h * len]) {
                    *v += *r;
                }
                zero_outside(&mut x, start, t);
                for (v, r) in out.data.iter_mut().zip(&rs.data[h * len..]) {
                    *v += *r;
                }
            } else {
                out.add_inplace(&rs);
            }
        }
        conv_serial(&cpl.post, &out)
    }

    /// Residual coupling flow (reverse, mean-only).
    fn flow_reverse(&self, mut z: Mat, g: Option<&[f32]>) -> Mat {
        let (half, t) = (z.rows / 2, z.cols);
        for cpl in &self.flows {
            let wn = &cpl.wn;
            let cond = match (&wn.cond, g) {
                (Some(c), Some(g)) => Some(conv_vec(c, g)),
                _ => None,
            };
            let halo: usize = wn.ins.iter().map(|c| c.pad).sum();
            let f = flip(&z);
            let x0 = rows(&f, 0, half);
            let m = par_time(t, half, halo, |start, len| {
                let w = Mat::from_vec(half, len, window(&x0, start, len));
                self.coupling_shift(cpl, &w, start, t, cond.as_deref())
            });
            let mut x1 = rows(&f, half, half);
            x1.data.iter_mut().zip(&m.data).for_each(|(a, b)| *a -= *b);
            z = Mat::vcat(&[&x0, &x1]);
        }
        z
    }
    /// Mean of the residual blocks of upsampling stage `stage`, followed by
    /// the next leaky ReLU. Runs on independent time chunks in parallel.
    fn res_stage(&self, stage: usize, x: &Mat, out_slope: f32) -> Mat {
        let (c, t) = (x.rows, x.cols);
        let nk = self.config.resblock_kernels.len();
        let blocks = &self.res[stage * nk..(stage + 1) * nk];
        // ~256 KiB of activations per window keeps a chunk in L2.
        let base = ((1usize << 16) / c).max(64);
        let threads = par::num_threads();
        let chunk = base.min(t.div_ceil(2 * threads).max(128));
        let tasks = t.div_ceil(chunk);
        let mut out = Mat::zeros(c, t);
        let op = par::SendPtr(out.data.as_mut_ptr());
        let div = nk as f32;
        par::for_each(tasks, |i| {
            let t0 = i * chunk;
            let n = chunk.min(t - t0);
            let mut acc = vec![0f32; c * n];
            for b in blocks {
                b.forward_chunk(x, t0, n, &mut acc);
            }
            for r in 0..c {
                // SAFETY: tasks write disjoint column ranges of every row.
                let dst = unsafe { std::slice::from_raw_parts_mut(op.get().add(r * t + t0), n) };
                for (d, a) in dst.iter_mut().zip(&acc[r * n..(r + 1) * n]) {
                    *d = lrelu(*a / div, out_slope);
                }
            }
        });
        out
    }

    /// HiFi-GAN generator: `z` `[inter, frames]` → waveform.
    fn generate(&self, z: &Mat, g: Option<&[f32]>, prof: &mut Profiler) -> Vec<f32> {
        let mut x = tconv(&self.dec_pre, z);
        if let (Some(cond), Some(g)) = (&self.dec_cond, g) {
            add_col(&mut x, &conv_vec(cond, g));
        }
        math::leaky_relu_inplace(&mut x.data, 0.1);
        prof.mark("gen.pre");
        let n = self.ups.len();
        for (i, up) in self.ups.iter().enumerate() {
            let y = up.forward(&x);
            prof.mark("gen.ups");
            x = self.res_stage(i, &y, if i + 1 < n { 0.1 } else { 0.01 });
            prof.mark("gen.res");
        }
        self.dec_post.forward(&x)
    }

    /// Synthesises Piper ids (see [`Piper::encode`]) for `speaker`.
    pub fn synthesize(
        &self,
        ids: &[i64],
        speaker: usize,
        opts: PiperOptions,
    ) -> Result<PiperOutput> {
        let c = &self.config;
        if let Some(&bad) = ids.iter().find(|&&i| i < 0 || i as usize >= c.n_vocab) {
            return err(format!(
                "piper: id {bad} out of range (vocabulary {})",
                c.n_vocab
            ));
        }
        if ids.is_empty() {
            return Ok(PiperOutput {
                audio: Vec::new(),
                durations: Vec::new(),
            });
        }
        let g = match &self.emb_g {
            Some(e) if speaker >= e.rows => {
                return err(format!(
                    "piper: speaker {speaker} out of range ({})",
                    e.rows
                ));
            }
            Some(e) => Some(e.row(speaker).to_vec()),
            None => None,
        };
        let g = g.as_deref();
        let mut prof = Profiler::new();
        let mut rng = Rng::new(opts.seed);
        let (x, m_p, logs_p) = self.encode_text(ids);
        prof.mark("text_encoder");
        let logw = self.log_durations(&x, g, opts.noise_w, &mut rng, &mut prof);
        let durations: Vec<usize> = logw
            .iter()
            .map(|l| (l.exp() * opts.length_scale).ceil().max(0.0) as usize)
            .collect();
        let frames: usize = durations.iter().sum();
        let mut z = if frames == 0 {
            // VITS clamps the output length to one (all-zero) frame.
            Mat::zeros(c.inter, 1)
        } else {
            let mut z = m_p.repeat_cols(&durations);
            if opts.noise_scale != 0.0 {
                let logs = logs_p.repeat_cols(&durations);
                for (v, ls) in z.data.iter_mut().zip(&logs.data) {
                    *v += rng.gaussian() * ls.exp() * opts.noise_scale;
                }
            }
            z
        };
        z = self.flow_reverse(z, g);
        prof.mark("flow");
        let audio = self.generate(&z, g, &mut prof);
        prof.mark("gen.post");
        prof.report();
        Ok(PiperOutput { audio, durations })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Spline {
        fn forward(&self, x: f64) -> f64 {
            let bins = self.cw.len() - 1;
            let bin = (0..bins).rev().find(|&i| x >= self.cw[i]).unwrap_or(0);
            let (x0, w) = (self.cw[bin], self.cw[bin + 1] - self.cw[bin]);
            let (y0, h) = (self.ch[bin], self.ch[bin + 1] - self.ch[bin]);
            let delta = h / w;
            let (d0, d1) = (self.d[bin], self.d[bin + 1]);
            let th = (x - x0) / w;
            let num = h * (delta * th * th + d0 * th * (1.0 - th));
            let den = delta + (d0 + d1 - 2.0 * delta) * th * (1.0 - th);
            y0 + num / den
        }
    }

    #[test]
    fn spline_inverse_inverts_forward() {
        let uw = [0.3, -1.0, 0.5, 2.0, 0.0, -0.2, 1.1, 0.7, -0.5, 0.1];
        let uh = [1.0, 0.2, -0.3, 0.0, 0.9, -1.2, 0.4, 0.3, 0.0, 0.5];
        let ud = [0.1, -0.4, 0.8, 0.0, 1.5, -1.0, 0.3, 0.2, -0.1];
        let s = Spline::new(&uw, &uh, &ud, 5.0);
        assert_eq!(s.inverse(7.0), 7.0);
        assert_eq!(s.inverse(-5.5), -5.5);
        let mut prev = f64::NEG_INFINITY;
        for i in 0..=1000 {
            let x = -5.0 + 10.0 * i as f64 / 1000.0;
            let y = s.forward(x);
            assert!(y > prev, "not monotone at {x}");
            prev = y;
            let back = s.inverse(y);
            assert!((back - x).abs() < 1e-9, "{x} -> {y} -> {back}");
        }
    }

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..n).map(|_| r.uniform() * 2.0 - 1.0).collect()
    }

    #[test]
    fn upsample_matches_conv_transpose() {
        for &(cin, cout, t, k, s) in &[
            (6, 4, 37, 16, 8),
            (5, 3, 300, 8, 4),
            (4, 2, 9, 4, 2),
            (3, 2, 20, 7, 3),
        ] {
            let x = Mat::from_vec(cin, t, rnd(cin * t, 1));
            let w = rnd(cin * cout * k, 2);
            let bias = rnd(cout, 3);
            let pad = (k - s) / 2;
            let phases = ops::polyphase_weights(&w, cin, cout, k, s)
                .into_iter()
                .map(|(a, taps)| (Matrix::from_f32(cout, cin * taps, a, false), taps))
                .collect();
            let want = ops::ConvTranspose1d {
                cin,
                cout,
                k,
                stride: s,
                pad,
                out_pad: 0,
                phases,
                bias: bias.clone(),
            }
            .forward(&x);
            let (a, taps) = stacked_polyphase(&w, cin, cout, k, s);
            let up = Upsample {
                w: Matrix::from_f32(s * cout, cin * taps, a, true),
                cin,
                cout,
                k,
                stride: s,
                pad,
                taps,
                bias,
            };
            let y = up.forward(&x);
            assert_eq!((y.rows, y.cols), (want.rows, want.cols));
            for (a, b) in y.data.iter().zip(&want.data) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn post_conv_matches_conv1d() {
        let (cin, t, k) = (5, 9000, 7);
        let x = Mat::from_vec(cin, t, rnd(cin * t, 4));
        let w = rnd(cin * k, 5);
        let conv = Conv1d {
            cin,
            cout: 1,
            k,
            stride: 1,
            pad: 3,
            dil: 1,
            w: Matrix::from_f32(1, cin * k, w.clone(), false),
            bias: Some(vec![0.25]),
        };
        let want = conv.forward(&x);
        let y = PostConv {
            cin,
            k,
            w,
            bias: 0.25,
        }
        .forward(&x);
        for (a, b) in y.iter().zip(&want.data) {
            assert!((a - b.tanh()).abs() < 1e-5);
        }
    }

    #[test]
    fn module_names() {
        assert_eq!(
            module_name("/flow/flows.0/enc/in_layers.0/Conv"),
            "flow.flows.0.enc.in_layers.0"
        );
        assert_eq!(module_name("/dec/ups.2/ConvTranspose"), "dec.ups.2");
    }
}
