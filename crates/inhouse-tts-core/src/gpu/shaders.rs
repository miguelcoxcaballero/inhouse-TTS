//! WGSL compute kernels. They are plain strings compiled by the browser
//! (WebGPU) or by naga (native); every parameter is passed through a
//! 128-byte uniform block whose layout mirrors the `Params` builders in
//! [`super`].

/// Implicit-GEMM 1-D convolution `Y[co, o] (+)= Σ W[co, ci·taps + kt] ·
/// X[ci, q·stride + kt·dil − pad]`, tiled 64×64×16 with 4×4 outputs per
/// thread. It also evaluates polyphase transposed convolutions (one phase
/// per `workgroup_id.z`, output column `o = q·ostride + phase + oshift`).
///
/// `LOAD_A` is replaced by the weight loader for the matrix format.
const CONV: &str = r#"
struct P {
  m: u32, n: u32, k: u32, kpad: u32,
  taps: u32, stride: u32, dil: u32, pad: i32,
  tin: u32, up: u32, xld: u32, xoff: u32,
  yld: u32, yoff: u32, ostride: u32, oshift: i32,
  tvalid: u32, dshift: u32, flags: u32, wphase: u32,
  sphase: u32, boff: u32, slope: f32, oscale: f32,
  r0: u32, r1: u32, r2: u32, r3: u32, r4: u32, r5: u32, r6: u32, r7: u32,
}

// flags: 1 accumulate into Y, 2 add bias, 4 leaky ReLU on the input,
//        8 reflection pad (also write column 1 to column 0), 16 row scales.

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> w: array<WTYPE>;
@group(0) @binding(2) var<storage, read> aux: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;

// As[row · 4 + kq] holds k = 4kq .. 4kq+3 of a weight row; Bs[k · 16 + c]
// holds columns 4c .. 4c+3 of one input row: all tile accesses are vec4.
var<workgroup> As: array<vec4<f32>, 256>;
var<workgroup> Bs: array<vec4<f32>, 256>;

fn i8x4(v: u32) -> vec4<f32> {
  let s = vec4<i32>(bitcast<i32>(v << 24u), bitcast<i32>(v << 16u), bitcast<i32>(v << 8u), bitcast<i32>(v));
  return vec4<f32>(s >> vec4<u32>(24u));
}

fn store_row(co: u32, q0: u32, a: vec4<f32>, phase: u32) {
  if (co >= p.m) {
    return;
  }
  var sc = 1.0;
  if ((p.flags & 16u) != 0u) {
    sc = aux[phase * p.sphase + co];
  }
  var bias = 0.0;
  if ((p.flags & 2u) != 0u) {
    bias = aux[p.boff + co];
  }
  let rowy = p.yoff + co * p.yld;
  var av = a;
  for (var j = 0u; j < 4u; j = j + 1u) {
    let q = q0 + j;
    if (q >= p.n) {
      break;
    }
    let o = i32(q * p.ostride + phase) + p.oshift;
    if (o < 0 || o >= i32(p.tvalid)) {
      continue;
    }
    let val = av[j] * sc + bias;
    let d = rowy + u32(o) + p.dshift;
    var r = val;
    if ((p.flags & 1u) != 0u) {
      r = r + y[d];
    }
    y[d] = r * p.oscale;
    if ((p.flags & 8u) != 0u && o == 1) {
      var r0 = val;
      if ((p.flags & 1u) != 0u) {
        r0 = r0 + y[rowy];
      }
      y[rowy] = r0 * p.oscale;
    }
  }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let m0 = wid.y * BM;
  let n0 = wid.x * BN;
  let phase = wid.z;
  let wbase = phase * p.wphase;
  let lim = i32(p.tin * p.up);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);
  let ktiles = (p.k + BK - 1u) / BK;
  for (var kt = 0u; kt < ktiles; kt = kt + 1u) {
    let k0 = kt * BK;
    LOAD_A
    {
      let kr = tid / 16u;
      let c4 = tid % 16u;
      let kk = k0 + kr;
      var v = vec4<f32>(0.0);
      if (kk < p.k) {
        let ci = kk / p.taps;
        let tap = kk - ci * p.taps;
        let base = i32(tap * p.dil) - p.pad;
        let row = p.xoff + ci * p.xld;
        let q0 = n0 + c4 * 4u;
        for (var j = 0u; j < 4u; j = j + 1u) {
          let q = q0 + j;
          let u = i32(q * p.stride) + base;
          if (q < p.n && u >= 0 && u < lim) {
            var src = u32(u);
            if (p.up != 1u) {
              src = src / p.up;
            }
            var xv = x[row + src];
            if ((p.flags & 4u) != 0u && xv < 0.0) {
              xv = xv * p.slope;
            }
            v[j] = xv;
          }
        }
      }
      Bs[kr * 16u + c4] = v;
    }
    workgroupBarrier();
    for (var kq = 0u; kq < 4u; kq = kq + 1u) {
      let b0 = Bs[kq * 64u + lid.x];
      let b1 = Bs[kq * 64u + 16u + lid.x];
      let b2 = Bs[kq * 64u + 32u + lid.x];
      let b3 = Bs[kq * 64u + 48u + lid.x];
      let ar = lid.y * 16u + kq;
      let a0 = As[ar];
      let a1 = As[ar + 4u];
      let a2 = As[ar + 8u];
      let a3 = As[ar + 12u];
      acc0 = fma(vec4<f32>(a0.x), b0, fma(vec4<f32>(a0.y), b1, fma(vec4<f32>(a0.z), b2, fma(vec4<f32>(a0.w), b3, acc0))));
      acc1 = fma(vec4<f32>(a1.x), b0, fma(vec4<f32>(a1.y), b1, fma(vec4<f32>(a1.z), b2, fma(vec4<f32>(a1.w), b3, acc1))));
      acc2 = fma(vec4<f32>(a2.x), b0, fma(vec4<f32>(a2.y), b1, fma(vec4<f32>(a2.z), b2, fma(vec4<f32>(a2.w), b3, acc2))));
      acc3 = fma(vec4<f32>(a3.x), b0, fma(vec4<f32>(a3.y), b1, fma(vec4<f32>(a3.z), b2, fma(vec4<f32>(a3.w), b3, acc3))));
    }
    workgroupBarrier();
  }
  let q0 = n0 + lid.x * 4u;
  let c0 = m0 + lid.y * 4u;
  store_row(c0, q0, acc0, phase);
  store_row(c0 + 1u, q0, acc1, phase);
  store_row(c0 + 2u, q0, acc2, phase);
  store_row(c0 + 3u, q0, acc3, phase);
}
"#;

/// int8 weights, four per word, with one scale per row (applied in the
/// epilogue). Thread `tid` loads k-quad `tid % 4` of tile row `tid / 4`.
const LOAD_Q8: &str = r#"{
      let r = tid / 4u;
      let kq = tid % 4u;
      let row = m0 + r;
      let kw = k0 / 4u + kq;
      var v = vec4<f32>(0.0);
      if (row < p.m && kw * 4u < p.kpad) {
        v = i8x4(w[wbase + row * (p.kpad / 4u) + kw]);
      }
      As[tid] = v;
    }"#;

/// f16 weights, two per word.
const LOAD_F16: &str = r#"{
      let r = tid / 4u;
      let kq = tid % 4u;
      let row = m0 + r;
      let kw = k0 / 2u + kq * 2u;
      var v = vec4<f32>(0.0);
      if (row < p.m && kw * 2u < p.kpad) {
        let base = wbase + row * (p.kpad / 2u) + kw;
        v = vec4<f32>(unpack2x16float(w[base]), unpack2x16float(w[base + 1u]));
      }
      As[tid] = v;
    }"#;

/// f32 weights (rows padded to a multiple of 4).
const LOAD_F32: &str = r#"{
      let r = tid / 4u;
      let kq = tid % 4u;
      let row = m0 + r;
      let kk = k0 + kq * 4u;
      var v = vec4<f32>(0.0);
      if (row < p.m && kk < p.kpad) {
        let base = wbase + row * p.kpad + kk;
        v = vec4<f32>(w[base], w[base + 1u], w[base + 2u], w[base + 3u]);
      }
      As[tid] = v;
    }"#;

/// Source of the convolution kernel for weights stored as `fmt`
/// (0 = f32, 1 = f16, 2 = int8).
pub fn conv(fmt: usize) -> String {
    let (ty, load) = match fmt {
        0 => ("f32", LOAD_F32),
        1 => ("u32", LOAD_F16),
        _ => ("u32", LOAD_Q8),
    };
    CONV.replace("WTYPE", ty).replace("LOAD_A", load)
}

/// Same range reduction and polynomial as `math::sin`, so CPU and GPU agree.
const SIN: &str = r#"
fn sin_(x: f32) -> f32 {
  let k = round(x * 0.31830987);
  let r = ((x - k * 3.140625) - k * 0.0009675026) - k * 1.5099579e-7;
  let r2 = r * r;
  var q = 2.601903e-6;
  q = q * r2 - 1.9807419e-4;
  q = q * r2 + 8.333066e-3;
  q = q * r2 - 1.6666657e-1;
  let s = r + r * r2 * q;
  let odd = f32(i32(k) & 1);
  return s * (1.0 - 2.0 * odd);
}
"#;

/// Adaptive instance norm `(1 + γ)·IN(x) + β` of one channel per workgroup,
/// fused with LeakyReLU (`act` 1) or Snake (`act` 2). The style buffer holds
/// `[γ (c) | β (c) | α (c)]` at `soff`.
const ADAIN: &str = r#"
struct P {
  c: u32, t: u32, xld: u32, xoff: u32,
  yld: u32, yoff: u32, soff: u32, act: u32,
  slope: f32, eps: f32, r0: u32, r1: u32,
  r2: u32, r3: u32, r4: u32, r5: u32,
  r6: u32, r7: u32, r8: u32, r9: u32,
  r10: u32, r11: u32, r12: u32, r13: u32,
  r14: u32, r15: u32, r16: u32, r17: u32,
  r18: u32, r19: u32, r20: u32, r21: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: array<f32>;
BINDINGS

var<workgroup> red: array<f32, 256>;

SIN

fn sum256(v: f32, tid: u32) -> f32 {
  red[tid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      red[tid] = red[tid] + red[tid + s];
    }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32) {
  let ch = wid.x;
  let xr = p.xoff + ch * p.xld;
  var s = 0.0;
  for (var i = tid; i < p.t; i = i + 256u) {
    s = s + SRC[xr + i];
  }
  let mean = sum256(s, tid) / f32(p.t);
  var q = 0.0;
  for (var i = tid; i < p.t; i = i + 256u) {
    let d = SRC[xr + i] - mean;
    q = q + d * d;
  }
  let vr = sum256(q, tid) / f32(p.t);
  let g = (1.0 + st[p.soff + ch]) / sqrt(vr + p.eps);
  let b = st[p.soff + p.c + ch] - mean * g;
  let alpha = st[p.soff + 2u * p.c + ch];
  let yr = p.yoff + ch * p.yld;
  for (var i = tid; i < p.t; i = i + 256u) {
    var v = SRC[xr + i] * g + b;
    if (p.act == 1u) {
      if (v < 0.0) {
        v = v * p.slope;
      }
    } else if (p.act == 2u) {
      let sn = sin_(alpha * v);
      v = v + sn * sn / alpha;
    }
    y[yr + i] = v;
  }
}
"#;

/// AdaIN kernel, out of place (`X` read-only, `Y` written) or in place.
pub fn adain(in_place: bool) -> String {
    let (bindings, src) = if in_place {
        (
            "@group(0) @binding(2) var<storage, read_write> y: array<f32>;",
            "y",
        )
    } else {
        (
            "@group(0) @binding(2) var<storage, read> x: array<f32>;\n\
             @group(0) @binding(3) var<storage, read_write> y: array<f32>;",
            "x",
        )
    };
    ADAIN
        .replace("BINDINGS", bindings)
        .replace("SIN", SIN)
        .replace("SRC", src)
}

/// Depthwise ConvTranspose1d(k=3, stride=2, padding=1, output_padding=1).
/// `wts` holds `[w0 w1 w2]` per channel followed by the biases.
pub const DWUP2: &str = r#"
struct P {
  c: u32, t: u32, xld: u32, xoff: u32,
  yld: u32, yoff: u32, r0: u32, r1: u32,
  r2: u32, r3: u32, r4: u32, r5: u32,
  r6: u32, r7: u32, r8: u32, r9: u32,
  r10: u32, r11: u32, r12: u32, r13: u32,
  r14: u32, r15: u32, r16: u32, r17: u32,
  r18: u32, r19: u32, r20: u32, r21: u32,
  r22: u32, r23: u32, r24: u32, r25: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> wts: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x;
  let ch = gid.y;
  if (q >= p.t || ch >= p.c) {
    return;
  }
  let w0 = wts[ch * 3u];
  let w1 = wts[ch * 3u + 1u];
  let w2 = wts[ch * 3u + 2u];
  let b = wts[3u * p.c + ch];
  let xr = p.xoff + ch * p.xld;
  let xq = x[xr + q];
  var nx = 0.0;
  if (q + 1u < p.t) {
    nx = x[xr + q + 1u];
  }
  let yr = p.yoff + ch * p.yld + 2u * q;
  y[yr] = w1 * xq + b;
  y[yr + 1u] = w0 * nx + w2 * xq + b;
}
"#;

/// `Y = (X (+ Y)) · scale` over `n` contiguous elements.
pub const ELTWISE: &str = r#"
struct P {
  n: u32, xoff: u32, yoff: u32, flags: u32,
  scale: f32, r0: u32, r1: u32, r2: u32,
  r3: u32, r4: u32, r5: u32, r6: u32,
  r7: u32, r8: u32, r9: u32, r10: u32,
  r11: u32, r12: u32, r13: u32, r14: u32,
  r15: u32, r16: u32, r17: u32, r18: u32,
  r19: u32, r20: u32, r21: u32, r22: u32,
  r23: u32, r24: u32, r25: u32, r26: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let step = nwg.x * 256u;
  for (var i = gid.x; i < p.n; i = i + step) {
    var v = x[p.xoff + i];
    if ((p.flags & 1u) != 0u) {
      v = v + y[p.yoff + i];
    }
    y[p.yoff + i] = v * p.scale;
  }
}
"#;
