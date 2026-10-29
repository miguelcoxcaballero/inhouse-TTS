//! Neural network layers on `[channels, time]` activations.

use crate::gemm::{self, Im2col};
use crate::math;
use crate::par;
use crate::tensor::{Mat, Matrix};

pub fn add_bias(y: &mut Mat, bias: &[f32]) {
    assert_eq!(bias.len(), y.rows);
    for (r, &b) in bias.iter().enumerate() {
        for v in y.row_mut(r) {
            *v += b;
        }
    }
}

/// 1-D convolution; weight is `cout × (cin·k)` (PyTorch layout flattened).
pub struct Conv1d {
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub dil: usize,
    pub w: Matrix,
    pub bias: Option<Vec<f32>>,
}

impl Conv1d {
    pub fn out_len(&self, t: usize) -> usize {
        let span = self.dil * (self.k - 1) + 1;
        let padded = t + 2 * self.pad;
        if padded < span {
            0
        } else {
            (padded - span) / self.stride + 1
        }
    }

    pub fn forward(&self, x: &Mat) -> Mat {
        assert_eq!(x.rows, self.cin, "conv1d input channels");
        let tout = self.out_len(x.cols);
        let mut y = Mat::zeros(self.cout, tout);
        if tout == 0 {
            return y;
        }
        if self.k == 1 && self.stride == 1 && self.pad == 0 {
            self.w.gemm(&x.as_b(), &mut y.data, tout, false);
        } else {
            let src = Im2col {
                x: &x.data,
                ld: x.cols,
                t_in: x.cols,
                channels: self.cin,
                taps: self.k,
                stride: self.stride,
                dil: self.dil,
                pad: self.pad,
                n: tout,
            };
            self.w.gemm(&src.into(), &mut y.data, tout, false);
        }
        if let Some(b) = &self.bias {
            add_bias(&mut y, b);
        }
        y
    }
}

/// Transposed 1-D convolution evaluated as `stride` polyphase convolutions.
///
/// Phase `r` owns output samples `j ≡ r (mod stride)` (before cropping) and
/// its matrix is `cout × (cin·taps)` with
/// `A_r[co][ci·taps + kt] = W[ci][co][r + (taps-1-kt)·stride]`.
pub struct ConvTranspose1d {
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub out_pad: usize,
    pub phases: Vec<(Matrix, usize)>,
    pub bias: Vec<f32>,
}

/// Re-arranges a PyTorch ConvTranspose1d weight `[cin][cout][k]` into the
/// polyphase matrices used by [`ConvTranspose1d`].
pub fn polyphase_weights(
    w: &[f32],
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
) -> Vec<(Vec<f32>, usize)> {
    assert_eq!(w.len(), cin * cout * k);
    (0..stride)
        .map(|r| {
            let taps = if r < k { (k - r).div_ceil(stride) } else { 0 };
            let mut a = vec![0f32; cout * cin * taps];
            for co in 0..cout {
                for ci in 0..cin {
                    for kt in 0..taps {
                        let tap = r + (taps - 1 - kt) * stride;
                        a[co * cin * taps + ci * taps + kt] = w[(ci * cout + co) * k + tap];
                    }
                }
            }
            (a, taps)
        })
        .collect()
}

impl ConvTranspose1d {
    pub fn out_len(&self, t: usize) -> usize {
        (t - 1) * self.stride + self.k + self.out_pad - 2 * self.pad
    }

    pub fn forward(&self, x: &Mat) -> Mat {
        assert_eq!(x.rows, self.cin, "conv_transpose1d input channels");
        let t = x.cols;
        let tout = self.out_len(t);
        let mut y = Mat::zeros(self.cout, tout);
        let s = self.stride;
        for (r, (a, taps)) in self.phases.iter().enumerate() {
            if *taps == 0 {
                continue;
            }
            let q = t + taps - 1;
            let mut tmp = Mat::zeros(self.cout, q);
            let src = Im2col {
                x: &x.data,
                ld: t,
                t_in: t,
                channels: self.cin,
                taps: *taps,
                stride: 1,
                dil: 1,
                pad: taps - 1,
                n: q,
            };
            a.gemm(&src.into(), &mut tmp.data, q, false);
            // Output o = q*s + r - pad.
            let first_q = if r >= self.pad {
                0
            } else {
                (self.pad - r).div_ceil(s)
            };
            for co in 0..self.cout {
                let src = tmp.row(co);
                let dst = y.row_mut(co);
                let mut qi = first_q;
                while qi < q {
                    let o = qi * s + r - self.pad;
                    if o >= tout {
                        break;
                    }
                    dst[o] = src[qi];
                    qi += 1;
                }
            }
        }
        add_bias(&mut y, &self.bias);
        y
    }
}

/// Depthwise ConvTranspose1d(k=3, stride=2, padding=1, output_padding=1).
pub struct DepthwiseUp2 {
    pub w: Vec<[f32; 3]>,
    pub bias: Vec<f32>,
}

impl DepthwiseUp2 {
    pub fn forward(&self, x: &Mat) -> Mat {
        let t = x.cols;
        let mut y = Mat::zeros(x.rows, 2 * t);
        for c in 0..x.rows {
            let [w0, w1, w2] = self.w[c];
            let b = self.bias[c];
            let xs = x.row(c);
            let ys = y.row_mut(c);
            for q in 0..t {
                let next = if q + 1 < t { xs[q + 1] } else { 0.0 };
                ys[2 * q] = w1 * xs[q] + b;
                ys[2 * q + 1] = w0 * next + w2 * xs[q] + b;
            }
        }
        y
    }
}

/// Nearest-neighbour ×2 upsampling along time.
pub fn upsample2(x: &Mat) -> Mat {
    let mut y = Mat::zeros(x.rows, x.cols * 2);
    for c in 0..x.rows {
        let xs = x.row(c);
        for (o, pair) in y.row_mut(c).chunks_exact_mut(2).enumerate() {
            pair[0] = xs[o];
            pair[1] = xs[o];
        }
    }
    y
}

/// Fully connected layer applied to every column of `[in, T]`.
pub struct Linear {
    pub w: Matrix,
    pub bias: Option<Vec<f32>>,
}

impl Linear {
    pub fn forward(&self, x: &Mat) -> Mat {
        assert_eq!(x.rows, self.w.cols, "linear input size");
        let mut y = Mat::zeros(self.w.rows, x.cols);
        self.w.gemm(&x.as_b(), &mut y.data, x.cols, false);
        if let Some(b) = &self.bias {
            add_bias(&mut y, b);
        }
        y
    }

    /// Applies the layer to a single vector.
    pub fn forward_vec(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0f32; self.w.rows];
        self.w.matvec(x, &mut y);
        if let Some(b) = &self.bias {
            for (v, bb) in y.iter_mut().zip(b) {
                *v += *bb;
            }
        }
        y
    }
}

/// Layer normalisation over channels (independently for every time step).
pub fn layer_norm_channels(x: &mut Mat, gamma: Option<&[f32]>, beta: Option<&[f32]>, eps: f32) {
    let (c, t) = (x.rows, x.cols);
    let mut mean = vec![0f32; t];
    let mut var = vec![0f32; t];
    for r in 0..c {
        for (m, v) in mean.iter_mut().zip(x.row(r)) {
            *m += *v;
        }
    }
    let inv_c = 1.0 / c as f32;
    for m in &mut mean {
        *m *= inv_c;
    }
    for r in 0..c {
        for ((s, v), m) in var.iter_mut().zip(x.row(r)).zip(&mean) {
            let d = *v - *m;
            *s += d * d;
        }
    }
    let inv_std: Vec<f32> = var.iter().map(|v| 1.0 / (v * inv_c + eps).sqrt()).collect();
    for r in 0..c {
        let g = gamma.map_or(1.0, |g| g[r]);
        let b = beta.map_or(0.0, |b| b[r]);
        for ((v, m), is) in x.row_mut(r).iter_mut().zip(&mean).zip(&inv_std) {
            *v = (*v - *m) * *is * g + b;
        }
    }
}

/// Instance normalisation of one channel (biased variance).
fn instance_norm_row(row: &mut [f32], eps: f32) -> (f32, f32) {
    let n = row.len().max(1) as f32;
    let mut sum = 0f64;
    for v in row.iter() {
        sum += *v as f64;
    }
    let mean = (sum / n as f64) as f32;
    let mut sq = 0f64;
    for v in row.iter() {
        let d = (*v - mean) as f64;
        sq += d * d;
    }
    let inv_std = 1.0 / ((sq / n as f64) as f32 + eps).sqrt();
    (mean, inv_std)
}

/// Adaptive instance normalisation: `(1 + γ) · IN(x) + β`, γ/β from a style.
pub struct AdaIn {
    pub fc: Linear,
}

impl AdaIn {
    /// Style projection `[γ; β]` (length `2C`).
    pub fn style(&self, s: &[f32]) -> Vec<f32> {
        self.fc.forward_vec(s)
    }

    pub fn forward_inplace(&self, x: &mut Mat, s: &[f32]) {
        let h = self.style(s);
        let c = x.rows;
        assert_eq!(h.len(), 2 * c, "AdaIN channels");
        let xp = par::SendPtr(x.data.as_mut_ptr());
        let t = x.cols;
        let rows_per = 8;
        par::for_each(c.div_ceil(rows_per), |blk| {
            for r in blk * rows_per..((blk + 1) * rows_per).min(c) {
                // SAFETY: each task owns distinct rows.
                let row = unsafe { std::slice::from_raw_parts_mut(xp.get().add(r * t), t) };
                let (mean, inv_std) = instance_norm_row(row, 1e-5);
                let g = (1.0 + h[r]) * inv_std;
                let b = h[c + r] - mean * g;
                for v in row.iter_mut() {
                    *v = *v * g + b;
                }
            }
        });
    }
}

/// Adaptive layer normalisation over channels.
pub struct AdaLayerNorm {
    pub fc: Linear,
}

impl AdaLayerNorm {
    pub fn forward_inplace(&self, x: &mut Mat, s: &[f32]) {
        let h = self.fc.forward_vec(s);
        let c = x.rows;
        let gamma: Vec<f32> = h[..c].iter().map(|g| 1.0 + g).collect();
        layer_norm_channels(x, Some(&gamma), Some(&h[c..]), 1e-5);
    }
}

/// One direction of an LSTM layer (PyTorch gate order i, f, g, o).
pub struct LstmDir {
    pub w_ih: Matrix,
    pub w_hh: Vec<f32>,
    pub bias: Vec<f32>,
}

/// Bidirectional single-layer LSTM over `[in, T]` → `[2H, T]`.
pub struct BiLstm {
    pub hidden: usize,
    pub fwd: LstmDir,
    pub bwd: LstmDir,
}

impl LstmDir {
    fn run(&self, x: &Mat, hidden: usize, reverse: bool) -> Mat {
        let t = x.cols;
        let g4 = 4 * hidden;
        let mut pre = Mat::zeros(g4, t);
        self.w_ih.gemm(&x.as_b(), &mut pre.data, t, false);
        add_bias(&mut pre, &self.bias);
        let pre = pre.transpose(); // [T, 4H]
        let mut h = vec![0f32; hidden];
        let mut c = vec![0f32; hidden];
        let mut gates = vec![0f32; g4];
        let mut out = Mat::zeros(t, hidden);
        for step in 0..t {
            let ti = if reverse { t - 1 - step } else { step };
            gemm::matvec(&self.w_hh, g4, hidden, &h, &mut gates);
            let p = pre.row(ti);
            for j in 0..hidden {
                let i = math::sigmoid(gates[j] + p[j]);
                let f = math::sigmoid(gates[hidden + j] + p[hidden + j]);
                let g = math::tanh(gates[2 * hidden + j] + p[2 * hidden + j]);
                let o = math::sigmoid(gates[3 * hidden + j] + p[3 * hidden + j]);
                c[j] = f * c[j] + i * g;
                h[j] = o * math::tanh(c[j]);
            }
            out.row_mut(ti).copy_from_slice(&h);
        }
        out.transpose()
    }
}

impl BiLstm {
    pub fn forward(&self, x: &Mat) -> Mat {
        let h = self.hidden;
        let (f, b) = par::join(|| self.fwd.run(x, h, false), || self.bwd.run(x, h, true));
        Mat::vcat(&[&f, &b])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_conv_t(
        x: &Mat,
        w: &[f32],
        cout: usize,
        k: usize,
        s: usize,
        p: usize,
        op: usize,
    ) -> Mat {
        let (cin, t) = (x.rows, x.cols);
        let full = (t - 1) * s + k + op;
        let tout = full - 2 * p;
        let mut y = Mat::zeros(cout, tout);
        for ci in 0..cin {
            for co in 0..cout {
                for ti in 0..t {
                    for kk in 0..k {
                        let j = ti * s + kk;
                        if j >= p && j - p < tout {
                            y.data[co * tout + j - p] +=
                                w[(ci * cout + co) * k + kk] * x.at(ci, ti);
                        }
                    }
                }
            }
        }
        y
    }

    fn naive_conv(x: &Mat, w: &[f32], cout: usize, k: usize, s: usize, p: usize, d: usize) -> Mat {
        let (cin, t) = (x.rows, x.cols);
        let tout = (t + 2 * p - d * (k - 1) - 1) / s + 1;
        let mut y = Mat::zeros(cout, tout);
        for co in 0..cout {
            for o in 0..tout {
                let mut acc = 0f32;
                for ci in 0..cin {
                    for kk in 0..k {
                        let ti = (o * s + kk * d) as isize - p as isize;
                        if ti >= 0 && (ti as usize) < t {
                            acc += w[(co * cin + ci) * k + kk] * x.at(ci, ti as usize);
                        }
                    }
                }
                y.data[co * tout + o] = acc;
            }
        }
        y
    }

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn conv1d_matches_naive() {
        for &(cin, cout, t, k, s, p, d) in &[
            (3, 5, 17, 3, 1, 1, 1),
            (22, 9, 61, 12, 6, 3, 1),
            (8, 8, 40, 7, 1, 9, 3),
            (4, 2, 9, 1, 1, 0, 1),
            (1, 1, 10, 3, 2, 1, 1),
        ] {
            let x = Mat::from_vec(cin, t, rnd(cin * t, 1));
            let w = rnd(cout * cin * k, 2);
            let conv = Conv1d {
                cin,
                cout,
                k,
                stride: s,
                pad: p,
                dil: d,
                w: Matrix::from_f32(cout, cin * k, w.clone(), false),
                bias: None,
            };
            let y = conv.forward(&x);
            let want = naive_conv(&x, &w, cout, k, s, p, d);
            assert_eq!((y.rows, y.cols), (want.rows, want.cols));
            for (a, b) in y.data.iter().zip(&want.data) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn conv_transpose_matches_naive() {
        for &(cin, cout, t, k, s, p, op) in &[
            (4, 3, 7, 20, 10, 5, 0),
            (5, 2, 9, 12, 6, 3, 0),
            (3, 3, 6, 3, 2, 1, 1),
            (2, 2, 5, 5, 2, 0, 0),
        ] {
            let x = Mat::from_vec(cin, t, rnd(cin * t, 3));
            let w = rnd(cin * cout * k, 4);
            let phases = polyphase_weights(&w, cin, cout, k, s)
                .into_iter()
                .map(|(a, taps)| (Matrix::from_f32(cout, cin * taps, a, false), taps))
                .collect();
            let ct = ConvTranspose1d {
                cin,
                cout,
                k,
                stride: s,
                pad: p,
                out_pad: op,
                phases,
                bias: vec![0.0; cout],
            };
            let y = ct.forward(&x);
            let want = naive_conv_t(&x, &w, cout, k, s, p, op);
            assert_eq!((y.rows, y.cols), (want.rows, want.cols));
            for (a, b) in y.data.iter().zip(&want.data) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn depthwise_up2_matches_naive() {
        let x = Mat::from_vec(2, 5, rnd(10, 5));
        let wf = rnd(6, 6);
        let dw = DepthwiseUp2 {
            w: vec![[wf[0], wf[1], wf[2]], [wf[3], wf[4], wf[5]]],
            bias: vec![0.0, 0.0],
        };
        let y = dw.forward(&x);
        for c in 0..2 {
            let xc = Mat::from_vec(1, 5, x.row(c).to_vec());
            let want = naive_conv_t(&xc, &wf[c * 3..c * 3 + 3], 1, 3, 2, 1, 1);
            for (a, b) in y.row(c).iter().zip(&want.data) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }
}
