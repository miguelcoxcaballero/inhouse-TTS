//! `K82`: StyleTTS2-family acoustic model + iSTFTNet vocoder, compatible
//! with the Apache-2.0 Kokoro-82M v1.0 checkpoint.
//!
//! Pipeline: phoneme ids → ALBERT (PL-BERT) context → prosody predictor
//! (durations, F0, energy) → text encoder → length regulation → AdaIN
//! decoder → harmonic-plus-noise iSTFTNet generator → 24 kHz waveform.

use crate::dsp::{self, Rng, SourceParams, Stft};
use crate::gemm::{self, AData, AMat, BMat};
use crate::math;
use crate::ops::{
    self, AdaIn, AdaLayerNorm, BiLstm, Conv1d, ConvTranspose1d, DepthwiseUp2, Linear, LstmDir,
};
use crate::tensor::{Mat, Matrix};
use crate::weights::{err, Result, WeightFile};
use std::collections::BTreeMap;

pub const SAMPLE_RATE: u32 = 24_000;
/// Samples per predicted duration frame (24 kHz / 40 Hz).
pub const SAMPLES_PER_FRAME: usize = 600;
pub const STYLE_DIM: usize = 128;
pub const MAX_TOKENS: usize = 510;

const HIDDEN: usize = 512;
const BERT_HIDDEN: usize = 768;
const BERT_EMB: usize = 128;
const BERT_HEADS: usize = 12;
const BERT_LAYERS: usize = 12;
const BERT_FFN: usize = 2048;
const MAX_DUR: usize = 50;

/// Loading options.
#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    /// Pre-pack weight matrices for the CPU micro-kernel (faster, uses f32
    /// memory). When false, matrices stay compact (f16/int8) and are
    /// unpacked on every call.
    pub prepack: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            prepack: !cfg!(target_arch = "wasm32"),
        }
    }
}

struct Loader<'a> {
    f: &'a mut WeightFile,
    prepack: bool,
}

impl Loader<'_> {
    fn vec(&mut self, name: &str, n: usize) -> Result<Vec<f32>> {
        self.f.f32(name, n)
    }

    fn matrix(&mut self, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
        self.f.matrix(name, rows, cols, self.prepack)
    }

    fn linear(&mut self, p: &str, inp: usize, out: usize, bias: bool) -> Result<Linear> {
        Ok(Linear {
            w: self.matrix(&format!("{p}.weight"), out, inp)?,
            bias: if bias {
                Some(self.vec(&format!("{p}.bias"), out)?)
            } else {
                None
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        p: &str,
        cin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dil: usize,
        bias: bool,
    ) -> Result<Conv1d> {
        Ok(Conv1d {
            cin,
            cout,
            k,
            stride,
            pad,
            dil,
            w: self.matrix(&format!("{p}.weight"), cout, cin * k)?,
            bias: if bias {
                Some(self.vec(&format!("{p}.bias"), cout)?)
            } else {
                None
            },
        })
    }

    fn conv_t(
        &mut self,
        p: &str,
        cin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        pad: usize,
    ) -> Result<ConvTranspose1d> {
        let mut phases = Vec::with_capacity(stride);
        for r in 0..stride {
            let taps = if r < k { (k - r).div_ceil(stride) } else { 0 };
            let m = if taps > 0 {
                self.matrix(&format!("{p}.phase{r}"), cout, cin * taps)?
            } else {
                Matrix::from_f32(0, 0, Vec::new(), false)
            };
            phases.push((m, taps));
        }
        Ok(ConvTranspose1d {
            cin,
            cout,
            k,
            stride,
            pad,
            out_pad: 0,
            phases,
            bias: self.vec(&format!("{p}.bias"), cout)?,
        })
    }

    fn lstm(&mut self, p: &str, inp: usize, hidden: usize) -> Result<BiLstm> {
        let mut dir = |sfx: &str| -> Result<LstmDir> {
            Ok(LstmDir {
                w_ih: self.matrix(&format!("{p}.weight_ih_l0{sfx}"), 4 * hidden, inp)?,
                w_hh: self.vec(&format!("{p}.weight_hh_l0{sfx}"), 4 * hidden * hidden)?,
                bias: self.vec(&format!("{p}.bias_l0{sfx}"), 4 * hidden)?,
            })
        };
        let fwd = dir("")?;
        let bwd = dir("_reverse")?;
        Ok(BiLstm { hidden, fwd, bwd })
    }

    fn adain(&mut self, p: &str, channels: usize) -> Result<AdaIn> {
        Ok(AdaIn {
            fc: self.linear(&format!("{p}.fc"), STYLE_DIM, 2 * channels, true)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

/// StyleTTS2 `AdainResBlk1d` (LeakyReLU 0.2, optional ×2 upsampling).
struct AdainResBlk {
    norm1: AdaIn,
    norm2: AdaIn,
    conv1: Conv1d,
    conv2: Conv1d,
    shortcut: Option<Conv1d>,
    pool: Option<DepthwiseUp2>,
}

impl AdainResBlk {
    fn load(l: &mut Loader, p: &str, cin: usize, cout: usize, upsample: bool) -> Result<Self> {
        let pool = if upsample {
            let w = l.vec(&format!("{p}.pool.weight"), cin * 3)?;
            Some(DepthwiseUp2 {
                w: w.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
                bias: l.vec(&format!("{p}.pool.bias"), cin)?,
            })
        } else {
            None
        };
        Ok(AdainResBlk {
            norm1: l.adain(&format!("{p}.norm1"), cin)?,
            norm2: l.adain(&format!("{p}.norm2"), cout)?,
            conv1: l.conv(&format!("{p}.conv1"), cin, cout, 3, 1, 1, 1, true)?,
            conv2: l.conv(&format!("{p}.conv2"), cout, cout, 3, 1, 1, 1, true)?,
            shortcut: if cin != cout {
                Some(l.conv(&format!("{p}.conv1x1"), cin, cout, 1, 1, 0, 1, false)?)
            } else {
                None
            },
            pool,
        })
    }

    fn forward(&self, x: &Mat, s: &[f32]) -> Mat {
        let mut r = x.clone();
        self.norm1.forward_inplace(&mut r, s);
        math::leaky_relu_inplace(&mut r.data, 0.2);
        if let Some(p) = &self.pool {
            r = p.forward(&r);
        }
        let mut r = self.conv1.forward(&r);
        self.norm2.forward_inplace(&mut r, s);
        math::leaky_relu_inplace(&mut r.data, 0.2);
        let mut r = self.conv2.forward(&r);
        let sc = if self.pool.is_some() {
            ops::upsample2(x)
        } else {
            x.clone()
        };
        let sc = match &self.shortcut {
            Some(c) => c.forward(&sc),
            None => sc,
        };
        r.add_inplace(&sc);
        r.scale_inplace(std::f32::consts::FRAC_1_SQRT_2);
        r
    }
}

/// HiFi-GAN style residual block with AdaIN and Snake activations.
struct AdaInResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    adain1: Vec<AdaIn>,
    adain2: Vec<AdaIn>,
    alpha1: Vec<Vec<f32>>,
    alpha2: Vec<Vec<f32>>,
}

impl AdaInResBlock1 {
    fn load(l: &mut Loader, p: &str, ch: usize, k: usize, dil: [usize; 3]) -> Result<Self> {
        let mut b = AdaInResBlock1 {
            convs1: vec![],
            convs2: vec![],
            adain1: vec![],
            adain2: vec![],
            alpha1: vec![],
            alpha2: vec![],
        };
        for (i, &d) in dil.iter().enumerate() {
            b.convs1.push(l.conv(
                &format!("{p}.convs1.{i}"),
                ch,
                ch,
                k,
                1,
                d * (k - 1) / 2,
                d,
                true,
            )?);
            b.convs2.push(l.conv(
                &format!("{p}.convs2.{i}"),
                ch,
                ch,
                k,
                1,
                (k - 1) / 2,
                1,
                true,
            )?);
            b.adain1.push(l.adain(&format!("{p}.adain1.{i}"), ch)?);
            b.adain2.push(l.adain(&format!("{p}.adain2.{i}"), ch)?);
            b.alpha1.push(l.vec(&format!("{p}.alpha1.{i}"), ch)?);
            b.alpha2.push(l.vec(&format!("{p}.alpha2.{i}"), ch)?);
        }
        Ok(b)
    }

    fn snake(x: &mut Mat, alpha: &[f32]) {
        let t = x.cols;
        let ptr = crate::par::SendPtr(x.data.as_mut_ptr());
        let rows = x.rows;
        crate::par::for_each(rows.div_ceil(8), |blk| {
            for r in blk * 8..((blk + 1) * 8).min(rows) {
                // SAFETY: distinct rows per task.
                let row = unsafe { std::slice::from_raw_parts_mut(ptr.get().add(r * t), t) };
                math::snake_inplace(row, alpha[r]);
            }
        });
    }

    fn forward(&self, x: &Mat, s: &[f32]) -> Mat {
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let mut xt = x.clone();
            self.adain1[i].forward_inplace(&mut xt, s);
            Self::snake(&mut xt, &self.alpha1[i]);
            let mut xt = self.convs1[i].forward(&xt);
            self.adain2[i].forward_inplace(&mut xt, s);
            Self::snake(&mut xt, &self.alpha2[i]);
            let xt = self.convs2[i].forward(&xt);
            x.add_inplace(&xt);
        }
        x
    }
}

// ---------------------------------------------------------------------------
// ALBERT context encoder
// ---------------------------------------------------------------------------

struct Albert {
    word: Vec<f32>,
    pos: Vec<f32>,
    token_type: Vec<f32>,
    emb_ln: (Vec<f32>, Vec<f32>),
    map_in: Linear,
    q: Linear,
    k: Linear,
    v: Linear,
    dense: Linear,
    attn_ln: (Vec<f32>, Vec<f32>),
    ffn: Linear,
    ffn_out: Linear,
    full_ln: (Vec<f32>, Vec<f32>),
    vocab: usize,
}

impl Albert {
    fn load(l: &mut Loader, vocab: usize) -> Result<Self> {
        let p = "bert.encoder.albert_layer_groups.0.albert_layers.0";
        let h = BERT_HIDDEN;
        Ok(Albert {
            word: l.vec("bert.embeddings.word_embeddings.weight", vocab * BERT_EMB)?,
            pos: l.vec("bert.embeddings.position_embeddings.weight", 512 * BERT_EMB)?,
            token_type: l.vec("bert.embeddings.token_type_embeddings.weight", 2 * BERT_EMB)?
                [..BERT_EMB]
                .to_vec(),
            emb_ln: (
                l.vec("bert.embeddings.LayerNorm.weight", BERT_EMB)?,
                l.vec("bert.embeddings.LayerNorm.bias", BERT_EMB)?,
            ),
            map_in: l.linear(
                "bert.encoder.embedding_hidden_mapping_in",
                BERT_EMB,
                h,
                true,
            )?,
            q: l.linear(&format!("{p}.attention.query"), h, h, true)?,
            k: l.linear(&format!("{p}.attention.key"), h, h, true)?,
            v: l.linear(&format!("{p}.attention.value"), h, h, true)?,
            dense: l.linear(&format!("{p}.attention.dense"), h, h, true)?,
            attn_ln: (
                l.vec(&format!("{p}.attention.LayerNorm.weight"), h)?,
                l.vec(&format!("{p}.attention.LayerNorm.bias"), h)?,
            ),
            ffn: l.linear(&format!("{p}.ffn"), h, BERT_FFN, true)?,
            ffn_out: l.linear(&format!("{p}.ffn_output"), BERT_FFN, h, true)?,
            full_ln: (
                l.vec(&format!("{p}.full_layer_layer_norm.weight"), h)?,
                l.vec(&format!("{p}.full_layer_layer_norm.bias"), h)?,
            ),
            vocab,
        })
    }

    /// Returns the last hidden state as `[768, T]`.
    fn forward(&self, ids: &[u32]) -> Mat {
        let t = ids.len();
        let mut x = Mat::zeros(BERT_EMB, t);
        for (j, &id) in ids.iter().enumerate() {
            let id = (id as usize).min(self.vocab - 1);
            for d in 0..BERT_EMB {
                x.data[d * t + j] =
                    self.word[id * BERT_EMB + d] + self.pos[j * BERT_EMB + d] + self.token_type[d];
            }
        }
        ops::layer_norm_channels(&mut x, Some(&self.emb_ln.0), Some(&self.emb_ln.1), 1e-12);
        let mut h = self.map_in.forward(&x);
        for _ in 0..BERT_LAYERS {
            h = self.layer(&h);
        }
        h
    }

    fn layer(&self, x: &Mat) -> Mat {
        let t = x.cols;
        let q = self.q.forward(x);
        let k = self.k.forward(x);
        let v = self.v.forward(x);
        let dh = BERT_HIDDEN / BERT_HEADS;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut ctx = Mat::zeros(BERT_HIDDEN, t);
        let mut scores = vec![0f32; t * t];
        for head in 0..BERT_HEADS {
            let r0 = head * dh;
            // scores[i][j] = Σ_d q[d][i] k[d][j]
            let qa = AMat {
                m: t,
                k: dh,
                data: AData::F32 {
                    data: &q.data[r0 * t..],
                    rs: 1,
                    cs: t,
                },
            };
            gemm::gemm(
                &qa,
                &BMat::row_major(&k.data[r0 * t..], dh, t, t).into(),
                &mut scores,
                t,
                false,
            );
            for i in 0..t {
                let row = &mut scores[i * t..(i + 1) * t];
                let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                let mut sum = 0f32;
                for v in row.iter_mut() {
                    *v = math::exp((*v - max) * scale);
                    sum += *v;
                }
                let inv = 1.0 / sum;
                for v in row.iter_mut() {
                    *v *= inv;
                }
            }
            // ctx[d][i] = Σ_j v[d][j] p[i][j]
            let va = AMat {
                m: dh,
                k: t,
                data: AData::F32 {
                    data: &v.data[r0 * t..],
                    rs: t,
                    cs: 1,
                },
            };
            let pt = BMat {
                k: t,
                n: t,
                data: &scores,
                rs: 1,
                cs: t,
            };
            gemm::gemm(&va, &pt.into(), &mut ctx.data[r0 * t..], t, false);
        }
        let mut a = self.dense.forward(&ctx);
        a.add_inplace(x);
        ops::layer_norm_channels(&mut a, Some(&self.attn_ln.0), Some(&self.attn_ln.1), 1e-12);
        let mut f = self.ffn.forward(&a);
        for v in &mut f.data {
            *v = math::gelu_tanh(*v);
        }
        let mut out = self.ffn_out.forward(&f);
        out.add_inplace(&a);
        ops::layer_norm_channels(
            &mut out,
            Some(&self.full_ln.0),
            Some(&self.full_ln.1),
            1e-12,
        );
        out
    }
}

// ---------------------------------------------------------------------------
// Prosody predictor, text encoder, decoder
// ---------------------------------------------------------------------------

struct Predictor {
    dur_lstms: Vec<BiLstm>,
    dur_norms: Vec<AdaLayerNorm>,
    lstm: BiLstm,
    dur_proj: Linear,
    shared: BiLstm,
    f0: Vec<AdainResBlk>,
    n: Vec<AdainResBlk>,
    f0_proj: Conv1d,
    n_proj: Conv1d,
}

fn style_rows(x: &Mat, s: &[f32]) -> Mat {
    let mut st = Mat::zeros(s.len(), x.cols);
    for (r, &v) in s.iter().enumerate() {
        st.row_mut(r).fill(v);
    }
    Mat::vcat(&[x, &st])
}

impl Predictor {
    fn load(l: &mut Loader) -> Result<Self> {
        let cat = HIDDEN + STYLE_DIM;
        let mut dur_lstms = Vec::new();
        let mut dur_norms = Vec::new();
        for i in 0..3 {
            dur_lstms.push(l.lstm(
                &format!("predictor.text_encoder.lstms.{}", 2 * i),
                cat,
                HIDDEN / 2,
            )?);
            dur_norms.push(AdaLayerNorm {
                fc: l.linear(
                    &format!("predictor.text_encoder.lstms.{}.fc", 2 * i + 1),
                    STYLE_DIM,
                    2 * HIDDEN,
                    true,
                )?,
            });
        }
        let blocks = |l: &mut Loader, p: &str| -> Result<Vec<AdainResBlk>> {
            Ok(vec![
                AdainResBlk::load(l, &format!("{p}.0"), HIDDEN, HIDDEN, false)?,
                AdainResBlk::load(l, &format!("{p}.1"), HIDDEN, HIDDEN / 2, true)?,
                AdainResBlk::load(l, &format!("{p}.2"), HIDDEN / 2, HIDDEN / 2, false)?,
            ])
        };
        Ok(Predictor {
            dur_lstms,
            dur_norms,
            lstm: l.lstm("predictor.lstm", cat, HIDDEN / 2)?,
            dur_proj: l.linear(
                "predictor.duration_proj.linear_layer",
                HIDDEN,
                MAX_DUR,
                true,
            )?,
            shared: l.lstm("predictor.shared", cat, HIDDEN / 2)?,
            f0: blocks(l, "predictor.F0")?,
            n: blocks(l, "predictor.N")?,
            f0_proj: l.conv("predictor.F0_proj", HIDDEN / 2, 1, 1, 1, 0, 1, true)?,
            n_proj: l.conv("predictor.N_proj", HIDDEN / 2, 1, 1, 1, 0, 1, true)?,
        })
    }

    /// DurationEncoder: returns `[512+128, T]`.
    fn encode(&self, d_en: &Mat, s: &[f32]) -> Mat {
        let mut x = style_rows(d_en, s);
        for (lstm, norm) in self.dur_lstms.iter().zip(&self.dur_norms) {
            let mut y = lstm.forward(&x);
            norm.forward_inplace(&mut y, s);
            x = style_rows(&y, s);
        }
        x
    }

    fn durations(&self, d: &Mat, speed: f32) -> Vec<usize> {
        let x = self.lstm.forward(d);
        let logits = self.dur_proj.forward(&x);
        (0..d.cols)
            .map(|j| {
                let mut sum = 0f32;
                for r in 0..MAX_DUR {
                    sum += math::sigmoid(logits.at(r, j));
                }
                (sum / speed).round_ties_even().max(1.0) as usize
            })
            .collect()
    }

    fn f0_n(&self, en: &Mat, s: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let x = self.shared.forward(en);
        let mut f0 = x.clone();
        for b in &self.f0 {
            f0 = b.forward(&f0, s);
        }
        let f0 = self.f0_proj.forward(&f0);
        let mut n = x;
        for b in &self.n {
            n = b.forward(&n, s);
        }
        let n = self.n_proj.forward(&n);
        (f0.data, n.data)
    }
}

struct TextEncoder {
    emb: Vec<f32>,
    cnn: Vec<(Conv1d, Vec<f32>, Vec<f32>)>,
    lstm: BiLstm,
    vocab: usize,
}

impl TextEncoder {
    fn load(l: &mut Loader, vocab: usize) -> Result<Self> {
        let mut cnn = Vec::new();
        for i in 0..3 {
            let p = format!("text_encoder.cnn.{i}");
            cnn.push((
                l.conv(&format!("{p}.0"), HIDDEN, HIDDEN, 5, 1, 2, 1, true)?,
                l.vec(&format!("{p}.1.gamma"), HIDDEN)?,
                l.vec(&format!("{p}.1.beta"), HIDDEN)?,
            ));
        }
        Ok(TextEncoder {
            emb: l.vec("text_encoder.embedding.weight", vocab * HIDDEN)?,
            cnn,
            lstm: l.lstm("text_encoder.lstm", HIDDEN, HIDDEN / 2)?,
            vocab,
        })
    }

    fn forward(&self, ids: &[u32]) -> Mat {
        let t = ids.len();
        let mut x = Mat::zeros(HIDDEN, t);
        for (j, &id) in ids.iter().enumerate() {
            let id = (id as usize).min(self.vocab - 1);
            for d in 0..HIDDEN {
                x.data[d * t + j] = self.emb[id * HIDDEN + d];
            }
        }
        for (conv, g, b) in &self.cnn {
            x = conv.forward(&x);
            ops::layer_norm_channels(&mut x, Some(g), Some(b), 1e-5);
            math::leaky_relu_inplace(&mut x.data, 0.2);
        }
        self.lstm.forward(&x)
    }
}

struct Generator {
    source_w: Vec<f32>,
    source_b: f32,
    noise_convs: Vec<Conv1d>,
    noise_res: Vec<AdaInResBlock1>,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<AdaInResBlock1>,
    conv_post: Conv1d,
    stft: Stft,
}

const UP_RATES: [usize; 2] = [10, 6];
const UP_KERNELS: [usize; 2] = [20, 12];
const RES_KERNELS: [usize; 3] = [3, 7, 11];
const N_FFT: usize = 20;
const HOP: usize = 5;

impl Generator {
    fn load(l: &mut Loader) -> Result<Self> {
        let p = "decoder.generator";
        let mut g = Generator {
            source_w: l.vec(&format!("{p}.m_source.l_linear.weight"), 9)?,
            source_b: l.vec(&format!("{p}.m_source.l_linear.bias"), 1)?[0],
            noise_convs: vec![],
            noise_res: vec![],
            ups: vec![],
            resblocks: vec![],
            conv_post: l.conv(&format!("{p}.conv_post"), 128, N_FFT + 2, 7, 1, 3, 1, true)?,
            stft: Stft::new(N_FFT, HOP),
        };
        let mut ch = 512;
        for i in 0..2 {
            let (u, k) = (UP_RATES[i], UP_KERNELS[i]);
            g.ups
                .push(l.conv_t(&format!("{p}.ups.{i}"), ch, ch / 2, k, u, (k - u) / 2)?);
            ch /= 2;
            for (j, &rk) in RES_KERNELS.iter().enumerate() {
                g.resblocks.push(AdaInResBlock1::load(
                    l,
                    &format!("{p}.resblocks.{}", i * 3 + j),
                    ch,
                    rk,
                    [1, 3, 5],
                )?);
            }
            if i == 0 {
                let s = UP_RATES[1];
                g.noise_convs.push(l.conv(
                    &format!("{p}.noise_convs.0"),
                    N_FFT + 2,
                    ch,
                    s * 2,
                    s,
                    s.div_ceil(2),
                    1,
                    true,
                )?);
                g.noise_res.push(AdaInResBlock1::load(
                    l,
                    &format!("{p}.noise_res.0"),
                    ch,
                    7,
                    [1, 3, 5],
                )?);
            } else {
                g.noise_convs.push(l.conv(
                    &format!("{p}.noise_convs.1"),
                    N_FFT + 2,
                    ch,
                    1,
                    1,
                    0,
                    1,
                    true,
                )?);
                g.noise_res.push(AdaInResBlock1::load(
                    l,
                    &format!("{p}.noise_res.1"),
                    ch,
                    11,
                    [1, 3, 5],
                )?);
            }
        }
        Ok(g)
    }

    /// Harmonic source signal at 24 kHz for the given F0 curve.
    fn source(&self, f0: &[f32], rng: Option<&mut Rng>) -> Vec<f32> {
        let up = UP_RATES[0] * UP_RATES[1] * HOP;
        dsp::harmonic_source(
            f0,
            &SourceParams {
                sample_rate: SAMPLE_RATE as f32,
                upsample: up,
                weights: &self.source_w,
                bias: self.source_b,
                voiced_threshold: 10.0,
            },
            rng,
        )
    }

    /// Source spectrum `[22, frames]` (magnitude rows, then phase rows).
    fn source_spec(&self, src: &[f32]) -> Mat {
        let (mag, phase) = self.stft.forward(src);
        Mat::vcat(&[&mag, &phase])
    }

    fn forward(&self, x: &Mat, s: &[f32], har: &Mat) -> Vec<f32> {
        let mut x = x.clone();
        for i in 0..self.ups.len() {
            math::leaky_relu_inplace(&mut x.data, 0.1);
            let xs = self.noise_convs[i].forward(har);
            let xs = self.noise_res[i].forward(&xs, s);
            x = self.ups[i].forward(&x);
            if i == self.ups.len() - 1 {
                x = reflect_pad_left1(&x);
            }
            assert_eq!(x.cols, xs.cols, "source/upsample length mismatch");
            x.add_inplace(&xs);
            let mut acc = self.resblocks[i * 3].forward(&x, s);
            for j in 1..3 {
                acc.add_inplace(&self.resblocks[i * 3 + j].forward(&x, s));
            }
            acc.scale_inplace(1.0 / 3.0);
            x = acc;
        }
        math::leaky_relu_inplace(&mut x.data, 0.01);
        let y = self.conv_post.forward(&x);
        let bins = N_FFT / 2 + 1;
        let frames = y.cols;
        let mut mag = Mat::zeros(bins, frames);
        let mut phase = Mat::zeros(bins, frames);
        for k in 0..bins {
            for (m, v) in mag.row_mut(k).iter_mut().zip(y.row(k)) {
                *m = math::exp(*v);
            }
            for (p, v) in phase.row_mut(k).iter_mut().zip(y.row(bins + k)) {
                *p = math::sin(*v);
            }
        }
        self.stft.inverse(&mag, &phase)
    }
}

fn reflect_pad_left1(x: &Mat) -> Mat {
    let mut y = Mat::zeros(x.rows, x.cols + 1);
    for r in 0..x.rows {
        let src = x.row(r);
        let dst = y.row_mut(r);
        dst[0] = src[1];
        dst[1..].copy_from_slice(src);
    }
    y
}

struct Decoder {
    encode: AdainResBlk,
    decode: Vec<AdainResBlk>,
    f0_conv: Conv1d,
    n_conv: Conv1d,
    asr_res: Conv1d,
    generator: Generator,
}

impl Decoder {
    fn load(l: &mut Loader) -> Result<Self> {
        let cat = 1024 + 2 + 64;
        Ok(Decoder {
            encode: AdainResBlk::load(l, "decoder.encode", HIDDEN + 2, 1024, false)?,
            decode: vec![
                AdainResBlk::load(l, "decoder.decode.0", cat, 1024, false)?,
                AdainResBlk::load(l, "decoder.decode.1", cat, 1024, false)?,
                AdainResBlk::load(l, "decoder.decode.2", cat, 1024, false)?,
                AdainResBlk::load(l, "decoder.decode.3", cat, 512, true)?,
            ],
            f0_conv: l.conv("decoder.F0_conv", 1, 1, 3, 2, 1, 1, true)?,
            n_conv: l.conv("decoder.N_conv", 1, 1, 3, 2, 1, 1, true)?,
            asr_res: l.conv("decoder.asr_res.0", HIDDEN, 64, 1, 1, 0, 1, true)?,
            generator: Generator::load(l)?,
        })
    }

    /// Frame-rate decoder trunk: returns `[512, 2F]` generator input.
    fn trunk(&self, asr: &Mat, f0: &[f32], n: &[f32], s: &[f32]) -> Mat {
        let f0m = self
            .f0_conv
            .forward(&Mat::from_vec(1, f0.len(), f0.to_vec()));
        let nm = self.n_conv.forward(&Mat::from_vec(1, n.len(), n.to_vec()));
        let mut x = self.encode.forward(&Mat::vcat(&[asr, &f0m, &nm]), s);
        let asr_res = self.asr_res.forward(asr);
        let mut res = true;
        for block in &self.decode {
            if res {
                x = Mat::vcat(&[&x, &asr_res, &f0m, &nm]);
            }
            x = block.forward(&x, s);
            if block.pool.is_some() {
                res = false;
            }
        }
        x
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Per-call synthesis options.
#[derive(Clone, Copy, Debug)]
pub struct InferOptions {
    pub speed: f32,
    /// Seed of the excitation noise (`None` disables noise, for testing).
    pub seed: Option<u64>,
}

impl Default for InferOptions {
    fn default() -> Self {
        InferOptions {
            speed: 1.0,
            seed: Some(0x1405_e),
        }
    }
}

/// Synthesis result.
pub struct Output {
    pub audio: Vec<f32>,
    /// Frames (25 ms) assigned to every input token, including pads.
    pub durations: Vec<usize>,
}

/// Intermediate tensors, exposed for numerical tests against the reference.
#[derive(Default)]
pub struct Trace {
    pub tensors: BTreeMap<String, Vec<f32>>,
}

impl Trace {
    fn put(&mut self, name: &str, v: &[f32]) {
        self.tensors.insert(name.to_string(), v.to_vec());
    }
}

pub struct K82 {
    pub vocab: BTreeMap<String, u32>,
    bert: Albert,
    bert_encoder: Linear,
    predictor: Predictor,
    text_encoder: TextEncoder,
    decoder: Decoder,
}

impl K82 {
    pub fn load(mut f: WeightFile, opts: LoadOptions) -> Result<Self> {
        if f.header.model != "k82" {
            return err(format!("expected a k82 model, found {:?}", f.header.model));
        }
        let vocab: BTreeMap<String, u32> =
            serde_json::from_value(f.header.meta["vocab"].clone())
                .map_err(|e| crate::weights::Error(format!("bad vocab: {e}")))?;
        let n_tok = f.header.meta["n_token"].as_u64().unwrap_or(178) as usize;
        let mut l = Loader {
            f: &mut f,
            prepack: opts.prepack,
        };
        let bert = Albert::load(&mut l, n_tok)?;
        let bert_encoder = l.linear("bert_encoder", BERT_HIDDEN, HIDDEN, true)?;
        let predictor = Predictor::load(&mut l)?;
        let text_encoder = TextEncoder::load(&mut l, n_tok)?;
        let decoder = Decoder::load(&mut l)?;
        Ok(K82 {
            vocab,
            bert,
            bert_encoder,
            predictor,
            text_encoder,
            decoder,
        })
    }

    /// Maps a phoneme string to token ids (unknown symbols are skipped).
    pub fn tokenize(&self, phonemes: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(phonemes.len() + 2);
        ids.push(0);
        let mut buf = [0u8; 4];
        for ch in phonemes.chars() {
            if let Some(&id) = self.vocab.get(ch.encode_utf8(&mut buf) as &str) {
                ids.push(id);
            }
        }
        ids.push(0);
        ids
    }

    /// Synthesises audio for token ids (with leading/trailing pad 0) and a
    /// 256-d reference style (`[decoder 128 | prosody 128]`).
    pub fn infer(&self, ids: &[u32], style: &[f32], opts: InferOptions) -> Output {
        self.infer_traced(ids, style, opts, None)
    }

    pub fn infer_traced(
        &self,
        ids: &[u32],
        style: &[f32],
        opts: InferOptions,
        mut trace: Option<&mut Trace>,
    ) -> Output {
        assert_eq!(
            style.len(),
            2 * STYLE_DIM,
            "style vector must have 256 values"
        );
        let (s_dec, s_pro) = style.split_at(STYLE_DIM);
        let mut tr = |name: &str, v: &[f32]| {
            if let Some(t) = trace.as_deref_mut() {
                t.put(name, v);
            }
        };
        let mut prof = crate::profile::Profiler::new();
        let bert = self.bert.forward(ids);
        prof.mark("bert");
        let d_en = self.bert_encoder.forward(&bert);
        tr("d_en", &d_en.data);
        let d = self.predictor.encode(&d_en, s_pro);
        tr("d", &d.transpose().data);
        let durations = self.predictor.durations(&d, opts.speed.max(0.05));
        prof.mark("durations");
        let en = d.repeat_cols(&durations);
        let (f0, n) = self.predictor.f0_n(&en, s_pro);
        prof.mark("f0_n");
        tr("F0_pred", &f0);
        tr("N_pred", &n);
        let t_en = self.text_encoder.forward(ids);
        tr("t_en", &t_en.data);
        let asr = t_en.repeat_cols(&durations);
        prof.mark("text_encoder");
        let x = self.decoder.trunk(&asr, &f0, &n, s_dec);
        prof.mark("decoder");
        tr("dec_decode3", &x.data);
        let mut rng = opts.seed.map(Rng::new);
        let src = self.decoder.generator.source(&f0, rng.as_mut());
        tr("har_source", &src);
        let har = self.decoder.generator.source_spec(&src);
        tr("har", &har.data);
        prof.mark("source");
        let audio = self.decoder.generator.forward(&x, s_dec, &har);
        prof.mark("generator");
        prof.report();
        if let Some(t) = trace {
            t.put("audio", &audio);
        }
        Output { audio, durations }
    }

    /// Runs only the generator on given inputs (numerical tests).
    pub fn generator_only(&self, x: &Mat, style: &[f32], har: &Mat) -> Vec<f32> {
        self.decoder.generator.forward(x, &style[..STYLE_DIM], har)
    }
}

// ---------------------------------------------------------------------------
// Checkpoint conversion (PyTorch state dict → .ihw)
// ---------------------------------------------------------------------------

#[cfg(feature = "convert")]
pub mod convert {
    use super::*;
    use crate::torchload::TorchTensor;
    use crate::weights::{DType, Writer};
    use std::collections::HashMap;

    fn fold_weight_norm(g: &TorchTensor, v: &TorchTensor) -> Vec<f32> {
        // weight_norm(dim=0): w = g * v / ||v|| over all dims but the first.
        let rows = v.shape[0];
        let cols = v.data.len() / rows;
        let mut w = vec![0f32; v.data.len()];
        for r in 0..rows {
            let row = &v.data[r * cols..(r + 1) * cols];
            let norm = row
                .iter()
                .map(|x| (*x as f64) * (*x as f64))
                .sum::<f64>()
                .sqrt();
            let s = g.data[r] as f64 / norm.max(1e-30);
            for (o, x) in w[r * cols..(r + 1) * cols].iter_mut().zip(row) {
                *o = (*x as f64 * s) as f32;
            }
        }
        w
    }

    /// Converts a Kokoro-82M v1.0 state dict into a `.ihw` container.
    /// `matrix_dtype` applies to large weight matrices; the rest stays f32.
    pub fn convert(
        tensors: Vec<(String, TorchTensor)>,
        config: &serde_json::Value,
        matrix_dtype: DType,
    ) -> Result<Vec<u8>> {
        let mut map: HashMap<String, TorchTensor> = HashMap::new();
        for (name, t) in tensors {
            let name = name.replacen(".module.", ".", 1);
            map.insert(name, t);
        }
        // Fold weight norm pairs.
        let gs: Vec<String> = map
            .keys()
            .filter(|k| k.ends_with(".weight_g"))
            .cloned()
            .collect();
        for g in gs {
            let base = g.trim_end_matches(".weight_g").to_string();
            let gt = map.remove(&g).unwrap();
            let vt = map
                .remove(&format!("{base}.weight_v"))
                .ok_or_else(|| crate::weights::Error(format!("{base}: weight_v missing")))?;
            let w = fold_weight_norm(&gt, &vt);
            map.insert(
                format!("{base}.weight"),
                TorchTensor {
                    shape: vt.shape.clone(),
                    data: w,
                },
            );
        }
        // LSTM biases: b_ih + b_hh.
        let bihs: Vec<String> = map
            .keys()
            .filter(|k| k.contains(".bias_ih_l0"))
            .cloned()
            .collect();
        for bih in bihs {
            let bhh = bih.replace("bias_ih", "bias_hh");
            let a = map.remove(&bih).unwrap();
            let b = map
                .remove(&bhh)
                .ok_or_else(|| crate::weights::Error(format!("{bhh} missing")))?;
            let sum: Vec<f32> = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
            map.insert(
                bih.replace("bias_ih", "bias"),
                TorchTensor {
                    shape: a.shape,
                    data: sum,
                },
            );
        }
        // Transposed convolutions → polyphase matrices.
        for i in 0..2 {
            let name = format!("decoder.generator.ups.{i}.weight");
            let t = map
                .remove(&name)
                .ok_or_else(|| crate::weights::Error(format!("{name} missing")))?;
            let (cin, cout, k) = (t.shape[0], t.shape[1], t.shape[2]);
            for (r, (a, taps)) in ops::polyphase_weights(&t.data, cin, cout, k, UP_RATES[i])
                .into_iter()
                .enumerate()
            {
                if taps > 0 {
                    map.insert(
                        format!("decoder.generator.ups.{i}.phase{r}"),
                        TorchTensor {
                            shape: vec![cout, cin * taps],
                            data: a,
                        },
                    );
                }
            }
        }
        map.retain(|k, _| !k.starts_with("bert.pooler"));

        let mut meta = serde_json::json!({
            "name": "Inhouse K82",
            "compat": "Kokoro-82M v1.0 (Apache-2.0)",
            "sample_rate": SAMPLE_RATE,
            "n_token": config["n_token"],
            "vocab": config["vocab"],
        });
        meta["matrix_dtype"] = serde_json::to_value(matrix_dtype).unwrap();
        let mut w = Writer::new("k82", meta);
        let mut names: Vec<&String> = map.keys().collect();
        names.sort();
        for name in names {
            let t = &map[name];
            let rows = t.shape.first().copied().unwrap_or(1);
            let cols = t.data.len() / rows.max(1);
            let is_matrix =
                name.ends_with(".weight") || name.contains(".phase") || name.contains("weight_ih");
            let small = rows < 16
                || cols < 16
                || name.contains("embeddings")
                || name.contains("embedding.")
                || name.contains("pool.")
                || name.contains("l_linear");
            // int8 only where the network tolerates it (decoder/vocoder, 65% of
            // the parameters); the ALBERT and prosody stack stay in f16.
            let dtype = match (is_matrix && !small, matrix_dtype) {
                (false, _) => DType::F32,
                (true, DType::Q8) if !name.starts_with("decoder.") => DType::F16,
                (true, d) => d,
            };
            w.add(name, &t.shape, &t.data, dtype);
        }
        Ok(w.finish())
    }

    /// Converts a voice pack tensor `[510, 1, 256]` to `[510, 256]` values.
    pub fn voice(t: &TorchTensor) -> Result<Vec<f32>> {
        if t.data.len() % 256 != 0 {
            return err(format!("unexpected voice shape {:?}", t.shape));
        }
        Ok(t.data.clone())
    }
}
