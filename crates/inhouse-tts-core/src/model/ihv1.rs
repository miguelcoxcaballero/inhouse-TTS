//! `IHV1`: Inhouse TTS's own model (trained from scratch with
//! `inhouse-tts-train`). VITS-family synthesiser: rotary-attention text
//! encoder with accent and speaker conditioning, duration predictor,
//! normalising flow and a multi-band iSTFT decoder (4 sub-bands, PQMF).
//!
//! This is the inference graph; it mirrors `crates/inhouse-tts-train`.

use crate::dsp::{Rng, Stft};
use crate::math;
use crate::ops::{self, Conv1d, ConvTranspose1d};
use crate::tensor::{Mat, Matrix};
use crate::weights::{err, Result, WeightFile};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize)]
pub struct Ihv1Model {
    pub hidden: usize,
    pub filter: usize,
    pub heads: usize,
    pub enc_layers: usize,
    pub inter: usize,
    pub flows: usize,
    pub flow_layers: usize,
    pub gin: usize,
    pub dec_channels: usize,
    pub ups: Vec<usize>,
    pub up_kernels: Vec<usize>,
    pub res_kernels: Vec<usize>,
    pub res_dilations: Vec<usize>,
    pub istft_n_fft: usize,
    pub istft_hop: usize,
    pub subbands: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Tables {
    pub symbols: Vec<String>,
    pub speakers: Vec<String>,
    pub locales: Vec<String>,
    #[serde(default)]
    pub speaker_locale: BTreeMap<String, String>,
}

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
    qkv: Conv1d,
    out: Conv1d,
    n1: Norm,
    ff1: Conv1d,
    ff2: Conv1d,
    n2: Norm,
}

struct WaveNet {
    ins: Vec<Conv1d>,
    res_skip: Vec<Conv1d>,
    cond: Conv1d,
    hidden: usize,
}

struct Coupling {
    pre: Conv1d,
    wn: WaveNet,
    post: Conv1d,
}

struct ResBlock {
    c1: Vec<Conv1d>,
    c2: Vec<Conv1d>,
}

pub struct Ihv1 {
    pub model: Ihv1Model,
    pub tables: Tables,
    pub sample_rate: u32,
    emb: Mat,
    lang: Mat,
    layers: Vec<EncLayer>,
    enc_spk: Conv1d,
    enc_proj: Conv1d,
    flow: Vec<Coupling>,
    dp: [Conv1d; 3],
    dp_norm: [Norm; 2],
    dp_cond: Conv1d,
    dec_pre: Conv1d,
    dec_cond: Conv1d,
    ups: Vec<ConvTranspose1d>,
    res: Vec<ResBlock>,
    dec_post: Conv1d,
    spk: Mat,
    stft: Stft,
    pqmf: Vec<f32>,
}

/// Inference options.
#[derive(Clone, Copy, Debug)]
pub struct Ihv1Options {
    pub noise: f32,
    pub length_scale: f32,
    pub seed: u64,
}

impl Default for Ihv1Options {
    fn default() -> Self {
        Ihv1Options {
            noise: 0.667,
            length_scale: 1.0,
            seed: 1,
        }
    }
}

struct L<'a> {
    f: &'a mut WeightFile,
    prepack: bool,
}

impl L<'_> {
    fn conv(&mut self, name: &str, k: usize, dil: usize) -> Result<Conv1d> {
        let shape = self.f.get(&format!("{name}.weight"))?.shape.clone();
        let (cout, cin) = (shape[0], shape[1]);
        let w = self
            .f
            .matrix(&format!("{name}.weight"), cout, cin * k, self.prepack)?;
        let bias = self.f.f32(&format!("{name}.bias"), cout).ok();
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
        let shape = self.f.get(name)?.shape.clone();
        let v = self.f.f32(name, shape.iter().product())?;
        Ok(Mat::from_vec(shape[0], shape[1], v))
    }

    fn wavenet(&mut self, name: &str, hidden: usize, layers: usize) -> Result<WaveNet> {
        Ok(WaveNet {
            ins: (0..layers)
                .map(|i| self.conv(&format!("{name}.in.{i}"), 5, 1))
                .collect::<Result<_>>()?,
            res_skip: (0..layers)
                .map(|i| self.conv(&format!("{name}.res_skip.{i}"), 1, 1))
                .collect::<Result<_>>()?,
            cond: self.conv(&format!("{name}.cond"), 1, 1)?,
            hidden,
        })
    }
}

/// Pseudo-QMF synthesis filters `[subbands][taps + 1]` (same design as the
/// trainer).
fn pqmf_synthesis(subbands: usize, taps: usize, cutoff: f64, beta: f64) -> Vec<f32> {
    use std::f64::consts::PI;
    let bessel = |x: f64| {
        let (mut s, mut t) = (1.0, 1.0);
        for k in 1..60 {
            t *= (x / 2.0) * (x / 2.0) / (k * k) as f64;
            s += t;
        }
        s
    };
    let n = taps + 1;
    let wc = PI * cutoff;
    let mut out = vec![0f32; subbands * n];
    for k in 0..subbands {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        for i in 0..n {
            let m = i as f64 - 0.5 * taps as f64;
            let h = if m == 0.0 {
                cutoff
            } else {
                (wc * m).sin() / (PI * m)
            };
            let r = 2.0 * i as f64 / taps as f64 - 1.0;
            let w = bessel(beta * (1.0 - r * r).max(0.0).sqrt()) / bessel(beta);
            let ph = (2 * k + 1) as f64 * (PI / (2 * subbands) as f64) * m;
            out[k * n + i] = (2.0 * h * w * (ph - sign * PI / 4.0).cos()) as f32;
        }
    }
    out
}

fn add_col(x: &mut Mat, v: &[f32]) {
    for r in 0..x.rows {
        let b = v[r];
        for e in x.row_mut(r) {
            *e += b;
        }
    }
}

fn conv_vec(c: &Conv1d, v: &[f32]) -> Vec<f32> {
    c.forward(&Mat::from_vec(v.len(), 1, v.to_vec())).data
}

fn rows(x: &Mat, r0: usize, n: usize) -> Mat {
    Mat::from_vec(n, x.cols, x.data[r0 * x.cols..(r0 + n) * x.cols].to_vec())
}

impl Ihv1 {
    pub fn load(mut f: WeightFile, prepack: bool) -> Result<Ihv1> {
        if f.header.model != "ihv1" {
            return err(format!(
                "expected an ihv1 model, found {:?}",
                f.header.model
            ));
        }
        let meta = f.header.meta.clone();
        let model: Ihv1Model = serde_json::from_value(meta["config"]["model"].clone())
            .map_err(|e| crate::weights::Error(format!("bad config: {e}")))?;
        let tables: Tables = serde_json::from_value(meta["tables"].clone())
            .map_err(|e| crate::weights::Error(format!("bad tables: {e}")))?;
        let sample_rate = meta["sample_rate"].as_u64().unwrap_or(24_000) as u32;
        let m = model.clone();
        let mut l = L { f: &mut f, prepack };
        let layers = (0..m.enc_layers)
            .map(|i| -> Result<EncLayer> {
                let p = format!("enc.layers.{i}");
                Ok(EncLayer {
                    qkv: l.conv(&format!("{p}.qkv"), 1, 1)?,
                    out: l.conv(&format!("{p}.out"), 1, 1)?,
                    n1: l.norm(&format!("{p}.norm1"), m.hidden)?,
                    ff1: l.conv(&format!("{p}.ff1"), 3, 1)?,
                    ff2: l.conv(&format!("{p}.ff2"), 3, 1)?,
                    n2: l.norm(&format!("{p}.norm2"), m.hidden)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let flow = (0..m.flows)
            .map(|i| -> Result<Coupling> {
                let p = format!("flow.{i}");
                Ok(Coupling {
                    pre: l.conv(&format!("{p}.pre"), 1, 1)?,
                    wn: l.wavenet(&format!("{p}.wn"), m.inter, m.flow_layers)?,
                    post: l.conv(&format!("{p}.post"), 1, 1)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut ch = m.dec_channels;
        let mut ups = Vec::new();
        for (i, (&u, &k)) in m.ups.iter().zip(&m.up_kernels).enumerate() {
            let name = format!("dec.ups.{i}.weight");
            let w = l.f.f32(&name, ch * (ch / 2) * k)?;
            let bias = l.f.f32(&format!("dec.ups.{i}.bias"), ch / 2)?;
            let phases = ops::polyphase_weights(&w, ch, ch / 2, k, u)
                .into_iter()
                .map(|(a, taps)| (Matrix::from_f32(ch / 2, ch * taps, a, prepack), taps))
                .collect();
            ups.push(ConvTranspose1d {
                cin: ch,
                cout: ch / 2,
                k,
                stride: u,
                pad: (k - u) / 2,
                out_pad: 0,
                phases,
                bias,
            });
            ch /= 2;
        }
        let nres = m.ups.len() * m.res_kernels.len();
        let mut res = Vec::with_capacity(nres);
        for i in 0..nres {
            let k = m.res_kernels[i % m.res_kernels.len()];
            res.push(ResBlock {
                c1: m
                    .res_dilations
                    .iter()
                    .enumerate()
                    .map(|(j, &d)| l.conv(&format!("dec.res.{i}.c1.{j}"), k, d))
                    .collect::<Result<_>>()?,
                c2: (0..m.res_dilations.len())
                    .map(|j| l.conv(&format!("dec.res.{i}.c2.{j}"), k, 1))
                    .collect::<Result<_>>()?,
            });
        }
        Ok(Ihv1 {
            emb: l.mat("enc.emb.weight")?,
            lang: l.mat("enc.lang.weight")?,
            layers,
            enc_spk: l.conv("enc.spk", 1, 1)?,
            enc_proj: l.conv("enc.proj", 1, 1)?,
            flow,
            dp: [
                l.conv("dp.c1", 3, 1)?,
                l.conv("dp.c2", 3, 1)?,
                l.conv("dp.proj", 1, 1)?,
            ],
            dp_norm: [l.norm("dp.n1", 256)?, l.norm("dp.n2", 256)?],
            dp_cond: l.conv("dp.cond", 1, 1)?,
            dec_pre: l.conv("dec.pre", 7, 1)?,
            dec_cond: l.conv("dec.cond", 1, 1)?,
            ups,
            res,
            dec_post: l.conv("dec.post", 7, 1)?,
            spk: l.mat("spk.weight")?,
            stft: Stft::new(m.istft_n_fft, m.istft_hop),
            pqmf: pqmf_synthesis(m.subbands, 62, 0.142, 9.0),
            model,
            tables,
            sample_rate,
        })
    }

    /// Phonemes → token ids with blanks between symbols.
    pub fn encode(&self, phonemes: &str) -> Vec<usize> {
        let mut out = vec![0];
        let mut buf = [0u8; 4];
        for ch in phonemes.chars() {
            let s = ch.encode_utf8(&mut buf);
            if let Some(i) = self.tables.symbols.iter().position(|x| x == s) {
                out.push(i);
                out.push(0);
            }
        }
        out
    }

    pub fn speaker_index(&self, name: &str) -> Option<usize> {
        self.tables.speakers.iter().position(|s| s == name)
    }

    fn attention(&self, l: &EncLayer, x: &Mat) -> Mat {
        let m = &self.model;
        let (hd, t) = (m.hidden, x.cols);
        let nh = m.heads;
        let d = hd / nh;
        let half = d / 2;
        let qkv = l.qkv.forward(x);
        let mut y = Mat::zeros(hd, t);
        // RoPE tables.
        let mut cos = vec![0f32; t * half];
        let mut sin = vec![0f32; t * half];
        for p in 0..t {
            for i in 0..half {
                let inv = 1.0 / 10_000f64.powf(2.0 * i as f64 / d as f64);
                let a = p as f64 * inv;
                cos[p * half + i] = a.cos() as f32;
                sin[p * half + i] = a.sin() as f32;
            }
        }
        let rot = |src: usize, h: usize| -> Vec<f32> {
            // [T, d] with rotary embedding applied.
            let mut out = vec![0f32; t * d];
            for p in 0..t {
                for i in 0..half {
                    let a = qkv.at(src + h * d + i, p);
                    let b = qkv.at(src + h * d + half + i, p);
                    let (c, s) = (cos[p * half + i], sin[p * half + i]);
                    out[p * d + i] = a * c - b * s;
                    out[p * d + half + i] = a * s + b * c;
                }
            }
            out
        };
        let scale = 1.0 / (d as f32).sqrt();
        for h in 0..nh {
            let q = rot(0, h);
            let k = rot(hd, h);
            let mut att = vec![0f32; t * t];
            for i in 0..t {
                let mut mx = f32::NEG_INFINITY;
                for j in 0..t {
                    let s: f32 = (0..d).map(|e| q[i * d + e] * k[j * d + e]).sum::<f32>() * scale;
                    att[i * t + j] = s;
                    mx = mx.max(s);
                }
                let mut sum = 0.0;
                for j in 0..t {
                    let e = (att[i * t + j] - mx).exp();
                    att[i * t + j] = e;
                    sum += e;
                }
                for j in 0..t {
                    att[i * t + j] /= sum;
                }
            }
            for e in 0..d {
                let vrow = qkv.row(2 * hd + h * d + e);
                let yrow = y.row_mut(h * d + e);
                for i in 0..t {
                    let mut acc = 0.0;
                    for j in 0..t {
                        acc += att[i * t + j] * vrow[j];
                    }
                    yrow[i] = acc;
                }
            }
        }
        l.out.forward(&y)
    }

    fn wavenet(&self, wn: &WaveNet, mut x: Mat, g: &[f32]) -> Mat {
        let h = wn.hidden;
        let cond = conv_vec(&wn.cond, g);
        let n = wn.ins.len();
        let mut out = Mat::zeros(h, x.cols);
        for i in 0..n {
            let mut a = wn.ins[i].forward(&x);
            add_col(&mut a, &cond[2 * h * i..2 * h * (i + 1)]);
            let mut acts = Mat::zeros(h, x.cols);
            for r in 0..h {
                let (ta, sa) = (a.row(r), a.row(h + r));
                for ((o, t), s) in acts.row_mut(r).iter_mut().zip(ta).zip(sa) {
                    *o = math::tanh(*t) * math::sigmoid(*s);
                }
            }
            let rs = wn.res_skip[i].forward(&acts);
            if i + 1 < n {
                x.add_inplace(&rows(&rs, 0, h));
                out.add_inplace(&rows(&rs, h, h));
            } else {
                out.add_inplace(&rs);
            }
        }
        out
    }

    /// Synthesises phonemes for `speaker` with the accent of `locale`.
    pub fn synthesize(
        &self,
        phonemes: &str,
        speaker: usize,
        locale: usize,
        opts: Ihv1Options,
    ) -> Vec<f32> {
        let m = &self.model;
        let ids = self.encode(phonemes);
        let t = ids.len();
        let g = self.spk.row(speaker.min(self.spk.rows - 1)).to_vec();
        // Text encoder.
        let sq = (m.hidden as f32).sqrt();
        let mut x = Mat::zeros(m.hidden, t);
        let lang = self.lang.row(locale.min(self.lang.rows - 1));
        for (p, &id) in ids.iter().enumerate() {
            let e = self.emb.row(id.min(self.emb.rows - 1));
            for c in 0..m.hidden {
                x.data[c * t + p] = e[c] * sq + lang[c];
            }
        }
        let half = self.layers.len() / 2;
        let spk_h = conv_vec(&self.enc_spk, &g);
        for (i, l) in self.layers.iter().enumerate() {
            if i == half {
                add_col(&mut x, &spk_h);
            }
            let y = self.attention(l, &x);
            x.add_inplace(&y);
            l.n1.apply(&mut x);
            let mut f = l.ff1.forward(&x);
            for v in &mut f.data {
                *v = v.max(0.0);
            }
            let f = l.ff2.forward(&f);
            x.add_inplace(&f);
            l.n2.apply(&mut x);
        }
        let stats = self.enc_proj.forward(&x);
        let m_p = rows(&stats, 0, m.inter);
        let logs_p = rows(&stats, m.inter, m.inter);
        // Durations.
        let mut h = x.clone();
        add_col(&mut h, &conv_vec(&self.dp_cond, &g));
        let mut h = self.dp[0].forward(&h);
        for v in &mut h.data {
            *v = v.max(0.0);
        }
        self.dp_norm[0].apply(&mut h);
        let mut h = self.dp[1].forward(&h);
        for v in &mut h.data {
            *v = v.max(0.0);
        }
        self.dp_norm[1].apply(&mut h);
        let logw = self.dp[2].forward(&h);
        let dur: Vec<usize> = logw
            .data
            .iter()
            .map(|v| (math::exp(*v) * opts.length_scale).ceil().max(0.0) as usize)
            .collect();
        if dur.iter().sum::<usize>() == 0 {
            return Vec::new();
        }
        let m_p = m_p.repeat_cols(&dur);
        let logs_p = logs_p.repeat_cols(&dur);
        let mut rng = Rng::new(opts.seed);
        let mut z = m_p;
        for (v, ls) in z.data.iter_mut().zip(&logs_p.data) {
            *v += rng.gaussian() * math::exp(*ls) * opts.noise;
        }
        // Flow (reverse).
        let half_c = m.inter / 2;
        for c in self.flow.iter().rev() {
            // Flip channels.
            let mut f = Mat::zeros(z.rows, z.cols);
            for r in 0..z.rows {
                f.row_mut(r).copy_from_slice(z.row(z.rows - 1 - r));
            }
            let x0 = rows(&f, 0, half_c);
            let hh = c.pre.forward(&x0);
            let hh = self.wavenet(&c.wn, hh, &g);
            let shift = c.post.forward(&hh);
            let mut x1 = rows(&f, half_c, half_c);
            for (a, b) in x1.data.iter_mut().zip(&shift.data) {
                *a -= *b;
            }
            z = Mat::vcat(&[&x0, &x1]);
        }
        self.decode(&z, &g)
    }

    fn decode(&self, z: &Mat, g: &[f32]) -> Vec<f32> {
        let m = &self.model;
        let mut x = self.dec_pre.forward(z);
        add_col(&mut x, &conv_vec(&self.dec_cond, g));
        let kn = m.res_kernels.len();
        for (i, up) in self.ups.iter().enumerate() {
            math::leaky_relu_inplace(&mut x.data, 0.1);
            x = up.forward(&x);
            let mut acc: Option<Mat> = None;
            for j in 0..kn {
                let mut r = x.clone();
                let rb = &self.res[i * kn + j];
                for (a, b) in rb.c1.iter().zip(&rb.c2) {
                    let mut tt = r.clone();
                    math::leaky_relu_inplace(&mut tt.data, 0.1);
                    let mut tt = a.forward(&tt);
                    math::leaky_relu_inplace(&mut tt.data, 0.1);
                    let tt = b.forward(&tt);
                    r.add_inplace(&tt);
                }
                acc = Some(match acc {
                    Some(mut s) => {
                        s.add_inplace(&r);
                        s
                    }
                    None => r,
                });
            }
            x = acc.unwrap();
            x.scale_inplace(1.0 / kn as f32);
        }
        math::leaky_relu_inplace(&mut x.data, 0.01);
        // Reflection pad (1, 0).
        let mut xp = Mat::zeros(x.rows, x.cols + 1);
        for r in 0..x.rows {
            let (src, dst) = (x.row(r), xp.row_mut(r));
            dst[0] = src[1];
            dst[1..].copy_from_slice(src);
        }
        let y = self.dec_post.forward(&xp);
        let (sb, n) = (m.subbands, m.istft_n_fft);
        let bins = n / 2 + 1;
        let frames = y.cols;
        let mut bands: Vec<Vec<f32>> = Vec::with_capacity(sb);
        for s in 0..sb {
            let base = s * (n + 2);
            let mut mag = Mat::zeros(bins, frames);
            let mut phase = Mat::zeros(bins, frames);
            for k in 0..bins {
                for (o, v) in mag.row_mut(k).iter_mut().zip(y.row(base + k)) {
                    *o = math::exp(v.min(10.0));
                }
                for (o, v) in phase.row_mut(k).iter_mut().zip(y.row(base + bins + k)) {
                    *o = std::f32::consts::PI * math::sin(*v);
                }
            }
            bands.push(self.stft.inverse(&mag, &phase));
        }
        // PQMF synthesis: ×sb zero-insertion upsampling, then filtering.
        let len = bands[0].len();
        let taps = 62;
        let half = taps / 2;
        let out_len = len * sb;
        let mut out = vec![0f32; out_len];
        for (s, band) in bands.iter().enumerate() {
            let h = &self.pqmf[s * (taps + 1)..(s + 1) * (taps + 1)];
            for (q, &v) in band.iter().enumerate() {
                let v = v * sb as f32;
                // Impulse at upsampled index q·sb contributes to outputs
                // t = q·sb + half - i for taps i.
                let center = q * sb + half;
                for (i, &hi) in h.iter().enumerate() {
                    if center >= i {
                        let t = center - i;
                        if t < out_len {
                            out[t] += hi * v;
                        }
                    }
                }
            }
        }
        out
    }
}
