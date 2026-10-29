//! IHV1, the first Inhouse TTS model: an end-to-end VITS-family
//! synthesiser (text encoder with rotary attention, posterior encoder,
//! normalising flow, monotonic alignment search, duration predictor) with a
//! multi-band iSTFT decoder (4 sub-bands × iSTFT, PQMF synthesis).
//!
//! Speakers and accents are global conditions: a speaker embedding feeds the
//! posterior encoder, flow, duration predictor and decoder; a locale (accent)
//! embedding is added to every text-encoder input.

use crate::config::{Config, ModelConfig};
use crate::dsp;
use crate::mas;
use crate::nn::{drop, ChannelNorm, Conv, ConvT, Embed, EncoderLayer, ResBlock, WaveNet};
use burn::module::Module;
use burn::tensor::module::{conv1d, conv_transpose1d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions, PadMode};
use burn::tensor::{activation, Device, Distribution, Int, Tensor, TensorData};

/// Fixed (non-trainable) filters, created once per device.
#[derive(Clone)]
pub struct Consts {
    pub istft_basis: Tensor<3>,
    pub pqmf_syn: Tensor<3>,
    pub updown: Tensor<3>,
    pub stft_basis: Tensor<3>,
    pub mel_bank: Tensor<2>,
    pub istft_n_fft: usize,
    pub istft_hop: usize,
    pub subbands: usize,
    pub n_fft: usize,
    pub hop: usize,
    pub taps: usize,
}

impl Consts {
    pub fn new(c: &Config, dev: &Device) -> Consts {
        let m = &c.model;
        let n = m.istft_n_fft;
        let bins = n / 2 + 1;
        let istft_basis =
            Tensor::<1>::from_floats(dsp::istft_basis(n).as_slice(), dev).reshape([2 * bins, 1, n]);
        let taps = 62;
        let (_, syn) = dsp::pqmf(m.subbands, taps, 0.142, 9.0);
        let sb = m.subbands;
        let pqmf_syn = Tensor::<1>::from_floats(syn.as_slice(), dev).reshape([1, sb, taps + 1]);
        let mut ud = vec![0f32; sb * sb * sb];
        for k in 0..sb {
            ud[(k * sb + k) * sb] = sb as f32;
        }
        let updown = Tensor::<1>::from_floats(ud.as_slice(), dev).reshape([sb, sb, sb]);
        let a = &c.audio;
        let fb = a.n_fft / 2 + 1;
        let stft_basis = Tensor::<1>::from_floats(dsp::stft_basis(a.n_fft).as_slice(), dev)
            .reshape([2 * fb, 1, a.n_fft]);
        let mel_bank = Tensor::<1>::from_floats(
            dsp::mel_bank(a.sample_rate, a.n_fft, a.n_mels, 0.0, a.fmax as f64).as_slice(),
            dev,
        )
        .reshape([a.n_mels, fb]);
        Consts {
            istft_basis,
            pqmf_syn,
            updown,
            stft_basis,
            mel_bank,
            istft_n_fft: n,
            istft_hop: m.istft_hop,
            subbands: sb,
            n_fft: a.n_fft,
            hop: a.hop,
            taps,
        }
    }

    /// Magnitude spectrogram `[B, n_fft/2 + 1, T/hop]` of waveforms
    /// `[B, 1, T]`: Hann window, frames centred by reflect padding (the
    /// framing of [`dsp::spectrogram`]).
    pub fn linear(&self, y: Tensor<3>) -> Tensor<3> {
        let pad = (self.n_fft - self.hop) / 2;
        let y = y.pad([(pad, pad)], PadMode::Reflect);
        let s = conv1d(
            y,
            self.stft_basis.clone(),
            None,
            ConvOptions::new([self.hop], [0], [1], 1),
        );
        let bins = self.n_fft / 2 + 1;
        let re = s.clone().narrow(1, 0, bins);
        let im = s.narrow(1, bins, bins);
        (re.powi_scalar(2) + im.powi_scalar(2) + 1e-9).sqrt()
    }

    /// Log-mel spectrogram `[B, n_mels, T/hop]` (differentiable).
    pub fn mel(&self, y: Tensor<3>) -> Tensor<3> {
        let mag = self.linear(y);
        let [b, bins, _] = mag.dims();
        let nm = self.mel_bank.dims()[0];
        let mel = self
            .mel_bank
            .clone()
            .unsqueeze_dim::<3>(0)
            .expand([b, nm, bins])
            .matmul(mag);
        mel.clamp_min(1e-5).log()
    }
}

/// Text (phoneme) encoder producing the prior statistics.
#[derive(Module, Debug)]
pub struct TextEncoder {
    pub emb: Embed,
    pub lang: Embed,
    pub layers: Vec<EncoderLayer>,
    pub spk: Conv,
    pub proj: Conv,
    pub hidden: usize,
    pub inter: usize,
}

impl TextEncoder {
    fn new(dev: &Device, m: &ModelConfig, vocab: usize, langs: usize) -> TextEncoder {
        TextEncoder {
            emb: Embed::new(dev, vocab, m.hidden, (m.hidden as f64).powf(-0.5)),
            lang: Embed::new(dev, langs.max(1), m.hidden, 0.1),
            layers: (0..m.enc_layers)
                .map(|_| EncoderLayer::new(dev, m.hidden, m.filter, m.heads))
                .collect(),
            spk: Conv::zeros(dev, m.gin, m.hidden),
            proj: Conv::pointwise(dev, m.hidden, 2 * m.inter),
            hidden: m.hidden,
            inter: m.inter,
        }
    }

    /// Returns `(x, m_p, logs_p)` at text rate.
    fn forward(
        &self,
        ids: Tensor<2, Int>,
        lang: Tensor<1, Int>,
        g: Tensor<3>,
        mask: Tensor<3>,
        dropout: f64,
        train: bool,
    ) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        let x = self.emb.forward(ids) * (self.hidden as f64).sqrt();
        let mut x = x.swap_dims(1, 2) + self.lang.global(lang);
        x = x * mask.clone();
        let half = self.layers.len() / 2;
        for (i, l) in self.layers.iter().enumerate() {
            if i == half {
                // Speaker conditioning in the middle of the stack (VITS2).
                x = x + self.spk.forward(g.clone());
            }
            x = l.forward(x, mask.clone(), dropout, train);
        }
        let stats = self.proj.forward(x.clone()) * mask;
        let m = stats.clone().narrow(1, 0, self.inter);
        let logs = stats.narrow(1, self.inter, self.inter);
        (x, m, logs)
    }
}

/// Posterior encoder: linear spectrogram → latent `z`.
#[derive(Module, Debug)]
pub struct PosteriorEncoder {
    pub pre: Conv,
    pub wn: WaveNet,
    pub proj: Conv,
    pub inter: usize,
}

impl PosteriorEncoder {
    fn new(dev: &Device, m: &ModelConfig, bins: usize) -> PosteriorEncoder {
        PosteriorEncoder {
            pre: Conv::pointwise(dev, bins, m.inter),
            wn: WaveNet::new(dev, m.inter, 5, m.post_layers, m.gin),
            proj: Conv::pointwise(dev, m.inter, 2 * m.inter),
            inter: m.inter,
        }
    }

    /// Returns `(z, m_q, logs_q)`.
    fn forward(
        &self,
        spec: Tensor<3>,
        mask: Tensor<3>,
        g: Tensor<3>,
    ) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        let x = self.pre.forward(spec) * mask.clone();
        let x = self.wn.forward(x, mask.clone(), Some(g));
        let stats = self.proj.forward(x) * mask.clone();
        let m = stats.clone().narrow(1, 0, self.inter);
        let logs = stats.narrow(1, self.inter, self.inter);
        let eps = Tensor::<3>::random(m.dims(), Distribution::Normal(0.0, 1.0), &m.device());
        let z = (m.clone() + eps * logs.clone().exp()) * mask;
        (z, m, logs)
    }
}

/// Mean-only affine coupling layer followed by a channel flip.
#[derive(Module, Debug)]
pub struct Coupling {
    pub pre: Conv,
    pub wn: WaveNet,
    pub post: Conv,
    pub half: usize,
}

impl Coupling {
    fn new(dev: &Device, m: &ModelConfig) -> Coupling {
        let half = m.inter / 2;
        Coupling {
            pre: Conv::pointwise(dev, half, m.inter),
            wn: WaveNet::new(dev, m.inter, 5, m.flow_layers, m.gin),
            post: Conv::zeros(dev, m.inter, half),
            half,
        }
    }

    fn forward(&self, x: Tensor<3>, mask: Tensor<3>, g: Tensor<3>, reverse: bool) -> Tensor<3> {
        let x0 = x.clone().narrow(1, 0, self.half);
        let x1 = x.narrow(1, self.half, self.half);
        let h = self.pre.forward(x0.clone()) * mask.clone();
        let h = self.wn.forward(h, mask.clone(), Some(g));
        let shift = self.post.forward(h) * mask.clone();
        let x1 = if reverse {
            (x1 - shift) * mask
        } else {
            (x1 + shift) * mask
        };
        Tensor::cat(vec![x0, x1], 1)
    }
}

#[derive(Module, Debug)]
pub struct Flow {
    pub layers: Vec<Coupling>,
}

impl Flow {
    pub fn forward(
        &self,
        mut x: Tensor<3>,
        mask: Tensor<3>,
        g: Tensor<3>,
        reverse: bool,
    ) -> Tensor<3> {
        if !reverse {
            for l in &self.layers {
                x = l.forward(x, mask.clone(), g.clone(), false).flip([1]);
            }
        } else {
            for l in self.layers.iter().rev() {
                x = l.forward(x.flip([1]), mask.clone(), g.clone(), true);
            }
        }
        x
    }
}

/// Deterministic log-duration predictor.
#[derive(Module, Debug)]
pub struct DurationPredictor {
    pub c1: Conv,
    pub n1: ChannelNorm,
    pub c2: Conv,
    pub n2: ChannelNorm,
    pub proj: Conv,
    pub cond: Conv,
}

impl DurationPredictor {
    fn new(dev: &Device, m: &ModelConfig) -> DurationPredictor {
        let f = 256;
        DurationPredictor {
            c1: Conv::same(dev, m.hidden, f, 3, 1, false),
            n1: ChannelNorm::new(dev, f),
            c2: Conv::same(dev, f, f, 3, 1, false),
            n2: ChannelNorm::new(dev, f),
            proj: Conv::pointwise(dev, f, 1),
            cond: Conv::pointwise(dev, m.gin, m.hidden),
        }
    }

    fn forward(
        &self,
        x: Tensor<3>,
        mask: Tensor<3>,
        g: Tensor<3>,
        dropout: f64,
        train: bool,
    ) -> Tensor<3> {
        let x = x + self.cond.forward(g);
        let x = self.c1.forward(x * mask.clone());
        let x = drop(self.n1.forward(activation::relu(x)), dropout, train);
        let x = self.c2.forward(x * mask.clone());
        let x = drop(self.n2.forward(activation::relu(x)), dropout, train);
        self.proj.forward(x * mask.clone()) * mask
    }
}

/// Multi-band iSTFT decoder.
#[derive(Module, Debug)]
pub struct Decoder {
    pub pre: Conv,
    pub cond: Conv,
    pub ups: Vec<ConvT>,
    pub res: Vec<ResBlock>,
    pub post: Conv,
    pub kernels: usize,
}

impl Decoder {
    fn new(dev: &Device, m: &ModelConfig) -> Decoder {
        let mut ch = m.dec_channels;
        let mut ups = Vec::new();
        let mut res = Vec::new();
        for (&u, &k) in m.ups.iter().zip(&m.up_kernels) {
            ups.push(ConvT::new(dev, ch, ch / 2, k, u));
            ch /= 2;
            for &rk in &m.res_kernels {
                res.push(ResBlock::new(dev, ch, rk, &m.res_dilations));
            }
        }
        let out = m.subbands * (m.istft_n_fft + 2);
        Decoder {
            pre: Conv::same(dev, m.inter, m.dec_channels, 7, 1, true),
            cond: Conv::pointwise(dev, m.gin, m.dec_channels),
            ups,
            res,
            post: Conv::same(dev, ch, out, 7, 1, true),
            kernels: m.res_kernels.len(),
        }
    }

    /// `z` `[B, inter, T]` → waveform `[B, 1, T · hop]`.
    pub fn forward(&self, z: Tensor<3>, g: Tensor<3>, k: &Consts) -> Tensor<3> {
        let mut x = self.pre.forward(z) + self.cond.forward(g);
        for (i, up) in self.ups.iter().enumerate() {
            x = up.forward(activation::leaky_relu(x, 0.1));
            let mut acc: Option<Tensor<3>> = None;
            for j in 0..self.kernels {
                let r = self.res[i * self.kernels + j].forward(x.clone());
                acc = Some(match acc {
                    Some(a) => a + r,
                    None => r,
                });
            }
            x = acc.unwrap() / self.kernels as f64;
        }
        let x = activation::leaky_relu(x, 0.01).pad([(1, 0)], PadMode::Reflect);
        let x = self.post.forward(x);
        let [b, _, t] = x.dims();
        let (sb, n) = (k.subbands, k.istft_n_fft);
        let bins = n / 2 + 1;
        let x = x.reshape([b * sb, n + 2, t]);
        let mag = x.clone().narrow(1, 0, bins).clamp_max(10.0).exp();
        let phase = x.narrow(1, bins, bins).sin() * std::f64::consts::PI;
        let re = mag.clone() * phase.clone().cos();
        let im = mag * phase.sin();
        let spec = Tensor::cat(vec![re, im], 1);
        // Overlap-add (iSTFT) as a transposed convolution, centred.
        let y = conv_transpose1d(
            spec,
            k.istft_basis.clone(),
            None,
            ConvTransposeOptions::new([k.istft_hop], [0], [0], [1], 1),
        );
        let len = (t - 1) * k.istft_hop;
        let y = y.narrow(2, n / 2, len);
        let norm =
            Tensor::<1>::from_floats(dsp::istft_norm(n, k.istft_hop, t).as_slice(), &y.device())
                .reshape([1, 1, len]);
        let y = (y * norm).reshape([b, sb, len]);
        // PQMF synthesis.
        let y = conv_transpose1d(
            y,
            k.updown.clone(),
            None,
            ConvTransposeOptions::new([sb], [0], [0], [1], 1),
        );
        let half = k.taps / 2;
        conv1d(
            y.pad([(half, half)], PadMode::Constant(0.0)),
            k.pqmf_syn.clone(),
            None,
            ConvOptions::new([1], [0], [1], 1),
        )
    }
}

/// The complete synthesiser.
#[derive(Module, Debug)]
pub struct Synthesizer {
    pub enc_p: TextEncoder,
    pub enc_q: PosteriorEncoder,
    pub flow: Flow,
    pub dp: DurationPredictor,
    pub dec: Decoder,
    pub spk: Embed,
    pub dropout: f64,
}

/// Training batch (padded).
pub struct Batch {
    pub ids: Tensor<2, Int>,
    pub id_mask: Tensor<3>,
    pub id_lens: Vec<usize>,
    pub spec_mask: Tensor<3>,
    pub spec_lens: Vec<usize>,
    pub wav: Tensor<3>,
    pub speaker: Tensor<1, Int>,
    pub lang: Tensor<1, Int>,
}

/// Outputs needed by the losses.
pub struct TrainOut {
    pub y_hat: Tensor<3>,
    pub slice_starts: Vec<usize>,
    pub l_dur: Tensor<1>,
    pub l_kl: Tensor<1>,
}

impl Synthesizer {
    pub fn new(
        dev: &Device,
        c: &Config,
        vocab: usize,
        speakers: usize,
        langs: usize,
    ) -> Synthesizer {
        let m = &c.model;
        Synthesizer {
            enc_p: TextEncoder::new(dev, m, vocab, langs),
            enc_q: PosteriorEncoder::new(dev, m, c.audio.n_fft / 2 + 1),
            flow: Flow {
                layers: (0..m.flows).map(|_| Coupling::new(dev, m)).collect(),
            },
            dp: DurationPredictor::new(dev, m),
            dec: Decoder::new(dev, m),
            spk: Embed::new(dev, speakers.max(1), m.gin, 0.1),
            dropout: m.dropout as f64,
        }
    }

    pub fn forward_train(&self, b: &Batch, segment: usize, k: &Consts, seed: u64) -> TrainOut {
        let g = self.spk.global(b.speaker.clone());
        let (x, m_p, logs_p) = self.enc_p.forward(
            b.ids.clone(),
            b.lang.clone(),
            g.clone(),
            b.id_mask.clone(),
            self.dropout,
            true,
        );
        // Linear spectrogram of the whole utterances, on the device.
        let spec = k.linear(b.wav.clone()).detach();
        let (z, _m_q, logs_q) = self.enc_q.forward(spec, b.spec_mask.clone(), g.clone());
        let z_p = self
            .flow
            .forward(z.clone(), b.spec_mask.clone(), g.clone(), false);

        // Monotonic alignment search on the detached log-likelihoods.
        let attn = {
            let s_p_sq_r = (logs_p.clone() * -2.0).exp().detach(); // [B, d, Tx]
            let mpd = m_p.clone().detach();
            let zpd = z_p.clone().detach();
            let lpd = logs_p.clone().detach();
            let c1 = ((lpd * -1.0) - 0.5 * (2.0 * std::f64::consts::PI).ln()).sum_dim(1); // [B,1,Tx]
            let c2 = (zpd.clone().powi_scalar(2) * -0.5)
                .swap_dims(1, 2)
                .matmul(s_p_sq_r.clone()); // [B,Ty,Tx]
            let c3 = zpd.swap_dims(1, 2).matmul(mpd.clone() * s_p_sq_r.clone());
            let c4 = (mpd.powi_scalar(2) * s_p_sq_r * -0.5).sum_dim(1);
            let neg = c2 + c3 + c1 + c4; // [B, Ty, Tx]
            mas::align(neg, &b.id_lens, &b.spec_lens)
        };
        // attn: [B, Ty, Tx]
        let w = attn.clone().sum_dim(1); // frames per token [B, 1, Tx]
        let logw_true = (w + 1e-6).log() * b.id_mask.clone();
        let logw = self.dp.forward(
            x.detach(),
            b.id_mask.clone(),
            g.clone().detach(),
            self.dropout,
            true,
        );
        let l_dur = (logw - logw_true).powi_scalar(2).sum() / b.id_mask.clone().sum();

        // Expand the prior to frame rate.
        let m_p = attn.clone().matmul(m_p.swap_dims(1, 2)).swap_dims(1, 2); // [B, d, Ty]
        let logs_p = attn.matmul(logs_p.swap_dims(1, 2)).swap_dims(1, 2);
        let kl = logs_p.clone() - logs_q - 0.5
            + (z_p - m_p).powi_scalar(2) * (logs_p * -2.0).exp() * 0.5;
        let l_kl = (kl * b.spec_mask.clone()).sum() / b.spec_mask.clone().sum();

        // Random decoder windows.
        let mut rng = crate::data::Rng::new(seed);
        let [bs, _, _] = z.dims();
        let mut parts = Vec::with_capacity(bs);
        let mut starts = Vec::with_capacity(bs);
        for i in 0..bs {
            let len = b.spec_lens[i];
            let s = if len > segment {
                rng.below(len - segment + 1)
            } else {
                0
            };
            starts.push(s);
            parts.push(z.clone().narrow(0, i, 1).narrow(2, s, segment));
        }
        let zs = Tensor::cat(parts, 0);
        let y_hat = self.dec.forward(zs, g, k);
        TrainOut {
            y_hat,
            slice_starts: starts,
            l_dur,
            l_kl,
        }
    }

    /// Inference: phoneme ids `[1, T]` → waveform samples.
    pub fn infer(
        &self,
        ids: Tensor<2, Int>,
        speaker: usize,
        lang: usize,
        k: &Consts,
        noise: f64,
        length_scale: f64,
    ) -> Vec<f32> {
        let dev = ids.device();
        let t = ids.dims()[1];
        let mask = Tensor::<3>::ones([1, 1, t], &dev);
        let g = self.spk.global(Tensor::<1, Int>::from_ints(
            [speaker as i64].as_slice(),
            &dev,
        ));
        let lang = Tensor::<1, Int>::from_ints([lang as i64].as_slice(), &dev);
        let (x, m_p, logs_p) = self
            .enc_p
            .forward(ids, lang, g.clone(), mask.clone(), 0.0, false);
        let logw = self.dp.forward(x, mask, g.clone(), 0.0, false);
        let w: Vec<f32> = logw
            .exp()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap_or_default();
        let dur: Vec<usize> = w
            .iter()
            .map(|v| ((*v as f64) * length_scale).ceil().max(0.0) as usize)
            .collect();
        let ty: usize = dur.iter().sum::<usize>().max(1);
        let mut path = vec![0f32; ty * t];
        let mut pos = 0;
        for (j, &d) in dur.iter().enumerate() {
            for f in pos..(pos + d).min(ty) {
                path[f * t + j] = 1.0;
            }
            pos += d;
        }
        let attn = Tensor::<3>::from_data(TensorData::new(path, [1, ty, t]), &dev);
        let m_p = attn.clone().matmul(m_p.swap_dims(1, 2)).swap_dims(1, 2);
        let logs_p = attn.matmul(logs_p.swap_dims(1, 2)).swap_dims(1, 2);
        let eps = Tensor::<3>::random(m_p.dims(), Distribution::Normal(0.0, 1.0), &dev);
        let z_p = m_p + eps * logs_p.exp() * noise;
        let ymask = Tensor::<3>::ones([1, 1, ty], &dev);
        let z = self.flow.forward(z_p, ymask, g.clone(), true);
        let y = self.dec.forward(z, g, k);
        y.into_data().try_to_vec::<f32>().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_spectrogram_matches_reference() {
        let cfg = Config::tiny();
        let dev = Device::flex();
        let k = Consts::new(&cfg, &dev);
        let hop = cfg.audio.hop;
        let n = hop * 20;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / 24_000.0;
                0.5 * (2.0 * std::f32::consts::PI * (220.0 + 900.0 * t) * t).sin()
                    + 0.1 * ((i * 7919 % 1000) as f32 / 500.0 - 1.0)
            })
            .collect();
        let want = dsp::spectrogram(&x, cfg.audio.n_fft, hop);
        let got = k
            .linear(Tensor::<1>::from_floats(x.as_slice(), &dev).reshape([1, 1, n]))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(got.len(), want.len());
        let peak = want.iter().fold(0f32, |m, v| m.max(*v));
        let err = got
            .iter()
            .zip(&want)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-4 * peak, "max error {err} (peak {peak})");
    }
}
