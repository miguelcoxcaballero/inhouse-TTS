//! Signal processing: small-FFT STFT/iSTFT, harmonic-plus-noise excitation,
//! a deterministic random generator and WAV encoding.

use crate::tensor::Mat;
use std::f64::consts::PI;

/// Short-time Fourier transform with a periodic Hann window, `center=True`
/// and reflect padding (PyTorch `torch.stft` / `torch.istft` semantics).
pub struct Stft {
    pub n_fft: usize,
    pub hop: usize,
    window: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        let n = n_fft;
        let bins = n / 2 + 1;
        let window = (0..n)
            .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()) as f32)
            .collect();
        let mut cos = vec![0f32; bins * n];
        let mut sin = vec![0f32; bins * n];
        for k in 0..bins {
            for i in 0..n {
                let a = 2.0 * PI * ((k * i) % n) as f64 / n as f64;
                cos[k * n + i] = a.cos() as f32;
                sin[k * n + i] = a.sin() as f32;
            }
        }
        Stft {
            n_fft,
            hop,
            window,
            cos,
            sin,
        }
    }

    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Magnitude and phase, each `[bins, frames]` with `frames = len/hop + 1`.
    pub fn forward(&self, x: &[f32]) -> (Mat, Mat) {
        let n = self.n_fft;
        let pad = n / 2;
        let len = x.len();
        assert!(len > pad, "signal too short for reflect padding");
        let reflect = |i: isize| -> f32 {
            let mut j = i - pad as isize;
            if j < 0 {
                j = -j;
            }
            let last = len as isize - 1;
            if j > last {
                j = 2 * last - j;
            }
            x[j as usize]
        };
        let frames = len / self.hop + 1;
        let bins = self.bins();
        let mut mag = Mat::zeros(bins, frames);
        let mut phase = Mat::zeros(bins, frames);
        let mut buf = vec![0f32; n];
        for f in 0..frames {
            let start = (f * self.hop) as isize;
            for (i, b) in buf.iter_mut().enumerate() {
                *b = reflect(start + i as isize) * self.window[i];
            }
            for k in 0..bins {
                let c = &self.cos[k * n..(k + 1) * n];
                let s = &self.sin[k * n..(k + 1) * n];
                let mut re = 0f32;
                let mut im = 0f32;
                for i in 0..n {
                    re += buf[i] * c[i];
                    im -= buf[i] * s[i];
                }
                if k == 0 || 2 * k == n {
                    im = 0.0;
                }
                mag.data[k * frames + f] = (re * re + im * im).sqrt();
                phase.data[k * frames + f] = im.atan2(re);
            }
        }
        (mag, phase)
    }

    /// Inverse transform of `mag · e^{i·phase}`; output length `hop·(frames-1)`.
    pub fn inverse(&self, mag: &Mat, phase: &Mat) -> Vec<f32> {
        let n = self.n_fft;
        let bins = self.bins();
        assert_eq!(mag.rows, bins);
        let frames = mag.cols;
        let total = n + self.hop * (frames - 1);
        let mut ola = vec![0f32; total];
        let mut env = vec![0f32; total];
        let mut re = vec![0f32; bins];
        let mut im = vec![0f32; bins];
        let inv_n = 1.0 / n as f32;
        for f in 0..frames {
            for k in 0..bins {
                let m = mag.data[k * frames + f];
                let p = phase.data[k * frames + f];
                re[k] = m * p.cos();
                im[k] = m * p.sin();
            }
            let off = f * self.hop;
            for i in 0..n {
                let mut acc = re[0];
                for k in 1..bins {
                    let w = if 2 * k == n { 1.0 } else { 2.0 };
                    let c = self.cos[k * n + i];
                    let s = self.sin[k * n + i];
                    acc += w * (re[k] * c - if 2 * k == n { 0.0 } else { im[k] * s });
                }
                let w = self.window[i];
                ola[off + i] += acc * inv_n * w;
                env[off + i] += w * w;
            }
        }
        let pad = n / 2;
        let len = self.hop * (frames - 1);
        (0..len)
            .map(|t| {
                let e = env[pad + t];
                if e > 1e-11 {
                    ola[pad + t] / e
                } else {
                    ola[pad + t]
                }
            })
            .collect()
    }
}

/// Deterministic PCG32 generator with Gaussian sampling (Box–Muller).
pub struct Rng {
    state: u64,
    inc: u64,
    spare: Option<f32>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut r = Rng {
            state: 0,
            inc: (seed << 1) | 1,
            spare: None,
        };
        r.next_u32();
        r.state = r.state.wrapping_add(seed ^ 0x853c_49e6_748f_ea9b);
        r.next_u32();
        r
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    pub fn gaussian(&mut self) -> f32 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let u1 = (self.uniform() as f64).max(1e-12);
        let u2 = self.uniform() as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (2.0 * PI * u2).sin_cos();
        self.spare = Some((r * s) as f32);
        (r * c) as f32
    }
}

/// Harmonic-plus-noise excitation of the iSTFTNet source module (hn-NSF).
///
/// `f0` is the frame-level pitch (Hz) and every frame spans `upsample`
/// samples. Returns `tanh(Σ_h w_h · e_h(t) + b)` where `e_h` is the h-th
/// harmonic (amplitude 0.1) in voiced samples plus Gaussian noise
/// (std 0.003 voiced, 0.1/3 unvoiced).
pub struct SourceParams<'a> {
    pub sample_rate: f32,
    pub upsample: usize,
    pub weights: &'a [f32],
    pub bias: f32,
    pub voiced_threshold: f32,
}

pub fn harmonic_source(f0: &[f32], p: &SourceParams, mut rng: Option<&mut Rng>) -> Vec<f32> {
    let frames = f0.len();
    let up = p.upsample;
    let harmonics = p.weights.len();
    let len = frames * up;
    // Cumulative phase in cycles at frame resolution (scaled by `up`), as in
    // the reference: per-frame increment frac(f0*h/sr), cumulative sum, ×up.
    let mut cycles = vec![0f64; frames * harmonics];
    for h in 0..harmonics {
        let mut acc = 0f64;
        for i in 0..frames {
            let r = (f0[i] * (h + 1) as f32) / p.sample_rate;
            let r = r - r.floor();
            acc += r as f64;
            cycles[h * frames + i] = acc * up as f64;
        }
    }
    let scale = 1.0 / up as f64;
    let mut out = vec![0f32; len];
    let mut harm = vec![0f64; harmonics];
    for (j, o) in out.iter_mut().enumerate() {
        // Linear interpolation (align_corners=False) of the frame phase.
        let mut src = scale * (j as f64 + 0.5) - 0.5;
        if src < 0.0 {
            src = 0.0;
        }
        let i0 = (src as usize).min(frames - 1);
        let i1 = (i0 + 1).min(frames - 1);
        let l1 = src - i0 as f64;
        let l0 = 1.0 - l1;
        let f = f0[j / up];
        let voiced = f > p.voiced_threshold;
        for (h, hv) in harm.iter_mut().enumerate() {
            let v = l0 * cycles[h * frames + i0] + l1 * cycles[h * frames + i1];
            *hv = (2.0 * PI * (v - v.floor())).sin();
        }
        let noise_amp = if voiced { 0.003 } else { 0.1 / 3.0 };
        let mut acc = p.bias as f64;
        for (h, &hv) in harm.iter().enumerate() {
            let mut e = if voiced { hv * 0.1 } else { 0.0 };
            if let Some(r) = rng.as_deref_mut() {
                e += noise_amp * r.gaussian() as f64;
            }
            acc += p.weights[h] as f64 * e;
        }
        *o = acc.tanh() as f32;
    }
    out
}

/// 16-bit PCM mono WAV.
pub fn wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_roundtrip() {
        let stft = Stft::new(20, 5);
        let x: Vec<f32> = (0..300)
            .map(|i| ((i as f32) * 0.07).sin() * 0.5 + ((i as f32) * 0.31).cos() * 0.2)
            .collect();
        let (m, p) = stft.forward(&x);
        let y = stft.inverse(&m, &p);
        assert_eq!(y.len(), 300);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    #[ignore]
    fn gaussian_moments_precise() {
        let mut r = Rng::new(7);
        let n = 20_000_000;
        let (mut s1, mut s2) = (0f64, 0f64);
        for _ in 0..n {
            let g = r.gaussian() as f64;
            s1 += g;
            s2 += g * g;
        }
        let mean = s1 / n as f64;
        println!("mean {mean} var {}", s2 / n as f64 - mean * mean);
    }

    #[test]
    fn gaussian_moments() {
        let mut r = Rng::new(42);
        let n = 100_000;
        let v: Vec<f32> = (0..n).map(|_| r.gaussian()).collect();
        let mean = v.iter().sum::<f32>() / n as f32;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.02 && (var - 1.0).abs() < 0.03);
    }
}
