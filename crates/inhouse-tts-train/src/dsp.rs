//! Fixed signal-processing filters (mel bank, STFT/iSTFT bases, PQMF) and
//! the CPU spectrogram used to cache posterior-encoder inputs.

use std::f64::consts::PI;

/// Periodic Hann window (as `torch.hann_window`).
pub fn hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos())
        .collect()
}

fn hz_to_mel(f: f64) -> f64 {
    // Slaney scale (librosa default).
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// Slaney-normalised mel filter bank `[n_mels, n_fft/2 + 1]` (librosa).
pub fn mel_bank(sr: u32, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let fft_f: Vec<f64> = (0..bins)
        .map(|k| k as f64 * sr as f64 / n_fft as f64)
        .collect();
    let (lo, hi) = (hz_to_mel(fmin), hz_to_mel(fmax));
    let pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut w = vec![0f32; n_mels * bins];
    for m in 0..n_mels {
        let (f0, f1, f2) = (pts[m], pts[m + 1], pts[m + 2]);
        let norm = 2.0 / (f2 - f0);
        for (k, &f) in fft_f.iter().enumerate() {
            let lower = (f - f0) / (f1 - f0);
            let upper = (f2 - f) / (f2 - f1);
            w[m * bins + k] = (lower.min(upper).max(0.0) * norm) as f32;
        }
    }
    w
}

/// Analysis basis for STFT-as-convolution: `[2·(n/2+1), 1, n]`, rows are
/// windowed cosines then windowed negative sines.
pub fn stft_basis(n: usize) -> Vec<f32> {
    let w = hann(n);
    let bins = n / 2 + 1;
    let mut out = vec![0f32; 2 * bins * n];
    for k in 0..bins {
        for t in 0..n {
            let a = 2.0 * PI * (k * t) as f64 / n as f64;
            out[k * n + t] = (a.cos() * w[t]) as f32;
            out[(bins + k) * n + t] = (-a.sin() * w[t]) as f32;
        }
    }
    out
}

/// Synthesis basis for iSTFT-as-transposed-convolution `[2·(n/2+1), 1, n]`
/// (real rows, then imaginary rows), including the synthesis window.
pub fn istft_basis(n: usize) -> Vec<f32> {
    let w = hann(n);
    let bins = n / 2 + 1;
    let mut out = vec![0f32; 2 * bins * n];
    for k in 0..bins {
        let c = if k == 0 || 2 * k == n { 1.0 } else { 2.0 };
        for t in 0..n {
            let a = 2.0 * PI * (k * t) as f64 / n as f64;
            out[k * n + t] = (c * a.cos() * w[t] / n as f64) as f32;
            out[(bins + k) * n + t] = (-c * a.sin() * w[t] / n as f64) as f32;
        }
    }
    out
}

/// Inverse of the overlap-added squared window for an iSTFT of `frames`
/// frames (centered, cropped by n/2 on both sides): length `(frames-1)·hop`.
pub fn istft_norm(n: usize, hop: usize, frames: usize) -> Vec<f32> {
    let w = hann(n);
    let full = (frames - 1) * hop + n;
    let mut env = vec![0f64; full];
    for f in 0..frames {
        for t in 0..n {
            env[f * hop + t] += w[t] * w[t];
        }
    }
    env[n / 2..full - n / 2]
        .iter()
        .map(|e| if *e > 1e-11 { (1.0 / e) as f32 } else { 0.0 })
        .collect()
}

fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..60 {
        term *= (x / 2.0) * (x / 2.0) / (k * k) as f64;
        sum += term;
    }
    sum
}

/// Pseudo-QMF bank (Parallel WaveGAN design): `(analysis [sb, 1, taps+1],
/// synthesis [1, sb, taps+1])`.
pub fn pqmf(subbands: usize, taps: usize, cutoff: f64, beta: f64) -> (Vec<f32>, Vec<f32>) {
    let n = taps + 1;
    let wc = PI * cutoff;
    let proto: Vec<f64> = (0..n)
        .map(|i| {
            let m = i as f64 - 0.5 * taps as f64;
            let h = if m == 0.0 {
                cutoff
            } else {
                (wc * m).sin() / (PI * m)
            };
            let r = 2.0 * i as f64 / taps as f64 - 1.0;
            h * bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / bessel_i0(beta)
        })
        .collect();
    let mut ana = vec![0f32; subbands * n];
    let mut syn = vec![0f32; subbands * n];
    for k in 0..subbands {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        for i in 0..n {
            let m = i as f64 - taps as f64 / 2.0;
            let ph = (2 * k + 1) as f64 * (PI / (2 * subbands) as f64) * m;
            ana[k * n + i] = (2.0 * proto[i] * (ph + sign * PI / 4.0).cos()) as f32;
            syn[k * n + i] = (2.0 * proto[i] * (ph - sign * PI / 4.0).cos()) as f32;
        }
    }
    (ana, syn)
}

/// Linear magnitude spectrogram `[n/2+1, len/hop]` of a waveform whose
/// length is a multiple of `hop` (reflect padding of (n-hop)/2 per side).
/// Reference for the on-device [`crate::model::Consts::linear`].
#[cfg(test)]
pub fn spectrogram(x: &[f32], n: usize, hop: usize) -> Vec<f32> {
    let pad = (n - hop) / 2;
    let len = x.len();
    let get = |i: isize| -> f32 {
        let mut j = i;
        if j < 0 {
            j = -j;
        }
        if j as usize >= len {
            j = 2 * (len as isize - 1) - j;
        }
        x[j.clamp(0, len as isize - 1) as usize]
    };
    let frames = len / hop;
    let bins = n / 2 + 1;
    let w = hann(n);
    let mut out = vec![0f32; bins * frames];
    let mut re = vec![0f64; n];
    let mut im = vec![0f64; n];
    for f in 0..frames {
        let start = (f * hop) as isize - pad as isize;
        for t in 0..n {
            re[t] = get(start + t as isize) as f64 * w[t];
            im[t] = 0.0;
        }
        fft(&mut re, &mut im);
        for k in 0..bins {
            out[k * frames + f] = ((re[k] * re[k] + im[k] * im[k]) + 1e-9).sqrt() as f32;
        }
    }
    out
}

/// In-place radix-2 FFT.
#[cfg(test)]
pub fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let (ws, wc) = ang.sin_cos();
        for i in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let (ar, ai) = (re[i + k + len / 2], im[i + k + len / 2]);
                let (tr, ti) = (ar * cr - ai * ci, ar * ci + ai * cr);
                re[i + k + len / 2] = re[i + k] - tr;
                im[i + k + len / 2] = im[i + k] - ti;
                re[i + k] += tr;
                im[i + k] += ti;
                let nr = cr * wc - ci * ws;
                ci = cr * ws + ci * wc;
                cr = nr;
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pqmf_is_near_perfect_reconstruction() {
        // Analysis then synthesis of a chirp reproduces it (delayed by taps).
        let (sb, taps) = (4, 62);
        let (ana, syn) = pqmf(sb, taps, 0.142, 9.0);
        let n = 4096;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * (0.002 * i as f64 + 1e-5 * (i * i) as f64)).sin())
            .collect();
        let half = taps / 2;
        // Analysis: filter (zero padding taps/2) and decimate.
        let mut bands = vec![vec![0f64; n / sb]; sb];
        for k in 0..sb {
            for q in 0..n / sb {
                let t = q * sb;
                let mut acc = 0.0;
                for i in 0..=taps {
                    let idx = t as isize + i as isize - half as isize;
                    if idx >= 0 && (idx as usize) < n {
                        acc += ana[k * (taps + 1) + i] as f64 * x[idx as usize];
                    }
                }
                bands[k][q] = acc;
            }
        }
        // Synthesis: upsample (×sb gain), filter.
        let mut y = vec![0f64; n];
        for (t, yt) in y.iter_mut().enumerate() {
            let mut acc = 0.0;
            for k in 0..sb {
                for i in 0..=taps {
                    let idx = t as isize + i as isize - half as isize;
                    if idx >= 0 && (idx as usize) < n && idx as usize % sb == 0 {
                        acc += syn[k * (taps + 1) + i] as f64
                            * bands[k][idx as usize / sb]
                            * sb as f64;
                    }
                }
            }
            *yt = acc;
        }
        let (mut e, mut p) = (0.0, 0.0);
        for t in 200..n - 200 {
            e += (y[t] - x[t]).powi(2);
            p += x[t].powi(2);
        }
        let snr = 10.0 * (p / e).log10();
        assert!(snr > 30.0, "PQMF reconstruction SNR {snr:.1} dB");
    }

    #[test]
    fn istft_inverts_stft() {
        let (n, hop) = (16, 4);
        let frames = 64;
        let len = (frames - 1) * hop;
        let x: Vec<f64> = (0..len)
            .map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0)
            .collect();
        let a = stft_basis(n);
        let s = istft_basis(n);
        let bins = n / 2 + 1;
        // Centered STFT with zero padding n/2.
        let mut spec = vec![0f64; 2 * bins * frames];
        for f in 0..frames {
            for r in 0..2 * bins {
                let mut acc = 0.0;
                for t in 0..n {
                    let idx = (f * hop + t) as isize - (n / 2) as isize;
                    if idx >= 0 && (idx as usize) < len {
                        acc += a[r * n + t] as f64 * x[idx as usize];
                    }
                }
                spec[r * frames + f] = acc;
            }
        }
        // Overlap-add with the synthesis basis, crop and normalise.
        let full = (frames - 1) * hop + n;
        let mut y = vec![0f64; full];
        for f in 0..frames {
            for r in 0..2 * bins {
                for t in 0..n {
                    y[f * hop + t] += s[r * n + t] as f64 * spec[r * frames + f];
                }
            }
        }
        let norm = istft_norm(n, hop, frames);
        let y: Vec<f64> = y[n / 2..full - n / 2]
            .iter()
            .zip(&norm)
            .map(|(v, g)| v * *g as f64)
            .collect();
        let err = y
            .iter()
            .zip(&x)
            .skip(n)
            .take(len - 2 * n)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-4, "iSTFT error {err}");
    }

    #[test]
    fn mel_bank_shape() {
        let m = mel_bank(24_000, 1024, 80, 0.0, 12_000.0);
        assert_eq!(m.len(), 80 * 513);
        assert!(m.iter().all(|v| *v >= 0.0));
    }
}
