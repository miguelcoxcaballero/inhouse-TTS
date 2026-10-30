//! Rational sample-rate conversion (Kaiser-windowed sinc, polyphase), used
//! to bring every model to the engine's output rate (e.g. 22.05 → 24 kHz).

use std::f64::consts::PI;

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Modified Bessel function I0 (Kaiser window).
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    let y = x * x / 4.0;
    for k in 1..60 {
        term *= y / (k * k) as f64;
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

/// A precomputed `from → to` resampler. The pass band ends at 96 % of the
/// lower Nyquist frequency; stop-band rejection is about 90 dB.
pub struct Resampler {
    up: usize,
    down: usize,
    /// Taps per phase.
    taps: usize,
    /// `phases[p * taps + j]`: weight of input `k0(p) - j` for phase `p`.
    phases: Vec<f32>,
}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Resampler {
        let g = gcd(from, to);
        let (up, down) = ((to / g) as usize, (from / g) as usize);
        let m = up.max(down);
        let cutoff = 0.96 * 0.5 / m as f64;
        let half = 64 * m;
        let beta = 9.0;
        let i0b = bessel_i0(beta);
        let h = |n: isize| -> f64 {
            if n.unsigned_abs() > half {
                return 0.0;
            }
            let n = n as f64;
            let sinc = if n == 0.0 {
                2.0 * cutoff
            } else {
                (2.0 * PI * cutoff * n).sin() / (PI * n)
            };
            let r = n / half as f64;
            sinc * bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b * up as f64
        };
        // Output o sits at upsampled position t = o·down; input k at k·up.
        // With t = q·up + p, input k = q - j has offset p + j·up.
        let taps = (2 * half) / up + 2;
        let mut phases = vec![0f32; up * taps];
        for p in 0..up {
            for j in 0..taps {
                // Inputs k = q - j + taps/2 so the window is centred.
                let off = p as isize + (j as isize - (taps / 2) as isize) * up as isize;
                phases[p * taps + j] = h(off) as f32;
            }
        }
        Resampler {
            up,
            down,
            taps,
            phases,
        }
    }

    /// Converts a whole signal (zero padding at both ends).
    pub fn process(&self, x: &[f32]) -> Vec<f32> {
        if self.up == self.down || x.is_empty() {
            return x.to_vec();
        }
        let out_len = (x.len() * self.up).div_ceil(self.down);
        let half_taps = (self.taps / 2) as isize;
        let mut y = vec![0f32; out_len];
        for (o, yo) in y.iter_mut().enumerate() {
            let t = o * self.down;
            let (q, p) = ((t / self.up) as isize, t % self.up);
            let w = &self.phases[p * self.taps..(p + 1) * self.taps];
            let mut acc = 0f32;
            for (j, &wj) in w.iter().enumerate() {
                let k = q - j as isize + half_taps;
                if k >= 0 && (k as usize) < x.len() {
                    acc += wj * x[k as usize];
                }
            }
            *yo = acc;
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f64, rate: f64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * hz * i as f64 / rate).sin() as f32)
            .collect()
    }

    #[test]
    fn keeps_pass_band_and_rejects_aliases() {
        let r = Resampler::new(22_050, 24_000);
        for hz in [440.0, 5_000.0, 10_000.0] {
            let y = r.process(&tone(hz, 22_050.0, 22_050));
            let want = tone(hz, 24_000.0, y.len());
            // Compare away from the edges.
            let (a, b) = (2_000, y.len() - 2_000);
            let err = (a..b).map(|i| (y[i] - want[i]).abs()).fold(0f32, f32::max);
            assert!(err < 2e-3, "{hz} Hz: max error {err}");
        }
        // 24 → 22.05 kHz: an 11.5 kHz tone is above the new Nyquist.
        let d = Resampler::new(24_000, 22_050);
        let y = d.process(&tone(11_500.0, 24_000.0, 24_000));
        let rms = (y[2_000..y.len() - 2_000].iter().map(|v| v * v).sum::<f32>()
            / (y.len() - 4_000) as f32)
            .sqrt();
        assert!(rms < 1e-3, "alias rms {rms}");
    }
}
