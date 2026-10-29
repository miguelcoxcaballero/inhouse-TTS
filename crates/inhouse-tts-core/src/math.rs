//! Branch-free, vectorisable approximations of transcendental functions.
//!
//! Accuracy is ~1e-6 relative over the ranges used by the network; all
//! functions are written with plain arithmetic so that LLVM vectorises the
//! slice loops on SSE/AVX, NEON and wasm SIMD128.

const MAGIC_ROUND: f32 = 12_582_912.0; // 1.5 * 2^23

#[inline(always)]
fn round_magic(x: f32) -> f32 {
    (x + MAGIC_ROUND) - MAGIC_ROUND
}

/// sin(x) for |x| < ~1e5.
#[inline(always)]
pub fn sin(x: f32) -> f32 {
    const INV_PI: f32 = std::f32::consts::FRAC_1_PI;
    const PI_A: f32 = 3.140_625;
    const PI_B: f32 = 0.000_967_502_6;
    const PI_C: f32 = 1.509_957_9e-7;
    let k = round_magic(x * INV_PI);
    let r = ((x - k * PI_A) - k * PI_B) - k * PI_C;
    let r2 = r * r;
    // Minimax polynomial for sin on [-pi/2, pi/2].
    let p = 2.601_903e-6f32;
    let p = p * r2 - 1.980_741_9e-4;
    let p = p * r2 + 8.333_066e-3;
    let p = p * r2 - 1.666_665_7e-1;
    let s = r + r * r2 * p;
    let odd = (k as i32 & 1) as f32;
    s * (1.0 - 2.0 * odd)
}

/// exp(x), saturating outside [-87, 88].
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_145_75;
    const LN2_LO: f32 = 1.428_606_8e-6;
    let x = x.clamp(-87.0, 88.0);
    let n = round_magic(x * LOG2E);
    let r = (x - n * LN2_HI) - n * LN2_LO;
    // exp(r) on [-ln2/2, ln2/2], degree-6 polynomial.
    let p = 1.0 / 720.0f32;
    let p = p * r + 1.0 / 120.0;
    let p = p * r + 1.0 / 24.0;
    let p = p * r + 1.0 / 6.0;
    let p = p * r + 0.5;
    let p = p * r + 1.0;
    let p = p * r + 1.0;
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    p * scale
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    // tanh(x) = sign(x) * (1 - e) / (1 + e) with e = exp(-2|x|).
    let a = x.abs();
    let e = exp(-2.0 * a);
    let t = (1.0 - e) / (1.0 + e);
    // Keep precision near zero with a short odd series.
    let small = x - x * x * x * (1.0 / 3.0) + 2.0 / 15.0 * x * x * x * x * x;
    if a < 0.02 {
        small
    } else {
        t.copysign(x)
    }
}

/// GELU with tanh approximation ("gelu_new").
#[inline(always)]
pub fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + tanh(C * (x + 0.044_715 * x * x * x)))
}

pub fn leaky_relu_inplace(x: &mut [f32], slope: f32) {
    for v in x.iter_mut() {
        *v = if *v >= 0.0 { *v } else { *v * slope };
    }
}

/// Snake activation `x + sin(a x)^2 / a` for one channel.
pub fn snake_inplace(x: &mut [f32], alpha: f32) {
    let inv = 1.0 / alpha;
    for v in x.iter_mut() {
        let s = sin(alpha * *v);
        *v += inv * s * s;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn approximations() {
        let mut x = -60.0f32;
        while x < 60.0 {
            assert!((super::sin(x) - x.sin()).abs() < 2e-6, "sin {x}");
            assert!(
                (super::tanh(x / 7.0) - (x / 7.0).tanh()).abs() < 2e-6,
                "tanh {x}"
            );
            let e = super::exp(x / 4.0);
            let r = (x / 4.0).exp();
            assert!((e - r).abs() <= r * 2e-6, "exp {x}");
            x += 0.0137;
        }
        for &v in &[0.0f32, 1e-4, -1e-3, 0.019, 0.021] {
            assert!((super::tanh(v) - v.tanh()).abs() < 1e-7);
        }
    }
}
