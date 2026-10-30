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

/// Error function in double precision (fdlibm `s_erf.c`, below 1 ulp).
///
/// Used where a network evaluates the exact GELU `x/2 · (1 + erf(x/√2))`.
pub fn erf(x: f64) -> f64 {
    const ERX: f64 = 0.845_062_911_510_467_5;
    const EFX: f64 = 0.128_379_167_095_512_6;
    const PP: [f64; 5] = [
        0.128_379_167_095_512_56,
        -0.325_042_107_247_001_5,
        -0.028_481_749_575_598_51,
        -0.005_770_270_296_489_442,
        -2.376_301_665_665_016_3e-5,
    ];
    const QQ: [f64; 5] = [
        0.397_917_223_959_155_35,
        0.065_022_249_988_767_3,
        0.005_081_306_281_875_766,
        0.000_132_494_738_004_321_64,
        -3.960_228_278_775_368e-6,
    ];
    const PA: [f64; 7] = [
        -0.002_362_118_560_752_659_4,
        0.414_856_118_683_748_33,
        -0.372_207_876_035_701_3,
        0.318_346_619_901_161_75,
        -0.110_894_694_282_396_68,
        0.035_478_304_325_618_236,
        -0.002_166_375_594_868_791,
    ];
    const QA: [f64; 6] = [
        0.106_420_880_400_844_23,
        0.540_397_917_702_171,
        0.071_828_654_414_196_27,
        0.126_171_219_808_761_64,
        0.013_637_083_912_029_05,
        0.011_984_499_846_799_107,
    ];
    const RA: [f64; 8] = [
        -0.009_864_944_034_847_148,
        -0.693_858_572_707_181_8,
        -10.558_626_225_323_291,
        -62.375_332_450_326_006,
        -162.396_669_462_573_47,
        -184.605_092_906_711_04,
        -81.287_435_506_306_6,
        -9.814_329_344_169_145,
    ];
    const SA: [f64; 8] = [
        19.651_271_667_439_257,
        137.657_754_143_519_04,
        434.565_877_475_229_23,
        645.387_271_733_267_9,
        429.008_140_027_567_83,
        108.635_005_541_779_44,
        6.570_249_770_319_282,
        -0.060_424_415_214_858_1,
    ];
    const RB: [f64; 7] = [
        -0.009_864_942_924_700_1,
        -0.799_283_237_680_523,
        -17.757_954_917_754_752,
        -160.636_384_855_821_92,
        -637.566_443_368_389_6,
        -1_025.095_131_611_077_2,
        -483.519_191_608_651_4,
    ];
    const SB: [f64; 7] = [
        30.338_060_743_482_46,
        325.792_512_996_573_9,
        1_536.729_586_084_437,
        3_199.858_219_508_595_5,
        2_553.050_406_433_164_4,
        474.528_541_206_955_37,
        -22.440_952_446_585_82,
    ];
    // c[0] + c[1] z + c[2] z² + ...
    fn poly(c: &[f64], z: f64) -> f64 {
        c.iter().rev().fold(0.0, |acc, &k| acc * z + k)
    }
    if x.is_nan() {
        return x;
    }
    let a = x.abs();
    if a < 0.843_75 {
        if a < 3.725_290_298_461_914e-9 {
            return x + EFX * x;
        }
        let z = x * x;
        let r = poly(&PP, z);
        let s = 1.0 + z * poly(&QQ, z);
        return x + x * (r / s);
    }
    if a < 1.25 {
        let s = a - 1.0;
        let p = poly(&PA, s);
        let q = 1.0 + s * poly(&QA, s);
        return (ERX + p / q).copysign(x);
    }
    if a >= 6.0 {
        return 1f64.copysign(x);
    }
    let s = 1.0 / (a * a);
    let (r, q) = if a < 1.0 / 0.35 {
        (poly(&RA, s), 1.0 + s * poly(&SA, s))
    } else {
        (poly(&RB, s), 1.0 + s * poly(&SB, s))
    };
    // z = a with the low 32 bits cleared, so that z·z is exact.
    let z = f64::from_bits(a.to_bits() & 0xffff_ffff_0000_0000);
    let e = (-z * z - 0.5625).exp() * ((z - a) * (z + a) + r / q).exp();
    (1.0 - e / a).copysign(x)
}

/// Exact GELU `x/2 · (1 + erf(x/√2))` (PyTorch `F.gelu` default).
#[inline]
pub fn gelu_erf(x: f32) -> f32 {
    let x = x as f64;
    (0.5 * x * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))) as f32
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

    #[test]
    fn erf_matches_libm() {
        // Reference values from CPython's math.erf.
        for &(x, want) in &[
            (0.0, 0.0),
            (1e-9, 1.128_379_167_095_512_7e-9),
            (0.1, 0.112_462_916_018_284_9),
            (0.5, 0.520_499_877_813_046_5),
            (0.84, 0.765_142_711_454_994_6),
            (0.9, 0.796_908_212_422_832_2),
            (1.2, 0.910_313_978_229_635_3),
            (1.3, 0.934_007_944_940_652_4),
            (2.0, 0.995_322_265_018_952_7),
            (2.8, 0.999_924_986_805_334_6),
            (3.0, 0.999_977_909_503_001_4),
            (4.5, 0.999_999_999_803_383_9),
            (5.9, 0.999_999_999_999_999_9),
            (-0.7, -0.677_801_193_837_418_4),
            (-1.1, -0.880_205_069_574_081_7),
            (-2.5, -0.999_593_047_982_555),
            (-3.5, -0.999_999_256_901_627_6),
        ] {
            let got = super::erf(x);
            assert!(
                (got - want).abs() <= 2.0 * f64::EPSILON * want.abs(),
                "erf({x}) = {got}, want {want}"
            );
        }
    }
}
