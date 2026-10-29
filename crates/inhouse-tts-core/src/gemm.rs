//! Packed single-precision matrix multiplication with SIMD micro-kernels.
//!
//! `C[m×n] (+)= A[m×k] · B[k×n]`, with `C` row-major (leading dimension
//! `ldc`). `A` is usually a weight matrix kept in a compact format (f32, f16
//! or int8 rows with per-row scales) that is unpacked into MR-row panels;
//! `B` is any strided f32 matrix (activations, im2col tiles, ...).
//!
//! Micro-kernels: AVX2+FMA 6×16 and SSE2 4×8 on x86_64, NEON 8×8 on aarch64,
//! SIMD128 4×8 on wasm32 and a portable 4×4 fallback.

use crate::par::{self, SendPtr};
use std::cell::RefCell;
use std::sync::OnceLock;

/// Depth of a packed block (rows of B / columns of A per pass).
pub const KC: usize = 256;
/// Width of a packed B block (multiple of every NR).
const NC: usize = 240;

/// Storage of the `A` operand.
#[derive(Clone, Copy)]
pub enum AData<'a> {
    /// Strided f32 matrix: element (i, j) at `data[i * rs + j * cs]`.
    F32 {
        data: &'a [f32],
        rs: usize,
        cs: usize,
    },
    /// Row-major IEEE half precision.
    F16 { data: &'a [u16], ld: usize },
    /// Row-major int8 with one scale per row.
    Q8 {
        data: &'a [i8],
        ld: usize,
        scales: &'a [f32],
    },
}

/// `A` operand of shape `m × k`.
#[derive(Clone, Copy)]
pub struct AMat<'a> {
    pub m: usize,
    pub k: usize,
    pub data: AData<'a>,
}

/// `B` operand of shape `k × n`; element (i, j) at `data[i * rs + j * cs]`.
#[derive(Clone, Copy)]
pub struct BMat<'a> {
    pub k: usize,
    pub n: usize,
    pub data: &'a [f32],
    pub rs: usize,
    pub cs: usize,
}

impl<'a> BMat<'a> {
    /// Row-major `k × n` matrix with leading dimension `ld`.
    pub fn row_major(data: &'a [f32], k: usize, n: usize, ld: usize) -> Self {
        BMat {
            k,
            n,
            data,
            rs: ld,
            cs: 1,
        }
    }
}

/// Virtual im2col matrix of a 1-D convolution input, read on the fly.
///
/// Row `ci * taps + kt`, column `j` is `x[ci][j * stride + kt * dil - pad]`
/// (zero outside `0..t_in`). Input channel `ci` starts at `x[ci * ld]`.
#[derive(Clone, Copy)]
pub struct Im2col<'a> {
    pub x: &'a [f32],
    pub ld: usize,
    pub t_in: usize,
    pub channels: usize,
    pub taps: usize,
    pub stride: usize,
    pub dil: usize,
    pub pad: usize,
    /// Number of output columns.
    pub n: usize,
}

/// Right-hand GEMM operand.
#[derive(Clone, Copy)]
pub enum BSrc<'a> {
    Mat(BMat<'a>),
    Im2col(Im2col<'a>),
}

impl BSrc<'_> {
    pub fn k(&self) -> usize {
        match self {
            BSrc::Mat(b) => b.k,
            BSrc::Im2col(c) => c.channels * c.taps,
        }
    }

    pub fn n(&self) -> usize {
        match self {
            BSrc::Mat(b) => b.n,
            BSrc::Im2col(c) => c.n,
        }
    }
}

impl<'a> From<BMat<'a>> for BSrc<'a> {
    fn from(b: BMat<'a>) -> Self {
        BSrc::Mat(b)
    }
}

impl<'a> From<Im2col<'a>> for BSrc<'a> {
    fn from(c: Im2col<'a>) -> Self {
        BSrc::Im2col(c)
    }
}

type KernelFn = unsafe fn(
    kc: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    acc: bool,
);

/// A register-blocked micro-kernel computing an `mr × nr` tile of C.
pub struct Kernel {
    pub mr: usize,
    pub nr: usize,
    pub name: &'static str,
    f: KernelFn,
}

/// The best micro-kernel for this CPU (detected once).
pub fn kernel() -> &'static Kernel {
    static K: OnceLock<Kernel> = OnceLock::new();
    K.get_or_init(select_kernel)
}

#[allow(unreachable_code)]
fn select_kernel() -> Kernel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return Kernel {
                mr: 6,
                nr: 16,
                name: "x86_64 AVX2+FMA 6x16",
                f: x86::k6x16,
            };
        }
        return Kernel {
            mr: 4,
            nr: 8,
            name: "x86_64 SSE2 4x8",
            f: x86::k4x8,
        };
    }
    #[cfg(target_arch = "aarch64")]
    {
        return Kernel {
            mr: 8,
            nr: 8,
            name: "aarch64 NEON 8x8",
            f: neon::k8x8,
        };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return Kernel {
            mr: 4,
            nr: 8,
            name: "wasm32 SIMD128 4x8",
            f: wasm::k4x8,
        };
    }
    Kernel {
        mr: 4,
        nr: 4,
        name: "portable 4x4",
        f: k_scalar,
    }
}

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

/// `A` unpacked into MR-row panels, block by block along k.
///
/// Block `kb` covers columns `kb*KC .. kb*KC + kc` and starts at
/// `kb*KC * panels * mr`; inside, panel `p` starts at `p * mr * kc` and holds
/// element (row `p*mr + r`, column `kk`) at `kk * mr + r`.
pub struct PackedA {
    pub m: usize,
    pub k: usize,
    mr: usize,
    panels: usize,
    data: Vec<f32>,
}

impl PackedA {
    #[inline]
    fn kblocks(&self) -> usize {
        self.k.div_ceil(KC)
    }

    #[inline]
    fn block(&self, kb: usize) -> (usize, usize, &[f32]) {
        let k0 = kb * KC;
        let kc = KC.min(self.k - k0);
        let start = k0 * self.panels * self.mr;
        (
            k0,
            kc,
            &self.data[start..start + self.panels * self.mr * kc],
        )
    }

    /// Memory used by the packed matrix in bytes.
    pub fn bytes(&self) -> usize {
        self.data.len() * 4
    }

    /// `y = A x` computed directly on the packed panels.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        let mr = self.mr;
        let mut acc = vec![0f32; self.panels * mr];
        for kb in 0..self.kblocks() {
            let (k0, kc, seg) = self.block(kb);
            for p in 0..self.panels {
                let panel = &seg[p * mr * kc..(p + 1) * mr * kc];
                let out = &mut acc[p * mr..(p + 1) * mr];
                for (kk, col) in panel.chunks_exact(mr).enumerate() {
                    let xv = x[k0 + kk];
                    for (o, &a) in out.iter_mut().zip(col) {
                        *o += a * xv;
                    }
                }
            }
        }
        y[..self.m].copy_from_slice(&acc[..self.m]);
    }
}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: normalise.
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[inline]
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut v = m >> shift;
        if rem > half || (rem == half && (v & 1) == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (v & 1) == 1) {
        v += 1;
    }
    sign | v as u16
}

/// Writes row `i` of `a` (as f32) into `out[..a.k]`.
fn load_row(a: &AMat, i: usize, out: &mut [f32]) {
    match a.data {
        AData::F32 { data, rs, cs } => {
            if cs == 1 {
                out[..a.k].copy_from_slice(&data[i * rs..i * rs + a.k]);
            } else {
                for (j, o) in out[..a.k].iter_mut().enumerate() {
                    *o = data[i * rs + j * cs];
                }
            }
        }
        AData::F16 { data, ld } => {
            for (o, &h) in out[..a.k].iter_mut().zip(&data[i * ld..i * ld + a.k]) {
                *o = f16_to_f32(h);
            }
        }
        AData::Q8 { data, ld, scales } => {
            let s = scales[i];
            for (o, &q) in out[..a.k].iter_mut().zip(&data[i * ld..i * ld + a.k]) {
                *o = q as f32 * s;
            }
        }
    }
}

/// Packs `a` for micro-kernels with `mr` rows.
pub fn pack_a_mr(a: &AMat, mr: usize) -> PackedA {
    let panels = a.m.div_ceil(mr);
    let mut data = vec![0f32; panels * mr * a.k];
    let kblocks = a.k.div_ceil(KC);
    let dst = SendPtr(data.as_mut_ptr());
    let rows_per_task = 64usize.max(mr).div_ceil(mr) * mr;
    let tasks = if a.m * a.k > 1 << 18 {
        a.m.div_ceil(rows_per_task)
    } else {
        1
    };
    let pack_rows = |r0: usize, r1: usize| {
        let mut row = vec![0f32; a.k];
        for i in r0..r1 {
            load_row(a, i, &mut row);
            let p = i / mr;
            let r = i % mr;
            for kb in 0..kblocks {
                let k0 = kb * KC;
                let kc = KC.min(a.k - k0);
                let base = k0 * panels * mr + p * mr * kc + r;
                for kk in 0..kc {
                    // SAFETY: each (row, column) maps to a unique slot.
                    unsafe { *dst.get().add(base + kk * mr) = row[k0 + kk] };
                }
            }
        }
    };
    if tasks <= 1 {
        pack_rows(0, a.m);
    } else {
        par::for_each(tasks, |t| {
            let r0 = t * rows_per_task;
            pack_rows(r0, (r0 + rows_per_task).min(a.m));
        });
    }
    PackedA {
        m: a.m,
        k: a.k,
        mr,
        panels,
        data,
    }
}

/// Packs `a` for the active micro-kernel.
pub fn pack_a(a: &AMat) -> PackedA {
    pack_a_mr(a, kernel().mr)
}

/// Packs rows `k0..k0+kc`, columns `n0..n0+nc` of `b` into NR-column panels.
fn pack_b(b: &BSrc, k0: usize, kc: usize, n0: usize, nc: usize, nr: usize, out: &mut [f32]) {
    let panels = nc.div_ceil(nr);
    for q in 0..panels {
        let c0 = n0 + q * nr;
        let cols = nr.min(n0 + nc - c0);
        let dst = &mut out[q * nr * kc..(q + 1) * nr * kc];
        match b {
            BSrc::Mat(b) if b.cs == 1 => {
                for kk in 0..kc {
                    let src = (k0 + kk) * b.rs + c0;
                    let d = &mut dst[kk * nr..kk * nr + nr];
                    d[..cols].copy_from_slice(&b.data[src..src + cols]);
                    d[cols..].fill(0.0);
                }
            }
            BSrc::Mat(b) => {
                for kk in 0..kc {
                    let d = &mut dst[kk * nr..kk * nr + nr];
                    for (j, v) in d.iter_mut().enumerate() {
                        *v = if j < cols {
                            b.data[(k0 + kk) * b.rs + (c0 + j) * b.cs]
                        } else {
                            0.0
                        };
                    }
                }
            }
            BSrc::Im2col(c) => {
                let t_in = c.t_in as isize;
                for kk in 0..kc {
                    let row = k0 + kk;
                    let (ci, kt) = (row / c.taps, row % c.taps);
                    let src = &c.x[ci * c.ld..ci * c.ld + c.t_in];
                    let off = (kt * c.dil) as isize - c.pad as isize;
                    let d = &mut dst[kk * nr..kk * nr + nr];
                    if c.stride == 1 {
                        let t0 = c0 as isize + off;
                        if t0 >= 0 && t0 + cols as isize <= t_in {
                            d[..cols].copy_from_slice(&src[t0 as usize..t0 as usize + cols]);
                        } else {
                            for (j, v) in d[..cols].iter_mut().enumerate() {
                                let t = t0 + j as isize;
                                *v = if t >= 0 && t < t_in {
                                    src[t as usize]
                                } else {
                                    0.0
                                };
                            }
                        }
                    } else {
                        for (j, v) in d[..cols].iter_mut().enumerate() {
                            let t = ((c0 + j) * c.stride) as isize + off;
                            *v = if t >= 0 && t < t_in {
                                src[t as usize]
                            } else {
                                0.0
                            };
                        }
                    }
                    d[cols..].fill(0.0);
                }
            }
        }
    }
}

thread_local! {
    static BBUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Serial GEMM over columns `n0..n1` and row panels `p0..p1`.
#[allow(clippy::too_many_arguments)]
fn gemm_block(
    kern: &Kernel,
    pa: &PackedA,
    b: &BSrc,
    n0: usize,
    n1: usize,
    p0: usize,
    p1: usize,
    c: SendPtr<f32>,
    ldc: usize,
    accumulate: bool,
) {
    let (mr, nr) = (kern.mr, kern.nr);
    BBUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < KC * NC {
            buf.resize(KC * NC, 0.0);
        }
        let mut nb0 = n0;
        while nb0 < n1 {
            let nc = NC.min(n1 - nb0);
            let npanels = nc.div_ceil(nr);
            for kb in 0..pa.kblocks() {
                let (k0, kc, aseg) = pa.block(kb);
                pack_b(b, k0, kc, nb0, nc, nr, &mut buf);
                let acc = accumulate || kb > 0;
                for q in 0..npanels {
                    let cols = nr.min(nc - q * nr);
                    let bp = buf[q * nr * kc..].as_ptr();
                    for p in p0..p1 {
                        let rows = mr.min(pa.m - p * mr);
                        let ap = aseg[p * mr * kc..].as_ptr();
                        // SAFETY: tile (p, q) lies inside C; disjoint per task.
                        unsafe {
                            let cp = c.get().add(p * mr * ldc + nb0 + q * nr);
                            (kern.f)(kc, ap, bp, cp, ldc, rows, cols, acc);
                        }
                    }
                }
            }
            nb0 += nc;
        }
    });
}

/// `C (+)= A · B` with an already packed `A`.
pub fn gemm_packed(pa: &PackedA, b: &BSrc, c: &mut [f32], ldc: usize, accumulate: bool) {
    let kern = kernel();
    assert_eq!(pa.mr, kern.mr, "packed for another kernel");
    assert_eq!(pa.k, b.k(), "inner dimensions differ");
    let (m, n) = (pa.m, b.n());
    if m == 0 || n == 0 {
        return;
    }
    assert!(ldc >= n && c.len() >= (m - 1) * ldc + n, "C too small");
    if pa.k == 0 {
        if !accumulate {
            for i in 0..m {
                c[i * ldc..i * ldc + n].fill(0.0);
            }
        }
        return;
    }
    let cp = SendPtr(c.as_mut_ptr());
    let threads = par::num_threads();
    let work = m * n * pa.k;
    let panels = pa.panels;
    if threads <= 1 || work < 1 << 21 {
        gemm_block(kern, pa, b, 0, n, 0, panels, cp, ldc, accumulate);
        return;
    }
    let nr = kern.nr;
    let col_tasks = n.div_ceil(nr * 4);
    if col_tasks >= threads * 2 || col_tasks >= panels {
        // Split columns: every task packs its own B columns.
        let tasks = (threads * 4).min(col_tasks);
        let chunk = n.div_ceil(tasks).div_ceil(nr) * nr;
        let tasks = n.div_ceil(chunk);
        par::for_each(tasks, |t| {
            let n0 = t * chunk;
            let n1 = (n0 + chunk).min(n);
            gemm_block(kern, pa, b, n0, n1, 0, panels, cp, ldc, accumulate);
        });
    } else {
        // Split row panels: tasks share nothing but read the same B.
        let tasks = (threads * 2).min(panels);
        let chunk = panels.div_ceil(tasks);
        let tasks = panels.div_ceil(chunk);
        par::for_each(tasks, |t| {
            let p0 = t * chunk;
            let p1 = (p0 + chunk).min(panels);
            gemm_block(kern, pa, b, 0, n, p0, p1, cp, ldc, accumulate);
        });
    }
}

/// `C (+)= A · B`, packing `A` on the fly.
pub fn gemm(a: &AMat, b: &BSrc, c: &mut [f32], ldc: usize, accumulate: bool) {
    let pa = pack_a(a);
    gemm_packed(&pa, b, c, ldc, accumulate);
}

// ---------------------------------------------------------------------------
// Micro-kernels
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn store_tile(
    tmp: &[f32],
    ld: usize,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    acc: bool,
) {
    for r in 0..mr {
        let cp = c.add(r * ldc);
        for j in 0..nr {
            let v = tmp[r * ld + j];
            if acc {
                *cp.add(j) += v;
            } else {
                *cp.add(j) = v;
            }
        }
    }
}

unsafe fn k_scalar(
    kc: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    acc: bool,
) {
    let mut t = [0f32; 16];
    for p in 0..kc {
        let ap = a.add(p * 4);
        let bp = b.add(p * 4);
        for r in 0..4 {
            let av = *ap.add(r);
            for j in 0..4 {
                t[r * 4 + j] += av * *bp.add(j);
            }
        }
    }
    store_tile(&t, 4, c, ldc, mr, nr, acc);
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::store_tile;
    use std::arch::x86_64::*;

    macro_rules! row6 {
        ($a:ident, $b0:ident, $b1:ident, $c0:ident, $c1:ident, $($r:literal),*) => {$(
            let av = _mm256_broadcast_ss(&*$a.add($r));
            $c0[$r] = _mm256_fmadd_ps(av, $b0, $c0[$r]);
            $c1[$r] = _mm256_fmadd_ps(av, $b1, $c1[$r]);
        )*};
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn k6x16(
        kc: usize,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        ldc: usize,
        mr: usize,
        nr: usize,
        acc: bool,
    ) {
        let mut c0 = [_mm256_setzero_ps(); 6];
        let mut c1 = [_mm256_setzero_ps(); 6];
        let mut a = a;
        let mut b = b;
        for _ in 0..kc {
            let b0 = _mm256_loadu_ps(b);
            let b1 = _mm256_loadu_ps(b.add(8));
            row6!(a, b0, b1, c0, c1, 0, 1, 2, 3, 4, 5);
            a = a.add(6);
            b = b.add(16);
        }
        if mr == 6 && nr == 16 {
            for r in 0..6 {
                let cp = c.add(r * ldc);
                let (mut v0, mut v1) = (c0[r], c1[r]);
                if acc {
                    v0 = _mm256_add_ps(v0, _mm256_loadu_ps(cp));
                    v1 = _mm256_add_ps(v1, _mm256_loadu_ps(cp.add(8)));
                }
                _mm256_storeu_ps(cp, v0);
                _mm256_storeu_ps(cp.add(8), v1);
            }
        } else {
            let mut t = [0f32; 96];
            for r in 0..6 {
                _mm256_storeu_ps(t.as_mut_ptr().add(r * 16), c0[r]);
                _mm256_storeu_ps(t.as_mut_ptr().add(r * 16 + 8), c1[r]);
            }
            store_tile(&t, 16, c, ldc, mr, nr, acc);
        }
    }

    macro_rules! row4 {
        ($a:ident, $b0:ident, $b1:ident, $c0:ident, $c1:ident, $($r:literal),*) => {$(
            let av = _mm_set1_ps(*$a.add($r));
            $c0[$r] = _mm_add_ps($c0[$r], _mm_mul_ps(av, $b0));
            $c1[$r] = _mm_add_ps($c1[$r], _mm_mul_ps(av, $b1));
        )*};
    }

    pub unsafe fn k4x8(
        kc: usize,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        ldc: usize,
        mr: usize,
        nr: usize,
        acc: bool,
    ) {
        let mut c0 = [_mm_setzero_ps(); 4];
        let mut c1 = [_mm_setzero_ps(); 4];
        let mut a = a;
        let mut b = b;
        for _ in 0..kc {
            let b0 = _mm_loadu_ps(b);
            let b1 = _mm_loadu_ps(b.add(4));
            row4!(a, b0, b1, c0, c1, 0, 1, 2, 3);
            a = a.add(4);
            b = b.add(8);
        }
        let mut t = [0f32; 32];
        for r in 0..4 {
            _mm_storeu_ps(t.as_mut_ptr().add(r * 8), c0[r]);
            _mm_storeu_ps(t.as_mut_ptr().add(r * 8 + 4), c1[r]);
        }
        store_tile(&t, 8, c, ldc, mr, nr, acc);
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::store_tile;
    use std::arch::aarch64::*;

    macro_rules! lanes {
        ($c:ident, $b0:ident, $b1:ident, $av:ident, $base:literal, $($l:literal),*) => {$(
            $c[$base + $l][0] = vfmaq_laneq_f32::<$l>($c[$base + $l][0], $b0, $av);
            $c[$base + $l][1] = vfmaq_laneq_f32::<$l>($c[$base + $l][1], $b1, $av);
        )*};
    }

    pub unsafe fn k8x8(
        kc: usize,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        ldc: usize,
        mr: usize,
        nr: usize,
        acc: bool,
    ) {
        let z = vdupq_n_f32(0.0);
        let mut cc = [[z; 2]; 8];
        let mut a = a;
        let mut b = b;
        for _ in 0..kc {
            let b0 = vld1q_f32(b);
            let b1 = vld1q_f32(b.add(4));
            let a0 = vld1q_f32(a);
            let a1 = vld1q_f32(a.add(4));
            lanes!(cc, b0, b1, a0, 0, 0, 1, 2, 3);
            lanes!(cc, b0, b1, a1, 4, 0, 1, 2, 3);
            a = a.add(8);
            b = b.add(8);
        }
        if mr == 8 && nr == 8 {
            for r in 0..8 {
                let cp = c.add(r * ldc);
                let (mut v0, mut v1) = (cc[r][0], cc[r][1]);
                if acc {
                    v0 = vaddq_f32(v0, vld1q_f32(cp));
                    v1 = vaddq_f32(v1, vld1q_f32(cp.add(4)));
                }
                vst1q_f32(cp, v0);
                vst1q_f32(cp.add(4), v1);
            }
        } else {
            let mut t = [0f32; 64];
            for r in 0..8 {
                vst1q_f32(t.as_mut_ptr().add(r * 8), cc[r][0]);
                vst1q_f32(t.as_mut_ptr().add(r * 8 + 4), cc[r][1]);
            }
            store_tile(&t, 8, c, ldc, mr, nr, acc);
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use super::store_tile;
    use std::arch::wasm32::*;

    #[inline(always)]
    unsafe fn madd(a: v128, b: v128, c: v128) -> v128 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            f32x4_relaxed_madd(a, b, c)
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            f32x4_add(c, f32x4_mul(a, b))
        }
    }

    macro_rules! row4 {
        ($a:ident, $b0:ident, $b1:ident, $c0:ident, $c1:ident, $($r:literal),*) => {$(
            let av = f32x4_splat(*$a.add($r));
            $c0[$r] = madd(av, $b0, $c0[$r]);
            $c1[$r] = madd(av, $b1, $c1[$r]);
        )*};
    }

    pub unsafe fn k4x8(
        kc: usize,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        ldc: usize,
        mr: usize,
        nr: usize,
        acc: bool,
    ) {
        let z = f32x4_splat(0.0);
        let mut c0 = [z; 4];
        let mut c1 = [z; 4];
        let mut a = a;
        let mut b = b;
        for _ in 0..kc {
            let b0 = v128_load(b as *const v128);
            let b1 = v128_load(b.add(4) as *const v128);
            row4!(a, b0, b1, c0, c1, 0, 1, 2, 3);
            a = a.add(4);
            b = b.add(8);
        }
        if mr == 4 && nr == 8 {
            for r in 0..4 {
                let cp = c.add(r * ldc);
                let (mut v0, mut v1) = (c0[r], c1[r]);
                if acc {
                    v0 = f32x4_add(v0, v128_load(cp as *const v128));
                    v1 = f32x4_add(v1, v128_load(cp.add(4) as *const v128));
                }
                v128_store(cp as *mut v128, v0);
                v128_store(cp.add(4) as *mut v128, v1);
            }
        } else {
            let mut t = [0f32; 32];
            for r in 0..4 {
                v128_store(t.as_mut_ptr().add(r * 8) as *mut v128, c0[r]);
                v128_store(t.as_mut_ptr().add(r * 8 + 4) as *mut v128, c1[r]);
            }
            store_tile(&t, 8, c, ldc, mr, nr, acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Dot products (LSTM recurrences, small matrix-vector products)
// ---------------------------------------------------------------------------

/// Dot product with 8 independent accumulators (vectorises without fast-math).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = [0f32; 8];
    let chunks = n / 8;
    for i in 0..chunks {
        let x = &a[i * 8..i * 8 + 8];
        let y = &b[i * 8..i * 8 + 8];
        for j in 0..8 {
            acc[j] += x[j] * y[j];
        }
    }
    let mut s = (acc[0] + acc[4]) + (acc[1] + acc[5]) + (acc[2] + acc[6]) + (acc[3] + acc[7]);
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
}

/// `y = W x` for row-major `W` (`rows × cols`).
pub fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    for (i, yi) in y[..rows].iter_mut().enumerate() {
        *yi = dot(&w[i * cols..(i + 1) * cols], &x[..cols]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f64;
                for p in 0..k {
                    s += a[i * k + p] as f64 * b[p * n + j] as f64;
                }
                c[i * n + j] = s as f32;
            }
        }
        c
    }

    fn rnd(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn gemm_matches_naive() {
        for &(m, k, n) in &[
            (1, 1, 1),
            (7, 3, 5),
            (13, 300, 37),
            (64, 513, 250),
            (129, 64, 1000),
            (6, 16, 16),
        ] {
            let a = rnd(m * k, 1);
            let b = rnd(k * n, 2);
            let want = naive(m, k, n, &a, &b);
            let mut c = vec![0f32; m * n];
            gemm(
                &AMat {
                    m,
                    k,
                    data: AData::F32 {
                        data: &a,
                        rs: k,
                        cs: 1,
                    },
                },
                &BMat::row_major(&b, k, n, n).into(),
                &mut c,
                n,
                false,
            );
            for (x, y) in c.iter().zip(&want) {
                assert!((x - y).abs() < 1e-3, "{m}x{k}x{n}: {x} vs {y}");
            }
            // accumulate
            gemm(
                &AMat {
                    m,
                    k,
                    data: AData::F32 {
                        data: &a,
                        rs: k,
                        cs: 1,
                    },
                },
                &BMat::row_major(&b, k, n, n).into(),
                &mut c,
                n,
                true,
            );
            for (x, y) in c.iter().zip(&want) {
                assert!((x - 2.0 * y).abs() < 2e-3);
            }
        }
    }

    #[test]
    fn half_roundtrip() {
        for &v in &[0.0f32, 1.0, -2.5, 65504.0, 1e-5, -3.14159, 6.1e-5, 1e-7] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert!((back - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} -> {back}");
        }
    }
}
