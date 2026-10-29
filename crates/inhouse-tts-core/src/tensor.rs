//! Dense activations (`Mat`) and compact weight matrices (`Matrix`).

use crate::gemm::{self, AData, AMat, BMat, BSrc, PackedA};

/// Row-major f32 matrix. Sequence activations use `[channels, time]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(rows * cols, data.len(), "shape mismatch");
        Mat { rows, cols, data }
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    #[inline]
    pub fn at(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    pub fn as_b(&self) -> BSrc<'_> {
        BMat::row_major(&self.data, self.rows, self.cols, self.cols).into()
    }

    pub fn transpose(&self) -> Mat {
        let mut out = Mat::zeros(self.cols, self.rows);
        const B: usize = 32;
        for r0 in (0..self.rows).step_by(B) {
            for c0 in (0..self.cols).step_by(B) {
                for r in r0..(r0 + B).min(self.rows) {
                    for c in c0..(c0 + B).min(self.cols) {
                        out.data[c * self.rows + r] = self.data[r * self.cols + c];
                    }
                }
            }
        }
        out
    }

    /// Stacks matrices with equal column counts vertically.
    pub fn vcat(parts: &[&Mat]) -> Mat {
        let cols = parts[0].cols;
        let rows = parts.iter().map(|m| m.rows).sum();
        let mut data = Vec::with_capacity(rows * cols);
        for p in parts {
            assert_eq!(p.cols, cols, "vcat column mismatch");
            data.extend_from_slice(&p.data);
        }
        Mat { rows, cols, data }
    }

    /// Repeats column `j` `counts[j]` times (length regulation).
    pub fn repeat_cols(&self, counts: &[usize]) -> Mat {
        assert_eq!(counts.len(), self.cols);
        let total: usize = counts.iter().sum();
        let mut out = Mat::zeros(self.rows, total);
        for r in 0..self.rows {
            let src = self.row(r);
            let dst = &mut out.data[r * total..(r + 1) * total];
            let mut o = 0;
            for (j, &n) in counts.iter().enumerate() {
                dst[o..o + n].fill(src[j]);
                o += n;
            }
        }
        out
    }

    pub fn scale_inplace(&mut self, s: f32) {
        for v in &mut self.data {
            *v *= s;
        }
    }

    pub fn add_inplace(&mut self, other: &Mat) {
        assert_eq!(self.data.len(), other.data.len());
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += *b;
        }
    }
}

/// Storage formats for weight matrices.
#[derive(Clone, Debug)]
pub enum Storage {
    F32(Vec<f32>),
    F16(Vec<u16>),
    /// int8 values with one f32 scale per row.
    Q8 {
        q: Vec<i8>,
        scales: Vec<f32>,
    },
}

impl Storage {
    pub fn bytes(&self) -> usize {
        match self {
            Storage::F32(v) => v.len() * 4,
            Storage::F16(v) => v.len() * 2,
            Storage::Q8 { q, scales } => q.len() + scales.len() * 4,
        }
    }
}

/// Weight matrix used as the left GEMM operand (`rows × cols`, row-major).
///
/// It is either kept compact and unpacked on every call (low memory, used on
/// the web) or pre-packed for the active micro-kernel (fastest, native).
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    storage: Option<Storage>,
    packed: Option<PackedA>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, storage: Storage, prepack: bool) -> Self {
        let mut m = Matrix {
            rows,
            cols,
            storage: Some(storage),
            packed: None,
        };
        if prepack {
            m.packed = Some(gemm::pack_a(&m.amat()));
            m.storage = None;
        }
        m
    }

    pub fn from_f32(rows: usize, cols: usize, data: Vec<f32>, prepack: bool) -> Self {
        assert_eq!(rows * cols, data.len());
        Matrix::new(rows, cols, Storage::F32(data), prepack)
    }

    fn amat(&self) -> AMat<'_> {
        let data = match self.storage.as_ref().expect("matrix storage released") {
            Storage::F32(v) => AData::F32 {
                data: v,
                rs: self.cols,
                cs: 1,
            },
            Storage::F16(v) => AData::F16 {
                data: v,
                ld: self.cols,
            },
            Storage::Q8 { q, scales } => AData::Q8 {
                data: q,
                ld: self.cols,
                scales,
            },
        };
        AMat {
            m: self.rows,
            k: self.cols,
            data,
        }
    }

    /// Compact weights, unless the matrix was pre-packed for the CPU.
    pub fn storage(&self) -> Option<&Storage> {
        self.storage.as_ref()
    }

    /// Packs the matrix for the current kernel (or returns the cached pack).
    pub fn with_packed<R>(&self, f: impl FnOnce(&PackedA) -> R) -> R {
        match &self.packed {
            Some(p) => f(p),
            None => f(&gemm::pack_a(&self.amat())),
        }
    }

    /// `C (+)= self · B`.
    pub fn gemm(&self, b: &BSrc, c: &mut [f32], ldc: usize, accumulate: bool) {
        self.with_packed(|p| gemm::gemm_packed(p, b, c, ldc, accumulate));
    }

    /// Dense f32 copy of row `r`.
    pub fn row_f32(&self, r: usize) -> Vec<f32> {
        let mut out = vec![0f32; self.cols];
        match self.storage.as_ref().expect("matrix storage released") {
            Storage::F32(v) => out.copy_from_slice(&v[r * self.cols..(r + 1) * self.cols]),
            Storage::F16(v) => {
                for (o, &h) in out.iter_mut().zip(&v[r * self.cols..(r + 1) * self.cols]) {
                    *o = gemm::f16_to_f32(h);
                }
            }
            Storage::Q8 { q, scales } => {
                for (o, &x) in out.iter_mut().zip(&q[r * self.cols..(r + 1) * self.cols]) {
                    *o = x as f32 * scales[r];
                }
            }
        }
        out
    }

    /// `y = self · x` for a single vector.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        assert!(x.len() >= self.cols && y.len() >= self.rows);
        if let Some(p) = &self.packed {
            p.matvec(x, y);
            return;
        }
        match self.storage.as_ref().expect("matrix storage released") {
            Storage::F32(v) => gemm::matvec(v, self.rows, self.cols, x, y),
            _ => {
                for (r, yr) in y[..self.rows].iter_mut().enumerate() {
                    *yr = gemm::dot(&self.row_f32(r), x);
                }
            }
        }
    }

    pub fn bytes(&self) -> usize {
        self.storage.as_ref().map_or(0, Storage::bytes)
            + self.packed.as_ref().map_or(0, PackedA::bytes)
    }
}
