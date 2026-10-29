//! Monotonic alignment search (Glow-TTS / VITS): the most likely monotonic,
//! surjective path from frames to text tokens.

use burn::tensor::{Tensor, TensorData};
use rayon::prelude::*;

/// Best path for one item. `value` is `[ty, tx]` log-likelihoods (row
/// major, frames × tokens); returns a 0/1 matrix of the same shape.
pub fn maximum_path(value: &[f32], ty: usize, tx: usize, stride: usize) -> Vec<f32> {
    let neg = f32::NEG_INFINITY;
    let mut v = vec![neg; ty * tx];
    for y in 0..ty {
        let x_lo = tx.saturating_sub(ty - y).min(tx);
        let x_hi = (y + 1).min(tx);
        for x in x_lo..x_hi {
            let cur = if x == y {
                neg
            } else if y == 0 {
                neg
            } else {
                v[(y - 1) * tx + x]
            };
            let prev = if x == 0 {
                if y == 0 {
                    0.0
                } else {
                    neg
                }
            } else if y == 0 {
                neg
            } else {
                v[(y - 1) * tx + x - 1]
            };
            v[y * tx + x] = value[y * stride + x] + cur.max(prev);
        }
    }
    let mut path = vec![0f32; ty * tx];
    let mut index = tx - 1;
    for y in (0..ty).rev() {
        path[y * tx + index] = 1.0;
        if index != 0
            && (index == y || (y > 0 && v[(y - 1) * tx + index] < v[(y - 1) * tx + index - 1]))
        {
            index -= 1;
        }
    }
    path
}

/// Batched alignment of `neg` `[B, Ty, Tx]` with per-item lengths; returns
/// the hard alignment `[B, Ty, Tx]` on the same device.
pub fn align(neg: Tensor<3>, tx_lens: &[usize], ty_lens: &[usize]) -> Tensor<3> {
    let [b, ty, tx] = neg.dims();
    let dev = neg.device();
    let data: Vec<f32> = neg
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32 log-likelihoods");
    let paths: Vec<Vec<f32>> = (0..b)
        .into_par_iter()
        .map(|i| {
            let (lx, ly) = (tx_lens[i].min(tx), ty_lens[i].min(ty));
            let mut out = vec![0f32; ty * tx];
            if lx == 0 || ly == 0 {
                return out;
            }
            let base = &data[i * ty * tx..];
            let p = maximum_path(base, ly, lx, tx);
            for y in 0..ly {
                out[y * tx..y * tx + lx].copy_from_slice(&p[y * lx..(y + 1) * lx]);
            }
            out
        })
        .collect();
    let flat: Vec<f32> = paths.into_iter().flatten().collect();
    Tensor::from_data(TensorData::new(flat, [b, ty, tx]), &dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_is_monotonic_and_covers_every_token() {
        let (ty, tx) = (10, 4);
        // Likelihood peaks on a diagonal-ish alignment.
        let mut v = vec![-10f32; ty * tx];
        for y in 0..ty {
            v[y * tx + (y * tx / ty)] = 0.0;
        }
        let p = maximum_path(&v, ty, tx, tx);
        let mut last = 0;
        for y in 0..ty {
            let row = &p[y * tx..(y + 1) * tx];
            assert_eq!(row.iter().sum::<f32>(), 1.0);
            let x = row.iter().position(|&a| a == 1.0).unwrap();
            assert!(x >= last && x <= last + 1);
            last = x;
        }
        for x in 0..tx {
            assert!((0..ty).any(|y| p[y * tx + x] == 1.0), "token {x} unused");
        }
    }
}
