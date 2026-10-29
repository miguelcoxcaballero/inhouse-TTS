//! `.ihw` — the Inhouse weights container.
//!
//! Layout: `b"IHW1"`, little-endian `u32` header length, UTF-8 JSON header,
//! zero padding to a 64-byte boundary, then the tensor data region. Every
//! tensor starts on a 64-byte boundary (offsets are relative to the data
//! region). `q8` tensors store `rows × cols` int8 values followed by `rows`
//! little-endian f32 scales (row = first dimension).

use crate::gemm::{f16_to_f32, f32_to_f16};
use crate::tensor::{Matrix, Storage};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

pub const MAGIC: &[u8; 4] = b"IHW1";

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    F16,
    Q8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub len: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Header {
    pub format: u32,
    /// Model family, e.g. `"k82"` or `"voices"`.
    pub model: String,
    #[serde(default)]
    pub meta: serde_json::Value,
    pub tensors: Vec<TensorInfo>,
}

/// One tensor decoded from a container.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub storage: Storage,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Dense f32 copy.
    pub fn to_f32(&self) -> Vec<f32> {
        match &self.storage {
            Storage::F32(v) => v.clone(),
            Storage::F16(v) => v.iter().map(|&h| f16_to_f32(h)).collect(),
            Storage::Q8 { q, scales } => {
                let cols = if scales.is_empty() {
                    0
                } else {
                    q.len() / scales.len()
                };
                q.iter()
                    .enumerate()
                    .map(|(i, &x)| x as f32 * scales[i / cols])
                    .collect()
            }
        }
    }
}

/// An opened container: header plus decoded tensors.
pub struct WeightFile {
    pub header: Header,
    tensors: BTreeMap<String, Tensor>,
}

impl WeightFile {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 || &bytes[..4] != MAGIC {
            return err("not an Inhouse weights file (bad magic)");
        }
        let hlen = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if 8 + hlen > bytes.len() {
            return err("truncated header");
        }
        let header: Header = serde_json::from_slice(&bytes[8..8 + hlen])
            .map_err(|e| Error(format!("bad header: {e}")))?;
        let data_start = (8 + hlen).div_ceil(64) * 64;
        let mut tensors = BTreeMap::new();
        for t in &header.tensors {
            let start = data_start + t.offset;
            let end = start + t.len;
            if end > bytes.len() {
                return err(format!("tensor {} out of bounds", t.name));
            }
            let raw = &bytes[start..end];
            let numel: usize = t.shape.iter().product();
            let storage = match t.dtype {
                DType::F32 => {
                    if raw.len() != numel * 4 {
                        return err(format!("tensor {} has wrong size", t.name));
                    }
                    Storage::F32(
                        raw.chunks_exact(4)
                            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                            .collect(),
                    )
                }
                DType::F16 => {
                    if raw.len() != numel * 2 {
                        return err(format!("tensor {} has wrong size", t.name));
                    }
                    Storage::F16(
                        raw.chunks_exact(2)
                            .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
                            .collect(),
                    )
                }
                DType::Q8 => {
                    let rows = t.shape.first().copied().unwrap_or(1);
                    if raw.len() != numel + rows * 4 {
                        return err(format!("tensor {} has wrong size", t.name));
                    }
                    let q = raw[..numel].iter().map(|&b| b as i8).collect();
                    let scales = raw[numel..]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    Storage::Q8 { q, scales }
                }
            };
            tensors.insert(
                t.name.clone(),
                Tensor {
                    shape: t.shape.clone(),
                    storage,
                },
            );
        }
        Ok(WeightFile { header, tensors })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error(format!("missing tensor {name}")))
    }

    /// Removes a tensor (lets loaders release memory as they go).
    pub fn take(&mut self, name: &str) -> Result<Tensor> {
        self.tensors
            .remove(name)
            .ok_or_else(|| Error(format!("missing tensor {name}")))
    }

    pub fn f32(&mut self, name: &str, numel: usize) -> Result<Vec<f32>> {
        let t = self.take(name)?;
        if t.numel() != numel {
            return err(format!(
                "tensor {name}: expected {numel} values, found {:?}",
                t.shape
            ));
        }
        Ok(t.to_f32())
    }

    /// Takes a tensor as a `rows × cols` weight matrix (first dim = rows).
    pub fn matrix(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
        prepack: bool,
    ) -> Result<Matrix> {
        let t = self.take(name)?;
        if t.numel() != rows * cols || t.shape.first() != Some(&rows) {
            return err(format!(
                "tensor {name}: expected {rows}x{cols}, found {:?}",
                t.shape
            ));
        }
        Ok(Matrix::new(rows, cols, t.storage, prepack))
    }
}

/// Builds containers.
pub struct Writer {
    model: String,
    meta: serde_json::Value,
    infos: Vec<TensorInfo>,
    data: Vec<u8>,
}

impl Writer {
    pub fn new(model: &str, meta: serde_json::Value) -> Self {
        Writer {
            model: model.to_string(),
            meta,
            infos: Vec::new(),
            data: Vec::new(),
        }
    }

    fn push(&mut self, name: &str, dtype: DType, shape: &[usize], bytes: &[u8]) {
        let offset = self.data.len().div_ceil(64) * 64;
        self.data.resize(offset, 0);
        self.data.extend_from_slice(bytes);
        self.infos.push(TensorInfo {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            offset,
            len: bytes.len(),
        });
    }

    /// Adds a tensor, converting it to `dtype` (q8 quantises per first-dim row).
    pub fn add(&mut self, name: &str, shape: &[usize], values: &[f32], dtype: DType) {
        assert_eq!(
            shape.iter().product::<usize>(),
            values.len(),
            "{name}: shape mismatch"
        );
        match dtype {
            DType::F32 => {
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                self.push(name, dtype, shape, &bytes);
            }
            DType::F16 => {
                let bytes: Vec<u8> = values
                    .iter()
                    .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                    .collect();
                self.push(name, dtype, shape, &bytes);
            }
            DType::Q8 => {
                let rows = shape[0];
                let cols = values.len() / rows;
                let (q, scales) = quantize_rows(values, rows, cols);
                let mut bytes: Vec<u8> = q.iter().map(|&x| x as u8).collect();
                for s in scales {
                    bytes.extend_from_slice(&s.to_le_bytes());
                }
                self.push(name, dtype, shape, &bytes);
            }
        }
    }

    pub fn finish(self) -> Vec<u8> {
        let header = Header {
            format: 1,
            model: self.model,
            meta: self.meta,
            tensors: self.infos,
        };
        let json = serde_json::to_vec(&header).expect("header serialises");
        let mut out = Vec::with_capacity(json.len() + self.data.len() + 128);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(&json);
        out.resize(out.len().div_ceil(64) * 64, 0);
        out.extend_from_slice(&self.data);
        out
    }
}

/// Symmetric per-row int8 quantisation.
pub fn quantize_rows(values: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * cols];
    let mut scales = vec![0f32; rows];
    for r in 0..rows {
        let row = &values[r * cols..(r + 1) * cols];
        let max = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        let s = if max > 0.0 { max / 127.0 } else { 1.0 };
        scales[r] = s;
        for (o, &v) in q[r * cols..(r + 1) * cols].iter_mut().zip(row) {
            *o = (v / s).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut w = Writer::new("test", serde_json::json!({"a": 1}));
        let v: Vec<f32> = (0..12).map(|i| i as f32 * 0.25 - 1.0).collect();
        w.add("x", &[3, 4], &v, DType::F32);
        w.add("h", &[3, 4], &v, DType::F16);
        w.add("q", &[3, 4], &v, DType::Q8);
        let bytes = w.finish();
        let mut f = WeightFile::parse(&bytes).unwrap();
        assert_eq!(f.header.meta["a"], 1);
        assert_eq!(f.f32("x", 12).unwrap(), v);
        assert_eq!(f.f32("h", 12).unwrap(), v);
        let q = f.f32("q", 12).unwrap();
        for (a, b) in q.iter().zip(&v) {
            assert!((a - b).abs() < 0.01);
        }
    }
}
