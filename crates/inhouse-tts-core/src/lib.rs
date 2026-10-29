//! # Inhouse TTS core
//!
//! Pure-Rust neural inference engine of Inhouse TTS: packed SIMD GEMM
//! (AVX2/FMA, SSE2, NEON, WASM SIMD128), 1-D convolutions, LSTMs,
//! attention, normalisation layers, an iSTFT vocoder and the `.ihw` weight
//! container. No C/C++ dependencies; runs natively, on Android and in the
//! browser through WebAssembly.

pub mod dsp;
pub mod gemm;
pub mod math;
pub mod model;
pub mod ops;
pub mod par;
pub mod profile;
pub mod tensor;
#[cfg(feature = "convert")]
pub mod torchload;
pub mod weights;

pub use model::k82::{InferOptions, LoadOptions, Output, K82};
pub use weights::{DType, WeightFile};
