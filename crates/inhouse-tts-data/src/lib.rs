//! # Inhouse TTS data
//!
//! Tooling that turns permissively licensed speech corpora into training
//! manifests for Inhouse TTS: download, decode, resample to 24 kHz mono,
//! trim, normalise loudness and phonemise every transcript with the Inhouse
//! front end for the corpus' accent.

pub mod audio;
pub mod corpus;
pub mod fetch;

pub use corpus::{prepare, read_manifest, CorpusSpec, Report, Utterance, SAMPLE_RATE};
