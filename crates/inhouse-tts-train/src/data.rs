//! Training data: manifests → symbol, speaker and locale tables, and padded
//! batches of phoneme ids and waveforms read from disk.

use crate::config::Config;
use crate::model::Batch;
use burn::tensor::{Device, Int, Tensor, TensorData};
use inhouse_tts_data::Utterance;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Small deterministic PRNG (PCG32).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut r = Rng(seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407));
        r.next();
        r
    }

    pub fn next(&mut self) -> u32 {
        let old = self.0;
        self.0 = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let xs = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xs.rotate_right(rot)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() as u64 * n as u64 >> 32) as usize
    }
}

/// Symbol, speaker and locale tables saved with every run.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Tables {
    /// Index 0 is the pad/blank symbol.
    pub symbols: Vec<String>,
    pub speakers: Vec<String>,
    pub locales: Vec<String>,
    /// Display name and locale of every speaker.
    pub speaker_locale: BTreeMap<String, String>,
}

impl Tables {
    pub fn build(utts: &[Utterance]) -> Tables {
        let mut syms: Vec<char> = utts.iter().flat_map(|u| u.phonemes.chars()).collect();
        syms.sort_unstable();
        syms.dedup();
        let mut speakers: Vec<String> = utts.iter().map(|u| u.speaker.clone()).collect();
        speakers.sort();
        speakers.dedup();
        let mut locales: Vec<String> = utts.iter().map(|u| u.locale.clone()).collect();
        locales.sort();
        locales.dedup();
        let speaker_locale = utts
            .iter()
            .map(|u| (u.speaker.clone(), u.locale.clone()))
            .collect();
        Tables {
            symbols: std::iter::once("_".to_string())
                .chain(syms.into_iter().map(|c| c.to_string()))
                .collect(),
            speakers,
            locales,
            speaker_locale,
        }
    }

    /// Phonemes → ids with a blank (0) between symbols (VITS `add_blank`).
    pub fn encode(&self, phonemes: &str) -> Vec<i64> {
        let mut out = vec![0i64];
        for ch in phonemes.chars() {
            let s = ch.to_string();
            if let Some(i) = self.symbols.iter().position(|x| *x == s) {
                out.push(i as i64);
                out.push(0);
            }
        }
        out
    }

    pub fn speaker(&self, s: &str) -> usize {
        self.speakers.iter().position(|x| x == s).unwrap_or(0)
    }

    pub fn locale(&self, s: &str) -> usize {
        self.locales.iter().position(|x| x == s).unwrap_or(0)
    }
}

/// One example. Its audio stays on disk and is read when a batch needs it
/// (the OS page cache keeps hot files in memory), so RAM does not grow with
/// the corpus.
struct Example {
    ids: Vec<i64>,
    path: PathBuf,
    frames: usize,
    speaker: usize,
    locale: usize,
}

pub struct Dataset {
    examples: Vec<Example>,
    hop: usize,
}

pub const SR: u32 = 24_000;

/// Sample count of a 24 kHz mono 16-bit WAV (as written by `ihtts-data`).
fn wav_len(path: &Path) -> Result<usize, String> {
    let r = hound::WavReader::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let s = r.spec();
    if s.sample_rate != SR || s.channels != 1 || s.bits_per_sample != 16 {
        return Err(format!(
            "{}: expected 24 kHz mono 16-bit WAV (prepare the corpus with ihtts-data)",
            path.display()
        ));
    }
    Ok(r.duration() as usize)
}

/// The first `n` samples of such a WAV, as floats.
fn read_wav(path: &Path, n: usize) -> Vec<f32> {
    let Ok(mut r) = hound::WavReader::open(path) else {
        return vec![0.0; n];
    };
    let mut x: Vec<f32> = r
        .samples::<i16>()
        .take(n)
        .map(|v| v.unwrap_or(0) as f32 / 32768.0)
        .collect();
    x.resize(n, 0.0);
    x
}

impl Dataset {
    /// Indexes every usable utterance of the manifests (lengths come from the
    /// WAV headers; no audio is decoded here).
    pub fn load(
        manifests: &[PathBuf],
        cfg: &Config,
        tables: &Tables,
        max_items: usize,
    ) -> Result<Dataset, String> {
        let hop = cfg.audio.hop;
        let mut items: Vec<(PathBuf, Utterance)> = Vec::new();
        for m in manifests {
            let dir = m.parent().unwrap_or(Path::new(".")).to_path_buf();
            for u in inhouse_tts_data::read_manifest(m)? {
                items.push((dir.join(&u.audio), u));
            }
        }
        items.truncate(max_items);
        let max_frames = cfg.train.max_frames;
        let examples: Vec<Result<Option<Example>, String>> = items
            .par_iter()
            .map(|(path, u)| {
                let frames = wav_len(path)? / hop;
                if frames < cfg.train.segment_frames + 2 || frames > max_frames {
                    return Ok(None);
                }
                Ok(Some(Example {
                    ids: tables.encode(&u.phonemes),
                    path: path.clone(),
                    frames,
                    speaker: tables.speaker(&u.speaker),
                    locale: tables.locale(&u.locale),
                }))
            })
            .collect();
        let mut kept = Vec::with_capacity(examples.len());
        for e in examples {
            if let Some(e) = e? {
                kept.push(e);
            }
        }
        if kept.is_empty() {
            return Err("no usable utterances".into());
        }
        Ok(Dataset {
            examples: kept,
            hop,
        })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn hours(&self) -> f64 {
        self.examples
            .iter()
            .map(|e| e.frames * self.hop)
            .sum::<usize>() as f64
            / SR as f64
            / 3600.0
    }

    /// Random batch of similar-length utterances (length bucketing keeps
    /// padding low).
    pub fn batch(&self, rng: &mut Rng, size: usize, dev: &Device) -> Batch {
        let n = self.examples.len();
        let pivot = rng.below(n);
        let target = self.examples[pivot].frames as i64;
        // Sample candidates and keep the closest in length.
        let mut cand: Vec<usize> = (0..size * 4).map(|_| rng.below(n)).collect();
        cand.sort_by_key(|&i| (self.examples[i].frames as i64 - target).abs());
        cand.truncate(size);
        let ex: Vec<&Example> = cand.iter().map(|&i| &self.examples[i]).collect();
        let b = ex.len();
        let tx = ex.iter().map(|e| e.ids.len()).max().unwrap_or(1);
        let ty = ex.iter().map(|e| e.frames).max().unwrap_or(1);
        let wavs: Vec<Vec<f32>> = ex
            .par_iter()
            .map(|e| read_wav(&e.path, e.frames * self.hop))
            .collect();
        let mut ids = vec![0i64; b * tx];
        let mut id_mask = vec![0f32; b * tx];
        let mut spec_mask = vec![0f32; b * ty];
        let mut wav = vec![0f32; b * ty * self.hop];
        for (i, e) in ex.iter().enumerate() {
            ids[i * tx..i * tx + e.ids.len()].copy_from_slice(&e.ids);
            id_mask[i * tx..i * tx + e.ids.len()].fill(1.0);
            spec_mask[i * ty..i * ty + e.frames].fill(1.0);
            wav[i * ty * self.hop..i * ty * self.hop + wavs[i].len()].copy_from_slice(&wavs[i]);
        }
        Batch {
            ids: Tensor::<2, Int>::from_data(TensorData::new(ids, [b, tx]), dev),
            id_mask: Tensor::from_data(TensorData::new(id_mask, [b, 1, tx]), dev),
            id_lens: ex.iter().map(|e| e.ids.len()).collect(),
            spec_mask: Tensor::from_data(TensorData::new(spec_mask, [b, 1, ty]), dev),
            spec_lens: ex.iter().map(|e| e.frames).collect(),
            wav: Tensor::from_data(TensorData::new(wav, [b, 1, ty * self.hop]), dev),
            speaker: Tensor::<1, Int>::from_data(
                TensorData::new(ex.iter().map(|e| e.speaker as i64).collect::<Vec<_>>(), [b]),
                dev,
            ),
            lang: Tensor::<1, Int>::from_data(
                TensorData::new(ex.iter().map(|e| e.locale as i64).collect::<Vec<_>>(), [b]),
                dev,
            ),
        }
    }
}
