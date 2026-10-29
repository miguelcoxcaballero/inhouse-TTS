//! Training data: manifests → symbol, speaker and locale tables, cached
//! spectrograms and padded batches.

use crate::config::Config;
use crate::dsp;
use crate::model::Batch;
use burn::tensor::{Device, Int, Tensor, TensorData};
use inhouse_tts_data::Utterance;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
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

/// One example in memory.
struct Example {
    ids: Vec<i64>,
    wav: Vec<f32>,
    spec: Vec<f32>,
    frames: usize,
    speaker: usize,
    locale: usize,
}

pub struct Dataset {
    examples: Vec<Example>,
    bins: usize,
    hop: usize,
}

pub const SR: u32 = 24_000;

fn read_wav(path: &Path) -> Result<Vec<f32>, String> {
    let a = inhouse_tts_data::audio::load(path)?;
    if a.rate != SR {
        return Ok(inhouse_tts_data::audio::resample(&a.samples, a.rate, SR));
    }
    Ok(a.samples)
}

impl Dataset {
    /// Loads every utterance of the manifests (spectrograms are cached as
    /// `<wav>.spec` files next to the audio).
    pub fn load(
        manifests: &[PathBuf],
        cfg: &Config,
        tables: &Tables,
        max_items: usize,
    ) -> Result<Dataset, String> {
        let (n_fft, hop) = (cfg.audio.n_fft, cfg.audio.hop);
        let bins = n_fft / 2 + 1;
        let mut items: Vec<(PathBuf, Utterance)> = Vec::new();
        for m in manifests {
            let dir = m.parent().unwrap_or(Path::new(".")).to_path_buf();
            for u in inhouse_tts_data::read_manifest(m)? {
                items.push((dir.join(&u.audio), u));
            }
        }
        items.truncate(max_items);
        let max_frames = cfg.train.max_frames;
        let examples: Vec<Example> = items
            .par_iter()
            .filter_map(|(path, u)| {
                let mut wav = read_wav(path).ok()?;
                let frames = wav.len() / hop;
                if frames < cfg.train.segment_frames + 2 || frames > max_frames {
                    return None;
                }
                wav.truncate(frames * hop);
                let cache = path.with_extension("spec");
                let spec = match fs::read(&cache) {
                    Ok(b) if b.len() == bins * frames * 4 => b
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                    _ => {
                        let s = dsp::spectrogram(&wav, n_fft, hop);
                        let bytes: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
                        let _ = fs::write(&cache, bytes);
                        s
                    }
                };
                Some(Example {
                    ids: tables.encode(&u.phonemes),
                    wav,
                    spec,
                    frames,
                    speaker: tables.speaker(&u.speaker),
                    locale: tables.locale(&u.locale),
                })
            })
            .collect();
        if examples.is_empty() {
            return Err("no usable utterances".into());
        }
        Ok(Dataset {
            examples,
            bins,
            hop,
        })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn hours(&self) -> f64 {
        self.examples.iter().map(|e| e.wav.len()).sum::<usize>() as f64 / SR as f64 / 3600.0
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
        let mut ids = vec![0i64; b * tx];
        let mut id_mask = vec![0f32; b * tx];
        let mut spec = vec![0f32; b * self.bins * ty];
        let mut spec_mask = vec![0f32; b * ty];
        let mut wav = vec![0f32; b * ty * self.hop];
        for (i, e) in ex.iter().enumerate() {
            ids[i * tx..i * tx + e.ids.len()].copy_from_slice(&e.ids);
            id_mask[i * tx..i * tx + e.ids.len()].fill(1.0);
            for k in 0..self.bins {
                let dst = &mut spec[(i * self.bins + k) * ty..(i * self.bins + k) * ty + e.frames];
                dst.copy_from_slice(&e.spec[k * e.frames..(k + 1) * e.frames]);
            }
            spec_mask[i * ty..i * ty + e.frames].fill(1.0);
            wav[i * ty * self.hop..i * ty * self.hop + e.wav.len()].copy_from_slice(&e.wav);
        }
        Batch {
            ids: Tensor::<2, Int>::from_data(TensorData::new(ids, [b, tx]), dev),
            id_mask: Tensor::from_data(TensorData::new(id_mask, [b, 1, tx]), dev),
            id_lens: ex.iter().map(|e| e.ids.len()).collect(),
            spec: Tensor::from_data(TensorData::new(spec, [b, self.bins, ty]), dev),
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
