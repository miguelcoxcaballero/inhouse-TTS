//! Corpus descriptions (`corpora/*.json`), training manifests and the
//! preparation pipeline.
//!
//! A corpus spec names the archives to download, the licence, the locale
//! (which selects the accent rules of the phonemiser) and the layout of its
//! transcript index. Preparation writes 24 kHz mono 16-bit WAV files and a
//! `manifest.jsonl` with one [`Utterance`] per line.

use crate::{audio, fetch};
use inhouse_tts_text::{find_locale, Frontend};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub const SAMPLE_RATE: u32 = 24_000;

/// One training utterance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Utterance {
    pub id: String,
    /// WAV path relative to the manifest directory.
    pub audio: String,
    pub text: String,
    pub phonemes: String,
    pub locale: String,
    /// Speaker id, unique across corpora (`corpus:speaker`).
    pub speaker: String,
    pub corpus: String,
    pub seconds: f32,
}

/// How speaker ids are derived.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Speaker {
    /// Single-speaker corpus.
    #[default]
    Single,
    /// From an index column.
    Column(usize),
    /// The first `n` `_`-separated fields of the utterance id
    /// (e.g. `arf_00295_0001` → `arf_00295` for n = 2).
    IdPrefix(usize),
    /// The parent directory of the audio file.
    Directory,
}

/// Layout of the transcript index.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Layout {
    /// Delimited text index (`id<TAB>text`, `file|text|…`, CSV…).
    Index {
        /// Index file(s), relative to the extraction directory; `*` globs one
        /// path component.
        files: Vec<String>,
        delimiter: char,
        #[serde(default)]
        header: bool,
        audio_col: usize,
        text_col: usize,
        /// Directory of the audio files relative to the index file.
        #[serde(default)]
        audio_dir: String,
        /// Extension appended to the audio column when it has none.
        #[serde(default)]
        audio_ext: String,
        #[serde(default)]
        speaker: Speaker,
    },
    /// Mozilla Common Voice release (`validated.tsv` + `clips/`), filtered by
    /// the self-reported accent field.
    CommonVoice {
        dir: String,
        /// Substrings of the `accents` column to keep (case-insensitive).
        accents: Vec<String>,
        #[serde(default)]
        min_up_votes: u32,
    },
}

#[derive(Clone, Debug, Deserialize)]
pub struct CorpusSpec {
    pub name: String,
    pub locale: String,
    pub license: String,
    #[serde(default)]
    pub source: String,
    /// Archives to download and extract.
    #[serde(default)]
    pub urls: Vec<String>,
    pub layout: Layout,
    /// Keep at most this many hours (longest speakers first).
    #[serde(default)]
    pub max_hours: Option<f32>,
    /// Only these speakers (after id derivation), if given.
    #[serde(default)]
    pub speakers: Vec<String>,
}

impl CorpusSpec {
    pub fn load(path: &Path) -> Result<CorpusSpec, String> {
        let s = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&s).map_err(|e| format!("{}: {e}", path.display()))
    }
}

struct Item {
    id: String,
    audio: PathBuf,
    text: String,
    speaker: String,
}

/// Expands `a/*/b.tsv` patterns (one `*` per component).
fn glob(base: &Path, pattern: &str) -> Vec<PathBuf> {
    let mut paths = vec![base.to_path_buf()];
    for comp in pattern.split('/') {
        let mut next = Vec::new();
        for p in &paths {
            if comp.contains('*') {
                let (pre, post) = comp.split_once('*').unwrap();
                if let Ok(rd) = fs::read_dir(p) {
                    let mut v: Vec<PathBuf> = rd
                        .filter_map(|e| e.ok().map(|e| e.path()))
                        .filter(|q| {
                            let n = q
                                .file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_default();
                            n.starts_with(pre) && n.ends_with(post)
                        })
                        .collect();
                    v.sort();
                    next.extend(v);
                }
            } else {
                next.push(p.join(comp));
            }
        }
        paths = next;
    }
    paths.into_iter().filter(|p| p.exists()).collect()
}

fn speaker_of(rule: &Speaker, id: &str, cols: &[&str], audio: &Path) -> String {
    match rule {
        Speaker::Single => "0".into(),
        Speaker::Column(c) => cols
            .get(*c)
            .map(|s| s.trim().to_string())
            .unwrap_or_default(),
        Speaker::IdPrefix(n) => id.split('_').take(*n).collect::<Vec<_>>().join("_"),
        Speaker::Directory => audio
            .parent()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
    }
}

fn list(spec: &CorpusSpec, root: &Path) -> Result<Vec<Item>, String> {
    let mut items = Vec::new();
    match &spec.layout {
        Layout::Index {
            files,
            delimiter,
            header,
            audio_col,
            text_col,
            audio_dir,
            audio_ext,
            speaker,
        } => {
            for pat in files {
                let found = glob(root, pat);
                if found.is_empty() {
                    return Err(format!("{}: no index matches {pat}", spec.name));
                }
                for idx in found {
                    let base = idx.parent().unwrap_or(root).join(audio_dir);
                    let text =
                        fs::read_to_string(&idx).map_err(|e| format!("{}: {e}", idx.display()))?;
                    for (n, line) in text.lines().enumerate() {
                        if (*header && n == 0) || line.trim().is_empty() {
                            continue;
                        }
                        let cols: Vec<&str> = line.split(*delimiter).collect();
                        let (Some(a), Some(t)) = (cols.get(*audio_col), cols.get(*text_col)) else {
                            continue;
                        };
                        let a = a.trim();
                        let mut audio = base.join(a);
                        if !audio_ext.is_empty() && audio.extension().is_none() {
                            audio.set_extension(audio_ext.trim_start_matches('.'));
                        }
                        let id = Path::new(a)
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| a.to_string());
                        let spk = speaker_of(speaker, &id, &cols, &audio);
                        items.push(Item {
                            id,
                            audio,
                            text: t.trim().to_string(),
                            speaker: spk,
                        });
                    }
                }
            }
        }
        Layout::CommonVoice {
            dir,
            accents,
            min_up_votes,
        } => {
            let d = root.join(dir);
            let tsv = fs::read_to_string(d.join("validated.tsv")).map_err(|e| e.to_string())?;
            let mut lines = tsv.lines();
            let head: Vec<&str> = lines.next().unwrap_or_default().split('\t').collect();
            let col = |n: &str| head.iter().position(|h| *h == n);
            let (Some(c_client), Some(c_path), Some(c_sent)) =
                (col("client_id"), col("path"), col("sentence"))
            else {
                return Err("validated.tsv: missing columns".into());
            };
            let c_acc = col("accents");
            let c_up = col("up_votes");
            let wanted: Vec<String> = accents.iter().map(|a| a.to_lowercase()).collect();
            for line in lines {
                let cols: Vec<&str> = line.split('\t').collect();
                let acc = c_acc
                    .and_then(|c| cols.get(c))
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                if !wanted.is_empty() && !wanted.iter().any(|w| acc.contains(w.as_str())) {
                    continue;
                }
                let up: u32 = c_up
                    .and_then(|c| cols.get(c))
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                if up < *min_up_votes {
                    continue;
                }
                let path = cols.get(c_path).copied().unwrap_or_default();
                items.push(Item {
                    id: Path::new(path)
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    audio: d.join("clips").join(path),
                    text: cols.get(c_sent).copied().unwrap_or_default().to_string(),
                    speaker: cols
                        .get(c_client)
                        .map(|s| s.chars().take(12).collect())
                        .unwrap_or_default(),
                });
            }
        }
    }
    if !spec.speakers.is_empty() {
        items.retain(|i| spec.speakers.contains(&i.speaker));
    }
    Ok(items)
}

/// Statistics of a prepared corpus.
#[derive(Debug, Default, Serialize)]
pub struct Report {
    pub utterances: usize,
    pub hours: f64,
    pub skipped: usize,
    pub speakers: BTreeMap<String, f64>,
    /// Mean share of energy above 8 kHz (detects band-limited audio).
    pub hf_energy: f64,
}

/// Downloads, extracts and converts a corpus into `out/<name>/`.
pub fn prepare(
    spec: &CorpusSpec,
    cache: &Path,
    out: &Path,
    limit: Option<usize>,
) -> Result<Report, String> {
    let raw = cache.join(&spec.name);
    for url in &spec.urls {
        let file = url
            .rsplit('/')
            .next()
            .unwrap_or("download")
            .split('?')
            .next()
            .unwrap_or("download");
        if [".zip", ".tar.gz", ".tgz", ".tar"]
            .iter()
            .any(|e| file.ends_with(e))
        {
            let archive = cache.join("archives").join(file);
            eprintln!("{}: downloading {url}", spec.name);
            fetch::download(url, &archive)?;
            eprintln!("{}: extracting {file}", spec.name);
            fetch::extract(&archive, &raw.join(file.split('.').next().unwrap_or(file)))?;
        } else {
            // Plain files (transcript indexes) go next to the extracted audio.
            fetch::download(url, &raw.join(file))?;
        }
    }
    let mut items = list(spec, &raw)?;
    if let Some(n) = limit {
        items.truncate(n);
    }
    let loc = find_locale(&spec.locale).ok_or_else(|| format!("unknown locale {}", spec.locale))?;
    let mut fe = Frontend::new();
    let phonemes: Vec<Option<String>> = items
        .iter()
        .map(|i| {
            fe.phonemize(&i.text, loc)
                .ok()
                .filter(|p| !p.trim().is_empty())
        })
        .collect();
    let dir = out.join(&spec.name);
    fs::create_dir_all(dir.join("wavs")).map_err(|e| e.to_string())?;
    let done = AtomicUsize::new(0);
    let total = items.len();
    let results: Vec<Option<(Utterance, f32)>> = items
        .par_iter()
        .zip(phonemes.par_iter())
        .map(|(it, ps)| {
            let ps = ps.as_ref()?;
            let a = audio::load(&it.audio).ok()?;
            let x = audio::resample(&a.samples, a.rate, SAMPLE_RATE);
            let mut x = audio::trim(&x, SAMPLE_RATE, 120).to_vec();
            let secs = x.len() as f32 / SAMPLE_RATE as f32;
            if !(0.6..=25.0).contains(&secs) {
                return None;
            }
            audio::normalize(&mut x, SAMPLE_RATE, -20.0);
            let hf = audio::energy_above(&x, SAMPLE_RATE, 8000.0);
            let rel = format!("wavs/{}.wav", it.id);
            audio::write_wav(&dir.join(&rel), &x, SAMPLE_RATE).ok()?;
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 500 == 0 {
                eprintln!("{}: {n}/{total}", spec.name);
            }
            Some((
                Utterance {
                    id: format!("{}:{}", spec.name, it.id),
                    audio: rel,
                    text: it.text.clone(),
                    phonemes: ps.clone(),
                    locale: loc.code.to_string(),
                    speaker: format!("{}:{}", spec.name, it.speaker),
                    corpus: spec.name.clone(),
                    seconds: secs,
                },
                hf,
            ))
        })
        .collect();
    let mut rep = Report::default();
    let mut kept: Vec<(Utterance, f32)> = Vec::new();
    for r in results {
        match r {
            Some(u) => kept.push(u),
            None => rep.skipped += 1,
        }
    }
    if let Some(h) = spec.max_hours {
        // Prefer the speakers with the most material.
        let mut per: BTreeMap<String, f64> = BTreeMap::new();
        for (u, _) in &kept {
            *per.entry(u.speaker.clone()).or_default() += u.seconds as f64;
        }
        kept.sort_by(|a, b| {
            per[&b.0.speaker]
                .total_cmp(&per[&a.0.speaker])
                .then(a.0.id.cmp(&b.0.id))
        });
        let mut acc = 0.0;
        kept.retain(|(u, _)| {
            acc += u.seconds as f64;
            acc <= h as f64 * 3600.0
        });
    }
    let mut f = fs::File::create(dir.join("manifest.jsonl")).map_err(|e| e.to_string())?;
    for (u, hf) in &kept {
        writeln!(f, "{}", serde_json::to_string(u).unwrap()).map_err(|e| e.to_string())?;
        rep.utterances += 1;
        rep.hours += u.seconds as f64 / 3600.0;
        rep.hf_energy += *hf as f64;
        *rep.speakers.entry(u.speaker.clone()).or_default() += u.seconds as f64 / 3600.0;
    }
    if rep.utterances > 0 {
        rep.hf_energy /= rep.utterances as f64;
    }
    fs::write(
        dir.join("report.json"),
        serde_json::to_string_pretty(&rep).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    Ok(rep)
}

/// Reads a manifest written by [`prepare`].
pub fn read_manifest(path: &Path) -> Result<Vec<Utterance>, String> {
    let s = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| format!("{}: {e}", path.display())))
        .collect()
}
