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
    /// One transcript per clip (`0001.webm` + `0001.txt`), as in the Open
    /// Home Foundation voice datasets.
    Pairs {
        /// Directories of the clips; `*` globs one path component.
        dirs: Vec<String>,
        audio_ext: String,
        #[serde(default)]
        speaker: Speaker,
    },
    /// Segments of long recordings, listed in a TSV next to the spec
    /// (`id`, `url`, `start`, `end`, `text`; optionally gzip-compressed), such
    /// as the index written by `ihtts-data cml-index`. Each recording is
    /// downloaded once and its segments are cut from it.
    Segments {
        index: String,
        /// Substring replacements applied to the recording URLs, e.g.
        /// `["_64kb.mp3", ".mp3"]` to cut from LibriVox's 128 kbps originals
        /// instead of archive.org's band-limited derivatives.
        #[serde(default)]
        rewrite: Vec<(String, String)>,
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
    /// Reject source recordings whose median 99 % spectral rolloff is
    /// below this frequency (narrow-band, "telephone" audio).
    #[serde(default)]
    pub min_rolloff_hz: Option<f32>,
    /// Directory of the spec file (relative index paths).
    #[serde(skip)]
    pub dir: PathBuf,
}

impl CorpusSpec {
    pub fn load(path: &Path) -> Result<CorpusSpec, String> {
        let s = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut spec: CorpusSpec =
            serde_json::from_str(&s).map_err(|e| format!("{}: {e}", path.display()))?;
        spec.dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        Ok(spec)
    }
}

struct Item {
    id: String,
    audio: PathBuf,
    text: String,
    speaker: String,
    /// Start and end in seconds, for segments of a longer recording.
    span: Option<(f64, f64)>,
    /// Where to download the recording from, if it is not local yet.
    url: Option<String>,
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
                            span: None,
                            url: None,
                        });
                    }
                }
            }
        }
        Layout::Pairs {
            dirs,
            audio_ext,
            speaker,
        } => {
            let ext = audio_ext.trim_start_matches('.');
            for pat in dirs {
                let found = glob(root, pat);
                if found.is_empty() {
                    return Err(format!("{}: no directory matches {pat}", spec.name));
                }
                for d in found {
                    let mut clips: Vec<PathBuf> = fs::read_dir(&d)
                        .map_err(|e| format!("{}: {e}", d.display()))?
                        .filter_map(|e| e.ok().map(|e| e.path()))
                        .filter(|p| p.extension().is_some_and(|e| e == ext))
                        .collect();
                    clips.sort();
                    for audio in clips {
                        let Ok(text) = fs::read_to_string(audio.with_extension("txt")) else {
                            continue;
                        };
                        let id = audio
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        let spk = speaker_of(speaker, &id, &[], &audio);
                        items.push(Item {
                            id,
                            audio,
                            text: text.trim().to_string(),
                            speaker: spk,
                            span: None,
                            url: None,
                        });
                    }
                }
            }
        }
        Layout::Segments {
            index,
            rewrite,
            speaker,
        } => {
            let path = spec.dir.join(index);
            let text = crate::cml::read_text(&path)?;
            for line in text.lines() {
                let c: Vec<&str> = line.splitn(5, '\t').collect();
                if c.len() < 5 {
                    continue;
                }
                let (Ok(start), Ok(end)) = (c[2].parse::<f64>(), c[3].parse::<f64>()) else {
                    continue;
                };
                // archive.org serves everything over HTTPS.
                let mut url = c[1].replacen("http://", "https://", 1);
                for (from, to) in rewrite {
                    url = url.replace(from.as_str(), to);
                }
                // cache/<corpus>/rec/<item>/<file> keeps archive.org's names.
                let rel: Vec<&str> = url.rsplit('/').take(2).collect();
                let audio = root
                    .join("rec")
                    .join(rel.get(1).copied().unwrap_or("item"))
                    .join(rel[0]);
                let id = c[0].to_string();
                let spk = speaker_of(speaker, &id, &[], &audio);
                items.push(Item {
                    id,
                    audio,
                    text: c[4].trim().to_string(),
                    speaker: spk,
                    span: Some((start, end)),
                    url: Some(url),
                });
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
                    span: None,
                    url: None,
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
    /// Source recordings rejected by the bandwidth gate.
    pub rejected_recordings: usize,
}

/// Keeps whole recordings, densest first (most segment time per second of
/// recording), until `seconds` of segments are covered, so that a capped
/// corpus downloads as little audio as possible.
fn densest_recordings(items: Vec<Item>, seconds: f64) -> Vec<Item> {
    let mut rec: BTreeMap<PathBuf, (f64, f64)> = BTreeMap::new();
    for it in &items {
        if let Some((s, e)) = it.span {
            let r = rec.entry(it.audio.clone()).or_default();
            r.0 += e - s;
            r.1 = r.1.max(e);
        }
    }
    let mut order: Vec<(PathBuf, f64, f64)> =
        rec.into_iter().map(|(p, (c, span))| (p, c, span)).collect();
    order.sort_by(|a, b| (b.1 / b.2.max(1.0)).total_cmp(&(a.1 / a.2.max(1.0))));
    let mut keep = std::collections::BTreeSet::new();
    let mut acc = 0.0;
    for (p, c, _) in order {
        if acc >= seconds {
            break;
        }
        acc += c;
        keep.insert(p);
    }
    items
        .into_iter()
        .filter(|i| keep.contains(&i.audio))
        .collect()
}

/// Downloads the recordings that segment layouts cut from (a few at a time).
fn download_recordings(spec: &CorpusSpec, items: &[Item]) -> Result<(), String> {
    let mut todo: BTreeMap<&Path, &str> = BTreeMap::new();
    for it in items {
        if let Some(u) = &it.url {
            if !it.audio.exists() {
                todo.insert(it.audio.as_path(), u.as_str());
            }
        }
    }
    if todo.is_empty() {
        return Ok(());
    }
    eprintln!("{}: downloading {} recordings", spec.name, todo.len());
    let todo: Vec<(&Path, &str)> = todo.into_iter().collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build()
        .map_err(|e| e.to_string())?;
    let done = AtomicUsize::new(0);
    let errors: Vec<String> = pool.install(|| {
        todo.par_iter()
            .filter_map(|(path, url)| {
                let r = fetch::download(url, path).err();
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n % 25 == 0 {
                    eprintln!("{}: {n}/{} recordings", spec.name, todo.len());
                }
                r
            })
            .collect()
    });
    // A few missing recordings only cost their segments.
    for e in errors.iter().take(5) {
        eprintln!("{}: {e}", spec.name);
    }
    if errors.len() * 10 > todo.len() {
        return Err(format!(
            "{} of {} downloads failed",
            errors.len(),
            todo.len()
        ));
    }
    Ok(())
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
    if let (Some(h), true) = (spec.max_hours, items.iter().any(|i| i.span.is_some())) {
        items = densest_recordings(items, h as f64 * 3600.0 * 1.15);
    }
    if let Some(n) = limit {
        items.truncate(n);
    }
    download_recordings(spec, &items)?;
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
    // Every source file is decoded once, for all the utterances cut from it.
    let mut groups: BTreeMap<&Path, Vec<usize>> = BTreeMap::new();
    for (i, it) in items.iter().enumerate() {
        groups.entry(it.audio.as_path()).or_default().push(i);
    }
    let groups: Vec<(&Path, Vec<usize>)> = groups.into_iter().collect();
    let done = AtomicUsize::new(0);
    let rejected = AtomicUsize::new(0);
    let total = items.len();
    let results: Vec<Option<(Utterance, f32)>> = groups
        .par_iter()
        .flat_map_iter(|(path, idx)| {
            let source = audio::load(path).ok().filter(|a| {
                let ok = spec
                    .min_rolloff_hz
                    .is_none_or(|hz| audio::rolloff(&a.samples, a.rate, 0.99) >= hz);
                if !ok {
                    rejected.fetch_add(1, Ordering::Relaxed);
                }
                ok
            });
            let out: Vec<Option<(Utterance, f32)>> = idx
                .iter()
                .map(|&i| {
                    let (it, ps) = (&items[i], phonemes[i].as_ref()?);
                    let a = source.as_ref()?;
                    let clip = match it.span {
                        Some((s, e)) => {
                            let r = a.rate as f64;
                            let (s, e) =
                                ((s * r) as usize, ((e * r) as usize).min(a.samples.len()));
                            a.samples.get(s..e)?
                        }
                        None => &a.samples[..],
                    };
                    let x = audio::resample(clip, a.rate, SAMPLE_RATE);
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
            out
        })
        .collect();
    let mut rep = Report {
        rejected_recordings: rejected.into_inner(),
        ..Report::default()
    };
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
