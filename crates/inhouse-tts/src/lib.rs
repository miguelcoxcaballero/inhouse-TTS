//! # Inhouse TTS
//!
//! Local, real-time, multilingual text-to-speech with selectable regional
//! accents, written entirely in Rust. This crate ties the text front end
//! ([`inhouse_tts_text`]) to the neural engine ([`inhouse_tts_core`]).
//!
//! Spanish voices are Piper models run by our engine; English voices use
//! the K82 model.
//!
//! ```no_run
//! use inhouse_tts::{Engine, SpeakOptions};
//! let mut tts = Engine::without_model();
//! let voice = std::fs::read("models/piper/es_MX-ald-medium.ihw").unwrap();
//! tts.add_piper("es_MX-ald", &voice).unwrap();
//! let opts = SpeakOptions { locale: "es-MX".into(), voice: "es_MX-ald".into(), ..Default::default() };
//! for chunk in tts.stream("Hola, ¿qué tal? Esto es Inhouse TTS.", &opts).unwrap() {
//!     let chunk = chunk.unwrap();
//!     println!("{} samples for {}", chunk.audio.len(), chunk.phonemes);
//! }
//! ```

pub use inhouse_tts_core as core;
pub use inhouse_tts_text as text;

#[cfg(feature = "gpu")]
pub use inhouse_tts_core::gpu::GpuOptions;
#[cfg(feature = "gpu")]
use inhouse_tts_core::model::k82::gpu::K82Gpu;
use inhouse_tts_core::model::k82::{
    InferOptions, LoadOptions, K82, MAX_TOKENS, SAMPLES_PER_FRAME, SAMPLE_RATE as SR,
};
use inhouse_tts_core::model::piper::Piper;
use inhouse_tts_core::resample::Resampler;
use inhouse_tts_core::WeightFile;
use inhouse_tts_text::piper::{piper_lang, piper_sentences};
use inhouse_tts_text::{chunk_phonemes, ChunkPlan, Frontend, Locale};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

pub const SAMPLE_RATE: u32 = SR;
const STYLE: usize = 256;

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<inhouse_tts_core::weights::Error> for Error {
    fn from(e: inhouse_tts_core::weights::Error) -> Self {
        Error(e.0)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// A voice: either a K82 style pack (one 256-d style vector per utterance
/// length) or one speaker of a Piper model.
#[derive(Clone)]
pub struct Voice {
    pub name: String,
    kind: Kind,
}

#[derive(Clone)]
enum Kind {
    Style(Vec<f32>),
    Piper { model: Arc<Piper>, speaker: usize },
}

impl Voice {
    pub fn new(name: &str, pack: Vec<f32>) -> Result<Voice> {
        if pack.is_empty() || pack.len() % STYLE != 0 {
            return Err(Error(format!("voice {name}: bad pack size {}", pack.len())));
        }
        Ok(Voice {
            name: name.to_string(),
            kind: Kind::Style(pack),
        })
    }

    /// Style vector for an utterance of `tokens` phonemes (K82 voices only).
    pub fn style(&self, tokens: usize) -> Option<&[f32]> {
        let Kind::Style(pack) = &self.kind else {
            return None;
        };
        let rows = pack.len() / STYLE;
        let r = tokens.saturating_sub(1).min(rows - 1);
        Some(&pack[r * STYLE..(r + 1) * STYLE])
    }

    /// Language of a Piper voice (`es-ES`, `en-US`…); `None` for K82 voices,
    /// whose language is in the name ([`voice_info`]).
    pub fn language(&self) -> Option<String> {
        match &self.kind {
            Kind::Piper { model, .. } => Some(model.language().replace('_', "-")),
            Kind::Style(_) => None,
        }
    }

    /// Weighted mix of K82 voices (voice design / blending).
    pub fn blend(name: &str, parts: &[(&Voice, f32)]) -> Result<Voice> {
        let total: f32 = parts.iter().map(|(_, w)| *w).sum();
        if parts.is_empty() || total <= 0.0 {
            return Err(Error("empty voice blend".into()));
        }
        let packs: Vec<(&Vec<f32>, f32)> = parts
            .iter()
            .map(|(v, w)| match &v.kind {
                Kind::Style(p) => Ok((p, *w)),
                Kind::Piper { .. } => Err(Error(format!("voice {} cannot be blended", v.name))),
            })
            .collect::<Result<_>>()?;
        let len = packs.iter().map(|(p, _)| p.len()).min().unwrap();
        let mut pack = vec![0f32; len];
        for (p, w) in packs {
            for (o, x) in pack.iter_mut().zip(p) {
                *o += x * w / total;
            }
        }
        Voice::new(name, pack)
    }
}

/// Human-readable metadata derived from a Kokoro-style voice name.
pub struct VoiceInfo {
    pub name: String,
    pub language: &'static str,
    pub female: bool,
    pub label: String,
}

/// Piper voices shipped with Inhouse TTS: name prefix and whether the
/// speaker is a woman.
const PIPER_GENDER: &[(&str, bool)] = &[
    ("es_ES-davefx", false),
    ("es_ES-sharvard-M", false),
    ("es_ES-sharvard-F", true),
    ("es_MX-ald", false),
    ("es_AR-daniela", true),
];

pub fn voice_info(name: &str) -> VoiceInfo {
    // Piper voices: `es_ES-davefx`, `es_ES-sharvard-F`.
    if name.len() > 6 && name.as_bytes()[2] == b'_' && name.as_bytes()[5] == b'-' {
        let language = match &name[..5] {
            "es_ES" => "es-ES",
            "es_MX" => "es-MX",
            "es_AR" => "es-AR",
            "en_US" => "en-US",
            "en_GB" => "en-GB",
            _ => "",
        };
        let female = PIPER_GENDER
            .iter()
            .find(|(p, _)| name.starts_with(p))
            .is_some_and(|(_, f)| *f);
        let base = name[6..].split('-').next().unwrap_or("");
        let mut label: String = base
            .chars()
            .take(1)
            .flat_map(char::to_uppercase)
            .chain(base.chars().skip(1))
            .collect();
        label.push_str(if female { " (F)" } else { " (M)" });
        return VoiceInfo {
            name: name.to_string(),
            language,
            female,
            label,
        };
    }
    let mut c = name.chars();
    let lang = match c.next() {
        Some('a') => "en-US",
        Some('b') => "en-GB",
        Some('e') => "es",
        Some('f') => "fr-FR",
        Some('h') => "hi-IN",
        Some('i') => "it-IT",
        Some('p') => "pt-BR",
        Some('j') => "ja-JP",
        Some('z') => "zh-CN",
        _ => "",
    };
    let female = c.next() == Some('f');
    let base = name.split('_').nth(1).unwrap_or(name);
    let mut label: String = base
        .chars()
        .take(1)
        .flat_map(char::to_uppercase)
        .chain(base.chars().skip(1))
        .collect();
    label.push_str(if female { " (F)" } else { " (M)" });
    VoiceInfo {
        name: name.to_string(),
        language: lang,
        female,
        label,
    }
}

/// Synthesis options.
#[derive(Clone, Debug)]
pub struct SpeakOptions {
    /// Locale code (`es-ES`, `es-MX`, `es-AR`, `en-US`, `en-GB`, ...).
    pub locale: String,
    /// Voice name, a blend `a+b` (equal weights) or `0.7*a+0.3*b`.
    pub voice: String,
    /// Speaking rate (1.0 = normal).
    pub speed: f32,
    /// Excitation noise seed (deterministic output).
    pub seed: u64,
    pub plan: ChunkPlan,
    /// Trim the model's leading/trailing silence and insert punctuation pauses.
    pub trim_silence: bool,
}

impl Default for SpeakOptions {
    fn default() -> Self {
        SpeakOptions {
            locale: "en-US".into(),
            voice: "af_heart".into(),
            speed: 1.0,
            seed: 1,
            plan: ChunkPlan::default(),
            trim_silence: true,
        }
    }
}

/// One synthesised chunk of a stream.
pub struct Chunk {
    pub index: usize,
    pub phonemes: String,
    pub audio: Vec<f32>,
}

pub struct Engine {
    /// The K82 model, needed only by K82 voices.
    model: Option<K82>,
    voices: BTreeMap<String, Voice>,
    frontend: Frontend,
    /// Converters from each Piper sample rate to [`SAMPLE_RATE`].
    resamplers: Mutex<BTreeMap<u32, Arc<Resampler>>>,
    #[cfg(feature = "gpu")]
    gpu: Option<K82Gpu>,
}

impl Engine {
    /// Loads a `.ihw` K82 model.
    pub fn new(model: &[u8], opts: LoadOptions) -> Result<Engine> {
        let mut e = Engine::without_model();
        e.model = Some(K82::load(WeightFile::parse(model)?, opts)?);
        Ok(e)
    }

    /// An engine for Piper voices only (no K82 model).
    pub fn without_model() -> Engine {
        Engine {
            model: None,
            voices: BTreeMap::new(),
            frontend: Frontend::new(),
            resamplers: Mutex::new(BTreeMap::new()),
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Loads (or replaces) the K82 model, e.g. on first use of a K82 voice.
    pub fn load_model(&mut self, model: &[u8], opts: LoadOptions) -> Result<()> {
        self.model = Some(K82::load(WeightFile::parse(model)?, opts)?);
        #[cfg(feature = "gpu")]
        {
            self.gpu = None;
        }
        Ok(())
    }

    /// Adds a Piper voice (`.ihw` written by `piper::convert`) under `name`;
    /// multi-speaker models add one voice per speaker, `name-<speaker>`.
    /// Returns the voice names.
    pub fn add_piper(&mut self, name: &str, bytes: &[u8]) -> Result<Vec<String>> {
        let prepack = !cfg!(target_arch = "wasm32");
        let model = Arc::new(Piper::load(WeightFile::parse(bytes)?, prepack)?);
        let speakers: Vec<(String, usize)> = if model.speakers().is_empty() {
            vec![(name.to_string(), 0)]
        } else {
            model
                .speakers()
                .iter()
                .map(|(s, &i)| (format!("{name}-{s}"), i))
                .collect()
        };
        for (n, speaker) in &speakers {
            self.voices.insert(
                n.clone(),
                Voice {
                    name: n.clone(),
                    kind: Kind::Piper {
                        model: model.clone(),
                        speaker: *speaker,
                    },
                },
            );
        }
        Ok(speakers.into_iter().map(|(n, _)| n).collect())
    }

    /// Moves the decoder and vocoder to the GPU and returns the adapter
    /// name. The model must be loaded with `LoadOptions { gpu: true, .. }`.
    #[cfg(feature = "gpu")]
    pub async fn enable_gpu(&mut self, opts: GpuOptions) -> Result<String> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error("no K82 model loaded".into()))?;
        let gpu = inhouse_tts_core::gpu::Gpu::new(opts).await.map_err(Error)?;
        let g = K82Gpu::new(gpu, model).map_err(Error)?;
        let name = g.name().to_string();
        self.gpu = Some(g);
        Ok(name)
    }

    /// Goes back to CPU-only synthesis.
    #[cfg(feature = "gpu")]
    pub fn disable_gpu(&mut self) {
        self.gpu = None;
    }

    /// Name of the GPU in use, if any.
    pub fn gpu_name(&self) -> Option<&str> {
        #[cfg(feature = "gpu")]
        if let Some(g) = &self.gpu {
            return Some(g.name());
        }
        None
    }

    /// Adds every voice of a `.ihw` voice container; returns how many.
    pub fn add_voices(&mut self, bytes: &[u8]) -> Result<usize> {
        let f = WeightFile::parse(bytes)?;
        let names: Vec<String> = f.names().map(str::to_string).collect();
        for n in &names {
            let pack = f.get(n)?.to_f32();
            self.voices.insert(n.clone(), Voice::new(n, pack)?);
        }
        Ok(names.len())
    }

    pub fn add_voice(&mut self, voice: Voice) {
        self.voices.insert(voice.name.clone(), voice);
    }

    pub fn voices(&self) -> impl Iterator<Item = &str> {
        self.voices.keys().map(String::as_str)
    }

    pub fn frontend(&mut self) -> &mut Frontend {
        &mut self.frontend
    }

    /// True if every voice named in a voice spec (`a`, `a+b`, `0.7*a+0.3*b`) is loaded.
    pub fn has_voices(&self, spec: &str) -> bool {
        Self::voice_names(spec)
            .iter()
            .all(|n| self.voices.contains_key(n.as_str()))
    }

    /// Voice names used by a voice spec.
    pub fn voice_names(spec: &str) -> Vec<String> {
        spec.split(['+', ','])
            .map(|t| t.split_once('*').map_or(t, |(_, n)| n).trim().to_string())
            .filter(|n| !n.is_empty())
            .collect()
    }

    /// Resolves a voice spec into a (possibly blended) voice.
    pub fn resolve_voice(&self, spec: &str) -> Result<Voice> {
        if let Some(v) = self.voices.get(spec) {
            return Ok(v.clone());
        }
        let mut parts = Vec::new();
        for term in spec.split(['+', ',']) {
            let term = term.trim();
            let (w, name) = match term.split_once('*') {
                Some((w, n)) => (
                    w.trim()
                        .parse::<f32>()
                        .map_err(|_| Error(format!("bad weight in {term}")))?,
                    n.trim(),
                ),
                None => (1.0, term),
            };
            let v = self
                .voices
                .get(name)
                .ok_or_else(|| Error(format!("unknown voice {name}")))?;
            parts.push((v, w));
        }
        Voice::blend(spec, &parts)
    }

    fn locale(code: &str) -> Result<&'static Locale> {
        inhouse_tts_text::find_locale(code)
            .ok_or_else(|| Error(format!("unsupported locale {code}")))
    }

    /// Text → model phonemes.
    pub fn phonemize(&mut self, text: &str, locale: &str) -> Result<String> {
        let loc = Self::locale(locale)?;
        self.frontend.phonemize(text, loc).map_err(Error)
    }

    /// Text → phoneme chunks in streaming order.
    pub fn plan(&mut self, text: &str, opts: &SpeakOptions) -> Result<Vec<String>> {
        let ps = self.phonemize(text, &opts.locale)?;
        Ok(chunk_phonemes(&ps, opts.plan))
    }

    fn ids(model: &K82, phonemes: &str) -> Vec<u32> {
        let ids = model.tokenize(phonemes);
        if ids.len() > MAX_TOKENS + 2 {
            [&ids[..MAX_TOKENS + 1], &[0]].concat()
        } else {
            ids
        }
    }

    /// Synthesises one phoneme chunk (≤ 510 symbols). Natively the decoder
    /// runs on the GPU when [`Engine::enable_gpu`] succeeded.
    pub fn synth_phonemes(&self, phonemes: &str, voice: &Voice, opts: &SpeakOptions) -> Vec<f32> {
        if let Kind::Piper { model, speaker } = &voice.kind {
            return self.synth_piper(phonemes, model, *speaker, opts);
        }
        let Some(model) = &self.model else {
            eprintln!("voice {} needs the K82 model", voice.name);
            return Vec::new();
        };
        let ids = Self::ids(model, phonemes);
        let style = voice.style(ids.len() - 2).unwrap_or_default();
        let inp = model.prepare(&ids, style, infer_options(opts));
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        if let Some(g) = &self.gpu {
            match inhouse_tts_core::gpu::block_on(g.decode(model, &inp, style)) {
                Ok(audio) => return finish(audio, &inp.durations, phonemes, opts),
                Err(e) => eprintln!("GPU decoder failed ({e}); using the CPU"),
            }
        }
        let audio = model.decode(&inp, style);
        finish(audio, &inp.durations, phonemes, opts)
    }

    /// Piper synthesis: our phonemes are converted to the espeak-style IPA
    /// the voice was trained on, synthesised sentence by sentence, trimmed
    /// and resampled to [`SAMPLE_RATE`].
    fn synth_piper(
        &self,
        phonemes: &str,
        model: &Piper,
        speaker: usize,
        opts: &SpeakOptions,
    ) -> Vec<f32> {
        let lang = piper_lang(&model.language().replace('_', "-"))
            .or_else(|| piper_lang(&opts.locale))
            .unwrap_or(inhouse_tts_text::piper::PiperLang::Es);
        let hop = model.hop();
        let rate = model.sample_rate as usize;
        let mut po = model.defaults();
        po.length_scale /= opts.speed.clamp(0.3, 3.0);
        let mut out = Vec::new();
        for (i, s) in piper_sentences(phonemes, lang).iter().enumerate() {
            po.seed = opts.seed.wrapping_add(i as u64);
            let ids = model.encode(s);
            let o = match model.synthesize(&ids, speaker, po) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("piper: {}", e.0);
                    continue;
                }
            };
            let d = &o.durations;
            let (mut a, mut b) = (0, o.audio.len());
            if opts.trim_silence && d.len() >= 4 {
                // ^ and the first pad lead, the last pad and $ trail.
                a = ((d[0] + d[1]) * hop).saturating_sub(rate / 20);
                let tail = (d[d.len() - 1] + d[d.len() - 2]) * hop;
                b = o
                    .audio
                    .len()
                    .saturating_sub(tail.saturating_sub(rate / 10))
                    .max(a);
            }
            out.extend_from_slice(&o.audio[a..b]);
            if opts.trim_silence {
                let ms = if s.ends_with(['.', '!', '?']) {
                    180
                } else {
                    60
                };
                out.extend(std::iter::repeat_n(0.0, rate * ms / 1000));
            }
        }
        if model.sample_rate == SAMPLE_RATE {
            return out;
        }
        let r = {
            let mut cache = self.resamplers.lock().unwrap_or_else(|e| e.into_inner());
            cache
                .entry(model.sample_rate)
                .or_insert_with(|| Arc::new(Resampler::new(model.sample_rate, SAMPLE_RATE)))
                .clone()
        };
        r.process(&out)
    }

    /// Like [`Engine::synth_phonemes`], but awaits the GPU instead of
    /// blocking (required in the browser). Fails if the GPU fails, so the
    /// caller can fall back with [`Engine::disable_gpu`].
    #[cfg(feature = "gpu")]
    pub async fn synth_phonemes_async(
        &self,
        phonemes: &str,
        voice: &Voice,
        opts: &SpeakOptions,
    ) -> Result<Vec<f32>> {
        let Some(model) = &self.model else {
            return Ok(self.synth_phonemes(phonemes, voice, opts));
        };
        if voice.style(0).is_none() {
            return Ok(self.synth_phonemes(phonemes, voice, opts));
        }
        let ids = Self::ids(model, phonemes);
        let style = voice.style(ids.len() - 2).unwrap_or_default();
        let inp = model.prepare(&ids, style, infer_options(opts));
        let audio = match &self.gpu {
            Some(g) => g.decode(model, &inp, style).await.map_err(Error)?,
            None => model.decode(&inp, style),
        };
        Ok(finish(audio, &inp.durations, phonemes, opts))
    }

    /// Synthesises a whole text.
    pub fn speak(&mut self, text: &str, opts: &SpeakOptions) -> Result<Vec<f32>> {
        let mut audio = Vec::new();
        for c in self.stream(text, opts)? {
            audio.extend(c?.audio);
        }
        Ok(audio)
    }

    /// Streams a text chunk by chunk (sentence/clause level).
    pub fn stream(
        &mut self,
        text: &str,
        opts: &SpeakOptions,
    ) -> Result<impl Iterator<Item = Result<Chunk>> + '_> {
        let chunks = self.plan(text, opts)?;
        let voice = self.resolve_voice(&opts.voice)?;
        let opts = opts.clone();
        Ok(chunks
            .into_iter()
            .enumerate()
            .map(move |(index, phonemes)| {
                let audio = self.synth_phonemes(&phonemes, &voice, &opts);
                Ok(Chunk {
                    index,
                    phonemes,
                    audio,
                })
            }))
    }
}

fn infer_options(opts: &SpeakOptions) -> InferOptions {
    InferOptions {
        speed: opts.speed,
        seed: Some(opts.seed),
    }
}

/// Trims the model's leading/trailing silence (keeping 50 ms / 100 ms) and
/// appends a punctuation pause.
fn finish(audio: Vec<f32>, durations: &[usize], phonemes: &str, opts: &SpeakOptions) -> Vec<f32> {
    if !opts.trim_silence || durations.len() < 3 {
        return audio;
    }
    let lead = (durations[0] * SAMPLES_PER_FRAME).saturating_sub(1200);
    let tail = (durations[durations.len() - 1] * SAMPLES_PER_FRAME).saturating_sub(2400);
    let end = audio.len().saturating_sub(tail).max(lead);
    let mut out = audio[lead..end].to_vec();
    let pause_ms = if phonemes.trim_end().ends_with(['.', '!', '?', '…']) {
        180
    } else {
        60
    };
    out.extend(std::iter::repeat_n(
        0.0,
        SAMPLE_RATE as usize * pause_ms / 1000,
    ));
    out
}
