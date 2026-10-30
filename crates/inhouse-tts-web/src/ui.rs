//! Main-thread application: DOM wiring, text front end, chunk scheduling
//! over synthesis workers and gapless streaming playback (Web Audio).

use crate::i18n::{self, Strings};
use crate::js::{err, fetch_bytes, fetch_fresh, get, get_f64, get_str, now, obj, prune_caches};
use inhouse_tts::text::{chunk_phonemes, find_locale, ChunkPlan, Frontend, Lang, LOCALES};
use inhouse_tts::voice_info;
use js_sys::{Array, Float32Array, Uint8Array};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet, VecDeque};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    AudioBufferSourceNode, AudioContext, AudioScheduledSourceNode, Blob, BlobPropertyBag, Document,
    Element, HtmlAnchorElement, HtmlButtonElement, HtmlElement, HtmlInputElement,
    HtmlOptGroupElement, HtmlOptionElement, HtmlSelectElement, HtmlTextAreaElement, MessageEvent,
    Url, Worker, WorkerOptions, WorkerType,
};

const SAMPLE_RATE: f32 = 24_000.0;
const CACHE_PREFIX: &str = "inhouse-tts-";

struct Slot {
    worker: Worker,
    ready: bool,
    busy: bool,
}

/// What the first worker found out about WebGPU.
#[derive(Clone, PartialEq)]
enum Gpu {
    /// Still probing (only worker 0 runs meanwhile).
    Unknown,
    /// Decoder on this adapter: a single worker is enough.
    On(String),
    /// No usable GPU: one CPU worker per spare core.
    Off,
}

struct Ui {
    text: HtmlTextAreaElement,
    locale: HtmlSelectElement,
    voice: HtmlSelectElement,
    speed: HtmlInputElement,
    speed_val: Element,
    threads: HtmlSelectElement,
    accel: HtmlSelectElement,
    speak: HtmlButtonElement,
    stop: HtmlButtonElement,
    download: HtmlButtonElement,
    status: Element,
    bar: HtmlElement,
    bar_wrap: HtmlElement,
    phonemes: Element,
    metrics: Element,
}

struct App {
    s: &'static Strings,
    ui: Ui,
    manifest: JsValue,
    cache: String,
    slots: Vec<Slot>,
    gpu: Gpu,
    frontend: Frontend,
    lexicons: HashSet<String>,
    last_sample: String,
    // Current utterance.
    job: f64,
    queue: VecDeque<(usize, String)>,
    total: usize,
    results: BTreeMap<usize, Vec<f32>>,
    next_play: usize,
    received: usize,
    voice: String,
    speed: f64,
    ctx: Option<AudioContext>,
    next_time: f64,
    sources: Vec<AudioBufferSourceNode>,
    all_audio: Vec<f32>,
    t_start: f64,
    t_first: Option<f64>,
    /// Models (Piper files, "k82") already in the cache and loaded by
    /// worker 0. Until then only worker 0 gets jobs that need them, so the
    /// workers don't all download the same model at once.
    loaded: HashSet<String>,
}

thread_local! {
    static APP: RefCell<Option<App>> = const { RefCell::new(None) };
}

fn with_app<R>(f: impl FnOnce(&mut App) -> R) -> Option<R> {
    APP.with(|a| a.borrow_mut().as_mut().map(f))
}

fn doc() -> Document {
    web_sys::window().unwrap().document().unwrap()
}

fn el<T: JsCast>(id: &str) -> Result<T, JsValue> {
    doc()
        .get_element_by_id(id)
        .ok_or_else(|| err(format!("missing #{id}")))?
        .dyn_into::<T>()
        .map_err(|_| err(format!("#{id} has wrong type")))
}

fn on(target: &web_sys::EventTarget, event: &str, f: impl FnMut(web_sys::Event) + 'static) {
    let c = Closure::<dyn FnMut(web_sys::Event)>::new(f);
    let _ = target.add_event_listener_with_callback(event, c.as_ref().unchecked_ref());
    c.forget();
}

fn set_text(e: &Element, t: &str) {
    e.set_text_content(Some(t));
}

pub fn start() -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| err("no window"))?;
    let nav_lang = window.navigator().language().unwrap_or_default();
    let s = i18n::pick(&nav_lang);
    let ui = Ui {
        text: el("text")?,
        locale: el("locale")?,
        voice: el("voice")?,
        speed: el("speed")?,
        speed_val: el("speed-val")?,
        threads: el("threads")?,
        accel: el("accel")?,
        speak: el("speak")?,
        stop: el("stop")?,
        download: el("download")?,
        status: el("status")?,
        bar: el("bar")?,
        bar_wrap: el("bar-wrap")?,
        phonemes: el("phonemes")?,
        metrics: el("metrics")?,
    };
    // Static labels.
    for (id, txt) in [
        ("subtitle", s.subtitle),
        ("l-text", s.text),
        ("l-locale", s.locale),
        ("l-voice", s.voice),
        ("l-speed", s.speed),
        ("l-threads", s.threads),
        ("l-accel", s.accel),
        ("l-phonemes", s.phonemes),
        ("offline", s.offline),
        ("about", s.about),
    ] {
        if let Some(e) = doc().get_element_by_id(id) {
            set_text(&e, txt);
        }
    }
    set_text(&ui.speak, s.speak);
    set_text(&ui.stop, s.stop);
    set_text(&ui.download, s.download);
    ui.speak.set_disabled(true);
    ui.download.set_disabled(true);

    // Locale menu grouped by language.
    for (label, lang) in [("Español", Lang::Es), ("English", Lang::En)] {
        let group: HtmlOptGroupElement = doc().create_element("optgroup")?.dyn_into()?;
        group.set_label(label);
        for l in LOCALES.iter().filter(|l| l.lang == lang) {
            let o = HtmlOptionElement::new_with_text_and_value(l.name, l.code)?;
            group.append_child(&o)?;
        }
        ui.locale.append_child(&group)?;
    }
    let initial = initial_locale(&nav_lang);
    ui.locale.set_value(initial);
    // One worker per spare core (each holds its own copy of the int8 model).
    let threads = window.navigator().hardware_concurrency() as usize;
    let default_threads = threads.saturating_sub(1).clamp(1, 4);
    for n in 1..=4usize {
        let o = HtmlOptionElement::new_with_text_and_value(&n.to_string(), &n.to_string())?;
        ui.threads.append_child(&o)?;
    }
    ui.threads.set_value(&default_threads.to_string());
    for (value, label) in [
        ("auto", s.accel_auto),
        ("gpu", s.accel_gpu),
        ("cpu", s.accel_cpu),
    ] {
        let o = HtmlOptionElement::new_with_text_and_value(label, value)?;
        ui.accel.append_child(&o)?;
    }
    let search = window.location().search().unwrap_or_default();
    let accel = ["gpu", "cpu"]
        .into_iter()
        .find(|m| search.contains(&format!("accel={m}")))
        .unwrap_or("auto");
    ui.accel.set_value(accel);

    let app = App {
        s,
        ui,
        manifest: JsValue::UNDEFINED,
        cache: String::new(),
        slots: Vec::new(),
        gpu: Gpu::Unknown,
        frontend: Frontend::new(),
        lexicons: HashSet::new(),
        last_sample: String::new(),
        job: 0.0,
        queue: VecDeque::new(),
        total: 0,
        results: BTreeMap::new(),
        next_play: 0,
        received: 0,
        voice: String::new(),
        speed: 1.0,
        ctx: None,
        next_time: 0.0,
        sources: Vec::new(),
        all_audio: Vec::new(),
        t_start: 0.0,
        t_first: None,
        loaded: HashSet::new(),
    };
    APP.with(|a| *a.borrow_mut() = Some(app));
    with_app(|a| a.locale_changed());

    // Events.
    with_app(|a| {
        on(a.ui.locale.as_ref(), "change", |_| {
            with_app(App::locale_changed);
        });
        on(a.ui.speed.as_ref(), "input", |_| {
            with_app(|a| {
                set_text(
                    &a.ui.speed_val,
                    &format!("{:.2}×", a.ui.speed.value_as_number()),
                )
            });
        });
        on(a.ui.speak.as_ref(), "click", |_| {
            wasm_bindgen_futures::spawn_local(async {
                if let Err(e) = speak().await {
                    with_app(|a| a.show_error(&e));
                }
            });
        });
        on(a.ui.stop.as_ref(), "click", |_| {
            with_app(App::stop);
        });
        on(a.ui.download.as_ref(), "click", |_| {
            with_app(App::download);
        });
        on(a.ui.threads.as_ref(), "change", |_| {
            with_app(App::ensure_workers);
        });
        on(a.ui.accel.as_ref(), "change", |_| {
            with_app(App::restart_workers);
        });
        on(a.ui.text.as_ref(), "keydown", |e| {
            let ke: &web_sys::KeyboardEvent = e.unchecked_ref();
            if (ke.ctrl_key() || ke.meta_key()) && ke.key() == "Enter" {
                e.prevent_default();
                wasm_bindgen_futures::spawn_local(async {
                    if let Err(e) = speak().await {
                        with_app(|a| a.show_error(&e));
                    }
                });
            }
        });
        set_text(&a.ui.speed_val, "1.00×");
    });

    wasm_bindgen_futures::spawn_local(async {
        if let Err(e) = boot().await {
            with_app(|a| a.show_error(&e));
        }
    });
    Ok(())
}

fn initial_locale(nav: &str) -> &'static str {
    let n = nav.replace('_', "-");
    if let Some(l) = find_locale(&n).filter(|l| l.code.eq_ignore_ascii_case(&n) || n.len() == 2) {
        return l.code;
    }
    if n.to_ascii_lowercase().starts_with("es") {
        return find_locale(&n).map_or("es-ES", |l| l.code);
    }
    if n.eq_ignore_ascii_case("en-GB") {
        "en-GB"
    } else {
        "es-ES"
    }
}

async fn boot() -> Result<(), JsValue> {
    let bytes = fetch_fresh("models/manifest.json").await?;
    let manifest = js_sys::JSON::parse(&String::from_utf8_lossy(&bytes))?;
    let cache = format!("{CACHE_PREFIX}{}", get_str(&manifest, "version"));
    prune_caches(CACHE_PREFIX, &cache).await;
    with_app(|a| {
        a.manifest = manifest;
        a.cache = cache;
        a.populate_voices();
        a.set_status(a.s.loading);
        a.ui.bar_wrap.style().set_property("display", "block").ok();
        a.ensure_workers();
    });
    Ok(())
}

async fn speak() -> Result<(), JsValue> {
    // Make sure the English lexicon is present before phonemising.
    let need = with_app(|a| {
        let code = a.ui.locale.value();
        let loc = find_locale(&code)?;
        (loc.lang == Lang::En && !a.lexicons.contains(loc.code))
            .then(|| (loc.code.to_string(), a.cache.clone()))
    })
    .flatten();
    if let Some((code, cache)) = need {
        with_app(|a| a.set_status(a.s.lexicon));
        let bytes = fetch_bytes(&format!("lexicons/{code}.lex"), &cache, |_| {}).await?;
        let text = String::from_utf8_lossy(&bytes).into_owned();
        with_app(|a| {
            a.frontend.load_english_lexicon(&text, code == "en-GB");
            a.lexicons.insert(code);
        });
    }
    with_app(App::begin).unwrap_or(Ok(()))
}

impl App {
    fn set_status(&self, t: &str) {
        set_text(&self.ui.status, t);
        let _ = self.ui.status.class_list().remove_1("error");
    }

    fn show_error(&self, e: &JsValue) {
        let msg = e.as_string().unwrap_or_else(|| format!("{e:?}"));
        set_text(&self.ui.status, &format!("{}: {msg}", self.s.error));
        let _ = self.ui.status.class_list().add_1("error");
        web_sys::console::error_1(e);
    }

    fn locale_changed(&mut self) {
        let code = self.ui.locale.value();
        let Some(loc) = find_locale(&code) else {
            return;
        };
        let current = self.ui.text.value();
        if current.trim().is_empty() || current == self.last_sample {
            self.ui.text.set_value(loc.sample);
        }
        self.last_sample = loc.sample.to_string();
        self.populate_voices();
    }

    fn populate_voices(&mut self) {
        let code = self.ui.locale.value();
        let Some(loc) = find_locale(&code) else {
            return;
        };
        let mut available: Vec<String> = get(&self.manifest, "voices")
            .dyn_into::<Array>()
            .map(|a| a.iter().filter_map(|v| v.as_string()).collect())
            .unwrap_or_default();
        if let Ok(models) = get(&self.manifest, "piper").dyn_into::<Array>() {
            for m in models.iter() {
                if let Ok(vs) = get(&m, "voices").dyn_into::<Array>() {
                    available.extend(vs.iter().filter_map(|v| v.as_string()));
                }
            }
        }
        let has = |v: &str| available.is_empty() || available.iter().any(|a| a == v);
        self.ui.voice.set_inner_html("");
        let add_group = |label: &str, names: Vec<&str>| -> Result<(), JsValue> {
            if names.is_empty() {
                return Ok(());
            }
            let group: HtmlOptGroupElement = doc().create_element("optgroup")?.dyn_into()?;
            group.set_label(label);
            for n in names {
                let info = voice_info(n);
                let text = format!("{} · {}", info.label, info.language);
                let opt = HtmlOptionElement::new_with_text_and_value(&text, n)?;
                group.append_child(&opt)?;
            }
            self.ui.voice.append_child(&group)?;
            Ok(())
        };
        // Only voices made for this language: no voice speaks a language
        // it was not trained on.
        let native: Vec<&str> = loc.voices.iter().copied().filter(|v| has(v)).collect();
        if native.is_empty() {
            // This build has no voice for the accent.
            self.ui.speak.set_disabled(true);
            self.set_status(self.s.no_voice);
            return;
        }
        let first = native[0];
        let _ = add_group(self.s.native_voices, native);
        self.ui.voice.set_value(first);
        if self.slots.first().is_some_and(|s| s.ready) {
            self.ui.speak.set_disabled(false);
            self.set_status(self.s.ready);
        }
    }

    fn ensure_workers(&mut self) {
        if self.manifest.is_undefined() {
            return;
        }
        let threads: usize = self.ui.threads.value().parse().unwrap_or(1);
        let want = match (&self.gpu, self.ui.accel.value().as_str()) {
            (_, "cpu") | (Gpu::Off, _) => threads,
            _ => 1,
        };
        self.ui.threads.set_disabled(matches!(self.gpu, Gpu::On(_)));
        while self.slots.len() < want {
            let index = self.slots.len();
            let opts = WorkerOptions::new();
            opts.set_type(WorkerType::Module);
            let worker = match Worker::new_with_options("worker.js", &opts) {
                Ok(w) => w,
                Err(e) => {
                    self.show_error(&e);
                    return;
                }
            };
            let handler = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
                let d = e.data();
                with_app(|a| a.on_worker(index, d));
            });
            worker.set_onmessage(Some(handler.as_ref().unchecked_ref()));
            handler.forget();
            self.slots.push(Slot {
                worker,
                ready: false,
                busy: false,
            });
        }
        while self.slots.len() > want {
            if let Some(s) = self.slots.pop() {
                s.worker.terminate();
            }
        }
    }

    /// Terminates every worker and starts again (accelerator changed).
    fn restart_workers(&mut self) {
        self.stop();
        for s in self.slots.drain(..) {
            s.worker.terminate();
        }
        self.gpu = Gpu::Unknown;
        self.loaded.clear();
        self.ui.speak.set_disabled(true);
        self.set_status(self.s.preparing);
        self.ensure_workers();
    }

    /// Sends the model location to a worker once its module has booted
    /// (messages posted earlier would be lost while it instantiates).
    fn init_worker(&self, index: usize) {
        let Some(slot) = self.slots.get(index) else {
            return;
        };
        let gpu = match (self.ui.accel.value().as_str(), &self.gpu) {
            ("cpu", _) | (_, Gpu::Off) => "off",
            _ if index > 0 => "off",
            ("gpu", _) => "force",
            _ => "auto",
        };
        let model = get(&self.manifest, "model");
        let msg = obj(&[
            ("type", "init".into()),
            ("base", "models".into()),
            ("cache", self.cache.clone().into()),
            ("files", get(&model, "files")),
            ("bytes", get(&model, "bytes")),
            ("piper", get(&self.manifest, "piper")),
            ("report", (index == 0).into()),
            ("gpu", gpu.into()),
        ]);
        let _ = slot.worker.post_message(&msg);
    }

    fn on_worker(&mut self, index: usize, d: JsValue) {
        match get_str(&d, "type").as_str() {
            "booted" => self.init_worker(index),
            "progress" => {
                let (l, t) = (get_f64(&d, "loaded"), get_f64(&d, "total").max(1.0));
                let shown = if l < t { "block" } else { "none" };
                let _ = self.ui.bar_wrap.style().set_property("display", shown);
                let _ = self
                    .ui
                    .bar
                    .style()
                    .set_property("width", &format!("{:.1}%", 100.0 * l / t));
                if l >= t {
                    self.set_status(self.s.preparing);
                } else {
                    set_text(
                        &self.ui.status,
                        &format!("{} · {:.0} / {:.0} MB", self.s.loading, l / 1e6, t / 1e6),
                    );
                }
            }
            "ready" => {
                if let Some(s) = self.slots.get_mut(index) {
                    s.ready = true;
                }
                if index == 0 {
                    let _ = self.ui.bar.style().set_property("width", "100%");
                    let _ = self.ui.bar_wrap.style().set_property("display", "none");
                    // Without a voice for the accent Speak stays disabled.
                    if self.ui.voice.length() > 0 {
                        self.ui.speak.set_disabled(false);
                        self.set_status(self.s.ready);
                    } else {
                        self.set_status(self.s.no_voice);
                    }
                    // Models load on first use; "loaded" reports the GPU.
                    self.gpu = Gpu::Off;
                    set_text(
                        &self.ui.metrics,
                        &format!("CPU · {}", get_str(&d, "kernel")),
                    );
                    self.ensure_workers();
                }
                self.dispatch();
            }
            "loaded" => {
                self.loaded.insert(get_str(&d, "model"));
                if index == 0 && get_str(&d, "model") == "k82" {
                    let gpu = get_str(&d, "gpu");
                    let gpu_error = get_str(&d, "gpu_error");
                    let device = if gpu.is_empty() {
                        let mut t = format!("CPU · {}", get_str(&d, "kernel"));
                        if !gpu_error.is_empty() {
                            web_sys::console::info_1(&format!("WebGPU: {gpu_error}").into());
                            if self.ui.accel.value() == "gpu" {
                                t = format!("{t} · WebGPU: {gpu_error}");
                            }
                        }
                        t
                    } else {
                        self.gpu = Gpu::On(gpu.clone());
                        if gpu == "WebGPU" {
                            gpu
                        } else {
                            format!("WebGPU · {gpu}")
                        }
                    };
                    set_text(
                        &self.ui.metrics,
                        &format!("{device} · {:.1} s", get_f64(&d, "ms") / 1e3),
                    );
                    self.ensure_workers();
                }
                self.dispatch();
            }
            "audio" => {
                if let Some(s) = self.slots.get_mut(index) {
                    s.busy = false;
                }
                if get_f64(&d, "job") == self.job {
                    let samples: Float32Array = get(&d, "samples").unchecked_into();
                    self.results
                        .insert(get_f64(&d, "index") as usize, samples.to_vec());
                    self.received += 1;
                    self.play_ready();
                }
                self.dispatch();
            }
            "error" => {
                if let Some(s) = self.slots.get_mut(index) {
                    s.busy = false;
                }
                self.show_error(&get(&d, "message"));
            }
            "gpu_lost" => {
                web_sys::console::warn_1(&get(&d, "message"));
                self.set_status(self.s.gpu_fallback);
                self.gpu = Gpu::Off;
                self.ensure_workers();
            }
            _ => {}
        }
    }

    fn begin(&mut self) -> Result<(), JsValue> {
        let text = self.ui.text.value();
        if text.trim().is_empty() {
            self.set_status(self.s.empty);
            return Ok(());
        }
        let code = self.ui.locale.value();
        let loc = find_locale(&code).ok_or_else(|| err("unknown locale"))?;
        let ps = self.frontend.phonemize(&text, loc).map_err(err)?;
        set_text(&self.ui.phonemes, &ps);
        let chunks = chunk_phonemes(
            &ps,
            ChunkPlan {
                first_max: 100,
                ..ChunkPlan::default()
            },
        );
        self.stop();
        if self.ctx.is_none() {
            self.ctx = Some(AudioContext::new()?);
        }
        if let Some(ctx) = &self.ctx {
            let _ = ctx.resume();
        }
        self.job += 1.0;
        self.total = chunks.len();
        self.queue = chunks.into_iter().enumerate().collect();
        self.results.clear();
        self.next_play = 0;
        self.received = 0;
        self.all_audio.clear();
        self.voice = self.ui.voice.value();
        self.speed = self.ui.speed.value_as_number();
        self.t_start = now();
        self.t_first = None;
        self.next_time = 0.0;
        self.ui.download.set_disabled(true);
        self.set_status(&format!("{} 1 / {}…", self.s.speaking, self.total));
        self.dispatch();
        Ok(())
    }

    /// Model a voice needs: its Piper file, or "k82" for English voices.
    fn model_key(&self, voice: &str) -> String {
        if let Ok(models) = get(&self.manifest, "piper").dyn_into::<Array>() {
            for m in models.iter() {
                let has = get(&m, "voices")
                    .dyn_into::<Array>()
                    .is_ok_and(|vs| vs.iter().any(|v| v.as_string().as_deref() == Some(voice)));
                if has {
                    return get_str(&m, "file");
                }
            }
        }
        "k82".into()
    }

    fn dispatch(&mut self) {
        let first_only = !self.loaded.contains(&self.model_key(&self.voice));
        for i in 0..self.slots.len() {
            if !self.slots[i].ready || self.slots[i].busy || (first_only && i > 0) {
                continue;
            }
            let Some((index, ps)) = self.queue.pop_front() else {
                return;
            };
            let msg = obj(&[
                ("type", "synth".into()),
                ("job", self.job.into()),
                ("index", (index as f64).into()),
                ("ps", ps.into()),
                ("voice", self.voice.clone().into()),
                ("speed", self.speed.into()),
                ("seed", ((index as f64) + 1.0).into()),
            ]);
            let _ = self.slots[i].worker.post_message(&msg);
            self.slots[i].busy = true;
        }
    }

    fn play_ready(&mut self) {
        let Some(ctx) = self.ctx.clone() else { return };
        while let Some(samples) = self.results.remove(&self.next_play) {
            if self.t_first.is_none() {
                self.t_first = Some(now() - self.t_start);
            }
            if let Ok(buf) = ctx.create_buffer(1, samples.len().max(1) as u32, SAMPLE_RATE) {
                let _ = buf.copy_to_channel(&samples, 0);
                if let Ok(src) = ctx.create_buffer_source() {
                    src.set_buffer(Some(&buf));
                    let _ = src.connect_with_audio_node(&ctx.destination());
                    let at = self.next_time.max(ctx.current_time() + 0.05);
                    let node: &AudioScheduledSourceNode = src.as_ref();
                    let _ = node.start_with_when(at);
                    self.next_time = at + samples.len() as f64 / SAMPLE_RATE as f64;
                    self.sources.push(src);
                }
            }
            self.all_audio.extend_from_slice(&samples);
            self.next_play += 1;
        }
        let audio_s = self.all_audio.len() as f64 / SAMPLE_RATE as f64;
        let wall = (now() - self.t_start) / 1e3;
        let first = self.t_first.unwrap_or(0.0) / 1e3;
        set_text(
            &self.ui.metrics,
            &format!(
                "{} {first:.2} s · {:.2}{} · {audio_s:.1} s audio",
                self.s.first_audio,
                audio_s / wall.max(1e-6),
                self.s.realtime
            ),
        );
        if self.next_play >= self.total {
            self.set_status(self.s.done);
            self.ui.download.set_disabled(false);
        } else {
            self.set_status(&format!(
                "{} {} / {}…",
                self.s.speaking,
                (self.next_play + 1).min(self.total),
                self.total
            ));
        }
    }

    fn stop(&mut self) {
        let was_playing = !self.sources.is_empty() || !self.queue.is_empty();
        self.job += 1.0;
        self.queue.clear();
        for s in self.sources.drain(..) {
            let node: &AudioScheduledSourceNode = s.as_ref();
            let _ = node.stop();
        }
        self.next_time = 0.0;
        if was_playing {
            self.set_status(self.s.stopped);
            self.ui.download.set_disabled(self.all_audio.is_empty());
        }
    }

    fn download(&mut self) {
        if self.all_audio.is_empty() {
            return;
        }
        let wav = inhouse_tts::core::dsp::wav_bytes(&self.all_audio, SAMPLE_RATE as u32);
        let arr = Uint8Array::from(&wav[..]);
        let parts = Array::of1(&arr);
        let props = BlobPropertyBag::new();
        props.set_type("audio/wav");
        let Ok(blob) = Blob::new_with_u8_array_sequence_and_options(&parts, &props) else {
            return;
        };
        let Ok(url) = Url::create_object_url_with_blob(&blob) else {
            return;
        };
        if let Ok(a) = doc()
            .create_element("a")
            .and_then(|e| e.dyn_into::<HtmlAnchorElement>().map_err(JsValue::from))
        {
            a.set_href(&url);
            a.set_download("inhouse-tts.wav");
            a.click();
        }
        let _ = Url::revoke_object_url(&url);
    }
}
