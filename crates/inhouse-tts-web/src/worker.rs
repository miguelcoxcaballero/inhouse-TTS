//! Synthesis Web Worker: downloads each model the first time a voice needs
//! it (cached) — a Piper voice file, or the K82 model for the English
//! voices — runs the K82 decoder on WebGPU when available (WebAssembly SIMD
//! otherwise) and posts the audio of every phoneme chunk back as a
//! transferable buffer.
//!
//! Messages are handled strictly in order, so a request never observes the
//! engine half-updated by another one.

use crate::js::{err, fetch_bytes, get, get_f64, get_str, now, obj};
use inhouse_tts::core::model::k82::LoadOptions;
use inhouse_tts::{Engine, GpuOptions, SpeakOptions};
use js_sys::{Array, Float32Array};
use std::cell::RefCell;
use std::collections::VecDeque;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

#[derive(Default)]
struct State {
    engine: Option<Engine>,
    base: String,
    cache: String,
    /// K82 model parts and size, downloaded on first use.
    model_files: Vec<String>,
    model_bytes: usize,
    /// "auto", "force" or "off" (K82 decoder on WebGPU).
    gpu: String,
    report: bool,
    /// Piper voice name → model file and its size.
    piper: Vec<(String, String, usize)>,
    inbox: VecDeque<JsValue>,
    running: bool,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

fn scope() -> DedicatedWorkerGlobalScope {
    js_sys::global().unchecked_into()
}

fn post(msg: &JsValue) {
    let _ = scope().post_message(msg);
}

fn post_error(e: JsValue) {
    let text = e.as_string().unwrap_or_else(|| format!("{e:?}"));
    post(&obj(&[("type", "error".into()), ("message", text.into())]));
}

pub fn start() {
    let handler = Closure::<dyn FnMut(MessageEvent)>::new(|e: MessageEvent| {
        STATE.with(|s| s.borrow_mut().inbox.push_back(e.data()));
        pump();
    });
    scope().set_onmessage(Some(handler.as_ref().unchecked_ref()));
    handler.forget();
    post(&obj(&[("type", "booted".into())]));
}

/// Processes queued messages one after another.
fn pump() {
    let idle = STATE.with(|s| !std::mem::replace(&mut s.borrow_mut().running, true));
    if !idle {
        return;
    }
    wasm_bindgen_futures::spawn_local(async {
        while let Some(d) = STATE.with(|s| s.borrow_mut().inbox.pop_front()) {
            let r = match get_str(&d, "type").as_str() {
                "init" => init(d).await,
                "synth" => synth(d).await,
                _ => Ok(()),
            };
            if let Err(e) = r {
                post_error(e);
            }
        }
        STATE.with(|s| s.borrow_mut().running = false);
    });
}

async fn init(d: JsValue) -> Result<(), JsValue> {
    let files: Array = get(&d, "files").dyn_into()?;
    let mut piper = Vec::new();
    if let Ok(models) = get(&d, "piper").dyn_into::<Array>() {
        for m in models.iter() {
            let file = get_str(&m, "file");
            let bytes = get_f64(&m, "bytes") as usize;
            if let Ok(vs) = get(&m, "voices").dyn_into::<Array>() {
                for v in vs.iter().filter_map(|v| v.as_string()) {
                    piper.push((v, file.clone(), bytes));
                }
            }
        }
    }
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.engine = Some(Engine::without_model());
        s.base = get_str(&d, "base");
        s.cache = get_str(&d, "cache");
        s.model_files = files.iter().filter_map(|f| f.as_string()).collect();
        s.model_bytes = get_f64(&d, "bytes") as usize;
        s.gpu = get_str(&d, "gpu");
        s.report = get(&d, "report").as_bool().unwrap_or(true);
        s.piper = piper;
    });
    post(&obj(&[
        ("type", "ready".into()),
        ("download_ms", 0.0.into()),
        ("parse_ms", 0.0.into()),
        ("gpu_ms", 0.0.into()),
        ("kernel", inhouse_tts::core::gemm::kernel().name.into()),
        ("gpu", "".into()),
        ("gpu_error", "".into()),
    ]));
    Ok(())
}

/// Downloads `files` (one model, possibly in parts) with progress reports.
async fn download(
    base: &str,
    cache: &str,
    files: &[String],
    total: usize,
    report: bool,
) -> Result<Vec<u8>, JsValue> {
    let mut bytes: Vec<u8> = Vec::with_capacity(total);
    let mut last = 0.0;
    for f in files {
        let before = bytes.len();
        let part = fetch_bytes(&format!("{base}/{f}"), cache, |n| {
            let t = now();
            if report && t - last > 100.0 {
                last = t;
                post(&obj(&[
                    ("type", "progress".into()),
                    ("loaded", ((before + n) as f64).into()),
                    ("total", (total.max(before + n) as f64).into()),
                ]));
            }
        })
        .await?;
        bytes.extend_from_slice(&part);
    }
    if report {
        let n = bytes.len() as f64;
        post(&obj(&[
            ("type", "progress".into()),
            ("loaded", n.into()),
            ("total", n.into()),
        ]));
    }
    Ok(bytes)
}

/// Tells the page that a model is loaded (`model`: a Piper file or "k82"),
/// with the GPU outcome for K82, so it can spread work over the workers.
fn post_loaded(model: &str, ms: f64, gpu: &str, gpu_error: &str) {
    post(&obj(&[
        ("type", "loaded".into()),
        ("model", model.into()),
        ("ms", ms.into()),
        ("kernel", inhouse_tts::core::gemm::kernel().name.into()),
        ("gpu", gpu.into()),
        ("gpu_error", gpu_error.into()),
    ]));
}

/// Loads the K82 model (English voices) and, if requested, its GPU decoder.
async fn ensure_k82(e: &mut Engine) -> Result<(), JsValue> {
    if e.has_model() {
        return Ok(());
    }
    let (base, cache, files, total, gpu, report) = STATE.with(|s| {
        let s = s.borrow();
        (
            s.base.clone(),
            s.cache.clone(),
            s.model_files.clone(),
            s.model_bytes,
            s.gpu.clone(),
            s.report,
        )
    });
    let t0 = now();
    let bytes = download(&base, &cache, &files, total, report).await?;
    let use_gpu = matches!(gpu.as_str(), "auto" | "force");
    e.load_model(
        &bytes,
        LoadOptions {
            prepack: false,
            gpu: use_gpu,
        },
    )
    .map_err(|e| err(e.0))?;
    drop(bytes);
    let (mut gpu_name, mut gpu_error) = (String::new(), String::new());
    if use_gpu {
        match e
            .enable_gpu(GpuOptions {
                allow_software: gpu == "force",
            })
            .await
        {
            Ok(n) => gpu_name = n,
            Err(g) => gpu_error = g.0,
        }
    }
    post_loaded("k82", now() - t0, &gpu_name, &gpu_error);
    Ok(())
}

async fn synth(d: JsValue) -> Result<(), JsValue> {
    let ps = get_str(&d, "ps");
    let voice = get_str(&d, "voice");
    let job = get_f64(&d, "job");
    let index = get_f64(&d, "index");
    let opts = SpeakOptions {
        speed: get_f64(&d, "speed").max(0.3) as f32,
        seed: get_f64(&d, "seed") as u64,
        ..Default::default()
    };
    let (mut engine, base, cache) = STATE.with(|s| {
        let mut s = s.borrow_mut();
        (s.engine.take(), s.base.clone(), s.cache.clone())
    });
    let r = run(engine.as_mut(), &base, &cache, &ps, &voice, &opts).await;
    STATE.with(|s| s.borrow_mut().engine = engine);
    let (audio, ms) = r?;
    let arr = Float32Array::new_with_length(audio.len() as u32);
    arr.copy_from(&audio);
    let msg = obj(&[
        ("type", "audio".into()),
        ("job", job.into()),
        ("index", index.into()),
        ("ms", ms.into()),
        ("samples", arr.clone().into()),
    ]);
    let _ = scope().post_message_with_transfer(&msg, &Array::of1(&arr.buffer()));
    Ok(())
}

async fn run(
    engine: Option<&mut Engine>,
    base: &str,
    cache: &str,
    ps: &str,
    voice: &str,
    opts: &SpeakOptions,
) -> Result<(Vec<f32>, f64), JsValue> {
    let e = engine.ok_or_else(|| err("engine not initialised"))?;
    // Fetch the models and voices this request needs.
    let (piper, report) = STATE.with(|s| (s.borrow().piper.clone(), s.borrow().report));
    for name in Engine::voice_names(voice) {
        if e.voices().any(|v| v == name) {
            continue;
        }
        if let Some((_, file, size)) = piper.iter().find(|(v, _, _)| *v == name) {
            let t0 = now();
            let bytes = download(base, cache, std::slice::from_ref(file), *size, report).await?;
            let stem = file
                .rsplit('/')
                .next()
                .unwrap_or(file)
                .trim_end_matches(".ihw");
            e.add_piper(stem, &bytes).map_err(|e| err(e.0))?;
            post_loaded(file, now() - t0, "", "");
        } else {
            ensure_k82(e).await?;
            let bytes = fetch_bytes(&format!("{base}/voices/{name}.ihw"), cache, |_| {}).await?;
            e.add_voices(&bytes).map_err(|e| err(e.0))?;
        }
    }
    let v = e.resolve_voice(voice).map_err(|e| err(e.0))?;
    let t = now();
    let audio = match e.synth_phonemes_async(ps, &v, opts).await {
        Ok(a) => a,
        Err(e2) if e.gpu_name().is_none() => return Err(err(e2.0)),
        Err(gpu_err) => {
            // The GPU failed (device lost, out of memory…): continue on the CPU.
            e.disable_gpu();
            post(&obj(&[
                ("type", "gpu_lost".into()),
                ("message", gpu_err.0.into()),
            ]));
            e.synth_phonemes(ps, &v, opts)
        }
    };
    Ok((audio, now() - t))
}
