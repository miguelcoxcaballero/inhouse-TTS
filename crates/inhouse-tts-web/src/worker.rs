//! Synthesis Web Worker: downloads the model once (cached), runs the
//! decoder on WebGPU when available (WebAssembly SIMD otherwise) and posts
//! the audio of every phoneme chunk back as a transferable buffer.
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
    let base = get_str(&d, "base");
    let cache = get_str(&d, "cache");
    let files: Array = get(&d, "files").dyn_into()?;
    let total = get_f64(&d, "bytes") as usize;
    let report = get(&d, "report").as_bool().unwrap_or(true);
    // "auto": hardware GPUs only; "force": also software adapters; "off".
    let gpu = get_str(&d, "gpu");
    let t0 = now();
    let mut bytes: Vec<u8> = Vec::with_capacity(total);
    let mut last = 0.0;
    for f in files.iter() {
        let url = format!("{base}/{}", f.as_string().unwrap_or_default());
        let before = bytes.len();
        let part = fetch_bytes(&url, &cache, |n| {
            let t = now();
            if report && t - last > 100.0 {
                last = t;
                post(&obj(&[
                    ("type", "progress".into()),
                    ("loaded", ((before + n) as f64).into()),
                    ("total", (total as f64).into()),
                ]));
            }
        })
        .await?;
        bytes.extend_from_slice(&part);
    }
    let t1 = now();
    let use_gpu = matches!(gpu.as_str(), "auto" | "force");
    let mut engine = Engine::new(
        &bytes,
        LoadOptions {
            prepack: false,
            gpu: use_gpu,
        },
    )
    .map_err(|e| err(e.0))?;
    drop(bytes);
    let t2 = now();
    let (mut gpu_name, mut gpu_error) = (String::new(), String::new());
    if use_gpu {
        match engine
            .enable_gpu(GpuOptions {
                allow_software: gpu == "force",
            })
            .await
        {
            Ok(n) => gpu_name = n,
            Err(e) => gpu_error = e.0,
        }
    }
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.engine = Some(engine);
        s.base = base;
        s.cache = cache;
    });
    post(&obj(&[
        ("type", "ready".into()),
        ("download_ms", (t1 - t0).into()),
        ("parse_ms", (t2 - t1).into()),
        ("gpu_ms", (now() - t2).into()),
        ("kernel", inhouse_tts::core::gemm::kernel().name.into()),
        ("gpu", gpu_name.into()),
        ("gpu_error", gpu_error.into()),
    ]));
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
    let e = engine.ok_or_else(|| err("model not loaded"))?;
    // Fetch the voice packs this request needs.
    for name in Engine::voice_names(voice) {
        if !e.voices().any(|v| v == name) {
            let bytes = fetch_bytes(&format!("{base}/voices/{name}.ihw"), cache, |_| {}).await?;
            e.add_voices(&bytes).map_err(|e| err(e.0))?;
        }
    }
    let v = e.resolve_voice(voice).map_err(|e| err(e.0))?;
    let t = now();
    let audio = match e.synth_phonemes_async(ps, &v, opts).await {
        Ok(a) => a,
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
