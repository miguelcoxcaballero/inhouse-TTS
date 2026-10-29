//! Synthesis Web Worker: downloads the model once (cached), then turns
//! phoneme chunks into audio and posts them back as transferable buffers.

use crate::js::{err, fetch_bytes, get, get_f64, get_str, now, obj};
use inhouse_tts::core::model::k82::LoadOptions;
use inhouse_tts::{Engine, SpeakOptions};
use js_sys::{Array, Float32Array};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

#[derive(Default)]
struct State {
    engine: Option<Engine>,
    base: String,
    cache: String,
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
        let data = e.data();
        let kind = get_str(&data, "type");
        wasm_bindgen_futures::spawn_local(async move {
            let r = match kind.as_str() {
                "init" => init(data).await,
                "synth" => synth(data).await,
                _ => Ok(()),
            };
            if let Err(e) = r {
                post_error(e);
            }
        });
    });
    scope().set_onmessage(Some(handler.as_ref().unchecked_ref()));
    handler.forget();
    post(&obj(&[("type", "booted".into())]));
}

async fn init(d: JsValue) -> Result<(), JsValue> {
    let base = get_str(&d, "base");
    let cache = get_str(&d, "cache");
    let files: Array = get(&d, "files").dyn_into()?;
    let total = get_f64(&d, "bytes") as usize;
    let report = get(&d, "report").as_bool().unwrap_or(true);
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
    let engine = Engine::new(&bytes, LoadOptions { prepack: false }).map_err(|e| err(e.0))?;
    drop(bytes);
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.engine = Some(engine);
        s.base = base;
        s.cache = cache;
    });
    post(&obj(&[
        ("type", "ready".into()),
        ("download_ms", (t1 - t0).into()),
        ("parse_ms", (now() - t1).into()),
        ("kernel", inhouse_tts::core::gemm::kernel().name.into()),
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
    // Load the voice packs this request needs.
    let (base, cache, missing) = STATE.with(|s| {
        let s = s.borrow();
        let loaded = |n: &str| {
            s.engine
                .as_ref()
                .is_some_and(|e| e.voices().any(|v| v == n))
        };
        let missing: Vec<String> = Engine::voice_names(&voice)
            .into_iter()
            .filter(|n| !loaded(n))
            .collect();
        (s.base.clone(), s.cache.clone(), missing)
    });
    for name in missing {
        let bytes = fetch_bytes(&format!("{base}/voices/{name}.ihw"), &cache, |_| {}).await?;
        STATE.with(|s| {
            s.borrow_mut()
                .engine
                .as_mut()
                .ok_or_else(|| err("model not loaded"))
                .and_then(|e| e.add_voices(&bytes).map_err(|e| err(e.0)))
        })?;
    }
    let t = now();
    let audio = STATE.with(|s| -> Result<Vec<f32>, JsValue> {
        let s = s.borrow();
        let e = s.engine.as_ref().ok_or_else(|| err("model not loaded"))?;
        let v = e.resolve_voice(&voice).map_err(|e| err(e.0))?;
        Ok(e.synth_phonemes(&ps, &v, &opts))
    })?;
    let ms = now() - t;
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
