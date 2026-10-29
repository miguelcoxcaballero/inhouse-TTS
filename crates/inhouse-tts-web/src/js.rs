//! Small helpers over js-sys/web-sys shared by the UI and the worker.

use js_sys::{Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Cache, CacheStorage, ReadableStreamDefaultReader, Response};

/// Builds a plain JS object from key/value pairs.
pub fn obj(pairs: &[(&str, JsValue)]) -> Object {
    let o = Object::new();
    for (k, v) in pairs {
        let _ = Reflect::set(&o, &JsValue::from_str(k), v);
    }
    o
}

pub fn get(v: &JsValue, k: &str) -> JsValue {
    Reflect::get(v, &JsValue::from_str(k)).unwrap_or(JsValue::UNDEFINED)
}

pub fn get_str(v: &JsValue, k: &str) -> String {
    get(v, k).as_string().unwrap_or_default()
}

pub fn get_f64(v: &JsValue, k: &str) -> f64 {
    get(v, k).as_f64().unwrap_or(0.0)
}

/// `performance.now()` in the window or in a worker.
pub fn now() -> f64 {
    let perf = get(&js_sys::global(), "performance");
    if perf.is_undefined() {
        return js_sys::Date::now();
    }
    let f: js_sys::Function = get(&perf, "now").unchecked_into();
    f.call0(&perf).ok().and_then(|v| v.as_f64()).unwrap_or(0.0)
}

pub fn err(msg: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&msg.to_string())
}

fn global_fetch(url: &str) -> js_sys::Promise {
    let g = js_sys::global();
    let f: js_sys::Function = get(&g, "fetch").unchecked_into();
    f.call1(&g, &JsValue::from_str(url))
        .map(|p| p.unchecked_into())
        .unwrap_or_else(|e| js_sys::Promise::reject(&e))
}

fn caches() -> Option<CacheStorage> {
    let c = get(&js_sys::global(), "caches");
    if c.is_undefined() || c.is_null() {
        None
    } else {
        Some(c.unchecked_into())
    }
}

async fn read_body(resp: &Response, mut progress: impl FnMut(usize)) -> Result<Vec<u8>, JsValue> {
    let mut out = Vec::new();
    let Some(body) = resp.body() else {
        let buf = JsFuture::from(resp.array_buffer()?).await?;
        let arr = Uint8Array::new(&buf);
        out.resize(arr.length() as usize, 0);
        arr.copy_to(&mut out);
        progress(out.len());
        return Ok(out);
    };
    let reader: ReadableStreamDefaultReader = body.get_reader().unchecked_into();
    loop {
        let chunk = JsFuture::from(reader.read()).await?;
        if get(&chunk, "done").as_bool().unwrap_or(true) {
            break;
        }
        let value: Uint8Array = get(&chunk, "value").unchecked_into();
        let start = out.len();
        out.resize(start + value.length() as usize, 0);
        value.copy_to(&mut out[start..]);
        progress(out.len());
    }
    Ok(out)
}

/// Fetches a file, keeping a copy in the Cache API so later visits work
/// offline and load instantly. `progress` receives the bytes read so far.
pub async fn fetch_bytes(
    url: &str,
    cache_name: &str,
    progress: impl FnMut(usize),
) -> Result<Vec<u8>, JsValue> {
    let cache: Option<Cache> = match caches() {
        Some(cs) => JsFuture::from(cs.open(cache_name))
            .await
            .ok()
            .map(|c| c.unchecked_into()),
        None => None,
    };
    if let Some(c) = &cache {
        if let Ok(hit) = JsFuture::from(c.match_with_str(url)).await {
            if let Ok(resp) = hit.dyn_into::<Response>() {
                return read_body(&resp, progress).await;
            }
        }
    }
    let resp: Response = JsFuture::from(global_fetch(url)).await?.dyn_into()?;
    if !resp.ok() {
        return Err(err(format!("HTTP {} loading {url}", resp.status())));
    }
    if let Some(c) = &cache {
        if let Ok(copy) = resp.clone() {
            let _ = JsFuture::from(c.put_with_str(url, &copy)).await;
        }
    }
    read_body(&resp, progress).await
}

/// Fetches a small file bypassing every cache (used for the manifest).
pub async fn fetch_fresh(url: &str) -> Result<Vec<u8>, JsValue> {
    let sep = if url.contains('?') { '&' } else { '?' };
    let fresh = format!("{url}{sep}t={}", js_sys::Date::now() as u64);
    let resp: Response = JsFuture::from(global_fetch(&fresh)).await?.dyn_into()?;
    if !resp.ok() {
        return Err(err(format!("HTTP {} loading {url}", resp.status())));
    }
    read_body(&resp, |_| {}).await
}

/// Removes every cache whose name starts with `prefix` except `keep`.
pub async fn prune_caches(prefix: &str, keep: &str) {
    let Some(cs) = caches() else { return };
    let Ok(keys) = JsFuture::from(cs.keys()).await else {
        return;
    };
    let keys: js_sys::Array = keys.unchecked_into();
    for k in keys.iter() {
        if let Some(name) = k.as_string() {
            if name.starts_with(prefix) && name != keep {
                let _ = JsFuture::from(cs.delete(&name)).await;
            }
        }
    }
}
