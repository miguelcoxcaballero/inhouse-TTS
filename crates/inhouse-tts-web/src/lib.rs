//! # Inhouse TTS web
//!
//! The browser build of Inhouse TTS. The same WebAssembly module runs in two
//! roles: `start_ui` drives the page (front end, chunk scheduling, streaming
//! playback) and `start_worker` runs the neural engine in a Web Worker.

mod i18n;
mod js;
mod ui;
mod worker;

use wasm_bindgen::prelude::*;

fn panic_to_console() {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&JsValue::from_str(&format!("inhouse-tts panic: {info}")));
    }));
}

/// Entry point of the page.
#[wasm_bindgen]
pub fn start_ui() -> Result<(), JsValue> {
    panic_to_console();
    ui::start()
}

/// Entry point of a synthesis worker.
#[wasm_bindgen]
pub fn start_worker() {
    panic_to_console();
    worker::start();
}

/// Micro-benchmark of the GEMM kernel (returns GMAC/s), used to tune the
/// WebAssembly build.
#[wasm_bindgen]
pub fn bench_gemm(m: usize, k: usize, n: usize, iters: usize) -> f64 {
    use inhouse_tts::core::gemm::{gemm_packed, pack_a, AData, AMat, BMat};
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7919) % 1000) as f32 * 1e-3)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 104729) % 1000) as f32 * 1e-3)
        .collect();
    let mut c = vec![0f32; m * n];
    let pa = pack_a(&AMat {
        m,
        k,
        data: AData::F32 {
            data: &a,
            rs: k,
            cs: 1,
        },
    });
    let bm = BMat::row_major(&b, k, n, n);
    gemm_packed(&pa, &bm.into(), &mut c, n, false);
    let t = js::now();
    for _ in 0..iters {
        gemm_packed(&pa, &bm.into(), &mut c, n, false);
    }
    let secs = (js::now() - t) / 1e3;
    (m * k * n * iters) as f64 / secs / 1e9
}
