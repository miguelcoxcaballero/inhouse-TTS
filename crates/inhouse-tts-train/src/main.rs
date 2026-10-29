//! `ihtts-train`: trains Inhouse TTS's own models (IHV1) in Rust.
//!
//! ```text
//! ihtts-train train --run runs/es --preset tiny|base --data a/manifest.jsonl [--data b/...]
//!                   [--steps N] [--device cpu|cuda|wgpu] [--max-items N] [--resume]
//! ihtts-train synth --run runs/es --speaker NAME --text "..." --out out.wav
//! ihtts-train export --run runs/es --out ihv1-es.ihw [--dtype f32|f16|q8]
//! ```

mod config;
mod data;
mod disc;
mod dsp;
mod export;
mod mas;
mod model;
mod nn;
mod train;

use burn::tensor::{Device, Int, Tensor, TensorData};
use config::Config;
use data::{Dataset, Tables};
use std::fs;
use std::path::PathBuf;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn args_all(args: &[String], name: &str) -> Vec<String> {
    args.windows(2)
        .filter(|w| w[0] == name)
        .map(|w| w[1].clone())
        .collect()
}

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}

fn device(name: &str) -> Device {
    let d = match name {
        #[cfg(feature = "cuda")]
        "cuda" => Device::cuda(0),
        #[cfg(feature = "wgpu")]
        "wgpu" => Device::wgpu(burn::tensor::DeviceKind::DefaultDevice),
        "cpu" => Device::flex(),
        other => die(format!("device {other} is not available in this build")),
    };
    d.autodiff()
}

/// Evaluation sentence per locale (phonemised with the Inhouse front end).
fn eval_sentences(locales: &[String]) -> Vec<(String, String)> {
    let mut fe = inhouse_tts_text::Frontend::new();
    locales
        .iter()
        .filter_map(|l| {
            let loc = inhouse_tts_text::find_locale(l)?;
            let ps = fe.phonemize(loc.sample, loc).ok()?;
            let first: String = ps
                .split_inclusive(['.', '!', '?'])
                .next()
                .unwrap_or(&ps)
                .to_string();
            Some((l.clone(), first))
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("train") => {
            let dir = PathBuf::from(arg(&args, "--run").unwrap_or_else(|| die("--run DIR")));
            fs::create_dir_all(&dir).unwrap_or_else(|e| die(e));
            let manifests: Vec<PathBuf> = args_all(&args, "--data")
                .into_iter()
                .map(PathBuf::from)
                .collect();
            if manifests.is_empty() {
                die("--data MANIFEST");
            }
            let cfg_path = dir.join("config.json");
            let cfg: Config = if cfg_path.exists() {
                serde_json::from_str(&fs::read_to_string(&cfg_path).unwrap())
                    .unwrap_or_else(|e| die(e))
            } else {
                let p = arg(&args, "--preset").unwrap_or_else(|| "tiny".into());
                let c = Config::preset(&p).unwrap_or_else(|| die(format!("unknown preset {p}")));
                fs::write(&cfg_path, serde_json::to_string_pretty(&c).unwrap()).unwrap();
                c
            };
            assert_eq!(
                cfg.decoder_hop(),
                cfg.audio.hop,
                "decoder upsampling must equal the hop"
            );
            let tables_path = dir.join("tables.json");
            let tables: Tables = if tables_path.exists() {
                serde_json::from_str(&fs::read_to_string(&tables_path).unwrap()).unwrap()
            } else {
                let mut utts = Vec::new();
                for m in &manifests {
                    utts.extend(inhouse_tts_data::read_manifest(m).unwrap_or_else(|e| die(e)));
                }
                let t = Tables::build(&utts);
                fs::write(&tables_path, serde_json::to_string_pretty(&t).unwrap()).unwrap();
                t
            };
            let max_items = arg(&args, "--max-items")
                .and_then(|s| s.parse().ok())
                .unwrap_or(usize::MAX);
            let data =
                Dataset::load(&manifests, &cfg, &tables, max_items).unwrap_or_else(|e| die(e));
            let steps = arg(&args, "--steps")
                .and_then(|s| s.parse().ok())
                .unwrap_or(100_000);
            let dev = device(&arg(&args, "--device").unwrap_or_else(|| "cpu".into()));
            let sentences = eval_sentences(&tables.locales);
            let run = train::Run { dir, cfg, tables };
            train::train(
                run,
                data,
                dev,
                steps,
                args.iter().any(|a| a == "--resume"),
                sentences,
            )
            .unwrap_or_else(|e| die(e));
        }
        Some("synth") => {
            let dir = PathBuf::from(arg(&args, "--run").unwrap_or_else(|| die("--run DIR")));
            let cfg: Config =
                serde_json::from_str(&fs::read_to_string(dir.join("config.json")).unwrap())
                    .unwrap();
            let tables: Tables =
                serde_json::from_str(&fs::read_to_string(dir.join("tables.json")).unwrap())
                    .unwrap();
            let dev = Device::flex();
            let mut net = model::Synthesizer::new(
                &dev,
                &cfg,
                tables.symbols.len(),
                tables.speakers.len(),
                tables.locales.len(),
            );
            train::load_generator(&mut net, &dir).unwrap_or_else(|e| die(e));
            let spk = arg(&args, "--speaker").unwrap_or_else(|| tables.speakers[0].clone());
            let loc = tables
                .speaker_locale
                .get(&spk)
                .cloned()
                .unwrap_or_else(|| tables.locales[0].clone());
            let text = arg(&args, "--text").unwrap_or_default();
            let mut fe = inhouse_tts_text::Frontend::new();
            let l = inhouse_tts_text::find_locale(&loc).unwrap_or_else(|| die("locale"));
            let ps = match arg(&args, "--ph") {
                Some(p) => p,
                None => fe.phonemize(&text, l).unwrap_or_else(|e| die(e)),
            };
            let ids = tables.encode(&ps);
            let n = ids.len();
            let k = model::Consts::new(&cfg, &dev);
            let wav = net.infer(
                Tensor::<2, Int>::from_data(TensorData::new(ids, [1, n]), &dev),
                tables.speaker(&spk),
                tables.locale(&loc),
                &k,
                arg(&args, "--noise")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.667),
                1.0,
            );
            let out = PathBuf::from(arg(&args, "--out").unwrap_or_else(|| "out.wav".into()));
            train::write_wav(&out, &wav, cfg.audio.sample_rate);
            println!(
                "{}: {:.2} s ({ps})",
                out.display(),
                wav.len() as f64 / cfg.audio.sample_rate as f64
            );
        }
        Some("export") => {
            let dir = PathBuf::from(arg(&args, "--run").unwrap_or_else(|| die("--run DIR")));
            let cfg: Config =
                serde_json::from_str(&fs::read_to_string(dir.join("config.json")).unwrap())
                    .unwrap();
            let tables: Tables =
                serde_json::from_str(&fs::read_to_string(dir.join("tables.json")).unwrap())
                    .unwrap();
            let dev = Device::flex();
            let mut net = model::Synthesizer::new(
                &dev,
                &cfg,
                tables.symbols.len(),
                tables.speakers.len(),
                tables.locales.len(),
            );
            train::load_generator(&mut net, &dir).unwrap_or_else(|e| die(e));
            let dtype = match arg(&args, "--dtype").as_deref().unwrap_or("f16") {
                "f32" => inhouse_tts_core::DType::F32,
                "q8" => inhouse_tts_core::DType::Q8,
                _ => inhouse_tts_core::DType::F16,
            };
            let (bytes, params) = export::export(&net, &cfg, &tables, dtype);
            let out = PathBuf::from(arg(&args, "--out").unwrap_or_else(|| "ihv1.ihw".into()));
            fs::write(&out, &bytes).unwrap_or_else(|e| die(e));
            println!(
                "{}: {:.1} MB, {:.2} M parameters",
                out.display(),
                bytes.len() as f64 / 1e6,
                params as f64 / 1e6
            );
        }
        _ => eprintln!("usage: ihtts-train train|synth|export … (see the source header)"),
    }
}
