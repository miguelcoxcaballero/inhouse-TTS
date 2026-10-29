//! `cargo xtask <command>` — build automation written in Rust.
//!
//! * `models --kokoro DIR [--out models] [--dtype f32|f16|q8]`
//!   converts the Kokoro-82M v1.0 checkpoint and voices to `.ihw`.
//! * `web --kokoro DIR [--out web/dist]` builds the GitHub Pages site:
//!   WebAssembly (SIMD128) + bindings, int8 model split in parts, one file
//!   per voice, English lexicons and `manifest.json`.
//! * `serve [--dir web/dist] [--port 8080]` serves a directory locally.

use inhouse_tts_core::model::k82::convert;
use inhouse_tts_core::weights::Writer;
use inhouse_tts_core::{torchload, DType};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Pages rejects files over 100 MiB; stay well below.
const PART_BYTES: usize = 40 * 1024 * 1024;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("xtask: {msg}");
    std::process::exit(1)
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn read(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| die(format!("{}: {e}", p.display())))
}

fn write(p: &Path, bytes: &[u8]) {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).unwrap();
    }
    std::fs::write(p, bytes).unwrap_or_else(|e| die(format!("{}: {e}", p.display())));
}

fn dtype(s: &str) -> DType {
    match s {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "q8" => DType::Q8,
        d => die(format!("unknown dtype {d}")),
    }
}

fn convert_model(kokoro: &Path, dt: DType) -> Vec<u8> {
    let cfg: serde_json::Value =
        serde_json::from_slice(&read(&kokoro.join("config.json"))).unwrap();
    let root = torchload::load(&read(&kokoro.join("kokoro-v1_0.pth"))).unwrap_or_else(|e| die(e));
    convert::convert(torchload::flatten(root), &cfg, dt).unwrap_or_else(|e| die(e))
}

fn voices(kokoro: &Path) -> Vec<(String, Vec<f32>)> {
    let mut out = Vec::new();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(kokoro.join("voices"))
        .unwrap_or_else(|e| die(e))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    for p in entries {
        if p.extension().is_some_and(|e| e == "pt") {
            let name = p.file_stem().unwrap().to_string_lossy().to_string();
            let t = match torchload::load(&read(&p)).unwrap_or_else(|e| die(e)) {
                torchload::Value::Tensor(t) => t,
                _ => die(format!("{}: not a tensor", p.display())),
            };
            out.push((name, convert::voice(&t).unwrap_or_else(|e| die(e))));
        }
    }
    out
}

fn voice_file(name: &str, pack: &[f32]) -> Vec<u8> {
    let mut w = Writer::new("voices", serde_json::json!({"style_dim": 256}));
    w.add(name, &[pack.len() / 256, 256], pack, DType::F32);
    w.finish()
}

fn models(args: &[String]) {
    let kokoro =
        PathBuf::from(arg(args, "--kokoro").unwrap_or_else(|| die("--kokoro DIR required")));
    let out = PathBuf::from(arg(args, "--out").unwrap_or_else(|| "models".into()));
    let dt = dtype(&arg(args, "--dtype").unwrap_or_else(|| "f32".into()));
    let model = convert_model(&kokoro, dt);
    write(&out.join("k82.ihw"), &model);
    let mut w = Writer::new("voices", serde_json::json!({"style_dim": 256}));
    let vs = voices(&kokoro);
    for (name, pack) in &vs {
        w.add(name, &[pack.len() / 256, 256], pack, DType::F32);
    }
    write(&out.join("voices.ihw"), &w.finish());
    println!(
        "models: k82.ihw ({:.1} MB) and {} voices in {}",
        model.len() as f64 / 1e6,
        vs.len(),
        out.display()
    );
}

fn run(cmd: &mut Command) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| die(format!("{cmd:?}: {e}")));
    if !status.success() {
        die(format!("{cmd:?} failed"));
    }
}

/// FNV-1a, used to version the browser cache.
fn fnv(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn web(args: &[String]) {
    let root = root();
    let kokoro =
        PathBuf::from(arg(args, "--kokoro").unwrap_or_else(|| die("--kokoro DIR required")));
    let out = PathBuf::from(
        arg(args, "--out").unwrap_or_else(|| root.join("web/dist").to_string_lossy().into()),
    );
    let _ = std::fs::remove_dir_all(&out);
    // 1. WebAssembly.
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    run(Command::new(cargo)
        .current_dir(&root)
        .env("RUSTFLAGS", "-C target-feature=+simd128")
        .args([
            "build",
            "--release",
            "--target",
            "wasm32-unknown-unknown",
            "-p",
            "inhouse-tts-web",
        ]));
    let wasm = root.join("target/wasm32-unknown-unknown/release/inhouse_tts_web.wasm");
    run(Command::new("wasm-bindgen")
        .args(["--target", "web", "--no-typescript", "--out-dir"])
        .arg(out.join("pkg"))
        .arg(&wasm));
    // 2. Static files.
    for f in ["index.html", "main.js", "worker.js"] {
        write(&out.join(f), &read(&root.join("web").join(f)));
    }
    write(&out.join(".nojekyll"), b"");
    // 3. Lexicons.
    for l in ["en-US", "en-GB"] {
        write(
            &out.join(format!("lexicons/{l}.lex")),
            &read(&root.join(format!("crates/inhouse-tts-text/data/{l}.lex"))),
        );
    }
    // 4. Model (int8 decoder, f16 elsewhere), split in parts.
    let model = convert_model(&kokoro, DType::Q8);
    let mut files = Vec::new();
    for (i, part) in model.chunks(PART_BYTES).enumerate() {
        let name = format!("k82-q8.part{i}");
        write(&out.join("models").join(&name), part);
        files.push(name);
    }
    let mut version = fnv(&model, 0xcbf2_9ce4_8422_2325);
    // 5. Voices, one file each.
    let mut names = Vec::new();
    for (name, pack) in voices(&kokoro) {
        let bytes = voice_file(&name, &pack);
        version = fnv(&bytes, version);
        write(&out.join(format!("models/voices/{name}.ihw")), &bytes);
        names.push(name);
    }
    let manifest = serde_json::json!({
        "version": format!("{version:016x}"),
        "model": { "files": files, "bytes": model.len(), "dtype": "q8" },
        "voices": names,
    });
    write(
        &out.join("models/manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap().as_bytes(),
    );
    println!(
        "web: site in {} (model {:.1} MB in {} parts, {} voices)",
        out.display(),
        model.len() as f64 / 1e6,
        files.len(),
        names.len()
    );
}

fn content_type(p: &Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" => "text/javascript; charset=utf-8",
        "wasm" => "application/wasm",
        "json" => "application/json",
        "css" => "text/css",
        "svg" => "image/svg+xml",
        "wav" => "audio/wav",
        "lex" | "txt" | "md" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

fn serve(args: &[String]) {
    let dir = PathBuf::from(
        arg(args, "--dir").unwrap_or_else(|| root().join("web/dist").to_string_lossy().into()),
    );
    let port = arg(args, "--port").unwrap_or_else(|| "8080".into());
    let listener =
        std::net::TcpListener::bind(format!("127.0.0.1:{port}")).unwrap_or_else(|e| die(e));
    println!("serving {} on http://127.0.0.1:{port}/", dir.display());
    for stream in listener.incoming().flatten() {
        let dir = dir.clone();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            if reader.read_line(&mut line).is_err() {
                return;
            }
            // Drain headers.
            let mut h = String::new();
            while reader.read_line(&mut h).is_ok_and(|n| n > 2) {
                h.clear();
            }
            let path = line
                .split_whitespace()
                .nth(1)
                .unwrap_or("/")
                .split('?')
                .next()
                .unwrap_or("/")
                .to_string();
            let rel = path.trim_start_matches('/');
            let mut file = dir.join(if rel.is_empty() { "index.html" } else { rel });
            if file.is_dir() {
                file = file.join("index.html");
            }
            let mut s = &stream;
            let ok = !rel.contains("..") && file.is_file();
            if ok {
                let body = std::fs::read(&file).unwrap_or_default();
                let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n", content_type(&file), body.len());
                let _ = s.write_all(&body);
            } else {
                let _ = write!(s, "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nnot found");
            }
        });
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("models") => models(&args),
        Some("web") => web(&args),
        Some("serve") => serve(&args),
        _ => eprintln!("usage: cargo xtask <models|web|serve> [options]"),
    }
}
