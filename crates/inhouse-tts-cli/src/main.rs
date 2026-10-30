//! `inhouse-tts` command line tool.

mod dev;

use inhouse_tts::core::model::k82::LoadOptions;
use inhouse_tts::{voice_info, Engine, SpeakOptions, SAMPLE_RATE};
use std::io::{Read, Write};
use std::time::Instant;

const USAGE: &str = "Inhouse TTS — local, real-time, multilingual text-to-speech in Rust

USAGE:
  inhouse-tts say [OPTIONS] [TEXT]      synthesise TEXT (or stdin) to a WAV file
  inhouse-tts phonemize [OPTIONS] TEXT  print model phonemes
  inhouse-tts locales                   list languages and accents
  inhouse-tts voices [OPTIONS]          list voices in the voice pack
  inhouse-tts bench [OPTIONS]           measure real-time factor

OPTIONS:
  -l, --locale CODE   es-ES, es-MX, es-AR, es-CO, es-CL, es-PE, es-VE, es-CU,
                      es-PR, es-AN, es-IC, en-US, en-GB   (default en-US)
  -v, --voice NAME    voice or blend, e.g. ef_dora, 0.6*af_heart+0.4*bf_emma
  -s, --speed X       speaking rate (default 1.0)
  -o, --out FILE      output WAV (default out.wav); '-' writes to stdout
      --model FILE    K82 model (.ihw)   [env INHOUSE_TTS_MODEL, default models/k82.ihw]
      --voices FILE   voice pack (.ihw)  [env INHOUSE_TTS_VOICES, default models/voices.ihw]
      --piper DIR     Piper voices (.ihw) [env INHOUSE_TTS_PIPER, default models/piper]
      --seed N        noise seed (default 1)
      --stream        write one WAV per chunk as soon as it is ready
      --gpu           run the decoder on the GPU (Vulkan, Metal or DX12)
";

pub fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn opt(args: &[String], long: &str, short: &str) -> Option<String> {
    arg(args, long).or_else(|| arg(args, short))
}

/// Positional text: everything that is not an option or option value.
fn positional(args: &[String]) -> Option<String> {
    let takes_value = [
        "-l", "--locale", "-v", "--voice", "-s", "--speed", "-o", "--out", "--model", "--voices",
        "--seed", "--piper",
    ];
    let mut out = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if takes_value.contains(&args[i].as_str()) {
            i += 2;
            continue;
        }
        if !args[i].starts_with("--") {
            out.push(args[i].clone());
        }
        i += 1;
    }
    if out.is_empty() {
        None
    } else {
        Some(out.join(" "))
    }
}

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}

/// Loads the K82 model and voice pack when present (English voices) and
/// every Piper voice in the Piper directory. A missing K82 model is fine
/// when Piper voices are available.
fn load_engine(args: &[String]) -> Engine {
    let explicit = arg(args, "--model").or_else(|| std::env::var("INHOUSE_TTS_MODEL").ok());
    let model = explicit.clone().unwrap_or_else(|| "models/k82.ihw".into());
    let voices = arg(args, "--voices")
        .or_else(|| std::env::var("INHOUSE_TTS_VOICES").ok())
        .unwrap_or_else(|| "models/voices.ihw".into());
    let piper = arg(args, "--piper")
        .or_else(|| std::env::var("INHOUSE_TTS_PIPER").ok())
        .unwrap_or_else(|| "models/piper".into());
    let gpu = args.iter().any(|a| a == "--gpu");
    let mut engine = match std::fs::read(&model) {
        Ok(bytes) => {
            let opts = LoadOptions {
                gpu,
                ..Default::default()
            };
            let mut e = Engine::new(&bytes, opts).unwrap_or_else(|e| die(e));
            if gpu {
                enable_gpu(&mut e);
            }
            let vb = std::fs::read(&voices)
                .unwrap_or_else(|e| die(format!("cannot read voices {voices}: {e}")));
            e.add_voices(&vb).unwrap_or_else(|e| die(e));
            e
        }
        // Only the default location may be missing (Piper-only setup).
        Err(e) if explicit.is_none() && e.kind() == std::io::ErrorKind::NotFound => {
            Engine::without_model()
        }
        Err(e) => die(format!("cannot read model {model}: {e}")),
    };
    let mut files: Vec<_> = std::fs::read_dir(&piper)
        .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    files.retain(|p| p.extension().is_some_and(|e| e == "ihw"));
    files.sort();
    for f in &files {
        let bytes = std::fs::read(f).unwrap_or_else(|e| die(format!("{}: {e}", f.display())));
        engine
            .add_piper(&f.file_stem().unwrap_or_default().to_string_lossy(), &bytes)
            .unwrap_or_else(|e| die(format!("{}: {e}", f.display())));
    }
    if !engine.has_model() && files.is_empty() {
        die(format!(
            "no voices: neither {model} nor Piper voices in {piper} (see README: models)"
        ));
    }
    engine
}

#[cfg(feature = "gpu")]
fn enable_gpu(engine: &mut Engine) {
    let opts = inhouse_tts::GpuOptions {
        allow_software: std::env::var_os("INHOUSE_TTS_SOFTWARE_GPU").is_some(),
    };
    match inhouse_tts::core::gpu::block_on(engine.enable_gpu(opts)) {
        Ok(name) => eprintln!("GPU: {name}"),
        Err(e) => eprintln!("GPU unavailable ({e}); using the CPU"),
    }
}

#[cfg(not(feature = "gpu"))]
fn enable_gpu(_: &mut Engine) {
    eprintln!("this build has no GPU support; using the CPU");
}

fn speak_options(args: &[String]) -> SpeakOptions {
    let locale = opt(args, "--locale", "-l").unwrap_or_else(|| "en-US".into());
    let loc = inhouse_tts::text::find_locale(&locale)
        .unwrap_or_else(|| die(format!("unsupported locale {locale}")));
    SpeakOptions {
        locale: loc.code.to_string(),
        // Empty: the best loaded voice of the locale (see `with_voice`).
        voice: opt(args, "--voice", "-v").unwrap_or_default(),
        speed: opt(args, "--speed", "-s")
            .map(|s| s.parse().unwrap_or_else(|_| die("bad speed")))
            .unwrap_or(1.0),
        seed: arg(args, "--seed")
            .map(|s| s.parse().unwrap_or_else(|_| die("bad seed")))
            .unwrap_or(1),
        ..Default::default()
    }
}

/// Fills in the default voice of the locale once the engine is loaded.
fn with_voice(mut opts: SpeakOptions, engine: &Engine) -> SpeakOptions {
    if opts.voice.is_empty() {
        opts.voice = engine
            .default_voice(&opts.locale)
            .unwrap_or_else(|| die(format!("no voice for {}", opts.locale)));
    }
    if !engine.has_voices(&opts.voice) {
        let hint = if engine.has_model() {
            "see `inhouse-tts voices`"
        } else {
            "English voices need the K82 model (--model, default models/k82.ihw); Spanish voices go in models/piper"
        };
        die(format!("voice {} is not loaded: {hint}", opts.voice));
    }
    opts
}

fn text_arg(args: &[String]) -> String {
    positional(args).unwrap_or_else(|| {
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .unwrap_or_else(|e| die(e));
        s
    })
}

fn write_wav(path: &str, audio: &[f32]) {
    let bytes = inhouse_tts::core::dsp::wav_bytes(audio, SAMPLE_RATE);
    if path == "-" {
        std::io::stdout()
            .write_all(&bytes)
            .unwrap_or_else(|e| die(e));
    } else {
        std::fs::write(path, bytes).unwrap_or_else(|e| die(format!("{path}: {e}")));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if dev::run(&args) {
        return;
    }
    match args.first().map(String::as_str) {
        Some("say") => {
            let opts = speak_options(&args);
            let text = text_arg(&args);
            let out = opt(&args, "--out", "-o").unwrap_or_else(|| "out.wav".into());
            let t0 = Instant::now();
            let mut engine = load_engine(&args);
            let opts = with_voice(opts, &engine);
            let load = t0.elapsed();
            let t1 = Instant::now();
            let stream = args.iter().any(|a| a == "--stream");
            let mut audio = Vec::new();
            let mut first = None;
            for chunk in engine.stream(&text, &opts).unwrap_or_else(|e| die(e)) {
                let chunk = chunk.unwrap_or_else(|e| die(e));
                first.get_or_insert(t1.elapsed());
                eprintln!(
                    "  chunk {:>2}: {:>5.2}s audio  {}",
                    chunk.index,
                    chunk.audio.len() as f64 / SAMPLE_RATE as f64,
                    chunk.phonemes
                );
                if stream && out != "-" {
                    write_wav(
                        &format!("{}.{:03}.wav", out.trim_end_matches(".wav"), chunk.index),
                        &chunk.audio,
                    );
                }
                audio.extend(chunk.audio);
            }
            let total = t1.elapsed().as_secs_f64();
            let secs = audio.len() as f64 / SAMPLE_RATE as f64;
            eprintln!(
                "{secs:.2}s of audio in {total:.2}s (RTF {:.3}, first chunk after {:.2}s, model load {:.2}s) → {out}",
                total / secs.max(1e-9),
                first.map_or(0.0, |d| d.as_secs_f64()),
                load.as_secs_f64()
            );
            write_wav(&out, &audio);
        }
        Some("phonemize") => {
            let opts = speak_options(&args);
            let text = text_arg(&args);
            let mut fe = inhouse_tts::text::Frontend::new();
            let loc = inhouse_tts::text::find_locale(&opts.locale).unwrap();
            println!("{}", fe.phonemize(&text, loc).unwrap_or_else(|e| die(e)));
        }
        Some("locales") => {
            for l in inhouse_tts::text::LOCALES {
                println!(
                    "{:<6} {:<40} voices: {}",
                    l.code,
                    l.name,
                    l.voices.join(", ")
                );
            }
        }
        Some("voices") => {
            let engine = load_engine(&args);
            for v in engine.voices() {
                let i = voice_info(v);
                println!("{:<14} {:<8} {}", v, i.language, i.label);
            }
        }
        Some("bench") => {
            let opts = speak_options(&args);
            let text = positional(&args).unwrap_or_else(|| {
                inhouse_tts::text::find_locale(&opts.locale)
                    .map(|l| l.sample.to_string())
                    .unwrap_or_default()
            });
            let mut engine = load_engine(&args);
            let opts = with_voice(opts, &engine);
            let _ = engine.speak(
                "Warm up.",
                &SpeakOptions {
                    locale: "en-US".into(),
                    voice: opts.voice.clone(),
                    ..opts.clone()
                },
            );
            let t = Instant::now();
            let audio = engine.speak(&text, &opts).unwrap_or_else(|e| die(e));
            let el = t.elapsed().as_secs_f64();
            let secs = audio.len() as f64 / SAMPLE_RATE as f64;
            let device = match engine.gpu_name() {
                Some(g) => format!("GPU {g}"),
                None => format!(
                    "{} threads, kernel {}",
                    inhouse_tts::core::par::num_threads(),
                    inhouse_tts::core::gemm::kernel().name
                ),
            };
            println!(
                "{device}: {secs:.2}s audio in {el:.2}s → RTF {:.3} ({:.1}x real time)",
                el / secs,
                secs / el
            );
        }
        _ => print!("{USAGE}"),
    }
}
