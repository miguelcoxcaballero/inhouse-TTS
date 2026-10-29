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
      --seed N        noise seed (default 1)
      --stream        write one WAV per chunk as soon as it is ready
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
        "--seed",
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

fn load_engine(args: &[String]) -> Engine {
    let model = arg(args, "--model")
        .or_else(|| std::env::var("INHOUSE_TTS_MODEL").ok())
        .unwrap_or_else(|| "models/k82.ihw".into());
    let voices = arg(args, "--voices")
        .or_else(|| std::env::var("INHOUSE_TTS_VOICES").ok())
        .unwrap_or_else(|| "models/voices.ihw".into());
    let bytes = std::fs::read(&model).unwrap_or_else(|e| {
        die(format!(
            "cannot read model {model}: {e} (see README: models)"
        ))
    });
    let mut engine = Engine::new(&bytes, LoadOptions::default()).unwrap_or_else(|e| die(e));
    let vb =
        std::fs::read(&voices).unwrap_or_else(|e| die(format!("cannot read voices {voices}: {e}")));
    engine.add_voices(&vb).unwrap_or_else(|e| die(e));
    engine
}

fn speak_options(args: &[String]) -> SpeakOptions {
    let locale = opt(args, "--locale", "-l").unwrap_or_else(|| "en-US".into());
    let loc = inhouse_tts::text::find_locale(&locale)
        .unwrap_or_else(|| die(format!("unsupported locale {locale}")));
    SpeakOptions {
        locale: loc.code.to_string(),
        voice: opt(args, "--voice", "-v").unwrap_or_else(|| loc.voices[0].to_string()),
        speed: opt(args, "--speed", "-s")
            .map(|s| s.parse().unwrap_or_else(|_| die("bad speed")))
            .unwrap_or(1.0),
        seed: arg(args, "--seed")
            .map(|s| s.parse().unwrap_or_else(|_| die("bad seed")))
            .unwrap_or(1),
        ..Default::default()
    }
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
            println!(
                "{} threads, kernel {}: {secs:.2}s audio in {el:.2}s → RTF {:.3} ({:.1}x real time)",
                inhouse_tts::core::par::num_threads(),
                inhouse_tts::core::gemm::kernel().name,
                el / secs,
                secs / el
            );
        }
        _ => print!("{USAGE}"),
    }
}
