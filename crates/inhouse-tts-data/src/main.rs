//! `ihtts-data`: prepares training corpora.
//!
//! ```text
//! ihtts-data prepare corpora/es-ES-*.json --cache data/cache --out data/corpora [--limit N]
//! ihtts-data inspect FILE.wav|FILE.flac|FILE.mp3
//! ihtts-data cml-index --speakers 10246,12367 --out corpora/es-ES/x.tsv[.gz]
//!                      [--items ITEM,...] [--repo freds0/cml_tts_dataset_spanish]
//!                      [--cache data/cache]
//! ```

use inhouse_tts_data::{audio, cml, prepare, CorpusSpec};
use std::path::PathBuf;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("prepare") => {
            let cache = PathBuf::from(arg(&args, "--cache").unwrap_or_else(|| "data/cache".into()));
            let out = PathBuf::from(arg(&args, "--out").unwrap_or_else(|| "data/corpora".into()));
            let limit = arg(&args, "--limit").and_then(|s| s.parse().ok());
            let specs: Vec<&String> = args[1..]
                .iter()
                .take_while(|a| !a.starts_with("--"))
                .collect();
            let mut failed = false;
            for path in specs {
                let spec = match CorpusSpec::load(path.as_ref()) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("error: {e}");
                        failed = true;
                        continue;
                    }
                };
                match prepare(&spec, &cache, &out, limit) {
                    Ok(r) => println!(
                        "{}: {} utterances, {:.2} h, {} speakers, {} skipped, energy > 8 kHz {:.2} %",
                        spec.name,
                        r.utterances,
                        r.hours,
                        r.speakers.len(),
                        r.skipped,
                        100.0 * r.hf_energy
                    ),
                    Err(e) => {
                        eprintln!("{}: error: {e}", spec.name);
                        failed = true;
                    }
                }
            }
            if failed {
                std::process::exit(1);
            }
        }
        Some("inspect") => {
            for p in &args[1..] {
                match audio::load(p.as_ref()) {
                    Ok(a) => {
                        let x = audio::resample(&a.samples, a.rate, 24_000);
                        println!(
                            "{p}: {} Hz, {:.2} s, speech {:.1} dBFS, energy > 4 kHz {:.2} %, > 8 kHz {:.2} %, 99 % rolloff {:.0} Hz",
                            a.rate,
                            a.samples.len() as f64 / a.rate as f64,
                            audio::speech_db(&x, 24_000),
                            100.0 * audio::energy_above(&x, 24_000, 4000.0),
                            100.0 * audio::energy_above(&x, 24_000, 8000.0),
                            audio::rolloff(&a.samples, a.rate, 0.99)
                        );
                    }
                    Err(e) => println!("{p}: {e}"),
                }
            }
        }
        Some("cml-index") => {
            let cache = PathBuf::from(arg(&args, "--cache").unwrap_or_else(|| "data/cache".into()));
            let repo =
                arg(&args, "--repo").unwrap_or_else(|| "freds0/cml_tts_dataset_spanish".into());
            let speakers = arg(&args, "--speakers")
                .unwrap_or_default()
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            let items = arg(&args, "--items")
                .unwrap_or_default()
                .split(',')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            let Some(out) = arg(&args, "--out") else {
                eprintln!("cml-index: --out is required");
                std::process::exit(2);
            };
            match cml::index(&repo, &speakers, &items, &cache, out.as_ref()) {
                Ok(n) => println!("{out}: {n} segments"),
                Err(e) => {
                    eprintln!("cml-index: error: {e}");
                    std::process::exit(1);
                }
            }
        }
        _ => eprintln!(
            "usage: ihtts-data prepare SPEC.json... [--cache DIR] [--out DIR] [--limit N] | inspect FILE... | cml-index --speakers IDS --out FILE"
        ),
    }
}
