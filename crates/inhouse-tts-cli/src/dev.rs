use inhouse_tts_core::model::k82::{self, InferOptions, LoadOptions, Trace, K82};
use inhouse_tts_core::tensor::Mat;
use inhouse_tts_core::{dsp, torchload, DType, WeightFile};

// Reference-dump helpers used by `compare`.
use std::time::Instant;

use crate::arg;

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn cmp(name: &str, ours: &[f32], want: &[f32]) {
    let n = ours.len().min(want.len());
    let mut max = 0f32;
    let mut sq = 0f64;
    let mut ref_sq = 0f64;
    for i in 0..n {
        let d = (ours[i] - want[i]).abs();
        max = max.max(d);
        sq += (d as f64).powi(2);
        ref_sq += (want[i] as f64).powi(2);
    }
    let snr = 10.0 * (ref_sq / sq.max(1e-30)).log10();
    println!(
        "{name:<14} len {:>7}/{:<7} max|d| {max:.3e}  SNR {snr:6.1} dB",
        ours.len(),
        want.len()
    );
}

/// Developer commands (weight conversion, reference comparisons).
/// Returns false when `args[0]` is not a developer command.
pub fn run(args: &[String]) -> bool {
    let args = args.to_vec();
    match args.first().map(String::as_str) {
        Some("convert") => {
            let pth = arg(&args, "--pth").expect("--pth");
            let cfg: serde_json::Value = serde_json::from_slice(
                &std::fs::read(arg(&args, "--config").expect("--config")).unwrap(),
            )
            .unwrap();
            let out = arg(&args, "--out").expect("--out");
            let dtype = match arg(&args, "--dtype").as_deref().unwrap_or("f32") {
                "f32" => DType::F32,
                "f16" => DType::F16,
                "q8" => DType::Q8,
                d => panic!("unknown dtype {d}"),
            };
            let t = Instant::now();
            let bytes = std::fs::read(&pth).unwrap();
            let root = torchload::load(&bytes).unwrap();
            let tensors = torchload::flatten(root);
            println!("read {} tensors in {:?}", tensors.len(), t.elapsed());
            let ihw = k82::convert::convert(tensors, &cfg, dtype).unwrap();
            std::fs::write(&out, &ihw).unwrap();
            println!("wrote {out} ({:.1} MB)", ihw.len() as f64 / 1e6);
        }
        Some("compare") => {
            let model = arg(&args, "--model").expect("--model");
            let dir = arg(&args, "--golden").expect("--golden");
            let t = Instant::now();
            let m = K82::load(
                WeightFile::parse(&std::fs::read(&model).unwrap()).unwrap(),
                LoadOptions::default(),
            )
            .unwrap();
            println!("loaded in {:?}", t.elapsed());
            let manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(format!("{dir}/manifest.json")).unwrap())
                    .unwrap();
            let ids: Vec<u32> = manifest["input_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let style = read_f32(&format!("{dir}/ref_s.bin"));
            let mut trace = Trace::default();
            let t = Instant::now();
            let out = m.infer_traced(
                &ids,
                &style,
                InferOptions {
                    speed: 1.0,
                    seed: None,
                },
                Some(&mut trace),
            );
            let el = t.elapsed();
            println!(
                "infer {:?} for {:.2}s audio (RTF {:.3})",
                el,
                out.audio.len() as f64 / 24000.0,
                el.as_secs_f64() / (out.audio.len() as f64 / 24000.0)
            );
            let want_dur: Vec<usize> = manifest["pred_dur"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            println!("durations match: {}", want_dur == out.durations);
            if want_dur != out.durations {
                println!(" ours {:?}\n want {:?}", out.durations, want_dur);
            }
            for name in [
                "d_en",
                "d",
                "F0_pred",
                "N_pred",
                "t_en",
                "dec_decode3",
                "har",
            ] {
                cmp(
                    name,
                    &trace.tensors[name],
                    &read_f32(&format!("{dir}/{name}.bin")),
                );
            }
            cmp("audio", &out.audio, &read_f32(&format!("{dir}/audio.bin")));
            // Our STFT on the reference source signal.
            let src_ref = read_f32(&format!("{dir}/har_source.bin"));
            let (mag, phase) = dsp::Stft::new(20, 5).forward(&src_ref);
            let har_ref = read_f32(&format!("{dir}/har.bin"));
            let nf = mag.cols;
            cmp("stft mag", &mag.data, &har_ref[..11 * nf]);
            let mut worst = 0f32;
            for i in 0..11 * nf {
                if har_ref[i] > 1e-3 {
                    let mut d = (phase.data[i] - har_ref[11 * nf + i]).abs();
                    if d > std::f32::consts::PI {
                        d = 2.0 * std::f32::consts::PI - d;
                    }
                    worst = worst.max(d);
                }
            }
            println!("stft phase max wrapped diff (mag>1e-3): {worst:.3e}");
            cmp("source", &trace.tensors["har_source"], &src_ref);
            // Generator alone on the reference inputs.
            let x = read_f32(&format!("{dir}/dec_decode3.bin"));
            let har = read_f32(&format!("{dir}/har.bin"));
            let frames = har.len() / 22;
            let g = m.generator_only(
                &Mat::from_vec(512, x.len() / 512, x),
                &style,
                &Mat::from_vec(22, frames, har),
            );
            cmp("gen(ref in)", &g, &read_f32(&format!("{dir}/audio.bin")));
            std::fs::write(format!("{dir}/ours.wav"), dsp::wav_bytes(&out.audio, 24000)).unwrap();
            for (name, v) in &trace.tensors {
                let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/ours_{name}.bin"), b).unwrap();
            }
        }
        Some("convert-voices") => {
            let out = arg(&args, "--out").expect("--out");
            let dtype = if arg(&args, "--dtype").as_deref() == Some("f32") {
                DType::F32
            } else {
                DType::F16
            };
            let mut w = inhouse_tts_core::weights::Writer::new(
                "voices",
                serde_json::json!({"style_dim": 256}),
            );
            for path in args.iter().skip(1).filter(|a| a.ends_with(".pt")) {
                let name = std::path::Path::new(path)
                    .file_stem()
                    .unwrap()
                    .to_string_lossy()
                    .to_string();
                let root = torchload::load(&std::fs::read(path).unwrap()).unwrap();
                let t = match root {
                    torchload::Value::Tensor(t) => t,
                    _ => panic!("{path}: not a tensor"),
                };
                let v = k82::convert::voice(&t).unwrap();
                w.add(&name, &[v.len() / 256, 256], &v, dtype);
                println!("{name}: {:?}", t.shape);
            }
            let bytes = w.finish();
            std::fs::write(&out, &bytes).unwrap();
            println!("wrote {out} ({:.1} MB)", bytes.len() as f64 / 1e6);
        }
        Some("synth-ph") => {
            let model = arg(&args, "--model").expect("--model");
            let voices = arg(&args, "--voices").expect("--voices");
            let voice = arg(&args, "--voice").expect("--voice");
            let out = arg(&args, "--out").expect("--out");
            let ph = arg(&args, "--ph").expect("--ph");
            let seed: u64 = arg(&args, "--seed")
                .map(|s| s.parse().unwrap())
                .unwrap_or(1);
            let t = Instant::now();
            let m = K82::load(
                WeightFile::parse(&std::fs::read(&model).unwrap()).unwrap(),
                LoadOptions::default(),
            )
            .unwrap();
            let mut vf = WeightFile::parse(&std::fs::read(&voices).unwrap()).unwrap();
            let pack = vf.get(&voice).unwrap().to_f32();
            println!("loaded in {:?}", t.elapsed());
            let ids = m.tokenize(&ph);
            if args.iter().any(|a| a == "--print-ids") {
                println!("ids {ids:?}");
            }
            let n = ph.chars().count();
            let style = &pack[(n - 1).min(pack.len() / 256 - 1) * 256..][..256];
            let t = Instant::now();
            let mut trace = Trace::default();
            let o = m.infer_traced(
                &ids,
                style,
                InferOptions {
                    speed: 1.0,
                    seed: Some(seed),
                },
                Some(&mut trace),
            );
            if let Some(dir) = arg(&args, "--dump") {
                for (name, v) in &trace.tensors {
                    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    std::fs::write(format!("{dir}/{name}.bin"), b).unwrap();
                }
            }
            let el = t.elapsed().as_secs_f64();
            let dur = o.audio.len() as f64 / 24000.0;
            println!("{:.2}s audio in {:.3}s (RTF {:.3})", dur, el, el / dur);
            std::fs::write(&out, dsp::wav_bytes(&o.audio, 24000)).unwrap();
            let _ = vf.take(&voice);
        }
        #[cfg(feature = "gpu")]
        Some("gpu-check") => {
            // Runs the decoder on the CPU and on the GPU from the same front
            // output and compares them.
            use inhouse_tts_core::gpu::{block_on, Gpu, GpuOptions};
            use inhouse_tts_core::model::k82::gpu::K82Gpu;
            let model = arg(&args, "--model").expect("--model");
            let voices = arg(&args, "--voices").expect("--voices");
            let voice = arg(&args, "--voice").expect("--voice");
            let ph = arg(&args, "--ph").expect("--ph");
            let m = K82::load(
                WeightFile::parse(&std::fs::read(&model).unwrap()).unwrap(),
                LoadOptions {
                    gpu: true,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut vf = WeightFile::parse(&std::fs::read(&voices).unwrap()).unwrap();
            let pack = vf.get(&voice).unwrap().to_f32();
            let n = ph.chars().count();
            let style = &pack[(n - 1).min(pack.len() / 256 - 1) * 256..][..256];
            let t = Instant::now();
            let gpu = block_on(Gpu::new(GpuOptions {
                allow_software: true,
            }))
            .unwrap();
            let g = K82Gpu::new(gpu, &m).unwrap();
            println!("GPU {} ready in {:?}", g.name(), t.elapsed());
            let ids = m.tokenize(&ph);
            let opts = InferOptions {
                speed: 1.0,
                seed: Some(1),
            };
            let t = Instant::now();
            let inp = m.prepare(&ids, style, opts);
            println!("front (CPU): {:?}", t.elapsed());
            let t = Instant::now();
            let cpu = m.decode(&inp, style);
            let dur = cpu.len() as f64 / 24000.0;
            let el = t.elapsed().as_secs_f64();
            println!("decoder CPU: {el:.3}s for {dur:.2}s (RTF {:.3})", el / dur);
            for run in 0..2 {
                let t = Instant::now();
                let gpu = block_on(g.decode(&m, &inp, style)).unwrap();
                let el = t.elapsed().as_secs_f64();
                println!("decoder GPU run {run}: {el:.3}s (RTF {:.3})", el / dur);
                cmp("gpu vs cpu", &gpu, &cpu);
                if let (0, Some(out)) = (run, arg(&args, "--out")) {
                    std::fs::write(&out, dsp::wav_bytes(&gpu, 24000)).unwrap();
                }
            }
            let _ = vf.take(&voice);
        }
        Some("ihv1-synth") => {
            // Inference of an exported IHV1 model (numerical checks).
            use inhouse_tts_core::model::ihv1::{Ihv1, Ihv1Options};
            let model = arg(&args, "--model").expect("--model");
            let ph = arg(&args, "--ph").expect("--ph");
            let out = arg(&args, "--out").expect("--out");
            let speaker: usize = arg(&args, "--speaker")
                .map(|s| s.parse().unwrap())
                .unwrap_or(0);
            let locale: usize = arg(&args, "--locale")
                .map(|s| s.parse().unwrap())
                .unwrap_or(0);
            let noise: f32 = arg(&args, "--noise")
                .map(|s| s.parse().unwrap())
                .unwrap_or(0.667);
            let t = Instant::now();
            let m = Ihv1::load(
                WeightFile::parse(&std::fs::read(&model).unwrap()).unwrap(),
                true,
            )
            .unwrap();
            println!("loaded in {:?}", t.elapsed());
            let t = Instant::now();
            let audio = m.synthesize(
                &ph,
                speaker,
                locale,
                Ihv1Options {
                    noise,
                    ..Default::default()
                },
            );
            let el = t.elapsed().as_secs_f64();
            let dur = audio.len() as f64 / m.sample_rate as f64;
            println!(
                "{dur:.2}s audio in {el:.3}s (RTF {:.3})",
                el / dur.max(1e-9)
            );
            std::fs::write(&out, dsp::wav_bytes(&audio, m.sample_rate)).unwrap();
        }
        Some("piper-convert") => {
            // Piper voice (.onnx + .onnx.json) → .ihw.
            use inhouse_tts_core::model::piper;
            let onnx = arg(&args, "--onnx").expect("--onnx");
            let json = arg(&args, "--json").unwrap_or_else(|| format!("{onnx}.json"));
            let out = arg(&args, "--out").expect("--out");
            let dtype = match arg(&args, "--dtype").as_deref().unwrap_or("f16") {
                "f32" => DType::F32,
                "f16" => DType::F16,
                "q8" => DType::Q8,
                d => panic!("unknown dtype {d}"),
            };
            let t = Instant::now();
            let ihw = piper::convert(
                &std::fs::read(&onnx).unwrap(),
                &std::fs::read(&json).unwrap(),
                dtype,
            )
            .unwrap();
            std::fs::write(&out, &ihw).unwrap();
            println!(
                "wrote {out} ({:.1} MB) in {:?}",
                ihw.len() as f64 / 1e6,
                t.elapsed()
            );
        }
        Some("piper-synth") => {
            // Piper inference from ids or from espeak-ng IPA (already NFD).
            use inhouse_tts_core::model::piper::{Piper, PiperOptions};
            let model = arg(&args, "--model").expect("--model");
            let out = arg(&args, "--out").expect("--out");
            let t = Instant::now();
            let m = Piper::load(
                WeightFile::parse(&std::fs::read(&model).unwrap()).unwrap(),
                true,
            )
            .unwrap();
            println!(
                "loaded {} ({}, espeak voice {:?}, {} Hz, {} speaker(s)) in {:?}",
                m.meta["name"].as_str().unwrap_or("piper"),
                m.language(),
                m.espeak_voice(),
                m.sample_rate,
                m.num_speakers(),
                t.elapsed()
            );
            let ids: Vec<i64> = match (arg(&args, "--ids"), arg(&args, "--ph")) {
                (Some(ids), _) => ids
                    .split(',')
                    .map(|s| s.trim().parse().expect("--ids: comma-separated integers"))
                    .collect(),
                (None, Some(ph)) => m.encode(&ph),
                (None, None) => panic!("--ids or --ph"),
            };
            if args.iter().any(|a| a == "--print-ids") {
                println!("ids {ids:?}");
            }
            let d = m.defaults();
            let num = |name: &str, default: f32| -> f32 {
                arg(&args, name).map_or(default, |s| s.parse().unwrap())
            };
            let opts = PiperOptions {
                noise_scale: num("--noise-scale", d.noise_scale),
                length_scale: num("--length-scale", d.length_scale),
                noise_w: num("--noise-w", d.noise_w),
                seed: arg(&args, "--seed").map_or(1, |s| s.parse().unwrap()),
            };
            let speaker: usize = arg(&args, "--speaker").map_or(0, |s| s.parse().unwrap());
            let runs: usize = arg(&args, "--runs").map_or(1, |s| s.parse().unwrap());
            let mut best = f64::INFINITY;
            let mut o = None;
            for _ in 0..runs.max(1) {
                let t = Instant::now();
                o = Some(m.synthesize(&ids, speaker, opts).unwrap());
                best = best.min(t.elapsed().as_secs_f64());
            }
            let o = o.unwrap();
            let dur = o.audio.len() as f64 / m.sample_rate as f64;
            println!(
                "{} ids, {} frames, {dur:.2}s audio in {best:.3}s (RTF {:.3}, {} threads)",
                ids.len(),
                o.durations.iter().sum::<usize>(),
                best / dur.max(1e-9),
                inhouse_tts_core::par::num_threads()
            );
            std::fs::write(&out, dsp::wav_bytes(&o.audio, m.sample_rate)).unwrap();
        }
        Some("build-lexicon") => {
            // Merges misaki gold + silver JSON lexicons into the compact
            // `.lex` text format: `word\tphonemes` or
            // `word\tdefault\tTAG=phonemes...` (gold entries win).
            let out = arg(&args, "--out").expect("--out");
            let mut entries: std::collections::BTreeMap<String, String> =
                std::collections::BTreeMap::new();
            for key in ["--silver", "--gold"] {
                let Some(path) = arg(&args, key) else {
                    continue;
                };
                let v: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
                for (word, ps) in v.as_object().unwrap() {
                    let line = match ps {
                        serde_json::Value::String(s) => s.clone(),
                        serde_json::Value::Object(m) => {
                            let mut parts = vec![m
                                .get("DEFAULT")
                                .and_then(|d| d.as_str())
                                .unwrap_or("")
                                .to_string()];
                            for (tag, val) in m {
                                if tag != "DEFAULT" {
                                    parts.push(format!("{tag}={}", val.as_str().unwrap_or("")));
                                }
                            }
                            parts.join("\t")
                        }
                        _ => continue,
                    };
                    if line.is_empty() {
                        continue;
                    }
                    entries.insert(word.clone(), line);
                }
            }
            let mut text = String::new();
            for (w, l) in &entries {
                text.push_str(w);
                text.push('\t');
                text.push_str(l);
                text.push('\n');
            }
            std::fs::write(&out, text).unwrap();
            println!("wrote {out}: {} entries", entries.len());
        }
        Some("phonemize-lines") => {
            use std::io::BufRead;
            let lang = arg(&args, "--lang").unwrap_or_else(|| "es-ES".into());
            if lang.starts_with("en") {
                let g = inhouse_tts_text::en::G2p::new(inhouse_tts_text::en::Lexicon::embedded(
                    lang.eq_ignore_ascii_case("en-GB"),
                ));
                for line in std::io::stdin().lock().lines() {
                    println!("{}", g.phonemize(&line.unwrap()));
                }
                return true;
            }
            let acc = inhouse_tts_text::es::accent(&lang).expect("unknown accent");
            for line in std::io::stdin().lock().lines() {
                println!("{}", inhouse_tts_text::es::phonemize(&line.unwrap(), acc));
            }
        }
        _ => return false,
    }
    true
}
