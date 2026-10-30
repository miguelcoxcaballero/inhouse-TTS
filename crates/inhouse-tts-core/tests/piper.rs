//! Piper voices against onnxruntime reference outputs.
//!
//! Needs the research test data: set `IHTTS_PIPER_TESTDATA` to the folder
//! holding `dl/*.onnx(.json)` and `out/testvec_*/*.npy` (inputs, ORT
//! waveforms and durations written by `piper_ref.py` / `realvec.py`).
//! Without it the test only prints a note.

use inhouse_tts_core::model::piper::{self, Piper, PiperOptions};
use inhouse_tts_core::{DType, WeightFile};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Minimal `.npy` reader (little-endian f4 / f8 / i8, C order).
fn npy(path: &Path) -> (Vec<usize>, Vec<f64>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&b[..6], b"\x93NUMPY", "{}: not a .npy file", path.display());
    let (hlen, start) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&b[start..start + hlen]).unwrap();
    assert!(!header.contains("'fortran_order': True"));
    let field = |key: &str| {
        let i = header.find(key).unwrap() + key.len();
        header[i..].trim_start_matches([':', ' ', '\''])
    };
    let descr = &field("'descr'")[..3];
    let shape_s = field("'shape'");
    let shape: Vec<usize> = shape_s[1..shape_s.find(')').unwrap()]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data = &b[start + hlen..];
    let v = match descr {
        "<f4" => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect(),
        "<f8" => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        "<i8" => data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect(),
        d => panic!("unsupported dtype {d}"),
    };
    (shape, v)
}

/// (max |a - b|, SNR of `a` against reference `b` in dB).
fn diff(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "lengths differ");
    let mut max = 0f64;
    let (mut err, mut sig) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = (x as f64 - y as f64).abs();
        max = max.max(d);
        err += d * d;
        sig += (y as f64) * (y as f64);
    }
    (max, 10.0 * (sig / err.max(1e-30)).log10())
}

fn load(root: &Path, voice: &str, dtype: DType) -> Piper {
    let dir = if root.join("dl").join(format!("{voice}.onnx")).exists() {
        root.join("dl")
    } else {
        root.join("dl2")
    };
    let onnx = std::fs::read(dir.join(format!("{voice}.onnx"))).unwrap();
    let json = std::fs::read(dir.join(format!("{voice}.onnx.json"))).unwrap();
    let t = Instant::now();
    let ihw = piper::convert(&onnx, &json, dtype).unwrap();
    println!(
        "{voice} {dtype:?}: converted in {:.2?}, {:.1} MB",
        t.elapsed(),
        ihw.len() as f64 / 1e6
    );
    Piper::load(WeightFile::parse(&ihw).unwrap(), true).unwrap()
}

fn run(m: &Piper, dir: &Path, speaker: usize) -> (Vec<f32>, Vec<usize>, PiperOptions) {
    let ids: Vec<i64> = npy(&dir.join("input_ids.npy"))
        .1
        .iter()
        .map(|&v| v as i64)
        .collect();
    let scales = npy(&dir.join("scales.npy")).1;
    let opts = PiperOptions {
        noise_scale: scales[0] as f32,
        length_scale: scales[1] as f32,
        noise_w: scales[2] as f32,
        seed: 1,
    };
    let out = m.synthesize(&ids, speaker, opts).unwrap();
    (out.audio, out.durations, opts)
}

fn root() -> Option<PathBuf> {
    match std::env::var_os("IHTTS_PIPER_TESTDATA") {
        Some(p) => Some(PathBuf::from(p)),
        None => {
            println!("note: IHTTS_PIPER_TESTDATA not set, skipping the Piper reference test");
            None
        }
    }
}

#[test]
fn matches_onnxruntime() {
    let Some(root) = root() else { return };
    let cases: &[(&str, &[(&str, usize)])] = &[
        (
            "es_ES-davefx-medium",
            &[("testvec_davefx", 0), ("testvec_davefx_hola", 0)],
        ),
        (
            "es_ES-sharvard-medium",
            &[("testvec_sharvard_sid0", 0), ("testvec_sharvard_sid1", 1)],
        ),
        ("es_ES-carlfm-x_low", &[("testvec_carlfm_xlow", 0)]),
        ("es_ES-mls_10246-low", &[("testvec_mls10246_low", 0)]),
    ];
    for (voice, vecs) in cases {
        let m = load(&root, voice, DType::F32);
        for (vec, speaker) in *vecs {
            let dir = root.join("out").join(vec);
            if !dir.exists() {
                println!("{vec}: missing, skipped");
                continue;
            }
            let t = Instant::now();
            let (audio, durations, opts) = run(&m, &dir, *speaker);
            let el = t.elapsed().as_secs_f64();
            let dur_file = if dir.join("durations_ort.npy").exists() {
                "durations_ort.npy"
            } else {
                "durations_ref.npy"
            };
            let want_dur: Vec<usize> = npy(&dir.join(dur_file))
                .1
                .iter()
                .map(|&v| v as usize)
                .collect();
            assert_eq!(durations, want_dur, "{vec}: durations ({dur_file})");
            let want: Vec<f32> = npy(&dir.join("output_ort.npy"))
                .1
                .iter()
                .map(|&v| v as f32)
                .collect();
            let (max, snr) = diff(&audio, &want);
            let secs = audio.len() as f64 / m.sample_rate as f64;
            println!(
                "{vec} (speaker {speaker}, {opts:?}): {} frames, {secs:.2} s audio in {el:.3} s; \
                 vs ORT max|d| {max:.2e}, SNR {snr:.1} dB",
                durations.iter().sum::<usize>()
            );
            assert!(max < 1e-3, "{vec}: max abs error {max}");
        }
    }
}

#[test]
fn compact_storage_snr() {
    let Some(root) = root() else { return };
    let dir = root.join("out").join("testvec_davefx_hola");
    let reference = run(&load(&root, "es_ES-davefx-medium", DType::F32), &dir, 0);
    for dtype in [DType::F16, DType::Q8] {
        let (audio, durations, _) = run(&load(&root, "es_ES-davefx-medium", dtype), &dir, 0);
        assert_eq!(durations, reference.1, "{dtype:?}: durations changed");
        let (max, snr) = diff(&audio, &reference.0);
        println!("{dtype:?} vs f32: max|d| {max:.2e}, SNR {snr:.1} dB");
        let min_snr = if dtype == DType::F16 { 40.0 } else { 25.0 };
        assert!(snr > min_snr, "{dtype:?}: SNR {snr:.1} dB");
    }
}
