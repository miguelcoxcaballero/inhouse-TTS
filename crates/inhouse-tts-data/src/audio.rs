//! Audio helpers for corpus preparation: decoding (WAV, FLAC, MP3, Ogg
//! Vorbis through Symphonia; Opus in WebM or Ogg through `opus-decoder`),
//! mono mixdown, band-limited resampling, silence trimming and loudness
//! normalisation.

use std::f64::consts::PI;
use std::fs::File;
use std::path::Path;

use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;
use symphonia::core::codecs::audio::{AudioCodecParameters, AudioDecoderOptions};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// Decoded mono audio.
pub struct Audio {
    pub samples: Vec<f32>,
    pub rate: u32,
}

/// Decodes a file and mixes it down to mono.
pub fn load(path: &Path) -> Result<Audio, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    load_from(Box::new(file), path.extension().and_then(|e| e.to_str()))
}

/// Decodes in-memory audio (e.g. a member of a zip or tar archive).
pub fn load_bytes(bytes: Vec<u8>, ext: Option<&str>) -> Result<Audio, String> {
    load_from(Box::new(std::io::Cursor::new(bytes)), ext)
}

fn load_from(
    src: Box<dyn symphonia::core::io::MediaSource>,
    ext: Option<&str>,
) -> Result<Audio, String> {
    let mss = MediaSourceStream::new(src, Default::default());
    let mut hint = Hint::new();
    if let Some(e) = ext {
        hint.with_extension(e);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| format!("unsupported audio: {e}"))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or("no audio track")?;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or("no audio codec parameters")?
        .clone();
    let track_id = track.id;
    if params.codec == CODEC_ID_OPUS {
        return decode_opus(&mut *format, track_id, &params);
    }
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|e| format!("no decoder: {e}"))?;
    let mut mono = Vec::new();
    let mut rate = params.sample_rate.unwrap_or(0);
    let mut buf: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymError::IoError(_)) => break,
            Err(e) => return Err(format!("read error: {e}")),
        };
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(ab) => {
                let spec = ab.spec();
                rate = spec.rate();
                let ch = spec.channels().count().max(1);
                buf.resize(ab.samples_interleaved(), 0.0);
                ab.copy_to_slice_interleaved(&mut buf);
                let inv = 1.0 / ch as f32;
                mono.extend(buf.chunks_exact(ch).map(|f| f.iter().sum::<f32>() * inv));
            }
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(format!("decode error: {e}")),
        }
    }
    if rate == 0 {
        return Err("unknown sample rate".into());
    }
    Ok(Audio {
        samples: mono,
        rate,
    })
}

/// Opus packets (from WebM/Matroska or Ogg) through the pure-Rust
/// `opus-decoder`, at 48 kHz, without the encoder's pre-skip.
fn decode_opus(
    format: &mut dyn FormatReader,
    track_id: u32,
    params: &AudioCodecParameters,
) -> Result<Audio, String> {
    // OpusHead: magic, version, channel count, pre-skip (little endian).
    let head = params.extra_data.as_deref().unwrap_or(&[]);
    let (channels, pre_skip) = if head.len() >= 12 && head.starts_with(b"OpusHead") {
        (
            head[9].max(1) as usize,
            u16::from_le_bytes([head[10], head[11]]) as usize,
        )
    } else {
        (
            params.channels.as_ref().map_or(1, |c| c.count()).max(1),
            312,
        )
    };
    let channels = channels.min(2);
    let mut dec =
        opus_decoder::OpusDecoder::new(48_000, channels).map_err(|e| format!("opus: {e}"))?;
    // 120 ms, the longest Opus packet.
    let mut pcm = vec![0f32; 5760 * channels];
    let mut mono = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymError::IoError(_)) => break,
            Err(e) => return Err(format!("read error: {e}")),
        };
        if packet.track_id != track_id {
            continue;
        }
        let n = dec
            .decode_float(&packet.data, &mut pcm, false)
            .map_err(|e| format!("opus: {e}"))?;
        let inv = 1.0 / channels as f32;
        mono.extend(
            pcm[..n * channels]
                .chunks_exact(channels)
                .map(|f| f.iter().sum::<f32>() * inv),
        );
    }
    mono.drain(..pre_skip.min(mono.len()));
    Ok(Audio {
        samples: mono,
        rate: 48_000,
    })
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Modified Bessel function I0 (Kaiser window).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let y = x * x / 4.0;
    for k in 1..50 {
        term *= y / (k * k) as f64;
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

/// Rational resampler: Kaiser-windowed sinc, polyphase, cutoff at 0.94 of the
/// lower Nyquist frequency (> 90 dB stop-band rejection).
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let g = gcd(from, to);
    let (up, down) = ((to / g) as usize, (from / g) as usize);
    // Filter designed at the upsampled rate `from * up`.
    let cutoff = 0.94 * 0.5 / up.max(down) as f64; // cycles per upsampled sample
                                                   // ~192 input taps per output sample: transition band ≈ 5 % of Nyquist.
    let half = 96 * up.max(down);
    let beta = 9.0;
    let i0b = bessel_i0(beta);
    let len = 2 * half + 1;
    let h: Vec<f64> = (0..len)
        .map(|i| {
            let n = i as f64 - half as f64;
            let sinc = if n == 0.0 {
                2.0 * cutoff
            } else {
                (2.0 * PI * cutoff * n).sin() / (PI * n)
            };
            let r = n / half as f64;
            let w = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b;
            sinc * w * up as f64
        })
        .collect();
    let out_len = (x.len() * up).div_ceil(down);
    let mut y = vec![0f32; out_len];
    for (o, yo) in y.iter_mut().enumerate() {
        // Output sample o sits at upsampled index o*down; input samples at k*up.
        let t = o * down;
        // Input indices k with |t - k*up| <= half.
        let k_lo = (t.saturating_sub(half)).div_ceil(up);
        let k_hi = ((t + half) / up).min(x.len() - 1);
        let mut acc = 0.0f64;
        let mut k = k_lo;
        while k <= k_hi {
            let idx = t as isize - (k * up) as isize + half as isize;
            acc += h[idx as usize] * x[k] as f64;
            k += 1;
        }
        *yo = acc as f32;
    }
    y
}

/// RMS level in dBFS of 20 ms frames.
fn frame_db(x: &[f32], rate: u32) -> Vec<f32> {
    let n = (rate as usize / 50).max(1);
    x.chunks(n)
        .map(|c| {
            let e = c.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / c.len() as f64;
            (10.0 * (e + 1e-12).log10()) as f32
        })
        .collect()
}

/// Removes leading/trailing silence (frames 40 dB below the loudest
/// frame), keeping `pad_ms` of margin.
pub fn trim(x: &[f32], rate: u32, pad_ms: u32) -> &[f32] {
    let db = frame_db(x, rate);
    let Some(peak) = db.iter().copied().reduce(f32::max) else {
        return x;
    };
    let thr = (peak - 40.0).max(-60.0);
    let n = (rate as usize / 50).max(1);
    let first = db.iter().position(|d| *d > thr).unwrap_or(0);
    let last = db
        .iter()
        .rposition(|d| *d > thr)
        .unwrap_or(db.len().saturating_sub(1));
    let pad = (rate * pad_ms / 1000) as usize;
    let start = (first * n).saturating_sub(pad);
    let end = ((last + 1) * n + pad).min(x.len());
    if start >= end {
        x
    } else {
        &x[start..end]
    }
}

/// Speech level estimate: mean power of the loudest 70 % of 20 ms frames.
pub fn speech_db(x: &[f32], rate: u32) -> f32 {
    let mut db = frame_db(x, rate);
    if db.is_empty() {
        return -100.0;
    }
    db.sort_by(|a, b| b.total_cmp(a));
    let keep = (db.len() * 7 / 10).max(1);
    let p: f64 = db[..keep]
        .iter()
        .map(|d| 10f64.powf(*d as f64 / 10.0))
        .sum::<f64>()
        / keep as f64;
    (10.0 * p.log10()) as f32
}

/// Scales speech to `target_db` (dBFS of the active frames), limiting peaks
/// to -1 dBFS.
pub fn normalize(x: &mut [f32], rate: u32, target_db: f32) {
    let level = speech_db(x, rate);
    let mut gain = 10f32.powf((target_db - level) / 20.0);
    let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let max_gain = 0.891 / peak.max(1e-9);
    gain = gain.min(max_gain);
    for v in x.iter_mut() {
        *v *= gain;
    }
}

/// Fraction of spectral energy above `hz` (bandwidth check for corpora that
/// were upsampled from narrow-band recordings).
pub fn energy_above(x: &[f32], rate: u32, hz: f32) -> f32 {
    let n = 1024;
    if x.len() < n {
        return 0.0;
    }
    let bin = (hz / rate as f32 * n as f32) as usize;
    let mut hi = 0f64;
    let mut tot = 0f64;
    let win: Vec<f64> = (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos())
        .collect();
    let mut frame = vec![0f64; n];
    for start in (0..x.len() - n).step_by(n / 2) {
        for i in 0..n {
            frame[i] = x[start + i] as f64 * win[i];
        }
        // Direct DFT on the bins we need is too slow; use a radix-2 FFT.
        let spec = fft_power(&frame);
        for (k, p) in spec.iter().enumerate() {
            tot += p;
            if k >= bin {
                hi += p;
            }
        }
    }
    (hi / tot.max(1e-30)) as f32
}

/// Median spectral rolloff in Hz: the frequency below which `share` of each
/// frame's spectral magnitude lies (as librosa computes it), over the frames
/// within 50 dB of the loudest. Full-band speech has a 99 % rolloff above
/// 10 kHz; narrow-band ("telephone") recordings stop near 6–7 kHz.
pub fn rolloff(x: &[f32], rate: u32, share: f64) -> f32 {
    let n = 2048;
    if x.len() < n {
        return 0.0;
    }
    let win: Vec<f64> = (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos())
        .collect();
    // Every fourth frame of a hop-512 analysis is plenty for a median.
    let starts: Vec<usize> = (0..x.len() - n).step_by(2048).collect();
    let energy: Vec<f64> = starts
        .iter()
        .map(|&s| x[s..s + n].iter().map(|v| (*v as f64).powi(2)).sum::<f64>())
        .collect();
    let peak = energy.iter().cloned().fold(0.0, f64::max);
    let mut frame = vec![0f64; n];
    let mut freqs: Vec<f32> = Vec::new();
    for (&s, &e) in starts.iter().zip(&energy) {
        if e <= peak * 1e-5 {
            continue;
        }
        for i in 0..n {
            frame[i] = x[s + i] as f64 * win[i];
        }
        let mag: Vec<f64> = fft_power(&frame).iter().map(|p| p.sqrt()).collect();
        let total: f64 = mag.iter().sum();
        let mut acc = 0.0;
        let k = mag
            .iter()
            .position(|m| {
                acc += m;
                acc >= share * total
            })
            .unwrap_or(mag.len() - 1);
        freqs.push(k as f32 * rate as f32 / n as f32);
    }
    if freqs.is_empty() {
        return 0.0;
    }
    freqs.sort_by(f32::total_cmp);
    freqs[freqs.len() / 2]
}

fn fft_power(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut re = x.to_vec();
    let mut im = vec![0f64; n];
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
        }
    }
    // Twiddles of the largest stage; stage `len` uses every (n/len)-th.
    let tw: Vec<(f64, f64)> = (0..n / 2)
        .map(|k| (-2.0 * PI * k as f64 / n as f64).sin_cos())
        .collect();
    let mut len = 2;
    while len <= n {
        let stride = n / len;
        for i in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = tw[k * stride];
                let (ar, ai) = (re[i + k + len / 2], im[i + k + len / 2]);
                let (tr, ti) = (ar * c - ai * s, ar * s + ai * c);
                re[i + k + len / 2] = re[i + k] - tr;
                im[i + k + len / 2] = im[i + k] - ti;
                re[i + k] += tr;
                im[i + k] += ti;
            }
        }
        len <<= 1;
    }
    (0..=n / 2).map(|k| re[k] * re[k] + im[k] * im[k]).collect()
}

/// Writes 16-bit PCM WAV.
pub fn write_wav(path: &Path, x: &[f32], rate: u32) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).map_err(|e| e.to_string())?;
    for v in x {
        w.write_sample((v.clamp(-1.0, 1.0) * 32767.0).round() as i16)
            .map_err(|e| e.to_string())?;
    }
    w.finalize().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f64, rate: u32, secs: f64) -> Vec<f32> {
        (0..(rate as f64 * secs) as usize)
            .map(|i| (2.0 * PI * hz * i as f64 / rate as f64).sin() as f32 * 0.5)
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn rolloff_separates_narrow_from_full_band() {
        // A tone sits at its own frequency; white noise spreads to Nyquist.
        let r = rolloff(&tone(5000.0, 48_000, 2.0), 48_000, 0.99);
        assert!((r - 5000.0).abs() < 100.0, "tone rolloff {r}");
        let mut seed = 1u32;
        let noise: Vec<f32> = (0..96_000)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
            })
            .collect();
        let r = rolloff(&noise, 48_000, 0.99);
        assert!(r > 23_000.0, "noise rolloff {r}");
        // The same noise band-limited to 6 kHz (telephone-like).
        let narrow = resample(&resample(&noise, 48_000, 12_000), 12_000, 48_000);
        let r = rolloff(&narrow, 48_000, 0.99);
        assert!(r < 7_000.0, "narrow-band rolloff {r}");
    }

    #[test]
    fn resampler_keeps_passband_and_rejects_aliases() {
        for (from, to) in [
            (48_000, 24_000),
            (44_100, 24_000),
            (22_050, 24_000),
            (16_000, 24_000),
        ] {
            // 1 kHz passes unchanged in level.
            let y = resample(&tone(1000.0, from, 0.5), from, to);
            let mid = &y[y.len() / 4..3 * y.len() / 4];
            assert!(
                (rms(mid) - 0.3536).abs() < 0.01,
                "{from}->{to}: {}",
                rms(mid)
            );
            // A tone above the output Nyquist frequency is removed.
            if from > to {
                let y = resample(&tone(to as f64 * 0.52, from, 0.5), from, to);
                let mid = &y[y.len() / 4..3 * y.len() / 4];
                assert!(rms(mid) < 0.001, "{from}->{to} alias {}", rms(mid));
            }
        }
    }

    #[test]
    fn trim_and_normalize() {
        let rate = 24_000;
        let mut x = vec![0f32; rate as usize];
        x.extend(tone(200.0, rate, 1.0));
        x.extend(vec![0f32; rate as usize]);
        let t = trim(&x, rate, 100);
        assert!(t.len() < 30_000 && t.len() > 24_000, "{}", t.len());
        let mut y = t.to_vec();
        normalize(&mut y, rate, -20.0);
        assert!((speech_db(&y, rate) + 20.0).abs() < 0.5);
    }
}
