//! Adversarial training loop (generator and discriminator steps), logging,
//! evaluation samples and checkpoints.

use crate::config::Config;
use crate::data::{Dataset, Rng, Tables};
use crate::disc::{disc_loss, feature_loss, gen_loss, Discriminator};
use crate::model::{Consts, Synthesizer};
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::{Device, Int, Tensor, TensorData};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct Run {
    pub dir: PathBuf,
    pub cfg: Config,
    pub tables: Tables,
}

fn scalar(t: Tensor<1>) -> f32 {
    t.into_data()
        .try_to_vec::<f32>()
        .map(|v| v[0])
        .unwrap_or(f32::NAN)
}

/// Writes 16-bit PCM.
pub fn write_wav(path: &Path, x: &[f32], rate: u32) {
    let _ = inhouse_tts_data::audio::write_wav(path, x, rate);
}

pub fn save(
    net: &Synthesizer,
    disc: &Discriminator,
    dir: &Path,
    step: usize,
) -> Result<(), String> {
    let mut s = SafetensorsStore::from_file(dir.join("generator.safetensors"));
    net.save_into(&mut s)
        .map_err(|e| format!("save generator: {e:?}"))?;
    let mut s = SafetensorsStore::from_file(dir.join("discriminator.safetensors"));
    disc.save_into(&mut s)
        .map_err(|e| format!("save discriminator: {e:?}"))?;
    fs::write(dir.join("step"), step.to_string()).map_err(|e| e.to_string())
}

pub fn load_generator(net: &mut Synthesizer, dir: &Path) -> Result<(), String> {
    let mut s = SafetensorsStore::from_file(dir.join("generator.safetensors"));
    net.load_from(&mut s)
        .map(|_| ())
        .map_err(|e| format!("load generator: {e:?}"))
}

/// Synthesises evaluation sentences for every speaker into `dir`.
pub fn eval(
    net: &Synthesizer,
    run: &Run,
    k: &Consts,
    dev: &Device,
    step: usize,
    sentences: &[(String, String)],
) {
    let out = run.dir.join("samples").join(format!("{step:07}"));
    let _ = fs::create_dir_all(&out);
    for (si, spk) in run.tables.speakers.iter().enumerate().take(12) {
        let loc = run
            .tables
            .speaker_locale
            .get(spk)
            .cloned()
            .unwrap_or_default();
        let Some((_, ph)) = sentences.iter().find(|(l, _)| *l == loc) else {
            continue;
        };
        let ids = run.tables.encode(ph);
        let n = ids.len();
        let ids = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, n]), dev);
        let wav = net.infer(ids, si, run.tables.locale(&loc), k, 0.667, 1.0);
        let name = spk.replace([':', '/'], "_");
        write_wav(
            &out.join(format!("{name}.wav")),
            &wav,
            run.cfg.audio.sample_rate,
        );
    }
}

pub fn train(
    run: Run,
    data: Dataset,
    dev: Device,
    steps: usize,
    resume: bool,
    sentences: Vec<(String, String)>,
) -> Result<(), String> {
    let cfg = &run.cfg;
    let k = Consts::new(cfg, &dev);
    let mut net = Synthesizer::new(
        &dev,
        cfg,
        run.tables.symbols.len(),
        run.tables.speakers.len(),
        run.tables.locales.len(),
    );
    let mut disc = Discriminator::new(&dev, &cfg.model.periods, cfg.model.disc_scale);
    let mut start = 0;
    if resume && run.dir.join("generator.safetensors").exists() {
        load_generator(&mut net, &run.dir)?;
        let mut s = SafetensorsStore::from_file(run.dir.join("discriminator.safetensors"));
        disc.load_from(&mut s)
            .map_err(|e| format!("load discriminator: {e:?}"))?;
        start = fs::read_to_string(run.dir.join("step"))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        eprintln!("resumed at step {start}");
    }
    let t = &cfg.train;
    let mut opt_g = AdamWConfig::new()
        .with_beta_1(t.betas.0)
        .with_beta_2(t.betas.1)
        .with_epsilon(1e-9)
        .with_weight_decay(0.01)
        .init();
    let mut opt_d = AdamWConfig::new()
        .with_beta_1(t.betas.0)
        .with_beta_2(t.betas.1)
        .with_epsilon(1e-9)
        .with_weight_decay(0.01)
        .init();
    let mut rng = Rng::new(t.seed + start as u64);
    let mut log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(run.dir.join("train.log"))
        .map_err(|e| e.to_string())?;
    let hop = cfg.audio.hop;
    let seg = t.segment_frames;
    let t0 = Instant::now();
    eprintln!(
        "training {} on {} utterances ({:.2} h), {} speakers, {} locales",
        cfg.name,
        data.len(),
        data.hours(),
        run.tables.speakers.len(),
        run.tables.locales.len()
    );
    for step in start..steps {
        // Learning rate decays once every 1000 steps (≈ per epoch of VITS).
        let lr = t.lr * t.lr_decay.powf(step as f64 / 1000.0 * 8.0);
        let batch = data.batch(&mut rng, t.batch, &dev);
        let out = net.forward_train(&batch, seg, &k, t.seed ^ step as u64);
        // Matching windows of the target waveform.
        let parts: Vec<Tensor<3>> = out
            .slice_starts
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                batch
                    .wav
                    .clone()
                    .narrow(0, i, 1)
                    .narrow(2, s * hop, seg * hop)
            })
            .collect();
        let y = Tensor::cat(parts, 0);
        let y_hat = out.y_hat;

        // Discriminator step.
        let d_real = disc.forward(y.clone());
        let d_fake = disc.forward(y_hat.clone().detach());
        let l_d = disc_loss(&d_real, &d_fake);
        let grads = l_d.backward();
        let gd = GradientsParams::from_grads(grads, &disc);
        disc = opt_d.step(lr, disc, gd);

        // Generator step.
        let d_real = disc.forward(y.clone());
        let d_fake = disc.forward(y_hat.clone());
        let l_adv = gen_loss(&d_fake);
        let l_fm = feature_loss(&d_real, &d_fake) * t.c_fm as f64;
        let l_mel = (k.mel(y) - k.mel(y_hat)).abs().mean() * t.c_mel as f64;
        let l_kl = out.l_kl * t.c_kl as f64;
        let l_g = l_adv.clone() + l_fm.clone() + l_mel.clone() + l_kl.clone() + out.l_dur.clone();
        let grads = l_g.backward();
        let gg = GradientsParams::from_grads(grads, &net);
        net = opt_g.step(lr, net, gg);

        if step % t.log_every == 0 || step + 1 == steps {
            let line = format!(
                "step {step} {:.0}s lr {lr:.2e} d {:.3} adv {:.3} fm {:.3} mel {:.3} kl {:.3} dur {:.3}",
                t0.elapsed().as_secs_f64(),
                scalar(l_d),
                scalar(l_adv),
                scalar(l_fm),
                scalar(l_mel),
                scalar(l_kl),
                scalar(out.l_dur),
            );
            eprintln!("{line}");
            let _ = writeln!(log, "{line}");
        }
        if (step + 1) % t.save_every == 0 || step + 1 == steps {
            save(&net, &disc, &run.dir, step + 1)?;
        }
        if (step + 1) % t.eval_every == 0 || step + 1 == steps {
            eval(&net, &run, &k, &dev, step + 1, &sentences);
        }
    }
    let _ = (net.num_params(), disc.num_params());
    Ok(())
}
