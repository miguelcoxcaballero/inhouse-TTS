//! Model and training configuration (`config.json` in every run directory).

use serde::{Deserialize, Serialize};

/// Audio features. The hop must equal the decoder's total upsampling
/// (`ups` product × `istft_hop` × `subbands`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop: usize,
    pub n_mels: usize,
    pub fmax: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Text encoder width and feed-forward width.
    pub hidden: usize,
    pub filter: usize,
    pub heads: usize,
    pub enc_layers: usize,
    /// Latent channels of the posterior / prior.
    pub inter: usize,
    /// WaveNet layers of the posterior encoder.
    pub post_layers: usize,
    /// Coupling layers of the flow and WaveNet layers inside each.
    pub flows: usize,
    pub flow_layers: usize,
    /// Speaker (global) conditioning width.
    pub gin: usize,
    /// Decoder: initial channels, upsampling rates and kernels, resblocks.
    pub dec_channels: usize,
    pub ups: Vec<usize>,
    pub up_kernels: Vec<usize>,
    pub res_kernels: Vec<usize>,
    pub res_dilations: Vec<usize>,
    pub istft_n_fft: usize,
    pub istft_hop: usize,
    pub subbands: usize,
    /// Discriminator width multiplier (1.0 = HiFi-GAN size).
    pub disc_scale: f32,
    pub periods: Vec<usize>,
    pub dropout: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    pub batch: usize,
    /// Decoder training window in latent frames.
    pub segment_frames: usize,
    pub lr: f64,
    pub betas: (f32, f32),
    pub lr_decay: f64,
    pub c_mel: f32,
    pub c_kl: f32,
    pub c_fm: f32,
    pub max_frames: usize,
    pub log_every: usize,
    pub eval_every: usize,
    pub save_every: usize,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub name: String,
    pub audio: AudioConfig,
    pub model: ModelConfig,
    pub train: TrainConfig,
}

impl Config {
    /// Full-size model for GPU training (~30M parameters, decoder ≈ 7 GMAC
    /// per second of audio at inference).
    pub fn base() -> Config {
        Config {
            name: "ihv1-base".into(),
            audio: AudioConfig {
                sample_rate: 24_000,
                n_fft: 1024,
                hop: 256,
                n_mels: 80,
                fmax: 12_000.0,
            },
            model: ModelConfig {
                hidden: 192,
                filter: 768,
                heads: 2,
                enc_layers: 6,
                inter: 192,
                post_layers: 16,
                flows: 4,
                flow_layers: 4,
                gin: 256,
                dec_channels: 512,
                ups: vec![4, 4],
                up_kernels: vec![16, 16],
                res_kernels: vec![3, 7, 11],
                res_dilations: vec![1, 3, 5],
                istft_n_fft: 16,
                istft_hop: 4,
                subbands: 4,
                disc_scale: 1.0,
                periods: vec![2, 3, 5, 7, 11],
                dropout: 0.1,
            },
            train: TrainConfig {
                batch: 32,
                segment_frames: 32,
                lr: 2e-4,
                betas: (0.8, 0.99),
                lr_decay: 0.999_875,
                c_mel: 45.0,
                c_kl: 1.0,
                c_fm: 2.0,
                max_frames: 1000,
                log_every: 50,
                eval_every: 2000,
                save_every: 2000,
                seed: 1234,
            },
        }
    }

    /// Small model that trains on a CPU (prototype; much lower quality).
    pub fn tiny() -> Config {
        let mut c = Config::base();
        c.name = "ihv1-tiny".into();
        c.model.hidden = 96;
        c.model.filter = 384;
        c.model.enc_layers = 4;
        c.model.inter = 96;
        c.model.post_layers = 8;
        c.model.flows = 4;
        c.model.flow_layers = 2;
        c.model.gin = 64;
        c.model.dec_channels = 192;
        c.model.res_kernels = vec![3, 7];
        c.model.disc_scale = 0.25;
        c.model.periods = vec![2, 3, 5];
        c.train.batch = 8;
        c.train.segment_frames = 24;
        c.train.max_frames = 600;
        c.train.log_every = 10;
        c.train.eval_every = 500;
        c.train.save_every = 250;
        c
    }

    /// Mid-size model for a 16 GB GPU such as Colab's T4.
    pub fn small() -> Config {
        let mut c = Config::base();
        c.name = "ihv1-small".into();
        c.model.dec_channels = 384;
        c.model.disc_scale = 0.5;
        c.train.batch = 16;
        c.train.save_every = 1000;
        c.train.eval_every = 2000;
        c
    }

    pub fn preset(name: &str) -> Option<Config> {
        match name {
            "base" => Some(Config::base()),
            "small" => Some(Config::small()),
            "tiny" => Some(Config::tiny()),
            _ => None,
        }
    }

    /// Samples generated per latent frame by the decoder.
    pub fn decoder_hop(&self) -> usize {
        self.model.ups.iter().product::<usize>() * self.model.istft_hop * self.model.subbands
    }
}
