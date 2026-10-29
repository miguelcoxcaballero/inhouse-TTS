//! Multi-period and multi-scale discriminators (HiFi-GAN), with a width
//! multiplier for small configurations.

use crate::nn::{Conv, Conv2};
use burn::module::Module;
use burn::tensor::ops::PadMode;
use burn::tensor::{activation, Device, Tensor};

fn ch(c: usize, s: f32) -> usize {
    ((c as f32 * s).round() as usize).max(4)
}

#[derive(Module, Debug)]
pub struct PeriodDisc {
    pub convs: Vec<Conv2>,
    pub post: Conv2,
    pub period: usize,
}

impl PeriodDisc {
    fn new(dev: &Device, period: usize, s: f32) -> PeriodDisc {
        let widths = [
            1,
            ch(32, s),
            ch(128, s),
            ch(512, s),
            ch(1024, s),
            ch(1024, s),
        ];
        let convs = (0..5)
            .map(|i| Conv2::new(dev, widths[i], widths[i + 1], 5, if i < 4 { 3 } else { 1 }))
            .collect();
        PeriodDisc {
            convs,
            post: Conv2::new(dev, widths[5], 1, 3, 1),
            period,
        }
    }

    fn forward(&self, x: Tensor<3>, fmaps: &mut Vec<Tensor<4>>) -> Tensor<2> {
        let [b, c, t] = x.dims();
        let p = self.period;
        let x = if t % p != 0 {
            let n = p - t % p;
            x.pad([(0, n)], PadMode::Reflect)
        } else {
            x
        };
        let t2 = x.dims()[2];
        let mut x = x.reshape([b, c, t2 / p, p]);
        for conv in &self.convs {
            x = activation::leaky_relu(conv.forward(x), 0.1);
            fmaps.push(x.clone());
        }
        let x = self.post.forward(x);
        fmaps.push(x.clone());
        x.flatten(1, 3)
    }
}

#[derive(Module, Debug)]
pub struct ScaleDisc {
    pub convs: Vec<Conv>,
    pub post: Conv,
}

impl ScaleDisc {
    fn new(dev: &Device, s: f32) -> ScaleDisc {
        // (cin, cout, kernel, stride, groups, pad) of HiFi-GAN's DiscriminatorS.
        let spec = [
            (1, 16, 15, 1, 1, 7),
            (16, 64, 41, 4, 4, 20),
            (64, 256, 41, 4, 16, 20),
            (256, 1024, 41, 4, 64, 20),
            (1024, 1024, 41, 4, 256, 20),
            (1024, 1024, 5, 1, 1, 2),
        ];
        let convs = spec
            .iter()
            .map(|&(ci, co, k, st, g, pad)| {
                let ci = if ci == 1 { 1 } else { ch(ci, s) };
                let co = ch(co, s);
                let g = ((g as f32 * s).round() as usize).max(1).min(ci);
                let mut c = Conv::new(
                    dev,
                    ci,
                    co,
                    k,
                    st,
                    pad,
                    1,
                    g,
                    true,
                    (1.0 / (ci / g * k) as f64).sqrt(),
                );
                c.groups = g;
                c
            })
            .collect();
        ScaleDisc {
            convs,
            post: Conv::new(dev, ch(1024, s), 1, 3, 1, 1, 1, 1, true, 0.02),
        }
    }

    fn forward(&self, mut x: Tensor<3>, fmaps: &mut Vec<Tensor<3>>) -> Tensor<2> {
        for c in &self.convs {
            x = activation::leaky_relu(c.forward(x), 0.1);
            fmaps.push(x.clone());
        }
        let x = self.post.forward(x);
        fmaps.push(x.clone());
        x.flatten(1, 2)
    }
}

/// All discriminators: one scale discriminator and one per period.
#[derive(Module, Debug)]
pub struct Discriminator {
    pub scale: ScaleDisc,
    pub periods: Vec<PeriodDisc>,
}

/// Logits and feature maps of every sub-discriminator.
pub struct DiscOut {
    pub logits: Vec<Tensor<2>>,
    pub fmaps3: Vec<Tensor<3>>,
    pub fmaps4: Vec<Tensor<4>>,
}

impl Discriminator {
    pub fn new(dev: &Device, periods: &[usize], s: f32) -> Discriminator {
        Discriminator {
            scale: ScaleDisc::new(dev, s),
            periods: periods
                .iter()
                .map(|&p| PeriodDisc::new(dev, p, s))
                .collect(),
        }
    }

    pub fn forward(&self, y: Tensor<3>) -> DiscOut {
        let mut out = DiscOut {
            logits: Vec::new(),
            fmaps3: Vec::new(),
            fmaps4: Vec::new(),
        };
        out.logits
            .push(self.scale.forward(y.clone(), &mut out.fmaps3));
        for p in &self.periods {
            out.logits.push(p.forward(y.clone(), &mut out.fmaps4));
        }
        out
    }
}

/// LSGAN discriminator loss.
pub fn disc_loss(real: &DiscOut, fake: &DiscOut) -> Tensor<1> {
    let mut acc: Option<Tensor<1>> = None;
    for (r, f) in real.logits.iter().zip(&fake.logits) {
        let l = (r.clone() * -1.0 + 1.0).powi_scalar(2).mean() + f.clone().powi_scalar(2).mean();
        acc = Some(match acc {
            Some(a) => a + l,
            None => l,
        });
    }
    acc.unwrap()
}

/// LSGAN generator loss.
pub fn gen_loss(fake: &DiscOut) -> Tensor<1> {
    let mut acc: Option<Tensor<1>> = None;
    for f in &fake.logits {
        let l = (f.clone() * -1.0 + 1.0).powi_scalar(2).mean();
        acc = Some(match acc {
            Some(a) => a + l,
            None => l,
        });
    }
    acc.unwrap()
}

/// Feature-matching loss (real features detached).
pub fn feature_loss(real: &DiscOut, fake: &DiscOut) -> Tensor<1> {
    let mut acc: Option<Tensor<1>> = None;
    let mut add = |l: Tensor<1>| {
        acc = Some(match acc.take() {
            Some(a) => a + l,
            None => l,
        });
    };
    for (r, f) in real.fmaps3.iter().zip(&fake.fmaps3) {
        add((r.clone().detach() - f.clone()).abs().mean());
    }
    for (r, f) in real.fmaps4.iter().zip(&fake.fmaps4) {
        add((r.clone().detach() - f.clone()).abs().mean());
    }
    acc.unwrap()
}
