//! Building blocks on `[batch, channels, time]` tensors.

use burn::module::{Module, Param};
use burn::tensor::module::{conv1d, conv2d, conv_transpose1d, embedding};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{activation, Device, Distribution, Int, Tensor};

fn normal<const D: usize>(shape: [usize; D], std: f64, dev: &Device) -> Tensor<D> {
    Tensor::random(shape, Distribution::Normal(0.0, std), dev)
}

/// Per-output-channel L2 norm of a weight tensor.
fn wn_norm<const D: usize>(v: &Tensor<D>) -> Tensor<D> {
    let mut n = v.clone().powi_scalar(2);
    for d in 1..D {
        n = n.sum_dim(d);
    }
    (n + 1e-12).sqrt()
}

/// Weight-normalised 1-D convolution (`w = g · v / ‖v‖`), optional.
#[derive(Module, Debug)]
pub struct Conv {
    pub v: Param<Tensor<3>>,
    pub g: Option<Param<Tensor<3>>>,
    pub b: Option<Param<Tensor<1>>>,
    pub stride: usize,
    pub pad: usize,
    pub dil: usize,
    pub groups: usize,
}

impl Conv {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dev: &Device,
        cin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dil: usize,
        groups: usize,
        weight_norm: bool,
        std: f64,
    ) -> Conv {
        let v = normal([cout, cin / groups, k], std, dev);
        let g = weight_norm.then(|| Param::from_tensor(wn_norm(&v).detach()));
        Conv {
            v: Param::from_tensor(v),
            g,
            b: Some(Param::from_tensor(Tensor::zeros([cout], dev))),
            stride,
            pad,
            dil,
            groups,
        }
    }

    /// "Same" convolution with odd kernel `k` and dilation `dil`.
    pub fn same(dev: &Device, cin: usize, cout: usize, k: usize, dil: usize, wn: bool) -> Conv {
        let std = if wn {
            0.01
        } else {
            (1.0 / (cin * k) as f64).sqrt()
        };
        Conv::new(dev, cin, cout, k, 1, dil * (k - 1) / 2, dil, 1, wn, std)
    }

    /// 1×1 convolution.
    pub fn pointwise(dev: &Device, cin: usize, cout: usize) -> Conv {
        Conv::new(
            dev,
            cin,
            cout,
            1,
            1,
            0,
            1,
            1,
            false,
            (1.0 / cin as f64).sqrt(),
        )
    }

    /// Zero-initialised 1×1 convolution (flow outputs).
    pub fn zeros(dev: &Device, cin: usize, cout: usize) -> Conv {
        let mut c = Conv::pointwise(dev, cin, cout);
        c.v = Param::from_tensor(Tensor::zeros([cout, cin, 1], dev));
        c
    }

    pub fn weight(&self) -> Tensor<3> {
        let v = self.v.val();
        match &self.g {
            Some(g) => v.clone() * (g.val() / wn_norm(&v)),
            None => v,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        conv1d(
            x,
            self.weight(),
            self.b.as_ref().map(|b| b.val()),
            ConvOptions::new([self.stride], [self.pad], [self.dil], self.groups),
        )
    }
}

/// Weight-normalised transposed convolution (HiFi-GAN upsampler).
#[derive(Module, Debug)]
pub struct ConvT {
    pub v: Param<Tensor<3>>,
    pub g: Param<Tensor<3>>,
    pub b: Param<Tensor<1>>,
    pub stride: usize,
    pub pad: usize,
}

impl ConvT {
    pub fn new(dev: &Device, cin: usize, cout: usize, k: usize, stride: usize) -> ConvT {
        // PyTorch layout [cin, cout, k]; weight norm over dims 1, 2 (dim=0).
        let v = normal([cin, cout, k], 0.01, dev);
        ConvT {
            g: Param::from_tensor(wn_norm(&v).detach()),
            v: Param::from_tensor(v),
            b: Param::from_tensor(Tensor::zeros([cout], dev)),
            stride,
            pad: (k - stride) / 2,
        }
    }

    pub fn weight(&self) -> Tensor<3> {
        let v = self.v.val();
        v.clone() * (self.g.val() / wn_norm(&v))
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        conv_transpose1d(
            x,
            self.weight(),
            Some(self.b.val()),
            ConvTransposeOptions::new([self.stride], [self.pad], [0], [1], 1),
        )
    }
}

/// Weight-normalised 2-D convolution with `(k, 1)` kernels (period
/// discriminator).
#[derive(Module, Debug)]
pub struct Conv2 {
    pub v: Param<Tensor<4>>,
    pub g: Param<Tensor<4>>,
    pub b: Param<Tensor<1>>,
    pub stride: usize,
    pub pad: usize,
}

impl Conv2 {
    pub fn new(dev: &Device, cin: usize, cout: usize, k: usize, stride: usize) -> Conv2 {
        let v = normal([cout, cin, k, 1], (1.0 / (cin * k) as f64).sqrt(), dev);
        Conv2 {
            g: Param::from_tensor(wn_norm(&v).detach()),
            v: Param::from_tensor(v),
            b: Param::from_tensor(Tensor::zeros([cout], dev)),
            stride,
            pad: (k - 1) / 2,
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let v = self.v.val();
        let w = v.clone() * (self.g.val() / wn_norm(&v));
        conv2d(
            x,
            w,
            Some(self.b.val()),
            ConvOptions::new([self.stride, 1], [self.pad, 0], [1, 1], 1),
        )
    }
}

/// Layer normalisation over channels of `[B, C, T]`.
#[derive(Module, Debug)]
pub struct ChannelNorm {
    pub gamma: Param<Tensor<1>>,
    pub beta: Param<Tensor<1>>,
}

impl ChannelNorm {
    pub fn new(dev: &Device, c: usize) -> ChannelNorm {
        ChannelNorm {
            gamma: Param::from_tensor(Tensor::ones([c], dev)),
            beta: Param::from_tensor(Tensor::zeros([c], dev)),
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let c = self.gamma.dims()[0];
        let mean = x.clone().mean_dim(1);
        let xc = x - mean;
        let var = xc.clone().powi_scalar(2).mean_dim(1);
        let y = xc / (var + 1e-5).sqrt();
        y * self.gamma.val().reshape([1, c, 1]) + self.beta.val().reshape([1, c, 1])
    }
}

/// Embedding table.
#[derive(Module, Debug)]
pub struct Embed {
    pub w: Param<Tensor<2>>,
}

impl Embed {
    pub fn new(dev: &Device, n: usize, d: usize, std: f64) -> Embed {
        Embed {
            w: Param::from_tensor(normal([n, d], std, dev)),
        }
    }

    /// `[B, T]` ids → `[B, T, D]`.
    pub fn forward(&self, ids: Tensor<2, Int>) -> Tensor<3> {
        embedding(self.w.val(), ids)
    }

    /// `[B]` ids → `[B, D, 1]` (global conditioning).
    pub fn global(&self, ids: Tensor<1, Int>) -> Tensor<3> {
        let [b] = ids.dims();
        let e = embedding(self.w.val(), ids.reshape([b, 1]));
        e.swap_dims(1, 2)
    }
}

/// Non-causal WaveNet stack with gated activations and global
/// conditioning (VITS `WN`).
#[derive(Module, Debug)]
pub struct WaveNet {
    pub ins: Vec<Conv>,
    pub res_skip: Vec<Conv>,
    pub cond: Option<Conv>,
    pub hidden: usize,
}

impl WaveNet {
    pub fn new(dev: &Device, hidden: usize, k: usize, layers: usize, gin: usize) -> WaveNet {
        let ins = (0..layers)
            .map(|_| Conv::same(dev, hidden, 2 * hidden, k, 1, true))
            .collect();
        let res_skip = (0..layers)
            .map(|i| {
                let out = if i + 1 < layers { 2 * hidden } else { hidden };
                let mut c = Conv::pointwise(dev, hidden, out);
                c.g = Some(Param::from_tensor(wn_norm(&c.v.val()).detach()));
                c
            })
            .collect();
        let cond = (gin > 0).then(|| Conv::pointwise(dev, gin, 2 * hidden * layers));
        WaveNet {
            ins,
            res_skip,
            cond,
            hidden,
        }
    }

    pub fn forward(&self, mut x: Tensor<3>, mask: Tensor<3>, g: Option<Tensor<3>>) -> Tensor<3> {
        let h = self.hidden;
        let layers = self.ins.len();
        let g = match (&self.cond, g) {
            (Some(c), Some(g)) => Some(c.forward(g)),
            _ => None,
        };
        let mut out: Option<Tensor<3>> = None;
        for i in 0..layers {
            let mut a = self.ins[i].forward(x.clone());
            if let Some(g) = &g {
                a = a + g.clone().narrow(1, 2 * h * i, 2 * h);
            }
            let t = activation::tanh(a.clone().narrow(1, 0, h));
            let s = activation::sigmoid(a.narrow(1, h, h));
            let acts = t * s;
            let rs = self.res_skip[i].forward(acts);
            if i + 1 < layers {
                x = (x + rs.clone().narrow(1, 0, h)) * mask.clone();
                let skip = rs.narrow(1, h, h);
                out = Some(match out {
                    Some(o) => o + skip,
                    None => skip,
                });
            } else {
                out = Some(match out {
                    Some(o) => o + rs,
                    None => rs,
                });
            }
        }
        out.unwrap() * mask
    }
}

/// HiFi-GAN residual block (type 1).
#[derive(Module, Debug)]
pub struct ResBlock {
    pub c1: Vec<Conv>,
    pub c2: Vec<Conv>,
}

impl ResBlock {
    pub fn new(dev: &Device, ch: usize, k: usize, dils: &[usize]) -> ResBlock {
        ResBlock {
            c1: dils
                .iter()
                .map(|&d| Conv::same(dev, ch, ch, k, d, true))
                .collect(),
            c2: dils
                .iter()
                .map(|_| Conv::same(dev, ch, ch, k, 1, true))
                .collect(),
        }
    }

    pub fn forward(&self, mut x: Tensor<3>) -> Tensor<3> {
        for (a, b) in self.c1.iter().zip(&self.c2) {
            let t = a.forward(activation::leaky_relu(x.clone(), 0.1));
            let t = b.forward(activation::leaky_relu(t, 0.1));
            x = x + t;
        }
        x
    }
}

/// Rotary position embedding for `[B, heads, T, d]`.
pub fn rope(x: Tensor<4>) -> Tensor<4> {
    let [b, h, t, d] = x.dims();
    let half = d / 2;
    let dev = x.device();
    let mut cos = Vec::with_capacity(t * half);
    let mut sin = Vec::with_capacity(t * half);
    for p in 0..t {
        for i in 0..half {
            let inv = 1.0 / 10_000f64.powf(2.0 * i as f64 / d as f64);
            let a = p as f64 * inv;
            cos.push(a.cos() as f32);
            sin.push(a.sin() as f32);
        }
    }
    let cos = Tensor::<1>::from_floats(cos.as_slice(), &dev).reshape([1, 1, t, half]);
    let sin = Tensor::<1>::from_floats(sin.as_slice(), &dev).reshape([1, 1, t, half]);
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.narrow(3, half, half);
    let r1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
    let r2 = x1 * sin + x2 * cos;
    let _ = (b, h);
    Tensor::cat(vec![r1, r2], 3)
}

/// Transformer encoder layer: RoPE self-attention and a convolutional
/// feed-forward network, post-norm (as in VITS).
#[derive(Module, Debug)]
pub struct EncoderLayer {
    pub qkv: Conv,
    pub out: Conv,
    pub norm1: ChannelNorm,
    pub ff1: Conv,
    pub ff2: Conv,
    pub norm2: ChannelNorm,
    pub heads: usize,
}

impl EncoderLayer {
    pub fn new(dev: &Device, hidden: usize, filter: usize, heads: usize) -> EncoderLayer {
        EncoderLayer {
            qkv: Conv::pointwise(dev, hidden, 3 * hidden),
            out: Conv::pointwise(dev, hidden, hidden),
            norm1: ChannelNorm::new(dev, hidden),
            ff1: Conv::same(dev, hidden, filter, 3, 1, false),
            ff2: Conv::same(dev, filter, hidden, 3, 1, false),
            norm2: ChannelNorm::new(dev, hidden),
            heads,
        }
    }

    /// `x` `[B, H, T]`, `mask` `[B, 1, T]`.
    pub fn forward(&self, x: Tensor<3>, mask: Tensor<3>, dropout: f64, train: bool) -> Tensor<3> {
        let [b, hd, t] = x.dims();
        let nh = self.heads;
        let d = hd / nh;
        let qkv = self.qkv.forward(x.clone() * mask.clone());
        let split = |i: usize| {
            qkv.clone()
                .narrow(1, i * hd, hd)
                .reshape([b, nh, d, t])
                .swap_dims(2, 3)
        };
        let (q, k, v) = (rope(split(0)), rope(split(1)), split(2));
        let scores = q.matmul(k.swap_dims(2, 3)) / (d as f64).sqrt();
        // Key mask: −∞ where the key position is padding.
        let km = mask.clone().reshape([b, 1, 1, t]);
        let scores = scores + (km - 1.0) * 1e4;
        let att = activation::softmax(scores, 3);
        let att = drop(att, dropout, train);
        let y = att.matmul(v).swap_dims(2, 3).reshape([b, hd, t]);
        let y = self.out.forward(y);
        let x = self.norm1.forward(x + drop(y, dropout, train));
        let f = self.ff1.forward(x.clone() * mask.clone());
        let f = drop(activation::relu(f), dropout, train);
        let f = self.ff2.forward(f * mask.clone());
        self.norm2.forward(x + drop(f, dropout, train)) * mask
    }
}

/// Dropout (inverted scaling) when training.
pub fn drop<const D: usize>(x: Tensor<D>, p: f64, train: bool) -> Tensor<D> {
    if !train || p <= 0.0 {
        return x;
    }
    let keep = Tensor::<D>::random(x.dims(), Distribution::Bernoulli(1.0 - p), &x.device());
    x * keep / (1.0 - p)
}
