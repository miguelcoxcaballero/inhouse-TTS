//! K82 decoder and iSTFTNet generator on the GPU (~95 % of the model's
//! arithmetic). The model front (ALBERT, prosody LSTMs, harmonic source)
//! stays on the CPU and hands over a [`DecoderInput`]; the GPU returns the
//! `conv_post` spectrum and the CPU finishes with the inverse STFT.

use super::*;
use crate::gpu::{wgpu, Act, ConvGeom, ConvRun, ConvTRun, Gpu, GpuConv, Recorder, View};

/// `AdainResBlk1d` weights.
struct Blk {
    cin: usize,
    cout: usize,
    conv1: GpuConv,
    conv2: GpuConv,
    shortcut: GpuConv,
    pool: Option<wgpu::Buffer>,
    /// Offsets of `norm1` / `norm2` in the style buffer.
    s1: usize,
    s2: usize,
}

/// `AdaINResBlock1` weights.
struct Res {
    ch: usize,
    c1: Vec<GpuConv>,
    c2: Vec<GpuConv>,
    s1: Vec<usize>,
    s2: Vec<usize>,
}

/// The K82 decoder resident on a GPU.
pub struct K82Gpu {
    gpu: Gpu,
    encode: Blk,
    decode: Vec<Blk>,
    asr_res: GpuConv,
    noise_convs: Vec<GpuConv>,
    noise_res: Vec<Res>,
    ups: Vec<GpuConv>,
    resblocks: Vec<Res>,
    conv_post: GpuConv,
    style_len: usize,
}

const CAT: usize = 1024 + 64 + 2;

fn conv(gpu: &Gpu, c: &Conv1d, label: &str) -> crate::gpu::Result<GpuConv> {
    gpu.upload_conv(
        label,
        &[&c.w],
        c.bias.as_deref(),
        ConvGeom {
            cin: c.cin,
            taps: c.k,
            stride: c.stride,
            pad: c.pad,
            dil: c.dil,
        },
    )
}

fn conv_t(gpu: &Gpu, c: &ConvTranspose1d, label: &str) -> crate::gpu::Result<GpuConv> {
    let taps = c.phases[0].1;
    if c.phases.iter().any(|(_, t)| *t != taps) {
        return Err(format!("{label}: phases with different tap counts"));
    }
    let mats: Vec<&Matrix> = c.phases.iter().map(|(m, _)| m).collect();
    gpu.upload_conv(
        label,
        &mats,
        Some(&c.bias),
        ConvGeom {
            cin: c.cin,
            taps,
            stride: 1,
            pad: 0,
            dil: 1,
        },
    )
}

impl K82Gpu {
    /// Uploads the decoder of `m` (loaded with [`LoadOptions::gpu`]).
    pub fn new(gpu: Gpu, m: &K82) -> crate::gpu::Result<K82Gpu> {
        let d = &m.decoder;
        let mut style_len = 0;
        let mut slot = |c: usize| {
            let o = style_len;
            style_len += 3 * c;
            o
        };
        let mut blk = |b: &AdainResBlk, name: &str| -> crate::gpu::Result<Blk> {
            let cin = b.conv1.cin;
            let cout = b.conv1.cout;
            Ok(Blk {
                cin,
                cout,
                conv1: conv(&gpu, &b.conv1, name)?,
                conv2: conv(&gpu, &b.conv2, name)?,
                shortcut: conv(
                    &gpu,
                    b.shortcut
                        .as_ref()
                        .ok_or_else(|| format!("{name}: identity shortcut"))?,
                    name,
                )?,
                pool: b.pool.as_ref().map(|p| {
                    let mut w: Vec<f32> = p.w.iter().flatten().copied().collect();
                    w.extend_from_slice(&p.bias);
                    gpu.upload_f32(name, &w)
                }),
                s1: slot(cin),
                s2: slot(cout),
            })
        };
        let encode = blk(&d.encode, "encode")?;
        let decode = d
            .decode
            .iter()
            .enumerate()
            .map(|(i, b)| blk(b, &format!("decode.{i}")))
            .collect::<crate::gpu::Result<Vec<_>>>()?;
        let g = &d.generator;
        let mut res = |b: &AdaInResBlock1, name: &str| -> crate::gpu::Result<Res> {
            let ch = b.convs1[0].cin;
            Ok(Res {
                ch,
                c1: b
                    .convs1
                    .iter()
                    .map(|c| conv(&gpu, c, name))
                    .collect::<crate::gpu::Result<_>>()?,
                c2: b
                    .convs2
                    .iter()
                    .map(|c| conv(&gpu, c, name))
                    .collect::<crate::gpu::Result<_>>()?,
                s1: (0..b.convs1.len()).map(|_| slot(ch)).collect(),
                s2: (0..b.convs2.len()).map(|_| slot(ch)).collect(),
            })
        };
        let noise_res = g
            .noise_res
            .iter()
            .map(|b| res(b, "noise_res"))
            .collect::<crate::gpu::Result<Vec<_>>>()?;
        let resblocks = g
            .resblocks
            .iter()
            .map(|b| res(b, "resblocks"))
            .collect::<crate::gpu::Result<Vec<_>>>()?;
        Ok(K82Gpu {
            asr_res: conv(&gpu, &d.asr_res, "asr_res")?,
            noise_convs: g
                .noise_convs
                .iter()
                .map(|c| conv(&gpu, c, "noise_convs"))
                .collect::<crate::gpu::Result<_>>()?,
            ups: g
                .ups
                .iter()
                .map(|c| conv_t(&gpu, c, "ups"))
                .collect::<crate::gpu::Result<_>>()?,
            conv_post: conv(&gpu, &g.conv_post, "conv_post")?,
            encode,
            decode,
            noise_res,
            resblocks,
            style_len,
            gpu,
        })
    }

    /// Adapter description.
    pub fn name(&self) -> &str {
        &self.gpu.name
    }

    /// True if the adapter is a software rasteriser.
    pub fn software(&self) -> bool {
        self.gpu.software
    }

    /// Per-utterance AdaIN parameters `[γ | β | α]` of every layer.
    fn styles(&self, m: &K82, s: &[f32]) -> Vec<f32> {
        let mut st = vec![0f32; self.style_len];
        let mut put = |off: usize, a: &AdaIn, alpha: Option<&[f32]>| {
            let h = a.style(s);
            let c = h.len() / 2;
            st[off..off + 2 * c].copy_from_slice(&h);
            if let Some(al) = alpha {
                st[off + 2 * c..off + 3 * c].copy_from_slice(al);
            }
        };
        let d = &m.decoder;
        let blocks =
            std::iter::once((&self.encode, &d.encode)).chain(self.decode.iter().zip(&d.decode));
        for (g, b) in blocks {
            put(g.s1, &b.norm1, None);
            put(g.s2, &b.norm2, None);
        }
        let gen = &d.generator;
        let res = self
            .noise_res
            .iter()
            .zip(&gen.noise_res)
            .chain(self.resblocks.iter().zip(&gen.resblocks));
        for (g, b) in res {
            for i in 0..g.s1.len() {
                put(g.s1[i], &b.adain1[i], Some(&b.alpha1[i]));
                put(g.s2[i], &b.adain2[i], Some(&b.alpha2[i]));
            }
        }
        st
    }

    fn block(&self, r: &mut Recorder, b: &Blk, x: &View, y: &View, st: &wgpu::Buffer) {
        let h = r.alloc(b.cin, x.cols);
        r.adain(x, Some(&h), st, b.s1, Act::Leaky(0.2));
        let (h, up) = match &b.pool {
            Some(w) => {
                let hu = r.alloc(b.cin, 2 * x.cols);
                r.dwup2(&h, &hu, w);
                r.free(h);
                (hu, 2)
            }
            None => (h, 1),
        };
        let h2 = r.alloc(b.cout, h.cols);
        r.conv(&b.conv1, &h, &h2, ConvRun::default());
        r.free(h);
        r.adain(&h2, None, st, b.s2, Act::Leaky(0.2));
        r.conv(
            &b.shortcut,
            x,
            y,
            ConvRun {
                up,
                ..Default::default()
            },
        );
        r.conv(
            &b.conv2,
            &h2,
            y,
            ConvRun {
                accumulate: true,
                scale: std::f32::consts::FRAC_1_SQRT_2,
                ..Default::default()
            },
        );
        r.free(h2);
    }

    /// `AdaINResBlock1` applied in place to `s`.
    fn resblock(&self, r: &mut Recorder, g: &Res, s: &View, st: &wgpu::Buffer) {
        for i in 0..g.c1.len() {
            let a = r.alloc(g.ch, s.cols);
            r.adain(s, Some(&a), st, g.s1[i], Act::Snake);
            let b = r.alloc(g.ch, s.cols);
            r.conv(&g.c1[i], &a, &b, ConvRun::default());
            r.free(a);
            r.adain(&b, None, st, g.s2[i], Act::Snake);
            r.conv(
                &g.c2[i],
                &b,
                s,
                ConvRun {
                    accumulate: true,
                    ..Default::default()
                },
            );
            r.free(b);
        }
    }

    /// Decoder + generator: returns the 24 kHz waveform.
    pub async fn decode(
        &self,
        m: &K82,
        inp: &DecoderInput,
        style: &[f32],
    ) -> crate::gpu::Result<Vec<f32>> {
        let s_dec = &style[..STYLE_DIM];
        let d = &m.decoder;
        let f = inp.asr.cols;
        let f0m = d
            .f0_conv
            .forward(&Mat::from_vec(1, inp.f0.len(), inp.f0.clone()));
        let nm = d
            .n_conv
            .forward(&Mat::from_vec(1, inp.n.len(), inp.n.clone()));
        if f0m.cols != f || nm.cols != f {
            return Err("F0/energy length does not match the alignment".into());
        }
        let st = self.gpu.upload_f32("styles", &self.styles(m, s_dec));
        let mut r = self.gpu.recorder();
        let x0 = r.upload(&Mat::vcat(&[&inp.asr, &f0m, &nm]));
        let har = r.upload(&inp.har);
        let cat = [r.alloc(CAT, f), r.alloc(CAT, f)];
        for c in &cat {
            r.write(&c.rows(1088, 1), &f0m.data);
            r.write(&c.rows(1089, 1), &nm.data);
            r.conv(
                &self.asr_res,
                &x0.rows(0, HIDDEN),
                &c.rows(1024, 64),
                ConvRun::default(),
            );
        }
        self.block(&mut r, &self.encode, &x0, &cat[0].rows(0, 1024), &st);
        r.free(x0);
        for (i, b) in self.decode[..3].iter().enumerate() {
            self.block(&mut r, b, &cat[i % 2], &cat[1 - i % 2].rows(0, 1024), &st);
        }
        let mut x = r.alloc(self.decode[3].cout, 2 * f);
        self.block(&mut r, &self.decode[3], &cat[1], &x, &st);
        let [a, b] = cat;
        r.free(a);
        r.free(b);
        for i in 0..self.ups.len() {
            let nc = &self.noise_convs[i];
            let tx = nc.out_len(har.cols);
            let reflect = i + 1 == self.ups.len();
            let (k, u) = (UP_KERNELS[i], UP_RATES[i]);
            let pad = (k - u) / 2;
            if (x.cols - 1) * u + k - 2 * pad + usize::from(reflect) != tx {
                return Err("source/upsample length mismatch".into());
            }
            let xs = r.alloc(nc.m, tx);
            r.conv(nc, &har, &xs, ConvRun::default());
            self.resblock(&mut r, &self.noise_res[i], &xs, &st);
            r.conv_t(
                &self.ups[i],
                &x,
                &xs,
                ConvTRun {
                    kernel: k,
                    stride: u,
                    pad,
                    accumulate: true,
                    leaky_in: Some(0.1),
                    reflect_left: reflect,
                },
            );
            r.free(x);
            // Mean of the three multi-receptive-field branches.
            let acc = r.alloc(nc.m, tx);
            r.eltwise(&xs, &acc, false, 1.0);
            self.resblock(&mut r, &self.resblocks[3 * i], &acc, &st);
            let s2 = r.alloc(nc.m, tx);
            r.eltwise(&xs, &s2, false, 1.0);
            self.resblock(&mut r, &self.resblocks[3 * i + 1], &s2, &st);
            r.eltwise(&s2, &acc, true, 1.0);
            r.free(s2);
            self.resblock(&mut r, &self.resblocks[3 * i + 2], &xs, &st);
            r.eltwise(&xs, &acc, true, 1.0 / 3.0);
            r.free(xs);
            x = acc;
        }
        let y = r.alloc(self.conv_post.m, x.cols);
        r.conv(
            &self.conv_post,
            &x,
            &y,
            ConvRun {
                leaky_in: Some(0.01),
                ..Default::default()
            },
        );
        let frames = x.cols;
        let spec = r.run(&[&y]).await?.pop().unwrap_or_default();
        Ok(d.generator
            .spec_to_audio(&Mat::from_vec(self.conv_post.m, frames, spec)))
    }
}
