//! Export of a trained IHV1 generator to the engine's `.ihw` container
//! (weight normalisation folded, posterior encoder dropped).

use crate::config::Config;
use crate::data::Tables;
use crate::model::Synthesizer;
use crate::nn::{ChannelNorm, Conv, ConvT, Embed, WaveNet};
use burn::tensor::Tensor;
use inhouse_tts_core::weights::{DType, Writer};

fn data<const D: usize>(t: Tensor<D>) -> (Vec<usize>, Vec<f32>) {
    let dims = t.dims().to_vec();
    let v = t.into_data().try_to_vec::<f32>().expect("f32 tensor");
    (dims, v)
}

struct Out {
    w: Writer,
    dtype: DType,
    params: usize,
}

impl Out {
    fn put<const D: usize>(&mut self, name: &str, t: Tensor<D>, big: bool) {
        let (shape, v) = data(t);
        self.params += v.len();
        let dt = if big { self.dtype } else { DType::F32 };
        self.w.add(name, &shape, &v, dt);
    }

    fn conv(&mut self, name: &str, c: &Conv) {
        let w = c.weight();
        let big = w.dims().iter().product::<usize>() >= 4096;
        self.put(&format!("{name}.weight"), w, big);
        if let Some(b) = &c.b {
            self.put(&format!("{name}.bias"), b.val(), false);
        }
    }

    fn conv_t(&mut self, name: &str, c: &ConvT) {
        self.put(&format!("{name}.weight"), c.weight(), true);
        self.put(&format!("{name}.bias"), c.b.val(), false);
    }

    fn norm(&mut self, name: &str, n: &ChannelNorm) {
        self.put(&format!("{name}.gamma"), n.gamma.val(), false);
        self.put(&format!("{name}.beta"), n.beta.val(), false);
    }

    fn embed(&mut self, name: &str, e: &Embed) {
        self.put(&format!("{name}.weight"), e.w.val(), false);
    }

    fn wavenet(&mut self, name: &str, wn: &WaveNet) {
        for (i, c) in wn.ins.iter().enumerate() {
            self.conv(&format!("{name}.in.{i}"), c);
        }
        for (i, c) in wn.res_skip.iter().enumerate() {
            self.conv(&format!("{name}.res_skip.{i}"), c);
        }
        if let Some(c) = &wn.cond {
            self.conv(&format!("{name}.cond"), c);
        }
    }
}

/// Serialises the inference part of `net`.
pub fn export(net: &Synthesizer, cfg: &Config, tables: &Tables, dtype: DType) -> (Vec<u8>, usize) {
    let meta = serde_json::json!({
        "config": cfg,
        "tables": tables,
        "sample_rate": cfg.audio.sample_rate,
    });
    let mut o = Out {
        w: Writer::new("ihv1", meta),
        dtype,
        params: 0,
    };
    let e = &net.enc_p;
    o.embed("enc.emb", &e.emb);
    o.embed("enc.lang", &e.lang);
    for (i, l) in e.layers.iter().enumerate() {
        let p = format!("enc.layers.{i}");
        o.conv(&format!("{p}.qkv"), &l.qkv);
        o.conv(&format!("{p}.out"), &l.out);
        o.norm(&format!("{p}.norm1"), &l.norm1);
        o.conv(&format!("{p}.ff1"), &l.ff1);
        o.conv(&format!("{p}.ff2"), &l.ff2);
        o.norm(&format!("{p}.norm2"), &l.norm2);
    }
    o.conv("enc.spk", &e.spk);
    o.conv("enc.proj", &e.proj);
    for (i, c) in net.flow.layers.iter().enumerate() {
        let p = format!("flow.{i}");
        o.conv(&format!("{p}.pre"), &c.pre);
        o.wavenet(&format!("{p}.wn"), &c.wn);
        o.conv(&format!("{p}.post"), &c.post);
    }
    let d = &net.dp;
    o.conv("dp.c1", &d.c1);
    o.norm("dp.n1", &d.n1);
    o.conv("dp.c2", &d.c2);
    o.norm("dp.n2", &d.n2);
    o.conv("dp.proj", &d.proj);
    o.conv("dp.cond", &d.cond);
    let g = &net.dec;
    o.conv("dec.pre", &g.pre);
    o.conv("dec.cond", &g.cond);
    for (i, u) in g.ups.iter().enumerate() {
        o.conv_t(&format!("dec.ups.{i}"), u);
    }
    for (i, r) in g.res.iter().enumerate() {
        for (j, c) in r.c1.iter().enumerate() {
            o.conv(&format!("dec.res.{i}.c1.{j}"), c);
        }
        for (j, c) in r.c2.iter().enumerate() {
            o.conv(&format!("dec.res.{i}.c2.{j}"), c);
        }
    }
    o.conv("dec.post", &g.post);
    o.embed("spk", &net.spk);
    let params = o.params;
    (o.w.finish(), params)
}
