//! Minimal, dependency-free reader for ONNX `ModelProto` files.
//!
//! Decodes just what the model converters need from the protobuf: the graph
//! initializers (name, dims, float32 / int64 data from `raw_data` or the
//! typed repeated fields) and the nodes (op type, name, inputs, outputs and
//! attributes). Other tensor element types are kept as
//! [`TensorData::Unsupported`] and only fail when their values are read;
//! tensors stored in external files are rejected.

use crate::weights::{err, Error, Result};

/// One decoded protobuf field value.
enum Wire<'a> {
    Varint(u64),
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32(u32),
}

/// Sequential reader of protobuf fields.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let Some(&b) = self.buf.get(self.pos) else {
                return err("onnx: truncated varint");
            };
            self.pos += 1;
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        err("onnx: varint too long")
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.buf.len() - self.pos < n {
            return err("onnx: truncated field");
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// Next `(field number, value)`, or `None` at the end of the message.
    fn next(&mut self) -> Result<Option<(u32, Wire<'a>)>> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = (key >> 3) as u32;
        let wire = match key & 7 {
            0 => Wire::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                Wire::Fixed64
            }
            2 => {
                let n = self.varint()? as usize;
                Wire::Bytes(self.take(n)?)
            }
            5 => Wire::Fixed32(u32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            w => return err(format!("onnx: unsupported wire type {w}")),
        };
        Ok(Some((field, wire)))
    }
}

fn string(w: &Wire) -> Result<String> {
    match w {
        Wire::Bytes(b) => {
            String::from_utf8(b.to_vec()).map_err(|_| Error("onnx: bad UTF-8".into()))
        }
        _ => err("onnx: expected a string"),
    }
}

/// Appends a repeated int64 field (packed or one value per field).
fn push_varints(w: &Wire, out: &mut Vec<i64>) -> Result<()> {
    match w {
        Wire::Varint(v) => out.push(*v as i64),
        Wire::Bytes(b) => {
            let mut r = Reader::new(b);
            while r.pos < b.len() {
                out.push(r.varint()? as i64);
            }
        }
        _ => return err("onnx: expected int64 values"),
    }
    Ok(())
}

/// Appends a repeated float field (packed or one value per field).
fn push_floats(w: &Wire, out: &mut Vec<f32>) -> Result<()> {
    match w {
        Wire::Fixed32(v) => out.push(f32::from_bits(*v)),
        Wire::Bytes(b) if b.len() % 4 == 0 => out.extend(
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
        ),
        _ => return err("onnx: expected float values"),
    }
    Ok(())
}

/// ONNX `TensorProto.DataType` codes this reader decodes.
pub const FLOAT: i32 = 1;
pub const INT64: i32 = 7;

/// Values of a tensor.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
    /// Element type the reader does not decode (ONNX data type code).
    Unsupported(i32),
}

/// A constant tensor (graph initializer or tensor attribute).
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub data: TensorData,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// The float32 values.
    pub fn floats(&self) -> Result<&[f32]> {
        match &self.data {
            TensorData::F32(v) => Ok(v),
            _ => err(format!("onnx: tensor {} is not float32", self.name)),
        }
    }

    /// The int64 values.
    pub fn ints(&self) -> Result<&[i64]> {
        match &self.data {
            TensorData::I64(v) => Ok(v),
            _ => err(format!("onnx: tensor {} is not int64", self.name)),
        }
    }

    fn parse(buf: &[u8]) -> Result<Tensor> {
        let mut r = Reader::new(buf);
        let mut name = String::new();
        let mut dims = Vec::new();
        let mut dtype = 0i32;
        let mut raw: Option<&[u8]> = None;
        let mut floats = Vec::new();
        let mut ints = Vec::new();
        let mut external = false;
        while let Some((field, w)) = r.next()? {
            match (field, &w) {
                (1, _) => push_varints(&w, &mut dims)?,
                (2, Wire::Varint(v)) => dtype = *v as i32,
                (4, _) => push_floats(&w, &mut floats)?,
                (7, _) => push_varints(&w, &mut ints)?,
                (8, _) => name = string(&w)?,
                (9, Wire::Bytes(b)) => raw = Some(b),
                (14, Wire::Varint(v)) => external = *v == 1,
                _ => {}
            }
        }
        if external {
            return err(format!("onnx: tensor {name} uses external data"));
        }
        if dims.iter().any(|&d| d < 0) {
            return err(format!("onnx: tensor {name} has negative dims"));
        }
        let dims: Vec<usize> = dims.into_iter().map(|d| d as usize).collect();
        let numel: usize = dims.iter().product();
        let data = match dtype {
            FLOAT => TensorData::F32(match raw {
                Some(b) => b
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                None => floats,
            }),
            INT64 => TensorData::I64(match raw {
                Some(b) => b
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                None => ints,
            }),
            other => TensorData::Unsupported(other),
        };
        let len = match &data {
            TensorData::F32(v) => v.len(),
            TensorData::I64(v) => v.len(),
            TensorData::Unsupported(_) => numel,
        };
        if len != numel {
            return err(format!(
                "onnx: tensor {name} has {len} values for dims {dims:?}"
            ));
        }
        Ok(Tensor { name, dims, data })
    }
}

/// Attribute value (the kinds the converters use; others are `Other`).
#[derive(Clone, Debug, PartialEq)]
pub enum AttrValue {
    Float(f32),
    Int(i64),
    String(Vec<u8>),
    Tensor(Tensor),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Other,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Attribute {
    pub name: String,
    pub value: AttrValue,
}

impl Attribute {
    fn parse(buf: &[u8]) -> Result<Attribute> {
        let mut r = Reader::new(buf);
        let mut name = String::new();
        let mut value = AttrValue::Other;
        let mut floats = Vec::new();
        let mut ints = Vec::new();
        let mut kind = 0u64;
        while let Some((field, w)) = r.next()? {
            match (field, &w) {
                (1, _) => name = string(&w)?,
                (2, Wire::Fixed32(v)) => value = AttrValue::Float(f32::from_bits(*v)),
                (3, Wire::Varint(v)) => value = AttrValue::Int(*v as i64),
                (4, Wire::Bytes(b)) => value = AttrValue::String(b.to_vec()),
                (5, Wire::Bytes(b)) => value = AttrValue::Tensor(Tensor::parse(b)?),
                (7, _) => push_floats(&w, &mut floats)?,
                (8, _) => push_varints(&w, &mut ints)?,
                (20, Wire::Varint(v)) => kind = *v,
                _ => {}
            }
        }
        // AttributeType: FLOATS = 6, INTS = 7 (also used when empty).
        if kind == 6 || !floats.is_empty() {
            value = AttrValue::Floats(floats);
        } else if kind == 7 || !ints.is_empty() {
            value = AttrValue::Ints(ints);
        }
        Ok(Attribute { name, value })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub op_type: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<Attribute>,
}

impl Node {
    fn parse(buf: &[u8]) -> Result<Node> {
        let mut r = Reader::new(buf);
        let mut n = Node {
            op_type: String::new(),
            name: String::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes: Vec::new(),
        };
        while let Some((field, w)) = r.next()? {
            match (field, &w) {
                (1, _) => n.inputs.push(string(&w)?),
                (2, _) => n.outputs.push(string(&w)?),
                (3, _) => n.name = string(&w)?,
                (4, _) => n.op_type = string(&w)?,
                (5, Wire::Bytes(b)) => n.attributes.push(Attribute::parse(b)?),
                _ => {}
            }
        }
        Ok(n)
    }

    pub fn attr(&self, name: &str) -> Option<&AttrValue> {
        self.attributes
            .iter()
            .find(|a| a.name == name)
            .map(|a| &a.value)
    }

    /// Integer list attribute (`dilations`, `pads`, `strides`, ...).
    pub fn ints(&self, name: &str) -> Option<&[i64]> {
        match self.attr(name)? {
            AttrValue::Ints(v) => Some(v),
            _ => None,
        }
    }

    pub fn int(&self, name: &str) -> Option<i64> {
        match self.attr(name)? {
            AttrValue::Int(v) => Some(*v),
            _ => None,
        }
    }
}

/// The parts of a `ModelProto` the converters use.
#[derive(Clone, Debug, Default)]
pub struct Model {
    pub producer: String,
    pub initializers: Vec<Tensor>,
    pub nodes: Vec<Node>,
}

impl Model {
    pub fn parse(bytes: &[u8]) -> Result<Model> {
        let mut m = Model::default();
        let mut graph = None;
        let mut r = Reader::new(bytes);
        while let Some((field, w)) = r.next()? {
            match (field, &w) {
                (2, _) => m.producer = string(&w)?,
                (7, Wire::Bytes(b)) => graph = Some(*b),
                _ => {}
            }
        }
        let Some(graph) = graph else {
            return err("onnx: model has no graph");
        };
        let mut r = Reader::new(graph);
        while let Some((field, w)) = r.next()? {
            match (field, &w) {
                (1, Wire::Bytes(b)) => m.nodes.push(Node::parse(b)?),
                (5, Wire::Bytes(b)) => m.initializers.push(Tensor::parse(b)?),
                _ => {}
            }
        }
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand-rolled protobuf encoder for the test messages.
    fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }

    fn key(field: u32, wire: u64, out: &mut Vec<u8>) {
        varint(((field as u64) << 3) | wire, out);
    }

    fn bytes(field: u32, b: &[u8], out: &mut Vec<u8>) {
        key(field, 2, out);
        varint(b.len() as u64, out);
        out.extend_from_slice(b);
    }

    fn int(field: u32, v: i64, out: &mut Vec<u8>) {
        key(field, 0, out);
        varint(v as u64, out);
    }

    #[test]
    fn parses_hand_encoded_model() {
        // Float tensor [2, 3] in raw_data, dims unpacked.
        let mut t1 = Vec::new();
        int(1, 2, &mut t1);
        int(1, 3, &mut t1);
        int(2, FLOAT as i64, &mut t1);
        bytes(8, b"w", &mut t1);
        let vals = [0.5f32, -1.0, 2.25, 3.0, -4.5, 1e-3];
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        bytes(9, &raw, &mut t1);
        // Int64 tensor [3] with packed int64_data (incl. a negative value).
        let mut t2 = Vec::new();
        let mut packed = Vec::new();
        varint(3, &mut packed);
        bytes(1, &packed, &mut t2);
        int(2, INT64 as i64, &mut t2);
        let mut data = Vec::new();
        for v in [7i64, -1, 300] {
            varint(v as u64, &mut data);
        }
        bytes(7, &data, &mut t2);
        bytes(8, b"ids", &mut t2);
        // Float tensor [2] with one fixed32 float_data entry per value.
        let mut t3 = Vec::new();
        int(1, 2, &mut t3);
        int(2, FLOAT as i64, &mut t3);
        for v in [1.5f32, -2.0] {
            key(4, 5, &mut t3);
            t3.extend_from_slice(&v.to_le_bytes());
        }
        bytes(8, b"f", &mut t3);
        // int32 tensor: kept but not decoded.
        let mut t4 = Vec::new();
        int(2, 6, &mut t4);
        bytes(8, b"i32", &mut t4);
        bytes(9, &5i32.to_le_bytes(), &mut t4);
        // Conv node with an ints attribute and an int attribute.
        let mut a1 = Vec::new();
        bytes(1, b"dilations", &mut a1);
        let mut p = Vec::new();
        varint(3, &mut p);
        bytes(8, &p, &mut a1);
        int(20, 7, &mut a1);
        let mut a2 = Vec::new();
        bytes(1, b"group", &mut a2);
        int(3, 1, &mut a2);
        int(20, 2, &mut a2);
        let mut node = Vec::new();
        bytes(1, b"x", &mut node);
        bytes(1, b"w", &mut node);
        bytes(2, b"y", &mut node);
        bytes(3, b"/enc/conv/Conv", &mut node);
        bytes(4, b"Conv", &mut node);
        bytes(5, &a1, &mut node);
        bytes(5, &a2, &mut node);
        let mut graph = Vec::new();
        bytes(1, &node, &mut graph);
        bytes(2, b"g", &mut graph);
        for t in [&t1, &t2, &t3, &t4] {
            bytes(5, t, &mut graph);
        }
        let mut model = Vec::new();
        int(1, 8, &mut model);
        bytes(2, b"pytorch", &mut model);
        bytes(7, &graph, &mut model);

        let m = Model::parse(&model).unwrap();
        assert_eq!(m.producer, "pytorch");
        assert_eq!(m.initializers.len(), 4);
        let w = &m.initializers[0];
        assert_eq!((w.name.as_str(), w.dims.clone()), ("w", vec![2, 3]));
        assert_eq!(w.floats().unwrap(), &vals);
        assert!(w.ints().is_err());
        let ids = &m.initializers[1];
        assert_eq!(ids.dims, vec![3]);
        assert_eq!(ids.ints().unwrap(), &[7, -1, 300]);
        assert_eq!(m.initializers[2].floats().unwrap(), &[1.5, -2.0]);
        assert_eq!(m.initializers[3].data, TensorData::Unsupported(6));
        assert!(m.initializers[3].floats().is_err());
        let n = &m.nodes[0];
        assert_eq!(n.op_type, "Conv");
        assert_eq!(n.name, "/enc/conv/Conv");
        assert_eq!(n.inputs, vec!["x", "w"]);
        assert_eq!(n.outputs, vec!["y"]);
        assert_eq!(n.ints("dilations"), Some(&[3i64][..]));
        assert_eq!(n.int("group"), Some(1));
        assert!(n.ints("pads").is_none());
        // Truncated input is an error, not a panic.
        assert!(Model::parse(&model[..model.len() - 3]).is_err());
    }
}
