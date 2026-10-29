//! Minimal reader for PyTorch `torch.save` archives (zip + pickle).
//!
//! Supports the subset produced when saving state dicts and plain tensors:
//! STORED zip entries, pickle protocols 2–4, typed storages and
//! `torch._utils._rebuild_tensor_v2`. No code is ever executed: unknown
//! globals become opaque values.

use crate::gemm::f16_to_f32;
use crate::weights::{err, Error, Result};
use std::collections::HashMap;

/// A decoded tensor (always converted to f32, contiguous, row-major).
#[derive(Clone, Debug)]
pub struct TorchTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Pickle values produced by the restricted unpickler.
#[derive(Clone, Debug)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String),
    Storage { key: String, dtype: String },
    Tensor(TorchTensor),
    Object(String),
}

impl Value {
    fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }
}

struct Zip<'a> {
    bytes: &'a [u8],
    entries: HashMap<String, (usize, usize)>,
}

fn u16le(b: &[u8], o: usize) -> usize {
    u16::from_le_bytes([b[o], b[o + 1]]) as usize
}

fn u32le(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize
}

fn u64le(b: &[u8], o: usize) -> usize {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) as usize
}

impl<'a> Zip<'a> {
    fn open(bytes: &'a [u8]) -> Result<Self> {
        let min = bytes.len().saturating_sub(65_557);
        let eocd = (min..bytes.len().saturating_sub(21))
            .rev()
            .find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
            .ok_or_else(|| Error("not a zip archive".into()))?;
        let mut count = u16le(bytes, eocd + 10);
        let mut cd = u32le(bytes, eocd + 16);
        if cd == 0xffff_ffff || count == 0xffff {
            // ZIP64 end of central directory locator precedes the EOCD.
            let loc = eocd - 20;
            if bytes[loc..loc + 4] != [0x50, 0x4b, 0x06, 0x07] {
                return err("bad zip64 locator");
            }
            let rec = u64le(bytes, loc + 8);
            count = u64le(bytes, rec + 32);
            cd = u64le(bytes, rec + 48);
        }
        let mut entries = HashMap::new();
        let mut p = cd;
        for _ in 0..count {
            if bytes[p..p + 4] != [0x50, 0x4b, 0x01, 0x02] {
                return err("bad central directory");
            }
            let method = u16le(bytes, p + 10);
            let mut csize = u32le(bytes, p + 20);
            let name_len = u16le(bytes, p + 28);
            let extra_len = u16le(bytes, p + 30);
            let comment_len = u16le(bytes, p + 32);
            let mut local = u32le(bytes, p + 42);
            let name = String::from_utf8_lossy(&bytes[p + 46..p + 46 + name_len]).into_owned();
            // ZIP64 extra field (id 1) carries 64-bit sizes / offset.
            let mut e = p + 46 + name_len;
            let end = e + extra_len;
            while e + 4 <= end {
                let id = u16le(bytes, e);
                let sz = u16le(bytes, e + 2);
                if id == 1 {
                    let mut q = e + 4;
                    if u32le(bytes, p + 24) == 0xffff_ffff {
                        q += 8;
                    }
                    if csize == 0xffff_ffff {
                        csize = u64le(bytes, q);
                        q += 8;
                    }
                    if local == 0xffff_ffff {
                        local = u64le(bytes, q);
                    }
                }
                e += 4 + sz;
            }
            if method != 0 {
                return err(format!("zip entry {name} is compressed (method {method})"));
            }
            let lname = u16le(bytes, local + 26);
            let lextra = u16le(bytes, local + 28);
            let start = local + 30 + lname + lextra;
            entries.insert(name, (start, csize));
            p += 46 + name_len + extra_len + comment_len;
        }
        Ok(Zip { bytes, entries })
    }

    fn find(&self, suffix: &str) -> Option<&'a [u8]> {
        self.entries
            .iter()
            .find(|(n, _)| n.ends_with(suffix))
            .map(|(_, &(s, l))| &self.bytes[s..s + l])
    }

    fn prefix(&self) -> Option<String> {
        self.entries
            .keys()
            .find_map(|n| n.strip_suffix("data.pkl").map(str::to_string))
    }
}

struct Unpickler<'a, 'z> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    marks: Vec<usize>,
    memo: HashMap<usize, Value>,
    zip: &'z Zip<'z>,
    prefix: String,
}

impl Unpickler<'_, '_> {
    fn byte(&mut self) -> Result<u8> {
        let b = *self
            .data
            .get(self.pos)
            .ok_or_else(|| Error("pickle truncated".into()))?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&[u8]> {
        if self.pos + n > self.data.len() {
            return err("pickle truncated");
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn line(&mut self) -> Result<String> {
        let start = self.pos;
        while self.byte()? != b'\n' {}
        Ok(String::from_utf8_lossy(&self.data[start..self.pos - 1]).into_owned())
    }

    fn pop(&mut self) -> Result<Value> {
        self.stack
            .pop()
            .ok_or_else(|| Error("pickle stack underflow".into()))
    }

    fn pop_mark(&mut self) -> Result<Vec<Value>> {
        let m = self
            .marks
            .pop()
            .ok_or_else(|| Error("pickle mark underflow".into()))?;
        Ok(self.stack.split_off(m))
    }

    fn storage(&self, key: &str, dtype: &str) -> Result<Vec<f32>> {
        let raw = self
            .zip
            .find(&format!("{}data/{}", self.prefix, key))
            .ok_or_else(|| Error(format!("missing storage {key}")))?;
        Ok(match dtype {
            "FloatStorage" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            "HalfStorage" => raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            "BFloat16Storage" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            "DoubleStorage" => raw
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            "LongStorage" => raw
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            "IntStorage" => raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            other => return err(format!("unsupported storage type {other}")),
        })
    }

    fn rebuild_tensor(&self, args: &[Value]) -> Result<Value> {
        let (key, dtype) = match args.first() {
            Some(Value::Storage { key, dtype }) => (key.clone(), dtype.clone()),
            _ => return err("tensor without storage"),
        };
        let offset = args.get(1).and_then(Value::as_int).unwrap_or(0) as usize;
        let ints = |v: Option<&Value>| -> Vec<usize> {
            match v {
                Some(Value::Tuple(t)) | Some(Value::List(t)) => t
                    .iter()
                    .filter_map(Value::as_int)
                    .map(|i| i as usize)
                    .collect(),
                _ => Vec::new(),
            }
        };
        let shape = ints(args.get(2));
        let stride = ints(args.get(3));
        let storage = self.storage(&key, &dtype)?;
        let numel: usize = shape.iter().product();
        let mut data = vec![0f32; numel];
        // Gather with arbitrary strides.
        let mut idx = vec![0usize; shape.len()];
        for d in data.iter_mut() {
            let mut off = offset;
            for (i, s) in idx.iter().zip(&stride) {
                off += i * s;
            }
            *d = *storage
                .get(off)
                .ok_or_else(|| Error("tensor out of storage bounds".into()))?;
            for a in (0..shape.len()).rev() {
                idx[a] += 1;
                if idx[a] < shape[a] {
                    break;
                }
                idx[a] = 0;
            }
        }
        Ok(Value::Tensor(TorchTensor { shape, data }))
    }

    fn reduce(&self, func: Value, args: Value) -> Result<Value> {
        let args = match args {
            Value::Tuple(a) => a,
            other => vec![other],
        };
        let name = match func {
            Value::Global(n) => n,
            _ => return Ok(Value::Object("reduce".into())),
        };
        match name.as_str() {
            "torch._utils._rebuild_tensor_v2" | "torch._utils._rebuild_tensor" => {
                self.rebuild_tensor(&args)
            }
            "torch._utils._rebuild_parameter" | "torch._utils._rebuild_parameter_with_state" => {
                Ok(args.into_iter().next().unwrap_or(Value::None))
            }
            "collections.OrderedDict" | "builtins.dict" => Ok(Value::Dict(Vec::new())),
            _ => Ok(Value::Object(name)),
        }
    }

    fn run(&mut self) -> Result<Value> {
        loop {
            let op = self.byte()?;
            match op {
                0x80 => {
                    self.byte()?;
                }
                0x95 => {
                    self.take(8)?;
                }
                b'.' => return self.pop(),
                b'(' => self.marks.push(self.stack.len()),
                b'}' => self.stack.push(Value::Dict(Vec::new())),
                b']' => self.stack.push(Value::List(Vec::new())),
                b')' => self.stack.push(Value::Tuple(Vec::new())),
                b'N' => self.stack.push(Value::None),
                0x88 => self.stack.push(Value::Bool(true)),
                0x89 => self.stack.push(Value::Bool(false)),
                b'K' => {
                    let v = self.byte()? as i64;
                    self.stack.push(Value::Int(v));
                }
                b'M' => {
                    let b = self.take(2)?;
                    let v = u16::from_le_bytes([b[0], b[1]]) as i64;
                    self.stack.push(Value::Int(v));
                }
                b'J' => {
                    let b = self.take(4)?;
                    let v = i32::from_le_bytes(b.try_into().unwrap()) as i64;
                    self.stack.push(Value::Int(v));
                }
                0x8a => {
                    let n = self.byte()? as usize;
                    let b = self.take(n)?.to_vec();
                    let mut v: i64 = 0;
                    for (i, &x) in b.iter().enumerate().take(8) {
                        v |= (x as i64) << (8 * i);
                    }
                    if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                        v -= 1i64 << (8 * n);
                    }
                    self.stack.push(Value::Int(v));
                }
                b'G' => {
                    let v = f64::from_be_bytes(self.take(8)?.try_into().unwrap());
                    self.stack.push(Value::Float(v));
                }
                b'X' => {
                    let b = self.take(4)?;
                    let n = u32::from_le_bytes(b.try_into().unwrap()) as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(s));
                }
                0x8c => {
                    let n = self.byte()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(s));
                }
                b'U' | b'C' => {
                    let n = self.byte()? as usize;
                    let b = self.take(n)?.to_vec();
                    self.stack.push(Value::Bytes(b));
                }
                b'B' | b'T' => {
                    let b = self.take(4)?;
                    let n = u32::from_le_bytes(b.try_into().unwrap()) as usize;
                    let b = self.take(n)?.to_vec();
                    self.stack.push(Value::Bytes(b));
                }
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(Value::Global(format!("{module}.{name}")));
                }
                0x93 => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    match (module, name) {
                        (Value::Str(m), Value::Str(n)) => {
                            self.stack.push(Value::Global(format!("{m}.{n}")))
                        }
                        _ => return err("bad STACK_GLOBAL"),
                    }
                }
                b'q' => {
                    let i = self.byte()? as usize;
                    let top = self
                        .stack
                        .last()
                        .cloned()
                        .ok_or_else(|| Error("memo of empty stack".into()))?;
                    self.memo.insert(i, top);
                }
                b'r' => {
                    let b = self.take(4)?;
                    let i = u32::from_le_bytes(b.try_into().unwrap()) as usize;
                    let top = self
                        .stack
                        .last()
                        .cloned()
                        .ok_or_else(|| Error("memo of empty stack".into()))?;
                    self.memo.insert(i, top);
                }
                0x94 => {
                    let i = self.memo.len();
                    let top = self
                        .stack
                        .last()
                        .cloned()
                        .ok_or_else(|| Error("memo of empty stack".into()))?;
                    self.memo.insert(i, top);
                }
                b'h' => {
                    let i = self.byte()? as usize;
                    let v = self
                        .memo
                        .get(&i)
                        .cloned()
                        .ok_or_else(|| Error("bad memo get".into()))?;
                    self.stack.push(v);
                }
                b'j' => {
                    let b = self.take(4)?;
                    let i = u32::from_le_bytes(b.try_into().unwrap()) as usize;
                    let v = self
                        .memo
                        .get(&i)
                        .cloned()
                        .ok_or_else(|| Error("bad memo get".into()))?;
                    self.stack.push(v);
                }
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Tuple(items));
                }
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a]));
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b]));
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b, c]));
                }
                b'Q' => {
                    let pid = self.pop()?;
                    let v = match pid {
                        Value::Tuple(t) if t.len() >= 3 => {
                            let dtype = match &t[1] {
                                Value::Global(g) => g.rsplit('.').next().unwrap_or("").to_string(),
                                Value::Str(s) => s.clone(),
                                _ => String::new(),
                            };
                            let key = match &t[2] {
                                Value::Str(s) => s.clone(),
                                Value::Int(i) => i.to_string(),
                                _ => return err("bad storage key"),
                            };
                            Value::Storage { key, dtype }
                        }
                        _ => return err("unsupported persistent id"),
                    };
                    self.stack.push(v);
                }
                b'R' => {
                    let args = self.pop()?;
                    let func = self.pop()?;
                    let v = self.reduce(func, args)?;
                    self.stack.push(v);
                }
                0x81 => {
                    let _args = self.pop()?;
                    let cls = self.pop()?;
                    let v = match cls {
                        Value::Global(n) if n == "collections.OrderedDict" => {
                            Value::Dict(Vec::new())
                        }
                        Value::Global(n) => Value::Object(n),
                        _ => Value::Object("newobj".into()),
                    };
                    self.stack.push(v);
                }
                b'b' => {
                    // BUILD: state is ignored (e.g. OrderedDict._metadata).
                    self.pop()?;
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    if let Some(Value::Dict(d)) = self.stack.last_mut() {
                        d.push((k, v));
                    }
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    if let Some(Value::Dict(d)) = self.stack.last_mut() {
                        let mut it = items.into_iter();
                        while let (Some(k), Some(v)) = (it.next(), it.next()) {
                            d.push((k, v));
                        }
                    }
                }
                b'a' => {
                    let v = self.pop()?;
                    if let Some(Value::List(l)) = self.stack.last_mut() {
                        l.push(v);
                    }
                }
                b'e' => {
                    let items = self.pop_mark()?;
                    if let Some(Value::List(l)) = self.stack.last_mut() {
                        l.extend(items);
                    }
                }
                other => {
                    return err(format!(
                        "unsupported pickle opcode 0x{other:02x} at {}",
                        self.pos - 1
                    ))
                }
            }
        }
    }
}

/// Loads a `torch.save` archive and returns the root pickled value.
pub fn load(bytes: &[u8]) -> Result<Value> {
    let zip = Zip::open(bytes)?;
    let prefix = zip
        .prefix()
        .ok_or_else(|| Error("archive has no data.pkl".into()))?;
    let pkl = zip.find(&format!("{prefix}data.pkl")).unwrap();
    let mut u = Unpickler {
        data: pkl,
        pos: 0,
        stack: Vec::new(),
        marks: Vec::new(),
        memo: HashMap::new(),
        zip: &zip,
        prefix,
    };
    u.run()
}

/// Flattens nested dicts of tensors into `"a.b.c" -> tensor`.
pub fn flatten(v: Value) -> Vec<(String, TorchTensor)> {
    fn walk(prefix: &str, v: Value, out: &mut Vec<(String, TorchTensor)>) {
        match v {
            Value::Tensor(t) => out.push((prefix.to_string(), t)),
            Value::Dict(items) => {
                for (k, v) in items {
                    let key = match k {
                        Value::Str(s) => s,
                        Value::Int(i) => i.to_string(),
                        _ => continue,
                    };
                    let name = if prefix.is_empty() {
                        key
                    } else {
                        format!("{prefix}.{key}")
                    };
                    walk(&name, v, out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk("", v, &mut out);
    out
}
