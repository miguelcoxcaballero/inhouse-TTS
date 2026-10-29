//! Opt-in stage timing (`IHTTS_PROFILE=1`), disabled on wasm.

pub struct Profiler {
    #[cfg(not(target_arch = "wasm32"))]
    inner: Option<(std::time::Instant, Vec<(&'static str, f64)>)>,
}

impl Profiler {
    pub fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let on = std::env::var_os("IHTTS_PROFILE").is_some();
            Profiler {
                inner: on.then(|| (std::time::Instant::now(), Vec::new())),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Profiler {}
        }
    }

    pub fn mark(&mut self, _name: &'static str) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some((t, v)) = &mut self.inner {
            v.push((_name, t.elapsed().as_secs_f64()));
            *t = std::time::Instant::now();
        }
    }

    pub fn report(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some((_, v)) = &self.inner {
            let total: f64 = v.iter().map(|x| x.1).sum();
            for (n, s) in v {
                eprintln!("  {n:<14} {:8.1} ms", s * 1e3);
            }
            eprintln!("  {:<14} {:8.1} ms", "total", total * 1e3);
        }
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}
