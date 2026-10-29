//! Tiny parallelism layer: rayon on native targets, sequential elsewhere (wasm).

/// Number of worker threads the kernels may use.
pub fn num_threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads().max(1)
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Runs `f(i)` for every `i in 0..n`, possibly in parallel.
pub fn for_each<F>(n: usize, f: F)
where
    F: Fn(usize) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        if n > 1 && num_threads() > 1 {
            use rayon::prelude::*;
            (0..n).into_par_iter().for_each(f);
            return;
        }
    }
    for i in 0..n {
        f(i);
    }
}

/// Runs two closures, possibly in parallel.
pub fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    #[cfg(feature = "parallel")]
    {
        if num_threads() > 1 {
            return rayon::join(a, b);
        }
    }
    (a(), b())
}

/// Raw pointer that may be shared between threads. Callers guarantee that
/// concurrent writes go to disjoint regions.
#[derive(Clone, Copy)]
pub struct SendPtr<T>(pub *mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    #[inline(always)]
    pub fn get(self) -> *mut T {
        self.0
    }
}
