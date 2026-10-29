//! Splitting phoneme strings into synthesis chunks for streaming.
//!
//! The acoustic model works on whole utterances (≤ 510 tokens) and sounds
//! best around 100–250 tokens; streaming wants a short first chunk so audio
//! starts quickly. Chunks break at sentence ends first, then at clause
//! punctuation, then at spaces.

/// Chunking parameters, in model tokens (one per phoneme character).
#[derive(Clone, Copy, Debug)]
pub struct ChunkPlan {
    /// Upper bound for the first chunk (latency).
    pub first_max: usize,
    /// Preferred upper bound for later chunks.
    pub max: usize,
    /// Chunks shorter than this are merged with a neighbour when possible.
    pub min: usize,
}

impl Default for ChunkPlan {
    fn default() -> Self {
        ChunkPlan {
            first_max: 120,
            max: 250,
            min: 12,
        }
    }
}

/// Hard limit of the K82 model (context 512 minus two pad tokens).
pub const HARD_MAX: usize = 510;

fn len(s: &str) -> usize {
    s.chars().count()
}

/// Splits after any char in `marks` that is followed by a space.
fn split_after(s: &str, marks: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        if marks.contains(chars[i]) {
            // Swallow closing quotes/brackets glued to the mark.
            let mut j = i + 1;
            while j < chars.len() && "”)\"".contains(chars[j]) {
                j += 1;
            }
            if j >= chars.len() || chars[j] == ' ' {
                let piece: String = chars[start..j].iter().collect();
                if !piece.trim().is_empty() {
                    out.push(piece.trim().to_string());
                }
                start = j;
                i = j;
                continue;
            }
        }
        i += 1;
    }
    let tail: String = chars[start..].iter().collect();
    if !tail.trim().is_empty() {
        out.push(tail.trim().to_string());
    }
    out
}

/// Splits a long unit at spaces into pieces of at most `max` tokens.
fn split_words(s: &str, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for w in s.split(' ') {
        if !cur.is_empty() && len(&cur) + 1 + len(w) > max {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(w);
        while len(&cur) > HARD_MAX {
            let head: String = cur.chars().take(HARD_MAX).collect();
            cur = cur.chars().skip(HARD_MAX).collect();
            out.push(head);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Breaks text units larger than `max` at clauses, then at spaces.
fn fit(unit: &str, max: usize) -> Vec<String> {
    if len(unit) <= max {
        return vec![unit.to_string()];
    }
    let mut out = Vec::new();
    for clause in split_after(unit, ",;:—") {
        if len(&clause) <= max {
            out.push(clause);
        } else {
            out.extend(split_words(&clause, max));
        }
    }
    out
}

/// Plans streaming chunks for a phoneme string.
pub fn chunk_phonemes(ps: &str, plan: ChunkPlan) -> Vec<String> {
    let max = plan.max.min(HARD_MAX);
    let mut units: Vec<String> = Vec::new();
    for sentence in split_after(ps, ".!?…") {
        units.extend(fit(&sentence, max));
    }
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for u in units {
        let limit = if chunks.is_empty() {
            plan.first_max.min(max)
        } else {
            max
        };
        // A first sentence longer than the latency budget is cut at clauses.
        let pieces = if chunks.is_empty() && cur.is_empty() && len(&u) > limit {
            fit(&u, limit)
        } else {
            vec![u]
        };
        for p in pieces {
            let limit = if chunks.is_empty() {
                plan.first_max.min(max)
            } else {
                max
            };
            // The first chunk is a single sentence/clause so audio starts early.
            let first_done = chunks.is_empty() && len(&cur) >= plan.min;
            if !cur.is_empty()
                && (first_done || len(&cur) + 1 + len(&p) > limit)
                && len(&cur) >= plan.min
            {
                chunks.push(std::mem::take(&mut cur));
            }
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(&p);
            if len(&cur) > HARD_MAX {
                let mut parts = split_words(&cur, max);
                cur = parts.pop().unwrap_or_default();
                chunks.extend(parts);
            }
        }
    }
    if !cur.is_empty() {
        if len(&cur) < plan.min
            && !chunks.is_empty()
            && len(chunks.last().unwrap()) + 1 + len(&cur) <= HARD_MAX
        {
            let last = chunks.last_mut().unwrap();
            last.push(' ');
            last.push_str(&cur);
        } else {
            chunks.push(cur);
        }
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_sentences() {
        let ps = "ˈola. kˈomo estˈas? mˈuj bjˈen, ɡɾˈaθjas.";
        let c = chunk_phonemes(
            ps,
            ChunkPlan {
                first_max: 10,
                max: 30,
                min: 3,
            },
        );
        assert_eq!(c[0], "ˈola.");
        assert!(c.iter().all(|x| x.chars().count() <= 30));
        assert_eq!(c.join(" "), ps);
    }

    #[test]
    fn respects_hard_limit() {
        let ps = "a ".repeat(800);
        let c = chunk_phonemes(ps.trim(), ChunkPlan::default());
        assert!(c.iter().all(|x| x.chars().count() <= HARD_MAX));
    }
}
