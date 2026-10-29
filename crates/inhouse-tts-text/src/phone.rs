//! Phone sequences shared by the rule-based G2P modules.
//!
//! The output alphabet is the IPA subset (plus a few capital-letter
//! diphthong shorthands) understood by the K82 acoustic model:
//! `A`=eɪ, `I`=aɪ, `W`=aʊ, `O`=oʊ, `Q`=əʊ, `Y`=ɔɪ, `ʧ`=tʃ, `ʤ`=dʒ, `T`=flap.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Syllable nucleus.
    Vowel,
    /// Semivowel (non-syllabic i/u).
    Glide,
    Consonant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ph {
    pub s: &'static str,
    pub kind: Kind,
    /// 0 = unstressed, 1 = secondary, 2 = primary (vowels only).
    pub stress: u8,
}

impl Ph {
    pub fn v(s: &'static str) -> Ph {
        Ph {
            s,
            kind: Kind::Vowel,
            stress: 0,
        }
    }
    pub fn g(s: &'static str) -> Ph {
        Ph {
            s,
            kind: Kind::Glide,
            stress: 0,
        }
    }
    pub fn c(s: &'static str) -> Ph {
        Ph {
            s,
            kind: Kind::Consonant,
            stress: 0,
        }
    }
    pub fn is_vowel(&self) -> bool {
        self.kind == Kind::Vowel
    }
    pub fn is_cons(&self) -> bool {
        self.kind == Kind::Consonant
    }
}

/// A phonemised word plus the punctuation that follows it.
#[derive(Clone, Debug, Default)]
pub struct Word {
    pub phones: Vec<Ph>,
    /// Punctuation emitted right after the word (e.g. `","`).
    pub punct: String,
    /// True when a pause (punctuation or start of text) precedes the word.
    pub after_pause: bool,
}

/// Serialises words in the espeak-like convention used by the model:
/// stress marks immediately before the stressed vowel, words separated by
/// spaces, punctuation glued to the preceding word.
pub fn render(words: &[Word]) -> String {
    let mut out = String::new();
    for (i, w) in words.iter().enumerate() {
        if w.phones.is_empty() && w.punct.is_empty() {
            continue;
        }
        if i > 0 && !out.is_empty() && !w.phones.is_empty() && !out.ends_with(' ') {
            out.push(' ');
        }
        for p in &w.phones {
            match p.stress {
                2 => out.push('ˈ'),
                1 => out.push('ˌ'),
                _ => {}
            }
            out.push_str(p.s);
        }
        out.push_str(&w.punct);
    }
    out.trim().to_string()
}

/// Marks secondary stress on every other syllable from the start of the
/// word, stopping two syllables before the primary stress (espeak style).
pub fn add_secondary_stress(phones: &mut [Ph]) {
    let nuclei: Vec<usize> = phones
        .iter()
        .enumerate()
        .filter(|(_, p)| p.is_vowel())
        .map(|(i, _)| i)
        .collect();
    let Some(primary) = nuclei.iter().position(|&i| phones[i].stress == 2) else {
        return;
    };
    let mut s = 0;
    while s + 2 <= primary {
        if phones[nuclei[s]].stress == 0 {
            phones[nuclei[s]].stress = 1;
        }
        s += 2;
    }
}
