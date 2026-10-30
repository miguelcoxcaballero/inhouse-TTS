//! # Inhouse TTS text front end
//!
//! Turns raw text into the phoneme strings consumed by the acoustic model:
//! normalisation (numbers, dates, money, abbreviations), grapheme-to-phoneme
//! conversion and regional accent rewrites. Pure Rust, no espeak.

pub mod chunk;
pub mod en;
pub mod es;
pub mod locale;
pub mod phone;
pub mod piper;
pub mod scan;

pub use chunk::{chunk_phonemes, ChunkPlan};
pub use locale::{find as find_locale, Lang, Locale, LOCALES};

use std::collections::HashMap;

/// Multilingual front end. English lexicons are loaded on first use (from
/// the embedded copy) or supplied by the caller (e.g. fetched on the web).
#[derive(Default)]
pub struct Frontend {
    english: HashMap<bool, en::G2p>,
}

impl Frontend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Installs an English lexicon (`.lex` text) for US (`british=false`) or GB.
    pub fn load_english_lexicon(&mut self, text: &str, british: bool) {
        self.english
            .insert(british, en::G2p::new(en::Lexicon::parse(text, british)));
    }

    /// True if the locale can be phonemised without loading more data.
    pub fn ready(&self, locale: &Locale) -> bool {
        match locale.lang {
            Lang::En => {
                self.english.contains_key(&locale.british()) || cfg!(feature = "embed-lexicons")
            }
            Lang::Es => true,
        }
    }

    fn english(&mut self, british: bool) -> Option<&en::G2p> {
        #[cfg(feature = "embed-lexicons")]
        self.english
            .entry(british)
            .or_insert_with(|| en::G2p::new(en::Lexicon::embedded(british)));
        self.english.get(&british)
    }

    /// Converts text to model phonemes for a locale.
    pub fn phonemize(&mut self, text: &str, locale: &Locale) -> Result<String, String> {
        // Paragraph breaks act as sentence ends.
        let text = text.replace("\r\n", "\n");
        let mut parts = Vec::new();
        for para in text.split("\n\n").flat_map(|p| p.split('\n')) {
            let para = para.trim();
            if para.is_empty() {
                continue;
            }
            let ps = match locale.lang {
                Lang::Es => {
                    let acc = es::accent(locale.code)
                        .ok_or_else(|| format!("unknown Spanish accent {}", locale.code))?;
                    es::phonemize(para, acc)
                }
                Lang::En => {
                    let g = self
                        .english(locale.british())
                        .ok_or("English lexicon not loaded")?;
                    g.phonemize(para)
                }
            };
            if ps.is_empty() {
                continue;
            }
            let ends = ps.ends_with(['.', '!', '?', '…', '”', ')']);
            parts.push(if ends { ps } else { format!("{ps}.") });
        }
        Ok(parts.join(" "))
    }
}
