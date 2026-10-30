//! Conversion of our front end's phonemes to the espeak-ng IPA that Piper
//! voices were trained on.
//!
//! Piper voices map every input codepoint through the `phoneme_id_map` of
//! their `.onnx.json`, which only knows espeak-ng symbols. Our front end is
//! espeak-like but writes diphthongs as capital-letter shorthands (`A`=eɪ,
//! `I`=aɪ, ...), affricates as single ligatures (`ʧ`, `ʤ`) and English
//! length/rhoticity differently, so each language gets a small rewrite
//! table, verified against espeak-ng on our evaluation corpus.
//!
//! Spanish voices with the espeak voice `es-419` (Latin America) should be
//! fed our `es-MX` phonemes: seseo and no further regional rewrites, which
//! is what espeak-ng produces for them. See [`espeak_voice_locale`].

/// The espeak-ng phoneme dialect a Piper voice expects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PiperLang {
    /// Spanish (espeak voices `es` and `es-419`).
    Es,
    /// American English (espeak voice `en-us`).
    EnUs,
    /// British English (espeak voices `en`, `en-gb`, `en-gb-x-rp`).
    EnGb,
}

/// The Piper dialect for one of our locale codes: `es-*` → Spanish,
/// `en-GB` → British, any other English → American.
pub fn piper_lang(locale: &str) -> Option<PiperLang> {
    let code = locale.trim().to_ascii_lowercase().replace('_', "-");
    if code == "es" || code.starts_with("es-") {
        Some(PiperLang::Es)
    } else if matches!(code.as_str(), "en-gb" | "en-uk") {
        Some(PiperLang::EnGb)
    } else if code == "en" || code.starts_with("en-") {
        Some(PiperLang::EnUs)
    } else {
        None
    }
}

/// The locale to phonemise with for a voice's espeak voice (the `espeak.voice`
/// field of its `.onnx.json`). Latin American voices (`es-419`) get `es-MX`;
/// espeak's plain `en` is British English.
pub fn espeak_voice_locale(voice: &str) -> Option<&'static str> {
    let v = voice.trim().to_ascii_lowercase().replace('_', "-");
    match v.as_str() {
        "es" | "es-es" => Some("es-ES"),
        "es-419" | "es-mx" => Some("es-MX"),
        "en-us" => Some("en-US"),
        "en" | "en-gb" | "en-uk" => Some("en-GB"),
        _ if v.starts_with("en-gb-") => Some("en-GB"),
        _ => None,
    }
}

/// Converts our phoneme output (one or more sentences) to espeak-style IPA,
/// one NFD-normalised string per sentence, ready for a voice's
/// `phoneme_id_map`. For text with several paragraphs use
/// [`piper_paragraphs`].
pub fn piper_sentences(phonemes: &str, lang: PiperLang) -> Vec<String> {
    punctuate(phonemes)
        .into_iter()
        .map(|s| {
            let s = match lang {
                PiperLang::Es => spanish(&s),
                PiperLang::EnUs => american(&s),
                PiperLang::EnGb => british(&s),
            };
            nfd(&s)
        })
        .filter(|s| !s.is_empty())
        .collect()
}

/// Converts paragraphs from [`crate::Frontend::phonemize_paragraphs`] to
/// Piper sentences. Every paragraph ends its last sentence with a terminator
/// (`.` when it has none), so a paragraph ending in `…`, a closing quote or a
/// bracket does not run into the next one.
pub fn piper_paragraphs<S: AsRef<str>>(paragraphs: &[S], lang: PiperLang) -> Vec<String> {
    let mut out = Vec::new();
    for p in paragraphs {
        let mut sentences = piper_sentences(p.as_ref(), lang);
        if let Some(last) = sentences.last_mut() {
            if !last.ends_with(is_terminator) {
                last.push('.');
            }
        }
        out.extend(sentences);
    }
    out
}

/// Characters espeak never emits and the voices never saw.
const DROPPED: &str = "“”«»\"()[]{}¿¡";
/// Clause and sentence punctuation kept in the output.
const KEPT: &str = ",.;:?!";

fn is_terminator(c: char) -> bool {
    matches!(c, '.' | '?' | '!')
}

/// One step of the punctuation pass.
enum Tok {
    Word(String),
    Punct(char),
}

/// Normalises punctuation the way espeak-ng (as driven by piper-tts) emits
/// it and splits into sentences after `.`, `?` and `!`.
///
/// - quotes, brackets and inverted marks are dropped;
/// - em/en dashes end a clause with `;`, except the opening dash of a pair
///   (`dˈixo— em bˈoθ βˈaxa— ke`), which is dropped: espeak only breaks on
///   the closing one. Our front end glues every dash to the word before, so
///   pairs are told apart by position within the sentence;
/// - `…` and runs of three or more dots become `,`: the voices never saw an
///   ellipsis symbol and a comma pause is better than Piper merging the
///   neighbouring words; `..` is ignored, as espeak does;
/// - a run of marks collapses to its first terminator (espeak emits exactly
///   one), or else to one clause mark (`;` / `:` before `,`); a clause mark
///   at the very end (a trailing ellipsis) becomes `.`;
/// - words are separated by one space and marks are glued to the word before.
fn punctuate(text: &str) -> Vec<String> {
    let mut toks = Vec::new();
    let mut word = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if DROPPED.contains(c) {
            continue;
        }
        let mark = match c {
            '—' | '–' => Some(Some(DASH)),
            '…' => Some(Some(',')),
            '.' => {
                let mut dots = 1;
                while chars.next_if_eq(&'.').is_some() {
                    dots += 1;
                }
                Some(match dots {
                    1 => Some('.'),
                    2 => None,
                    _ => Some(','),
                })
            }
            c if KEPT.contains(c) => Some(Some(c)),
            c if c.is_whitespace() => Some(None),
            _ => None,
        };
        let Some(mark) = mark else {
            word.push(c);
            continue;
        };
        if !word.is_empty() {
            toks.push(Tok::Word(std::mem::take(&mut word)));
        }
        if let Some(m) = mark {
            toks.push(Tok::Punct(m));
        }
    }
    if !word.is_empty() {
        toks.push(Tok::Word(word));
    }

    let mut sentences = Vec::new();
    let mut cur = String::new();
    // An opening dash was dropped and its closing one is still to come.
    let mut dash_open = false;
    let mut i = 0;
    while i < toks.len() {
        match &toks[i] {
            Tok::Word(w) => {
                if !cur.is_empty() {
                    cur.push(' ');
                }
                cur.push_str(w);
                i += 1;
            }
            Tok::Punct(_) => {
                let mut run = String::new();
                while let Some(Tok::Punct(p)) = toks.get(i) {
                    run.push(*p);
                    i += 1;
                }
                // Marks with no word before them have nothing to attach to.
                if cur.is_empty() {
                    continue;
                }
                if let Some(t) = run.chars().find(|&c| is_terminator(c)) {
                    cur.push(t);
                    sentences.push(std::mem::take(&mut cur));
                    dash_open = false;
                    continue;
                }
                if run.contains(DASH) {
                    if dash_open {
                        dash_open = false;
                    } else if dash_later(&toks[i..]) {
                        dash_open = true;
                        run.retain(|c| c != DASH);
                    }
                }
                if run.is_empty() {
                    continue;
                }
                if i == toks.len() {
                    cur.push('.');
                } else {
                    cur.push(
                        [';', ':']
                            .into_iter()
                            .find(|&m| run.contains(m) || (m == ';' && run.contains(DASH)))
                            .unwrap_or(','),
                    );
                }
            }
        }
    }
    if !cur.is_empty() {
        sentences.push(cur);
    }
    sentences
}

/// Internal mark for an em/en dash during [`punctuate`].
const DASH: char = '—';

/// True if another dash follows before the end of the sentence.
fn dash_later(toks: &[Tok]) -> bool {
    for t in toks {
        match t {
            Tok::Punct(DASH) => return true,
            Tok::Punct(c) if is_terminator(*c) => return false,
            _ => {}
        }
    }
    false
}

/// Replaces the capital-letter diphthong shorthands and ligatures.
fn expand(s: &str, table: &[(char, &str)]) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match table.iter().find(|(k, _)| *k == c) {
            Some((_, v)) => out.push_str(v),
            None => out.push(c),
        }
    }
    out
}

fn map_words(s: &str, f: impl Fn(&str) -> String) -> String {
    s.split(' ').map(f).collect::<Vec<_>>().join(" ")
}

fn spanish(s: &str) -> String {
    let s = expand(
        s,
        &[
            ('A', "eɪ"),
            ('I', "aɪ"),
            ('W', "aʊ"),
            ('O', "oʊ"),
            ('ʧ', "tʃ"),
            ('ç', "x"),
        ],
    );
    // espeak's hiatus glide: stressed i before a vowel (`rˈiʲo`) and
    // word-final i before a vowel-initial word (`iʲ ˈes`).
    let c: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len() + 8);
    for (k, &ch) in c.iter().enumerate() {
        out.push(ch);
        if ch != 'i' {
            continue;
        }
        let next = c.get(k + 1).copied();
        let stressed_hiatus = k > 0 && c[k - 1] == 'ˈ' && next.is_some_and(|n| "aeoɛu".contains(n));
        let across_words = next == Some(' ') && {
            let mut j = k + 2;
            if c.get(j) == Some(&'ˈ') {
                j += 1;
            }
            c.get(j).is_some_and(|&n| "aeiouɛ".contains(n))
        };
        if stressed_hiatus || across_words {
            out.push('ʲ');
        }
    }
    out
}

/// Appends `ː` to every `v` not already followed by `ː` or a char in `except`.
fn lengthen(w: &str, v: char, except: &str) -> String {
    let c: Vec<char> = w.chars().collect();
    let mut out = String::with_capacity(w.len() + 4);
    for (k, &ch) in c.iter().enumerate() {
        out.push(ch);
        if ch == v
            && !c
                .get(k + 1)
                .is_some_and(|&n| n == 'ː' || except.contains(n))
        {
            out.push('ː');
        }
    }
    out
}

fn count_vowels(w: &str, vowels: &str) -> usize {
    w.chars().filter(|&c| vowels.contains(c)).count()
}

fn american_word(w: &str) -> String {
    let w = w.replace("ɜɹ", "ɜː").replace("əɹ", "ɚ");
    let w = lengthen(&w, 'ɑ', "");
    let w = lengthen(&w, 'ɔ', "ɪ");
    let w = lengthen(&w, 'u', "");
    // espeak's /i/ is long unless it is an unstressed word-final vowel of a
    // polysyllable (happy-tensing is written short).
    let nv = count_vowels(&w, "aeiouæɑɒɔəɛɜɪʊʌᵻɚAIOWYQ");
    let c: Vec<char> = w.chars().collect();
    let mut out = String::with_capacity(w.len() + 4);
    for (k, &ch) in c.iter().enumerate() {
        out.push(ch);
        let next = c.get(k + 1).copied();
        if ch != 'i' || next.is_some_and(|n| matches!(n, 'ː' | 'ə' | 'ʲ')) {
            continue;
        }
        let stressed = k > 0 && matches!(c[k - 1], 'ˈ' | 'ˌ');
        let last = next.is_none_or(|n| !(n.is_alphanumeric() || n == '_'));
        if stressed || nv <= 1 || !last {
            out.push('ː');
        }
    }
    out
}

fn american(s: &str) -> String {
    let s = expand(s, &[('T', "ɾ"), ('ᵊ', "ə"), ('ʧ', "tʃ"), ('ʤ', "dʒ")]);
    let s = map_words(&s, american_word);
    expand(
        &s,
        &[
            ('A', "eɪ"),
            ('I', "aɪ"),
            ('W', "aʊ"),
            ('O', "oʊ"),
            ('Y', "ɔɪ"),
        ],
    )
}

const GB_VOWELS: &str = "aeiouæɑɒɔəɛɜɪʊʌAIQWY";

fn british_word(w: &str) -> String {
    let w = w.replace("ɛː", "eə").replace("ɪə", "iə");
    if count_vowels(&w, GB_VOWELS) <= 1 {
        return w;
    }
    // espeak reduces a word-final schwa to ɐ (unless it closes a diphthong)
    // and writes an unstressed word-final i as ɪ.
    let mut c: Vec<char> = w.chars().collect();
    let Some(k) = c.iter().rposition(|&ch| !KEPT.contains(ch)) else {
        return w;
    };
    let prev = k.checked_sub(1).map(|j| c[j]);
    if c[k] == 'ə' && !prev.is_some_and(|p| p == 'ː' || GB_VOWELS.contains(p)) {
        c[k] = 'ɐ';
    } else if c[k] == 'i' && !prev.is_some_and(|p| matches!(p, 'ˈ' | 'ˌ')) {
        c[k] = 'ɪ';
    }
    c.into_iter().collect()
}

fn british(s: &str) -> String {
    let s = expand(
        s,
        &[('a', "æ"), ('ᵊ', "ə"), ('ʧ', "tʃ"), ('ʤ', "dʒ"), ('T', "t")],
    );
    let s = map_words(&s, british_word);
    expand(
        &s,
        &[
            ('A', "eɪ"),
            ('I', "aɪ"),
            ('W', "aʊ"),
            ('Q', "əʊ"),
            ('Y', "ɔɪ"),
            ('O', "əʊ"),
        ],
    )
}

const ACUTE: char = '\u{301}';
const GRAVE: char = '\u{300}';
const CIRCUMFLEX: char = '\u{302}';
const TILDE: char = '\u{303}';
const DIAERESIS: char = '\u{308}';
const RING: char = '\u{30A}';
const CEDILLA: char = '\u{327}';

/// Canonical decompositions of the Latin-1 letters. Our front ends emit IPA
/// letters, modifier letters and Greek β/θ, none of which decompose (the
/// English lexicons contain no precomposed letter at all), so only a stray
/// Latin letter passed through verbatim can need one. A table this size keeps
/// the crate free of dependencies; anything else is already in NFD.
const DECOMPOSE: &[(char, char, char)] = &[
    ('À', 'A', GRAVE),
    ('Á', 'A', ACUTE),
    ('Â', 'A', CIRCUMFLEX),
    ('Ã', 'A', TILDE),
    ('Ä', 'A', DIAERESIS),
    ('Å', 'A', RING),
    ('Ç', 'C', CEDILLA),
    ('È', 'E', GRAVE),
    ('É', 'E', ACUTE),
    ('Ê', 'E', CIRCUMFLEX),
    ('Ë', 'E', DIAERESIS),
    ('Ì', 'I', GRAVE),
    ('Í', 'I', ACUTE),
    ('Î', 'I', CIRCUMFLEX),
    ('Ï', 'I', DIAERESIS),
    ('Ñ', 'N', TILDE),
    ('Ò', 'O', GRAVE),
    ('Ó', 'O', ACUTE),
    ('Ô', 'O', CIRCUMFLEX),
    ('Õ', 'O', TILDE),
    ('Ö', 'O', DIAERESIS),
    ('Ù', 'U', GRAVE),
    ('Ú', 'U', ACUTE),
    ('Û', 'U', CIRCUMFLEX),
    ('Ü', 'U', DIAERESIS),
    ('Ý', 'Y', ACUTE),
    ('à', 'a', GRAVE),
    ('á', 'a', ACUTE),
    ('â', 'a', CIRCUMFLEX),
    ('ã', 'a', TILDE),
    ('ä', 'a', DIAERESIS),
    ('å', 'a', RING),
    ('ç', 'c', CEDILLA),
    ('è', 'e', GRAVE),
    ('é', 'e', ACUTE),
    ('ê', 'e', CIRCUMFLEX),
    ('ë', 'e', DIAERESIS),
    ('ì', 'i', GRAVE),
    ('í', 'i', ACUTE),
    ('î', 'i', CIRCUMFLEX),
    ('ï', 'i', DIAERESIS),
    ('ñ', 'n', TILDE),
    ('ò', 'o', GRAVE),
    ('ó', 'o', ACUTE),
    ('ô', 'o', CIRCUMFLEX),
    ('õ', 'o', TILDE),
    ('ö', 'o', DIAERESIS),
    ('ù', 'u', GRAVE),
    ('ú', 'u', ACUTE),
    ('û', 'u', CIRCUMFLEX),
    ('ü', 'u', DIAERESIS),
    ('ý', 'y', ACUTE),
    ('ÿ', 'y', DIAERESIS),
];

/// Unicode NFD for every character our front ends can produce.
fn nfd(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match DECOMPOSE.iter().find(|(k, _, _)| *k == c) {
            Some(&(_, base, mark)) => {
                out.push(base);
                out.push(mark);
            }
            None => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Keys of the davefx (es) `phoneme_id_map`; lessac (en-US) and alan
    /// (en-GB) share it plus the two apical/laminal diacritics.
    const ES_MAP: &str = " !\"#$'(),-.0123456789:;?^_abcdefhijklmnopqrstuvwxyzæçðøħŋœǀǁǂǃɐɑɒɓɔɕɖɗɘəɚɛɜɞɟɠɡɢɣɤɥɦɧɨɪɫɬɭɮɯɰɱɲɳɴɵɶɸɹɺɻɽɾʀʁʂʃʄʈʉʊʋʌʍʎʏʐʑʒʔʕʘʙʛʜʝʟʡʢʰʲˈˌːˑ˞ˤ\u{303}\u{327}\u{329}\u{32a}\u{32f}βεθχᵻ↑↓ⱱ";
    const EN_EXTRA: &str = "\u{33a}\u{33b}";

    fn mapped(lang: PiperLang, s: &str) {
        for c in s.chars() {
            let ok = ES_MAP.contains(c) || (lang != PiperLang::Es && EN_EXTRA.contains(c));
            assert!(ok, "{c:?} ({:04X}) not in the {lang:?} map: {s}", c as u32);
        }
    }

    /// Our phonemes → the reference conversion (sentences joined by ` | `).
    fn check(lang: PiperLang, ours: &str, want: &str) {
        let got = piper_sentences(ours, lang);
        assert_eq!(got.join(" | "), nfd(want), "input: {ours}");
        for s in &got {
            mapped(lang, s);
        }
    }

    #[test]
    fn spanish_oracle() {
        use PiperLang::Es;
        check(
            Es,
            "ˈoɪ ˈI ˈuna wˈelɣa ðe ˌWtoβˈuses.",
            "ˈoɪ ˈaɪ ˈuna wˈelɣa ðe ˌaʊtoβˈuses.",
        );
        check(
            Es,
            "el rˈAno ðe los θjˈelos ˈes pˌaɾa los ˈombɾes βwˈenos.",
            "el rˈeɪno ðe los θjˈelos ˈes pˌaɾa los ˈombɾes βwˈenos.",
        );
        check(
            Es,
            "los ʧˈikos xwˈeɣan kon la pelˈota en la plˈaθa maʝˈoɾ.",
            "los tʃˈikos xwˈeɣan kon la pelˈota en la plˈaθa maʝˈoɾ.",
        );
        check(
            Es,
            "la kˈWsa pɾˌinθipˈal ðel Wmˈɛnto ˈes la ðˈeʊða, dˈixo…“ el minˈistɾo”.",
            "la kˈaʊsa pɾˌinθipˈal ðel aʊmˈɛnto ˈes la ðˈeʊða, dˈixo, el minˈistɾo.",
        );
        check(
            Es,
            "dˈonde estˈa la ˌestaθjˈon? mˈuj θˈeɾka!",
            "dˈonde estˈa la ˌestaθjˈon? | mˈuj θˈeɾka!",
        );
        // es-MX and es-AR samples from the comparison corpus.
        check(
            Es,
            "la xˈɛnte ðel pwˈeβlo kˈome ʧoɾˈiso, xamˈon i kˈeso tˈoðos los ðˈias.",
            "la xˈɛnte ðel pwˈeβlo kˈome tʃoɾˈiso, xamˈon i kˈeso tˈoðos los ðˈiʲas.",
        );
        check(
            Es,
            "el ˈaɣwa ðel rˈio ehtˈa mˈuj fɾˈia en imbjˈeɾno.",
            "el ˈaɣwa ðel rˈiʲo ehtˈa mˈuj fɾˈiʲa en imbjˈeɾno.",
        );
        check(
            Es,
            "tˈɛŋɡo βˌAntitɾˈes ˈaɲos i βˈiβo en ˈum pˈiso pekˈeɲo sˈeɾka ðel sˈɛntɾo.",
            "tˈɛŋɡo βˌeɪntitɾˈes ˈaɲos i βˈiβo en ˈum pˈiso pekˈeɲo sˈeɾka ðel sˈɛntɾo.",
        );
        // Word-final i before a vowel-initial word; Chilean ç.
        check(Es, "kˈasi ˈoko i ˈes", "kˈasiʲ ˈoko iʲ ˈes");
        check(Es, "mˈuçe", "mˈuxe");
    }

    #[test]
    fn american_oracle() {
        use PiperLang::EnUs;
        check(
            EnUs,
            "həlˈO ðˈɛɹ, hˌW ɑɹ ju tədˈA?",
            "həlˈoʊ ðˈɛɹ, hˌaʊ ɑːɹ juː tədˈeɪ?",
        );
        check(
            EnUs,
            "ðə ʧˈɜɹʧ ʤˈʌʤ θˈɔt mI fˈɑðəɹ wʌz ɐ ɡˈʊd bˈY.",
            "ðə tʃˈɜːtʃ dʒˈʌdʒ θˈɔːt maɪ fˈɑːðɚ wʌz ɐ ɡˈʊd bˈɔɪ.",
        );
        check(
            EnUs,
            "wˈɔTəɹ ænd bˈʌTəɹ ɑɹ bˈɛTəɹ ðən nˈʌθɪŋ æt ˈɔl.",
            "wˈɔːɾɚ ænd bˈʌɾɚ ɑːɹ bˈɛɾɚ ðən nˈʌθɪŋ æt ˈɔːl.",
        );
        check(
            EnUs,
            "ʤˈY ænd nˈYz fˈɪld ðə tˈY ʃˈɑp.",
            "dʒˈɔɪ ænd nˈɔɪz fˈɪld ðə tˈɔɪ ʃˈɑːp.",
        );
        check(
            EnUs,
            "ˌWəɹ kwˈIəɹ sˈæŋ bjˈuTəfəli; nˈObˌɑdi nˈOz wˌI!",
            "ˌaʊɚ kwˈaɪɚ sˈæŋ bjˈuːɾəfəli; nˈoʊbˌɑːdi nˈoʊz wˌaɪ!",
        );
    }

    #[test]
    fn british_oracle() {
        use PiperLang::EnGb;
        check(
            EnGb,
            "həlˈQ ðˈɛː, hˌW ɑː juː tədˈA? ˌIm fˈIn, θˈaŋks.",
            "həlˈəʊ ðˈeə, hˌaʊ ɑː juː tədˈeɪ? | ˌaɪm fˈaɪn, θˈæŋks.",
        );
        check(
            EnGb,
            "ðə kwˈɪk bɹˈWn fˈɒks ʤˈʌmps ˈQvə ðə lˈAzi dˈɒɡ.",
            "ðə kwˈɪk bɹˈaʊn fˈɒks dʒˈʌmps ˈəʊvɐ ðə lˈeɪzɪ dˈɒɡ.",
        );
        check(
            EnGb,
            "ðə ʧˈɜːʧ ʤˈʌʤ θˈɔːt mI fˈɑːðə wɒz ɐ ɡˈʊd bˈY.",
            "ðə tʃˈɜːtʃ dʒˈʌdʒ θˈɔːt maɪ fˈɑːðɐ wɒz ɐ ɡˈʊd bˈɔɪ.",
        );
        check(
            EnGb,
            "wˈɔːtə and bˈʌtə ɑː bˈɛtə ðən nˈʌθɪŋ at ˈɔːl.",
            "wˈɔːtɐ ænd bˈʌtɐ ɑː bˈɛtɐ ðən nˈʌθɪŋ æt ˈɔːl.",
        );
    }

    #[test]
    fn punctuation() {
        let p = |s| piper_sentences(s, PiperLang::Es);
        assert_eq!(p("«ˈola» — dˈixo."), ["ˈola; dˈixo."]);
        assert_eq!(p("ˈuno–dˈos"), ["ˈuno; dˈos"]);
        assert_eq!(p("pwˈes... nˈo"), ["pwˈes, nˈo"]);
        assert_eq!(p("pwˈes…"), ["pwˈes."]);
        assert_eq!(p("(ˈuno) , [dˈos]:tɾˈes ,."), ["ˈuno, dˈos: tɾˈes."]);
        assert_eq!(p("¿kˈe?  ¡nˈaða!  …  βˈale"), ["kˈe?", "nˈaða!", "βˈale"]);
        assert!(p(" “ ” ").is_empty());
    }

    /// espeak emits exactly one terminator per sentence, the first of a run,
    /// and does not end a sentence on `..`.
    #[test]
    fn repeated_marks() {
        let p = |s| piper_sentences(s, PiperLang::Es);
        assert_eq!(p("ˈola!!! nˈo???"), ["ˈola!", "nˈo?"]);
        assert_eq!(p("kˈe?! βˈale"), ["kˈe?", "βˈale"]);
        assert_eq!(p("ˈola!? bjˈen.!"), ["ˈola!", "bjˈen."]);
        assert_eq!(p("ˈola.. aðjˈos"), ["ˈola aðjˈos"]);
        assert_eq!(p("ˈola.... bjˈen."), ["ˈola, bjˈen."]);
        assert_eq!(p("ˈola...."), ["ˈola."]);
    }

    /// Only the closing dash of a pair ends a clause; a lone dash does too.
    #[test]
    fn dashes() {
        let p = |s| piper_sentences(s, PiperLang::Es);
        assert_eq!(
            p("ˈel ðˈixo— em bˈoθ βˈaxa— ke nˈo."),
            ["ˈel ðˈixo em bˈoθ βˈaxa; ke nˈo."]
        );
        assert_eq!(p("sˈi— dˈixo ˈel."), ["sˈi; dˈixo ˈel."]);
        // The pair is closed by the sentence end; the next sentence starts afresh.
        assert_eq!(
            p("ˈola— dˈixo ˈel—. ˈuno— dˈos."),
            ["ˈola dˈixo ˈel.", "ˈuno; dˈos."]
        );
        assert_eq!(p("ˈa— ˈbe— ˈθe— ˈde— ˈe."), ["ˈa ˈbe; ˈθe ˈde; ˈe."]);
        assert_eq!(p("ˈa—, ˈbe— ˈθe"), ["ˈa, ˈbe; ˈθe"]);
    }

    /// A paragraph ending without a terminator (`…`, a closing quote or
    /// bracket) stays a sentence of its own.
    #[test]
    fn paragraphs() {
        let mut fe = crate::Frontend::new();
        let locale = crate::find_locale("es-ES").unwrap();
        let run = |fe: &mut crate::Frontend, text| {
            let paras = fe.phonemize_paragraphs(text, locale).unwrap();
            piper_paragraphs(&paras, PiperLang::Es)
        };
        assert_eq!(
            run(&mut fe, "Pues…\nOtro día empezó."),
            ["pwes.", "ˈotɾo ðˈiʲa ˌempeθˈo."]
        );
        assert_eq!(
            run(&mut fe, "Dijo «hola»\nY se fue."),
            ["dˈixo ˈola.", "i se fwˈe."]
        );
        assert_eq!(
            run(&mut fe, "Hola (amigo)\n\nAdiós."),
            ["ˈola amˈiɣo.", "aðjˈos."]
        );
        // The joined form is unchanged.
        let joined = fe.phonemize("Pues…\nOtro día.", locale).unwrap();
        assert_eq!(joined, "pwes… ˈotɾo ðˈia.");
        assert_eq!(piper_paragraphs(&["ˈola?"], PiperLang::Es), ["ˈola?"]);
        assert!(piper_paragraphs::<&str>(&[], PiperLang::Es).is_empty());
    }

    #[test]
    fn nfd_latin() {
        assert_eq!(nfd("ñé ç ʃ"), "n\u{303}e\u{301} c\u{327} ʃ");
    }

    #[test]
    fn languages() {
        assert_eq!(piper_lang("es-AR"), Some(PiperLang::Es));
        assert_eq!(piper_lang("es"), Some(PiperLang::Es));
        assert_eq!(piper_lang("en-US"), Some(PiperLang::EnUs));
        assert_eq!(piper_lang("en_GB"), Some(PiperLang::EnGb));
        assert_eq!(piper_lang("fr-FR"), None);
        assert_eq!(espeak_voice_locale("es"), Some("es-ES"));
        assert_eq!(espeak_voice_locale("es-419"), Some("es-MX"));
        assert_eq!(espeak_voice_locale("en-us"), Some("en-US"));
        assert_eq!(espeak_voice_locale("en"), Some("en-GB"));
        assert_eq!(espeak_voice_locale("en-gb-x-rp"), Some("en-GB"));
        assert_eq!(espeak_voice_locale("de"), None);
    }

    /// The whole pipeline (text → our phonemes → Piper IPA) only produces
    /// symbols the voices know.
    #[test]
    fn front_end_symbols_are_mapped() {
        let mut fe = crate::Frontend::new();
        let mut cases = vec![
            (
                "es-ES",
                "Ayer llovió mucho en la calle; ¡qué bonito es el cielo azul de Madrid!",
            ),
            (
                "es-MX",
                "«El agua del río» —dijo— está muy fría... en invierno.",
            ),
            (
                "es-CL",
                "Los jóvenes gentiles están cansados (después de la fiesta).",
            ),
        ];
        if cfg!(feature = "embed-lexicons") {
            cases.extend([
                (
                    "en-US",
                    "Our choir sang beautifully in the cathedral on Sunday.",
                ),
                (
                    "en-US",
                    "Please measure the treasure carefully before the vision fades!",
                ),
                (
                    "en-GB",
                    "The world's largest ocean is the Pacific: it covers a third of the planet.",
                ),
                (
                    "en-GB",
                    "They're going to visit their aunt in New York next month?",
                ),
            ]);
        }
        for (code, text) in cases {
            let locale = crate::find_locale(code).unwrap();
            let ours = fe.phonemize(text, locale).unwrap();
            let lang = piper_lang(code).unwrap();
            let out = piper_sentences(&ours, lang);
            assert!(!out.is_empty(), "{code}: {text}");
            for s in &out {
                mapped(lang, s);
            }
        }
    }
}
