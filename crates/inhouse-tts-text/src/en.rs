//! English front end (American and British): dictionary lookup with
//! morphological stemming, numbers, compound splitting and letter-to-sound
//! rules for out-of-vocabulary words.
//!
//! The lexicons are the misaki gold + silver dictionaries (Apache-2.0),
//! whose phoneme alphabet matches the K82 English voices.

use crate::scan::{scan, Quotes, Tok};
use std::collections::HashMap;

const PRIMARY: char = 'ˈ';
const SECONDARY: char = 'ˌ';
const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
const CONSONANTS: &str = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθT";

fn is_vowel(c: char) -> bool {
    VOWELS.contains(c)
}

#[derive(Clone, Debug)]
enum Entry {
    Plain(String),
    Tagged {
        default: String,
        tags: Vec<(String, Option<String>)>,
    },
}

impl Entry {
    fn get(&self, tag: Option<&str>) -> Option<&str> {
        match self {
            Entry::Plain(s) => Some(s),
            Entry::Tagged { default, tags } => {
                if let Some(t) = tag {
                    if let Some((_, v)) = tags.iter().find(|(k, _)| k == t) {
                        return v.as_deref();
                    }
                }
                Some(default)
            }
        }
    }
}

/// A pronunciation dictionary in the `.lex` text format.
pub struct Lexicon {
    map: HashMap<String, Entry>,
    pub british: bool,
}

impl Lexicon {
    pub fn parse(text: &str, british: bool) -> Lexicon {
        let mut map = HashMap::with_capacity(200_000);
        for line in text.lines() {
            let mut parts = line.split('\t');
            let (Some(word), Some(first)) = (parts.next(), parts.next()) else {
                continue;
            };
            let rest: Vec<&str> = parts.collect();
            let entry = if rest.is_empty() {
                Entry::Plain(first.to_string())
            } else {
                let tags = rest
                    .iter()
                    .filter_map(|p| p.split_once('='))
                    .map(|(k, v)| {
                        (
                            k.to_string(),
                            if v.is_empty() {
                                None
                            } else {
                                Some(v.to_string())
                            },
                        )
                    })
                    .collect();
                Entry::Tagged {
                    default: first.to_string(),
                    tags,
                }
            };
            map.insert(word.to_string(), entry);
        }
        Lexicon { map, british }
    }

    #[cfg(feature = "embed-lexicons")]
    pub fn embedded(british: bool) -> Lexicon {
        let text = if british {
            include_str!("../data/en-GB.lex")
        } else {
            include_str!("../data/en-US.lex")
        };
        Lexicon::parse(text, british)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Exact lookup with misaki's case growing (Word ↔ word).
    fn raw(&self, w: &str) -> Option<&Entry> {
        if let Some(e) = self.map.get(w) {
            return Some(e);
        }
        if w.chars().count() < 2 {
            return None;
        }
        let lower = w.to_lowercase();
        if lower != w && capitalize(&lower) == w {
            return self.map.get(&lower);
        }
        if lower == w {
            return self.map.get(&capitalize(w));
        }
        None
    }

    fn has(&self, w: &str) -> bool {
        self.raw(w).is_some()
    }

    fn get(&self, w: &str) -> Option<String> {
        self.get_tagged(w, None)
    }

    fn get_tagged(&self, w: &str, tag: Option<&str>) -> Option<String> {
        self.raw(w).and_then(|e| e.get(tag)).map(str::to_string)
    }

    /// True when the entry has a clause-final variant (misaki tag "None").
    fn has_final_variant(&self, w: &str) -> bool {
        matches!(self.raw(w), Some(Entry::Tagged { tags, .. }) if tags.iter().any(|(k, _)| k == "None"))
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}

/// Moves every stress mark right before the vowel that follows it.
fn restress(ps: &str) -> String {
    let chars: Vec<char> = ps.chars().collect();
    let mut keyed: Vec<(f64, char)> = Vec::with_capacity(chars.len());
    for (i, &c) in chars.iter().enumerate() {
        if c == PRIMARY || c == SECONDARY {
            let j = chars[i..]
                .iter()
                .position(|&v| is_vowel(v))
                .map(|p| p + i)
                .unwrap_or(i);
            keyed.push((j as f64 - 0.5, c));
        } else {
            keyed.push((i as f64, c));
        }
    }
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    keyed.into_iter().map(|(_, c)| c).collect()
}

/// misaki's `apply_stress`.
fn apply_stress(ps: &str, stress: Option<f32>) -> String {
    let Some(stress) = stress else {
        return ps.to_string();
    };
    let has_p = ps.contains(PRIMARY);
    let has_s = ps.contains(SECONDARY);
    let any_vowel = ps.chars().any(is_vowel);
    if stress < -1.0 {
        ps.replace([PRIMARY, SECONDARY], "")
    } else if stress == -1.0 || ((stress == 0.0 || stress == -0.5) && has_p) {
        ps.replace(SECONDARY, "")
            .replace(PRIMARY, &SECONDARY.to_string())
    } else if (stress == 0.0 || stress == 0.5 || stress == 1.0) && !has_p && !has_s {
        if !any_vowel {
            ps.to_string()
        } else {
            restress(&format!("{SECONDARY}{ps}"))
        }
    } else if stress >= 1.0 && !has_p && has_s {
        ps.replace(SECONDARY, &PRIMARY.to_string())
    } else if stress > 1.0 && !has_p && !has_s {
        if !any_vowel {
            ps.to_string()
        } else {
            restress(&format!("{PRIMARY}{ps}"))
        }
    } else {
        ps.to_string()
    }
}

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

fn below_thousand(n: u64, out: &mut Vec<String>) {
    let h = n / 100;
    let r = n % 100;
    if h > 0 {
        out.push(ONES[h as usize].into());
        out.push("hundred".into());
    }
    if r > 0 {
        if r < 20 {
            out.push(ONES[r as usize].into());
        } else {
            out.push(TENS[(r / 10) as usize].into());
            if r % 10 > 0 {
                out.push(ONES[(r % 10) as usize].into());
            }
        }
    }
}

/// Cardinal number words (no "and"), e.g. 1234 → one thousand two hundred thirty four.
pub fn cardinal(n: u64) -> Vec<String> {
    if n == 0 {
        return vec!["zero".into()];
    }
    let mut out = Vec::new();
    let scales = [
        (1_000_000_000_000u64, "trillion"),
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1000, "thousand"),
    ];
    let mut n = n;
    for (size, name) in scales {
        if n >= size {
            below_thousand(n / size, &mut out);
            out.push(name.into());
            n %= size;
        }
    }
    if n > 0 {
        below_thousand(n, &mut out);
    }
    out
}

fn ordinal_word(w: &str) -> String {
    match w {
        "one" => "first".into(),
        "two" => "second".into(),
        "three" => "third".into(),
        "five" => "fifth".into(),
        "eight" => "eighth".into(),
        "nine" => "ninth".into(),
        "twelve" => "twelfth".into(),
        w if w.ends_with('y') => format!("{}ieth", &w[..w.len() - 1]),
        w => format!("{w}th"),
    }
}

pub fn ordinal(n: u64) -> Vec<String> {
    let mut w = cardinal(n);
    if let Some(last) = w.last_mut() {
        *last = ordinal_word(last);
    }
    w
}

/// Years as spoken: 1999 → nineteen ninety nine, 2026 → twenty twenty six.
pub fn year(n: u64) -> Vec<String> {
    if !(1000..10000).contains(&n) {
        return cardinal(n);
    }
    let (hi, lo) = (n / 100, n % 100);
    if (2000..2010).contains(&n) || n % 1000 == 0 {
        return cardinal(n);
    }
    let mut out = cardinal(hi);
    if lo == 0 {
        out.push("hundred".into());
    } else if lo < 10 {
        out.push("oh".into());
        out.extend(cardinal(lo));
    } else {
        out.extend(cardinal(lo));
    }
    out
}

fn digits(s: &str) -> Vec<String> {
    s.chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| ONES[d as usize].to_string())
        .collect()
}

/// English grapheme-to-phoneme converter.
pub struct G2p {
    lex: Lexicon,
}

impl G2p {
    pub fn new(lex: Lexicon) -> G2p {
        G2p { lex }
    }

    pub fn british(&self) -> bool {
        self.lex.british
    }

    fn known(&self, w: &str) -> bool {
        if self.lex.has(w) {
            return true;
        }
        if !w.chars().all(|c| c.is_ascii_alphabetic() || c == '\'') {
            return false;
        }
        if w.chars().count() == 1 {
            return true;
        }
        if w == w.to_uppercase() && self.lex.has(&w.to_lowercase()) {
            return true;
        }
        false
    }

    /// Spells a word letter by letter (acronyms).
    fn spell(&self, w: &str) -> Option<String> {
        let mut ps = String::new();
        for c in w.chars().filter(|c| c.is_alphabetic()) {
            ps.push_str(&self.lex.get(&c.to_uppercase().to_string())?);
        }
        let ps = apply_stress(&ps, Some(0.0));
        // Last letter carries the primary stress.
        match ps.rfind(SECONDARY) {
            Some(i) => Some(format!(
                "{}{}{}",
                &ps[..i],
                PRIMARY,
                &ps[i + SECONDARY.len_utf8()..]
            )),
            None => Some(ps),
        }
    }

    fn lookup(&self, w: &str, stress: Option<f32>) -> Option<String> {
        self.lookup_tagged(w, stress, None)
    }

    fn lookup_tagged(&self, w: &str, stress: Option<f32>, tag: Option<&str>) -> Option<String> {
        let mut word = w.to_string();
        if word == word.to_uppercase() && !self.lex.has(&word) {
            word = word.to_lowercase();
        }
        let ps = self.lex.get_tagged(&word, tag)?;
        Some(apply_stress(&ps, stress))
    }

    fn suffix_s(&self, stem: &str) -> String {
        match stem.chars().last() {
            Some(c) if "ptkfθ".contains(c) => format!("{stem}s"),
            Some(c) if "szʃʒʧʤ".contains(c) => {
                format!("{stem}{}z", if self.british() { "ɪ" } else { "ᵻ" })
            }
            _ => format!("{stem}z"),
        }
    }

    fn suffix_ed(&self, stem: &str) -> String {
        let chars: Vec<char> = stem.chars().collect();
        match chars.last() {
            Some(c) if "pkfθʃsʧ".contains(*c) => format!("{stem}t"),
            Some('d') => format!("{stem}{}d", if self.british() { "ɪ" } else { "ᵻ" }),
            Some('t') if self.british() || chars.len() < 2 => format!("{stem}ɪd"),
            Some('t') if "AIOWYiuæɑəɛɪɹʊʌ".contains(chars[chars.len() - 2]) => {
                format!("{}ɾᵻd", chars[..chars.len() - 1].iter().collect::<String>())
            }
            Some('t') => format!("{stem}ᵻd"),
            _ => format!("{stem}d"),
        }
    }

    fn suffix_ing(&self, stem: &str) -> String {
        let chars: Vec<char> = stem.chars().collect();
        if !self.british()
            && chars.len() > 1
            && chars[chars.len() - 1] == 't'
            && "AIOWYiuæɑəɛɪɹʊʌ".contains(chars[chars.len() - 2])
        {
            return format!("{}ɾɪŋ", chars[..chars.len() - 1].iter().collect::<String>());
        }
        format!("{stem}ɪŋ")
    }

    fn stems(&self, w: &str, stress: Option<f32>) -> Option<String> {
        let n = w.len();
        if n >= 3 && w.ends_with('s') {
            let stem = if !w.ends_with("ss") && self.known(&w[..n - 1]) {
                Some(w[..n - 1].to_string())
            } else if (w.ends_with("'s") || (n > 4 && w.ends_with("es") && !w.ends_with("ies")))
                && self.known(&w[..n - 2])
            {
                Some(w[..n - 2].to_string())
            } else if n > 4 && w.ends_with("ies") && self.known(&format!("{}y", &w[..n - 3])) {
                Some(format!("{}y", &w[..n - 3]))
            } else {
                None
            };
            if let Some(s) = stem {
                return self.lookup(&s, stress).map(|p| self.suffix_s(&p));
            }
        }
        if n >= 4 && w.ends_with('d') {
            let stem = if !w.ends_with("dd") && self.known(&w[..n - 1]) {
                Some(&w[..n - 1])
            } else if n > 4 && w.ends_with("ed") && !w.ends_with("eed") && self.known(&w[..n - 2]) {
                Some(&w[..n - 2])
            } else {
                None
            };
            if let Some(s) = stem {
                return self.lookup(s, stress).map(|p| self.suffix_ed(&p));
            }
        }
        if n >= 5 && w.ends_with("ing") {
            let base = &w[..n - 3];
            let doubled = base.len() > 2 && {
                let b = base.as_bytes();
                b[b.len() - 1] == b[b.len() - 2]
                    && "bcdgklmnprstvxz".contains(b[b.len() - 1] as char)
            };
            let stem = if n > 5 && self.known(base) {
                Some(base.to_string())
            } else if self.known(&format!("{base}e")) {
                Some(format!("{base}e"))
            } else if n > 5
                && (doubled || base.ends_with("ck"))
                && self.known(&base[..base.len() - 1])
            {
                Some(base[..base.len() - 1].to_string())
            } else {
                None
            };
            if let Some(s) = stem {
                return self
                    .lookup(&s, Some(stress.unwrap_or(0.5)))
                    .map(|p| self.suffix_ing(&p));
            }
        }
        None
    }

    /// Splits an unknown word into two dictionary words (inhouse = in + house).
    fn compound(&self, w: &str) -> Option<String> {
        let lw = w.to_lowercase();
        let chars: Vec<char> = lw.chars().collect();
        if chars.len() < 5 {
            return None;
        }
        let mut best: Option<(usize, String, String)> = None;
        for cut in 2..chars.len() - 1 {
            let (a, b): (String, String) =
                (chars[..cut].iter().collect(), chars[cut..].iter().collect());
            if a.len() < 2 || b.len() < 3 {
                continue;
            }
            if self.lex.has(&a) && self.lex.has(&b) {
                let score = a.len().min(b.len());
                if best.as_ref().is_none_or(|(s, _, _)| score > *s) {
                    best = Some((score, a, b));
                }
            }
        }
        let (_, a, b) = best?;
        let pa = self.lex.get(&a)?;
        let pb = self.lex.get(&b)?;
        // Compound stress: primary on the first element, secondary on the second.
        let pa = apply_stress(&pa, Some(2.0));
        let pa = if pa.contains(PRIMARY) {
            pa
        } else {
            restress(&format!("{PRIMARY}{}", pa.replace(SECONDARY, "")))
        };
        let pb = pb.replace(PRIMARY, &SECONDARY.to_string());
        Some(format!("{pa}{pb}"))
    }

    /// Phonemes of a single word token (no context).
    fn word(&self, w: &str) -> String {
        let stress = if w == w.to_lowercase() {
            None
        } else if w == w.to_uppercase() && w.chars().count() > 1 {
            Some(2.0)
        } else {
            Some(0.5)
        };
        let lw = w.to_lowercase();
        if w.chars().count() == 1
            && lw != "a"
            && w != "I"
            && w.chars().all(|c| c.is_ascii_alphabetic())
        {
            if let Some(p) = self.spell(w) {
                return p;
            }
        }
        let caps = w.chars().count() > 1 && w == w.to_uppercase();
        if caps && !self.lex.has(w) && w.chars().count() <= 5 && !self.lex.has(&lw) {
            if let Some(p) = self.spell(w) {
                return p;
            }
        }
        if let Some(p) = self.lookup(w, stress) {
            return p;
        }
        if let Some(p) = self.stems(w, stress).or_else(|| self.stems(&lw, stress)) {
            return p;
        }
        if let Some(p) = self.compound(w) {
            return p;
        }
        if caps {
            if let Some(p) = self.spell(w) {
                return p;
            }
        }
        lts(&lw, self.british())
    }

    fn number(&self, num: &str, next: Option<&Tok>) -> Vec<String> {
        let clean = num.replace(',', "");
        if let Some((h, m)) = num.split_once(':') {
            if let (Ok(h), Ok(m)) = (h.parse::<u64>(), m.parse::<u64>()) {
                let mut w = cardinal(h);
                match m {
                    0 => w.push("o'clock".into()),
                    1..=9 => {
                        w.push("oh".into());
                        w.extend(cardinal(m));
                    }
                    _ => w.extend(cardinal(m)),
                }
                return w;
            }
        }
        if let Some((a, b)) = clean.split_once('.') {
            let mut w = if a.is_empty() {
                vec![]
            } else {
                cardinal(a.parse().unwrap_or(0))
            };
            w.push("point".into());
            w.extend(digits(b));
            return w;
        }
        let Ok(n) = clean.parse::<u64>() else {
            return digits(num);
        };
        if let Some(Tok::Word(s)) = next {
            if ["st", "nd", "rd", "th"].contains(&s.to_lowercase().as_str()) {
                return ordinal(n);
            }
        }
        if clean.len() > 1 && clean.starts_with('0') {
            return digits(&clean);
        }
        if num.len() == 4 && !num.contains(',') {
            return year(n);
        }
        cardinal(n)
    }

    fn money(&self, num: &str, sym: char) -> Vec<String> {
        let (one, many, sub_one, sub_many) = match sym {
            '£' => ("pound", "pounds", "penny", "pence"),
            '€' => ("euro", "euros", "cent", "cents"),
            _ => ("dollar", "dollars", "cent", "cents"),
        };
        let clean = num.replace(',', "");
        let (a, b) = clean.split_once('.').unwrap_or((&clean, ""));
        let units: u64 = a.parse().unwrap_or(0);
        let cents: u64 = if b.len() == 2 {
            b.parse().unwrap_or(0)
        } else {
            0
        };
        let mut w = cardinal(units);
        w.push(if units == 1 { one } else { many }.into());
        if cents > 0 {
            w.push("and".into());
            w.extend(cardinal(cents));
            w.push(if cents == 1 { sub_one } else { sub_many }.into());
        }
        w
    }

    /// Converts English text to model phonemes.
    pub fn phonemize(&self, text: &str) -> String {
        let toks = scan(text);
        // Expand into word strings and punctuation.
        enum Item {
            W(String),
            /// Precomputed phonemes (spelled initialisms).
            Ph(String),
            P(&'static str),
        }
        let mut items: Vec<Item> = Vec::new();
        let mut quotes = Quotes::default();
        let mut i = 0;
        while i < toks.len() {
            match &toks[i] {
                Tok::Word(w) => {
                    let lw = w.to_lowercase();
                    if ["st", "nd", "rd", "th"].contains(&lw.as_str())
                        && i > 0
                        && matches!(toks[i - 1], Tok::Num(_))
                    {
                        i += 1;
                        continue;
                    }
                    // Dotted initialisms: U.S., a.m., e.g.
                    if w.chars().count() == 1
                        && matches!(toks.get(i + 1), Some(Tok::Punct('.')))
                        && matches!(toks.get(i + 2), Some(Tok::Word(x)) if x.chars().count() == 1)
                    {
                        let mut letters = String::new();
                        let mut j = i;
                        while let (Some(Tok::Word(x)), Some(Tok::Punct('.'))) =
                            (toks.get(j), toks.get(j + 1))
                        {
                            if x.chars().count() != 1 {
                                break;
                            }
                            letters.push_str(x);
                            j += 2;
                        }
                        let spoken = match letters.to_lowercase().as_str() {
                            "eg" => Some("for example"),
                            "ie" => Some("that is"),
                            _ => None,
                        };
                        match spoken {
                            Some(sp) => items.extend(sp.split(' ').map(|x| Item::W(x.into()))),
                            None => {
                                if let Some(p) = self.spell(&letters) {
                                    items.push(Item::Ph(p));
                                }
                            }
                        }
                        // Keep a sentence-final dot.
                        let ends = matches!(toks.get(j), None | Some(Tok::Word(_))) && toks.get(j).is_none_or(|t| matches!(t, Tok::Word(x) if x.chars().next().is_some_and(char::is_uppercase)));
                        if ends {
                            items.push(Item::P("."));
                        }
                        i = j;
                        continue;
                    }
                    let abbrev = match lw.as_str() {
                        "mr" => Some("mister"),
                        "mrs" => Some("missus"),
                        "ms" => Some("miz"),
                        "dr" => Some("doctor"),
                        "prof" => Some("professor"),
                        "vs" => Some("versus"),
                        "etc" => Some("et cetera"),
                        "no" if matches!(toks.get(i + 1), Some(Tok::Punct('.')))
                            && matches!(toks.get(i + 2), Some(Tok::Num(_))) =>
                        {
                            Some("number")
                        }
                        _ => None,
                    };
                    if let Some(a) = abbrev
                        .filter(|_| matches!(toks.get(i + 1), Some(Tok::Punct('.'))) || lw == "vs")
                    {
                        items.extend(a.split(' ').map(|x| Item::W(x.into())));
                        if matches!(toks.get(i + 1), Some(Tok::Punct('.'))) {
                            i += 1;
                        }
                        i += 1;
                        continue;
                    }
                    if w.contains('-') && self.lex.has(w) {
                        items.push(Item::W(w.clone()));
                    } else {
                        for part in w.split('-').filter(|p| !p.is_empty()) {
                            items.push(Item::W(part.to_string()));
                        }
                    }
                }
                Tok::Num(n) => {
                    let prev = if i > 0 { Some(&toks[i - 1]) } else { None };
                    let next = toks.get(i + 1);
                    let words = match (prev, next) {
                        (Some(Tok::Sym(c @ ('$' | '£' | '€'))), _) => self.money(n, *c),
                        (_, Some(Tok::Sym('%'))) => {
                            i += 1;
                            let mut w = self.number(n, None);
                            w.push("percent".into());
                            w
                        }
                        _ => self.number(n, next),
                    };
                    items.extend(words.into_iter().map(Item::W));
                }
                Tok::Sym(c) => {
                    let w = match c {
                        '%' => "percent",
                        '&' => "and",
                        '+' => "plus",
                        '=' => "equals",
                        '@' => "at",
                        _ => "",
                    };
                    let money_next = matches!(c, '$' | '£' | '€')
                        && matches!(toks.get(i + 1), Some(Tok::Num(_)));
                    if !w.is_empty() && !money_next {
                        items.push(Item::W(w.into()));
                    }
                }
                Tok::Punct(c) => {
                    if let Some(p) = quotes.map(*c) {
                        items.push(Item::P(p));
                    }
                }
            }
            i += 1;
        }
        // Phonemise words, then resolve context-dependent function words
        // from right to left (the/a/to depend on the following sound).
        let mut ps: Vec<Option<String>> = items
            .iter()
            .map(|it| match it {
                Item::W(w) => Some(self.word(w)),
                Item::Ph(p) => Some(p.clone()),
                Item::P(_) => None,
            })
            .collect();
        let mut next_vowel: Option<bool> = None;
        for k in (0..items.len()).rev() {
            match &items[k] {
                Item::P(p) => {
                    if !matches!(*p, "“" | "”" | "(" | ")") {
                        next_vowel = None;
                    }
                }
                Item::Ph(_) => {
                    let first = ps[k]
                        .as_deref()
                        .and_then(|p| p.chars().find(|c| is_vowel(*c) || CONSONANTS.contains(*c)));
                    next_vowel = first.map(is_vowel).or(next_vowel);
                }
                Item::W(w) => {
                    let lw = w.to_lowercase();
                    let special = match lw.as_str() {
                        "the" => Some(
                            if next_vowel == Some(true) {
                                "ði"
                            } else {
                                "ðə"
                            }
                            .to_string(),
                        ),
                        "a" if w != "A" || next_vowel.is_some() => Some("ɐ".to_string()),
                        "an" => Some("ɐn".to_string()),
                        "to" => match next_vowel {
                            None => None,
                            Some(false) => Some("tə".into()),
                            Some(true) => Some("tʊ".into()),
                        },
                        "i" if w == "I" => Some("ˌI".to_string()),
                        "in" => Some(if next_vowel.is_none() { "ˈɪn" } else { "ɪn" }.to_string()),
                        _ => None,
                    };
                    if let Some(s) = special {
                        ps[k] = Some(s);
                    } else if next_vowel.is_none() && self.lex.has_final_variant(w) {
                        // Clause-final function words are stressed (misaki "None" tag).
                        let stress = if *w == lw { None } else { Some(0.5) };
                        if let Some(p) = self.lookup_tagged(w, stress, Some("None")) {
                            ps[k] = Some(p.replace('ɾ', "T"));
                        }
                    }
                    let first = ps[k]
                        .as_deref()
                        .and_then(|p| p.chars().find(|c| is_vowel(*c) || CONSONANTS.contains(*c)));
                    next_vowel = first.map(is_vowel).or(next_vowel);
                }
            }
        }
        let mut out = String::new();
        for (k, it) in items.iter().enumerate() {
            match it {
                Item::W(_) | Item::Ph(_) => {
                    let p = ps[k].take().unwrap_or_default();
                    if p.is_empty() {
                        continue;
                    }
                    if !out.is_empty() && !out.ends_with(['“', '(']) {
                        out.push(' ');
                    }
                    out.push_str(&p);
                }
                Item::P(p) => {
                    if matches!(*p, "“" | "(") && !out.is_empty() {
                        out.push(' ');
                    }
                    out.push_str(p);
                }
            }
        }
        // Kokoro v1.0 symbol set: flap as T, glottal stop as t.
        out.replace('ɾ', "T").replace('ʔ', "t")
    }
}

// ---------------------------------------------------------------------------
// Letter-to-sound fallback for out-of-vocabulary words
// ---------------------------------------------------------------------------

/// (grapheme, left context, right context, phonemes). Contexts: `#` word
/// boundary, `V` vowel letter, `C` consonant letter, `+` e/i/y, `*` any;
/// otherwise a literal string.
const RULES: &[(&str, &str, &str, &str)] = &[
    ("tion", "*", "*", "ʃən"),
    ("sion", "V", "*", "ʒən"),
    ("sion", "*", "*", "ʃən"),
    ("cious", "*", "*", "ʃəs"),
    ("tious", "*", "*", "ʃəs"),
    ("ture", "*", "*", "ʧəɹ"),
    ("ough", "*", "t", "ɔ"),
    ("ough", "*", "*", "O"),
    ("augh", "*", "*", "ɔ"),
    ("eigh", "*", "*", "A"),
    ("igh", "*", "*", "I"),
    ("ph", "*", "*", "f"),
    ("sh", "*", "*", "ʃ"),
    ("ch", "#", "r", "k"),
    ("ch", "*", "*", "ʧ"),
    ("th", "*", "*", "θ"),
    ("wh", "#", "*", "w"),
    ("ck", "*", "*", "k"),
    ("ng", "*", "#", "ŋ"),
    ("nk", "*", "*", "ŋk"),
    ("qu", "*", "*", "kw"),
    ("gh", "#", "*", "ɡ"),
    ("gh", "*", "*", ""),
    ("kn", "#", "*", "n"),
    ("wr", "#", "*", "ɹ"),
    ("mb", "*", "#", "m"),
    ("dge", "*", "*", "ʤ"),
    ("tch", "*", "*", "ʧ"),
    ("ee", "*", "*", "i"),
    ("ea", "*", "*", "i"),
    ("oo", "*", "k", "ʊ"),
    ("oo", "*", "*", "u"),
    ("ou", "*", "*", "W"),
    ("ow", "*", "#", "O"),
    ("ow", "*", "*", "W"),
    ("oi", "*", "*", "Y"),
    ("oy", "*", "*", "Y"),
    ("ai", "*", "*", "A"),
    ("ay", "*", "*", "A"),
    ("au", "*", "*", "ɔ"),
    ("aw", "*", "*", "ɔ"),
    ("ew", "*", "*", "u"),
    ("ie", "*", "#", "I"),
    ("ie", "*", "*", "i"),
    ("ei", "*", "*", "A"),
    ("ey", "*", "#", "i"),
    ("oa", "*", "*", "O"),
    ("ue", "*", "#", "u"),
    ("ar", "*", "*", "ɑɹ"),
    ("or", "*", "*", "ɔɹ"),
    ("er", "*", "*", "əɹ"),
    ("ir", "*", "*", "ɜɹ"),
    ("ur", "*", "*", "ɜɹ"),
    ("a", "*", "C+", "A"),
    ("a", "*", "Ce#", "A"),
    ("a", "*", "*", "æ"),
    ("e", "C", "#", ""),
    ("e", "*", "C+", "i"),
    ("e", "*", "*", "ɛ"),
    ("i", "*", "Ce#", "I"),
    ("i", "*", "*", "ɪ"),
    ("o", "*", "Ce#", "O"),
    ("o", "*", "#", "O"),
    ("o", "*", "*", "ɑ"),
    ("u", "*", "Ce#", "u"),
    ("u", "*", "*", "ʌ"),
    ("y", "#", "*", "j"),
    ("y", "C", "#", "i"),
    ("y", "*", "*", "ɪ"),
    ("c", "*", "+", "s"),
    ("c", "*", "*", "k"),
    ("g", "*", "+", "ʤ"),
    ("g", "*", "*", "ɡ"),
    ("s", "V", "V", "z"),
    ("s", "*", "#", "z"),
    ("s", "*", "*", "s"),
    ("x", "#", "*", "z"),
    ("x", "*", "*", "ks"),
    ("j", "*", "*", "ʤ"),
    ("r", "*", "*", "ɹ"),
    ("w", "*", "*", "w"),
    ("h", "*", "*", "h"),
    ("b", "*", "*", "b"),
    ("d", "*", "*", "d"),
    ("f", "*", "*", "f"),
    ("k", "*", "*", "k"),
    ("l", "*", "*", "l"),
    ("m", "*", "*", "m"),
    ("n", "*", "*", "n"),
    ("p", "*", "*", "p"),
    ("q", "*", "*", "k"),
    ("t", "*", "*", "t"),
    ("v", "*", "*", "v"),
    ("z", "*", "*", "z"),
];

fn is_vowel_letter(c: char) -> bool {
    "aeiouy".contains(c)
}

fn ctx_matches(ctx: &str, s: &[char], left: bool) -> bool {
    if ctx == "*" {
        return true;
    }
    // Contexts are matched outward from the grapheme.
    let pat: Vec<char> = if left {
        ctx.chars().rev().collect()
    } else {
        ctx.chars().collect()
    };
    let mut pos = 0usize;
    for p in pat {
        let c = s.get(pos).copied();
        let ok = match p {
            '#' => c.is_none(),
            'V' => c.is_some_and(is_vowel_letter),
            'C' => c.is_some_and(|c| c.is_alphabetic() && !is_vowel_letter(c)),
            '+' => c.is_some_and(|c| "eiy".contains(c)),
            lit => c == Some(lit),
        };
        if !ok {
            return false;
        }
        if p != '#' {
            pos += 1;
        }
    }
    true
}

/// Rule-based letter-to-sound conversion with a simple stress heuristic.
pub fn lts(word: &str, british: bool) -> String {
    let chars: Vec<char> = word.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let mut matched = false;
        for (g, l, r, ph) in RULES {
            let gc: Vec<char> = g.chars().collect();
            if chars.len() < i + gc.len() || chars[i..i + gc.len()] != gc[..] {
                continue;
            }
            let left: Vec<char> = chars[..i].iter().rev().copied().collect();
            let right: Vec<char> = chars[i + gc.len()..].to_vec();
            if ctx_matches(l, &left, true) && ctx_matches(r, &right, false) {
                out.push_str(ph);
                i += gc.len();
                matched = true;
                break;
            }
        }
        if !matched {
            i += 1;
        }
    }
    if british {
        out = out
            .replace("ɑɹ", "ɑː")
            .replace("ɔɹ", "ɔː")
            .replace("əɹ", "ə")
            .replace("ɜɹ", "ɜː")
            .replace('O', "Q");
    }
    // Stress: first vowel, or the vowel before -tion/-ic style endings.
    let vowel_pos: Vec<usize> = out
        .char_indices()
        .filter(|(_, c)| is_vowel(*c))
        .map(|(i, _)| i)
        .collect();
    if vowel_pos.is_empty() {
        return out;
    }
    let target = if (word.ends_with("tion")
        || word.ends_with("sion")
        || word.ends_with("ic")
        || word.ends_with("ical"))
        && vowel_pos.len() >= 2
    {
        vowel_pos[vowel_pos.len() - 2]
    } else if [
        "be", "de", "re", "con", "com", "dis", "ex", "un", "pre", "pro",
    ]
    .iter()
    .any(|p| word.starts_with(p) && word.len() > p.len() + 3)
        && vowel_pos.len() >= 2
    {
        vowel_pos[1]
    } else {
        vowel_pos[0]
    };
    format!("{}{PRIMARY}{}", &out[..target], &out[target..])
}

#[cfg(all(test, feature = "embed-lexicons"))]
mod tests {
    use super::*;

    #[test]
    fn american() {
        let g = G2p::new(Lexicon::embedded(false));
        assert_eq!(g.phonemize("Hello world!"), "həlˈO wˈɜɹld!");
        assert_eq!(
            g.phonemize("the apple and the pear"),
            "ði ˈæpᵊl ænd ðə pˈɛɹ"
        );
        assert_eq!(
            g.phonemize("It costs $5.50."),
            "ˌɪt kˈɔsts fˈIv dˈɑləɹz ænd fˈɪfti sˈɛnts."
        );
        assert_eq!(g.phonemize("She said \"hi\" now"), "ʃˌi sˈɛd “hˈI” nˈW");
        assert!(g.phonemize("Inhouse").starts_with("ˈɪn"));
        assert_eq!(g.phonemize("in 2026"), "ɪn twˈɛnti twˈɛnti sˈɪks");
        assert_eq!(g.phonemize("water"), "wˈɔTəɹ");
        assert_eq!(
            g.phonemize("Mr. Smith is in the U.S."),
            "mˈɪstəɹ smˈɪθ ɪz ɪn ðə jˌuˈɛs."
        );
    }

    #[test]
    fn british() {
        let g = G2p::new(Lexicon::embedded(true));
        assert_eq!(g.phonemize("water"), "wˈɔːtə");
    }
}
