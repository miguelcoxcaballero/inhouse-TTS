//! Spanish front end: normalisation, rule-based G2P and regional accents.
//!
//! The base transcription follows the conventions the Spanish K82 voices
//! were trained on (espeak-ng `es` style: stress mark before the vowel,
//! β/ð/ɣ allophones, falling diphthongs as `I`/`A`/`W`). Regional accents
//! are phonological rewrites on top of it.

use crate::phone::{add_secondary_stress, render, Kind, Ph, Word};
use crate::scan::{is_pause, scan, Quotes, Tok};

/// Realisation of orthographic `ll` / consonantal `y`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ll {
    /// ll = [ʎ], y = [ʝ] (conservative).
    Lleismo,
    /// ll = y = [ʝ].
    Yeismo,
    /// ll = y = [ʃ] (Río de la Plata).
    Sheismo,
}

/// Realisation of `j` / `g(e,i)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Jota {
    Velar,
    /// Aspirated [h] (Caribbean, Colombia, Andalusia, Canaries).
    Glottal,
    /// [ç] before front vowels (Chile).
    Fronted,
}

/// Treatment of syllable-final /s/.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodaS {
    Keep,
    /// [h] before consonants.
    Aspirate,
    /// [h] before consonants and pauses.
    AspirateFinal,
}

/// A Spanish regional accent as a bundle of phonological features.
#[derive(Clone, Copy, Debug)]
pub struct Accent {
    pub code: &'static str,
    pub name: &'static str,
    /// θ/s distinction (Spain); `false` means seseo.
    pub distincion: bool,
    pub ll: Ll,
    pub jota: Jota,
    pub coda_s: CodaS,
    /// `-ado` → `-ao`.
    pub d_elision: bool,
    /// Word-final /d/ dropped (verdad → verdá).
    pub final_d_drop: bool,
    /// Word-final /n/ velarised to [ŋ].
    pub final_n_velar: bool,
    /// Syllable-final /ɾ/ → [l] (Puerto Rico).
    pub coda_r_to_l: bool,
    /// ch = [ʃ].
    pub ch_fricative: bool,
    /// Decimal separator is a comma (thousands with dots).
    pub decimal_comma: bool,
    /// Currency read for `$`.
    pub dollar: (&'static str, &'static str),
}

const BASE: Accent = Accent {
    code: "es-ES",
    name: "Español (España)",
    distincion: true,
    ll: Ll::Yeismo,
    jota: Jota::Velar,
    coda_s: CodaS::Keep,
    d_elision: false,
    final_d_drop: false,
    final_n_velar: false,
    coda_r_to_l: false,
    ch_fricative: false,
    decimal_comma: true,
    dollar: ("dólar", "dólares"),
};

const PESO: (&str, &str) = ("peso", "pesos");

/// All supported Spanish accents.
pub const ACCENTS: &[Accent] = &[
    BASE,
    Accent {
        code: "es-MX",
        name: "Español (México)",
        distincion: false,
        decimal_comma: false,
        dollar: PESO,
        ..BASE
    },
    Accent {
        code: "es-AR",
        name: "Español (Argentina, rioplatense)",
        distincion: false,
        ll: Ll::Sheismo,
        coda_s: CodaS::Aspirate,
        dollar: PESO,
        ..BASE
    },
    Accent {
        code: "es-CO",
        name: "Español (Colombia)",
        distincion: false,
        jota: Jota::Glottal,
        dollar: PESO,
        ..BASE
    },
    Accent {
        code: "es-CL",
        name: "Español (Chile)",
        distincion: false,
        jota: Jota::Fronted,
        coda_s: CodaS::AspirateFinal,
        d_elision: true,
        dollar: PESO,
        ..BASE
    },
    Accent {
        code: "es-PE",
        name: "Español (Perú)",
        distincion: false,
        decimal_comma: false,
        dollar: ("sol", "soles"),
        ..BASE
    },
    Accent {
        code: "es-VE",
        name: "Español (Venezuela)",
        distincion: false,
        jota: Jota::Glottal,
        coda_s: CodaS::Aspirate,
        final_n_velar: true,
        d_elision: true,
        ..BASE
    },
    Accent {
        code: "es-CU",
        name: "Español (Caribe: Cuba, R. Dominicana)",
        distincion: false,
        jota: Jota::Glottal,
        coda_s: CodaS::AspirateFinal,
        d_elision: true,
        final_d_drop: true,
        final_n_velar: true,
        decimal_comma: false,
        dollar: PESO,
        ..BASE
    },
    Accent {
        code: "es-PR",
        name: "Español (Puerto Rico)",
        distincion: false,
        jota: Jota::Glottal,
        coda_s: CodaS::AspirateFinal,
        d_elision: true,
        final_n_velar: true,
        coda_r_to_l: true,
        decimal_comma: false,
        ..BASE
    },
    Accent {
        code: "es-AN",
        name: "Español (Andalucía)",
        distincion: false,
        jota: Jota::Glottal,
        coda_s: CodaS::AspirateFinal,
        d_elision: true,
        final_d_drop: true,
        ..BASE
    },
    Accent {
        code: "es-IC",
        name: "Español (Canarias)",
        distincion: false,
        jota: Jota::Glottal,
        coda_s: CodaS::Aspirate,
        d_elision: true,
        ..BASE
    },
];

pub fn accent(code: &str) -> Option<&'static Accent> {
    let code = code.to_ascii_lowercase().replace('_', "-");
    ACCENTS
        .iter()
        .find(|a| a.code.to_ascii_lowercase() == code)
        .or_else(|| match code.as_str() {
            "es" | "es-es" => Some(&ACCENTS[0]),
            "es-uy" => accent("es-AR"),
            "es-do" => accent("es-CU"),
            "es-us" | "es-419" | "es-gt" | "es-hn" | "es-sv" | "es-ni" | "es-cr" => accent("es-MX"),
            "es-ec" | "es-bo" => accent("es-PE"),
            "es-pa" => accent("es-VE"),
            "es-py" => accent("es-AR"),
            _ => None,
        })
}

// ---------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------

const UNITS: [&str; 30] = [
    "cero",
    "uno",
    "dos",
    "tres",
    "cuatro",
    "cinco",
    "seis",
    "siete",
    "ocho",
    "nueve",
    "diez",
    "once",
    "doce",
    "trece",
    "catorce",
    "quince",
    "dieciséis",
    "diecisiete",
    "dieciocho",
    "diecinueve",
    "veinte",
    "veintiuno",
    "veintidós",
    "veintitrés",
    "veinticuatro",
    "veinticinco",
    "veintiséis",
    "veintisiete",
    "veintiocho",
    "veintinueve",
];
const TENS: [&str; 10] = [
    "",
    "",
    "veinte",
    "treinta",
    "cuarenta",
    "cincuenta",
    "sesenta",
    "setenta",
    "ochenta",
    "noventa",
];
const HUNDREDS: [&str; 10] = [
    "",
    "ciento",
    "doscientos",
    "trescientos",
    "cuatrocientos",
    "quinientos",
    "seiscientos",
    "setecientos",
    "ochocientos",
    "novecientos",
];

fn below_thousand(n: u64, apocope: bool) -> String {
    debug_assert!(n < 1000);
    let mut parts = Vec::new();
    let h = n / 100;
    let r = n % 100;
    if h > 0 {
        parts.push(if n == 100 {
            "cien".to_string()
        } else {
            HUNDREDS[h as usize].to_string()
        });
    }
    if r > 0 {
        let w = if r < 30 {
            UNITS[r as usize].to_string()
        } else {
            let t = TENS[(r / 10) as usize];
            match r % 10 {
                0 => t.to_string(),
                u => format!("{t} y {}", UNITS[u as usize]),
            }
        };
        parts.push(w);
    }
    let mut s = parts.join(" ");
    if apocope {
        if s.ends_with("veintiuno") {
            s = format!("{}ún", &s[..s.len() - 3]);
        } else if s.ends_with("uno") {
            s.truncate(s.len() - 1);
        }
    }
    s
}

/// Cardinal number in words. `apocope` gives "un"/"veintiún" (before nouns).
pub fn cardinal(n: u64, apocope: bool) -> String {
    if n == 0 {
        return "cero".into();
    }
    let groups = [
        (1_000_000_000_000u64, "billón", "billones"),
        (1_000_000, "millón", "millones"),
    ];
    let mut n = n;
    let mut parts: Vec<String> = Vec::new();
    for (size, one, many) in groups {
        if n >= size {
            let q = n / size;
            n %= size;
            if q == 1 {
                parts.push(format!("un {one}"));
            } else {
                parts.push(format!("{} {many}", cardinal(q, true)));
            }
        }
    }
    if n >= 1000 {
        let q = n / 1000;
        n %= 1000;
        if q == 1 {
            parts.push("mil".into());
        } else {
            parts.push(format!("{} mil", below_thousand(q, true)));
        }
    }
    if n > 0 {
        parts.push(below_thousand(n, apocope));
    }
    parts.join(" ")
}

const ORDINALS_M: [&str; 11] = [
    "", "primero", "segundo", "tercero", "cuarto", "quinto", "sexto", "séptimo", "octavo",
    "noveno", "décimo",
];

fn ordinal(n: u64, feminine: bool, apocope: bool) -> String {
    if (1..=10).contains(&n) {
        let mut w = ORDINALS_M[n as usize].to_string();
        if feminine {
            w.pop();
            w.push('a');
        } else if apocope && (n == 1 || n == 3) {
            w.pop();
        }
        w
    } else {
        cardinal(n, false)
    }
}

const MONTHS: [&str; 12] = [
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
];

fn digits_words(d: &str) -> String {
    d.chars()
        .filter_map(|c| c.to_digit(10))
        .map(|v| UNITS[v as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

/// Splits a numeral into integer and fractional digits. Thousands groups
/// are exactly three digits; any other separator is a decimal mark.
fn split_number(num: &str, dec: char, thou: char) -> (String, Option<String>) {
    let segs: Vec<&str> = num.split(['.', ',']).collect();
    let seps: Vec<char> = num.chars().filter(|c| *c == '.' || *c == ',').collect();
    let mut int = segs[0].to_string();
    for (k, sep) in seps.iter().enumerate() {
        let seg = segs[k + 1];
        let last = k + 1 == seps.len();
        if *sep == thou && seg.len() == 3 {
            int.push_str(seg);
        } else if *sep == dec || last {
            return (int, Some(segs[k + 1..].concat()));
        } else {
            int.push_str(seg);
        }
    }
    (int, None)
}

fn parse_u64(s: &str) -> Option<u64> {
    if s.is_empty() || s.len() > 15 {
        return None;
    }
    s.parse().ok()
}

/// Expands a numeric token to words.
fn number_words(num: &str, acc: &Accent, next_is_word: bool) -> String {
    // Times: 10:30
    if let Some((h, m)) = num.split_once(':') {
        if let (Some(h), Some(m)) = (parse_u64(h), parse_u64(m)) {
            if h < 24 && m < 60 {
                let mut hw = cardinal(h, false);
                if hw.ends_with("uno") {
                    hw.truncate(hw.len() - 1);
                    hw.push('a');
                }
                return match m {
                    0 => format!("{hw} en punto"),
                    15 => format!("{hw} y cuarto"),
                    30 => format!("{hw} y media"),
                    _ => format!("{hw} y {}", cardinal(m, false)),
                };
            }
        }
    }
    // Dates: 29/10/2026
    let parts: Vec<&str> = num.split('/').collect();
    if parts.len() == 3 {
        if let (Some(d), Some(m), Some(y)) = (
            parse_u64(parts[0]),
            parse_u64(parts[1]),
            parse_u64(parts[2]),
        ) {
            if (1..=31).contains(&d) && (1..=12).contains(&m) {
                let y = if parts[2].len() == 2 { 2000 + y } else { y };
                let day = if d == 1 {
                    "primero".to_string()
                } else {
                    cardinal(d, false)
                };
                return format!(
                    "{day} de {} de {}",
                    MONTHS[(m - 1) as usize],
                    cardinal(y, false)
                );
            }
        }
    }
    if parts.len() == 2 {
        if let (Some(a), Some(b)) = (parse_u64(parts[0]), parse_u64(parts[1])) {
            return format!("{} entre {}", cardinal(a, false), cardinal(b, false));
        }
    }
    let (dec_sep, thou_sep) = if acc.decimal_comma {
        (',', '.')
    } else {
        ('.', ',')
    };
    let (int_part, frac_part) = split_number(num, dec_sep, thou_sep);
    let Some(n) = parse_u64(&int_part) else {
        return digits_words(num);
    };
    let mut words = if int_part.len() > 1 && int_part.starts_with('0') {
        digits_words(&int_part)
    } else {
        cardinal(n, next_is_word && frac_part.is_none())
    };
    if let Some(f) = frac_part {
        words.push_str(" coma ");
        if f.starts_with('0') || f.len() > 3 {
            words.push_str(&digits_words(&f));
        } else {
            words.push_str(&cardinal(parse_u64(&f).unwrap_or(0), false));
        }
    }
    words
}

const ABBREVIATIONS: &[(&str, &str)] = &[
    ("sr", "señor"),
    ("sra", "señora"),
    ("srta", "señorita"),
    ("sres", "señores"),
    ("dr", "doctor"),
    ("dra", "doctora"),
    ("ud", "usted"),
    ("uds", "ustedes"),
    ("vd", "usted"),
    ("etc", "etcétera"),
    ("pág", "página"),
    ("núm", "número"),
    ("tel", "teléfono"),
    ("avda", "avenida"),
    ("av", "avenida"),
    ("aprox", "aproximadamente"),
    ("p.ej", "por ejemplo"),
    ("km", "kilómetros"),
    ("kg", "kilos"),
    ("cm", "centímetros"),
    ("mm", "milímetros"),
    ("ml", "mililitros"),
    ("min", "minutos"),
    ("seg", "segundos"),
];

fn letter_name(c: char) -> &'static str {
    match c {
        'a' => "a",
        'b' => "be",
        'c' => "ce",
        'd' => "de",
        'e' => "e",
        'f' => "efe",
        'g' => "ge",
        'h' => "hache",
        'i' => "i",
        'j' => "jota",
        'k' => "ka",
        'l' => "ele",
        'm' => "eme",
        'n' => "ene",
        'ñ' => "eñe",
        'o' => "o",
        'p' => "pe",
        'q' => "cu",
        'r' => "erre",
        's' => "ese",
        't' => "te",
        'u' => "u",
        'v' => "uve",
        'w' => "uve doble",
        'x' => "equis",
        'y' => "ye",
        'z' => "zeta",
        _ => "",
    }
}

fn is_vowel_letter(c: char) -> bool {
    "aeiouáéíóúü".contains(c)
}

/// All-caps tokens without a vowel pattern are spelled (DNI, PP, BBVA).
fn is_spelled_acronym(w: &str) -> bool {
    let n = w.chars().count();
    if n < 2 || n > 6 || !w.chars().all(|c| c.is_uppercase()) {
        return false;
    }
    let lower: Vec<char> = w.to_lowercase().chars().collect();
    let vowels = lower.iter().filter(|c| is_vowel_letter(**c)).count();
    vowels == 0
        || n == 2
        || lower
            .windows(3)
            .any(|t| t.iter().all(|c| !is_vowel_letter(*c)))
}

/// Normalises raw text into a sequence of plain lower-case words and
/// punctuation tokens.
fn normalize(text: &str, acc: &Accent) -> Vec<Tok> {
    let toks = scan(text);
    let mut out: Vec<Tok> = Vec::new();
    let push_words = |out: &mut Vec<Tok>, s: &str| {
        for w in s.split_whitespace() {
            out.push(Tok::Word(w.to_string()));
        }
    };
    let is_money = |c: &char| matches!(c, '€' | '$' | '£');
    let mut i = 0;
    while i < toks.len() {
        match &toks[i] {
            Tok::Word(w) => {
                let lw = w.to_lowercase();
                let dot_next = matches!(toks.get(i + 1), Some(Tok::Punct('.')));
                if lw == "ee"
                    && dot_next
                    && matches!(toks.get(i + 2), Some(Tok::Word(u)) if u.eq_ignore_ascii_case("uu"))
                {
                    push_words(&mut out, "estados unidos");
                    i += 3;
                    if matches!(toks.get(i), Some(Tok::Punct('.'))) {
                        i += 1;
                    }
                    continue;
                }
                let unit = ["km", "kg", "cm", "mm", "ml"].contains(&lw.as_str());
                if let Some((_, e)) = ABBREVIATIONS
                    .iter()
                    .find(|(a, _)| *a == lw)
                    .filter(|_| dot_next || unit)
                {
                    push_words(&mut out, e);
                    if dot_next {
                        // The dot also ends the sentence when a capital or the end follows.
                        let ends = match toks.get(i + 2) {
                            None => true,
                            Some(Tok::Word(n)) => n.chars().next().is_some_and(char::is_uppercase),
                            Some(_) => false,
                        };
                        i += if ends { 1 } else { 2 };
                    } else {
                        i += 1;
                    }
                    continue;
                }
                let no_vowel = !lw.chars().any(is_vowel_letter) && !lw.contains('y');
                if is_spelled_acronym(w) || (no_vowel && lw.chars().count() <= 4) {
                    let spelled: Vec<&str> = lw
                        .chars()
                        .map(letter_name)
                        .filter(|s| !s.is_empty())
                        .collect();
                    push_words(&mut out, &spelled.join(" "));
                } else {
                    for part in lw.split('-') {
                        if !part.is_empty() {
                            out.push(Tok::Word(part.replace('\'', "")));
                        }
                    }
                }
            }
            Tok::Num(n) => {
                let next = toks.get(i + 1);
                let prev_money = if i > 0 {
                    if let Tok::Sym(c) = &toks[i - 1] {
                        Some(*c).filter(is_money)
                    } else {
                        None
                    }
                } else {
                    None
                };
                let next_word = matches!(next, Some(Tok::Word(_)));
                let simple = parse_u64(n);
                let consumed = match next {
                    Some(Tok::Sym('%')) => {
                        push_words(
                            &mut out,
                            &format!("{} por ciento", number_words(n, acc, false)),
                        );
                        true
                    }
                    Some(Tok::Sym('°')) => {
                        push_words(&mut out, &format!("{} grados", number_words(n, acc, false)));
                        true
                    }
                    Some(Tok::Sym('º')) if simple.is_some() => {
                        let apocope = matches!(toks.get(i + 2), Some(Tok::Word(_)));
                        push_words(&mut out, &ordinal(simple.unwrap(), false, apocope));
                        true
                    }
                    Some(Tok::Sym('ª')) if simple.is_some() => {
                        push_words(&mut out, &ordinal(simple.unwrap(), true, false));
                        true
                    }
                    Some(Tok::Sym(c)) if is_money(c) => {
                        push_words(&mut out, &money(n, *c, acc));
                        true
                    }
                    _ => {
                        match prev_money {
                            Some(c) => push_words(&mut out, &money(n, c, acc)),
                            None => push_words(&mut out, &number_words(n, acc, next_word)),
                        }
                        false
                    }
                };
                i += if consumed { 2 } else { 1 };
                continue;
            }
            Tok::Sym(c) => {
                let next_is_num = matches!(toks.get(i + 1), Some(Tok::Num(_)));
                match c {
                    c if is_money(c) && next_is_num => {}
                    '%' => push_words(&mut out, "por ciento"),
                    '&' => push_words(&mut out, "y"),
                    '+' => push_words(&mut out, "más"),
                    '=' => push_words(&mut out, "igual a"),
                    '@' => push_words(&mut out, "arroba"),
                    '€' => push_words(&mut out, "euros"),
                    '$' => push_words(&mut out, acc.dollar.1),
                    _ => {}
                }
            }
            Tok::Punct(p) => out.push(Tok::Punct(*p)),
        }
        i += 1;
    }
    out
}

fn money(num: &str, sym: char, acc: &Accent) -> String {
    let (one, many) = match sym {
        '€' => ("euro", "euros"),
        '£' => ("libra", "libras"),
        _ => acc.dollar,
    };
    let (dec_sep, thou_sep) = if acc.decimal_comma {
        (',', '.')
    } else {
        ('.', ',')
    };
    let (int, frac) = split_number(num, dec_sep, thou_sep);
    let cents = frac.filter(|f| f.len() == 2).and_then(|f| parse_u64(&f));
    let n = parse_u64(&int).unwrap_or(0);
    let mut s = format!("{} {}", cardinal(n, true), if n == 1 { one } else { many });
    if let Some(c) = cents.filter(|c| *c > 0) {
        s.push_str(&format!(" con {}", cardinal(c, false)));
    }
    s
}

// ---------------------------------------------------------------------------
// Grapheme to phoneme
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum U {
    /// Vowel letter: symbol, strong (a/e/o or accented i/u), written accent.
    V(&'static str, bool, bool),
    /// Word-final `y` after a vowel (falling glide).
    FinalY,
    C(&'static str),
}

fn base_vowel(c: char) -> Option<(&'static str, bool, bool)> {
    Some(match c {
        'a' | 'à' | 'â' | 'ä' => ("a", true, false),
        'á' => ("a", true, true),
        'e' | 'è' | 'ê' | 'ë' => ("e", true, false),
        'é' => ("e", true, true),
        'o' | 'ò' | 'ô' | 'ö' => ("o", true, false),
        'ó' => ("o", true, true),
        'i' | 'ì' | 'î' | 'ï' => ("i", false, false),
        'í' => ("i", true, true),
        'u' | 'ù' | 'û' | 'ü' => ("u", false, false),
        'ú' => ("u", true, true),
        _ => return None,
    })
}

fn is_front(c: Option<&char>) -> bool {
    matches!(c, Some('e' | 'é' | 'i' | 'í' | 'ê' | 'è' | 'ë' | 'î' | 'ï'))
}

fn letters_to_units(word: &str) -> Vec<U> {
    let ch: Vec<char> = word.chars().collect();
    let n = ch.len();
    let mut u = Vec::with_capacity(n + 2);
    let mut i = 0;
    // Silent initial consonants of learned clusters.
    if n > 2 {
        let two: String = ch[..2].iter().collect();
        if ["ps", "gn", "pn", "mn", "pt"].contains(&two.as_str()) && base_vowel(ch[2]).is_some()
            || two == "ps"
        {
            i = 1;
        }
    }
    let x_as_jota = [
        "méxic", "mexic", "oaxac", "texas", "texan", "xavier", "ximena",
    ]
    .iter()
    .any(|p| word.starts_with(p));
    while i < n {
        let c = ch[i];
        let next = ch.get(i + 1);
        if let Some((s, strong, acc)) = base_vowel(c) {
            u.push(U::V(s, strong, acc));
            i += 1;
            continue;
        }
        let prev_vowel = i > 0 && base_vowel(ch[i - 1]).is_some();
        match c {
            'y' => {
                if n == 1 {
                    u.push(U::V("i", false, false));
                } else if i + 1 == n && prev_vowel {
                    u.push(U::FinalY);
                } else if next.is_some_and(|c| base_vowel(*c).is_some()) {
                    u.push(U::C("ʝ"));
                } else {
                    u.push(U::V("i", false, false));
                }
            }
            'h' => {}
            'c' => {
                if next == Some(&'k') {
                    // Loanword "ck" = /k/.
                } else if next == Some(&'h') {
                    u.push(U::C("ʧ"));
                    i += 1;
                } else if is_front(next) {
                    u.push(U::C("θ"));
                } else {
                    u.push(U::C("k"));
                }
            }
            'q' => {
                u.push(U::C("k"));
                if next == Some(&'u') {
                    i += 1;
                }
            }
            'g' => {
                if is_front(next) {
                    u.push(U::C("x"));
                } else if next == Some(&'u') && is_front(ch.get(i + 2)) {
                    u.push(U::C("ɡ"));
                    i += 1;
                } else {
                    u.push(U::C("ɡ"));
                }
            }
            'j' => u.push(U::C("x")),
            'l' => {
                if next == Some(&'l') {
                    u.push(U::C("ʎ"));
                    i += 1;
                } else {
                    u.push(U::C("l"));
                }
            }
            'r' => {
                if next == Some(&'r') {
                    u.push(U::C("r"));
                    i += 1;
                } else if i == 0 || matches!(ch.get(i.wrapping_sub(1)), Some('l' | 'n' | 's')) {
                    u.push(U::C("r"));
                } else {
                    u.push(U::C("ɾ"));
                }
            }
            'ñ' => u.push(U::C("ɲ")),
            'n' => {
                if next == Some(&'y') && ch.get(i + 2).is_some_and(|c| base_vowel(*c).is_some()) {
                    u.push(U::C("ɲ"));
                    i += 1;
                } else {
                    u.push(U::C("n"));
                }
            }
            'b' | 'v' => u.push(U::C("b")),
            'w' => u.push(U::V("u", false, false)),
            'x' => {
                if i == 0 {
                    u.push(U::C("s"));
                } else if x_as_jota {
                    u.push(U::C("x"));
                } else {
                    u.push(U::C("k"));
                    u.push(U::C("s"));
                }
            }
            'z' => u.push(U::C("θ")),
            'ç' => u.push(U::C("s")),
            'k' => u.push(U::C("k")),
            'p' => u.push(U::C("p")),
            't' => u.push(U::C("t")),
            'f' => u.push(U::C("f")),
            'm' => u.push(U::C("m")),
            's' => u.push(U::C("s")),
            'd' => u.push(U::C("d")),
            _ => {}
        }
        i += 1;
    }
    u
}

/// Monosyllables and clitics that carry no stress.
const UNSTRESSED: &[&str] = &[
    "el", "la", "los", "las", "lo", "le", "les", "me", "te", "se", "nos", "os", "mi", "mis", "tu",
    "tus", "su", "sus", "de", "del", "a", "al", "en", "con", "por", "y", "e", "o", "u", "ni",
    "que", "si", "sin", "tan", "pues",
];

/// Function words realised with secondary stress.
const WEAK: &[&str] = &[
    "como", "cuando", "donde", "pero", "para", "porque", "aunque", "mientras", "sobre", "entre",
    "hacia", "desde", "nuestro", "nuestra", "nuestros", "nuestras", "cuanto", "cuanta", "tras",
    "mas", "cual", "quien",
];

/// Common English loanwords and brand names, as pronounced in Spanish.
const LOANWORDS: &[(&str, &str)] = &[
    ("inhouse", "ˈinxaws"),
    ("rust", "rˈast"),
    ("software", "sˈofwer"),
    ("hardware", "xˈaɾwer"),
    ("online", "onlˈain"),
    ("offline", "oflˈain"),
    ("email", "imˈeil"),
    ("app", "ˈap"),
    ("apps", "ˈaps"),
    ("android", "ˈandɾoid"),
    ("iphone", "ˈaifon"),
    ("google", "ɡˈuɣel"),
    ("youtube", "ʝutˈuβ"),
    ("whatsapp", "wˈatsap"),
    ("facebook", "fˈeisβuk"),
    ("instagram", "ˈinstaɣɾam"),
    ("github", "ɡˈitxaβ"),
    ("windows", "wˈindows"),
    ("linux", "lˈinuks"),
    ("ok", "okˈei"),
    ("okay", "okˈei"),
    ("show", "ʧˈow"),
    ("streaming", "estɾˈimin"),
    ("podcast", "pˈoðkast"),
    ("smartphone", "esmˈaɾtfon"),
    ("chat", "ʧˈat"),
    ("pizza", "pˈitsa"),
    ("jazz", "ʝˈas"),
    ("marketing", "mˈaɾketin"),
];

fn loanword(word: &str) -> Option<Vec<Ph>> {
    let (_, ipa) = LOANWORDS.iter().find(|(w, _)| *w == word)?;
    let mut out = Vec::new();
    let mut stress = 0u8;
    for c in ipa.chars() {
        let s: &'static str = match c {
            'ˈ' => {
                stress = 2;
                continue;
            }
            'a' => "a",
            'e' => "e",
            'i' => "i",
            'o' => "o",
            'u' => "u",
            'w' => "w",
            'j' => "j",
            'b' => "b",
            'β' => "β",
            'd' => "d",
            'ð' => "ð",
            'ɡ' => "ɡ",
            'ɣ' => "ɣ",
            'f' => "f",
            'k' => "k",
            'l' => "l",
            'm' => "m",
            'n' => "n",
            'p' => "p",
            'r' => "r",
            'ɾ' => "ɾ",
            's' => "s",
            't' => "t",
            'x' => "x",
            'ʧ' => "ʧ",
            'ʝ' => "ʝ",
            _ => continue,
        };
        let mut p = if "aeiou".contains(s) {
            Ph::v(s)
        } else if s == "w" || s == "j" {
            Ph::g(s)
        } else {
            Ph::c(s)
        };
        if p.is_vowel() && stress > 0 {
            p.stress = stress;
            stress = 0;
        }
        out.push(p);
    }
    Some(out)
}

/// Phonemises one normalised (lower-case) word.
fn word_phones(word: &str) -> Vec<Ph> {
    if let Some(p) = loanword(word) {
        return p;
    }
    // -mente adverbs keep the adjective stress plus a stress on "men".
    if word.len() > 7 && word.ends_with("mente") {
        let stem = &word[..word.len() - 5];
        if stem.chars().count() >= 3 {
            let mut a = word_phones(stem);
            let mut b = word_phones("mente");
            for p in &mut b {
                if p.s == "ɛ" {
                    p.s = "e";
                }
            }
            a.extend(b);
            return a;
        }
    }
    let units = letters_to_units(word);
    // Vowel groups → nuclei / glides.
    #[derive(Clone, Copy, PartialEq)]
    enum Role {
        Cons,
        Nucleus,
        Rising,
        Falling,
    }
    let mut roles = vec![Role::Cons; units.len()];
    let mut i = 0;
    while i < units.len() {
        if !matches!(units[i], U::V(..) | U::FinalY) {
            i += 1;
            continue;
        }
        let start = i;
        while i < units.len() && matches!(units[i], U::V(..) | U::FinalY) {
            i += 1;
        }
        let group = start..i;
        let strong: Vec<usize> = group
            .clone()
            .filter(|&k| matches!(units[k], U::V(_, true, _)))
            .collect();
        if strong.is_empty() {
            // Only weak vowels: the last one is the nucleus (ciu-, cui-),
            // except a final y which is always a glide (muy).
            let mut last = group.end - 1;
            if matches!(units[last], U::FinalY) && last > group.start {
                last -= 1;
            }
            let after_liquid =
                group.start > 0 && matches!(units[group.start - 1], U::C("l" | "ɾ" | "r"));
            for k in group.clone() {
                roles[k] = if k == last || (after_liquid && k + 1 == last) {
                    Role::Nucleus
                } else if k < last {
                    Role::Rising
                } else {
                    Role::Falling
                };
            }
        } else {
            for k in group.clone() {
                if strong.contains(&k) {
                    roles[k] = Role::Nucleus;
                }
            }
            for k in group.clone() {
                if roles[k] == Role::Cons {
                    // Weak vowel: falling if it follows a nucleus, else rising;
                    // after l/r it stays syllabic (espeak: lu.e.go, pru.e.ba).
                    let follows = k > group.start && roles[k - 1] == Role::Nucleus;
                    let prev = if k == group.start && k > 0 {
                        Some(units[k - 1])
                    } else {
                        None
                    };
                    let cluster = k >= 2 && matches!(units[k - 2], U::C(_));
                    let after_liquid = match (units[k], prev) {
                        (U::V("u", ..), Some(U::C("l" | "ɾ" | "r"))) => true,
                        (U::V("i", ..), Some(U::C("r"))) => true,
                        (U::V("i", ..), Some(U::C("l" | "ɾ"))) => cluster,
                        _ => false,
                    };
                    roles[k] = if follows {
                        Role::Falling
                    } else if after_liquid && matches!(units[k], U::V(..)) {
                        Role::Nucleus
                    } else {
                        Role::Rising
                    };
                }
            }
        }
    }
    let nuclei: Vec<usize> = (0..units.len())
        .filter(|&k| roles[k] == Role::Nucleus)
        .collect();
    // Stress position.
    let accented = nuclei
        .iter()
        .position(|&k| matches!(units[k], U::V(_, _, true)));
    let stress = accented.unwrap_or_else(|| {
        let last = word.chars().last().unwrap_or('a');
        let paroxytone = "aeiouns".contains(last);
        if nuclei.len() >= 2 && paroxytone {
            nuclei.len() - 2
        } else {
            nuclei.len().saturating_sub(1)
        }
    });
    let level = if UNSTRESSED.contains(&word) {
        0
    } else if WEAK.contains(&word) {
        1
    } else {
        2
    };
    let mut out: Vec<Ph> = Vec::with_capacity(units.len() + 2);
    let mut nucleus_idx = 0;
    for (k, unit) in units.iter().enumerate() {
        match (unit, roles[k]) {
            (U::C(s), _) => out.push(Ph::c(s)),
            (U::V(s, ..), Role::Nucleus) => {
                let mut p = Ph::v(s);
                if nucleus_idx == stress && !nuclei.is_empty() {
                    p.stress = level;
                }
                nucleus_idx += 1;
                out.push(p);
            }
            (U::V(s, ..), Role::Rising) => out.push(Ph::g(if *s == "i" { "j" } else { "w" })),
            (U::V(..), _) | (U::FinalY, _) => {
                let is_i = matches!(unit, U::FinalY | U::V("i", ..));
                // Falling glide: merge with the preceding nucleus.
                let prev = out.last_mut();
                match prev {
                    Some(p) if p.kind == Kind::Vowel && p.s != "u" && p.s != "i" => {
                        let merged = match (p.s, is_i) {
                            ("a", true) => Some("I"),
                            ("e", true) => Some("A"),
                            ("a", false) => Some("W"),
                            _ => None,
                        };
                        match merged {
                            Some(m) => p.s = m,
                            None => out.push(Ph::g(if is_i { "ɪ" } else { "ʊ" })),
                        }
                    }
                    _ => out.push(Ph::g(if is_i { "j" } else { "w" })),
                }
            }
        }
    }
    if level == 2 {
        add_secondary_stress(&mut out);
    }
    // espeak opens a stressed penultimate /e/ closed by /n/ (tengo, cliente).
    let vowels: Vec<usize> = (0..out.len()).filter(|&k| out[k].is_vowel()).collect();
    if vowels.len() >= 2 {
        let k = vowels[vowels.len() - 2];
        let n1 = out.get(k + 1).map(|p| p.s);
        if out[k].s == "e"
            && out[k].stress == 2
            && n1 == Some("n")
            && out.get(k + 2).is_some_and(|p| p.is_cons())
        {
            out[k].s = "ɛ";
        }
    }
    out
}

fn is_nasal(s: &str) -> bool {
    matches!(s, "m" | "n" | "ŋ" | "ɲ")
}

fn is_voiced_stop(s: &str) -> bool {
    matches!(s, "b" | "d" | "ɡ")
}

fn is_vowelish(s: &str) -> bool {
    matches!(
        s,
        "a" | "e" | "i" | "o" | "u" | "ɛ" | "I" | "A" | "W" | "O" | "j" | "w"
    )
}

/// Phrase-level allophony and accent rewrites.
fn postprocess(words: &mut [Word], acc: &Accent) {
    // Flatten positions to see across word boundaries.
    let mut idx: Vec<(usize, usize)> = Vec::new();
    for (w, word) in words.iter().enumerate() {
        for p in 0..word.phones.len() {
            idx.push((w, p));
        }
    }
    let get = |words: &[Word], k: usize| -> &'static str {
        let (w, p) = idx[k];
        words[w].phones[p].s
    };
    let boundary_pause = |words: &[Word], k: usize| -> bool {
        // True if position k starts a word that follows a pause.
        let (w, p) = idx[k];
        p == 0 && words[w].after_pause
    };
    // Nasal place assimilation.
    for k in 0..idx.len().saturating_sub(1) {
        if get(words, k) == "n" && !boundary_pause(words, k + 1) {
            let next = get(words, k + 1);
            let new = match next {
                "p" | "b" | "f" => Some("m"),
                "ɡ" | "x" => Some("ŋ"),
                _ => None,
            };
            if let Some(s) = new {
                let (w, p) = idx[k];
                words[w].phones[p].s = s;
            }
        }
    }
    // Spirantisation of b, d, g except after a pause or a nasal.
    for k in 0..idx.len() {
        let s = get(words, k);
        if !is_voiced_stop(s) {
            continue;
        }
        // Coda stops stay occlusive (signi[ɡ]nifica, clu[b], verda[d]).
        let onset = k + 1 < idx.len() && {
            let (w0, _) = idx[k];
            let (w1, _) = idx[k + 1];
            let nxt = get(words, k + 1);
            w0 == w1 && (is_vowelish(nxt) || matches!(nxt, "ɾ" | "l" | "ɛ" | "ʝ"))
        };
        // ...except /b/ before another consonant (a[β]soluto, o[β]jetivo).
        let next_cons_b = s == "b"
            && k + 1 < idx.len()
            && idx[k].0 == idx[k + 1].0
            && !is_vowelish(get(words, k + 1))
            && get(words, k + 1) != "b";
        let keep = k == 0
            || boundary_pause(words, k)
            || is_nasal(get(words, k - 1))
            || (!onset && !next_cons_b);
        if !keep {
            let (w, p) = idx[k];
            words[w].phones[p].s = match s {
                "b" => "β",
                "d" => "ð",
                _ => "ɣ",
            };
        }
    }
    // Regional features.
    let n_words = words.len();
    for wi in 0..n_words {
        let next_first: Option<&'static str> = if wi + 1 < n_words && !words[wi + 1].after_pause {
            words[wi + 1].phones.first().map(|p| p.s)
        } else {
            None
        };
        let next_pause =
            wi + 1 >= n_words || words[wi + 1].after_pause || !words[wi].punct.is_empty();
        let ph = &mut words[wi].phones;
        let len = ph.len();
        for k in 0..len {
            let s = ph[k].s;
            let next = if k + 1 < len {
                Some(ph[k + 1].clone())
            } else {
                None
            };
            let next_is_cons = match &next {
                Some(p) => p.kind == Kind::Consonant,
                None => next_first.is_some_and(|f| !is_vowelish(f)),
            };
            let word_final = k + 1 == len;
            let new: Option<&'static str> = match s {
                "θ" if !acc.distincion => Some("s"),
                "ʎ" => match acc.ll {
                    Ll::Lleismo => None,
                    Ll::Yeismo => Some("ʝ"),
                    Ll::Sheismo => Some("ʃ"),
                },
                "ʝ" if acc.ll == Ll::Sheismo => Some("ʃ"),
                "x" => match acc.jota {
                    Jota::Velar => None,
                    Jota::Glottal => Some("h"),
                    Jota::Fronted
                        if next
                            .as_ref()
                            .is_some_and(|p| matches!(p.s, "i" | "e" | "ɛ" | "j")) =>
                    {
                        Some("ç")
                    }
                    Jota::Fronted => None,
                },
                "ʧ" if acc.ch_fricative => Some("ʃ"),
                "ɾ" if acc.coda_r_to_l && (next_is_cons || word_final) && len > 1 => Some("l"),
                "n" if acc.final_n_velar && word_final => Some("ŋ"),
                _ => None,
            };
            if let Some(n) = new {
                ph[k].s = n;
            }
        }
        // Syllable-final /s/ (θ under seseo was already rewritten above).
        if acc.coda_s != CodaS::Keep {
            for k in 0..len {
                if ph[k].s != "s" {
                    continue;
                }
                let word_final = k + 1 == len;
                let before_cons = if word_final {
                    next_first.is_some_and(|f| !is_vowelish(f))
                } else {
                    ph[k + 1].kind == Kind::Consonant
                };
                let aspirate =
                    before_cons || (word_final && next_pause && acc.coda_s == CodaS::AspirateFinal);
                if aspirate {
                    ph[k].s = "h";
                }
            }
        }
        if acc.d_elision && len >= 4 {
            let l = len;
            if ph[l - 1].s == "o"
                && ph[l - 2].s == "ð"
                && ph[l - 3].s == "a"
                && ph[l - 3].stress == 2
            {
                ph.remove(l - 2);
            }
        }
        if acc.final_d_drop && ph.len() > 2 && matches!(ph.last().map(|p| p.s), Some("ð" | "d")) {
            ph.pop();
        }
    }
}

/// Converts Spanish text to model phonemes for the given accent.
pub fn phonemize(text: &str, acc: &Accent) -> String {
    let toks = normalize(text, acc);
    let mut words: Vec<Word> = Vec::new();
    let mut quotes = Quotes::default();
    let mut after_pause = true;
    for t in toks {
        match t {
            Tok::Word(w) => {
                let phones = word_phones(&w);
                if phones.is_empty() {
                    continue;
                }
                words.push(Word {
                    phones,
                    punct: String::new(),
                    after_pause,
                });
                after_pause = false;
            }
            Tok::Punct(c) => {
                if let Some(p) = quotes.map(c) {
                    if let Some(last) = words.last_mut() {
                        last.punct.push_str(p);
                    }
                }
                if is_pause(c) {
                    after_pause = true;
                }
            }
            _ => {}
        }
    }
    postprocess(&mut words, acc);
    render(&words)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn es(t: &str) -> String {
        phonemize(t, accent("es-ES").unwrap())
    }

    #[test]
    fn matches_espeak_conventions() {
        assert_eq!(es("Hola, ¿cómo estás?"), "ˈola, kˈomo estˈas?");
        assert_eq!(es("ciudad de Zaragoza"), "θjuðˈad ðe θˌaɾaɣˈoθa");
        assert_eq!(
            es("los chicos juegan al fútbol"),
            "los ʧˈikos xwˈeɣan al fˈutβol"
        );
        assert_eq!(es("un beso"), "ˈum bˈeso");
        assert_eq!(es("tengo"), "tˈɛŋɡo");
        assert_eq!(es("aire"), "ˈIɾe");
        assert_eq!(es("rey"), "rˈA");
        assert_eq!(es("muy"), "mˈuj");
        assert_eq!(es("guerra"), "ɡˈera");
        assert_eq!(es("electricidad"), "ˌelektɾˌiθiðˈad");
        assert_eq!(es("rápidamente"), "rˈapiðamˈente");
    }

    #[test]
    fn accents() {
        let mx = accent("es-MX").unwrap();
        assert_eq!(phonemize("cielo y calle", mx), "sjˈelo i kˈaʝe");
        let ar = accent("es-AR").unwrap();
        assert_eq!(phonemize("yo me llamo", ar), "ʃˈo me ʃˈamo");
        assert_eq!(phonemize("mismo", ar), "mˈihmo");
        let cu = accent("es-CU").unwrap();
        assert_eq!(phonemize("cansado", cu), "kansˈao");
    }

    #[test]
    fn numbers() {
        let es_acc = accent("es-ES").unwrap();
        assert_eq!(cardinal(1999, false), "mil novecientos noventa y nueve");
        assert_eq!(cardinal(21, true), "veintiún");
        assert_eq!(cardinal(2_026, false), "dos mil veintiséis");
        assert_eq!(cardinal(1_000_000, false), "un millón");
        assert_eq!(
            number_words("15,50", es_acc, false),
            "quince coma cincuenta"
        );
        assert_eq!(
            number_words("1.234", es_acc, false),
            "mil doscientos treinta y cuatro"
        );
        assert_eq!(number_words("10:30", es_acc, false), "diez y media");
        let toks = normalize("Cuesta 20€ y el 15%", es_acc);
        let words: Vec<String> = toks
            .iter()
            .filter_map(|t| {
                if let Tok::Word(w) = t {
                    Some(w.clone())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            words.join(" "),
            "cuesta veinte euros y el quince por ciento"
        );
    }
}
