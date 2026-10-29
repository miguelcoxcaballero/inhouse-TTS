//! Language-neutral tokenisation of raw text into words, numbers and
//! punctuation, used by every language front end.

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    /// Letters (any script), possibly with inner apostrophes or hyphens.
    Word(String),
    /// Digits with inner separators (`.`, `,`, `:`, `/`), e.g. `1.234,5`.
    Num(String),
    /// Sentence or clause punctuation (`.`, `,`, `?`, ...).
    Punct(char),
    /// Other symbols (`%`, `€`, `$`, `&`, `@`, `+`, `º`, ...).
    Sym(char),
}

pub fn is_letter(c: char) -> bool {
    c.is_alphabetic() || c == '\u{0301}' || c == '\u{0303}' || c == '\u{0308}'
}

const PUNCT: &str = ".,;:!?¡¿…—–-\"“”«»()[]{}";

/// Splits text into tokens. Whitespace is dropped; a token's position in the
/// output is enough for the front ends.
pub fn scan(text: &str) -> Vec<Tok> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() {
            let start = i;
            i += 1;
            while i < chars.len() {
                let d = chars[i];
                if d.is_ascii_digit() {
                    i += 1;
                } else if matches!(d, '.' | ',' | ':' | '/' | '\'')
                    && i + 1 < chars.len()
                    && chars[i + 1].is_ascii_digit()
                {
                    i += 1;
                } else {
                    break;
                }
            }
            out.push(Tok::Num(chars[start..i].iter().collect()));
        } else if is_letter(c) {
            let start = i;
            i += 1;
            while i < chars.len() {
                let d = chars[i];
                if is_letter(d) {
                    i += 1;
                } else if matches!(d, '\'' | '’' | '-')
                    && i + 1 < chars.len()
                    && is_letter(chars[i + 1])
                {
                    i += 1;
                } else {
                    break;
                }
            }
            out.push(Tok::Word(
                chars[start..i].iter().collect::<String>().replace('’', "'"),
            ));
        } else if PUNCT.contains(c) {
            // Collapse "..." into an ellipsis.
            if c == '.' && i + 2 < chars.len() && chars[i + 1] == '.' && chars[i + 2] == '.' {
                out.push(Tok::Punct('…'));
                i += 3;
                while i < chars.len() && chars[i] == '.' {
                    i += 1;
                }
            } else {
                out.push(Tok::Punct(c));
                i += 1;
            }
        } else {
            out.push(Tok::Sym(c));
            i += 1;
        }
    }
    out
}

/// Maps punctuation to the symbols the acoustic model knows (others vanish).
pub fn model_punct(c: char) -> Option<&'static str> {
    Some(match c {
        '.' => ".",
        ',' => ",",
        ';' => ";",
        ':' => ":",
        '!' => "!",
        '?' => "?",
        '…' => "…",
        '—' | '–' => "—",
        '"' | '“' | '«' => "“",
        '”' | '»' => "”",
        '(' | '[' | '{' => "(",
        ')' | ']' | '}' => ")",
        _ => return None,
    })
}

/// Maps punctuation like [`model_punct`], alternating straight double quotes
/// between opening and closing curly quotes.
#[derive(Default)]
pub struct Quotes {
    open: bool,
}

impl Quotes {
    pub fn map(&mut self, c: char) -> Option<&'static str> {
        if c == '"' {
            self.open = !self.open;
            return Some(if self.open { "“" } else { "”" });
        }
        model_punct(c)
    }
}

/// True for punctuation that implies a pause before the next word.
pub fn is_pause(c: char) -> bool {
    matches!(
        c,
        '.' | ','
            | ';'
            | ':'
            | '!'
            | '?'
            | '…'
            | '—'
            | '–'
            | '('
            | ')'
            | '"'
            | '“'
            | '”'
            | '«'
            | '»'
            | '¡'
            | '¿'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scans_mixed_text() {
        let t = scan("Hola, ¿qué tal? Son 1.234,50€ (el 20%)...");
        assert_eq!(t[0], Tok::Word("Hola".into()));
        assert_eq!(t[1], Tok::Punct(','));
        assert!(t.contains(&Tok::Num("1.234,50".into())));
        assert!(t.contains(&Tok::Sym('€')));
        assert!(t.contains(&Tok::Sym('%')));
        assert_eq!(*t.last().unwrap(), Tok::Punct('…'));
    }
}
