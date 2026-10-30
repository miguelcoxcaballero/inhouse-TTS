//! Supported languages and regional accents.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lang {
    Es,
    En,
}

/// A language + regional accent with its recommended voices.
#[derive(Clone, Copy, Debug)]
pub struct Locale {
    /// BCP-47 style code, e.g. `es-MX`.
    pub code: &'static str,
    pub lang: Lang,
    /// Native display name.
    pub name: &'static str,
    /// English display name.
    pub name_en: &'static str,
    /// Recommended voices, best first.
    pub voices: &'static [&'static str],
    /// Demo sentence that shows off the accent.
    pub sample: &'static str,
}

impl Locale {
    pub fn british(&self) -> bool {
        self.code.eq_ignore_ascii_case("en-GB")
    }
}

/// Castilian voices (Spain, Andalusia and the Canary Islands use them with
/// their own accent rules).
const ES_SPAIN: &[&str] = &["es_ES-sharvard-F", "es_ES-davefx", "es_ES-sharvard-M"];
/// Latin American voice (seseo); the accent rules of each country apply.
const ES_LATAM: &[&str] = &["es_MX-ald"];
const ES_AR: &[&str] = &["es_AR-daniela", "es_MX-ald"];

pub const LOCALES: &[Locale] = &[
    Locale {
        code: "es-ES",
        lang: Lang::Es,
        name: "Español — España",
        name_en: "Spanish — Spain",
        voices: ES_SPAIN,
        sample: "Hola, soy Inhouse TTS, un motor de voz escrito en Rust. En España distinguimos la zeta de Zaragoza y la ce de cielo.",
    },
    Locale {
        code: "es-MX",
        lang: Lang::Es,
        name: "Español — México",
        name_en: "Spanish — Mexico",
        voices: ES_LATAM,
        sample: "¿Qué tal? Soy Inhouse TTS y hablo con acento de México. Mañana vamos a la ciudad a comer unos tacos con mis amigos.",
    },
    Locale {
        code: "es-AR",
        lang: Lang::Es,
        name: "Español — Argentina (rioplatense)",
        name_en: "Spanish — Argentina",
        voices: ES_AR,
        sample: "¡Hola! Soy Inhouse TTS. Yo me llamo así y hablo como en Buenos Aires: la calle, la lluvia y las estrellas.",
    },
    Locale {
        code: "es-CO",
        lang: Lang::Es,
        name: "Español — Colombia",
        name_en: "Spanish — Colombia",
        voices: ES_LATAM,
        sample: "Hola, ¿cómo está? Soy Inhouse TTS y hablo con acento de Colombia. El gerente trabaja junto al jardín.",
    },
    Locale {
        code: "es-CL",
        lang: Lang::Es,
        name: "Español — Chile",
        name_en: "Spanish — Chile",
        voices: ES_LATAM,
        sample: "Hola, soy Inhouse TTS y hablo con acento chileno. Los jóvenes están cansados después de la fiesta.",
    },
    Locale {
        code: "es-PE",
        lang: Lang::Es,
        name: "Español — Perú",
        name_en: "Spanish — Peru",
        voices: ES_LATAM,
        sample: "Buenos días, soy Inhouse TTS y hablo con acento peruano. La ciudad de Lima está junto al mar.",
    },
    Locale {
        code: "es-VE",
        lang: Lang::Es,
        name: "Español — Venezuela",
        name_en: "Spanish — Venezuela",
        voices: ES_LATAM,
        sample: "Hola, ¿cómo están? Soy Inhouse TTS con acento venezolano. Los muchachos se fueron para la playa.",
    },
    Locale {
        code: "es-CU",
        lang: Lang::Es,
        name: "Español — Caribe (Cuba, R. Dominicana)",
        name_en: "Spanish — Caribbean",
        voices: ES_LATAM,
        sample: "Oye, soy Inhouse TTS y hablo con acento caribeño. Estoy cansado, pero vamos a bailar esta noche.",
    },
    Locale {
        code: "es-PR",
        lang: Lang::Es,
        name: "Español — Puerto Rico",
        name_en: "Spanish — Puerto Rico",
        voices: ES_LATAM,
        sample: "Saludos, soy Inhouse TTS y hablo como en Puerto Rico. Abre la puerta, que vamos a comer arroz.",
    },
    Locale {
        code: "es-AN",
        lang: Lang::Es,
        name: "Español — Andalucía",
        name_en: "Spanish — Andalusia",
        voices: ES_SPAIN,
        sample: "Hola, soy Inhouse TTS y hablo con acento andaluz. Los amigos se han quedado en Sevilla toda la tarde.",
    },
    Locale {
        code: "es-IC",
        lang: Lang::Es,
        name: "Español — Canarias",
        name_en: "Spanish — Canary Islands",
        voices: ES_SPAIN,
        sample: "Buenas, soy Inhouse TTS y hablo con acento canario. Las guaguas salen temprano de Las Palmas.",
    },
    Locale {
        code: "en-US",
        lang: Lang::En,
        name: "English — United States",
        name_en: "English — United States",
        voices: &["af_heart", "af_bella", "af_nicole", "am_michael", "am_fenrir", "am_puck", "af_aoede", "af_kore", "af_sarah", "am_adam"],
        sample: "Hi! I'm Inhouse TTS, a realistic text-to-speech engine written entirely in Rust. It runs locally, on your phone or your computer.",
    },
    Locale {
        code: "en-GB",
        lang: Lang::En,
        name: "English — United Kingdom",
        name_en: "English — United Kingdom",
        voices: &["bf_emma", "bm_george", "bm_fable", "bf_isabella", "bm_lewis", "bf_alice", "bf_lily", "bm_daniel"],
        sample: "Hello! I'm Inhouse TTS, speaking with a British accent. Shall we have a cup of tea and a biscuit in the garden?",
    },
];

/// Finds a locale by code (case-insensitive; `es` → `es-ES`, `en` → `en-US`).
pub fn find(code: &str) -> Option<&'static Locale> {
    let c = code.trim().replace('_', "-");
    if let Some(l) = LOCALES.iter().find(|l| l.code.eq_ignore_ascii_case(&c)) {
        return Some(l);
    }
    let lower = c.to_ascii_lowercase();
    if lower == "en" || lower.starts_with("en-") {
        return LOCALES
            .iter()
            .find(|l| l.code == if lower == "en-uk" { "en-GB" } else { "en-US" });
    }
    if lower.starts_with("es") {
        let acc = crate::es::accent(&lower)?;
        return LOCALES.iter().find(|l| l.code == acc.code);
    }
    None
}
