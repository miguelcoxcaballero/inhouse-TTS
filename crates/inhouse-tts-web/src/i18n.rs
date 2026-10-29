//! Interface strings (Spanish / English, chosen from the browser language).

pub struct Strings {
    pub subtitle: &'static str,
    pub text: &'static str,
    pub locale: &'static str,
    pub voice: &'static str,
    pub speed: &'static str,
    pub threads: &'static str,
    pub speak: &'static str,
    pub stop: &'static str,
    pub download: &'static str,
    pub loading: &'static str,
    pub preparing: &'static str,
    pub ready: &'static str,
    pub lexicon: &'static str,
    pub speaking: &'static str,
    pub done: &'static str,
    pub stopped: &'static str,
    pub first_audio: &'static str,
    pub realtime: &'static str,
    pub native_voices: &'static str,
    pub other_voices: &'static str,
    pub phonemes: &'static str,
    pub error: &'static str,
    pub empty: &'static str,
    pub offline: &'static str,
    pub about: &'static str,
}

pub const ES: Strings = Strings {
    subtitle: "Voz neuronal realista, multilingüe y con acentos regionales. 100 % Rust, en tu dispositivo, sin servidores.",
    text: "Texto",
    locale: "Idioma y acento",
    voice: "Voz",
    speed: "Velocidad",
    threads: "Hilos de síntesis",
    speak: "▶ Hablar",
    stop: "■ Parar",
    download: "⬇ Descargar WAV",
    loading: "Descargando el modelo de voz (solo la primera vez)",
    preparing: "Preparando el motor…",
    ready: "Listo. Todo se ejecuta en tu navegador.",
    lexicon: "Cargando el diccionario de inglés…",
    speaking: "Sintetizando fragmento",
    done: "Terminado.",
    stopped: "Detenido.",
    first_audio: "primer audio en",
    realtime: "× tiempo real",
    native_voices: "Voces nativas",
    other_voices: "Otras voces (acento cruzado)",
    phonemes: "Fonemas generados (IPA)",
    error: "Error",
    empty: "Escribe algún texto.",
    offline: "Funciona sin conexión tras la primera carga.",
    about: "Motor y frontend: Inhouse TTS (Apache-2.0/MIT). Voces puente: pesos Kokoro-82M (Apache-2.0) convertidos al formato propio .ihw.",
};

pub const EN: Strings = Strings {
    subtitle: "Realistic multilingual neural voice with regional accents. 100% Rust, on your device, no servers.",
    text: "Text",
    locale: "Language & accent",
    voice: "Voice",
    speed: "Speed",
    threads: "Synthesis threads",
    speak: "▶ Speak",
    stop: "■ Stop",
    download: "⬇ Download WAV",
    loading: "Downloading the voice model (first time only)",
    preparing: "Preparing the engine…",
    ready: "Ready. Everything runs in your browser.",
    lexicon: "Loading the English dictionary…",
    speaking: "Synthesizing chunk",
    done: "Done.",
    stopped: "Stopped.",
    first_audio: "first audio after",
    realtime: "× real time",
    native_voices: "Native voices",
    other_voices: "Other voices (cross-lingual)",
    phonemes: "Generated phonemes (IPA)",
    error: "Error",
    empty: "Type some text.",
    offline: "Works offline after the first load.",
    about: "Engine and front end: Inhouse TTS (Apache-2.0/MIT). Bootstrap voices: Kokoro-82M weights (Apache-2.0) converted to the native .ihw format.",
};

pub fn pick(lang: &str) -> &'static Strings {
    if lang.to_ascii_lowercase().starts_with("es") {
        &ES
    } else {
        &EN
    }
}
