# Inhouse TTS

**Voz neuronal realista, multilingüe y con acento local seleccionable. 100 % Rust, en tu dispositivo, en directo.**
*Realistic, multilingual neural text-to-speech with selectable regional accents — written entirely in Rust, running locally and in real time.*

**▶ Demo web (se ejecuta en tu navegador, sin servidor):** https://miguelcoxcaballero.github.io/inhouse-TTS/

| | |
|---|---|
| 🦀 **Todo en Rust** | motor neuronal propio (GEMM SIMD, convoluciones, LSTM, atención, vocoder iSTFT), front-end de texto, interfaz web (WASM), biblioteca Android (JNI), CLI y herramientas de build. Sin C/C++, sin ONNX Runtime, sin espeak (GPL) |
| 🌍 **13 acentos** | español de España, México, Argentina, Colombia, Chile, Perú, Venezuela, Caribe, Puerto Rico, Andalucía y Canarias, e inglés de EE. UU. y Reino Unido |
| ⚡ **Local y en streaming** | el audio empieza con la primera frase; RTF ≈ 0,45 en un PC de 4 núcleos; funciona sin conexión |
| 🎮 **GPU** | WebGPU en el navegador, y Vulkan, Metal o DX12 en nativo y Android (`wgpu`). Salida idéntica a la CPU (101–104 dB) |
| 📱 **Cualquier dispositivo** | x86-64 (AVX2/FMA, SSE2), ARM64/Android (NEON), navegador (WebAssembly SIMD128 o WebGPU) |
| 🔓 **Open source** | código Apache-2.0 / MIT; voces puente con pesos Apache-2.0 |

## Ejemplos

Generados con `inhouse-tts say`, sin retoques. La misma voz (`ef_dora`) cambia de acento con las reglas fonológicas de cada región:

| Locale | Rasgos audibles | Audio |
|---|---|---|
| es-ES | distinción z/c ≠ s | [es-ES_ef_dora.wav](samples/es-ES_ef_dora.wav) |
| es-MX | seseo | [es-MX_ef_dora.wav](samples/es-MX_ef_dora.wav) |
| es-AR | sheísmo («ʃo me ʃamo»), /s/ aspirada | [es-AR_ef_dora.wav](samples/es-AR_ef_dora.wav) |
| es-CO | jota suave [h] | [es-CO_ef_dora.wav](samples/es-CO_ef_dora.wav) |
| es-CL | /s/ final aspirada, «cansao» | [es-CL_ef_dora.wav](samples/es-CL_ef_dora.wav) |
| es-PE | seseo | [es-PE_ef_dora.wav](samples/es-PE_ef_dora.wav) |
| es-VE | /n/ final velar, /s/ aspirada | [es-VE_ef_dora.wav](samples/es-VE_ef_dora.wav) |
| es-CU | Caribe: /s/ y /d/ finales, «cansao» | [es-CU_ef_dora.wav](samples/es-CU_ef_dora.wav) |
| es-PR | además /r/ → [l] en coda («comel») | [es-PR_ef_dora.wav](samples/es-PR_ef_dora.wav) |
| es-AN | andaluz: «andalú», /s/ aspirada | [es-AN_ef_dora.wav](samples/es-AN_ef_dora.wav) |
| es-IC | canario | [es-IC_ef_dora.wav](samples/es-IC_ef_dora.wav) |
| en-US | American English (`af_heart`) | [en-US_af_heart.wav](samples/en-US_af_heart.wav) |
| en-GB | British English (`bf_emma`) | [en-GB_bf_emma.wav](samples/en-GB_bf_emma.wav) |

Inteligibilidad medida con Whisper-small sobre estos ejemplos: **WER medio del 10,8 %**. Casi todos los «errores» son la marca («Inhao TTS») o el propio acento: Whisper escribe «Asento» con seseo, «Andalú» y «Saludo».

## Uso rápido (CLI)

```bash
cargo build --release -p inhouse-tts-cli
```

Conversión de los pesos puente a nuestro formato `.ihw`, que hace falta una sola vez. Antes descarga [Kokoro-82M v1.0](https://huggingface.co/hexgrad/Kokoro-82M) (`config.json`, `kokoro-v1_0.pth` y `voices/*.pt`) a `ruta/Kokoro-82M`:

```bash
cargo xtask models --kokoro ruta/Kokoro-82M --out models
```

```bash
./target/release/inhouse-tts say -l es-AR -o hola.wav "¡Hola! Yo me llamo Inhouse TTS."
```

```bash
./target/release/inhouse-tts say -l en-GB -v "0.6*bf_emma+0.4*af_heart" -o blend.wav "Voices can be blended."
```

```bash
./target/release/inhouse-tts phonemize -l es-CU "Estoy cansado, vamos a la playa."
```

`inhouse-tts locales` lista los acentos, `voices` las voces y `bench` mide la velocidad. Con `--gpu`, el decodificador va en la GPU (Vulkan, Metal o DX12) y, si no hay una GPU real, se queda en la CPU.

```bash
./target/release/inhouse-tts bench --gpu -l es-ES
```

## Como biblioteca (Rust)

```rust
use inhouse_tts::{Engine, SpeakOptions};

let mut tts = Engine::new(&std::fs::read("models/k82.ihw")?, Default::default())?;
tts.add_voices(&std::fs::read("models/voices.ihw")?)?;
let opts = SpeakOptions { locale: "es-MX".into(), voice: "ef_dora".into(), ..Default::default() };
for chunk in tts.stream("Hola, ¿qué tal? Esto es Inhouse TTS.", &opts)? {
    let chunk = chunk?;          // PCM f32, 24 kHz, en cuanto está listo
    play(&chunk.audio);
}
```

## Android, web y otros

- **Android**: biblioteca JNI con streaming por *callback* para `AudioTrack` y C ABI. Ver [docs/ANDROID.md](docs/ANDROID.md). CI publica los `.so` de arm64-v8a, armeabi-v7a y x86_64.
- **Web**: `cargo xtask web --kokoro ruta/Kokoro-82M` genera el sitio en `web/dist` y `cargo xtask serve` lo sirve en local. La interfaz y el *worker* de síntesis son Rust compilado a WASM; el JavaScript son dos líneas de arranque. El selector **Acelerador** usa WebGPU cuando hay una GPU real y, si no, WebAssembly SIMD en varios *workers*; también se puede forzar con `?accel=gpu` o `?accel=cpu`.

## Arquitectura

```
texto → normalización → G2P + reglas de acento → troceado → modelo acústico → vocoder → PCM
        (números,       (español: reglas,       (streaming)  (K82 hoy,        (iSTFT hoy,
         fechas, $)      inglés: léxicos)                     IH-1 mañana)     IH-Codec mañana)
```

- **Motor propio** (`inhouse-tts-core`):
  - GEMM empaquetada con micro-kernels AVX2+FMA 6×16, SSE2 4×8, NEON 8×8 y WASM SIMD128 6×8.
  - Convoluciones con im2col virtual y convoluciones transpuestas polifásicas.
  - LSTM, ALBERT, AdaIN y Snake; STFT/iSTFT con semántica de PyTorch.
  - Pesos f32/f16/int8 en el formato propio `.ihw` y conversor seguro de checkpoints PyTorch (sin ejecutar pickle).
  - **Backend de GPU** con `wgpu`: convolución como GEMM implícita sobre los pesos int8, AdaIN con Snake fusionado y convolución transpuesta polifásica. Kernels WGSL embebidos en Rust.
- **Modelo puente K82**: StyleTTS2 + iSTFTNet, compatible con los pesos Apache-2.0 de Kokoro-82M. **Reproduce la referencia con 106–131 dB de SNR** en cada etapa.
- **Modelo propio IH-1**, en planificación: latentes continuos a 12,5 Hz, fonemas con *tokens* de acento y voz disociada del acento, entrenado en Rust (burn). Unas 40× menos cómputo que K82, para tiempo real en cualquier móvil y en WASM.

Documentación:

- [docs/RESEARCH.md](docs/RESEARCH.md) — investigación del estado del arte (modelos, licencias, ecosistema Rust, datos, eficiencia) y medidas propias.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — diseño del sistema, arquitectura del modelo IH-1, plan de entrenamiento y hoja de ruta.
- [docs/ANDROID.md](docs/ANDROID.md) — integración en Android.

## Rendimiento (Xeon de 4 vCPU a 2,8 GHz)

| Plataforma | RTF (< 1 = más rápido que el tiempo real) |
|---|---|
| Nativo, AVX2+FMA, 4 hilos | **0,42–0,51** |
| Nativo, 1 hilo | 1,3 |
| Navegador, WASM SIMD128 | ~4 por *worker* (el sitio usa varios *workers*) |
| Navegador o nativo con GPU (WebGPU, Vulkan, Metal) | estimado 0,05–0,15 en GPUs integradas; **sin medir en hardware real** (este entorno no tiene GPU) |

El 88 % del cómputo es el vocoder iSTFTNet, ~25 GMAC por segundo de audio, y por eso va en la GPU cuando la hay. El siguiente salto será el modelo IH-1. Detalles en [docs/RESEARCH.md §7](docs/RESEARCH.md#7-medidas-propias-de-inhouse-tts).

## Hoja de ruta

- [x] Motor neuronal propio en Rust (x86-64, ARM64, WASM), validado frente a la referencia
- [x] Front-end propio sin GPL: español (11 acentos) e inglés (US/GB)
- [x] CLI, API con streaming, web en GitHub Pages, biblioteca Android
- [x] GPU con `wgpu` (WebGPU, Vulkan, Metal, DX12) para el decodificador y el vocoder
- [ ] AVX-512; hilos en WASM; medir y ajustar los kernels en GPUs reales
- [ ] Portugués (BR/PT), francés (FR/CA), italiano e hindi en el front-end
- [ ] IH-Codec e IH-1 entrenados en Rust con datos permisivos y destilados a 8–30M parámetros

## Licencias y créditos

- Código: **Apache-2.0 o MIT**, a tu elección.
- Voces puente: pesos de [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) de hexgrad (Apache-2.0), convertidos al formato `.ihw` durante el build. No se incluyen en este repositorio.
- Léxicos ingleses: derivados de [misaki](https://github.com/hexgrad/misaki) (Apache-2.0). Ver [NOTICE](NOTICE).
