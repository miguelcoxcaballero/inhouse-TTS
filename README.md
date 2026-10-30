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
| 🔓 **Open source** | código Apache-2.0 / MIT; voces de [Piper](https://huggingface.co/rhasspy/piper-voices) (MIT) en español y de Kokoro-82M (Apache-2.0) en inglés, todas ejecutadas en nuestro motor |

## Ejemplos

Generados con `inhouse-tts say`, sin retoques. Cada acento usa una voz nativa de su zona y las reglas fonológicas de la región:

| Locale | Voz | Rasgos audibles | Audio |
|---|---|---|---|
| es-ES | Sharvard (F) | distinción z/c ≠ s | [es-ES_es_ES-sharvard-F.wav](samples/es-ES_es_ES-sharvard-F.wav) |
| es-ES | Davefx (M) | distinción z/c ≠ s | [es-ES_es_ES-davefx.wav](samples/es-ES_es_ES-davefx.wav) |
| es-AN | Sharvard (F) | andaluz: «andalú», /s/ aspirada | [es-AN_es_ES-sharvard-F.wav](samples/es-AN_es_ES-sharvard-F.wav) |
| es-IC | Sharvard (F) | canario | [es-IC_es_ES-sharvard-F.wav](samples/es-IC_es_ES-sharvard-F.wav) |
| es-MX | Ald (M) | seseo | [es-MX_es_MX-ald.wav](samples/es-MX_es_MX-ald.wav) |
| es-AR | Daniela (F) | rioplatense: sheísmo («ʃo me ʃamo»), /s/ aspirada | [es-AR_es_AR-daniela.wav](samples/es-AR_es_AR-daniela.wav) |
| es-CO | Ald (M) | jota suave [h] | [es-CO_es_MX-ald.wav](samples/es-CO_es_MX-ald.wav) |
| es-CL | Ald (M) | /s/ final aspirada, «cansao» | [es-CL_es_MX-ald.wav](samples/es-CL_es_MX-ald.wav) |
| es-PE | Ald (M) | seseo | [es-PE_es_MX-ald.wav](samples/es-PE_es_MX-ald.wav) |
| es-VE | Ald (M) | /n/ final velar, /s/ aspirada | [es-VE_es_MX-ald.wav](samples/es-VE_es_MX-ald.wav) |
| es-CU | Ald (M) | Caribe: /s/ y /d/ finales, «cansao» | [es-CU_es_MX-ald.wav](samples/es-CU_es_MX-ald.wav) |
| es-PR | Ald (M) | además /r/ → [l] en coda («comel») | [es-PR_es_MX-ald.wav](samples/es-PR_es_MX-ald.wav) |
| en-US | Heart (F, `af_heart`) | American English | [en-US_af_heart.wav](samples/en-US_af_heart.wav) |
| en-GB | Emma (F, `bf_emma`) | British English | [en-GB_bf_emma.wav](samples/en-GB_bf_emma.wav) |

Inteligibilidad medida con Whisper-small sobre los ejemplos en español: **WER medio del 16 %**, sin contar el nombre «Inhouse TTS» (que Whisper no conoce). Casi todos los «errores» son el propio acento («asento» con seseo, «andalú»). En frases neutras de castellano, las voces de España dan un **0 %**. La voz argentina (Daniela) es la más débil (10–15 %) y la mexicana (Ald) a veces confunde «hablo» con «abro».

## Uso rápido (CLI)

```bash
cargo build --release -p inhouse-tts-cli
```

Las voces se convierten una sola vez a nuestro formato `.ihw`.

**Español (Piper).** Descarga el `.onnx` y el `.onnx.json` de cada voz de [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices): `es_ES-davefx-medium`, `es_ES-sharvard-medium`, `es_MX-ald-medium` y `es_AR-daniela-high`. Luego conviértelas en `models/piper/`, que la CLI carga automáticamente:

```bash
./target/release/inhouse-tts piper-convert --onnx es_ES-davefx-medium.onnx --json es_ES-davefx-medium.onnx.json --out models/piper/es_ES-davefx-medium.ihw
```

**Inglés (Kokoro).** Descarga [Kokoro-82M v1.0](https://huggingface.co/hexgrad/Kokoro-82M) (`config.json`, `kokoro-v1_0.pth` y `voices/*.pt`) a `ruta/Kokoro-82M`:

```bash
cargo xtask models --kokoro ruta/Kokoro-82M --out models
```

```bash
./target/release/inhouse-tts say -l es-ES -v es_ES-sharvard-F -o hola.wav "¡Hola! La zeta de Zaragoza y la ce de cielo."
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

let mut tts = Engine::without_model();   // solo voces Piper; Engine::new(k82) para el inglés
tts.add_piper("es_ES-davefx", &std::fs::read("models/piper/es_ES-davefx-medium.ihw")?)?;
let opts = SpeakOptions { locale: "es-ES".into(), voice: "es_ES-davefx".into(), ..Default::default() };
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
- **Voces en español: Piper (VITS)**, reimplementado en el motor.
  - Un lector propio de ONNX convierte los modelos a `.ihw`; no se usa onnxruntime ni espeak, porque nuestros fonemas se convierten al IPA con el que se entrenaron.
  - Reproduce onnxruntime con duraciones idénticas y error máximo de 5·10⁻⁵ en voces medium, low, x_low y high.
  - Va 20× más rápido que el tiempo real en 4 núcleos (RTF 0,045) y 2,7× en el navegador.
- **Voces en inglés: modelo K82** (StyleTTS2 + iSTFTNet), compatible con los pesos Apache-2.0 de Kokoro-82M. **Reproduce la referencia con 106–131 dB de SNR** en cada etapa.
- **Modelo propio IHV1**, entrenado desde cero sin ningún modelo de terceros: VITS con decodificador iSTFT multibanda, acento y voz como condiciones separadas. Los datos y el entrenamiento son Rust (`ihtts-data`, `ihtts-train` con *burn*), y la inferencia corre en el mismo motor (paridad de 1 LSB con el entrenador). Está listo para entrenarse en Google Colab con grabaciones castellanas de banda completa y acento verificado, pero de momento las voces publicadas son las de Piper. Ver [Entrenar el modelo propio](#entrenar-el-modelo-propio).
- **Modelo IH-1**, el siguiente paso: latentes continuos a 12,5 Hz, unas 40× menos cómputo que K82, para tiempo real en cualquier móvil y en WASM.

Documentación:

- [docs/RESEARCH.md](docs/RESEARCH.md) — investigación del estado del arte (modelos, licencias, ecosistema Rust, datos, eficiencia) y medidas propias.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — diseño del sistema, arquitectura del modelo IH-1, plan de entrenamiento y hoja de ruta.
- [docs/ANDROID.md](docs/ANDROID.md) — integración en Android.

## Entrenar el modelo propio

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miguelcoxcaballero/inhouse-TTS/blob/main/train/colab.ipynb)

1. Abre el cuaderno [`train/colab.ipynb`](train/colab.ipynb) en Colab y elige *Entorno de ejecución → GPU*.
2. Pulsa *Ejecutar todo*. El cuaderno descarga el entrenador CUDA precompilado y el corpus preparado, y entrena con checkpoints en tu Google Drive.
3. Si la sesión se corta, vuelve a ejecutarlo: continúa desde el último checkpoint. Las muestras de audio se guardan en `MyDrive/inhouse-tts/runs/<voz>/samples/`.

En local, `ihtts-data prepare corpora/es-ES/*.json` prepara los datos e `ihtts-train train --run runs/es-ES --preset small --data …/manifest.jsonl --device cuda|wgpu|cpu --resume` entrena. Los datos, sus licencias y la verificación de acento se describen en [corpora/README.md](corpora/README.md).

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
- [x] Datos y entrenamiento en Rust: corpus permisivos de banda completa con acento verificado, modelo propio IHV1, Colab
- [x] Voces nativas por acento en español (Piper en nuestro motor): castellano con distinción, México, Argentina
- [ ] Voces propias entrenadas (IHV1) para sustituir a las de terceros
- [ ] IH-Codec e IH-1 entrenados en Rust con datos permisivos y destilados a 8–30M parámetros

## Licencias y créditos

- Código: **Apache-2.0 o MIT**, a tu elección.
- **Voces en español**: modelos VITS de [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) (MIT), convertidos al formato `.ihw` y ejecutados por nuestro motor, sin onnxruntime ni espeak. Datos de entrenamiento según cada ficha:
  - *davefx*: CC0;
  - *sharvard*: CC BY 3.0 (Aubanel, García Lecumberri y Cooke, 2014);
  - *ald*: Unlicense;
  - *daniela*: CC BY-SA 4.0 (Google, OpenSLR 61).

  Ojo: según sus fichas, davefx, sharvard y daniela se ajustaron partiendo de la voz inglesa *lessac* de Piper, cuyos datos (Blizzard 2013) tienen una licencia solo para investigación; ald parte de davefx. Antes de un uso comercial conviene revisarlo legalmente.
- **Voces en inglés**: pesos de [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) de hexgrad (Apache-2.0), convertidos al formato `.ihw` durante el build. No se incluyen en este repositorio.
- Léxicos ingleses: derivados de [misaki](https://github.com/hexgrad/misaki) (Apache-2.0). Ver [NOTICE](NOTICE).
