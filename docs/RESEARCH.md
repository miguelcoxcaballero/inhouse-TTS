# Investigación: estado del arte en TTS abierto (octubre de 2026)

> Fecha de la investigación: **29-10-2026**. Fuentes primarias: fichas de modelos en Hugging Face, READMEs y licencias de los repositorios, API de crates.io, OpenSLR, arXiv y blogs técnicos. Lo que no se pudo confirmar en una fuente primaria aparece como **[sin verificar]**. Las cifras de velocidad vienen de máquinas distintas, así que solo sirven para orientarse.
>
> Al final hay una sección con **medidas propias** del motor de Inhouse TTS.

## Índice

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Modelos TTS abiertos](#2-modelos-tts-abiertos)
3. [Ecosistema Rust](#3-ecosistema-rust)
4. [Front-end de texto, G2P y acentos](#4-front-end-de-texto-g2p-y-acentos)
5. [Datos de entrenamiento con licencias permisivas](#5-datos-de-entrenamiento-con-licencias-permisivas)
6. [Eficiencia en el dispositivo](#6-eficiencia-en-el-dispositivo)
7. [Medidas propias de Inhouse TTS](#7-medidas-propias-de-inhouse-tts)
8. [Conclusiones y decisiones](#8-conclusiones-y-decisiones)

---

## 1. Resumen ejecutivo

1. **El avance más relevante de 2026 para este proyecto es la línea CALM / Pocket TTS de Kyutai.** Predice latentes continuos de audio a **12,5 Hz** (tramas de 80 ms) con un transformer causal y un cabezal de muestreo de un solo paso.
   - Pocket TTS (100M) va unas 6× más rápido que el tiempo real en 2 núcleos de portátil y clona voces.
   - El código es MIT y los pesos CC-BY-4.0. El **código de entrenamiento** se publicó en agosto de 2026.
   - El 20-10-2026 salieron modelos ingleses de **18,2M / 6,9M / 1,9M** parámetros: 9,9× / 12,6× / 25× el tiempo real **en un solo hilo de CPU**, con WER < 1,8 %.
   - Fuentes: [README](https://github.com/kyutai-labs/pocket-tts), [modelos pequeños](https://kyutai.org/blog/2026-10-20-pocket-tts-small), [entrenamiento](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md).
2. **Ya existe un runtime de TTS 100 % Rust: Phonon / `ptts` de Gradium** (v0.5.0, 27-10-2026), sobre la librería de tensores `xn` (CPU con wasm-simd128, wgpu, Metal, CUDA).
   - Funciona en CLI, navegador, iOS, Android y C.
   - Phonon 31m (inglés) es Apache-2.0. El 40m multilingüe es **no comercial**.
   - [xn-ptts](https://github.com/gradium-ai/xn-ptts), [xn](https://github.com/LaurentMazare/xn).
3. **Kokoro-82M (Apache-2.0)** sigue siendo el modelo pequeño permisivo más conocido. Está en el **8.º puesto de los modelos abiertos en Artificial Analysis** (Elo 1062) y es el único de menos de 100M en ese grupo.
   - No hace streaming.
   - Sus idiomas no ingleses dependen de fonemas de espeak-ng (GPL).
   - En móviles de 2019 va más lento que el tiempo real (RTF 1,13 en un Galaxy S10).
   - [Ficha](https://huggingface.co/hexgrad/Kokoro-82M), [ranking AA](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice/open-weights).
4. **La cima de calidad está ahora en modelos de 0,5B a 35B**, casi todos con licencias restrictivas:
   - Fish Drama 3, Breeze TTS 2, Fish S2 Pro y Voxtral.
   - Excepciones permisivas: Ming-3.0-tts-0.5B (MIT), Step-Audio-EditX (Apache-2.0) y Magpie 357M (licencia NVIDIA).
   - Ninguno corre en tiempo real en un móvil ni en el navegador.
5. **Trampas de licencia**:
   - Emilia (101,7k h) es **CC BY-NC**, y por eso F5-TTS, ZipVoice y OmniVoice son no comerciales.
   - Piper pasó a **GPL-3.0**.
   - **espeak-ng es GPL-3.0+**, también su port a Rust puro.
   - Supertonic usa OpenRAIL-M y su repositorio está archivado.
6. **Tendencia en el front-end de texto**: los modelos pequeños modernos ya no usan G2P (Pocket/Phonon usan SentencePiece y Supertonic, caracteres), y el acento sale del prompt de voz. Para **controlar el acento de forma explícita** los fonemas siguen siendo útiles:
   - un *fine-tune* de Pocket TTS acepta IPA ([willwade](https://huggingface.co/willwade/pocket-tts-en-phonemes-24l));
   - un estudio de 2026 muestra que reglas fonológicas más *embeddings* de hablante dan acentos más auténticos ([arXiv 2601.14417](https://arxiv.org/abs/2601.14417)).
7. **Datos de acentos permisivos**:
   - **Buenos** para el inglés (VCTK, GLOBE_V2 con 164 acentos, SLR83) y el español de Latinoamérica (sets de Google en OpenSLR: AR, CL, CO, PE, PR, VE).
   - **Escasos** para pt-PT y fr-CA.
8. **Lecciones de eficiencia**:
   - La **frecuencia de tramas** es la palanca principal (12,5 Hz frente a 40–86 Hz).
   - Los cabezales de un paso y los decodificadores destilados funcionan; el de Kyutai pasa de 10,3M a 1,0M parámetros.
   - **int8 no es automáticamente más rápido en ARM.**
   - Los hilos WASM necesitan las cabeceras COOP/COEP, que GitHub Pages no permite poner.

---

## 2. Modelos TTS abiertos

### 2.1 Clase «en el dispositivo» (≤ ~150M parámetros)

| Modelo | Parámetros | Arquitectura | Idiomas / acentos | Licencia código / pesos | Velocidad conocida | Streaming |
|---|---|---|---|---|---|---|
| **Pocket TTS** (Kyutai, 2026) | 100M; 18,2M / 6,9M / 1,9M (inglés) | CALM: transformer causal sobre latentes continuos estilo Mimi (12,5 Hz) + cabezal de 1 paso; SentencePiece, **sin G2P** | en, fr, de, pt, it, es, nl (+ comunidad); el acento lo da la voz de referencia | MIT / CC-BY-4.0 | ~6× TR en 2 núcleos M4; pequeños: 9,9×/12,6×/25× TR en 1 hilo | ✔ por trama |
| **Phonon 31m / 40m** (Gradium) | 31M / 40M | Derivado de Pocket TTS, decodificador Mimi | 31m: en; 40m: en, fr, de, es (España), pt (Brasil) | MIT/Apache / 31m **Apache-2.0**, 40m CC BY-NC | M5, 1 núcleo: 26× TR; RPi 5: 2,3× TR | ✔ |
| **Kokoro-82M v1.0** | 82M | StyleTTS2 + decodificador iSTFTNet; entrada de fonemas | en-US, en-GB, es, fr, hi, it, pt-BR, ja, zh (54 voces) | Apache-2.0 / Apache-2.0 | Galaxy S10 RTF 1,13; WebGPU (M2) RTF 0,2–0,4 | ✖ (por frases) |
| Supertonic 2 / 3 | 66M / ~99M | Autoencoder + *flow matching*, caracteres | 5 / 31 idiomas | MIT / OpenRAIL-M | M4 Pro RTF 0,012 | ✖; repo **archivado** |
| Kitten TTS 0.8 | 15M / 40M / 80M | StyleTTS2, espeak-ng | inglés | Apache-2.0 | Galaxy S10 RTF 0,39 | ✖ |
| Piper (`piper1-gpl`) | 5–30M | VITS + espeak-ng | 35+ idiomas, muchas voces regionales | **GPL-3.0** / según voz (muchas derivan de Lessac, solo investigación) | Galaxy S10 RTF 0,08 | ◐ |
| MeloTTS | ~50M **[sin verificar]** | VITS2/Bert-VITS2 | en-US/GB/IN/AU, es, fr, zh, ja, ko | MIT / MIT | «tiempo real en CPU» | ✖ |
| Matcha-TTS | 18M (+ vocoder) | *Flow matching* (2–10 pasos) | por dataset | MIT | Galaxy S10 RTF 0,135 | ✖ |
| Chatterbox-Nano | 110M | LM de tokens + decodificador de 1 paso | inglés | MIT / MIT | 3× TR en 8 núcleos | ◐ |
| Soprano-1.1-80M | 80M | LM pequeño + decodificador Vocos | inglés | Apache-2.0 | ~TR en 1 núcleo | ✔ |

### 2.2 Clase grande (≥ 0,3B), orientada a GPU

| Modelo | Tamaño | Licencia de pesos | Notas |
|---|---|---|---|
| Ming-3.0-tts-0.5B (oct. 2026) | 0,5B | **MIT** | #3 abierto en AA; 7 idiomas + cantonés y sichuanés |
| Qwen3-TTS 0.6B / 1.7B | 0,6–1,7B | Apache-2.0 | 10 idiomas; un port a Rust puro va a ~0,5× TR en el navegador con una descarga de 1,77 GB |
| CosyVoice 3 (0.5B) | 0,5B | Apache-2.0 | 9 idiomas + 18 dialectos chinos |
| Chatterbox Multilingual V3 | 0,5B | MIT | **packs es-ES vs es-MX y pt-PT vs pt-BR** |
| Kyutai TTS 1.6B | 1,8B | CC-BY-4.0 | servidor en Rust (candle), en/fr |
| VibeVoice-Realtime-0.5B | 0,5B | MIT | ~300 ms al primer audio |
| Magpie-TTS-Multilingual | 357M | NVIDIA OML | 12 idiomas, incluye variantes de árabe |
| F5-TTS, OmniVoice, ZipVoice | 0,3–0,6B | **CC-BY-NC** (Emilia) | no comerciales |
| Fish S2 Pro, Voxtral, Breeze 2, XTTS-v2, Spark-TTS | 0,5–5B | **no comerciales** | — |

### 2.3 Rankings (29-10-2026)

- **Artificial Analysis**, modelos abiertos (Elo): 1 Fish Drama 3 (1240) · 2 Breeze TTS 2 (1228) · 3 Ming-3.0 (1187) · 4 Fish S2 Pro (1109) · 5 Step-Audio-EditX (1099) · 6 Voxtral (1090) · 7 Magpie (1067) · **8 Kokoro-82M (1062)** · … · 12 Chatterbox (1024) · 17 StyleTTS 2 (891). Pocket TTS, Supertonic y Phonon no aparecen en este ranking.
- **Open TTS Leaderboard de Hugging Face** (30-09-2026, métricas automáticas): en WER de inglés lideran Kokoro-82M, Supertonic-3 y Fish S2 Pro, y Pocket TTS destaca en streaming en CPU ([blog](https://huggingface.co/blog/open-tts-leaderboard)). Las cifras exactas están **[sin verificar]**.
- **Evaluación de Kyutai** (WER, Elo de calidad): Pocket TTS (100M) obtiene 1,84 / 2016, al nivel de F5-TTS (336M) y Kyutai TTS (750M). Solo Pocket TTS y Kokoro fueron más rápidos que el tiempo real en la CPU de un portátil.

### 2.4 Trampas de licencia

| Trampa | Afecta a |
|---|---|
| Emilia (101,7k h) es CC BY-NC | F5-TTS, ZipVoice, OmniVoice, Kani-TTS-2 |
| espeak-ng es GPL-3.0+, también en Rust | Kokoro fuera del inglés, Kitten, Piper, Zonos, NeuTTS |
| La licencia de los datos y voces se hereda | voces Piper (Lessac), voces de referencia NC |
| «Pesos abiertos» no es lo mismo que código abierto | Fish, Higgs, IndexTTS2, Kitten 2, NeuTTS Nano |
| Restricciones de uso | Supertonic (OpenRAIL-M) |

**Consecuencia para Inhouse TTS:** el motor no enlaza espeak-ng; el front-end de texto es nuestro (Apache/MIT). El modelo puente usa los pesos Apache-2.0 de Kokoro, y el modelo propio se entrenará solo con datos CC0/CC-BY (y CC-BY-SA en módulos separados).

---

## 3. Ecosistema Rust

### 3.1 Motores de inferencia

| Motor | ¿Rust puro? | Backends | WASM | WebGPU | Comentario |
|---|---|---|---|---|---|
| **candle** 0.11 | Sí | CPU, CUDA, Metal | Sí (CPU) | No oficial | muy usado (moshi, any-tts) |
| **burn** 0.22 + CubeCL | Sí | CUDA, ROCm, Metal, Vulkan, **WebGPU**, CPU | Sí | Sí | **entrenamiento e inferencia en Rust** |
| **tract** 0.23 | Sí | CPU con SIMD propio | Sí | No | ejecutor ONNX/NNEF |
| **rten** 0.27 | Sí | CPU (AVX2/512, NEON, WASM SIMD) | Sí | No | ONNX |
| **xn** 0.2 | Sí | CPU (gemm con wasm-simd128), CUDA, Metal, Vulkan, wgpu | Sí | Sí | base de Phonon |
| ort 2.0-rc | **No** (C++) | todos los de ONNX Runtime | ORT Web | ORT Web | descartado por «TODO EN RUST» |

### 3.2 TTS en Rust que ya existen

- **Phonon / `ptts`** (xn): el único runtime TTS completo 100 % Rust de extremo a extremo.
- **any-tts**, **voice** (candle): Kokoro y otros, con fonetizadores parciales.
- **Kokoros**, **kokoro-tts**, **piper-rs**, **sherpa-onnx**: dependen de **ONNX Runtime (C++)** y algunos de espeak-ng.

### 3.3 G2P en Rust

| Crate | Licencia | Comentario |
|---|---|---|
| `misaki-rs` | MIT | port del G2P inglés de Kokoro; trae espeak como opción por defecto (desactivable) |
| `floravox-g2p` | Apache/MIT | léxicos FST de ~13–28 idiomas (gruut), uroman |
| `piper-plus-g2p` | MIT | 8 idiomas sin espeak |
| `espeak-ng` (port) | **GPL-3.0+** | evitar |
| `jpreprocess`, `jieba-rs`, `pinyin` | BSD/MIT | japonés y chino |

### 3.4 TTS en el navegador

- **kokoro-js** (ONNX Runtime Web): RTF 0,2–0,4 con WebGPU en un M2 y ≈1,1 con WASM q8 en un x86 de gama media.
- **Piper** en WASM: RTF ≈0,11.
- **Phonon** (Rust a WASM): compilaciones con hilos y *relaxed SIMD* y WebGPU opcional. Observa que «en móviles la CPU con hilos puede ganar a WebGPU».
- **Restricciones de GitHub Pages**:
  - Los hilos WASM requieren COOP/COEP, que hay que inyectar con `coi-serviceworker`.
  - El sitio puede ocupar como máximo 1 GB y cada fichero, 100 MiB.

---

## 4. Front-end de texto, G2P y acentos

### 4.1 Cómo se modelan hoy los acentos

1. **Acento = variante de G2P + voz**: misaki en-US/en-GB, espeak `es` frente a `es-419` y voces Piper por *locale*.
2. **Acento = prompt de voz** (clonación): Pocket TTS, Chatterbox. El acento depende de la muestra.
3. **Acento = *fine-tune* por región**: Chatterbox *Single Language Packs* (es-ES/es-MX, pt-PT/pt-BR).
4. **Acento = atributo aprendido**: *embeddings* de acento separados del hablante ([2406.10844](https://arxiv.org/abs/2406.10844)), intensidad interpolable ([2606.25403](https://arxiv.org/abs/2606.25403)).
5. **Reglas + *embeddings***: las reglas fonológicas explícitas combinadas con *embeddings* de hablante dan los acentos más auténticos ([2601.14417](https://arxiv.org/abs/2601.14417)).

### 4.2 Qué diferencia a cada variedad

| Variedades | Diferencias sistemáticas |
|---|---|
| Español peninsular / americano / rioplatense / caribeño | distinción θ/s frente a seseo (o ceceo); yeísmo frente a lleísmo; sheísmo/zheísmo [ʃ ʒ]; aspiración o pérdida de /s/ en coda; jota [x] frente a [h]; /n/ final velar; neutralización r/l (PR) |
| Inglés EE. UU. / Reino Unido | rotacidad, *flapping* de /t/, TRAP–BATH, LOT/THOUGHT, GOAT oʊ/əʊ, diferencias léxicas |
| Portugués BR / PT | reducción vocálica fuerte en PT; palatalización de t/d ante i en BR; /s/ en coda como [ʃ] en PT; róticas; vocalización de /l/ en BR |
| Francés FR / CA | africación de t/d ante i/y; relajación de vocales altas; diptongación; contrastes a/ɑ y ɛ/ɛː |

**La mayoría son transformaciones regulares sobre una fonetización base.** Por eso el diseño «G2P base + reglas por acento + excepciones por léxico» es viable. La entonación y la calidad vocálica fina, en cambio, tienen que salir de datos de audio etiquetados por acento.

### 4.3 Léxicos

- **CMUdict** (BSD) y **Britfone** (MIT) para el inglés.
- **Diccionarios gold/silver de misaki** (Apache-2.0; procedencia no documentada).
- **WikiPron** (datos CC BY-SA, con dialectos: es Castilla frente a Latinoamérica y pt BR frente a PT).
- **ipa-dict** (es_ES/es_MX, fr_FR/fr_QC).

---

## 5. Datos de entrenamiento con licencias permisivas

| Dataset | Tamaño | Idiomas / acentos | Licencia |
|---|---|---|---|
| LJSpeech | 24 h, 1 voz | en-US | dominio público |
| VCTK | 44 h, 110 hablantes | inglés con acentos (Inglaterra, Escocia, Irlanda, Gales, EE. UU., India…) | CC BY 4.0 |
| LibriTTS-R | 585 h | inglés | CC BY 4.0 |
| **HiFiTTS-2** | 36,7k h | inglés | CC BY 4.0 |
| MLS / CML-TTS | ~1k h es, fr, pt, it, de… | 8 idiomas | CC BY 4.0 |
| **Common Voice** | miles de horas | 294 idiomas con metadatos de acento | CC0 |
| **GLOBE_V2** | 535 h, 23,5k hablantes | inglés, **164 acentos** | CC0 |
| Emilia-YODAS | 113,9k h | en, fr, de, ko, ja, zh | CC BY 4.0 |
| VoxPopuli | 1,8k h + 400k h | 16–23 idiomas europeos | CC0 |
| **Google crowdsourced (OpenSLR)** | 1k–18k frases por set | **es-AR, es-CL, es-CO, es-PE, es-PR, es-VE**, ca, eu, gl, en-UK/IE, en-NG | CC BY-SA 4.0 |
| OHF voice-datasets | 1 voz por set | es_ES, pt_BR ×3, pt_PT, it_IT… | CC0 |
| SIWIS | <11 h | fr-FR | CC BY 4.0 |
| ~~Emilia~~, ~~Expresso~~, ~~EARS~~ | — | — | **NC: excluir** |

**Cobertura por acento:** en-US es abundante; en-GB y otros acentos ingleses, buenos. El español de Latinoamérica es bueno, salvo México, donde hay que extraer de Common Voice y MLS. pt-BR y es-ES son moderados; **pt-PT y fr-CA son un hueco**. El presupuesto de Kyutai sirve de referencia: 2.000 h de HiFiTTS-2 → ~8,8 h en una H100, menos de 200 $.

---

## 6. Eficiencia en el dispositivo

### 6.1 Coste de los decodificadores de forma de onda (GMAC por segundo de audio)

| Decodificador | Parámetros | GMAC/s | Comentario |
|---|---|---|---|
| HiFi-GAN V1 | 13,8M | 28,0 | referencia en el dominio temporal |
| iSTFTNet | 13,7M | 25,9 | **el que usa Kokoro/K82** |
| MS-iSTFTNet | 12,3M | 8,4 | multi-stream |
| Vocos | 13,5M | 1,35 | ConvNeXt a frecuencia de trama + iSTFT |
| MS-Wavehax | 0,33M | 1,6 | el más rápido con latencia < 80 ms |
| Decodificador latente Pocket (destilado) | 1,0M | — | 29× TR en un hilo |

### 6.2 Cuantización

- **int8 por canal suele ser seguro, pero no siempre más rápido**: Kokoro int8 fue ~2× más lento que fp32 en un Galaxy S10.
- Las **últimas capas del decodificador** son las más sensibles; conviene mantenerlas en mayor precisión.
- Con 4 bits Kokoro pierde poco (−0,07 UTMOS), pero Supertonic pierde mucho (−2,8).

### 6.3 Streaming

- **AR sobre latentes continuos** (CALM): audio cada 80 ms, con un TTFB de 5–20 ms en Phonon.
- **NAR por bloques** (Kokoro): hay que trocear en frases de 100–200 tokens; Kokoro flojea con menos de 10–20 tokens y se acelera con más de 400.
- **Vocoders causales** con un frame de *lookahead* rinden igual que los no causales.

### 6.4 Velocidad medida en dispositivos

| Sistema | Dispositivo | RTF (<1 = más rápido que el tiempo real) |
|---|---|---|
| Piper low | Galaxy S10 (2019) | 0,08 |
| Matcha + HiFi-GAN | Galaxy S10 | 0,135 |
| Kitten nano | Galaxy S10 | 0,39 |
| **Kokoro fp32** | **Galaxy S10** | **1,13** |
| Phonon 31m | RPi 5, 1 núcleo | ~0,43 (2,3× TR) |
| Pocket TTS 6,9M | 1 hilo CPU | ~0,08 (12,6× TR) |

---

## 7. Medidas propias de Inhouse TTS

Medidas sobre el motor de este repositorio (`inhouse-tts-core`), un Xeon de 4 vCPU a 2,8 GHz con AVX2/FMA y Chromium headless:

### 7.1 Fidelidad numérica del motor K82 frente a la implementación de referencia (PyTorch)

Comparación capa a capa, sin el ruido de excitación:

| Tensor | SNR |
|---|---|
| contexto ALBERT → `d_en` | 106 dB |
| codificador de duración `d` | 109 dB |
| F0 / energía predichas | 120 / 124 dB |
| codificador de texto | 131 dB |
| decodificador (antes del generador) | 122 dB |
| generador iSTFTNet con la misma excitación | 107 dB |
| duraciones | **idénticas** |

Con el ruido activado, la diferencia espectral entre nuestro audio y la referencia (L1 log-espectral 0,1307) es igual a la que hay entre dos ejecuciones de la propia referencia con semillas distintas (0,1310). Son **estadísticamente indistinguibles**. Nuestra fuente armónica acumula la fase en f64, mientras que la referencia lo hace en f32 y pierde precisión en frases largas.

### 7.2 Cuantización del modelo puente

| Pesos | Tamaño | SNR de la forma de onda vs f32 | L1 log-espectral |
|---|---|---|---|
| f32 | 325 MB | — | — |
| f16 | 169 MB | 27,6 dB | 0,039 |
| q8 en todo | 92 MB | (las duraciones cambian) | 0,92 ✖ |
| **q8 solo en el decodificador y f16 en el resto** | **116 MB** | 25,9 dB | 0,082 ✔ |

Con q8, ALBERT se degrada (29,7 dB) porque comparte la misma capa en 12 iteraciones. Por eso solo el decodificador o vocoder, que es el 65 % de los parámetros, va en int8.

### 7.3 Velocidad

| Plataforma | Configuración | RTF |
|---|---|---|
| Nativo x86-64 | AVX2+FMA, 4 hilos, f32 preempaquetado | **0,46–0,51** |
| Nativo x86-64 | AVX2+FMA, 1 hilo | 1,26–1,30 |
| Nativo x86-64 | solo SSE2, 1 hilo, q8 (emula WASM) | 3,6 |
| Navegador (WASM SIMD128) | 1 worker | ~4 |
| Navegador (WASM SIMD128) | GEMM aislado | 8–10 GMAC/s |

**Backend de GPU** (`wgpu`, mismo modelo q8, misma entrada):

| Prueba | Resultado |
|---|---|
| Vulkan (lavapipe) frente a la CPU, frase de 3,1 s | **101,4 dB** de SNR (q8) y **103,7 dB** (f32); error máximo 4,5·10⁻⁶ |
| WebGPU en Chromium (Tint + SwiftShader) frente a WASM en CPU, WAV de 16 bits | **85,7 dB**: como mucho 1 LSB de diferencia |
| Velocidad en adaptadores por software (lavapipe, SwiftShader) | RTF 7–40, porque emulan la GPU en la CPU. El modo automático los descarta |

En este entorno de desarrollo no hay ninguna GPU física. Según el cómputo (~25 GMAC/s de audio) y la eficiencia típica de una GEMM de 64×64 en WGSL (20–40 % del pico), se espera:

- RTF ≈ 0,05–0,15 en GPUs integradas recientes (Intel Xe, Apple M, Adreno 7xx);
- RTF ≪ 0,05 en GPUs dedicadas.

Está pendiente medirlo en hardware real.

**Dónde se va el tiempo**: el 88 % del cómputo está en el generador iSTFTNet, ~25 GMAC por segundo de audio, igual que la tabla 6.1. La GEMM del motor ya va cerca del límite de cada ISA. Para el tiempo real en el navegador y en móviles modestos solo queda:

- usar la GPU (WebGPU/wgpu), **ya implementado** (ver arriba), o
- un modelo con decodificador a frecuencia de trama: unas 20–40× menos cómputo. Es el plan del modelo propio, en [ARCHITECTURE.md](ARCHITECTURE.md).

### 7.4 Front-end

| Prueba | Resultado |
|---|---|
| Español frente a espeak-ng (solo como oráculo local, no se distribuye), 5.000 palabras más frecuentes | **95,5 %** idénticas, **97,8 %** sin contar el acento prosódico. Las diferencias restantes son sobre todo préstamos y abreviaturas, que expandimos mejor (`Sr.` → «señor»). |
| Inglés frente a la lógica de léxico de misaki, 4.776 palabras | **99,4 %** idénticas |

---

## 8. Conclusiones y decisiones

1. **Motor propio en Rust puro, con CPU SIMD y GPU (WebGPU, Vulkan, Metal o DX12 vía wgpu).** Las opciones de Rust «puro» (Phonon/xn, burn, candle) confirman que es viable. Hemos escrito nuestro propio motor para controlar el formato de pesos, los kernels de SIMD de cada plataforma y el tamaño del binario WASM (0,9 MB).
2. **Modelo puente K82**: es compatible con los pesos Apache-2.0 de Kokoro-82M. Ofrece calidad top-10 abierta desde el primer día en inglés y español, y en francés, italiano, hindi y portugués de Brasil cuando estén sus front-ends. También servirá de **profesor** para destilar el modelo propio.
3. **Front-end propio sin GPL**: reglas y léxicos con **13 acentos** (11 de español y 2 de inglés) y control explícito del acento por fonemas, la vía que la literatura de 2026 señala como la más auténtica cuando se combina con *embeddings*.
4. **Modelo propio IH-1**: latentes continuos a 12,5 Hz, en la línea de CALM/Pocket TTS, con fonemas y *tokens* de acento, entrenado en Rust (burn) solo con datos permisivos. Objetivo: 20–40× menos cómputo que K82, tiempo real en cualquier móvil y en el navegador. Detalle en [ARCHITECTURE.md](ARCHITECTURE.md).
5. **Datos**: HiFiTTS-2, LibriTTS-R, VCTK, GLOBE_V2, Common Voice, MLS/CML-TTS, VoxPopuli y los sets de Google para los acentos del español. Hay que conseguir grabaciones CC0 para pt-PT y fr-CA.
