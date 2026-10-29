# Inhouse TTS en Android

La biblioteca nativa `libinhouse_tts_jni.so` (crate `inhouse-tts-android`) es Rust puro y usa la GEMM NEON 8×8 en ARM64. Expone la clase JNI `com.inhouse.tts.InhouseTts` y una C ABI (`ihtts_*`).

> Validación: la salida en ARM64 (NEON, ejecutada bajo qemu) coincide con la de x86-64 a **91,9 dB de SNR**, el límite del PCM de 16 bits.

## 1. Compilar

Con el NDK instalado (`ANDROID_NDK_HOME`) y [`cargo-ndk`](https://github.com/bbqsrc/cargo-ndk):

```bash
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
```

```bash
cargo install cargo-ndk
```

```bash
cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 -o app/src/main/jniLibs build --release -p inhouse-tts-android
```

El workflow **Android** de GitHub Actions publica estos `.so` como artefacto en cada push. También publica el binario de línea de comandos `inhouse-tts` para ARM64, que se puede usar en Termux.

## 2. Modelos

Copia `k82.ihw` y `voices.ihw` a los *assets* de la app o descárgalos la primera vez; los genera `cargo xtask models`. En el móvil conviene el modelo `--dtype f16` (169 MB) o `q8` (116 MB).

```bash
cargo xtask models --kokoro ruta/a/Kokoro-82M --out models --dtype f16
```

## 3. Clase Kotlin

La única pieza que no es Rust es esta clase de pegamento de unas 30 líneas, porque Android carga el código nativo a través de JNI:

```kotlin
package com.inhouse.tts

class InhouseTts(model: ByteArray, voices: ByteArray) : AutoCloseable {
    fun interface ChunkListener { fun onChunk(samples: FloatArray): Boolean }

    private var handle = nativeCreate(model, voices)

    /** Sintetiza todo el texto (PCM float mono, 24 kHz). */
    fun speak(text: String, locale: String = "es-ES", voice: String = "", speed: Float = 1f): FloatArray =
        nativeSpeak(handle, text, locale, voice, speed)

    /** Streaming: llama a [listener] con cada fragmento en cuanto está listo. */
    fun stream(text: String, locale: String, voice: String = "", speed: Float = 1f, listener: ChunkListener) =
        nativeStream(handle, text, locale, voice, speed, listener)

    fun phonemize(text: String, locale: String): String = nativePhonemize(handle, text, locale)

    override fun close() { if (handle != 0L) { nativeDestroy(handle); handle = 0L } }

    companion object {
        const val SAMPLE_RATE = 24_000
        init { System.loadLibrary("inhouse_tts_jni") }
        @JvmStatic private external fun nativeCreate(model: ByteArray, voices: ByteArray): Long
        @JvmStatic private external fun nativeDestroy(handle: Long)
        @JvmStatic private external fun nativeSpeak(h: Long, text: String, locale: String, voice: String, speed: Float): FloatArray
        @JvmStatic private external fun nativeStream(h: Long, text: String, locale: String, voice: String, speed: Float, l: ChunkListener)
        @JvmStatic private external fun nativePhonemize(h: Long, text: String, locale: String): String
    }
}
```

## 4. Reproducir en directo con `AudioTrack`

```kotlin
val track = AudioTrack.Builder()
    .setAudioFormat(AudioFormat.Builder()
        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
        .setSampleRate(InhouseTts.SAMPLE_RATE)
        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
    .setTransferMode(AudioTrack.MODE_STREAM)
    .setBufferSizeInBytes(InhouseTts.SAMPLE_RATE * 4)
    .build()
track.play()
thread {
    tts.stream("Hola, ¿qué tal? Esto es Inhouse TTS.", "es-MX") { chunk ->
        track.write(chunk, 0, chunk.size, AudioTrack.WRITE_BLOCKING)
        true // devuelve false para cortar
    }
}
```

## 5. C ABI (iOS, escritorio, otros lenguajes)

```c
void*  ihtts_engine_new(const uint8_t* model, size_t model_len, const uint8_t* voices, size_t voices_len);
float* ihtts_speak(void* engine, const char* text, const char* locale, const char* voice, float speed, size_t* out_len);
void   ihtts_audio_free(float* samples, size_t len);
void   ihtts_engine_free(void* engine);
uint32_t ihtts_sample_rate(void);
```

## 6. Rendimiento esperado

El modelo puente K82 cuesta ~26 GMAC por segundo de audio. En ARM64 con 4 núcleos grandes (Snapdragon 8 Gen 2 o similares) se espera un RTF de 0,3–0,6. En móviles de 2019, alrededor de 1, como mide la investigación con Kokoro. El modelo propio **IH-1 Mobile** apunta a un RTF < 0,1 en cualquier Android; ver [ARCHITECTURE.md](ARCHITECTURE.md).
