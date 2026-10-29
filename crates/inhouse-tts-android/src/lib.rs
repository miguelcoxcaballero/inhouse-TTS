//! Native library of Inhouse TTS for Android and other hosts.
//!
//! * JNI: `com.inhouse.tts.InhouseTts` (see `docs/ANDROID.md`), with full
//!   synthesis and chunk-by-chunk streaming through a Java callback.
//! * C ABI: `ihtts_*` functions for iOS, desktop apps or other languages.

#![allow(clippy::missing_safety_doc)]

use inhouse_tts::core::model::k82::LoadOptions;
use inhouse_tts::{Engine, SpeakOptions};
use jni_sys::{
    jboolean, jbyteArray, jclass, jfloat, jfloatArray, jlong, jobject, jsize, jstring, JNIEnv,
};
use std::ffi::{c_char, CStr, CString};
use std::sync::Mutex;

/// Handle shared with Java/C (an engine behind a mutex).
pub struct Handle {
    engine: Mutex<Engine>,
}

fn create(model: &[u8], voices: &[u8]) -> Result<Box<Handle>, String> {
    let mut engine = Engine::new(model, LoadOptions::default()).map_err(|e| e.0)?;
    engine.add_voices(voices).map_err(|e| e.0)?;
    Ok(Box::new(Handle {
        engine: Mutex::new(engine),
    }))
}

fn options(locale: &str, voice: &str, speed: f32) -> SpeakOptions {
    let loc = inhouse_tts::text::find_locale(locale);
    SpeakOptions {
        locale: loc.map_or(locale.to_string(), |l| l.code.to_string()),
        voice: if voice.is_empty() {
            loc.and_then(|l| l.voices.first())
                .unwrap_or(&"af_heart")
                .to_string()
        } else {
            voice.to_string()
        },
        speed: if speed > 0.0 { speed } else { 1.0 },
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// JNI helpers
// ---------------------------------------------------------------------------

unsafe fn bytes(env: *mut JNIEnv, arr: jbyteArray) -> Vec<u8> {
    let f = &**env;
    let len = (f.GetArrayLength.unwrap())(env, arr) as usize;
    let mut out = vec![0u8; len];
    (f.GetByteArrayRegion.unwrap())(env, arr, 0, len as jsize, out.as_mut_ptr() as *mut i8);
    out
}

unsafe fn string(env: *mut JNIEnv, s: jstring) -> String {
    if s.is_null() {
        return String::new();
    }
    let f = &**env;
    let p = (f.GetStringUTFChars.unwrap())(env, s, std::ptr::null_mut());
    let out = CStr::from_ptr(p).to_string_lossy().into_owned();
    (f.ReleaseStringUTFChars.unwrap())(env, s, p);
    out
}

unsafe fn floats(env: *mut JNIEnv, v: &[f32]) -> jfloatArray {
    let f = &**env;
    let arr = (f.NewFloatArray.unwrap())(env, v.len() as jsize);
    if !arr.is_null() {
        (f.SetFloatArrayRegion.unwrap())(env, arr, 0, v.len() as jsize, v.as_ptr());
    }
    arr
}

unsafe fn throw(env: *mut JNIEnv, msg: &str) {
    let f = &**env;
    let cls = (f.FindClass.unwrap())(env, c"java/lang/RuntimeException".as_ptr());
    if !cls.is_null() {
        let m = CString::new(msg.replace('\0', " ")).unwrap_or_default();
        (f.ThrowNew.unwrap())(env, cls, m.as_ptr());
    }
}

unsafe fn handle<'a>(h: jlong) -> Option<&'a Handle> {
    (h as *const Handle).as_ref()
}

/// `static native long nativeCreate(byte[] model, byte[] voices)`
#[no_mangle]
pub unsafe extern "system" fn Java_com_inhouse_tts_InhouseTts_nativeCreate(
    env: *mut JNIEnv,
    _: jclass,
    model: jbyteArray,
    voices: jbyteArray,
) -> jlong {
    match create(&bytes(env, model), &bytes(env, voices)) {
        Ok(h) => Box::into_raw(h) as jlong,
        Err(e) => {
            throw(env, &e);
            0
        }
    }
}

/// `static native void nativeDestroy(long handle)`
#[no_mangle]
pub unsafe extern "system" fn Java_com_inhouse_tts_InhouseTts_nativeDestroy(
    _: *mut JNIEnv,
    _: jclass,
    h: jlong,
) {
    if h != 0 {
        drop(Box::from_raw(h as *mut Handle));
    }
}

/// `static native float[] nativeSpeak(long handle, String text, String locale, String voice, float speed)`
#[no_mangle]
pub unsafe extern "system" fn Java_com_inhouse_tts_InhouseTts_nativeSpeak(
    env: *mut JNIEnv,
    _: jclass,
    h: jlong,
    text: jstring,
    locale: jstring,
    voice: jstring,
    speed: jfloat,
) -> jfloatArray {
    let Some(h) = handle(h) else {
        throw(env, "engine is closed");
        return std::ptr::null_mut();
    };
    let opts = options(&string(env, locale), &string(env, voice), speed);
    let text = string(env, text);
    let result = h
        .engine
        .lock()
        .map_err(|_| "engine poisoned".to_string())
        .and_then(|mut e| e.speak(&text, &opts).map_err(|e| e.0));
    match result {
        Ok(audio) => floats(env, &audio),
        Err(e) => {
            throw(env, &e);
            std::ptr::null_mut()
        }
    }
}

/// `static native void nativeStream(long handle, String text, String locale, String voice, float speed, InhouseTts.ChunkListener listener)`
///
/// Calls `listener.onChunk(float[] samples)` for every chunk as soon as it is
/// ready; returning `false` from `onChunk` stops the stream.
#[no_mangle]
pub unsafe extern "system" fn Java_com_inhouse_tts_InhouseTts_nativeStream(
    env: *mut JNIEnv,
    _: jclass,
    h: jlong,
    text: jstring,
    locale: jstring,
    voice: jstring,
    speed: jfloat,
    listener: jobject,
) {
    let Some(h) = handle(h) else {
        throw(env, "engine is closed");
        return;
    };
    let f = &**env;
    let cls = (f.GetObjectClass.unwrap())(env, listener);
    let method = (f.GetMethodID.unwrap())(env, cls, c"onChunk".as_ptr(), c"([F)Z".as_ptr());
    if method.is_null() {
        return; // NoSuchMethodError is pending in Java.
    }
    let opts = options(&string(env, locale), &string(env, voice), speed);
    let text = string(env, text);
    let Ok(mut engine) = h.engine.lock() else {
        throw(env, "engine poisoned");
        return;
    };
    let stream = match engine.stream(&text, &opts) {
        Ok(s) => s,
        Err(e) => {
            throw(env, &e.0);
            return;
        }
    };
    for chunk in stream {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                throw(env, &e.0);
                return;
            }
        };
        let arr = floats(env, &chunk.audio);
        let keep: jboolean = (f.CallBooleanMethod.unwrap())(env, listener, method, arr);
        (f.DeleteLocalRef.unwrap())(env, arr);
        if (f.ExceptionCheck.unwrap())(env) != 0 || keep == 0 {
            return;
        }
    }
}

/// `static native String nativePhonemize(long handle, String text, String locale)`
#[no_mangle]
pub unsafe extern "system" fn Java_com_inhouse_tts_InhouseTts_nativePhonemize(
    env: *mut JNIEnv,
    _: jclass,
    h: jlong,
    text: jstring,
    locale: jstring,
) -> jstring {
    let Some(h) = handle(h) else {
        throw(env, "engine is closed");
        return std::ptr::null_mut();
    };
    let text = string(env, text);
    let locale = string(env, locale);
    let r = h
        .engine
        .lock()
        .map_err(|_| "engine poisoned".to_string())
        .and_then(|mut e| e.phonemize(&text, &locale).map_err(|e| e.0));
    match r {
        Ok(ps) => {
            let c = CString::new(ps).unwrap_or_default();
            ((**env).NewStringUTF.unwrap())(env, c.as_ptr())
        }
        Err(e) => {
            throw(env, &e);
            std::ptr::null_mut()
        }
    }
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

/// Creates an engine from a `.ihw` model and voice pack. Returns NULL on error.
#[no_mangle]
pub unsafe extern "C" fn ihtts_engine_new(
    model: *const u8,
    model_len: usize,
    voices: *const u8,
    voices_len: usize,
) -> *mut Handle {
    if model.is_null() || voices.is_null() {
        return std::ptr::null_mut();
    }
    let m = std::slice::from_raw_parts(model, model_len);
    let v = std::slice::from_raw_parts(voices, voices_len);
    create(m, v).map_or(std::ptr::null_mut(), Box::into_raw)
}

#[no_mangle]
pub unsafe extern "C" fn ihtts_engine_free(h: *mut Handle) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Synthesises UTF-8 `text` (24 kHz mono f32). Returns NULL on error; free
/// the result with `ihtts_audio_free(ptr, len)`.
#[no_mangle]
pub unsafe extern "C" fn ihtts_speak(
    h: *mut Handle,
    text: *const c_char,
    locale: *const c_char,
    voice: *const c_char,
    speed: f32,
    out_len: *mut usize,
) -> *mut f32 {
    let (Some(h), false, false) = (h.as_ref(), text.is_null(), out_len.is_null()) else {
        return std::ptr::null_mut();
    };
    let s = |p: *const c_char| {
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    let opts = options(&s(locale), &s(voice), speed);
    let Ok(mut e) = h.engine.lock() else {
        return std::ptr::null_mut();
    };
    match e.speak(&s(text), &opts) {
        Ok(audio) => {
            let mut b = audio.into_boxed_slice();
            *out_len = b.len();
            let p = b.as_mut_ptr();
            std::mem::forget(b);
            p
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn ihtts_audio_free(p: *mut f32, len: usize) {
    if !p.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(p, len)));
    }
}

/// Sample rate of the produced audio.
#[no_mangle]
pub extern "C" fn ihtts_sample_rate() -> u32 {
    inhouse_tts::SAMPLE_RATE
}
