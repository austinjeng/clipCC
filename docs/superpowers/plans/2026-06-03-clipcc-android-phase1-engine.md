# clipCC-Android Phase 1 — Headless Inference Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A headless, instrumented-tested Android inference engine that turns (video + text labels + model) into the same `[F×L]` cosine/confidence matrices and aggregation results as the Python clipCC pipeline, verified against the Phase 0 golden fixtures within tolerance.

**Architecture:** Plain Kotlin classes in the existing `:app` module (no new Gradle modules — keeps native `.so` in the application module, sidestepping AGP-9 library-native uncertainty). Pure-math pieces (resampler core, scoring) are JVM-unit-testable (`src/test`); native/IO pieces (tokenizer JNI, ORT, Media3) are device-instrumented-tested (`src/androidTest`). The engine consumes the Phase 0 `ModelBundleManifest` + ONNX towers staged on device.

**Tech Stack:** Kotlin 2.0, ORT `onnxruntime-android:1.26.0`, HF `tokenizers` 0.23.x (Rust cdylib via cargo-ndk, NDK r28c), Media3 `media3-inspector-frame:1.10.1`, JUnit4 + androidx.test.

**Inputs from Phase 0:** spec `docs/superpowers/specs/2026-06-02-clipcc-android-design.md`; results `docs/superpowers/plans/phase0-spike-results.md`; bundles `build/android_assets/<model_id>/` (manifest + ONNX + tokenizer.json); fixtures `build/android_assets/fixtures/` (`tokenizer_golden.json`, `preprocess_golden.npz`, `scores_golden.json`, `resample_contract.json`).

**Android project:** `/Users/austin/AndroidStudioProjects/ClipCC` (module `:app`, package `com.example.clipcc`, minSdk 24, AGP 9.0.0-rc01). Build with `JAVA_HOME=/Applications/Android Studio.app/Contents/jbr/Contents/Home`. adb at `~/Library/Android/sdk/platform-tools/adb`. **Not a git repo** — commit engine code into the **clipCC repo is N/A**; instead this plan's "commit" steps mean *save the file in the Android project*; track progress via the checkboxes and the parity gates. (If the user later `git init`s the Android project, switch to real commits.)

---

## Resolved facts carried from Phase 0 (do not re-litigate)
- **Tokenizer is CASE-SENSITIVE** → the Kotlin wrapper must NOT lowercase. Pipeline = `rust.encode(text)` → truncate to 64 → pad with id `0` (right). Parity target = Python `AutoProcessor` (fast `GemmaTokenizerFast`), proven byte-exact via the same Rust engine.
- **Resize = BILINEAR with antialiasing (PIL convolution bilinear)**, stretch-to-square (non-aspect-preserving), then `×1/255`, then `(x−0.5)/0.5` → `[−1,1]`, CHW. **`Bitmap.createScaledBitmap` is NOT acceptable** (no antialias prefilter → tens-of-LSB drift on downscale → label flips). Port PIL's separable triangle-filter resize.
- **Scoring** (per the §2 numerical contract): `cosine[f,l] = normalize(img[f])·normalize(txt[l])`; `confidence = sigmoid(cosine·exp(logit_scale) + logit_bias)`. `logit_scale`/`logit_bias` come from the manifest. Aggregations: mean/max/temporal/contrast (port `app/services/scoring.py`, minus the dead softmax `compute_frame_scores`).
- **ORT** ⚠️ **CORRECTED IN EXECUTION (see `phase1-report.md` ERRATA):** the embedding is the output named **`pooler_output`** ([rows,dim]) — select BY NAME, NOT `res[0]` (which is `last_hidden_state` [rows,seq,dim], the WRONG tensor). Host-verified: normalized `pooler_output` reproduces the golden cosine to 1.96e-7. The **XNNPACK EP collapses the symbolic batch dim to 1 for BOTH towers** (N>1 input → `[1,dim]`), so batched output is correct ONLY on the CPU EP — run **batch=1 per item** (vision per-frame, text per-label) under XNNPACK. Use direct `FloatBuffer`/`LongBuffer` inputs; copy outputs out before closing the `Result`.
- **Backends:** CPU/XNNPACK is the only real lane (NNAPI 0% delegated). so400m peak ~3.19 GB, batch ≤ 8, ~16 s/frame.

**Verified-correct by adversarial review — do NOT "fix" these:** the Rust JNI signatures + symbol name; ORT `createTensor(FloatBuffer/LongBuffer)`, ~~`res[0]`~~ (⚠️ WRONG — use `pooler_output` by name, see ERRATA above), `.floatBuffer`, `addXnnpack`; `Bitmap.getPixels` ARGB unpacking (`R = (p shr 16) and 0xFF`); `allocateDirect(...).order(nativeOrder()).asFloatBuffer()`; `@JvmStatic`/`System.loadLibrary`; the resampler center formula `(o+0.5)*scale-0.5`, `support=max(scale,1)`, per-axis W/H, border clamp + weight renormalization. These were checked and confirmed (EXCEPT `res[0]`, which adversarial review got wrong — corrected during Task 5 execution against the golden).

---

## File map (created in the Android project under `app/`)
```
app/src/main/cpp-tokenizer/           # Rust crate (built out-of-band via cargo-ndk)
  Cargo.toml
  src/lib.rs                          # JNI: createTokenizer / tokenize / deleteTokenizer
app/src/main/jniLibs/arm64-v8a/libhftokenizer.so   # cargo-ndk output (+ x86_64 for emulator)
app/src/main/java/com/example/clipcc/engine/
  Manifest.kt          # ModelBundleManifest data class + JSON parse (mirror Phase 0 schema v1)
  HfTokenizer.kt       # JNI wrapper + truncate-64/pad-0 (NO lowercasing)
  Resampler.kt         # PIL-faithful separable-triangle bilinear (pure: FloatArray in/out)
  Preprocess.kt        # Bitmap -> RGB float -> Resampler -> normalize -> CHW FloatBuffer
  FrameSampler.kt      # (DEFERRED to Plan 2 — NOT built in Phase 1; end-to-end parity uses lossless PNGs)
  OrtTower.kt          # load .onnx, run vision/text, L2-normalized embeddings
  Scoring.kt           # cosine/confidence + mean/max/temporal/contrast (pure)
  Engine.kt            # orchestrates: model -> frames -> preprocess -> encode -> score
app/src/test/java/com/example/clipcc/engine/        # JVM unit tests (fast)
  ResamplerTest.kt, ScoringTest.kt, ManifestTest.kt
app/src/androidTest/java/com/example/clipcc/engine/  # device tests (need .so/ORT/Media3/files)
  TokenizerParityTest.kt, PreprocessParityTest.kt, OrtTowerTest.kt, EndToEndParityTest.kt
app/src/androidTest/assets/fixtures/   # golden fixtures pushed into test assets
```
Test data (ONNX towers + tokenizer.json) read from `/data/local/tmp/clipcc_models/<model_id>/` (Phase 0 lesson: app can read /data/local/tmp, cannot write it; write outputs to `cacheDir`). Golden fixtures small enough to ship in `androidTest/assets`.

---

## Task 1: Tokenizer JNI bring-up (the gating de-risk)

**Files:** `app/src/main/cpp-tokenizer/{Cargo.toml,src/lib.rs}`, `app/src/main/jniLibs/<abi>/libhftokenizer.so`, `app/src/main/java/com/example/clipcc/engine/HfTokenizer.kt`, `app/src/androidTest/.../TokenizerParityTest.kt`, golden `app/src/androidTest/assets/fixtures/tokenizer_golden.json`.

- [ ] **Step 1: Toolchain**

Run:
```bash
rustup target add aarch64-linux-android x86_64-linux-android
cargo install cargo-ndk
# NDK r28c (28.2.13676358) — install via Android Studio SDK Manager if absent
ls "$HOME/Library/Android/sdk/ndk" 2>/dev/null
```
Expected: rust targets added; `cargo ndk --version` works; an NDK r28c dir exists.

- [ ] **Step 2: Rust crate** — create `app/src/main/cpp-tokenizer/Cargo.toml`:

```toml
[package]
name = "hftokenizer"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]
name = "hftokenizer"

[dependencies]
tokenizers = { version = "0.23.1", default-features = false, features = ["onig"] }
jni = "0.21.1"
serde_json = "1"
```
(Keep `onig` for now; dropping it is a later optimization — validate parity first.)

- [ ] **Step 3: JNI implementation** — create `app/src/main/cpp-tokenizer/src/lib.rs`:

```rust
use jni::objects::{JClass, JString, JByteArray};
use jni::sys::{jlong, jstring};
use jni::JNIEnv;
use tokenizers::Tokenizer;

#[no_mangle]
pub extern "system" fn Java_com_example_clipcc_engine_HfTokenizer_createTokenizer(
    mut env: JNIEnv, _c: JClass, bytes: JByteArray) -> jlong {
    let buf = env.convert_byte_array(&bytes).expect("bytes");
    let tk = Tokenizer::from_bytes(&buf).expect("tokenizer.json parse");
    Box::into_raw(Box::new(tk)) as jlong
}

#[no_mangle]
pub extern "system" fn Java_com_example_clipcc_engine_HfTokenizer_tokenize(
    mut env: JNIEnv, _c: JClass, ptr: jlong, text: JString) -> jstring {
    let tk = unsafe { &*(ptr as *const Tokenizer) };
    let s: String = env.get_string(&text).expect("text").into();
    let enc = tk.encode(s, true).expect("encode");           // add_special_tokens=true; NO lowercasing
    let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
    let out = serde_json::to_string(&ids).unwrap();
    env.new_string(out).expect("new_string").into_raw()
}

#[no_mangle]
pub extern "system" fn Java_com_example_clipcc_engine_HfTokenizer_deleteTokenizer(
    _env: JNIEnv, _c: JClass, ptr: jlong) {
    if ptr != 0 { unsafe { drop(Box::from_raw(ptr as *mut Tokenizer)); } }
}
```

- [ ] **Step 4: Build the .so into jniLibs**

Run:
```bash
cd app/src/main/cpp-tokenizer
cargo ndk -t arm64-v8a -t x86_64 --platform 24 -o ../jniLibs build --release
ls ../jniLibs/arm64-v8a/libhftokenizer.so
```
Expected: `libhftokenizer.so` present under `jniLibs/arm64-v8a/` (and `x86_64/`). Add a README note that this is a manual pre-build step (not wired into Gradle).

- [ ] **Step 5: Kotlin wrapper** — create `HfTokenizer.kt`:

```kotlin
package com.example.clipcc.engine

import org.json.JSONArray

class HfTokenizer private constructor(private var ptr: Long) : AutoCloseable {
    companion object {
        init { System.loadLibrary("hftokenizer") }
        const val MAX_LEN = 64
        const val PAD_ID = 0L
        fun fromJson(bytes: ByteArray) = HfTokenizer(createTokenizer(bytes))
        @JvmStatic private external fun createTokenizer(bytes: ByteArray): Long
        @JvmStatic private external fun tokenize(ptr: Long, text: String): String
        @JvmStatic private external fun deleteTokenizer(ptr: Long)
    }

    /** Raw subword ids from the Rust engine (case-sensitive; specials included). */
    fun encodeRaw(text: String): LongArray {
        val arr = JSONArray(tokenize(ptr, text))
        return LongArray(arr.length()) { arr.getLong(it) }
    }

    /** AutoProcessor-equivalent: encode -> truncate to 64 -> right-pad with 0. NO lowercasing. */
    fun encodePadded(text: String): LongArray {
        val ids = encodeRaw(text)
        return LongArray(MAX_LEN) { if (it < ids.size) ids[it] else PAD_ID }
    }

    override fun close() { if (ptr != 0L) { deleteTokenizer(ptr); ptr = 0L } }
}
```

- [ ] **Step 6: Stage tokenizer.json + golden on device/assets**

Run:
```bash
ADB="$HOME/Library/Android/sdk/platform-tools/adb"
$ADB shell mkdir -p /data/local/tmp/clipcc_models/siglip2-base-patch16-256
$ADB push build/android_assets/siglip2-base-patch16-256/tokenizer.json /data/local/tmp/clipcc_models/siglip2-base-patch16-256/
$ADB shell chmod -R a+rX /data/local/tmp/clipcc_models
cp build/android_assets/fixtures/tokenizer_golden.json /Users/austin/AndroidStudioProjects/ClipCC/app/src/androidTest/assets/fixtures/
```

- [ ] **Step 7: Instrumented parity test** — create `TokenizerParityTest.kt`:

```kotlin
package com.example.clipcc.engine

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.junit.Assert.assertArrayEquals
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class TokenizerParityTest {
    @Test fun matches_python_golden_byte_exact() {
        val ctx = InstrumentationRegistry.getInstrumentation().context
        val tokBytes = File("/data/local/tmp/clipcc_models/siglip2-base-patch16-256/tokenizer.json").readBytes()
        val golden = JSONArray(ctx.assets.open("fixtures/tokenizer_golden.json").bufferedReader().readText())
        HfTokenizer.fromJson(tokBytes).use { tk ->
            for (i in 0 until golden.length()) {
                val row = golden.getJSONObject(i)
                val text = row.getString("text")
                val exp = row.getJSONArray("input_ids").let { LongArray(it.length()) { j -> it.getLong(j) } }
                val got = tk.encodePadded(text)
                assertArrayEquals("token ids for '$text'", exp, got)
            }
        }
    }
}
```

- [ ] **Step 8: Build + run; GATE**

Run:
```bash
cd /Users/austin/AndroidStudioProjects/ClipCC
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
./gradlew :app:connectedDebugAndroidTest --console=plain \
  -Pandroid.testInstrumentationRunnerArguments.class=com.example.clipcc.engine.TokenizerParityTest 2>&1 | tail -6
```
Expected: BUILD SUCCESSFUL, test passes — on-device tokenization is byte-exact vs the Python golden for all cases (incl. mixed case). **If any case differs**, the Metaspace/BOS prefix-space is the usual culprit — inspect `encodeRaw("Car")` vs the golden and adjust (do NOT lowercase). This gate de-risks the whole engine.

- [ ] **Step 9: Save** (Android project — keep files; no git commit unless repo initialized).

---

## Task 2: PIL-faithful bilinear resampler (pure, JVM-tested)

**Files:** `app/src/main/java/com/example/clipcc/engine/Resampler.kt`, `app/src/test/java/com/example/clipcc/engine/ResamplerTest.kt`.

- [ ] **Step 1: Failing JVM test** — create `ResamplerTest.kt`. It checks the separable triangle filter against a hand-computed 2→1 downscale (antialiased average), proving the prefilter:

```kotlin
package com.example.clipcc.engine

import org.junit.Assert.assertEquals
import org.junit.Test

class ResamplerTest {
    @Test fun downscale_2to1_averages_like_pil_triangle() {
        // 2x2 single-channel image [[0,1],[1,1]] -> 1x1 should be the antialiased mean = 0.75
        val src = floatArrayOf(0f, 1f, 1f, 1f)
        val out = Resampler.resizeChannelMajor(src, srcW = 2, srcH = 2, dstW = 1, dstH = 1, channels = 1)
        assertEquals(0.75f, out[0], 1e-4f)
    }

    @Test fun identity_when_same_size() {
        val src = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)
        val out = Resampler.resizeChannelMajor(src, 2, 2, 2, 2, 1)
        assertEquals(0.1f, out[0], 1e-4f); assertEquals(0.4f, out[3], 1e-4f)
    }

    /** Asymmetric 4->1 downscale of [0,0,0,4]: PIL antialiased triangle (support=scale=4) gives
     *  ~0.833 (weights .2083/.2917/.2917/.2083); a NO-antialias impl (support=1) would give 0.0.
     *  This is the test that actually guards the prefilter the resampler exists for. */
    @Test fun downscale_4to1_antialiased_not_plain_bilinear() {
        val out = Resampler.resizeChannelMajor(floatArrayOf(0f, 0f, 0f, 4f), 4, 1, 1, 1, 1)
        assertEquals(0.8333f, out[0], 1e-3f)
    }
}
```

- [ ] **Step 2: Run — fails** (`Resampler` undefined).

Run: `./gradlew :app:testDebugUnitTest --tests "*ResamplerTest*"` → FAIL.

- [ ] **Step 3: Implement `Resampler.kt`** — separable triangle (PIL bilinear) with per-axis support:

```kotlin
package com.example.clipcc.engine

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max

/** PIL/Pillow convolution bilinear (triangle filter), separable, half-pixel centers,
 *  antialiased on downscale. Input/output are HWC-interleaved float planes (channels last). */
object Resampler {
    private fun triangle(x: Float): Float { val a = if (x < 0) -x else x; return if (a < 1f) 1f - a else 0f }

    private fun axisWeights(srcN: Int, dstN: Int): Pair<IntArray, Array<FloatArray>> {
        val scale = srcN.toFloat() / dstN
        val support = if (scale > 1f) scale else 1f           // triangle support; widen on downscale
        val starts = IntArray(dstN); val weights = Array(dstN) { FloatArray(0) }
        for (o in 0 until dstN) {
            val center = (o + 0.5f) * scale - 0.5f            // half-pixel center
            val lo = max(0, floor(center - support).toInt())
            val hi = minOf(srcN - 1, ceil(center + support).toInt())
            val w = FloatArray(hi - lo + 1); var sum = 0f
            for (s in lo..hi) { val ww = triangle((s - center) / support); w[s - lo] = ww; sum += ww }
            if (sum > 0f) for (k in w.indices) w[k] /= sum
            starts[o] = lo; weights[o] = w
        }
        return starts to weights
    }

    /** resize one interleaved float image [srcH*srcW*channels] -> [dstH*dstW*channels]. */
    fun resizeChannelMajor(src: FloatArray, srcW: Int, srcH: Int, dstW: Int, dstH: Int, channels: Int): FloatArray {
        val (xs, xw) = axisWeights(srcW, dstW)
        val (ys, yw) = axisWeights(srcH, dstH)
        // horizontal pass: srcH x dstW
        val tmp = FloatArray(srcH * dstW * channels)
        for (y in 0 until srcH) for (ox in 0 until dstW) {
            val w = xw[ox]; val s0 = xs[ox]
            for (c in 0 until channels) {
                var acc = 0f
                for (k in w.indices) acc += w[k] * src[(y * srcW + (s0 + k)) * channels + c]
                tmp[(y * dstW + ox) * channels + c] = acc
            }
        }
        // vertical pass: dstH x dstW
        val out = FloatArray(dstH * dstW * channels)
        for (oy in 0 until dstH) for (x in 0 until dstW) {
            val w = yw[oy]; val s0 = ys[oy]
            for (c in 0 until channels) {
                var acc = 0f
                for (k in w.indices) acc += w[k] * tmp[((s0 + k) * dstW + x) * channels + c]
                out[(oy * dstW + x) * channels + c] = acc
            }
        }
        return out
    }
}
```

- [ ] **Step 4: Run — passes.** `./gradlew :app:testDebugUnitTest --tests "*ResamplerTest*"` → PASS (both cases; 0.75 proves the antialiased prefilter).

- [ ] **Step 5: Save.**

---

## Task 3: Preprocess (Bitmap → CHW FloatBuffer) + device parity gate

**Files:** `Preprocess.kt`, `app/src/androidTest/.../PreprocessParityTest.kt`, golden `preprocess_golden` (+ the lossless PNG frames) in androidTest assets.

- [ ] **Step 1: Implement `Preprocess.kt`**:

```kotlin
package com.example.clipcc.engine

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

object Preprocess {
    /** Bitmap -> SigLIP2 tensor: RGB, stretch-to-square via PIL-bilinear, /255, (x-0.5)/0.5, CHW. */
    fun toCHW(bitmap: Bitmap, res: Int): FloatBuffer {
        val w = bitmap.width; val h = bitmap.height
        val px = IntArray(w * h); bitmap.getPixels(px, 0, w, 0, 0, w, h)
        val rgb = FloatArray(w * h * 3)                       // interleaved RGB float [0,255]
        for (i in px.indices) {
            val p = px[i]
            rgb[i * 3] = ((p shr 16) and 0xFF).toFloat()     // R (NOT BGR)
            rgb[i * 3 + 1] = ((p shr 8) and 0xFF).toFloat()  // G
            rgb[i * 3 + 2] = (p and 0xFF).toFloat()           // B
        }
        val resized = Resampler.resizeChannelMajor(rgb, w, h, res, res, 3)  // stretch to res x res
        val buf = ByteBuffer.allocateDirect(3 * res * res * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        // CHW + normalize: (v/255 - 0.5)/0.5 = v/127.5 - 1
        for (c in 0 until 3) for (i in 0 until res * res)
            buf.put(resized[i * 3 + c] / 127.5f - 1f)
        buf.rewind(); return buf
    }
}
```

- [ ] **Step 2: Stage frames + golden + a manifest into androidTest assets**

Run:
```bash
A=/Users/austin/AndroidStudioProjects/ClipCC/app/src/androidTest/assets/fixtures
cp tools/android_assets/tests/fixtures/lossless/*.png "$A/"
# Convert preprocess_golden.npz -> JSON the test can read (flat per-frame CHW), via the export venv:
.venv-export/bin/python - <<'PY'
import numpy as np, json
from pathlib import Path
d = np.load("build/android_assets/fixtures/preprocess_golden.npz")["pixel_values"]  # [F,3,R,R]
out = {"shape": list(d.shape), "data": d.astype(float).round(6).flatten().tolist()}
Path("/Users/austin/AndroidStudioProjects/ClipCC/app/src/androidTest/assets/fixtures/preprocess_golden.json").write_text(json.dumps(out))
print("frames", d.shape)
PY
```

- [ ] **Step 3: Device parity test** — create `PreprocessParityTest.kt`:

```kotlin
package com.example.clipcc.engine

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs

@RunWith(AndroidJUnit4::class)
class PreprocessParityTest {
    @Test fun chw_matches_pil_golden_within_tolerance() {
        val ctx = InstrumentationRegistry.getInstrumentation().context
        val g = JSONObject(ctx.assets.open("fixtures/preprocess_golden.json").bufferedReader().readText())
        val shape = g.getJSONArray("shape"); val res = shape.getInt(2)
        val data = g.getJSONArray("data")
        val frames = shape.getInt(0); val per = 3 * res * res
        // frame files were generated as frame_000.png, frame_001.png ...
        var maxAbs = 0f
        for (f in 0 until frames) {
            val bmp = ctx.assets.open("fixtures/frame_%03d.png".format(f)).use { BitmapFactory.decodeStream(it) }
            val buf = Preprocess.toCHW(bmp, res)
            for (i in 0 until per) maxAbs = maxOf(maxAbs, abs(buf.get(i) - data.getDouble(f * per + i).toFloat()))
        }
        println("PREPROCESS max_abs_diff=$maxAbs")
        assertTrue("max abs diff $maxAbs <= 0.016", maxAbs <= 0.016f)
    }
}
```

- [ ] **Step 4: Build + run; GATE.** `connectedDebugAndroidTest --tests "*PreprocessParityTest*"`. Expected: PASS, `PREPROCESS max_abs_diff` ≤ 0.016. If it exceeds: check R/G/B channel order, half-pixel center, and per-axis support (stretch uses different x/y scales). Capture the printed diff for the Plan-1 report.

- [ ] **Step 5: Save.**

---

## Task 4: Manifest consumer (pure, JVM-tested)

**Files:** `Manifest.kt`, `app/src/test/.../ManifestTest.kt`.

- [ ] **Step 1: Failing test** — parse a sample manifest JSON (copy a real one from `build/android_assets/siglip2-base-patch16-256/manifest.json` into `app/src/test/resources/`), assert key fields:

```kotlin
package com.example.clipcc.engine

import org.junit.Assert.assertEquals
import org.junit.Test

class ManifestTest {
    @Test fun parses_schema_v1() {
        val json = javaClass.classLoader!!.getResource("manifest_base256.json")!!.readText()
        val m = ModelBundleManifest.parse(json)
        assertEquals(1, m.schemaVersion)
        assertEquals("siglip2-base-patch16-256", m.modelId)
        assertEquals(256, m.resolution)
        assertEquals("bilinear", m.resample)
        assertEquals("tokenizer_json", m.lowercaseAppliedBy)
        assertEquals("vision_model.onnx", m.visionFile)
        assertEquals("text_model.onnx", m.textFile)
        assertNull(m.visionDataFile)   // base-256 single-file; guards the org.json null-handling
        assertNull(m.textDataFile)
        // logit_scale/bias are read as Doubles
        assertEquals(4.7265, m.logitScale, 1e-3)
    }
}
```

- [ ] **Step 2: Run — fails.**

- [ ] **Step 3: Implement `Manifest.kt`** (mirror Phase 0 schema v1; only the fields the engine needs):

```kotlin
package com.example.clipcc.engine

import org.json.JSONObject

data class ModelBundleManifest(
    val schemaVersion: Int, val modelId: String, val resolution: Int,
    val precision: String, val logitScale: Double, val logitBias: Double,
    val visionFile: String, val visionDataFile: String?,
    val textFile: String, val textDataFile: String?,
    val tokenizerFile: String, val lowercaseAppliedBy: String,
    val resample: String, val maxLength: Int, val padId: Int,
) {
    companion object {
        fun parse(json: String): ModelBundleManifest {
            val d = JSONObject(json)
            require(d.getInt("schema_version") == 1) { "unsupported schema_version" }
            val v = d.getJSONObject("vision"); val t = d.getJSONObject("text")
            val tok = d.getJSONObject("tokenizer"); val pp = d.getJSONObject("preprocess")
            return ModelBundleManifest(
                schemaVersion = 1, modelId = d.getString("model_id"),
                resolution = d.getInt("resolution"), precision = d.getString("precision"),
                logitScale = d.getDouble("logit_scale"), logitBias = d.getDouble("logit_bias"),
                visionFile = v.getString("file"),
                visionDataFile = if (v.isNull("data_file")) null else v.getString("data_file"),
                textFile = t.getString("file"),
                textDataFile = if (t.isNull("data_file")) null else t.getString("data_file"),
                tokenizerFile = tok.getString("file"), lowercaseAppliedBy = tok.getString("lowercase_applied_by"),
                resample = pp.getString("resample"), maxLength = tok.getInt("max_length"), padId = tok.getInt("pad_id"),
            )
        }
    }
}
```
(Use `isNull(...)`, NOT `optString(...).ifEmpty{null}`: Android's `org.json.optString` on an
explicit JSON `null` can return the literal `"null"`, not `""`. `isNull` is correct on all API levels.)

- [ ] **Step 4: Run — passes. Step 5: Save.**

---

## Task 5: ORT towers — embeddings (device-tested)

**Files:** `OrtTower.kt`, `app/src/androidTest/.../OrtTowerTest.kt`.

- [ ] **Step 1: Implement `OrtTower.kt`**:

```kotlin
package com.example.clipcc.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.LongBuffer
import java.nio.FloatBuffer
import kotlin.math.sqrt

class OrtTower(private val session: OrtSession, private val env: OrtEnvironment) : AutoCloseable {
    companion object {
        fun open(path: String, env: OrtEnvironment): OrtTower =
            OrtTower(env.createSession(path, OrtSession.SessionOptions().apply {
                addXnnpack(mapOf("intra_op_num_threads" to "4"))
            }), env)
    }
    private fun inputName() = session.inputNames.first()

    /** Run with a [rows, *] tensor; return L2-normalized embeddings as [rows][dim]. */
    private fun runEmbed(buf: java.nio.Buffer, shape: LongArray, rows: Int): Array<FloatArray> {
        val tensor = when (buf) {
            is FloatBuffer -> OnnxTensor.createTensor(env, buf, shape)
            is LongBuffer -> OnnxTensor.createTensor(env, buf, shape)
            else -> error("buffer type")
        }
        tensor.use { t ->
            session.run(mapOf(inputName() to t)).use { res ->
                val out = (res[0] as OnnxTensor).floatBuffer        // flat [rows*dim], copy before close
                val dim = out.capacity() / rows
                return Array(rows) { r ->
                    val v = FloatArray(dim) { out.get(r * dim + it) }
                    var n = 0f; for (x in v) n += x * x; n = sqrt(n)
                    if (n > 0f) for (i in v.indices) v[i] = v[i] / n
                    v
                }
            }
        }
    }

    fun encodeVision(pixelValues: FloatBuffer, frames: Int, res: Int): Array<FloatArray> {
        // NB: rewind() as a separate statement via Buffer — the covariant FloatBuffer.rewind():FloatBuffer
        // override only exists in Android libcore on API 33+, so `x.rewind() as FloatBuffer` would
        // NoSuchMethodError on minSdk 24–32. Cast the RECEIVER to Buffer, then pass the typed var.
        (pixelValues as java.nio.Buffer).rewind()
        return runEmbed(pixelValues, longArrayOf(frames.toLong(), 3, res.toLong(), res.toLong()), frames)
    }

    fun encodeText(inputIds: LongBuffer, labels: Int, maxLen: Int): Array<FloatArray> {
        (inputIds as java.nio.Buffer).rewind()
        return runEmbed(inputIds, longArrayOf(labels.toLong(), maxLen.toLong()), labels)
    }

    override fun close() = session.close()
}
```

- [ ] **Step 2: Device test** — `OrtTowerTest.kt`: load base-256 vision tower from `/data/local/tmp/clipcc_models/...`, run a single dummy frame, assert embeddings shape `[1][D]` and that the vector is L2-normalized (‖v‖≈1). Run the text tower with one padded label, assert `[1][D]`. Also assert **batch correctness**: run 2 identical frames batched and confirm both rows equal the single-frame embedding within 1e-3 (proves batched output isn't corrupted by the shape-reuse warning).

```kotlin
// key assertion sketch
val env = OrtEnvironment.getEnvironment()
OrtTower.open("$dir/vision_model.onnx", env).use { v ->
    val one = Preprocess.toCHW(dummyBitmap, 256)
    val e1 = v.encodeVision(one, 1, 256)
    assertEquals(1, e1.size); val norm = sqrt(e1[0].sumOf { (it*it).toDouble() }); assertEquals(1.0, norm, 1e-2)
    val two = batchOf(dummyBitmap, dummyBitmap, 256)   // [2,3,256,256]
    val e2 = v.encodeVision(two, 2, 256)
    for (i in e1[0].indices) assertEquals(e1[0][i], e2[0][i], 1e-3f)   // batched == single
}
```

- [ ] **Step 3: Build + run; GATE.** Expected PASS: embeddings normalized, batched rows match single. Captures whether N>1 is safe per model (note in report; if a model's axis is fixed at 1, fall back to per-frame loop for it).

- [ ] **Step 4: Save.**

---

## Task 6: Scoring port (pure, JVM-tested — mirrors `tests/test_scoring.py`)

**Files:** `Scoring.kt`, `app/src/test/.../ScoringTest.kt`.

- [ ] **Step 1: Failing tests** — port these SPECIFIC invariants from `tests/test_scoring.py`, reproducing the exact Python numbers (read the referenced assertions):
  - **mean**: per-label mean confidence + argmax best-match.
  - **max**: per-label max confidence + peak frame index + approx timestamp.
  - **temporal end-clamp**: `test_temporal_basic_segment_detection` — a segment's `end_time == 8.0` (`test_scoring.py:195`) and `== 5.0` (`:211`), both derived from `FrameInterval.end = min(start + 1/fps, video_duration)`. This case REQUIRES the FrameTimeline port (B3) — a flat timestamp list cannot reproduce final-frame clamping.
  - **contrast** difference-vs-group-mean invariant (`test_scoring.py:544`: `difference == 0.8` while `group_diff == 0.08`); `top_k_mean` with `k = ceil(0.10 * n)` (`scoring.py:90`); verdict by `±threshold`.

- [ ] **Step 2: Run — fails.**

- [ ] **Step 3: Implement `Scoring.kt` + `FrameTimeline`** — pure functions/types:
  - `data class ScoreMatrices(val cosine: Array<FloatArray>, val confidence: Array<FloatArray>)` — the carrier from `scoreMatrix` into aggregation (Task 7 uses this exact name). (Python `ScoreBatch` also carries `logits`; intentionally omitted — aggregations read only confidence/raw_similarity.)
  - `scoreMatrix(img: Array<FloatArray>, txt: Array<FloatArray>, logitScale: Double, logitBias: Double): ScoreMatrices` — `cosine = img·txtᵀ`, `confidence = sigmoid(cosine*exp(scale)+bias)`.
  - **`FrameTimeline` port** (from `app/services/frame_timeline.py`): built from `fps` + `videoDuration`, producing `FrameInterval(index, start, end)` with `end = min(start + 1/fps, videoDuration)`; expose `timestamp(i)`, `gapSeconds(a,b)`, `segmentDuration(start,end)`, `intervals[i].end`. Temporal aggregation depends on this (B3) — a flat timestamp list cannot reproduce end-clamping or active-duration sums.
  - `aggregateMean`, `aggregateMax`, `aggregateTemporal(threshold, gapTol, minDur, timeline: FrameTimeline)`, `aggregateContrast(posCount, reduce, threshold)`.
  - Data classes mirroring `schemas/response.py` (ScoreItem, Segment, LabelSummary, ContrastResult). Round to 6 dp to match Python.
  - **Do NOT** port `compute_frame_scores` (dead softmax).

- [ ] **Step 4: Run — passes** (all ported invariants green). **Step 5: Save.**

---

## Task 7: Engine orchestration + end-to-end parity (device GATE)

**Files:** `Engine.kt`, `app/src/androidTest/.../EndToEndParityTest.kt`, golden `scores_golden.json` in assets.

- [ ] **Step 1: Implement `Engine.kt`** — ties it together:

```kotlin
class Engine(private val modelDir: String, private val manifest: ModelBundleManifest, private val env: OrtEnvironment) {
    fun scoreFrames(bitmaps: List<Bitmap>, labels: List<String>): ScoreMatrices {
        // §5.5 memory order (Spike 0d): encode text, RELEASE the text session, THEN open vision —
        // only ONE large session resident at a time (so400m peak ~3.19 GB if both; avoid it).
        val txt: Array<FloatArray> =
            HfTokenizer.fromJson(File("$modelDir/${manifest.tokenizerFile}").readBytes()).use { tk ->
                val ids = labels.map { tk.encodePadded(it) }          // NO lowercasing
                OrtTower.open("$modelDir/${manifest.textFile}", env).use { t ->
                    t.encodeText(packLongs(ids), labels.size, manifest.maxLength)   // text session closed here
                }
            }
        return OrtTower.open("$modelDir/${manifest.visionFile}", env).use { v ->
            // batch ≤ 8 for so400m (Spike 0d); base/large may batch larger. Chunk bitmaps accordingly.
            val pix = packFrames(bitmaps, manifest.resolution)
            val img = v.encodeVision(pix, bitmaps.size, manifest.resolution)
            Scoring.scoreMatrix(img, txt, manifest.logitScale, manifest.logitBias)
        }
    }
}
```
Helper signatures to implement in `Engine.kt`:
- `packLongs(idsPerLabel: List<LongArray>): LongBuffer` — direct `LongBuffer` of `[L*64]` int64, row-major.
- `packFrames(bitmaps: List<Bitmap>, res: Int): FloatBuffer` — direct `FloatBuffer` `[N*3*res*res]`, each frame via `Preprocess.toCHW` concatenated (or run `Preprocess.toCHW` per frame and copy in).

For so400m, chunk `bitmaps` into batches ≤ 8 and concatenate the resulting embeddings (don't allocate one batch-300 buffer). base/large can use a larger chunk.

- [ ] **Step 2: Stage `scores_golden.json` into assets** (already produced by Phase 0; copy it).

- [ ] **Step 3: End-to-end parity test** — `EndToEndParityTest.kt`: load the lossless frames + labels `["Car","texting while driving","a dog"]`, run `Engine.scoreFrames`, compare `confidence` AND `cosine` to `scores_golden.json`. Gate (all three):
  - **`cosine` max abs diff ≤ 0.01** — the sensitive signal that actually validates embedding parity (golden cosines ~0.02–0.07; relies on the SigLIP2 identity `logits_per_image == cosine*exp(scale)+bias`, true by construction).
  - **`confidence` max abs diff ≤ 0.02** — secondary (confidence sits in the near-flat tail of sigmoid here, so this alone is a weak check).
  - **no best-match label flips** vs golden.
  Print all diffs. (If cosine drifts > 0.01, suspect the resampler antialias or a channel-order bug — Task 3, not the model.)

- [ ] **Step 4: Build + run; GATE.** Expected PASS within tolerance. This is the Phase 1 acceptance gate — on-device scores match the Python fp32 reference. Capture the diff.

- [ ] **Step 5: Save.**

---

## Phase 1 Definition of Done
- [ ] Tokenizer `.so` builds; on-device tokenization byte-exact vs Python golden (Task 1 gate).
- [ ] Resampler JVM tests pass; preprocess CHW matches PIL golden ≤ 0.016 on device (Tasks 2–3).
- [ ] ORT towers return L2-normalized embeddings; batched output == single (Task 5).
- [ ] Scoring port passes all `test_scoring.py`-equivalent invariants (Task 6).
- [ ] End-to-end on-device scores match `scores_golden.json` ≤ 0.02, no label flips (Task 7).
- [ ] A short Phase-1 report records: tokenizer parity, preprocess max-diff, batched-safe models, end-to-end diff, and any per-device color caveat.

## Open items carried to Plan 2 (benchmark) / Plan 3 (UI)
- Resampler tolerance vs torchvision-fast server path (fixtures are slow-PIL); decide if the running server must match.
- Real network downloader (resume/Xet/external-data) — engine currently reads from `/data/local/tmp`.
- Per-model batch defaults + so400m text-first/release wiring (from Spike 0d).
- FrameSampler (Media3) is specced in §5.4 but the end-to-end parity test uses lossless PNGs; wire Media3 decode + its color caveat in Plan 2/benchmark over real video.
