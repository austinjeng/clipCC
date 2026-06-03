# clipCC-Android Phase 2 — Benchmark Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A headless on-device benchmark harness that measures SigLIP2 inference speed for the 4 `benchmark-v1` models across ORT backend configs over a real decoded video, emits honest per-model backend-capability evidence, and retrieves structured metrics off-device — without breaking Plan-1 parity.

**Architecture:** Extend the Plan-1 engine in the existing `:app` module. Refactor `OrtTower` into a backend-aware session factory with per-item (XNNPACK) vs batched (CPU_EP) encode strategies; add `FrameSampler` (Media3), `BackendCapabilityReport` (untimed profiling probe), and a `Benchmark` runner (FramePrep-once + TimedRun-per-backend, warm-up + median-of-3 + thermal metadata). Results are written to the app external files dir and pulled via `am instrument` (no auto-uninstall). `Preprocess`/`Scoring` are reused unchanged.

**Tech Stack:** Kotlin 2.0, ORT `onnxruntime-android:1.26.0` (XNNPACK + CPU EP + NNAPI EP), Media3 `androidx.media3:media3-inspector-frame:1.10.1` (`FrameExtractor`), JUnit4 + androidx.test (instrumented).

**Inputs:** spec `docs/superpowers/specs/2026-06-03-clipcc-android-phase2-benchmark-design.md`; Plan-1 engine (`HfTokenizer`, `Resampler`, `Preprocess`, `Manifest`, `Scoring`, `OrtTower`, `Engine`) in `app/src/main/java/com/example/clipcc/engine/`; Plan-1 report `phase1-report.md`.

**Android project:** `/Users/austin/AndroidStudioProjects/ClipCC` (`:app`, `com.example.clipcc`, minSdk 24, AGP 9.0.0-rc01; NOT a git repo — "commit" = save the file). Build env: `JAVA_HOME=/Applications/Android Studio.app/Contents/jbr/Contents/Home`. adb: `$HOME/Library/Android/sdk/platform-tools/adb`. Device: Pixel 7a `36161JEHN16600` (Android 16 / API 36).

---

## Resolved facts carried from Plan 0/1 + the design (do not re-litigate)
- **Embedding = ONNX `pooler_output` selected BY NAME**, not `res[0]`. `OrtTower.runEmbed` already does this and has a `check(rowsData.size == rows)` guard — keep it.
- **XNNPACK collapses the symbolic batch dim → 1** for both towers (so per-frame under XNNPACK; the `check` enforces it). **CPU EP batches correctly** (host-verified to 1.96e-7 vs golden). This is the whole point of the two CPU lanes.
- **NNAPI is one ORT EP + `NNAPIFlags`**, not GPU-vs-NPU. `NNAPI_CPU_DISABLED` disables NNAPI's CPU *device* (not "hardware-only"); unsupported nodes still fall back to ORT CPU and session-create may fail. NNAPI lanes are **capability probes only**, never timed.
- **so400m ≈ 16 s/frame**; peak ~3.19 GB with both towers — keep the §5.5 memory order (encode text → release text session → open vision). Vision batch map (CPU_EP): base-256=16, base-384=16, large-384=8, so400m-384=4.
- **Model precisions:** base-256/384 = fp32; large-384/so400m-384 = fp16. All single-file (no `.onnx_data`).
- **rewind gotcha:** `(buf as java.nio.Buffer).rewind()` (covariant override absent minSdk<33).
- **ORT is an `implementation` dep**; `org.json:json` is `testImplementation` (Android stubs org.json on the JVM). `media3-inspector-frame:1.10.1` is verified the latest on Google Maven (artifact only ships 1.10.x).

## File map (created/modified in the Android project under `app/`)
```
app/build.gradle.kts                                   # MODIFY: add media3 dep (catalog)
gradle/libs.versions.toml                              # MODIFY: media3 version + lib alias
app/src/main/java/com/example/clipcc/engine/
  Backend.kt              # NEW: Backend enum + BackendConfig + OpenOutcome
  OrtTower.kt             # MODIFY: backend-aware open(); encodeVision(batch); record outcomes
  Engine.kt              # MODIFY: backend-parameterized; encode strategy via OrtTower
  FrameSampler.kt        # NEW: Media3 FrameExtractor wrapper (single-thread) + VideoMeta + SampledFrame
  BackendCapability.kt   # NEW: BackendCapabilityReport + profiling-JSON provider-coverage parser
  Benchmark.kt           # NEW: FramePrepResult, TimedRun, RunMetadata, BenchmarkResult, Benchmark runner + JSON
app/src/androidTest/java/com/example/clipcc/engine/
  OrtBackendTest.kt          # NEW: CPU lanes open + batched==per-frame consistency + NNAPI attempt records outcome
  BackendCapabilityTest.kt   # NEW: untimed probe emits provider coverage (base-256)
  FrameSamplerTest.kt        # NEW: decode test video; frame count/dims/rotation/color smoke
  BenchmarkSmokeTest.kt      # NEW: base-256/CPU_EP/2-frame fast smoke; writes+reads result JSON
  BenchmarkMatrixTest.kt     # NEW: full 4-model matrix (Task 7 gate); long-running
```
Models read from `/data/local/tmp/clipcc_models/<model_id>/`; test video from `/data/local/tmp/clipcc_bench/test.mp4`; benchmark JSON written to the app external files dir and pulled via `am instrument`.

---

## Task 0: Prerequisites — stage models + test video on device

**No code.** Run these once; they are offline-safe (onnxruntime cached) except none need network.

- [ ] **Step 1: Push the 3 not-yet-staged bundles**

Run:
```bash
ADB="$HOME/Library/Android/sdk/platform-tools/adb"
SRC=/Users/austin/MITAC/clipCC/build/android_assets
for m in siglip2-base-patch16-384 siglip2-large-patch16-384 siglip2-so400m-patch14-384; do
  $ADB shell mkdir -p /data/local/tmp/clipcc_models/$m
  $ADB push $SRC/$m/vision_model.onnx $SRC/$m/text_model.onnx $SRC/$m/tokenizer.json $SRC/$m/manifest.json \
    /data/local/tmp/clipcc_models/$m/
done
$ADB shell chmod -R a+rX /data/local/tmp/clipcc_models
$ADB shell 'for d in /data/local/tmp/clipcc_models/*/; do echo "$d"; ls "$d"; done'
```
Expected: all 4 model dirs each list `manifest.json text_model.onnx tokenizer.json vision_model.onnx`.

- [ ] **Step 2: Stage the SDR test clip**

Run:
```bash
ADB="$HOME/Library/Android/sdk/platform-tools/adb"
$ADB shell mkdir -p /data/local/tmp/clipcc_bench
$ADB shell cp "/sdcard/Movies/FlexibilityCC/FlexibilityCC_20260513_091459.mp4" /data/local/tmp/clipcc_bench/test.mp4
$ADB shell chmod -R a+rX /data/local/tmp/clipcc_bench
# Verify it is SDR (no HDR transfer); print video stream metadata:
$ADB shell 'ls -la /data/local/tmp/clipcc_bench/test.mp4'
```
Expected: `test.mp4` present (~9 MB). Note its color/HDR status (the clip is a phone screen/camera recording; treat as SDR; if `mediainfo`/`ffprobe` is available on host, confirm `transfer_characteristics` is not PQ/HLG). Record the finding for `FrameSampler` (Task 4).

- [ ] **Step 3: Confirm Plan-1 engine still builds + parity holds (regression baseline before refactor)**

Run:
```bash
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
cd /Users/austin/AndroidStudioProjects/ClipCC
./gradlew :app:testDebugUnitTest 2>&1 | tail -5
```
Expected: BUILD SUCCESSFUL (Resampler/Manifest/Scoring green). This is the pre-refactor baseline.

---

## Task 1: Backend config + backend-aware `OrtTower` session factory

**Files:**
- Create: `app/src/main/java/com/example/clipcc/engine/Backend.kt`
- Modify: `app/src/main/java/com/example/clipcc/engine/OrtTower.kt`
- Test: `app/src/androidTest/java/com/example/clipcc/engine/OrtBackendTest.kt`

- [ ] **Step 1: Create `Backend.kt`**

```kotlin
package com.example.clipcc.engine

import ai.onnxruntime.OrtSession

/** ORT backend configurations. NNAPI is ONE EP with flags (no GPU-vs-NPU selection);
 *  the NNAPI runtime picks hardware opaquely. NNAPI lanes are capability probes only. */
enum class Backend { CPU_XNNPACK, CPU_EP, NNAPI_DEFAULT, NNAPI_CPU_DISABLED }

/** Pinned session settings, recorded in every result for fair comparison. */
data class BackendConfig(
    val intraOpThreads: Int = 4,
    val interOpThreads: Int = 1,
    val graphOpt: OrtSession.SessionOptions.OptLevel = OrtSession.SessionOptions.OptLevel.ALL_OPT,
    val memoryPattern: Boolean = false,
)

/** Outcome of trying to build a session for a backend (for honest capability reporting). */
data class OpenOutcome(
    val backend: Backend,
    val addEpOutcome: String,       // "ok" | "threw: <msg>" | "n/a"
    val sessionCreateOutcome: String, // "ok" | "threw: <msg>"
)
```

- [ ] **Step 2: Replace `OrtTower.kt` with the backend-aware version**

The current `OrtTower` hardcodes XNNPACK and loops per-frame. Replace it with this (keeps `pooler_output`-by-name + the `check(rows)` guard + the rewind cast; adds backend selection, a chunked `encodeVision(batch)`, and outcome capture):

```kotlin
package com.example.clipcc.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.EnumSet
import kotlin.math.sqrt

class OrtTower private constructor(
    private val session: OrtSession,
    private val env: OrtEnvironment,
    val backend: Backend,
) : AutoCloseable {
    companion object {
        /** Build SessionOptions for [backend]+[config]. NNAPI EP-add is wrapped: on throw we record it
         *  in [outcome] and continue with whatever EPs applied (never relabel). Throws only if
         *  createSession itself fails. [profilePath] (non-null) enables ORT profiling to that base path. */
        fun open(
            path: String,
            env: OrtEnvironment,
            backend: Backend,
            config: BackendConfig = BackendConfig(),
            profilePath: String? = null,
            outcome: (OpenOutcome) -> Unit = {},
        ): OrtTower {
            var addEp = "n/a"
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(config.intraOpThreads)
                setInterOpNumThreads(config.interOpThreads)
                setOptimizationLevel(config.graphOpt)
                setMemoryPatternOptimization(config.memoryPattern)
                if (profilePath != null) enableProfiling(profilePath)
                try {
                    when (backend) {
                        Backend.CPU_XNNPACK -> {
                            addXnnpack(mapOf("intra_op_num_threads" to config.intraOpThreads.toString())); addEp = "ok"
                        }
                        Backend.CPU_EP -> addEp = "ok"  // default ORT CPU EP, no extra EP
                        Backend.NNAPI_DEFAULT -> { addNnapi(EnumSet.noneOf(NNAPIFlags::class.java)); addEp = "ok" }
                        Backend.NNAPI_CPU_DISABLED -> { addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED)); addEp = "ok" }
                    }
                } catch (t: Throwable) { addEp = "threw: ${t.message}" }
            }
            return try {
                val s = env.createSession(path, opts)
                outcome(OpenOutcome(backend, addEp, "ok"))
                OrtTower(s, env, backend)
            } catch (t: Throwable) {
                outcome(OpenOutcome(backend, addEp, "threw: ${t.message}"))
                throw t
            }
        }
    }

    private fun inputName() = session.inputNames.first()

    /** Run a [rows, *] tensor; return L2-normalized POOLED embeddings [rows][dim] (by-name pooler_output). */
    private fun runEmbed(buf: Buffer, shape: LongArray, rows: Int): Array<FloatArray> {
        val tensor = when (buf) {
            is FloatBuffer -> OnnxTensor.createTensor(env, buf, shape)
            is LongBuffer -> OnnxTensor.createTensor(env, buf, shape)
            else -> error("buffer type")
        }
        tensor.use { t ->
            session.run(mapOf(inputName() to t)).use { res ->
                val pooled = res.get("pooler_output").orElseThrow {
                    IllegalStateException("no pooler_output; outputs=${session.outputNames}")
                } as OnnxTensor
                @Suppress("UNCHECKED_CAST")
                val rowsData = pooled.value as Array<FloatArray>
                check(rowsData.size == rows) { "expected $rows rows, got ${rowsData.size}; shape=${pooled.info.shape.toList()}" }
                return Array(rows) { r ->
                    val v = rowsData[r].copyOf()
                    var n = 0f; for (x in v) n += x * x; n = sqrt(n)
                    if (n > 0f) for (i in v.indices) v[i] = v[i] / n
                    v
                }
            }
        }
    }

    /** Encode [frames] in fixed chunks of [batch] (use 1 for XNNPACK; the D5 map for CPU_EP).
     *  pixelValues is the full [frames*3*res*res] buffer. Stacks rows in frame order. */
    fun encodeVision(pixelValues: FloatBuffer, frames: Int, res: Int, batch: Int): Array<FloatArray> {
        (pixelValues as Buffer).rewind()
        val per = 3 * res * res
        val out = ArrayList<FloatArray>(frames)
        var f = 0
        while (f < frames) {
            val n = minOf(batch, frames - f)
            val chunk = ByteBuffer.allocateDirect(n * per * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            val base = f * per
            for (i in 0 until n * per) chunk.put(pixelValues.get(base + i))
            (chunk as Buffer).rewind()
            for (row in runEmbed(chunk, longArrayOf(n.toLong(), 3, res.toLong(), res.toLong()), n)) out.add(row)
            f += n
        }
        return out.toTypedArray()
    }

    /** Encode [labels] padded ids. Batched if [batch] > 1 (CPU_EP); per-label if 1 (XNNPACK). */
    fun encodeText(inputIds: LongArray, labels: Int, maxLen: Int, batch: Int): Array<FloatArray> {
        val out = ArrayList<FloatArray>(labels)
        var l = 0
        while (l < labels) {
            val n = minOf(batch, labels - l)
            val buf = ByteBuffer.allocateDirect(n * maxLen * 8).order(ByteOrder.nativeOrder()).asLongBuffer()
            for (i in 0 until n * maxLen) buf.put(inputIds[l * maxLen + i])
            (buf as Buffer).rewind()
            for (row in runEmbed(buf, longArrayOf(n.toLong(), maxLen.toLong()), n)) out.add(row)
            l += n
        }
        return out.toTypedArray()
    }

    /** Run one frame with profiling on, end profiling, return the profile-file path (vision capability probe). */
    fun runOnceForProfile(pixelValues: FloatBuffer, res: Int): String {
        encodeVision(pixelValues, 1, res, 1)
        return session.endProfiling()
    }

    /** Run one label with profiling on (text capability probe). */
    fun runOnceForProfileText(inputIds: LongArray, maxLen: Int): String {
        encodeText(inputIds, 1, maxLen, 1)
        return session.endProfiling()
    }

    override fun close() = session.close()
}
```

NOTE: `encodeText` now takes a flat `LongArray` of `[labels*maxLen]` (callers pack it). This replaces the old `LongBuffer` signature — Task 2 updates `Engine` to match.

- [ ] **Step 3: Write `OrtBackendTest.kt` (device)**

```kotlin
package com.example.clipcc.engine

import ai.onnxruntime.OrtEnvironment
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs

@RunWith(AndroidJUnit4::class)
class OrtBackendTest {
    private val dir = "/data/local/tmp/clipcc_models/siglip2-base-patch16-256"
    private val res = 256
    private val dim = 768

    private fun twoFrames(): FloatBuffer {
        val ctx = InstrumentationRegistry.getInstrumentation().context
        val per = 3 * res * res
        val buf = ByteBuffer.allocateDirect(2 * per * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        for (name in listOf("frame_000.png", "frame_001.png")) {
            val bmp = ctx.assets.open("fixtures/$name").use { BitmapFactory.decodeStream(it) }
            val one = Preprocess.toCHW(bmp, res)
            for (i in 0 until per) buf.put(one.get(i))
        }
        (buf as Buffer).rewind(); return buf
    }

    @Test fun cpu_lanes_open_and_batched_equals_perframe() {
        val env = OrtEnvironment.getEnvironment()
        // XNNPACK per-frame (batch=1)
        val xnn = OrtTower.open("$dir/vision_model.onnx", env, Backend.CPU_XNNPACK).use {
            it.encodeVision(twoFrames(), 2, res, 1)
        }
        // CPU EP batched (batch=2 in one session.run)
        val cpu = OrtTower.open("$dir/vision_model.onnx", env, Backend.CPU_EP).use {
            it.encodeVision(twoFrames(), 2, res, 2)
        }
        assertEquals(2, xnn.size); assertEquals(dim, xnn[0].size)
        assertEquals(2, cpu.size); assertEquals(dim, cpu[0].size)
        var maxAbs = 0f
        for (f in 0 until 2) for (i in 0 until dim) maxAbs = maxOf(maxAbs, abs(xnn[f][i] - cpu[f][i]))
        println("BACKEND xnnpack_vs_cpuep max_abs=$maxAbs")
        assertTrue("XNNPACK per-frame vs CPU_EP batched agree ($maxAbs)", maxAbs <= 1e-2f)
    }

    @Test fun nnapi_attempt_records_outcome_without_crashing() {
        val env = OrtEnvironment.getEnvironment()
        var got: OpenOutcome? = null
        try {
            OrtTower.open("$dir/vision_model.onnx", env, Backend.NNAPI_DEFAULT, outcome = { got = it }).use {
                it.encodeVision(twoFrames(), 1, res, 1)  // runs on whatever applied (likely CPU fallback)
            }
        } catch (t: Throwable) {
            // session-create failure is an acceptable, recorded outcome
            assertTrue("outcome captured on throw", got != null)
        }
        println("BACKEND nnapi_default outcome=$got")
        assertTrue("an outcome was recorded", got != null)
    }

    @Test fun text_lane_shapes_and_consistency() {
        val env = OrtEnvironment.getEnvironment()
        val ids = HfTokenizer.fromJson(java.io.File("$dir/tokenizer.json").readBytes()).use {
            it.encodePadded("a photo of a car")
        }
        val xnn = OrtTower.open("$dir/text_model.onnx", env, Backend.CPU_XNNPACK).use {
            it.encodeText(ids, 1, HfTokenizer.MAX_LEN, 1)
        }
        val cpu = OrtTower.open("$dir/text_model.onnx", env, Backend.CPU_EP).use {
            it.encodeText(ids, 1, HfTokenizer.MAX_LEN, 1)
        }
        assertEquals(1, xnn.size); assertEquals(dim, xnn[0].size)
        var m = 0f; for (i in 0 until dim) m = maxOf(m, abs(xnn[0][i] - cpu[0][i]))
        println("BACKEND text_xnn_vs_cpu max_abs=$m")
        assertTrue("text lanes agree ($m)", m <= 1e-2f)
    }
}
```

- [ ] **Step 2b: Remove the superseded Plan-1 `OrtTowerTest.kt`**

The Task-1 `OrtTower` refactor removes the old 2-arg `open` / 3-arg `encodeVision` / `LongBuffer encodeText` signatures, so the Plan-1 `OrtTowerTest.kt` will no longer compile. Its coverage (per-row mapping, distinct frames, text shape) is now in `OrtBackendTest`. Delete it:
```bash
rm /Users/austin/AndroidStudioProjects/ClipCC/app/src/androidTest/java/com/example/clipcc/engine/OrtTowerTest.kt
```

- [ ] **Step 4: Run; GATE**

Run:
```bash
ADB="$HOME/Library/Android/sdk/platform-tools/adb"; $ADB logcat -c
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
cd /Users/austin/AndroidStudioProjects/ClipCC
./gradlew :app:connectedDebugAndroidTest --console=plain \
  -Pandroid.testInstrumentationRunnerArguments.class=com.example.clipcc.engine.OrtBackendTest 2>&1 | tail -15
$ADB logcat -d | grep -i "BACKEND " | tail -5
```
Expected: BUILD SUCCESSFUL; `xnnpack_vs_cpuep max_abs` ≤ 1e-2 (proves both lanes correct + consistent); NNAPI outcome recorded (applied EP or recorded throw), no crash. If `cpu` batched throws the `check(rows)` (got 1, expected 2), CPU_EP is NOT batching on this device — STOP and report (it contradicts the host finding; do not silently fall back to per-frame).

- [ ] **Step 5: Save.**

---

## Task 2: Backend-parameterized `Engine`

**Files:**
- Modify: `app/src/main/java/com/example/clipcc/engine/Engine.kt`
- Test: reuse Plan-1 `EndToEndParityTest.kt` (parity must still hold on CPU_EP) + add a CPU_EP case.

- [ ] **Step 1: Replace `Engine.kt` with the backend-parameterized version**

The current `Engine` hardcodes the XNNPACK per-label workaround. Parameterize it by `Backend` + per-model `visionBatch`, and route through the new `OrtTower` API:

```kotlin
package com.example.clipcc.engine

import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap
import java.io.File
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Headless engine: (frames + labels) -> [F x L] cosine/confidence matrices on a chosen [backend].
 * Memory order (Spike 0d): encode text, RELEASE the text session, THEN open vision.
 * Under XNNPACK both towers run per-item (batch=1, batch-collapse); under CPU_EP they batch.
 */
class Engine(
    private val modelDir: String,
    private val manifest: ModelBundleManifest,
    private val env: OrtEnvironment,
    private val backend: Backend = Backend.CPU_XNNPACK,
    private val visionBatch: Int = 1,
    private val config: BackendConfig = BackendConfig(),
) {
    private fun itemBatch() = if (backend == Backend.CPU_EP) Int.MAX_VALUE else 1

    fun scoreFrames(bitmaps: List<Bitmap>, labels: List<String>): ScoreMatrices {
        val effVisionBatch = if (backend == Backend.CPU_EP) visionBatch else 1
        val txt: Array<FloatArray> =
            HfTokenizer.fromJson(File("$modelDir/${manifest.tokenizerFile}").readBytes()).use { tk ->
                val ids = labels.map { tk.encodePadded(it) }
                OrtTower.open("$modelDir/${manifest.textFile}", env, backend, config).use { t ->
                    t.encodeText(flatten(ids), labels.size, manifest.maxLength, minOf(itemBatch(), labels.size))
                }
            }
        return OrtTower.open("$modelDir/${manifest.visionFile}", env, backend, config).use { v ->
            val pix = packFrames(bitmaps, manifest.resolution)
            val img = v.encodeVision(pix, bitmaps.size, manifest.resolution, effVisionBatch)
            Scoring.scoreMatrix(img, txt, manifest.logitScale, manifest.logitBias)
        }
    }

    private fun flatten(ids: List<LongArray>): LongArray {
        val maxLen = manifest.maxLength
        val out = LongArray(ids.size * maxLen)
        for (l in ids.indices) { require(ids[l].size == maxLen); System.arraycopy(ids[l], 0, out, l * maxLen, maxLen) }
        return out
    }

    private fun packFrames(bitmaps: List<Bitmap>, res: Int): FloatBuffer {
        val per = 3 * res * res
        val buf = ByteBuffer.allocateDirect(bitmaps.size * per * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        for (bmp in bitmaps) { val one = Preprocess.toCHW(bmp, res); for (i in 0 until per) buf.put(one.get(i)) }
        (buf as Buffer).rewind(); return buf
    }
}
```
NOTE: `itemBatch()` returns `Int.MAX_VALUE` for CPU_EP so `encodeText`/`encodeVision`'s `minOf(batch, remaining)` batches the whole set in one run; for XNNPACK it's 1 (per-item). `visionBatch` (the D5 map) only applies on CPU_EP.

- [ ] **Step 2: Add a CPU_EP case to `EndToEndParityTest.kt`**

Append this test method to the existing `app/src/androidTest/java/com/example/clipcc/engine/EndToEndParityTest.kt` class (keep the existing XNNPACK-default test):

```kotlin
    @Test fun scores_match_golden_on_cpu_ep() {
        val ctx = InstrumentationRegistry.getInstrumentation().context
        val g = JSONObject(ctx.assets.open("fixtures/scores_golden.json").bufferedReader().readText())
        val labels = g.getJSONArray("labels").let { la -> List(la.length()) { la.getString(it) } }
        val frameNames = g.getJSONArray("frames").let { fa -> List(fa.length()) { fa.getString(it) } }
        val gCos = mat(g.getJSONArray("cosine"))
        val bitmaps = frameNames.map { n -> ctx.assets.open("fixtures/$n").use { BitmapFactory.decodeStream(it) } }
        val manifest = ModelBundleManifest.parse(File("$dir/manifest.json").readText())
        val sm = Engine(dir, manifest, OrtEnvironment.getEnvironment(),
            backend = Backend.CPU_EP, visionBatch = 16).scoreFrames(bitmaps, labels)
        var cosMax = 0f
        for (f in gCos.indices) for (l in labels.indices) cosMax = maxOf(cosMax, kotlin.math.abs(sm.cosine[f][l] - gCos[f][l]))
        println("E2E_CPU_EP cosine_max=$cosMax")
        assertTrue("cpu_ep cosine max $cosMax <= 0.01", cosMax <= 0.01f)
    }
```
(The existing test already imports `mat`, `JSONObject`, `BitmapFactory`, `dir`, etc.)

- [ ] **Step 3: Run; GATE — parity preserved on both lanes**

Run:
```bash
ADB="$HOME/Library/Android/sdk/platform-tools/adb"; $ADB logcat -c
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
cd /Users/austin/AndroidStudioProjects/ClipCC
./gradlew :app:connectedDebugAndroidTest --console=plain \
  -Pandroid.testInstrumentationRunnerArguments.class=com.example.clipcc.engine.EndToEndParityTest 2>&1 | tail -15
$ADB logcat -d | grep -iE "E2E |E2E_CPU_EP" | tail -5
```
Expected: both tests pass; XNNPACK `cosine_max` (Plan-1 ~9e-5) and `E2E_CPU_EP cosine_max` both ≤ 0.01. The refactor did not change scores.

- [ ] **Step 4: Save.**

---

## Task 3: `BackendCapabilityReport` — untimed provider-coverage probe

**Files:**
- Create: `app/src/main/java/com/example/clipcc/engine/BackendCapability.kt`
- Test: `app/src/androidTest/java/com/example/clipcc/engine/BackendCapabilityTest.kt`

- [ ] **Step 1: Create `BackendCapability.kt`**

```kotlin
package com.example.clipcc.engine

import ai.onnxruntime.OrtEnvironment
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Locale

/** Per-(model,tower,backend) capability evidence from an UNTIMED, profiling-ON one-frame run. */
data class BackendCapabilityReport(
    val modelId: String,
    val tower: String,          // "vision" | "text"
    val backend: Backend,
    val addEpOutcome: String,
    val sessionCreateOutcome: String,
    val providerCounts: Map<String, Int>,  // provider -> node count (from profiling JSON)
    val totalNodes: Int,
    val delegatedPctByProvider: Map<String, Double>,
) {
    fun toJson(): String {
        val pc = providerCounts.entries.joinToString(",") { "\"${it.key}\":${it.value}" }
        val pct = delegatedPctByProvider.entries.joinToString(",") { "\"${it.key}\":${String.format(Locale.US, "%.2f", it.value)}" }
        // JSONObject.quote() adds surrounding quotes + escapes — addEp/sessionCreate may hold raw exception text.
        return """{"model":"$modelId","tower":"$tower","backend":"$backend","addEp":${JSONObject.quote(addEpOutcome)},""" +
               """"sessionCreate":${JSONObject.quote(sessionCreateOutcome)},"totalNodes":$totalNodes,""" +
               """"providerCounts":{$pc},"delegatedPct":{$pct}}"""
    }
}

object BackendCapability {
    /** Parse ORT profiling JSON: count Node events by their args.provider. (Spike 0b format.) */
    fun parseProviderCounts(profileJsonPath: String): Map<String, Int> {
        val text = File(profileJsonPath).readText()
        val arr = JSONArray(text)
        val counts = HashMap<String, Int>()
        for (i in 0 until arr.length()) {
            val ev = arr.optJSONObject(i) ?: continue
            if (ev.optString("cat") != "Node") continue
            val args = ev.optJSONObject("args") ?: continue
            val provider = args.optString("provider", "")
            if (provider.isEmpty()) continue
            counts[provider] = (counts[provider] ?: 0) + 1
        }
        return counts
    }

    /** Probe vision tower of [modelDir] under [backend]; one untimed frame; profiling -> coverage. */
    fun probeVision(
        modelDir: String, modelId: String, res: Int, backend: Backend,
        env: OrtEnvironment, cacheDir: File, dummy: FloatBuffer,
    ): BackendCapabilityReport {
        var outcome = OpenOutcome(backend, "n/a", "not-attempted")
        val profileBase = File(cacheDir, "prof_${modelId}_vision_$backend").absolutePath
        var counts: Map<String, Int> = emptyMap()
        try {
            OrtTower.open("$modelDir/vision_model.onnx", env, backend,
                profilePath = profileBase, outcome = { outcome = it }).use { tower ->
                val path = tower.runOnceForProfile(dummy, res)
                counts = parseProviderCounts(path)
            }
        } catch (t: Throwable) {
            // outcome already records the throw; counts stays empty
        }
        val total = counts.values.sum()
        val pct = if (total > 0) counts.mapValues { 100.0 * it.value / total } else emptyMap()
        return BackendCapabilityReport(modelId, "vision", backend, outcome.addEpOutcome,
            outcome.sessionCreateOutcome, counts, total, pct)
    }

    /** Probe text tower of [modelDir] under [backend]; one untimed label; profiling -> coverage. */
    fun probeText(
        modelDir: String, modelId: String, maxLen: Int, backend: Backend,
        env: OrtEnvironment, cacheDir: File,
    ): BackendCapabilityReport {
        var outcome = OpenOutcome(backend, "n/a", "not-attempted")
        val profileBase = File(cacheDir, "prof_${modelId}_text_$backend").absolutePath
        var counts: Map<String, Int> = emptyMap()
        try {
            OrtTower.open("$modelDir/text_model.onnx", env, backend,
                profilePath = profileBase, outcome = { outcome = it }).use { tower ->
                counts = parseProviderCounts(tower.runOnceForProfileText(LongArray(maxLen) { 0L }, maxLen))
            }
        } catch (t: Throwable) { /* outcome records the throw */ }
        val total = counts.values.sum()
        val pct = if (total > 0) counts.mapValues { 100.0 * it.value / total } else emptyMap()
        return BackendCapabilityReport(modelId, "text", backend, outcome.addEpOutcome,
            outcome.sessionCreateOutcome, counts, total, pct)
    }

    fun dummyFrame(res: Int): FloatBuffer {
        val per = 3 * res * res
        val buf = ByteBuffer.allocateDirect(per * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        for (i in 0 until per) buf.put(0f)
        (buf as Buffer).rewind(); return buf
    }
}
```

- [ ] **Step 2: Write `BackendCapabilityTest.kt` (device, base-256 only — fast)**

```kotlin
package com.example.clipcc.engine

import ai.onnxruntime.OrtEnvironment
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class BackendCapabilityTest {
    private val dir = "/data/local/tmp/clipcc_models/siglip2-base-patch16-256"

    @Test fun probe_emits_provider_coverage_for_each_backend() {
        val env = OrtEnvironment.getEnvironment()
        val cache = InstrumentationRegistry.getInstrumentation().targetContext.cacheDir
        val dummy = BackendCapability.dummyFrame(256)
        for (b in listOf(Backend.CPU_XNNPACK, Backend.CPU_EP, Backend.NNAPI_DEFAULT, Backend.NNAPI_CPU_DISABLED)) {
            val r = BackendCapability.probeVision(dir, "siglip2-base-patch16-256", 256, b, env, cache,
                BackendCapability.dummyFrame(256))
            println("CAP ${r.toJson()}")
        }
        // CPU_XNNPACK must produce a real coverage histogram with >0 total nodes.
        val xnn = BackendCapability.probeVision(dir, "siglip2-base-patch16-256", 256, Backend.CPU_XNNPACK, env, cache, dummy)
        println("CAP_ASSERT xnn_total=${xnn.totalNodes} providers=${xnn.providerCounts.keys}")
        assertTrue("XNNPACK probe found nodes", xnn.totalNodes > 0)
    }
}
```

- [ ] **Step 3: Run; GATE**

Run:
```bash
ADB="$HOME/Library/Android/sdk/platform-tools/adb"; $ADB logcat -c
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
cd /Users/austin/AndroidStudioProjects/ClipCC
./gradlew :app:connectedDebugAndroidTest --console=plain \
  -Pandroid.testInstrumentationRunnerArguments.class=com.example.clipcc.engine.BackendCapabilityTest 2>&1 | tail -15
$ADB logcat -d | grep -iE "CAP |CAP_ASSERT" | tail -10
```
Expected: PASS; four `CAP {...}` JSON lines printed; XNNPACK total nodes > 0 with a `providerCounts` map (expect mostly `CPUExecutionProvider` + some `XnnpackExecutionProvider`, matching Spike 0b ~12%). If the profiling JSON has no `args.provider` keys, inspect one profile file at `cacheDir/prof_*` and adjust `parseProviderCounts` to the actual key (e.g. `"provider"` may be nested differently per ORT version) — do NOT hardcode coverage numbers.

- [ ] **Step 4: Save.**

---

## Task 4: `FrameSampler` (Media3 real-video decode)

**Files:**
- Modify: `gradle/libs.versions.toml`, `app/build.gradle.kts` (add Media3 dep)
- Create: `app/src/main/java/com/example/clipcc/engine/FrameSampler.kt`
- Test: `app/src/androidTest/java/com/example/clipcc/engine/FrameSamplerTest.kt`

> Needs network for the Media3 AAR (not cached). Internet confirmed available 2026-06-03.

- [ ] **Step 1: Add the Media3 dependency**

In `gradle/libs.versions.toml`, under `[versions]` add:
```toml
media3 = "1.10.1"
```
and under `[libraries]` add:
```toml
media3-inspector-frame = { group = "androidx.media3", name = "media3-inspector-frame", version.ref = "media3" }
```
In `app/build.gradle.kts`, in `dependencies { }` after the onnxruntime line add:
```kotlin
    implementation(libs.media3.inspector.frame)
```

- [ ] **Step 2: Verify the artifact resolves**

Run:
```bash
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
cd /Users/austin/AndroidStudioProjects/ClipCC
./gradlew :app:dependencies --configuration debugRuntimeClasspath 2>&1 | grep -i "media3-inspector-frame"
```
Expected: `androidx.media3:media3-inspector-frame:1.10.1` resolved. If resolution fails, you are offline — STOP and report (this task needs network).

- [ ] **Step 3: Create `FrameSampler.kt`**

`FrameExtractor` is `@UnstableApi` and must be accessed from one thread; it exposes a per-position frame getter returning a `Bitmap`. The exact method/return-type names can vary across Media3 versions — verify against 1.10.1 sources/Javadoc and adjust the marked lines if needed; the contract is "get a `Bitmap` at a timestamp, single-threaded."

```kotlin
package com.example.clipcc.engine

import android.content.Context
import android.graphics.Bitmap
import androidx.media3.common.MediaItem
import androidx.media3.common.util.UnstableApi
import androidx.media3.inspector.frame.FrameExtractor

data class SampledFrame(val bitmap: Bitmap, val timestampSec: Double, val index: Int)
data class VideoMeta(val width: Int, val height: Int, val rotationDegrees: Int, val frameCount: Int)

@UnstableApi
class FrameSampler(private val context: Context) {
    /** Sample [maxFrames] frames at [fps] from [videoPath]; rotation already applied by the extractor.
     *  Single-threaded: call from one worker thread (the benchmark runner provides it). */
    fun sample(videoPath: String, fps: Double, maxFrames: Int): Pair<VideoMeta, List<SampledFrame>> {
        val item = MediaItem.fromUri(android.net.Uri.fromFile(java.io.File(videoPath)))
        // VERIFY API (1.10.1): builder + per-position getFrame returning a Frame with a Bitmap + pts.
        val extractor = FrameExtractor.Builder(context, item).build()
        try {
            val frames = ArrayList<SampledFrame>(maxFrames)
            var w = 0; var h = 0; var rot = 0
            val stepMs = (1000.0 / fps).toLong()
            var i = 0
            while (i < maxFrames) {
                val posMs = i * stepMs
                // VERIFY API: getFrame(posMs) returns a ListenableFuture<Frame>; Frame has .bitmap + .presentationTimeMs
                val frame = extractor.getFrame(posMs).get()   // blocks; single-thread caller
                    ?: break
                val bmp = frame.bitmap
                if (i == 0) { w = bmp.width; h = bmp.height }
                frames.add(SampledFrame(bmp, posMs / 1000.0, i))
                i++
            }
            return VideoMeta(w, h, rot, frames.size) to frames
        } finally {
            extractor.release()  // VERIFY API name
        }
    }
}
```

- [ ] **Step 4: Write `FrameSamplerTest.kt` (device)**

```kotlin
package com.example.clipcc.engine

import androidx.media3.common.util.UnstableApi
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@UnstableApi
@RunWith(AndroidJUnit4::class)
class FrameSamplerTest {
    @Test fun decodes_expected_frames_with_dims_and_rotation() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val (meta, frames) = FrameSampler(ctx).sample("/data/local/tmp/clipcc_bench/test.mp4", fps = 1.0, maxFrames = 8)
        println("FRAMES count=${frames.size} dims=${meta.width}x${meta.height} rot=${meta.rotationDegrees}")
        assertTrue("got frames", frames.isNotEmpty())
        assertTrue("dims positive", meta.width > 0 && meta.height > 0)
        // timestamps increase by ~1s
        for (k in 1 until frames.size) assertTrue("ts increasing", frames[k].timestampSec > frames[k-1].timestampSec)
        assertEquals("indices sequential", frames.indices.toList(), frames.map { it.index })
    }
}
```

- [ ] **Step 5: Run; GATE**

Run:
```bash
ADB="$HOME/Library/Android/sdk/platform-tools/adb"; $ADB logcat -c
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
cd /Users/austin/AndroidStudioProjects/ClipCC
./gradlew :app:connectedDebugAndroidTest --console=plain \
  -Pandroid.testInstrumentationRunnerArguments.class=com.example.clipcc.engine.FrameSamplerTest 2>&1 | tail -20
$ADB logcat -d | grep -i "FRAMES " | tail -3
```
Expected: PASS; `FRAMES count=8 dims=WxH rot=R`; timestamps increasing. If the FrameExtractor API differs from the skeleton, fix `FrameSampler.kt` per the 1.10.1 Javadoc (the test contract is unchanged) and re-run.

- [ ] **Step 6: Save.**

---

## Task 5: `Benchmark` runner — FramePrep once + TimedRun per backend

**Files:**
- Create: `app/src/main/java/com/example/clipcc/engine/Benchmark.kt`
- Test: `app/src/androidTest/java/com/example/clipcc/engine/BenchmarkSmokeTest.kt`

- [ ] **Step 1: Create `Benchmark.kt`**

```kotlin
package com.example.clipcc.engine

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.graphics.Bitmap
import android.os.BatteryManager
import android.os.PowerManager
import androidx.media3.common.util.UnstableApi
import java.io.File

data class FramePrepResult(val modelId: String, val res: Int, val frames: Int, val decodeMs: Long, val preprocessMs: Long)

data class RunMetadata(
    val thermalStatus: Int, val thermalThrottled: Boolean, val batteryPct: Int,
    val charging: Boolean, val runOrder: Int, val wallClockMs: Long, val media3Version: String,
)

data class TimedRun(
    val modelId: String, val backend: Backend, val effectiveBatch: Int,
    val loadMs: Long, val textMs: Long, val visionMsMedian: Long, val visionMsMin: Long,
    val visionMsMax: Long, val scoringMs: Long, val msPerFrame: Double, val fps: Double,
    val endToEndMsSynthetic: Long, val config: BackendConfig, val meta: RunMetadata,
)

@UnstableApi
class Benchmark(private val context: Context, private val env: OrtEnvironment) {
    /** D5 batch map (CPU_EP only). */
    fun visionBatchFor(modelId: String): Int = when {
        modelId.contains("so400m") -> 4
        modelId.contains("large") -> 8
        else -> 16  // base-256 / base-384
    }
    fun framesFor(modelId: String): Int = if (modelId.contains("so400m")) 4 else 16

    private fun now() = System.nanoTime()
    private fun msSince(t0: Long) = (now() - t0) / 1_000_000L

    private fun meta(order: Int): RunMetadata {
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val thermal = pm.currentThermalStatus
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val pct = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        val charging = bm.isCharging
        return RunMetadata(thermal, thermal >= PowerManager.THERMAL_STATUS_MODERATE, pct, charging,
            order, System.currentTimeMillis(), "1.10.1")
    }

    /** Decode + preprocess ONCE per model (timed); returns cached CHW tensors + prep timing. */
    fun prepFrames(modelDir: String, manifest: ModelBundleManifest, videoPath: String):
            Triple<FramePrepResult, List<Bitmap>, FloatArrayHolder> {
        val n = framesFor(manifest.modelId)
        val tDecode = now()
        val (_, sampled) = FrameSampler(context).sample(videoPath, 1.0, n)
        val decodeMs = msSince(tDecode)
        val bitmaps = sampled.map { it.bitmap }
        val tPre = now()
        val per = 3 * manifest.resolution * manifest.resolution
        val flat = FloatArray(bitmaps.size * per)
        for ((i, bmp) in bitmaps.withIndex()) {
            val one = Preprocess.toCHW(bmp, manifest.resolution)
            for (j in 0 until per) flat[i * per + j] = one.get(j)
        }
        val preMs = msSince(tPre)
        return Triple(FramePrepResult(manifest.modelId, manifest.resolution, bitmaps.size, decodeMs, preMs),
            bitmaps, FloatArrayHolder(flat))
    }

    /** Time one CPU lane: warm-up (discard) + median-of-3 vision; load/text/scoring once. */
    fun timeLane(modelDir: String, manifest: ModelBundleManifest, backend: Backend,
                 prep: FloatArrayHolder, frames: Int, labels: List<String>, runOrder: Int): TimedRun {
        require(backend == Backend.CPU_XNNPACK || backend == Backend.CPU_EP) { "only CPU lanes are timed" }
        val res = manifest.resolution
        val visionBatch = if (backend == Backend.CPU_EP) visionBatchFor(manifest.modelId) else 1
        val per = 3 * res * res
        val cfg = BackendConfig()

        // text (once)
        val tLoadT = now()
        val txt = HfTokenizer.fromJson(File("$modelDir/${manifest.tokenizerFile}").readBytes()).use { tk ->
            val ids = labels.map { tk.encodePadded(it) }
            val flat = LongArray(ids.size * manifest.maxLength)
            for (l in ids.indices) System.arraycopy(ids[l], 0, flat, l * manifest.maxLength, manifest.maxLength)
            OrtTower.open("$modelDir/${manifest.textFile}", env, backend, cfg).use { t ->
                val textBatch = if (backend == Backend.CPU_EP) labels.size else 1
                t.encodeText(flat, labels.size, manifest.maxLength, textBatch)
            }
        }
        val textMs = msSince(tLoadT)

        // vision: load, warm-up, median-of-3 (one session reused)
        val tLoadV = now()
        OrtTower.open("$modelDir/${manifest.visionFile}", env, backend, cfg).use { v ->
            val loadMs = msSince(tLoadV)
            // Build the pixel buffer ONCE (outside the timed region) and reuse it: encodeVision rewinds
            // its input and reads via absolute get(), so reuse is safe. Keeps the alloc+fill out of vision_ms.
            val pix = run {
                val b = java.nio.ByteBuffer.allocateDirect(frames * per * 4)
                    .order(java.nio.ByteOrder.nativeOrder()).asFloatBuffer()
                b.put(prep.data); (b as java.nio.Buffer).rewind(); b
            }
            v.encodeVision(pix, frames, res, visionBatch)  // warm-up (discarded)
            val times = LongArray(3)
            lateinit var img: Array<FloatArray>
            for (r in 0 until 3) {
                val t0 = now()
                img = v.encodeVision(pix, frames, res, visionBatch)
                times[r] = msSince(t0)
                Thread.sleep(1500)  // cool-down between timed runs
            }
            times.sort()
            val median = times[1]; val mn = times[0]; val mx = times[2]
            val tScore = now()
            Scoring.scoreMatrix(img, txt, manifest.logitScale, manifest.logitBias)
            val scoringMs = msSince(tScore)
            val msPerFrame = median.toDouble() / frames
            val fps = if (median > 0) frames * 1000.0 / median else 0.0
            return TimedRun(manifest.modelId, backend, visionBatch, loadMs, textMs, median, mn, mx,
                scoringMs, msPerFrame, fps, prep.decodeMs + prep.preMs + textMs + median + scoringMs,
                cfg, meta(runOrder))
        }
    }
}

/** Holder so prep timing travels with the flat tensor. */
class FloatArrayHolder(val data: FloatArray) { var decodeMs: Long = 0; var preMs: Long = 0 }
```
NOTE: set `holder.decodeMs`/`holder.preMs` from the `FramePrepResult` before calling `timeLane` (the runner in Task 6 wires this). Cool-down = 1.5 s; warm-up discarded; median-of-3.

- [ ] **Step 2: Write `BenchmarkSmokeTest.kt` (fast: base-256, CPU_EP, 2 frames)**

```kotlin
package com.example.clipcc.engine

import ai.onnxruntime.OrtEnvironment
import androidx.media3.common.util.UnstableApi
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@UnstableApi
@RunWith(AndroidJUnit4::class)
class BenchmarkSmokeTest {
    @Test fun base256_cpuep_two_frames_smoke() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val env = OrtEnvironment.getEnvironment()
        val dir = "/data/local/tmp/clipcc_models/siglip2-base-patch16-256"
        val manifest = ModelBundleManifest.parse(File("$dir/manifest.json").readText())
        val bench = Benchmark(ctx, env)
        val (prep, bitmaps, holder) = bench.prepFrames(dir, manifest, "/data/local/tmp/clipcc_bench/test.mp4")
        holder.decodeMs = prep.decodeMs; holder.preMs = prep.preprocessMs
        val run = bench.timeLane(dir, manifest, Backend.CPU_EP, holder, prep.frames,
            listOf("Car", "a dog"), runOrder = 0)
        println("SMOKE ${run.modelId} vision_ms=${run.visionMsMedian} fps=${run.fps} batch=${run.effectiveBatch} thermal=${run.meta.thermalStatus}")
        assertTrue("vision timed", run.visionMsMedian >= 0)
        assertTrue("decode timed", prep.decodeMs >= 0)
    }
}
```

- [ ] **Step 3: Run; GATE**

Run:
```bash
ADB="$HOME/Library/Android/sdk/platform-tools/adb"; $ADB logcat -c
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
cd /Users/austin/AndroidStudioProjects/ClipCC
./gradlew :app:connectedDebugAndroidTest --console=plain \
  -Pandroid.testInstrumentationRunnerArguments.class=com.example.clipcc.engine.BenchmarkSmokeTest 2>&1 | tail -15
$ADB logcat -d | grep -i "SMOKE " | tail -3
```
Expected: PASS; `SMOKE siglip2-base-patch16-256 vision_ms=<n> fps=<n> batch=16 thermal=<n>`. Confirms prep+timing+metadata wire together.

- [ ] **Step 4: Save.**

---

## Task 6: Result serialization + off-device retrieval

**Files:**
- Modify: `app/src/main/java/com/example/clipcc/engine/Benchmark.kt` (add JSON writer + Bundle emit helper)
- Test: extend `BenchmarkSmokeTest.kt` to write + read back the JSON

- [ ] **Step 1: Add a results writer to `Benchmark.kt`**

Append to the `Benchmark` class:
```kotlin
    /** Write all results as JSON to the app external files dir; return the absolute path. */
    fun writeResults(prep: List<FramePrepResult>, runs: List<TimedRun>,
                     caps: List<BackendCapabilityReport>): String {
        fun run(r: TimedRun) = """{"model":"${r.modelId}","backend":"${r.backend}","batch":${r.effectiveBatch},""" +
            """"loadMs":${r.loadMs},"textMs":${r.textMs},"visionMsMedian":${r.visionMsMedian},""" +
            """"visionMsMin":${r.visionMsMin},"visionMsMax":${r.visionMsMax},"scoringMs":${r.scoringMs},""" +
            """"msPerFrame":${String.format(java.util.Locale.US, "%.3f", r.msPerFrame)},"fps":${String.format(java.util.Locale.US, "%.3f", r.fps)},""" +
            """"endToEndMsSynthetic":${r.endToEndMsSynthetic},"intraOpThreads":${r.config.intraOpThreads},""" +
            """"thermal":${r.meta.thermalStatus},"thermalThrottled":${r.meta.thermalThrottled},""" +
            """"batteryPct":${r.meta.batteryPct},"runOrder":${r.meta.runOrder},"media3":"${r.meta.media3Version}"}"""
        fun prep(p: FramePrepResult) = """{"model":"${p.modelId}","res":${p.res},"frames":${p.frames},""" +
            """"decodeMs":${p.decodeMs},"preprocessMs":${p.preprocessMs}}"""
        val json = """{"prep":[${prep.joinToString(",") { prep(it) }}],""" +
            """"runs":[${runs.joinToString(",") { run(it) }}],""" +
            """"capabilities":[${caps.joinToString(",") { it.toJson() }}]}"""
        val out = File(context.getExternalFilesDir(null), "benchmark_result.json")
        out.writeText(json)
        return out.absolutePath
    }
```

- [ ] **Step 2: Extend `BenchmarkSmokeTest` to write + read back + emit the path via the instrumentation Bundle**

Append to `BenchmarkSmokeTest`:
```kotlin
    @Test fun writes_and_reads_result_json_and_emits_path() {
        val inst = InstrumentationRegistry.getInstrumentation()
        val ctx = inst.targetContext
        val env = OrtEnvironment.getEnvironment()
        val dir = "/data/local/tmp/clipcc_models/siglip2-base-patch16-256"
        val manifest = ModelBundleManifest.parse(File("$dir/manifest.json").readText())
        val bench = Benchmark(ctx, env)
        val (prep, _, holder) = bench.prepFrames(dir, manifest, "/data/local/tmp/clipcc_bench/test.mp4")
        holder.decodeMs = prep.decodeMs; holder.preMs = prep.preprocessMs
        val run = bench.timeLane(dir, manifest, Backend.CPU_EP, holder, prep.frames, listOf("Car", "a dog"), 0)
        val path = bench.writeResults(listOf(prep), listOf(run), emptyList())
        // emit the absolute path to the instrumentation runner (survives, pullable)
        val b = android.os.Bundle(); b.putString("benchmark_result_path", path); inst.sendStatus(0, b)
        println("RESULT_PATH $path")
        val back = File(path).readText()
        org.junit.Assert.assertTrue("json has runs", back.contains("\"visionMsMedian\""))
    }
```

- [ ] **Step 3: Run via `am instrument` (no auto-uninstall) + pull the JSON; GATE**

Run:
```bash
ADB="$HOME/Library/Android/sdk/platform-tools/adb"
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
cd /Users/austin/AndroidStudioProjects/ClipCC
# install app + test APKs WITHOUT the auto-uninstall connectedAndroidTest lifecycle:
./gradlew :app:installDebug :app:installDebugAndroidTest 2>&1 | tail -3
PKG=com.example.clipcc
RUNNER=$($ADB shell pm list instrumentation | sed -n 's/^instrumentation:\([^ ]*\) .*/\1/p' | grep "$PKG.test" | head -1)
echo "runner=$RUNNER"
$ADB logcat -c
$ADB shell am instrument -w \
  -e class com.example.clipcc.engine.BenchmarkSmokeTest#writes_and_reads_result_json_and_emits_path \
  "$RUNNER" 2>&1 | tee /tmp/instr_out.txt
# retrieve: prefer the external files dir path; fallback to run-as cacheDir
RP=$(grep -o "benchmark_result_path=[^ ]*" /tmp/instr_out.txt | head -1 | cut -d= -f2)
echo "result_path=$RP"
$ADB pull "$RP" /tmp/benchmark_result.json 2>&1 | tail -1 || \
  $ADB shell run-as $PKG cat files/benchmark_result.json > /tmp/benchmark_result.json
echo "=== pulled JSON ==="; head -c 400 /tmp/benchmark_result.json
```
Expected: instrumentation reports the test passed; `benchmark_result_path=` printed in the status Bundle; the JSON pulled to `/tmp/benchmark_result.json` containing `"runs":[...]` with `visionMsMedian`. If `adb pull` of the external path is denied, the `run-as` fallback retrieves it from the app cacheDir-adjacent `files/` dir.

NOTE on retrieval path: `getExternalFilesDir(null)` maps to `/sdcard/Android/data/<pkg>/files/`; `adb pull` of that path works while the app is installed. The `run-as` fallback reads the app's internal `files/` dir. If you prefer internal-only, change `writeResults` to `File(context.filesDir, ...)` and pull exclusively via `run-as`.

- [ ] **Step 4: Save.**

---

## Task 7: Full benchmark matrix + acceptance gate + report

> ⚠️ **EXECUTED DESIGN DIFFERS (see `phase2-report.md`).** The single-process `BenchmarkMatrixTest`
> below OOM-crashes on so400m (`shortMsg=Process crashed`) after ~30 min of accumulated native ORT
> memory — a native death no try/catch can catch. The shipped test instead benchmarks **one model per
> `am instrument` invocation** (`@Test bench_one_model`, `-e model <id>`, fresh process), writes a
> per-model `benchmark_<id>.json` (via `Benchmark.writeResults(..., fileName)` — a 4-arg overload), and a
> host driver runs it 4× + merges. Results merged to `phase2-benchmark-result.json`; gate MET (prep=4,
> runs=8, capabilities=24). The blocks below are the original (pre-OOM) design, kept for context.

**Files:**
- Create: `app/src/androidTest/java/com/example/clipcc/engine/BenchmarkMatrixTest.kt`
- Create (host repo): `docs/superpowers/plans/phase2-report.md`

- [ ] **Step 1: Create `BenchmarkMatrixTest.kt` (the full, long-running matrix)**

```kotlin
package com.example.clipcc.engine

import ai.onnxruntime.OrtEnvironment
import androidx.media3.common.util.UnstableApi
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@UnstableApi
@RunWith(AndroidJUnit4::class)
class BenchmarkMatrixTest {
    private val models = listOf(
        "siglip2-base-patch16-256", "siglip2-base-patch16-384",
        "siglip2-large-patch16-384", "siglip2-so400m-patch14-384",
    )
    private val labels = listOf("Car", "texting while driving", "a dog")
    private val video = "/data/local/tmp/clipcc_bench/test.mp4"

    @Test fun full_matrix_cpu_lanes_plus_capability() {
        val inst = InstrumentationRegistry.getInstrumentation()
        val ctx = inst.targetContext
        val env = OrtEnvironment.getEnvironment()
        val bench = Benchmark(ctx, env)
        val preps = ArrayList<FramePrepResult>()
        val runs = ArrayList<TimedRun>()
        val caps = ArrayList<BackendCapabilityReport>()
        var order = 0
        for (modelId in models) {
            val dir = "/data/local/tmp/clipcc_models/$modelId"
            val manifest = ModelBundleManifest.parse(File("$dir/manifest.json").readText())
            // capability probe (untimed): vision tower for all 4 backends; text tower for CPU lanes only
            for (b in listOf(Backend.CPU_XNNPACK, Backend.CPU_EP, Backend.NNAPI_DEFAULT, Backend.NNAPI_CPU_DISABLED)) {
                caps.add(BackendCapability.probeVision(dir, modelId, manifest.resolution, b, env, ctx.cacheDir,
                    BackendCapability.dummyFrame(manifest.resolution)))
            }
            for (b in listOf(Backend.CPU_XNNPACK, Backend.CPU_EP)) {
                caps.add(BackendCapability.probeText(dir, modelId, manifest.maxLength, b, env, ctx.cacheDir))
            }
            // prep once, then both timed CPU lanes over the SAME cached tensors
            val (prep, _, holder) = bench.prepFrames(dir, manifest, video)
            holder.decodeMs = prep.decodeMs; holder.preMs = prep.preprocessMs
            preps.add(prep)
            for (b in listOf(Backend.CPU_XNNPACK, Backend.CPU_EP)) {
                runs.add(bench.timeLane(dir, manifest, b, holder, prep.frames, labels, order++))
            }
        }
        val path = bench.writeResults(preps, runs, caps)
        val bundle = android.os.Bundle(); bundle.putString("benchmark_result_path", path); inst.sendStatus(0, bundle)
        println("MATRIX_RESULT_PATH $path")
        // gate: every model has both CPU lanes timed
        for (m in models) {
            assertTrue("$m XNNPACK timed", runs.any { it.modelId == m && it.backend == Backend.CPU_XNNPACK })
            assertTrue("$m CPU_EP timed", runs.any { it.modelId == m && it.backend == Backend.CPU_EP })
        }
        // gate: each model has a VISION capability report for all 4 backends + TEXT for the 2 CPU lanes
        for (m in models) {
            for (b in Backend.values())
                assertTrue("$m vision cap $b", caps.any { it.modelId == m && it.tower == "vision" && it.backend == b })
            for (b in listOf(Backend.CPU_XNNPACK, Backend.CPU_EP))
                assertTrue("$m text cap $b", caps.any { it.modelId == m && it.tower == "text" && it.backend == b })
        }
    }
}
```

- [ ] **Step 2: Run the matrix via `am instrument` + pull JSON; GATE**

Run (expect long — so400m dominates):
```bash
ADB="$HOME/Library/Android/sdk/platform-tools/adb"
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
cd /Users/austin/AndroidStudioProjects/ClipCC
./gradlew :app:installDebug :app:installDebugAndroidTest 2>&1 | tail -3
PKG=com.example.clipcc
RUNNER=$($ADB shell pm list instrumentation | sed -n 's/^instrumentation:\([^ ]*\) .*/\1/p' | grep "$PKG.test" | head -1)
$ADB logcat -c
$ADB shell am instrument -w -e class com.example.clipcc.engine.BenchmarkMatrixTest "$RUNNER" 2>&1 | tee /tmp/matrix_out.txt
RP=$(grep -o "benchmark_result_path=[^ ]*" /tmp/matrix_out.txt | head -1 | cut -d= -f2)
$ADB pull "$RP" /tmp/benchmark_result.json 2>&1 | tail -1 || $ADB shell run-as $PKG cat files/benchmark_result.json > /tmp/benchmark_result.json
python3 -m json.tool /tmp/benchmark_result.json | head -60
```
Expected: instrumentation passes; JSON pulled with `prep` (4 entries), `runs` (8 = 4 models × 2 CPU lanes), `capabilities` (16 = 4 models × 4 backends). Verify by inspection: CPU_EP `fps` ≥ CPU_XNNPACK `fps` on multi-frame (batched should help base/large); NNAPI capability rows show `addEp`/`sessionCreate` outcomes + provider counts; any `thermalThrottled:true` run is flagged.

- [ ] **Step 3: Verify the acceptance-gate criteria explicitly**

Confirm against the pulled JSON + the prior task gates:
- 4 models × {CPU_XNNPACK, CPU_EP} timed with component breakdown + `BackendConfig` + `effective_batch_size` + `RunMetadata`. ✓ from JSON.
- Per-model `BackendCapabilityReport` for all 4 backends (incl. NNAPI vision-tower probes) with outcomes + coverage. ✓ from JSON.
- Reproducibility: each timed run has median + min/max; throttled runs flagged. ✓ from JSON.
- **Parity preserved:** `EndToEndParityTest` (both XNNPACK + CPU_EP) green from Task 2. ✓
- **fp16 coverage:** create `app/src/androidTest/java/com/example/clipcc/engine/Fp16ConsistencyTest.kt` and run it — it asserts XNNPACK-vs-CPU_EP cosine agreement per model (the fp16 smoke for large/so400m), using the 2 lossless fixture frames (deterministic, ~fast):
```kotlin
package com.example.clipcc.engine

import ai.onnxruntime.OrtEnvironment
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs

@RunWith(AndroidJUnit4::class)
class Fp16ConsistencyTest {
    private val models = listOf(
        "siglip2-base-patch16-256" to 16, "siglip2-base-patch16-384" to 16,
        "siglip2-large-patch16-384" to 8, "siglip2-so400m-patch14-384" to 4,
    )
    private val labels = listOf("Car", "texting while driving", "a dog")

    @Test fun xnnpack_vs_cpuep_cosine_consistency_per_model() {
        val ctx = InstrumentationRegistry.getInstrumentation().context
        val env = OrtEnvironment.getEnvironment()
        val frames = listOf("frame_000.png", "frame_001.png").map { n ->
            ctx.assets.open("fixtures/$n").use { BitmapFactory.decodeStream(it) }
        }
        for ((modelId, batch) in models) {
            val dir = "/data/local/tmp/clipcc_models/$modelId"
            val manifest = ModelBundleManifest.parse(File("$dir/manifest.json").readText())
            val xnn = Engine(dir, manifest, env, Backend.CPU_XNNPACK).scoreFrames(frames, labels)
            val cpu = Engine(dir, manifest, env, Backend.CPU_EP, visionBatch = batch).scoreFrames(frames, labels)
            var maxAbs = 0f
            for (f in xnn.cosine.indices) for (l in labels.indices)
                maxAbs = maxOf(maxAbs, abs(xnn.cosine[f][l] - cpu.cosine[f][l]))
            println("FP16CONSIST $modelId max_abs=$maxAbs")
            assertTrue("$modelId XNNPACK vs CPU_EP cosine ($maxAbs)", maxAbs <= 1e-2f)
        }
    }
}
```
Run:
```bash
$ADB logcat -c
./gradlew :app:connectedDebugAndroidTest --console=plain \
  -Pandroid.testInstrumentationRunnerArguments.class=com.example.clipcc.engine.Fp16ConsistencyTest 2>&1 | tail -8
$ADB logcat -d | grep -i "FP16CONSIST" | tail -5
```
Expected: PASS; one `FP16CONSIST <model> max_abs=<n>` per model, all ≤ 1e-2. If any (esp. large/so400m fp16) exceeds, STOP and report — fp16 EP divergence is a real finding.

- [ ] **Step 4: Write `phase2-report.md` (host repo)**

Create `docs/superpowers/plans/phase2-report.md` recording: per-model CPU_XNNPACK vs CPU_EP `vision_ms`/`fps`/`ms_per_frame` (from the pulled JSON), decode+preprocess prep ms per model, per-model backend-capability table (applied EP + node coverage % + NNAPI outcomes), fp16 cross-lane `max_abs` per model, parity status (both lanes vs golden), thermal/battery conditions of the run, and the media3 version. Note any model that throttled or required a batch deviation.

- [ ] **Step 5: Save + commit the report (host repo only)**

```bash
cd /Users/austin/MITAC/clipCC
git add docs/superpowers/plans/phase2-report.md docs/superpowers/plans/2026-06-03-clipcc-android-phase2-benchmark.md
git commit -m "docs(phase2): benchmark harness plan + completion report"
```

---

## Phase 2 Definition of Done
- [ ] `OrtTower` is backend-aware (4 configs); CPU_XNNPACK (per-frame) and CPU_EP (batched, D5 map) both produce correct, mutually-consistent embeddings (Task 1).
- [ ] `Engine` is backend-parameterized; base-256 end-to-end cosine still matches `scores_golden.json` ≤ 0.01 on BOTH lanes (Task 2).
- [ ] `BackendCapabilityReport` emits per-(model,tower,backend) provider coverage from untimed profiling; NNAPI lanes record `addEp`/`sessionCreate` outcomes (Tasks 3, 7).
- [ ] `FrameSampler` decodes the real test clip (frame count/dims/rotation/timestamps) single-threaded (Task 4).
- [ ] `Benchmark` produces FramePrep-once + per-lane TimedRun with warm-up + median-of-3 + cool-down + `RunMetadata` (thermal/battery/run-order); fast smoke test separate from the full matrix (Tasks 5, 6).
- [ ] Results retrieved off-device as JSON via `am instrument` (+ `run-as` fallback) (Task 6).
- [ ] Full matrix: 4 models × 2 CPU lanes timed + 4×4 capability reports; fp16 cross-lane consistency for large/so400m; report written (Task 7).

## Open items carried to Plan 3 (UI) / later
- Compose UI + benchmark panel consuming the JSON; experimental-badge for NNAPI rows.
- Full network downloader (Xet/resume/external-data) — adb-push suffices for the bench phone.
- Optional fp32 transformers goldens for base-384/large/so400m to measure true fp16 drift (not just cross-lane consistency).
- Manifest schema-v2 (temporal/contrast defaults) when the UI consumes those modes.
- CPU clock locking (needs userdebug/root) for lower-variance numbers.
