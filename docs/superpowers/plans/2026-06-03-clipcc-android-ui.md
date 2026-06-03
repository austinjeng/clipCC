# clipCC-Android Plan 3 — Compose UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Jetpack Compose / Material 3 UI (Classify + Benchmark tabs) on top of the completed headless engine, so the four aggregation modes run on-device and render, and the captured Plan-2 benchmark is shown read-only.

**Architecture:** Single-Activity, two-tab Compose app. A `ClassifyViewModel` (StateFlow + `SavedStateHandle`) drives a memory-bounded, chunked, cancellable run through an injected `Classifier` seam (`RealClassifier` = `FrameSampler` + `Engine` + `Scoring`). Charts are hand-drawn Canvas. The benchmark tab parses a bundled JSON asset. Engine changes are additive and parity-neutral (existing `scoreFrames` + Plan-1/2 tests untouched).

**Tech Stack:** Kotlin, Jetpack Compose (Material 3), AndroidX Lifecycle (ViewModel/Compose/SavedState), Coroutines, ONNX Runtime 1.26.0, Media3 1.10.1, JUnit4 + kotlinx-coroutines-test.

**Source of truth:** `docs/superpowers/specs/2026-06-03-clipcc-android-ui-design.md` (Rev 3). Read §0 for the project-root / paths / sync policy — **all `app/...` paths below are relative to the Android project root `/Users/austin/AndroidStudioProjects/ClipCC` (NOT git; "commit" = save).** Specs/plans/reports are committed in the host repo `/Users/austin/MITAC/clipCC`.

**Environment:** `JAVA_HOME=/Applications/Android Studio.app/Contents/jbr/Contents/Home`; adb `~/Library/Android/sdk/platform-tools/adb`; device Pixel 7a (serial `36161JEHN16600`, API 36). JVM unit tests: `./gradlew testDebugUnitTest`. Instrumented: `./gradlew connectedDebugAndroidTest`. Run a single JVM test class: `./gradlew testDebugUnitTest --tests "com.example.clipcc.*ClassName*"`.

---

## File structure

**New (UI):**
- `app/src/main/java/com/example/clipcc/ui/app/ClipCCApp.kt` — 2-tab scaffold
- `app/src/main/java/com/example/clipcc/ui/classify/UiBackend.kt` — UI backend enum + mapping
- `app/src/main/java/com/example/clipcc/ui/classify/ClassifyModels.kt` — DTOs (ClassifyRequest/RunResult/RunMeta/options/state)
- `app/src/main/java/com/example/clipcc/ui/classify/Classifier.kt` — seam interface + RunCancelledException re-export
- `app/src/main/java/com/example/clipcc/ui/classify/RealClassifier.kt` — chunked pipeline
- `app/src/main/java/com/example/clipcc/ui/classify/ClassifyViewModel.kt` — state holder
- `app/src/main/java/com/example/clipcc/ui/classify/LabelValidation.kt` — pure validation
- `app/src/main/java/com/example/clipcc/ui/classify/SetupCard.kt`, `RunStatus.kt`, `ResultsSection.kt`, `ModeExtras.kt`
- `app/src/main/java/com/example/clipcc/ui/benchmark/BenchmarkData.kt`, `BenchmarkScreen.kt`
- `app/src/main/java/com/example/clipcc/ui/charts/ChartData.kt`, `BarChart.kt`, `TimelineChart.kt`
- `app/src/main/java/com/example/clipcc/data/ModelRepository.kt`
- `app/src/main/assets/phase2-benchmark-result.json` — bundled benchmark snapshot

**New (engine):**
- `app/src/main/java/com/example/clipcc/engine/ScoringPolicy.kt`

**Modified (engine, additive only):**
- `engine/Manifest.kt` — read `display_name`, `score_semantics`, per-file `bytes`/`sha256`
- `engine/FrameSampler.kt` — streaming `sample(uri, fps, max, onFrame)` overload
- `engine/OrtTower.kt` — `encodeVision` gains optional `onItem`/`isCancelled`
- `engine/Engine.kt` — additive chunked primitives (`encodeTextEmbeddings`, `withVisionEncoder`) + `RunCancelledException`

**Modified (app shell):**
- `app/build.gradle.kts` — add lifecycle/coroutines-test deps
- `app/src/main/java/com/example/clipcc/MainActivity.kt` — host `ClipCCApp`, keep-screen-on
- `app/src/main/AndroidManifest.xml` — (no change needed; `OpenDocument` needs no permission)

**New tests:**
- JVM: `ScoringPolicyTest`, `ManifestExtensionTest`, `ModelRepositoryTest`, `LabelValidationTest`, `ChartDataTest`, `BenchmarkDataTest`, `ClassifyViewModelTest`
- Instrumented: `ClassifyEndToEndSmokeTest` (+ existing Plan-1/2 tests must stay green)

---

## Task 1: Gradle deps + 2-tab scaffold

**Files:**
- Modify: `app/build.gradle.kts`
- Create: `app/src/main/java/com/example/clipcc/ui/app/ClipCCApp.kt`
- Modify: `app/src/main/java/com/example/clipcc/MainActivity.kt`
- Delete: the template `Greeting`/`GreetingPreview` in `MainActivity.kt`

- [ ] **Step 1: Add dependencies** to `app/build.gradle.kts` inside the `dependencies { }` block (after the existing `media3.inspector.frame` line), matching the project's existing direct-string style:

```kotlin
    // Plan 3 UI: ViewModel + Compose state + SavedStateHandle (pinned to the catalog lifecycle 2.6.1)
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.6.1")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.6.1")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.6.1")
    implementation("androidx.lifecycle:lifecycle-viewmodel-savedstate:2.6.1")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.1")
```

- [ ] **Step 2: Create the placeholder scaffold** `ui/app/ClipCCApp.kt` (the two real screens are filled in later tasks; for now stub bodies prove the tabs build):

```kotlin
package com.example.clipcc.ui.app

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier

@Composable
fun ClipCCApp() {
    var tab by remember { mutableIntStateOf(0) }
    val titles = listOf("Classify", "Benchmark")
    Scaffold(modifier = Modifier.fillMaxSize()) { pad ->
        androidx.compose.foundation.layout.Column(Modifier.padding(pad)) {
            TabRow(selectedTabIndex = tab) {
                titles.forEachIndexed { i, t ->
                    Tab(selected = tab == i, onClick = { tab = i }, text = { Text(t) })
                }
            }
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(if (tab == 0) "Classify (TODO)" else "Benchmark (TODO)")
            }
        }
    }
}
```

- [ ] **Step 3: Replace `MainActivity.kt`** entirely (drops the template `Greeting`):

```kotlin
package com.example.clipcc

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.example.clipcc.ui.app.ClipCCApp
import com.example.clipcc.ui.theme.ClipCCTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent { ClipCCTheme { ClipCCApp() } }
    }
}
```

- [ ] **Step 4: Build** to verify deps resolve and Compose compiles.

Run: `./gradlew :app:assembleDebug`
Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 5: Commit (save)** — Android project is not git; "commit" = files saved. Record in the task log: files touched + build result.

---

## Task 2: `ScoringPolicy` (engine constants) + test

**Files:**
- Create: `app/src/main/java/com/example/clipcc/engine/ScoringPolicy.kt`
- Test: `app/src/test/java/com/example/clipcc/engine/ScoringPolicyTest.kt`

- [ ] **Step 1: Write the failing test** `ScoringPolicyTest.kt` (pure JVM; values pinned to the Python sources — `app/services/temporal_policy.py`, `app/schemas/response.py`, `app/config.py` in the host repo):

```kotlin
package com.example.clipcc.engine

import org.junit.Assert.assertEquals
import org.junit.Test

class ScoringPolicyTest {
    @Test fun constants_match_python() {
        assertEquals(0.5, ScoringPolicy.THRESHOLD, 0.0)
        assertEquals("absolute", ScoringPolicy.THRESHOLD_MODE)
        assertEquals(2.0, ScoringPolicy.GAP_TOLERANCE, 0.0)
        assertEquals(1.0, ScoringPolicy.MIN_DURATION, 0.0)
        assertEquals(0.15, ScoringPolicy.CONTRAST_THRESHOLD, 0.0)
        assertEquals("mean", ScoringPolicy.CONTRAST_REDUCE)
        assertEquals(listOf("mean", "top_k_mean", "max", "quantile"), ScoringPolicy.CONTRAST_REDUCE_MODES)
        assertEquals("siglip2_pairwise_sigmoid", ScoringPolicy.SCORE_SEMANTICS)
        assertEquals(1.0, ScoringPolicy.FPS, 0.0)
        assertEquals(300, ScoringPolicy.MAX_FRAMES)
        assertEquals(
            listOf("texting while driving", "sleeping while driving", "eating while driving"),
            ScoringPolicy.DEFAULT_LABELS,
        )
    }

    @Test fun vision_chunk_matches_phase2_batches() {
        assertEquals(16, ScoringPolicy.visionChunkFor("siglip2-base-patch16-256", Backend.CPU_EP))
        assertEquals(16, ScoringPolicy.visionChunkFor("siglip2-base-patch16-384", Backend.CPU_EP))
        assertEquals(8, ScoringPolicy.visionChunkFor("siglip2-large-patch16-384", Backend.CPU_EP))
        assertEquals(4, ScoringPolicy.visionChunkFor("siglip2-so400m-patch14-384", Backend.CPU_EP))
        // Non-CPU_EP (XNNPACK/NNAPI) encodes per-frame; chunk is decode/release granularity = 16.
        assertEquals(16, ScoringPolicy.visionChunkFor("siglip2-so400m-patch14-384", Backend.CPU_XNNPACK))
    }
}
```

- [ ] **Step 2: Run it — verify it fails.**

Run: `./gradlew testDebugUnitTest --tests "com.example.clipcc.engine.ScoringPolicyTest"`
Expected: FAIL (`ScoringPolicy` unresolved).

- [ ] **Step 3: Create `ScoringPolicy.kt`:**

```kotlin
package com.example.clipcc.engine

/**
 * Model-INDEPENDENT scoring/policy defaults, pinned to the Python reference (single source of truth;
 * a per-model manifest bump would duplicate these 4×). See spec §2 Decision 4.
 *   THRESHOLD/THRESHOLD_MODE/CONTRAST_* : app/services/temporal_policy.py SigLip2Policy
 *   GAP_TOLERANCE/MIN_DURATION          : app/schemas/response.py ResolvedTemporalOptions
 *   DEFAULT_LABELS                      : app/config.py default_labels
 *   FPS/MAX_FRAMES                      : app/services/video.py
 */
object ScoringPolicy {
    const val THRESHOLD = 0.5
    const val THRESHOLD_MODE = "absolute"
    const val GAP_TOLERANCE = 2.0
    const val MIN_DURATION = 1.0
    const val CONTRAST_THRESHOLD = 0.15
    const val CONTRAST_REDUCE = "mean"
    val CONTRAST_REDUCE_MODES = listOf("mean", "top_k_mean", "max", "quantile")
    const val SCORE_SEMANTICS = "siglip2_pairwise_sigmoid"
    const val FPS = 1.0
    const val MAX_FRAMES = 300
    val DEFAULT_LABELS = listOf(
        "texting while driving", "sleeping while driving", "eating while driving",
    )

    /** Chunked vision-encode size per (model, backend), mirroring the phase-2 CPU_EP batches.
     *  Non-CPU_EP backends encode per-frame; the chunk is then just decode/release granularity (16). */
    fun visionChunkFor(modelId: String, backend: Backend): Int = when {
        backend != Backend.CPU_EP -> 16
        modelId.contains("so400m") -> 4
        modelId.contains("large") -> 8
        else -> 16
    }
}
```

- [ ] **Step 4: Run it — verify it passes.**

Run: `./gradlew testDebugUnitTest --tests "com.example.clipcc.engine.ScoringPolicyTest"`
Expected: PASS.

- [ ] **Step 5: Commit (save).**

---

## Task 3: `Manifest` extension (display_name, score_semantics, bytes, sha256) + test

**Files:**
- Modify: `app/src/main/java/com/example/clipcc/engine/Manifest.kt`
- Test: `app/src/test/java/com/example/clipcc/engine/ManifestExtensionTest.kt`
- Fixture already present: `app/src/test/resources/manifest_base256.json`

- [ ] **Step 1: Write the failing test** `ManifestExtensionTest.kt`:

```kotlin
package com.example.clipcc.engine

import org.junit.Assert.assertEquals
import org.junit.Test

class ManifestExtensionTest {
    private val json = ManifestExtensionTest::class.java.classLoader!!
        .getResource("manifest_base256.json")!!.readText()

    @Test fun parses_new_v1_fields() {
        val m = ModelBundleManifest.parse(json)
        assertEquals("SigLIP2 Base (256px)", m.displayName)
        assertEquals("siglip2_pairwise_sigmoid", m.scoreSemantics)
        assertEquals(371992072L, m.visionBytes)
        assertEquals(1129469657L, m.textBytes)
        assertEquals("f5cb16728a704703f05516ded628397e11dbca4de2eb5db04b0c0bcee988aa7a", m.visionSha256)
        assertEquals("d3de4a6bbbfcb429b6615ac496790353cf4a4fc0f19fbbe7179e523ae60daaef", m.textSha256)
        assertEquals("cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322", m.tokenizerSha256)
    }
}
```

- [ ] **Step 2: Run — verify it fails.**

Run: `./gradlew testDebugUnitTest --tests "com.example.clipcc.engine.ManifestExtensionTest"`
Expected: FAIL (unresolved `displayName`).

- [ ] **Step 3: Extend `Manifest.kt`** — add the fields to the data class and parse them (additive; `schema_version == 1` unchanged). Replace the whole file with:

```kotlin
package com.example.clipcc.engine

import org.json.JSONObject

data class ModelBundleManifest(
    val schemaVersion: Int, val modelId: String, val displayName: String, val resolution: Int,
    val precision: String, val scoreSemantics: String,
    val logitScale: Double, val logitBias: Double,
    val visionFile: String, val visionDataFile: String?, val visionBytes: Long, val visionSha256: String,
    val textFile: String, val textDataFile: String?, val textBytes: Long, val textSha256: String,
    val tokenizerFile: String, val tokenizerSha256: String, val lowercaseAppliedBy: String,
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
                displayName = d.getString("display_name"),
                resolution = d.getInt("resolution"), precision = d.getString("precision"),
                scoreSemantics = d.getString("score_semantics"),
                logitScale = d.getDouble("logit_scale"), logitBias = d.getDouble("logit_bias"),
                visionFile = v.getString("file"),
                visionDataFile = if (v.isNull("data_file")) null else v.getString("data_file"),
                visionBytes = v.getLong("bytes"), visionSha256 = v.getString("sha256"),
                textFile = t.getString("file"),
                textDataFile = if (t.isNull("data_file")) null else t.getString("data_file"),
                textBytes = t.getLong("bytes"), textSha256 = t.getString("sha256"),
                tokenizerFile = tok.getString("file"), tokenizerSha256 = tok.getString("sha256"),
                lowercaseAppliedBy = tok.getString("lowercase_applied_by"),
                resample = pp.getString("resample"), maxLength = tok.getInt("max_length"),
                padId = tok.getInt("pad_id"),
            )
        }
    }
}
```

- [ ] **Step 4: Run the extension test AND the existing `ManifestTest`** (the existing test asserts the original fields parse byte-identically — it must stay green since the change is additive):

Run: `./gradlew testDebugUnitTest --tests "com.example.clipcc.engine.ManifestExtensionTest" --tests "com.example.clipcc.engine.ManifestTest"`
Expected: both PASS. (If `ManifestTest` constructs a `ModelBundleManifest` directly with positional args, update it to the new constructor — keep its assertions.)

- [ ] **Step 5: Commit (save).**

---

## Task 4: `ModelRepository` (discovery + readiness) + test

**Files:**
- Create: `app/src/main/java/com/example/clipcc/data/ModelRepository.kt`
- Test: `app/src/test/java/com/example/clipcc/data/ModelRepositoryTest.kt`

Readiness (spec §8): parse manifest → all referenced files exist → **size-match vision/text only** (tokenizer has no `bytes` in v1) → `score_semantics == siglip2_pairwise_sigmoid`. Takes a `File` root (not `Context`) so it is plain-JVM testable.

- [ ] **Step 1: Write the failing test** `ModelRepositoryTest.kt` (uses JUnit `TemporaryFolder`; writes a real manifest + dummy files of the right size):

```kotlin
package com.example.clipcc.data

import com.example.clipcc.engine.ScoringPolicy
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class ModelRepositoryTest {
    @get:Rule val tmp = TemporaryFolder()

    /** Minimal valid v1 manifest with caller-controlled file sizes + semantics. */
    private fun manifest(visBytes: Long, txtBytes: Long, semantics: String = ScoringPolicy.SCORE_SEMANTICS) = """
        {"schema_version":1,"model_id":"siglip2-base-patch16-256","display_name":"SigLIP2 Base (256px)",
         "resolution":256,"precision":"fp32","score_semantics":"$semantics",
         "logit_scale":4.7,"logit_bias":-16.7,
         "vision":{"file":"vision_model.onnx","data_file":null,"bytes":$visBytes,"sha256":"a"},
         "text":{"file":"text_model.onnx","data_file":null,"bytes":$txtBytes,"sha256":"b"},
         "tokenizer":{"file":"tokenizer.json","sha256":"c","max_length":64,"pad_id":0,
           "lowercase_applied_by":"tokenizer_json"},
         "preprocess":{"resample":"bilinear"}}
    """.trimIndent()

    private fun bundle(id: String, visBytes: Long, txtBytes: Long,
                       visActual: Int, txtActual: Int, semantics: String = ScoringPolicy.SCORE_SEMANTICS,
                       withTokenizer: Boolean = true): File {
        val dir = File(tmp.root, "models/$id").apply { mkdirs() }
        File(dir, "manifest.json").writeText(manifest(visBytes, txtBytes, semantics))
        File(dir, "vision_model.onnx").writeBytes(ByteArray(visActual))
        File(dir, "text_model.onnx").writeBytes(ByteArray(txtActual))
        if (withTokenizer) File(dir, "tokenizer.json").writeText("{}")
        return dir
    }

    private fun repo() = ModelRepository(File(tmp.root, "models"))

    @Test fun ready_when_files_present_and_sizes_match() {
        bundle("m1", visBytes = 10, txtBytes = 20, visActual = 10, txtActual = 20)
        val info = repo().scan().single()
        assertEquals("m1", info.id)
        assertEquals("SigLIP2 Base (256px)", info.displayName)
        assertEquals(true, info.ready)
        assertNull(info.reason)
    }

    @Test fun not_ready_when_vision_size_mismatch() {
        bundle("m1", visBytes = 10, txtBytes = 20, visActual = 9, txtActual = 20)
        val info = repo().scan().single()
        assertEquals(false, info.ready)
        assertEquals("size mismatch", info.reason)
    }

    @Test fun not_ready_when_file_missing() {
        bundle("m1", visBytes = 10, txtBytes = 20, visActual = 10, txtActual = 20, withTokenizer = false)
        val info = repo().scan().single()
        assertEquals(false, info.ready)
        assertEquals("not provisioned", info.reason)
    }

    @Test fun not_ready_when_unsupported_semantics() {
        bundle("m1", visBytes = 10, txtBytes = 20, visActual = 10, txtActual = 20, semantics = "clip_softmax")
        val info = repo().scan().single()
        assertEquals(false, info.ready)
        assertEquals("unsupported semantics", info.reason)
    }

    @Test fun empty_root_returns_empty() {
        assertEquals(emptyList<ModelInfo>(), repo().scan())
    }
}
```

- [ ] **Step 2: Run — verify it fails.**

Run: `./gradlew testDebugUnitTest --tests "com.example.clipcc.data.ModelRepositoryTest"`
Expected: FAIL (unresolved `ModelRepository`).

- [ ] **Step 3: Create `ModelRepository.kt`:**

```kotlin
package com.example.clipcc.data

import com.example.clipcc.engine.ModelBundleManifest
import com.example.clipcc.engine.ScoringPolicy
import java.io.File

data class ModelInfo(
    val id: String, val displayName: String, val resolution: Int, val precision: String,
    val scoreSemantics: String, val ready: Boolean, val reason: String?, val dir: String,
)

/** Scans [modelsRoot]/<id>/manifest.json bundles. Readiness = parse + files-exist + size-match
 *  (vision/text only — v1 has no tokenizer `bytes`) + supported semantics. See spec §8. */
class ModelRepository(private val modelsRoot: File) {
    fun scan(): List<ModelInfo> {
        val dirs = modelsRoot.listFiles { f -> f.isDirectory }?.sortedBy { it.name } ?: return emptyList()
        return dirs.mapNotNull { dir -> infoFor(dir) }
    }

    private fun infoFor(dir: File): ModelInfo? {
        val manifestFile = File(dir, "manifest.json")
        if (!manifestFile.exists()) return null
        val m = try {
            ModelBundleManifest.parse(manifestFile.readText())
        } catch (t: Throwable) {
            return ModelInfo(dir.name, dir.name, 0, "", "", false, "bad manifest: ${t.message}", dir.absolutePath)
        }
        val reason = readinessReason(dir, m)
        return ModelInfo(
            id = m.modelId, displayName = m.displayName, resolution = m.resolution,
            precision = m.precision, scoreSemantics = m.scoreSemantics,
            ready = reason == null, reason = reason, dir = dir.absolutePath,
        )
    }

    private fun readinessReason(dir: File, m: ModelBundleManifest): String? {
        val required = buildList {
            add(m.visionFile); add(m.textFile); add(m.tokenizerFile)
            m.visionDataFile?.let { add(it) }; m.textDataFile?.let { add(it) }
        }
        if (required.any { !File(dir, it).exists() }) return "not provisioned"
        if (File(dir, m.visionFile).length() != m.visionBytes) return "size mismatch"
        if (File(dir, m.textFile).length() != m.textBytes) return "size mismatch"
        if (m.scoreSemantics != ScoringPolicy.SCORE_SEMANTICS) return "unsupported semantics"
        return null
    }
}
```

- [ ] **Step 4: Run — verify it passes.**

Run: `./gradlew testDebugUnitTest --tests "com.example.clipcc.data.ModelRepositoryTest"`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit (save).**

---

## Task 5: `LabelValidation` (pure) + test

**Files:**
- Create: `app/src/main/java/com/example/clipcc/ui/classify/LabelValidation.kt`
- Test: `app/src/test/java/com/example/clipcc/ui/classify/LabelValidationTest.kt`

Rules (spec §5.1 / P2-11): trim; reject blanks; reject exact duplicates within and across groups; preserve case.

- [ ] **Step 1: Write the failing test** `LabelValidationTest.kt`:

```kotlin
package com.example.clipcc.ui.classify

import org.junit.Assert.assertEquals
import org.junit.Test

class LabelValidationTest {
    @Test fun trims_and_keeps_case() {
        val r = LabelValidation.validate(listOf("  Car ", "truck"), emptyList(), contrast = false)
        assertEquals(listOf("Car", "truck"), r.cleaned)
        assertEquals(null, r.error)
    }

    @Test fun rejects_all_blank() {
        val r = LabelValidation.validate(listOf("   ", ""), emptyList(), contrast = false)
        assertEquals("Add at least one label", r.error)
    }

    @Test fun rejects_duplicate_within_group() {
        val r = LabelValidation.validate(listOf("car", "car"), emptyList(), contrast = false)
        assertEquals("Duplicate label: car", r.error)
    }

    @Test fun contrast_requires_both_groups() {
        val r = LabelValidation.validate(listOf("car"), emptyList(), contrast = true)
        assertEquals("Contrast needs at least one positive and one negative label", r.error)
    }

    @Test fun contrast_rejects_cross_group_duplicate() {
        val r = LabelValidation.validate(listOf("car"), listOf("car"), contrast = true)
        assertEquals("Duplicate label: car", r.error)
    }

    @Test fun contrast_ok_returns_pos_then_neg_and_count() {
        val r = LabelValidation.validate(listOf("a", "b"), listOf("c"), contrast = true)
        assertEquals(listOf("a", "b", "c"), r.cleaned)
        assertEquals(2, r.posCount)
        assertEquals(null, r.error)
    }
}
```

- [ ] **Step 2: Run — verify it fails.**

Run: `./gradlew testDebugUnitTest --tests "com.example.clipcc.ui.classify.LabelValidationTest"`
Expected: FAIL.

- [ ] **Step 3: Create `LabelValidation.kt`:**

```kotlin
package com.example.clipcc.ui.classify

/** Pure label validation. `cleaned` is trimmed, blanks removed; for contrast it is posLabels + negLabels
 *  and `posCount` = positive count (the engine's contrast contract). */
data class LabelCheck(val cleaned: List<String>, val posCount: Int, val error: String?)

object LabelValidation {
    fun validate(positives: List<String>, negatives: List<String>, contrast: Boolean): LabelCheck {
        val pos = positives.map { it.trim() }.filter { it.isNotEmpty() }
        val neg = negatives.map { it.trim() }.filter { it.isNotEmpty() }

        if (!contrast) {
            if (pos.isEmpty()) return LabelCheck(emptyList(), 0, "Add at least one label")
            dup(pos)?.let { return LabelCheck(emptyList(), 0, it) }
            return LabelCheck(pos, 0, null)
        }
        if (pos.isEmpty() || neg.isEmpty())
            return LabelCheck(emptyList(), 0, "Contrast needs at least one positive and one negative label")
        dup(pos + neg)?.let { return LabelCheck(emptyList(), 0, it) }
        return LabelCheck(pos + neg, pos.size, null)
    }

    private fun dup(all: List<String>): String? {
        val seen = HashSet<String>()
        for (s in all) if (!seen.add(s)) return "Duplicate label: $s"
        return null
    }
}
```

- [ ] **Step 4: Run — verify it passes.**

Run: `./gradlew testDebugUnitTest --tests "com.example.clipcc.ui.classify.LabelValidationTest"`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit (save).**

---

## Task 6: `ChartData` (pure chart prep) + test

**Files:**
- Create: `app/src/main/java/com/example/clipcc/ui/charts/ChartData.kt`
- Test: `app/src/test/java/com/example/clipcc/ui/charts/ChartDataTest.kt`

Separate-scale design (spec §6): confidence on [0,1]; raw cosine on a symmetric axis sized to its own magnitude.

- [ ] **Step 1: Write the failing test** `ChartDataTest.kt`:

```kotlin
package com.example.clipcc.ui.charts

import com.example.clipcc.engine.ScoreItem
import org.junit.Assert.assertEquals
import org.junit.Test

class ChartDataTest {
    private val items = listOf(
        ScoreItem("a", confidence = 0.9, rawSimilarity = 0.12),
        ScoreItem("b", confidence = 0.2, rawSimilarity = -0.05),
    )

    @Test fun confidence_bars_use_unit_scale() {
        val bars = ChartData.confidenceBars(items)
        assertEquals(listOf("a", "b"), bars.map { it.label })
        assertEquals(listOf(0.9f, 0.2f), bars.map { it.value })
        assertEquals(1f, ChartData.UNIT_MAX, 0f)
    }

    @Test fun cosine_axis_is_symmetric_to_max_abs() {
        // max |value| = 0.12 → symmetric axis half-range rounds up to a 0.05 step boundary (0.15)
        assertEquals(0.15f, ChartData.symmetricMax(items.map { it.rawSimilarity.toFloat() }), 1e-6f)
    }

    @Test fun cosine_axis_floor_is_nonzero_for_tiny_values() {
        assertEquals(0.05f, ChartData.symmetricMax(listOf(0.001f, -0.002f)), 1e-6f)
    }
}
```

- [ ] **Step 2: Run — verify it fails.**

Run: `./gradlew testDebugUnitTest --tests "com.example.clipcc.ui.charts.ChartDataTest"`
Expected: FAIL.

- [ ] **Step 3: Create `ChartData.kt`:**

```kotlin
package com.example.clipcc.ui.charts

import com.example.clipcc.engine.ScoreItem
import kotlin.math.abs
import kotlin.math.ceil

data class Bar(val label: String, val value: Float)

/** Pure preparation for the Canvas charts. Confidence bars share a [0,1] scale; cosine bars use a
 *  symmetric axis sized to their own magnitude (so a tiny cosine isn't drawn as if on a 0..1 scale). */
object ChartData {
    const val UNIT_MAX = 1f
    private const val STEP = 0.05f

    fun confidenceBars(items: List<ScoreItem>): List<Bar> =
        items.map { Bar(it.label, it.confidence.toFloat()) }

    fun cosineBars(items: List<ScoreItem>): List<Bar> =
        items.map { Bar(it.label, it.rawSimilarity.toFloat()) }

    /** Smallest STEP-aligned half-range that covers max|value|, with a non-zero floor of one STEP. */
    fun symmetricMax(values: List<Float>): Float {
        val maxAbs = values.maxOfOrNull { abs(it) } ?: 0f
        val steps = ceil(maxAbs / STEP).toInt().coerceAtLeast(1)
        return steps * STEP
    }
}
```

- [ ] **Step 4: Run — verify it passes.**

Run: `./gradlew testDebugUnitTest --tests "com.example.clipcc.ui.charts.ChartDataTest"`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit (save).**

---

## Task 7: `BenchmarkData` (parse snapshot) + asset ingestion + test

**Files:**
- Create: `app/src/main/assets/phase2-benchmark-result.json` (copy of the host repo's `docs/superpowers/plans/phase2-benchmark-result.json`)
- Create: `app/src/main/java/com/example/clipcc/ui/benchmark/BenchmarkData.kt`
- Test: `app/src/test/java/com/example/clipcc/ui/benchmark/BenchmarkDataTest.kt`
- Test fixture: `app/src/test/resources/phase2-benchmark-result.json` (same file copied to test resources)

- [ ] **Step 1: Ingest the asset.** Copy the snapshot into BOTH the app assets and the test resources:

```bash
SRC=/Users/austin/MITAC/clipCC/docs/superpowers/plans/phase2-benchmark-result.json
DST=/Users/austin/AndroidStudioProjects/ClipCC/app/src/main
mkdir -p "$DST/assets" "$DST/../test/resources"
cp "$SRC" "$DST/assets/phase2-benchmark-result.json"
cp "$SRC" "$DST/../test/resources/phase2-benchmark-result.json"
```

- [ ] **Step 2: Write the failing test** `BenchmarkDataTest.kt` (CPU lanes become timed rows; NNAPI lanes become capability-only rows; vision coverage joined from `capabilities`):

```kotlin
package com.example.clipcc.ui.benchmark

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class BenchmarkDataTest {
    private val json = BenchmarkDataTest::class.java.classLoader!!
        .getResource("phase2-benchmark-result.json")!!.readText()

    @Test fun parses_cpu_timed_rows_with_coverage() {
        val groups = BenchmarkData.parse(json)
        val base = groups.first { it.modelId == "siglip2-base-patch16-256" }
        val xnn = base.timed.first { it.backend == "CPU_XNNPACK" }
        assertEquals(2372.286, xnn.msPerFrame, 1e-3)
        assertEquals(0.422, xnn.fps, 1e-3)
        // vision node coverage joined from capabilities (XNNPACK 11.84%)
        assertEquals(11.84, xnn.visionDelegatedPct!!, 1e-2)
    }

    @Test fun nnapi_is_capability_only_not_timed() {
        val groups = BenchmarkData.parse(json)
        val base = groups.first { it.modelId == "siglip2-base-patch16-256" }
        assertTrue(base.timed.none { it.backend.startsWith("NNAPI") })
        val nnapi = base.capabilityOnly.first { it.backend == "NNAPI_DEFAULT" }
        assertEquals(0.0, nnapi.visionDelegatedPct, 1e-6)   // 0% delegated
        assertTrue(nnapi.experimental)
    }
}
```

- [ ] **Step 3: Run — verify it fails.**

Run: `./gradlew testDebugUnitTest --tests "com.example.clipcc.ui.benchmark.BenchmarkDataTest"`
Expected: FAIL.

- [ ] **Step 4: Create `BenchmarkData.kt`** (uses `org.json`, available on JVM via the existing `testImplementation("org.json:json")` and at runtime via Android):

```kotlin
package com.example.clipcc.ui.benchmark

import org.json.JSONObject

data class TimedRow(
    val backend: String, val loadMs: Long, val visionMsMedian: Long,
    val msPerFrame: Double, val fps: Double, val visionDelegatedPct: Double?,
)
data class CapabilityRow(val backend: String, val visionDelegatedPct: Double, val experimental: Boolean)
data class ModelGroup(val modelId: String, val timed: List<TimedRow>, val capabilityOnly: List<CapabilityRow>)

/** Parses the captured phase-2 snapshot. CPU lanes (with timing) → [timed] rows; NNAPI lanes (no
 *  timing, capability probe only) → [capabilityOnly]. Vision node-coverage % joined from `capabilities`. */
object BenchmarkData {
    private const val XNNPACK = "XnnpackExecutionProvider"

    fun parse(json: String): List<ModelGroup> {
        val root = JSONObject(json)
        val runs = root.getJSONArray("runs")
        val caps = root.getJSONArray("capabilities")

        // visionCoverage[model][backend] = non-CPU delegated % (XNNPACK% for XNNPACK lane; 0 for NNAPI/CPU_EP)
        val visionCoverage = HashMap<Pair<String, String>, Double>()
        for (i in 0 until caps.length()) {
            val c = caps.getJSONObject(i)
            if (c.getString("tower") != "vision") continue
            val pct = c.getJSONObject("delegatedPct")
            visionCoverage[c.getString("model") to c.getString("backend")] = pct.optDouble(XNNPACK, 0.0)
        }

        val timedByModel = LinkedHashMap<String, MutableList<TimedRow>>()
        for (i in 0 until runs.length()) {
            val r = runs.getJSONObject(i)
            val model = r.getString("model"); val backend = r.getString("backend")
            timedByModel.getOrPut(model) { mutableListOf() }.add(
                TimedRow(
                    backend = backend, loadMs = r.getLong("loadMs"),
                    visionMsMedian = r.getLong("visionMsMedian"),
                    msPerFrame = r.getDouble("msPerFrame"), fps = r.getDouble("fps"),
                    visionDelegatedPct = visionCoverage[model to backend],
                )
            )
        }

        // capability-only = vision capability rows whose backend never appears in runs (NNAPI lanes)
        val timedBackends = HashMap<String, MutableSet<String>>()
        timedByModel.forEach { (m, rows) -> timedBackends[m] = rows.map { it.backend }.toMutableSet() }
        val capByModel = LinkedHashMap<String, MutableList<CapabilityRow>>()
        for (i in 0 until caps.length()) {
            val c = caps.getJSONObject(i)
            if (c.getString("tower") != "vision") continue
            val model = c.getString("model"); val backend = c.getString("backend")
            if (timedBackends[model]?.contains(backend) == true) continue
            val pct = c.getJSONObject("delegatedPct").optDouble(XNNPACK, 0.0)
            capByModel.getOrPut(model) { mutableListOf() }
                .add(CapabilityRow(backend, pct, experimental = backend.startsWith("NNAPI")))
        }

        return timedByModel.keys.map { m ->
            ModelGroup(m, timedByModel[m] ?: emptyList(), capByModel[m] ?: emptyList())
        }
    }
}
```

- [ ] **Step 5: Run — verify it passes.**

Run: `./gradlew testDebugUnitTest --tests "com.example.clipcc.ui.benchmark.BenchmarkDataTest"`
Expected: PASS.

- [ ] **Step 6: Commit (save).**

---

## Task 8: `FrameSampler` streaming overload (engine touch §7.1)

**Files:**
- Modify: `app/src/main/java/com/example/clipcc/engine/FrameSampler.kt`

Add a single forward streaming pass with an `onFrame` callback that returns `false` to stop; the existing path-based `sample` delegates to it (so `FrameSamplerTest` stays green).

- [ ] **Step 1: Replace the body of `FrameSampler`** (keep imports; add `Uri` already imported). New methods:

```kotlin
@UnstableApi
class FrameSampler(private val context: Context) {

    /** Streaming forward pass: invoke [onFrame] per decoded frame; stop early when it returns false
     *  (cancel) or at EOF. Returns [VideoMeta] (rotation already applied; reported 0). */
    fun sample(uri: Uri, fps: Double, maxFrames: Int, onFrame: (SampledFrame) -> Boolean): VideoMeta {
        val extractor = FrameExtractor.Builder(context, MediaItem.fromUri(uri)).build()
        try {
            var w = 0; var h = 0
            val stepMs = (1000.0 / fps).toLong()
            var lastPtsMs = Long.MIN_VALUE
            var i = 0; var produced = 0
            while (i < maxFrames) {
                val frame = extractor.getFrame(i * stepMs).get() ?: break
                if (frame.presentationTimeMs <= lastPtsMs) break   // seek past EOF clamps PTS
                lastPtsMs = frame.presentationTimeMs
                val bmp = frame.bitmap
                if (produced == 0) { w = bmp.width; h = bmp.height }
                val cont = onFrame(SampledFrame(bmp, frame.presentationTimeMs / 1000.0, produced))
                produced++
                if (!cont) break
                i++
            }
            return VideoMeta(w, h, rotationDegrees = 0, frameCount = produced)
        } finally {
            extractor.close()
        }
    }

    /** Path-based variant used by the Benchmark harness — buffers the streaming pass into a list. */
    fun sample(videoPath: String, fps: Double, maxFrames: Int): Pair<VideoMeta, List<SampledFrame>> {
        val frames = ArrayList<SampledFrame>(maxFrames)
        val meta = sample(Uri.fromFile(File(videoPath)), fps, maxFrames) { frames.add(it); true }
        return meta to frames
    }
}
```

- [ ] **Step 2: Build + run the existing instrumented `FrameSamplerTest` on the device** (it calls the path variant; must stay green). Ensure the device is connected (`adb devices` shows `36161JEHN16600`).

Run: `./gradlew :app:connectedDebugAndroidTest --tests "com.example.clipcc.engine.FrameSamplerTest"`
Expected: PASS (7 frames @720×1280, unchanged from Plan 2).

- [ ] **Step 3: Commit (save).**

---

## Task 9: `OrtTower` cancel/progress + `Engine` chunked primitives (engine touch §7.2/§7.3)

**Files:**
- Modify: `app/src/main/java/com/example/clipcc/engine/OrtTower.kt`
- Modify: `app/src/main/java/com/example/clipcc/engine/Engine.kt`

Additive only: existing `scoreFrames` and `encodeVision` callers are untouched (new params default to no-op).

- [ ] **Step 1: Add `RunCancelledException`** at the top of `Engine.kt` (after the package line / imports):

```kotlin
/** Thrown by the chunked encode path when the cooperative cancel flag is observed. */
class RunCancelledException : RuntimeException("run cancelled")
```

- [ ] **Step 2: Modify `OrtTower.encodeVision`** to accept optional callbacks (default null → unchanged). Replace the existing `encodeVision` with:

```kotlin
    /** Encode [frames] in fixed chunks of [batch] (1 for XNNPACK; the D5 map for CPU_EP).
     *  [onItem] is invoked with the running count after each chunk; [isCancelled] is checked
     *  before each chunk and throws [RunCancelledException] if true. */
    fun encodeVision(
        pixelValues: FloatBuffer, frames: Int, res: Int, batch: Int,
        onItem: ((Int) -> Unit)? = null, isCancelled: (() -> Boolean)? = null,
    ): Array<FloatArray> {
        (pixelValues as Buffer).rewind()
        val per = 3 * res * res
        val out = ArrayList<FloatArray>(frames)
        var f = 0
        while (f < frames) {
            if (isCancelled?.invoke() == true) throw RunCancelledException()
            val n = minOf(batch, frames - f)
            val chunk = ByteBuffer.allocateDirect(n * per * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            val base = f * per
            for (i in 0 until n * per) chunk.put(pixelValues.get(base + i))
            (chunk as Buffer).rewind()
            for (row in runEmbed(chunk, longArrayOf(n.toLong(), 3, res.toLong(), res.toLong()), n)) out.add(row)
            f += n
            onItem?.invoke(f)
        }
        return out.toTypedArray()
    }
```

- [ ] **Step 3: Add chunked primitives to `Engine`** (do NOT change `scoreFrames`, `flatten`, `packFrames`, `itemBatch`). Append inside the `Engine` class:

```kotlin
    /** Open the text tower, encode [labels] → L2-normalized embeddings, release the session.
     *  Mirrors scoreFrames' text-first/release ordering so the vision session is the only large one. */
    fun encodeTextEmbeddings(labels: List<String>): Array<FloatArray> =
        HfTokenizer.fromJson(File("$modelDir/${manifest.tokenizerFile}").readBytes()).use { tk ->
            val ids = labels.map { tk.encodePadded(it) }
            OrtTower.open("$modelDir/${manifest.textFile}", env, backend, config).use { t ->
                t.encodeText(flatten(ids), labels.size, manifest.maxLength, minOf(itemBatch(), labels.size))
            }
        }

    /** A chunk encoder bound to an open vision session. Encode one frame chunk → [chunk][D] embeddings. */
    inner class VisionEncoder internal constructor(private val tower: OrtTower) {
        private val effBatch = if (backend == Backend.CPU_EP) visionBatch else 1
        fun encodeChunk(
            bitmaps: List<Bitmap>, onItem: ((Int) -> Unit)? = null, isCancelled: (() -> Boolean)? = null,
        ): Array<FloatArray> {
            val pix = packFrames(bitmaps, manifest.resolution)
            return tower.encodeVision(pix, bitmaps.size, manifest.resolution,
                minOf(effBatch, bitmaps.size), onItem, isCancelled)
        }
    }

    /** Open the vision tower for the duration of [block], releasing it afterward. */
    fun <R> withVisionEncoder(block: (VisionEncoder) -> R): R =
        OrtTower.open("$modelDir/${manifest.visionFile}", env, backend, config).use { v -> block(VisionEncoder(v)) }
```

`packFrames` currently takes a `List<Bitmap>` — confirm it is the existing `private fun packFrames(bitmaps: List<Bitmap>, res: Int): FloatBuffer` (it is). No signature change needed.

- [ ] **Step 4: Verify the engine parity gate still holds** — run the existing instrumented `EndToEndParityTest` and `OrtBackendTest` on the device (they exercise `scoreFrames`/`encodeVision`; the additive changes must not alter results):

Run: `./gradlew :app:connectedDebugAndroidTest --tests "com.example.clipcc.engine.EndToEndParityTest" --tests "com.example.clipcc.engine.OrtBackendTest"`
Expected: PASS (cosine_max ≈ 9.09e-5, best_match_flips = 0 — unchanged from Plan 1).

- [ ] **Step 5: Commit (save).**

---

## Task 10: `UiBackend` + DTOs + `Classifier` seam + `RealClassifier`

**Files:**
- Create: `app/src/main/java/com/example/clipcc/ui/classify/UiBackend.kt`
- Create: `app/src/main/java/com/example/clipcc/ui/classify/ClassifyModels.kt`
- Create: `app/src/main/java/com/example/clipcc/ui/classify/Classifier.kt`
- Create: `app/src/main/java/com/example/clipcc/ui/classify/RealClassifier.kt`

- [ ] **Step 1: Create `UiBackend.kt`** (UI lanes → engine `Backend`; spec §2 backend mapping):

```kotlin
package com.example.clipcc.ui.classify

import com.example.clipcc.engine.Backend

enum class UiBackend(val label: String, val engine: Backend, val experimental: Boolean, val note: String) {
    CPU_XNNPACK("CPU·XNNPACK", Backend.CPU_XNNPACK, false, "per-frame; partial XNNPACK delegation"),
    CPU_EP("CPU·EP", Backend.CPU_EP, false, "batched ORT CPU EP"),
    NNAPI("NNAPI", Backend.NNAPI_DEFAULT, true, "experimental — 0% delegated on Tensor G2");
}
```

- [ ] **Step 2: Create `ClassifyModels.kt`** (frozen DTOs from spec §4.1; platform-light state — `videoUriString`, not `Uri`):

```kotlin
package com.example.clipcc.ui.classify

import android.graphics.Bitmap
import com.example.clipcc.data.ModelInfo
import com.example.clipcc.engine.AggregationResult
import com.example.clipcc.engine.ScoringPolicy

enum class AggMode { MEAN, MAX, TEMPORAL, CONTRAST }

data class TemporalOptions(
    val threshold: Double = ScoringPolicy.THRESHOLD,
    val gap: Double = ScoringPolicy.GAP_TOLERANCE,
    val minDuration: Double = ScoringPolicy.MIN_DURATION,
    val thresholdWasDefaulted: Boolean = true,
)
data class ContrastOptions(
    val threshold: Double = ScoringPolicy.CONTRAST_THRESHOLD,
    val reduce: String = ScoringPolicy.CONTRAST_REDUCE,
    val thresholdWasDefaulted: Boolean = true,
)

data class ClassifyRequest(
    val modelDir: String, val modelId: String, val backend: UiBackend, val videoUriString: String,
    val labels: List<String>, val posCount: Int, val mode: AggMode,
    val temporal: TemporalOptions, val contrast: ContrastOptions,
)
data class RunMeta(
    val modelId: String, val requestedBackend: UiBackend,
    val frameCount: Int, val elapsedMs: Long, val scoreSemantics: String,
)
data class RunResult(
    val result: AggregationResult, val thumbnails: Map<Int, Bitmap>,
    val timestamps: DoubleArray, val meta: RunMeta,
)

sealed interface RunState {
    data object Idle : RunState
    data class Running(val stage: Stage, val chunkDone: Int, val chunkTotal: Int) : RunState
    data object Cancelling : RunState
    data class Success(val result: RunResult) : RunState
    data class Error(val message: String) : RunState
    data object Cancelled : RunState
}
enum class Stage { LOADING_MODEL, ENCODING_TEXT, DECODING, ENCODING_VISION, AGGREGATING }

data class SetupState(
    val availableModels: List<ModelInfo> = emptyList(),
    val selectedModelId: String? = null,
    val backend: UiBackend = UiBackend.CPU_XNNPACK,
    val videoUriString: String? = null, val videoName: String? = null, val grantPersisted: Boolean = false,
    val positives: List<String> = ScoringPolicy.DEFAULT_LABELS,
    val negatives: List<String> = emptyList(),
    val mode: AggMode = AggMode.MEAN,
    val temporal: TemporalOptions = TemporalOptions(),
    val contrast: ContrastOptions = ContrastOptions(),
    val validationError: String? = null,
    val etaPerFrameMs: Long? = null,
) {
    val selectedModel: ModelInfo? get() = availableModels.firstOrNull { it.id == selectedModelId }
    val canRun: Boolean get() =
        selectedModel?.ready == true && videoUriString != null && validationError == null
}

data class ClassifyUiState(
    val setup: SetupState = SetupState(),
    val run: RunState = RunState.Idle,
) {
    val keepAwake: Boolean get() = run is RunState.Running || run is RunState.Cancelling
}
```

- [ ] **Step 3: Create `Classifier.kt`** (the seam):

```kotlin
package com.example.clipcc.ui.classify

/** Progress callback: (stage, chunkDone, chunkTotal). */
typealias ProgressSink = (Stage, Int, Int) -> Unit

interface Classifier {
    /** Runs one classification. [isCancelled] is polled cooperatively at checkpoints.
     *  Throws RunCancelledException (engine) on cancel; other throwables surface as errors. */
    suspend fun classify(req: ClassifyRequest, onProgress: ProgressSink, isCancelled: () -> Boolean): RunResult
}
```

- [ ] **Step 4: Create `RealClassifier.kt`** (the chunked, memory-bounded pipeline, spec §4.2):

```kotlin
package com.example.clipcc.ui.classify

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.media3.common.util.UnstableApi
import com.example.clipcc.engine.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

@UnstableApi
class RealClassifier(
    private val context: Context,
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment(),
) : Classifier {

    override suspend fun classify(
        req: ClassifyRequest, onProgress: ProgressSink, isCancelled: () -> Boolean,
    ): RunResult = withContext(Dispatchers.Default) {
        val start = System.currentTimeMillis()
        val manifest = ModelBundleManifest.parse(java.io.File("${req.modelDir}/manifest.json").readText())
        val backend = req.backend.engine
        val chunkSize = ScoringPolicy.visionChunkFor(req.modelId, backend)
        val engine = Engine(req.modelDir, manifest, env, backend, visionBatch = chunkSize)

        fun ckCancel() { if (isCancelled()) throw RunCancelledException() }

        onProgress(Stage.LOADING_MODEL, 0, 0); ckCancel()
        onProgress(Stage.ENCODING_TEXT, 0, 0)
        val txt = engine.encodeTextEmbeddings(req.labels); ckCancel()

        val sampler = FrameSampler(context)
        val uri = Uri.parse(req.videoUriString)
        val embeddings = ArrayList<FloatArray>()
        val timestamps = ArrayList<Double>()
        val thumbnails = HashMap<Int, Bitmap>()
        val chunk = ArrayList<Bitmap>(chunkSize)
        var chunkDone = 0
        // Upper bound on chunks for progress (frame count unknown until decode finishes).
        val chunkTotalGuess = (ScoringPolicy.MAX_FRAMES + chunkSize - 1) / chunkSize

        engine.withVisionEncoder { enc ->
            fun flushChunk() {
                if (chunk.isEmpty()) return
                onProgress(Stage.ENCODING_VISION, chunkDone, chunkTotalGuess)
                for (row in enc.encodeChunk(chunk, isCancelled = isCancelled)) embeddings.add(row)
                chunk.forEach { it.recycle() }   // release full bitmaps; thumbnails already kept
                chunk.clear()
                chunkDone++
            }
            fun decodePass(decodeUri: Uri) {
                onProgress(Stage.DECODING, 0, 0)
                sampler.sample(decodeUri, ScoringPolicy.FPS, ScoringPolicy.MAX_FRAMES) { frame ->
                    thumbnails[frame.index] = thumbnail(frame.bitmap)
                    timestamps.add(frame.timestampSec)
                    chunk.add(frame.bitmap)
                    if (chunk.size == chunkSize) flushChunk()
                    !isCancelled()
                }
            }
            try {
                decodePass(uri)
            } catch (c: RunCancelledException) {
                throw c
            } catch (t: Throwable) {
                // Decode failed at open/first-seek (some content:// providers Media3 can't seek directly).
                // Only safe to retry if nothing was produced yet (spec §9 / P2-6 copy-to-cache fallback).
                if (timestamps.isNotEmpty()) throw t
                decodePass(Uri.fromFile(copyToCache(uri)))
            }
            flushChunk()
        }
        ckCancel()

        onProgress(Stage.AGGREGATING, chunkDone, chunkDone)
        val matrices = Scoring.scoreMatrix(
            embeddings.toTypedArray(), txt, manifest.logitScale, manifest.logitBias)
        val ts = timestamps.toDoubleArray()
        val agg = aggregate(req, matrices, ts)

        RunResult(
            result = agg, thumbnails = thumbnails, timestamps = ts,
            meta = RunMeta(req.modelId, req.backend, ts.size,
                System.currentTimeMillis() - start, manifest.scoreSemantics),
        )
    }

    private fun aggregate(req: ClassifyRequest, m: ScoreMatrices, ts: DoubleArray): AggregationResult =
        when (req.mode) {
            AggMode.MEAN -> Scoring.aggregateMean(m.confidence, m.cosine, req.labels)
            AggMode.MAX -> Scoring.aggregateMax(m.confidence, m.cosine, req.labels, ts)
            AggMode.TEMPORAL -> Scoring.aggregateTemporal(
                m.confidence, m.cosine, req.labels,
                req.temporal.threshold, req.temporal.gap, req.temporal.minDuration,
                FrameTimeline(ts, ScoringPolicy.FPS, videoDuration(ts)),
                req.temporal.thresholdWasDefaulted)
            AggMode.CONTRAST -> Scoring.aggregateContrast(
                m.confidence, m.cosine, req.labels, req.posCount,
                req.contrast.reduce, req.contrast.threshold, req.contrast.thresholdWasDefaulted)
        }

    private fun videoDuration(ts: DoubleArray): Double =
        if (ts.isEmpty()) 0.0 else ts.last() + 1.0 / ScoringPolicy.FPS

    /** ~96px longest-edge thumbnail (small, bounded retention for MAX peaks). */
    private fun thumbnail(src: Bitmap): Bitmap {
        val max = 96
        val scale = max.toFloat() / maxOf(src.width, src.height)
        if (scale >= 1f) return src.copy(src.config ?: Bitmap.Config.ARGB_8888, false)
        return Bitmap.createScaledBitmap(src, (src.width * scale).toInt(), (src.height * scale).toInt(), true)
    }

    /** Copy a content:// video into app cache so Media3 can decode a seekable local file (SAF fallback). */
    private fun copyToCache(uri: Uri): java.io.File {
        val out = java.io.File(context.cacheDir, "clip_input.mp4")
        context.contentResolver.openInputStream(uri)?.use { input ->
            out.outputStream().use { input.copyTo(it) }
        } ?: error("cannot open $uri")
        return out
    }
}
```

> Note: `OrtEnvironment` import comes from `ai.onnxruntime.OrtEnvironment` (already used by the engine). Add `import ai.onnxruntime.OrtEnvironment` to the `engine.*` star import if the IDE flags it (the star import `com.example.clipcc.engine.*` does NOT cover `ai.onnxruntime`).

- [ ] **Step 5: Build** (no unit test here — exercised by the ViewModel test via a fake, and on-device by the smoke test).

Run: `./gradlew :app:assembleDebug`
Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 6: Commit (save).**

---

## Task 11: `ClassifyViewModel` + test

**Files:**
- Create: `app/src/main/java/com/example/clipcc/ui/classify/ClassifyViewModel.kt`
- Test: `app/src/test/java/com/example/clipcc/ui/classify/ClassifyViewModelTest.kt`

The ViewModel is plain-JVM testable: state is platform-light, the `Classifier` is injected (fake), and the run dispatcher is injectable.

- [ ] **Step 1: Write the failing test** `ClassifyViewModelTest.kt`:

```kotlin
package com.example.clipcc.ui.classify

import com.example.clipcc.data.ModelInfo
import com.example.clipcc.engine.AggregationResult
import com.example.clipcc.engine.BestMatch
import com.example.clipcc.engine.ScoreItem
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.*
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class ClassifyViewModelTest {
    private val dispatcher = StandardTestDispatcher()

    private val readyModel = ModelInfo(
        "siglip2-base-patch16-256", "Base", 256, "fp32",
        "siglip2_pairwise_sigmoid", ready = true, reason = null, dir = "/tmp/m")

    private fun okResult() = AggregationResult(
        scores = listOf(ScoreItem("a", 0.9, 0.1)), bestMatch = BestMatch("a", 0.9))

    private class FakeClassifier(
        val result: AggregationResult,
        val onClassify: suspend (ClassifyRequest, ProgressSink, () -> Boolean) -> Unit = { _, _, _ -> },
    ) : Classifier {
        override suspend fun classify(req: ClassifyRequest, onProgress: ProgressSink, isCancelled: () -> Boolean): RunResult {
            onClassify(req, onProgress, isCancelled)
            return RunResult(result, emptyMap(), DoubleArray(0),
                RunMeta(req.modelId, req.backend, 0, 0, "siglip2_pairwise_sigmoid"))
        }
    }

    @Before fun setUp() = Dispatchers.setMain(dispatcher)
    @After fun tearDown() = Dispatchers.resetMain()

    private fun vm(classifier: Classifier) = ClassifyViewModel(
        classifier = classifier, models = listOf(readyModel),
        benchmarkMsPerFrame = { _, _ -> 1202.0 }, runDispatcher = dispatcher,
    ).apply {
        selectModel(readyModel.id)
        setVideo("content://v/1", "clip.mp4", granted = true)
    }

    @Test fun run_disabled_until_model_and_video_set() {
        val v = ClassifyViewModel(FakeClassifier(okResult()), listOf(readyModel), { _, _ -> null }, dispatcher)
        assertFalse(v.state.value.setup.canRun)         // no video yet
        v.selectModel(readyModel.id); v.setVideo("content://v/1", "c.mp4", true)
        assertTrue(v.state.value.setup.canRun)
    }

    @Test fun successful_run_reaches_Success() = runTest(dispatcher) {
        val v = vm(FakeClassifier(okResult()))
        v.run()
        advanceUntilIdle()
        val run = v.state.value.run
        assertTrue(run is RunState.Success)
        assertEquals("a", (run as RunState.Success).result.result.bestMatch.label)
    }

    @Test fun classifier_error_reaches_Error() = runTest(dispatcher) {
        val v = vm(FakeClassifier(okResult()) { _, _, _ -> throw IllegalStateException("boom") })
        v.run(); advanceUntilIdle()
        assertTrue(v.state.value.run is RunState.Error)
        assertEquals("boom", (v.state.value.run as RunState.Error).message)
    }

    @Test fun cancel_reaches_Cancelled() = runTest(dispatcher) {
        val v = vm(FakeClassifier(okResult()) { _, _, isCancelled ->
            // simulate engine honoring the flag
            if (isCancelled()) throw com.example.clipcc.engine.RunCancelledException()
        })
        v.run()
        v.cancel()                 // sets flag before the classify body runs
        advanceUntilIdle()
        assertTrue(v.state.value.run is RunState.Cancelled)
    }

    @Test fun contrast_mode_sets_posCount_and_concatenates() = runTest(dispatcher) {
        var captured: ClassifyRequest? = null
        val v = vm(FakeClassifier(okResult()) { req, _, _ -> captured = req })
        v.setMode(AggMode.CONTRAST)
        v.setLabels(positives = listOf("yes"), negatives = listOf("no"))
        v.run(); advanceUntilIdle()
        assertEquals(listOf("yes", "no"), captured!!.labels)
        assertEquals(1, captured!!.posCount)
    }

    @Test fun eta_per_frame_comes_from_benchmark_lookup() {
        val v = vm(FakeClassifier(okResult()))
        assertEquals(1202L, v.state.value.setup.etaPerFrameMs)
    }

    @Test fun invalid_labels_block_run() {
        val v = vm(FakeClassifier(okResult()))
        v.setLabels(positives = listOf("dup", "dup"), negatives = emptyList())
        assertFalse(v.state.value.setup.canRun)
        assertEquals("Duplicate label: dup", v.state.value.setup.validationError)
    }

    @Test fun setup_restores_from_saved_state() {
        val handle = androidx.lifecycle.SavedStateHandle()
        val v1 = ClassifyViewModel(FakeClassifier(okResult()), listOf(readyModel), { _, _ -> null }, dispatcher, handle)
        v1.selectModel(readyModel.id); v1.setMode(AggMode.MAX)
        val v2 = ClassifyViewModel(FakeClassifier(okResult()), listOf(readyModel), { _, _ -> null }, dispatcher, handle)
        assertEquals(readyModel.id, v2.state.value.setup.selectedModelId)
        assertEquals(AggMode.MAX, v2.state.value.setup.mode)
    }
}
```

- [ ] **Step 2: Run — verify it fails.**

Run: `./gradlew testDebugUnitTest --tests "com.example.clipcc.ui.classify.ClassifyViewModelTest"`
Expected: FAIL (unresolved `ClassifyViewModel`).

- [ ] **Step 3: Create `ClassifyViewModel.kt`:**

```kotlin
package com.example.clipcc.ui.classify

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.clipcc.data.ModelInfo
import com.example.clipcc.engine.RunCancelledException
import com.example.clipcc.engine.ScoringPolicy
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean

/** [benchmarkMsPerFrame] looks up captured ms/frame for (modelId, backend) for the ETA; returns null
 *  if unknown. [runDispatcher] lets tests drive the run deterministically. */
class ClassifyViewModel(
    private val classifier: Classifier,
    private val models: List<ModelInfo>,
    private val benchmarkMsPerFrame: (String, UiBackend) -> Double?,
    private val runDispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val savedState: SavedStateHandle = SavedStateHandle(),
) : ViewModel() {

    private val _state = MutableStateFlow(ClassifyUiState())
    val state: StateFlow<ClassifyUiState> = _state.asStateFlow()

    private var job: Job? = null
    private val cancelFlag = AtomicBoolean(false)

    init {
        // Restore platform-light Setup across process death (spec §9 / P1-4).
        val restored = SetupState(
            availableModels = models,
            selectedModelId = savedState["selectedModelId"],
            backend = savedState.get<String>("backend")
                ?.let { runCatching { UiBackend.valueOf(it) }.getOrNull() } ?: UiBackend.CPU_XNNPACK,
            videoUriString = savedState["videoUriString"],
            videoName = savedState["videoName"],
            grantPersisted = savedState["grantPersisted"] ?: false,
            mode = savedState.get<String>("mode")
                ?.let { runCatching { AggMode.valueOf(it) }.getOrNull() } ?: AggMode.MEAN,
            positives = savedState.get<ArrayList<String>>("positives") ?: ArrayList(ScoringPolicy.DEFAULT_LABELS),
            negatives = savedState.get<ArrayList<String>>("negatives") ?: arrayListOf(),
        )
        _state.value = ClassifyUiState(setup = withDerived(restored))
    }

    private fun updateSetup(block: (SetupState) -> SetupState) {
        val next = withDerived(block(_state.value.setup))
        persist(next)
        _state.value = _state.value.copy(setup = next)
    }

    private fun persist(s: SetupState) {
        savedState["selectedModelId"] = s.selectedModelId
        savedState["backend"] = s.backend.name
        savedState["videoUriString"] = s.videoUriString
        savedState["videoName"] = s.videoName
        savedState["grantPersisted"] = s.grantPersisted
        savedState["mode"] = s.mode.name
        savedState["positives"] = ArrayList(s.positives)
        savedState["negatives"] = ArrayList(s.negatives)
    }

    /** Recompute validation + ETA whenever setup changes. */
    private fun withDerived(s: SetupState): SetupState {
        val check = LabelValidation.validate(s.positives, s.negatives, s.mode == AggMode.CONTRAST)
        val eta = s.selectedModel?.let { benchmarkMsPerFrame(it.id, s.backend)?.toLong() }
        return s.copy(validationError = check.error, etaPerFrameMs = eta)
    }

    fun selectModel(id: String) = updateSetup { it.copy(selectedModelId = id) }
    fun setBackend(b: UiBackend) = updateSetup { it.copy(backend = b) }
    fun setVideo(uriString: String, name: String, granted: Boolean) =
        updateSetup { it.copy(videoUriString = uriString, videoName = name, grantPersisted = granted) }
    fun setMode(m: AggMode) = updateSetup { it.copy(mode = m) }
    fun setLabels(positives: List<String>, negatives: List<String>) =
        updateSetup { it.copy(positives = positives, negatives = negatives) }
    fun setTemporal(o: TemporalOptions) = updateSetup { it.copy(temporal = o) }
    fun setContrast(o: ContrastOptions) = updateSetup { it.copy(contrast = o) }

    fun run() {
        val s = _state.value.setup
        val model = s.selectedModel ?: return
        val check = LabelValidation.validate(s.positives, s.negatives, s.mode == AggMode.CONTRAST)
        if (!s.canRun || check.error != null) return
        val req = ClassifyRequest(
            modelDir = model.dir, modelId = model.id, backend = s.backend,
            videoUriString = s.videoUriString!!, labels = check.cleaned, posCount = check.posCount,
            mode = s.mode, temporal = s.temporal, contrast = s.contrast,
        )
        cancelFlag.set(false)
        _state.value = _state.value.copy(run = RunState.Running(Stage.LOADING_MODEL, 0, 0))
        job = viewModelScope.launch(runDispatcher) {
            try {
                val result = classifier.classify(
                    req,
                    onProgress = { stage, done, total -> postProgress(stage, done, total) },
                    isCancelled = { cancelFlag.get() },
                )
                _state.value = _state.value.copy(run = RunState.Success(result))
            } catch (c: RunCancelledException) {
                _state.value = _state.value.copy(run = RunState.Cancelled)
            } catch (t: Throwable) {
                _state.value = _state.value.copy(run = RunState.Error(t.message ?: t.javaClass.simpleName))
            }
        }
    }

    private fun postProgress(stage: Stage, done: Int, total: Int) {
        if (cancelFlag.get()) { _state.value = _state.value.copy(run = RunState.Cancelling); return }
        _state.value = _state.value.copy(run = RunState.Running(stage, done, total))
    }

    fun cancel() {
        cancelFlag.set(true)
        if (_state.value.run is RunState.Running) _state.value = _state.value.copy(run = RunState.Cancelling)
    }

    fun reset() { _state.value = _state.value.copy(run = RunState.Idle) }
}
```

- [ ] **Step 4: Run — verify it passes.**

Run: `./gradlew testDebugUnitTest --tests "com.example.clipcc.ui.classify.ClassifyViewModelTest"`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit (save).**

---

## Task 12: Canvas charts (`BarChart`, `TimelineChart`)

**Files:**
- Create: `app/src/main/java/com/example/clipcc/ui/charts/BarChart.kt`
- Create: `app/src/main/java/com/example/clipcc/ui/charts/TimelineChart.kt`

Pure rendering (verified by build + manual screenshots; logic was unit-tested in `ChartData`). Each carries a `contentDescription` for accessibility.

- [ ] **Step 1: Create `BarChart.kt`:**

```kotlin
package com.example.clipcc.ui.charts

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp

/** Grouped bars on a single declared scale. [zeroAtCenter] draws a symmetric axis (for signed cosine);
 *  otherwise bars rise from the bottom on [0, max]. [thresholdLine] draws an optional guide (e.g. 0.5). */
@Composable
fun BarChart(
    bars: List<Bar>, max: Float, barColor: Color,
    modifier: Modifier = Modifier, zeroAtCenter: Boolean = false, thresholdLine: Float? = null,
) {
    val desc = bars.joinToString("; ") { "${it.label} ${"%.3f".format(it.value)}" }
    Canvas(
        modifier.fillMaxWidth().height(160.dp).semantics { contentDescription = "Bar chart: $desc" }
    ) {
        val n = bars.size.coerceAtLeast(1)
        val slot = size.width / n
        val barW = slot * 0.6f
        val zeroY = if (zeroAtCenter) size.height / 2f else size.height
        fun y(v: Float) = zeroY - (v / max) * (if (zeroAtCenter) size.height / 2f else size.height)

        thresholdLine?.let { t ->
            val ty = y(t)
            drawLine(Color.Gray, androidx.compose.ui.geometry.Offset(0f, ty),
                androidx.compose.ui.geometry.Offset(size.width, ty), strokeWidth = 2f)
        }
        bars.forEachIndexed { i, b ->
            val cx = i * slot + slot / 2f
            val top = minOf(zeroY, y(b.value))
            val h = kotlin.math.abs(zeroY - y(b.value))
            drawRect(barColor,
                topLeft = androidx.compose.ui.geometry.Offset(cx - barW / 2f, top),
                size = androidx.compose.ui.geometry.Size(barW, h))
        }
        if (zeroAtCenter) drawLine(Color.DarkGray,
            androidx.compose.ui.geometry.Offset(0f, zeroY),
            androidx.compose.ui.geometry.Offset(size.width, zeroY), strokeWidth = 1f)
    }
}
```

- [ ] **Step 2: Create `TimelineChart.kt`** (temporal: one polyline per label + threshold + shaded segments):

```kotlin
package com.example.clipcc.ui.charts

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp

data class TimelineSeries(val label: String, val color: Color, val values: List<Float>)
data class TimelineBand(val color: Color, val startFrac: Float, val endFrac: Float)

/** Score-over-time chart on [0,1] y, frame-index x. [bands] are shaded segments (fractions of width). */
@Composable
fun TimelineChart(
    series: List<TimelineSeries>, threshold: Float, bands: List<TimelineBand>,
    modifier: Modifier = Modifier,
) {
    val desc = "Timeline of ${series.size} labels over ${series.firstOrNull()?.values?.size ?: 0} frames"
    Canvas(modifier.fillMaxWidth().height(200.dp).semantics { contentDescription = desc }) {
        fun x(i: Int, n: Int) = if (n <= 1) 0f else size.width * i / (n - 1)
        fun y(v: Float) = size.height * (1f - v.coerceIn(0f, 1f))

        bands.forEach { b ->
            drawRect(b.color.copy(alpha = 0.15f),
                topLeft = Offset(size.width * b.startFrac, 0f),
                size = androidx.compose.ui.geometry.Size(size.width * (b.endFrac - b.startFrac), size.height))
        }
        val dashed = PathEffect.dashPathEffect(floatArrayOf(12f, 8f))
        drawLine(Color.Gray, Offset(0f, y(threshold)), Offset(size.width, y(threshold)),
            strokeWidth = 2f, pathEffect = dashed)
        series.forEach { s ->
            val path = Path()
            s.values.forEachIndexed { i, v ->
                val px = x(i, s.values.size); val py = y(v)
                if (i == 0) path.moveTo(px, py) else path.lineTo(px, py)
            }
            drawPath(path, s.color, style = Stroke(width = 4f))
        }
    }
}
```

- [ ] **Step 3: Build.**

Run: `./gradlew :app:assembleDebug`
Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 4: Commit (save).**

---

## Task 13: `SetupCard` composable

**Files:**
- Create: `app/src/main/java/com/example/clipcc/ui/classify/SetupCard.kt`

- [ ] **Step 1: Create `SetupCard.kt`** (model dropdown, backend segmented control, video picker, label editor, mode + options, Run/Cancel, ETA):

```kotlin
package com.example.clipcc.ui.classify

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SetupCard(state: SetupState, vm: ClassifyViewModel, running: Boolean) {
    val ctx = LocalContext.current
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            val granted = try {
                ctx.contentResolver.takePersistableUriPermission(
                    uri, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                true
            } catch (t: Throwable) { false }
            val name = uri.lastPathSegment ?: "video"
            vm.setVideo(uri.toString(), name, granted)
        }
    }

    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        // Model dropdown
        var modelMenu by remember { mutableStateOf(false) }
        ExposedDropdownMenuBox(expanded = modelMenu, onExpandedChange = { modelMenu = it }) {
            OutlinedTextField(
                value = state.selectedModel?.let { "${it.displayName} · ${it.resolution}px · ${it.precision}" }
                    ?: "Select model",
                onValueChange = {}, readOnly = true, label = { Text("Model") },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(modelMenu) },
                modifier = Modifier.menuAnchor().fillMaxWidth(), enabled = !running,
            )
            ExposedDropdownMenu(expanded = modelMenu, onDismissRequest = { modelMenu = false }) {
                if (state.availableModels.isEmpty()) {
                    DropdownMenuItem(text = { Text("No models provisioned — adb push to files/models/") },
                        onClick = {}, enabled = false)
                }
                state.availableModels.forEach { m ->
                    DropdownMenuItem(
                        text = { Text("${m.displayName} · ${m.resolution}px · ${m.precision}" +
                            if (m.ready) "" else "  (${m.reason})") },
                        enabled = m.ready,
                        onClick = { vm.selectModel(m.id); modelMenu = false })
                }
            }
        }

        // Backend segmented control
        Text("Backend", style = MaterialTheme.typography.labelMedium)
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            UiBackend.entries.forEachIndexed { i, b ->
                SegmentedButton(
                    selected = state.backend == b, onClick = { vm.setBackend(b) },
                    enabled = !running,
                    shape = SegmentedButtonDefaults.itemShape(i, UiBackend.entries.size),
                ) { Text(b.label) }
            }
        }
        if (state.backend.experimental) {
            Text(state.backend.note, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
        }

        // Video picker
        Button(onClick = { picker.launch(arrayOf("video/*")) }, enabled = !running) {
            Text(state.videoName?.let { "Video: $it" } ?: "Pick video")
        }

        // Mode selector
        Text("Mode", style = MaterialTheme.typography.labelMedium)
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            AggMode.entries.forEachIndexed { i, m ->
                SegmentedButton(
                    selected = state.mode == m, onClick = { vm.setMode(m) }, enabled = !running,
                    shape = SegmentedButtonDefaults.itemShape(i, AggMode.entries.size),
                ) { Text(m.name.lowercase()) }
            }
        }

        // Label editor (contrast → two groups)
        LabelEditor(state, vm, running)

        // Mode options
        when (state.mode) {
            AggMode.TEMPORAL -> TemporalOptionsEditor(state.temporal, vm, running)
            AggMode.CONTRAST -> ContrastOptionsEditor(state.contrast, vm, running)
            else -> {}
        }

        state.validationError?.let { Text(it, color = MaterialTheme.colorScheme.error) }
        state.etaPerFrameMs?.let { Text("≈ ${it} ms/frame on this model+backend (captured estimate)",
            style = MaterialTheme.typography.bodySmall) }
    }
}

@Composable
private fun LabelEditor(state: SetupState, vm: ClassifyViewModel, running: Boolean) {
    if (state.mode == AggMode.CONTRAST) {
        Text("Positive labels", style = MaterialTheme.typography.labelMedium)
        EditableList(state.positives, running) { vm.setLabels(it, state.negatives) }
        Text("Negative labels", style = MaterialTheme.typography.labelMedium)
        EditableList(state.negatives, running) { vm.setLabels(state.positives, it) }
    } else {
        Text("Labels", style = MaterialTheme.typography.labelMedium)
        EditableList(state.positives, running) { vm.setLabels(it, state.negatives) }
    }
}

@Composable
private fun EditableList(items: List<String>, running: Boolean, onChange: (List<String>) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        items.forEachIndexed { i, v ->
            Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                OutlinedTextField(value = v, onValueChange = { nv ->
                    onChange(items.toMutableList().also { it[i] = nv })
                }, modifier = Modifier.weight(1f), enabled = !running, singleLine = true)
                TextButton(onClick = { onChange(items.toMutableList().also { it.removeAt(i) }) },
                    enabled = !running) { Text("×") }
            }
        }
        TextButton(onClick = { onChange(items + "") }, enabled = !running) { Text("+ Add label") }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun TemporalOptionsEditor(o: TemporalOptions, vm: ClassifyViewModel, running: Boolean) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        NumberField("Threshold", o.threshold, running) { vm.setTemporal(o.copy(threshold = it, thresholdWasDefaulted = false)) }
        NumberField("Gap tolerance (s)", o.gap, running) { vm.setTemporal(o.copy(gap = it)) }
        NumberField("Min duration (s)", o.minDuration, running) { vm.setTemporal(o.copy(minDuration = it)) }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ContrastOptionsEditor(o: ContrastOptions, vm: ClassifyViewModel, running: Boolean) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        var menu by remember { mutableStateOf(false) }
        ExposedDropdownMenuBox(expanded = menu, onExpandedChange = { menu = it }) {
            OutlinedTextField(value = o.reduce, onValueChange = {}, readOnly = true,
                label = { Text("Reduce") }, modifier = Modifier.menuAnchor().fillMaxWidth(), enabled = !running)
            ExposedDropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                com.example.clipcc.engine.ScoringPolicy.CONTRAST_REDUCE_MODES.forEach { r ->
                    DropdownMenuItem(text = { Text(r) }, onClick = { vm.setContrast(o.copy(reduce = r)); menu = false })
                }
            }
        }
        NumberField("Threshold", o.threshold, running) { vm.setContrast(o.copy(threshold = it, thresholdWasDefaulted = false)) }
    }
}

@Composable
private fun NumberField(label: String, value: Double, running: Boolean, onChange: (Double) -> Unit) {
    OutlinedTextField(
        value = value.toString(),
        onValueChange = { it.toDoubleOrNull()?.let(onChange) },
        label = { Text(label) }, enabled = !running, singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        modifier = Modifier.fillMaxWidth(),
    )
}
```

- [ ] **Step 2: Build.**

Run: `./gradlew :app:assembleDebug`
Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 3: Commit (save).**

---

## Task 14: `RunStatus` + `ResultsSection` + `ModeExtras`

**Files:**
- Create: `app/src/main/java/com/example/clipcc/ui/classify/RunStatus.kt`
- Create: `app/src/main/java/com/example/clipcc/ui/classify/ResultsSection.kt`
- Create: `app/src/main/java/com/example/clipcc/ui/classify/ModeExtras.kt`

- [ ] **Step 1: Create `RunStatus.kt`:**

```kotlin
package com.example.clipcc.ui.classify

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun RunStatus(run: RunState, onCancel: () -> Unit) {
    when (run) {
        is RunState.Running -> Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            val label = when (run.stage) {
                Stage.LOADING_MODEL -> "Loading model…"
                Stage.ENCODING_TEXT -> "Encoding labels…"
                Stage.DECODING -> "Decoding video…"
                Stage.ENCODING_VISION -> "Encoding frames (chunk ${run.chunkDone}/${run.chunkTotal})…"
                Stage.AGGREGATING -> "Aggregating…"
            }
            Text(label)
            if (run.stage == Stage.ENCODING_VISION && run.chunkTotal > 0)
                LinearProgressIndicator(progress = { run.chunkDone.toFloat() / run.chunkTotal },
                    modifier = Modifier.fillMaxWidth())
            else LinearProgressIndicator(Modifier.fillMaxWidth())
            OutlinedButton(onClick = onCancel) { Text("Cancel") }
        }
        is RunState.Cancelling -> Row(Modifier.padding(16.dp)) {
            CircularProgressIndicator(); Spacer(Modifier.width(12.dp)); Text("Cancelling…")
        }
        else -> {}
    }
}
```

- [ ] **Step 2: Create `ResultsSection.kt`:**

```kotlin
package com.example.clipcc.ui.classify

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.clipcc.ui.charts.BarChart
import com.example.clipcc.ui.charts.ChartData

@Composable
fun ResultsSection(success: RunState.Success) {
    val r = success.result
    val agg = r.result
    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        ElevatedCard {
            Column(Modifier.padding(16.dp)) {
                Text("Best match", style = MaterialTheme.typography.labelMedium)
                Text(agg.bestMatch.label, style = MaterialTheme.typography.headlineSmall)
                Text("confidence ${"%.3f".format(agg.bestMatch.confidence)}")
                Text("${r.meta.modelId} · ${r.meta.requestedBackend.label} · ${r.meta.frameCount} frames · " +
                    "${r.meta.elapsedMs} ms · ${r.meta.scoreSemantics}",
                    style = MaterialTheme.typography.bodySmall)
                Text("live node coverage not profiled — see Benchmark",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.outline)
            }
        }

        Text("Confidence", style = MaterialTheme.typography.labelLarge)
        BarChart(ChartData.confidenceBars(agg.scores), max = ChartData.UNIT_MAX,
            barColor = MaterialTheme.colorScheme.primary, thresholdLine = ChartData.UNIT_MAX * 0.5f)

        Text("Raw similarity (cosine)", style = MaterialTheme.typography.labelLarge)
        val cos = ChartData.cosineBars(agg.scores)
        BarChart(cos, max = ChartData.symmetricMax(cos.map { it.value }),
            barColor = Color(0xFF7E57C2), zeroAtCenter = true)

        // value table (also the screen-reader text)
        agg.scores.forEach {
            Text("${it.label}: conf ${"%.3f".format(it.confidence)}, cos ${"%.3f".format(it.rawSimilarity)}",
                style = MaterialTheme.typography.bodySmall)
        }

        ModeExtras(r)
    }
}
```

- [ ] **Step 3: Create `ModeExtras.kt`:**

```kotlin
package com.example.clipcc.ui.classify

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import com.example.clipcc.engine.ScoreItem
import com.example.clipcc.ui.charts.TimelineChart
import com.example.clipcc.ui.charts.TimelineBand
import com.example.clipcc.ui.charts.TimelineSeries

@Composable
fun ModeExtras(r: RunResult) {
    val agg = r.result
    when {
        agg.temporal != null -> TemporalExtras(r)
        agg.contrast != null -> ContrastExtras(agg.contrast!!)
        // MAX is detectable by per-label peakFrameIndex being set
        agg.scores.any { it.peakFrameIndex != null } -> MaxExtras(r)
        else -> {}
    }
}

@Composable
private fun MaxExtras(r: RunResult) {
    Text("Peak frames", style = MaterialTheme.typography.labelLarge)
    r.result.scores.forEach { s: ScoreItem ->
        Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            s.peakFrameIndex?.let { idx ->
                r.thumbnails[idx]?.let { Image(it.asImageBitmap(), contentDescription = "${s.label} peak frame",
                    modifier = Modifier.size(72.dp)) }
            }
            Text("${s.label} @ ${"%.1f".format(s.approxTimestampSeconds ?: 0.0)}s")
        }
    }
}

@Composable
private fun TemporalExtras(r: RunResult) {
    val t = r.result.temporal!!
    val labels = r.result.scores.map { it.label }
    val palette = listOf(Color(0xFF1565C0), Color(0xFFEF6C00), Color(0xFF2E7D32), Color(0xFFC62828))
    val series = labels.mapIndexed { i, lbl ->
        TimelineSeries(lbl, palette[i % palette.size],
            t.timeline.map { fr -> (fr.scores[lbl] ?: 0.0).toFloat() })
    }
    val nFrames = t.timeline.size.coerceAtLeast(1)
    val bands = t.segments.map { seg ->
        val li = labels.indexOf(seg.label).coerceAtLeast(0)
        // approximate band by start/end timestamp fraction of total
        val total = (t.timeline.lastOrNull()?.timestamp ?: 1.0).coerceAtLeast(1e-6)
        TimelineBand(palette[li % palette.size], (seg.startTime / total).toFloat(), (seg.endTime / total).toFloat())
    }
    Text("Timeline", style = MaterialTheme.typography.labelLarge)
    TimelineChart(series, threshold = t.effectiveThreshold.toFloat(), bands = bands)
    Text("Segments", style = MaterialTheme.typography.labelLarge)
    t.segments.forEach { seg ->
        Text("${seg.label}: ${"%.1f".format(seg.startTime)}–${"%.1f".format(seg.endTime)}s " +
            "(${"%.1f".format(seg.duration)}s, peak ${"%.3f".format(seg.peakConfidence)})",
            style = MaterialTheme.typography.bodySmall)
    }
    Text("Label summaries", style = MaterialTheme.typography.labelLarge)
    t.labelSummaries.forEach { ls ->
        Text("${ls.label}: ${ls.segmentCount} seg, active ${"%.1f".format(ls.totalActiveDuration)}s, " +
            "DWC ${"%.3f".format(ls.durationWeightedConfidence)}", style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun ContrastExtras(c: com.example.clipcc.engine.ContrastResult) {
    val color = when (c.verdict) {
        "positive" -> Color(0xFF2E7D32); "negative" -> Color(0xFFC62828); else -> Color(0xFF757575)
    }
    Surface(color = color) {
        Text("  Verdict: ${c.verdict.uppercase()}  (margin ${"%.3f".format(c.difference)})  ",
            modifier = Modifier.padding(8.dp), color = Color.White)
    }
    Text("Positive group mean ${"%.3f".format(c.positive.meanGroupScore)} · " +
        "Negative group mean ${"%.3f".format(c.negative.meanGroupScore)}")
    c.dominantLabel?.let { Text("Dominant: $it") }
    Text("threshold ${c.threshold} (${c.thresholdSource}), ${c.calibrationStatus}",
        style = MaterialTheme.typography.bodySmall)
}
```

> Note: MAX detection uses `peakFrameIndex != null` because `aggregateMax` sets it while `aggregateMean` does not (verify against `Scoring.kt`: `aggregateMax` populates `peakFrameIndex`/`approxTimestampSeconds`, `aggregateMean` leaves them null). Temporal/contrast set their own `agg.temporal`/`agg.contrast`.

- [ ] **Step 4: Build.**

Run: `./gradlew :app:assembleDebug`
Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 5: Commit (save).**

---

## Task 15: `BenchmarkScreen` composable

**Files:**
- Create: `app/src/main/java/com/example/clipcc/ui/benchmark/BenchmarkScreen.kt`

- [ ] **Step 1: Create `BenchmarkScreen.kt`** (loads the asset once; CPU timed rows + NNAPI capability-only rows + provenance header):

```kotlin
package com.example.clipcc.ui.benchmark

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp

private const val HEADER =
    "Pixel 7a · Tensor G2 · median-of-3, 1 warm-up discarded · CPU-only · Media3 1.10.1 · captured 2026-06-03"

@Composable
fun BenchmarkScreen() {
    val ctx = LocalContext.current
    val groups = remember {
        BenchmarkData.parse(ctx.assets.open("phase2-benchmark-result.json").bufferedReader().use { it.readText() })
    }
    Column(Modifier.padding(16.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(HEADER, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.outline)
        groups.forEach { g ->
            ElevatedCard {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(g.modelId, style = MaterialTheme.typography.titleMedium)
                    g.timed.forEach { t ->
                        Text("${t.backend}: ${"%.0f".format(t.msPerFrame)} ms/frame · ${"%.3f".format(t.fps)} fps · " +
                            "load ${t.loadMs} ms" +
                            (t.visionDelegatedPct?.let { " · ${"%.1f".format(it)}% delegated" } ?: ""),
                            style = MaterialTheme.typography.bodyMedium)
                    }
                    g.capabilityOnly.forEach { c ->
                        Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            AssistChip(onClick = {}, label = { Text("experimental") }, enabled = false)
                            Text("${c.backend}: not timed · ${"%.0f".format(c.visionDelegatedPct)}% delegated",
                                style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 2: Build.**

Run: `./gradlew :app:assembleDebug`
Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 3: Commit (save).**

---

## Task 16: App wiring (`ClipCCApp` real, inline saved-state factory, MainActivity keep-screen-on)

**Files:**
- Modify: `app/src/main/java/com/example/clipcc/ui/app/ClipCCApp.kt`
- Modify: `app/src/main/java/com/example/clipcc/MainActivity.kt`

The ViewModel is built with an inline `viewModelFactory` so `createSavedStateHandle()` provides a **real** `SavedStateHandle` (process-death restore actually works — a standalone `ViewModelProvider.Factory` would hand it a throwaway handle).

- [ ] **Step 1: Replace `ClipCCApp.kt`** with the real wiring (model scan, benchmark ETA lookup, saved-state ViewModel, padding for tabs):

```kotlin
package com.example.clipcc.ui.app

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.createSavedStateHandle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import androidx.media3.common.util.UnstableApi
import com.example.clipcc.data.ModelRepository
import com.example.clipcc.ui.benchmark.BenchmarkData
import com.example.clipcc.ui.benchmark.BenchmarkScreen
import com.example.clipcc.ui.classify.*
import java.io.File

@OptIn(UnstableApi::class)
@Composable
fun ClipCCApp(onKeepAwake: (Boolean) -> Unit) {
    var tab by remember { mutableIntStateOf(0) }
    val titles = listOf("Classify", "Benchmark")
    Scaffold(modifier = Modifier.fillMaxSize()) { pad ->
        Column(Modifier.padding(pad)) {
            TabRow(selectedTabIndex = tab) {
                titles.forEachIndexed { i, t ->
                    Tab(selected = tab == i, onClick = { tab = i }, text = { Text(t) })
                }
            }
            if (tab == 0) ClassifyTab(onKeepAwake) else BenchmarkScreen()
        }
    }
}

@OptIn(UnstableApi::class)
@Composable
private fun ClassifyTab(onKeepAwake: (Boolean) -> Unit) {
    val appCtx = LocalContext.current.applicationContext
    val vm: ClassifyViewModel = viewModel(factory = viewModelFactory {
        initializer {
            val models = ModelRepository(File(appCtx.filesDir, "models")).scan()
            val groups = BenchmarkData.parse(
                appCtx.assets.open("phase2-benchmark-result.json").bufferedReader().use { it.readText() })
            val lookup: (String, UiBackend) -> Double? = { id, backend ->
                val g = groups.firstOrNull { it.modelId == id }
                (g?.timed?.firstOrNull { it.backend == backend.engine.name } ?: g?.timed?.firstOrNull())?.msPerFrame
            }
            ClassifyViewModel(RealClassifier(appCtx), models, lookup, savedState = createSavedStateHandle())
        }
    })
    val ui by vm.state.collectAsStateWithLifecycle()
    LaunchedEffect(ui.keepAwake) { onKeepAwake(ui.keepAwake) }
    val running = ui.run is RunState.Running || ui.run is RunState.Cancelling

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        SetupCard(ui.setup, vm, running)
        Button(onClick = { vm.run() }, enabled = ui.setup.canRun && !running,
            modifier = Modifier.padding(horizontal = 16.dp)) { Text("Run") }
        RunStatus(ui.run, onCancel = { vm.cancel() })
        (ui.run as? RunState.Success)?.let { ResultsSection(it) }
        (ui.run as? RunState.Error)?.let {
            Column(Modifier.padding(16.dp)) {
                Text("Error: ${it.message}", color = MaterialTheme.colorScheme.error)
                OutlinedButton(onClick = { vm.reset() }) { Text("Back") }
            }
        }
    }
}
```

- [ ] **Step 2: Update `MainActivity.kt`** to apply keep-screen-on:

```kotlin
package com.example.clipcc

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.media3.common.util.UnstableApi
import com.example.clipcc.ui.app.ClipCCApp
import com.example.clipcc.ui.theme.ClipCCTheme

@UnstableApi
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ClipCCTheme {
                ClipCCApp(onKeepAwake = { keep ->
                    if (keep) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                })
            }
        }
    }
}
```

- [ ] **Step 3: Build + run the full JVM unit suite** (everything must be green):

Run: `./gradlew testDebugUnitTest`
Expected: PASS — includes Plan-1/2 (`Resampler`, `Manifest`, `Scoring`) + the 7 new JVM classes (`ScoringPolicy`, `ManifestExtension`, `ModelRepository`, `LabelValidation`, `ChartData`, `BenchmarkData`, `ClassifyViewModel`).

- [ ] **Step 4: Install on the device and smoke it by hand.**

Run: `./gradlew :app:installDebug` then launch the app; confirm both tabs render and the Benchmark tab shows the 4 model groups.
Expected: app launches; Benchmark populated.

- [ ] **Step 5: Commit (save).**

---

## Task 17: Instrumented end-to-end smoke + manual screenshots (gate)

**Files:**
- Create: `app/src/androidTest/java/com/example/clipcc/ClassifyEndToEndSmokeTest.kt`

Provisions base-256 (already on the device at `/data/local/tmp/clipcc_models/...` from Plans 1–2 → copy into the app's `files/models/`), runs the real classifier on the test clip, asserts a populated result + bounded thumbnail retention.

- [ ] **Step 1: Provision the model + clip into app storage** (debuggable `run-as`):

```bash
ADB=~/Library/Android/sdk/platform-tools/adb
$ADB shell run-as com.example.clipcc mkdir -p files/models
$ADB shell run-as com.example.clipcc sh -c 'cp -r /data/local/tmp/clipcc_models/siglip2-base-patch16-256 files/models/'
$ADB shell run-as com.example.clipcc sh -c 'cp /data/local/tmp/clipcc_bench/test.mp4 files/test.mp4'
```

- [ ] **Step 2: Write the instrumented smoke test** `ClassifyEndToEndSmokeTest.kt`:

```kotlin
package com.example.clipcc

import androidx.test.platform.app.InstrumentationRegistry
import androidx.media3.common.util.UnstableApi
import com.example.clipcc.ui.classify.*
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

@UnstableApi
class ClassifyEndToEndSmokeTest {
    @Test fun real_clip_runs_and_populates_mean_result() = runBlocking {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(ctx.filesDir, "models/siglip2-base-patch16-256")
        assertTrue("provision base-256 into files/models first", dir.exists())
        val video = File(ctx.filesDir, "test.mp4")
        assertTrue("provision test.mp4 first", video.exists())

        val req = ClassifyRequest(
            modelDir = dir.absolutePath, modelId = "siglip2-base-patch16-256",
            backend = UiBackend.CPU_XNNPACK, videoUriString = android.net.Uri.fromFile(video).toString(),
            labels = ScoringPolicyDefaults(), posCount = 0, mode = AggMode.MEAN,
            temporal = TemporalOptions(), contrast = ContrastOptions(),
        )
        val result = RealClassifier(ctx).classify(req, onProgress = { _, _, _ -> }, isCancelled = { false })

        assertTrue(result.result.scores.isNotEmpty())
        assertTrue(result.result.bestMatch.confidence in 0.0..1.0)
        assertTrue("thumbnails retained per frame", result.thumbnails.isNotEmpty())
        assertTrue("frames decoded", result.meta.frameCount > 0)
    }

    private fun ScoringPolicyDefaults() =
        com.example.clipcc.engine.ScoringPolicy.DEFAULT_LABELS
}
```

- [ ] **Step 3: Run the smoke test on the device.**

Run: `./gradlew :app:connectedDebugAndroidTest --tests "com.example.clipcc.ClassifyEndToEndSmokeTest"`
Expected: PASS (non-empty scores, best match in [0,1], thumbnails non-empty, frameCount > 0).

- [ ] **Step 4: Manual acceptance — screenshots** of each mode + the benchmark panel:

```bash
ADB=~/Library/Android/sdk/platform-tools/adb
# in the app: pick base-256, pick a video, run each mode; capture:
$ADB exec-out screencap -p > /tmp/clipcc_mean.png
# repeat for MAX / TEMPORAL / CONTRAST / Benchmark tab
```
Expected: best-match card + both bar charts render; MAX shows peak thumbnails; TEMPORAL shows the timeline + segments; CONTRAST shows the verdict banner; Benchmark shows 4 model groups with CPU timed rows + NNAPI experimental capability rows.

- [ ] **Step 5: Run the FULL instrumented suite** to confirm no Plan-1/2 regressions (the engine touches were additive):

Run: `./gradlew :app:connectedDebugAndroidTest`
Expected: all green (Tokenizer/Preprocess/OrtBackend/EndToEndParity/BackendCapability/FrameSampler/Benchmark* + the new smoke). Note: the Phase-0 spike tests may still pad the run (Plan-1 housekeeping) — out of scope here.

- [ ] **Step 6: Write the phase report** `docs/superpowers/plans/phase3-report.md` (host repo, git) mirroring Plans 1–2: DoD table, files touched, test evidence, screenshots list, deviations. Commit it in the host repo.

---

## Self-review notes (for the executor)

- **Engine purity:** Tasks 8–9 are additive. The gates are the **existing** `FrameSamplerTest` (Task 8) and `EndToEndParityTest`/`OrtBackendTest` (Task 9) — if either regresses, the touch was not additive; revert and re-do.
- **Type consistency:** DTO names are fixed in Task 10 (`ClassifyRequest`, `RunResult`, `RunMeta`, `RunState`, `Stage`, `SetupState`, `UiBackend`, `AggMode`, `TemporalOptions`, `ContrastOptions`) and consumed unchanged in Tasks 11–16. `ModelInfo` is fixed in Task 4. `Bar`/`TimelineSeries`/`TimelineBand` in Tasks 6/12. `TimedRow`/`CapabilityRow`/`ModelGroup` in Task 7.
- **Platform-light rule:** only `RealClassifier` (Task 10), `SetupCard` (Task 13), `ModeExtras` thumbnails (Task 14), and the factory/activity (Task 16) touch Android types (`Uri`/`Bitmap`/`Context`). `ClassifyViewModel` and all `*Test` JVM classes do not — keep it that way so the ViewModel test stays Robolectric-free.
- **Aggregation dispatch** in `RealClassifier.aggregate` matches the engine signatures in `Scoring.kt` exactly (mean/max/temporal/contrast). MAX is detected in `ModeExtras` via `peakFrameIndex != null`.
- **Deviation from spec §4.1:** `ScoringPolicy.visionChunkFor` takes engine `Backend` (not `UiBackend`) to keep `engine` UI-agnostic; mapping happens in `RealClassifier`. Functionally identical.
- **Spec §9 coverage:** `SavedStateHandle` restore (P1-4) is in `ClassifyViewModel` (Task 11) — wired with a real handle via `createSavedStateHandle()` in Task 16's inline factory; `SavedStateHandle()` is a plain class so the ViewModel test stays JVM (no Robolectric). The SAF copy-to-cache fallback (P2-6) is in `RealClassifier.decodePass`/`copyToCache` (Task 10), only retried when no frames were produced (no mid-stream double-decode).
- **ETA deviation from spec §5.1:** the spec phrased ETA as ms/frame × frame-count, but frame count is unknown before decode without an extra probe; the plan shows the captured **per-frame** ms in Setup (`etaPerFrameMs`) and the run reports actual elapsed. Documented, intentional.
