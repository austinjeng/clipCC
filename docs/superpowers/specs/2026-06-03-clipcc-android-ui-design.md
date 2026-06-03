# clipCC-Android — Plan 3 (Compose UI) Design

**Date:** 2026-06-03
**Status:** Approved design (pre-plan) — **Revision 2** (post external review; all P1/P2/P3 findings folded in)
**Parent spec:** `docs/superpowers/specs/2026-06-02-clipcc-android-design.md` (§6 UI, §10 phase 3)
**Handoff:** `docs/superpowers/plans/phase3-handoff.md`
**Target device:** Pixel 7a (Tensor G2), Android 16 / API 36
**Predecessors:** Plan 0 (assets/spikes) ✅ · Plan 1 (headless engine) ✅ · Plan 2 (benchmark harness) ✅

---

## 0. Project root, paths, and sync policy (read first)

This design doc lives in the **host Python repo** (`/Users/austin/MITAC/clipCC`, git), but the code it
describes lives in a **separate, non-git Android project**. To remove all path ambiguity:

- **Android project root:** `/Users/austin/AndroidStudioProjects/ClipCC` — **not** under git ("commit"
  there means *save the file*; there is no VCS).
- **Gradle module:** `:app` · **package / namespace:** `com.example.clipcc` · **applicationId:**
  `com.example.clipcc` · **minSdk 24 / target+compile 36**.
- **Path convention in this spec:** every `app/src/...`, `assets/...`, or `engine/...` path is
  **relative to the Android project root above**, *not* to the host repo. (The host repo's own `app/`
  is the unrelated Python service — never a target here.)
- **Sync / commit policy:**
  - Design specs, plans, and phase reports → written under the **host repo** `docs/superpowers/` and
    **committed to git** (this file included).
  - Android source/asset changes → **saved** in the Android project (no git). Verification is by
    build + test output, not by diff. Each subagent task records its touched files + test evidence in
    the plan/report, mirroring Plans 1–2.

---

## 1. Goal

Build the Jetpack Compose / Material 3 UI on top of the completed headless engine: a **live
classification** flow (pick model + backend + video + labels + aggregation mode → run on-device →
view best match, per-label charts, and mode-specific extras for all four modes) plus a **read-only
benchmark panel** that displays the already-captured Plan 2 results.

**Primary emphasis (decided): live classification is the star.** The Setup → Run → Results flow is
the centerpiece, closest in spirit to the Python web UI. The benchmark panel is a peer tab that
renders `phase2-benchmark-result.json` — captured data, not a live re-run.

### Success criteria (the Plan-3 gate, from parent §10)

1. All four aggregation modes (`mean` / `max` / `temporal` / `contrast`) render correctly from real
   on-device engine output.
2. The benchmark panel shows the CPU timed lanes (load / vision / ms-per-frame / fps) as rows and the
   NNAPI lanes as **capability-only / not-timed** rows with node-coverage % + experimental badges,
   from the captured data.
3. A long run shows honest per-chunk progress, is cancellable at every stage checkpoint, displays an
   ETA, and keeps the screen awake while running.
4. The UI reuses the existing engine with only **additive, parity-neutral** touches (§7); all Plan-1/2
   tests stay green.

---

## 2. Decisions locked in brainstorming

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | UI emphasis | **Live classification is the star**; benchmark = read-only view of the JSON | Matches the Python web UI's spirit; benchmark data already captured in Plan 2 |
| 2 | Model provisioning | **adb-push / local import; defer the §9 network downloader** | "adb-push sufficed" (Plan 2); downloader is a large orthogonal subsystem |
| 3 | Charting | **Custom Compose Canvas** (no library), **separate stacked scales** | Charts are simple; zero-dep; confidence (0–1) and cosine (signed, small) need independent axes — dual-axis would distort |
| 4 | Temporal/contrast defaults | **App-side `ScoringPolicy` constants; no schema-v2 bump** | These are model-*independent* policy constants; a per-model manifest bump would duplicate them 4× |
| 5 | Run lifecycle | **ViewModel coroutine + keep-screen-on + cooperative cancel; attended/foreground-only** | Fits a watched bench run; foreground service is out of scope (deferred with the downloader) |

Decision 4 **retires the parent spec's "schema-v2 required in Plan 3" open item** (§5.0/§12): the
deferred `ScoringPolicySpec` *policy* values (gap 2.0, min-dur 1.0, threshold 0.5, contrast defaults)
are global constants, not per-model data, so they live in one Kotlin object. The genuinely per-model
values (`logit_scale`, `logit_bias`, `score_semantics`, per-file `sha256`/`bytes`) are **already in
manifest v1** (verified against `manifest_base256.json`) — so no schema bump is needed, and §8 reads
those existing fields rather than inventing new ones.

Decision 1's honesty refinement to parent §6: the backend selector exposes the **real engine lanes**,
not the aspirational "CPU / GPU / NPU" (GPU/NPU don't exist as separate lanes on Tensor G2; verified
Plans 0/2). See §5.1 / §2-backend-mapping below.

**`engine/ScoringPolicy.kt` constants** (pinned to the exact Python sources; `ScoringPolicyTest`
guards drift):

```kotlin
object ScoringPolicy {
    const val THRESHOLD = 0.5             // temporal_policy.py SigLip2Policy.default_threshold
    const val THRESHOLD_MODE = "absolute" // SigLip2Policy.threshold_mode
    const val GAP_TOLERANCE = 2.0         // response.py ResolvedTemporalOptions default
    const val MIN_DURATION = 1.0          // response.py ResolvedTemporalOptions default
    const val CONTRAST_THRESHOLD = 0.15   // SigLip2Policy.contrast_default_threshold
    const val CONTRAST_REDUCE = "mean"    // SigLip2Policy.contrast_default_reduction
    val CONTRAST_REDUCE_MODES = listOf("mean", "top_k_mean", "max", "quantile")
    const val SCORE_SEMANTICS = "siglip2_pairwise_sigmoid"
    const val FPS = 1.0                   // services/video.py
    const val MAX_FRAMES = 300            // services/video.py
    val DEFAULT_LABELS = listOf(          // config.py default_labels
        "texting while driving", "sleeping while driving", "eating while driving")

    // Chunked vision-encode sizing is per-(model,backend) — mirrors the phase-2 CPU_EP batches
    // (replaces the earlier single VISION_CHUNK const; resolves review P3-6).
    fun visionChunkFor(modelId: String, backend: UiBackend): Int = when {
        backend != UiBackend.CPU_EP -> 16            // XNNPACK: decode/release granularity; encode is batch=1
        modelId.contains("so400m")  -> 4
        modelId.contains("large")   -> 8
        else                        -> 16            // base-256 / base-384
    }
}
```

### Backend mapping (resolves review P2-8)

The engine `Backend` enum has four values; two of the NNAPI lanes are capability-probes, not live
options. A `UiBackend` enum maps cleanly:

| `UiBackend` (live options) | → engine `Backend` | Notes shown in UI |
|---|---|---|
| `CPU_XNNPACK` (default) | `CPU_XNNPACK` | per-frame; partial XNNPACK delegation |
| `CPU_EP` | `CPU_EP` | batched ORT CPU EP |
| `NNAPI` (experimental) | `NNAPI_DEFAULT` | "experimental — 0 % delegated on Tensor G2" |

`NNAPI_CPU_DISABLED` is **not** a live option — it is a capability-probe lane only (it appears in the
benchmark panel's capability rows, never in the live selector). A run's result shows the **requested**
backend only; we do **not** claim an "effective"/delegated backend live, because actual node placement
is readable only from an untimed profiling pass (the `BackendCapabilityReport` path), not from a normal
`createSession` (which "succeeds" for NNAPI even at 0 % delegation). Live node coverage is therefore
shown in the Benchmark panel (captured), and the Results meta links there rather than asserting it
(resolves review P2-3).

---

## 3. Architecture

Single-Activity Compose app. Two top-level tabs (`TabRow`): **Classify** | **Benchmark**. No
navigation library, no chart library, no image library.

```
com.example.clipcc/
  MainActivity.kt                 # 2-tab Scaffold; applies FLAG_KEEP_SCREEN_ON while a run is active
  ui/
    app/ClipCCApp.kt              # top scaffold + TabRow (Classify | Benchmark)
    classify/
      ClassifyViewModel.kt        # single state holder (StateFlow<ClassifyUiState>) + SavedStateHandle
      ClassifyUiState.kt          # SetupState + sealed RunState + ModelInfo + DTOs (§4.1)
      Classifier.kt               # interface seam (real = FrameSampler+Engine+Scoring; fake in tests)
      RealClassifier.kt           # chunked decode→encode→release pipeline (§4.2)
      SetupCard.kt                # model/backend/video/labels/mode + options, Run/Cancel, ETA
      RunStatus.kt                # stage + per-chunk progress, Cancel, "Cancelling…"
      ResultsSection.kt           # best-match card + bar charts + mode-extra dispatch
      ModeExtras.kt               # MaxExtras / TemporalExtras / ContrastExtras
      LabelValidation.kt          # trim/blank/duplicate rules (pure, JVM-testable)
    benchmark/
      BenchmarkScreen.kt          # grouped table (CPU timed + NNAPI capability-only) + device header
      BenchmarkData.kt            # parse bundled JSON → rows + capability join (pure, JVM-testable)
    charts/
      BarChart.kt                 # grouped bars, single scale (Canvas) + Compose semantics
      TimelineChart.kt            # temporal per-frame line + segment overlay (Canvas) + semantics
      ChartData.kt                # pure data-prep helpers (JVM-testable)
    theme/                        # exists (Color/Theme/Type)
  data/
    ModelRepository.kt            # scan filesDir/models/<id>/manifest.json → ready models (+ size/semantics checks)
  engine/
    ScoringPolicy.kt              # NEW: model-independent defaults (mirror Python)
    Manifest.kt                   # EXTENDED: also read score_semantics, bytes, sha256 (§7.4)
    FrameSampler.kt               # EXTENDED: Uri overload + per-frame callback hook (§7.1)
    Engine.kt / OrtTower.kt       # EXTENDED: chunked encode + onProgress/isCancelled (§7.2/§7.3)
    …rest unchanged…
app/src/main/assets/phase2-benchmark-result.json   # bundled, read-only (ingestion task)
```

**New dependencies** (Gradle version catalog): `androidx.lifecycle:lifecycle-viewmodel-compose`,
`androidx.lifecycle:lifecycle-runtime-compose`, and `androidx.lifecycle:lifecycle-viewmodel-savedstate`
(for `SavedStateHandle`). Everything else (Compose BOM, Material 3, activity-compose,
lifecycle-runtime-ktx, ONNX Runtime, Media3) is already wired.

### Module boundaries

- **`ClassifyViewModel`** owns all Classify state + the run lifecycle; depends on a `Classifier`
  interface and `ModelRepository`; no Compose/Android-UI imports → JVM-unit-testable. Setup fields are
  mirrored into `SavedStateHandle` for process-death restore.
- **`Classifier`** seam: `suspend fun classify(req: ClassifyRequest, onProgress, isCancelled): RunResult`.
  `RealClassifier` runs the chunked pipeline (§4.2); tests inject a fake returning canned results.
- **`ModelRepository`** owns discovery (scan + parse + readiness + semantics validation); no UI.
- **`BenchmarkData`** is a pure parser; the screen only renders.
- **`charts/`** composables are dumb renderers; all numeric prep is in `ChartData` (pure); each
  exposes Compose semantics + a textual summary for accessibility.

---

## 4. State model & run pipeline

### 4.1 Frozen DTOs (resolves review P1-2)

```kotlin
data class ClassifyRequest(
    val modelDir: String, val manifest: ModelBundleManifest,
    val backend: UiBackend, val videoUriString: String,   // parsed to Uri inside RealClassifier (platform-light)
    val labels: List<String>,          // mean/max/temporal: as-is; contrast: posLabels + negLabels
    val posCount: Int,                 // contrast only (0 for non-contrast)
    val mode: AggMode,
    val temporal: TemporalOptions, val contrast: ContrastOptions,
)
data class RunResult(
    val result: AggregationResult,     // engine output (existing type)
    val thumbnails: Map<Int, Bitmap>,  // frameIndex → small (~96px) thumbnail; all frames, bounded (§4.2)
    val timestamps: DoubleArray,
    val meta: RunMeta,
)
data class RunMeta(   // requestedBackend only — no live "effective" claim (see backend-mapping note, P2-3)
    val modelId: String, val requestedBackend: UiBackend,
    val frameCount: Int, val elapsedMs: Long, val scoreSemantics: String,
)
data class TemporalOptions(                       // defaults from ScoringPolicy
    val threshold: Double = ScoringPolicy.THRESHOLD, val gap: Double = ScoringPolicy.GAP_TOLERANCE,
    val minDuration: Double = ScoringPolicy.MIN_DURATION, val thresholdWasDefaulted: Boolean = true)
data class ContrastOptions(
    val threshold: Double = ScoringPolicy.CONTRAST_THRESHOLD,
    val reduce: String = ScoringPolicy.CONTRAST_REDUCE, val thresholdWasDefaulted: Boolean = true)
```

```kotlin
data class SetupState(
    val availableModels: List<ModelInfo>, val selectedModelId: String?,
    val backend: UiBackend = UiBackend.CPU_XNNPACK,
    val videoUriString: String?, val videoName: String?, val grantPersisted: Boolean = false,
    val labels: List<String> = ScoringPolicy.DEFAULT_LABELS,
    val posLabels: List<String> = ScoringPolicy.DEFAULT_LABELS, val negLabels: List<String> = emptyList(),
    val mode: AggMode = AggMode.MEAN,
    val temporal: TemporalOptions = TemporalOptions(), val contrast: ContrastOptions = ContrastOptions(),
    val validation: ValidationState,   // per-field errors + canRun
    val etaMs: Long?,                  // estimated run time (§9)
)
enum class AggMode { MEAN, MAX, TEMPORAL, CONTRAST }

sealed interface RunState {
    data object Idle : RunState
    data class Running(val stage: Stage, val chunkDone: Int, val chunkTotal: Int) : RunState
    data object Cancelling : RunState
    data class Success(val result: RunResult) : RunState
    data class Error(val message: String) : RunState
    data object Cancelled : RunState
}
enum class Stage { LOADING_MODEL, DECODING, ENCODING_TEXT, ENCODING_VISION, AGGREGATING }
data class ClassifyUiState(val setup: SetupState, val run: RunState, val keepAwake: Boolean)
```

`StateFlow<ClassifyUiState>`, collected with `collectAsStateWithLifecycle`. The run executes in
`viewModelScope` on `Dispatchers.Default`; the active `Job` is retained for Cancel. `keepAwake =
run is Running`. Full bitmaps are never retained in `Success` — only small per-frame thumbnails
(§4.2; resolves review P1-3).

**Platform-light state (resolves review P2-4):** ViewModel/Setup state holds a `videoUriString`
(String) + grant flag, **not** an Android `Uri`; the launcher (UI) produces the `Uri` and the
`RealClassifier` parses the string back. The only Android type in the state graph is the pass-through
`Bitmap` thumbnails in `RunResult` (produced by `RealClassifier`, rendered by the UI). Unit tests
inject a fake `Classifier` returning an empty thumbnail map, so `ClassifyViewModelTest` constructs no
Android type and runs on the plain JVM (Robolectric only if a test asserts on an Android type).

### 4.2 Run pipeline (chunked, memory-bounded — resolves review P1-3 & P1-5)

The ViewModel validates, sets `keepAwake`, launches the run, and owns `RunState`. The
`RealClassifier` works in **chunks of `ScoringPolicy.visionChunkFor(modelId, backend)` frames** so peak
native + bitmap memory stays bounded regardless of frame count:

1. `LOADING_MODEL` — open text tower; **cancel checkpoint** before/after.
2. `ENCODING_TEXT` — encode labels once; release the text session (existing memory strategy). **cancel
   checkpoint.**
3. Open vision tower, then make **one forward decode pass** —
   `FrameSampler.sample(uri, fps, maxFrames, onFrame)` (§7.1). For each `onFrame(SampledFrame)`:
   downscale to a small (~96px) **thumbnail and keep it** (all-frame thumbnails total ≈ 11 MB at the
   300-frame cap — cheap and bounded), add the full bitmap to the current chunk buffer, and **return
   `false` to cancel** (the **per-frame cancel checkpoint**).
   - When the chunk fills (or the pass ends): `ENCODING_VISION` — encode the chunk, accumulate the small
     `[chunk×D]` embeddings, **emit `onProgress(chunkDone, chunkTotal)`**, then **release that chunk's
     full bitmaps** (thumbnails already kept). **cancel checkpoint** between chunks.
4. `AGGREGATING` — build `[F×L]` cosine/confidence from the accumulated embeddings + cached text
   embeddings via `Scoring.scoreMatrix`, then dispatch by mode (§5.3 mapping). MAX peak indices index
   straight into the retained thumbnails — **no re-decode, no random-access** needed.
5. `Success(RunResult)` (`requestedBackend` recorded; effective node placement not claimed live — see
   the backend-mapping note).

**Cancellation** is cooperative at every checkpoint above; on cancel the UI shows `Cancelling…` until
the worker returns, then `Cancelled`. Progress granularity is **per chunk** (honest — not a fake
per-frame number under batched EPs); the label reads "Encoding frames a–b of N".

**Engine implication (flagged):** the current `Engine.scoreFrames` packs *all* frames into one buffer
and encodes once. Chunked encoding is a new engine capability (§7.3) — additive, with the existing
all-at-once path preserved as the default for `Benchmark`/tests. This is the third engine touch and is
called out so it is planned, not discovered mid-build.

---

## 5. Screens

### 5.1 Setup (`SetupCard`)

- **Model dropdown** — `ExposedDropdownMenuBox` over `ModelRepository` ready bundles, labelled
  `"<display_name> · <res>px · <precision>"`. Not-ready bundles greyed with the reason ("not
  provisioned" / "size mismatch" / "unsupported semantics"). Empty-state card shows the §8 adb-push
  recipe if none are ready.
- **Backend** — `SingleChoiceSegmentedButtonRow` over `UiBackend` (CPU·XNNPACK / CPU·EP / NNAPI); the
  NNAPI option carries an "experimental — 0 % delegated on Tensor G2" caption.
- **Video picker** — `rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument())`, then
  `launcher.launch(arrayOf("video/*"))` (the MIME array goes to `launch`, **not** the contract
  constructor — resolves review P2-5). On result, `takePersistableUriPermission` wrapped in try/catch
  (resolves review P2-6): on failure the URI is used for this session only with a note shown, and the
  grant flag is stored. The `Uri` is converted to `videoUriString` for the ViewModel; the filename is
  shown.
- **Label editor** — add/remove text fields, default `ScoringPolicy.DEFAULT_LABELS`. In **contrast**
  mode it switches to two grouped lists (Positive / Negative). Validation (`LabelValidation`, resolves
  review P2-11): trim whitespace, reject blank entries, reject exact duplicates within *and across*
  groups, **preserve case** (the tokenizer is case-sensitive — parent §5.2); case-sensitivity is noted
  in the UI.
- **Mode selector** — MEAN / MAX / TEMPORAL / CONTRAST + a mode-options panel: temporal → threshold
  (0.5) / gap (2.0) / min-dur (1.0); contrast → reduce mode (`mean`/`top_k_mean`/`max`/`quantile`,
  default `mean`) + threshold (0.15). Defaults from `ScoringPolicy`.
- **ETA** — when model+backend+video are chosen, show an estimated run time computed from the captured
  benchmark ms/frame for that (model, nearest backend) × the clip's frame count (§9). Labelled an
  estimate.
- **Run** (enabled iff `validation.canRun`) / **Cancel** (while running, in `RunStatus`).

### 5.2 Run status (`RunStatus`)

While `Running`: stage label (Loading model / Encoding labels / Decoding / Encoding frames a–b of N /
Aggregating), a progress indicator (indeterminate for non-chunked stages, determinate
`chunkDone/chunkTotal` for vision), elapsed time, and a **Cancel** button. On cancel → `Cancelling…`
spinner until the worker unwinds.

### 5.3 Results (`ResultsSection`, on `Success`)

- **BestMatchCard** — best-match label + confidence + a meta line: model · **requested backend** · N
  frames · elapsed · `score_semantics`, with a caption "live node coverage not profiled — see Benchmark"
  (no false "effective backend" claim — resolves review P2-3; surfaces semantics — P2-10).
- **Per-label charts** (`BarChart` ×2, separate scales — resolves review P3-12 / chart-scale):
  - **Confidence** row — bars 0..1 with the 0.5 threshold guide line.
  - **Raw similarity (cosine)** row — signed/symmetric axis (cosine is small and may be negative).
  - A **value table** beneath (label · confidence · cosine) doubles as the screen-reader text; charts
    carry `contentDescription` summaries.
- **Mode extras** (`ModeExtras.kt`):
  - **MEAN** — the two bar rows only.
  - **MAX** — per-label peak-frame **thumbnail** (`Image` from the retained downscaled bitmap) +
    timestamp.
  - **TEMPORAL** — `TimelineChart` (per-label confidence over time + threshold line + shaded segments)
    → segment list (label, start–end, duration, active-avg, peak) → label summaries (segment count,
    total active duration, duration-weighted confidence); best segment highlighted.
  - **CONTRAST** — colored verdict banner (positive/negative/uncertain) + margin (`difference`) +
    pos/neg group means + per-label bars + dominant label + threshold source / calibration status.

### 5.4 Benchmark (`BenchmarkScreen`) — resolves review P2-7

- **Asset-ingestion task:** `phase2-benchmark-result.json` is copied into
  `app/src/main/assets/` (its own plan task) and parsed by `BenchmarkData` (pure).
- **Provenance header** rendered from fixed constants (the snapshot is frozen, so no JSON enrichment —
  YAGNI): *Pixel 7a · Tensor G2 · median-of-3, 1 warm-up discarded · CPU-only · Media3 1.10.1 · captured
  2026-06-03*.
- **Per-model groups:**
  - **CPU·XNNPACK** and **CPU·EP** → **timed rows**: load ms · vision ms (median) · ms/frame · fps ·
    vision node-coverage % (joined from `capabilities`).
  - **NNAPI_DEFAULT** and **NNAPI_CPU_DISABLED** → **capability-only rows**, explicitly labelled "not
    timed", showing the 0 %-delegated coverage + an **experimental** badge. (The captured JSON has no
    timed NNAPI runs; the panel must not imply otherwise.)
- Optional compact bars comparing ms/frame across the 4 models (CPU lanes only).

---

## 6. Charts (`charts/`, Canvas)

All numeric preparation lives in `ChartData` (pure, JVM-tested); composables only draw and attach
semantics.

- **`BarChart`** — input `(label, value, scale)` groups on a **single declared scale** per instance
  (so confidence and cosine are *separate* `BarChart`s, never dual-axis). Zero baseline supports
  negative cosine; optional threshold guide; axis ticks; value labels; a `contentDescription`
  summarising each bar.
- **`TimelineChart`** — input timestamps + per-label score series + threshold + segments; draws
  time-x / score-y axes, one polyline per label, a dashed threshold line, shaded segment bands, and a
  textual summary for accessibility. ≤300 points, few labels.

---

## 7. Engine touches (additive, parity-neutral)

All default to current behavior → `Benchmark` and every Plan-1/2 test stay green.

1. **Streaming `FrameSampler` API (resolves review P1-2):**
   `sample(uri: Uri, fps, maxFrames, onFrame: (SampledFrame) -> Boolean): VideoMeta` — a single forward
   decode pass that invokes `onFrame` per decoded frame and **stops early when it returns `false`**
   (cancel; lets the classifier chunk/encode/release without holding all frames). Internally uses the
   same `FrameExtractor.getFrame(posMs)` forward seek as today; because all-frame thumbnails are
   retained (§4.2), no `sampleAt`/`sampleRange` random-access variant is needed. The existing
   `sample(videoPath: String, …): Pair<VideoMeta, List<SampledFrame>>` (used by `Benchmark`) is kept and
   may delegate to a buffering wrapper of the streaming pass. Existing `FrameSamplerTest` stays green.
2. **`Engine.scoreFrames(..., onProgress: ((Int,Int)->Unit)? = null, isCancelled: (()->Boolean)? =
   null)`** — threaded into `OrtTower.encodeVision`; checked between frames/batches; both default
   `null` ⇒ no change to existing callers.
3. **Chunked vision encode** — a path that encodes a frame chunk and returns its `[chunk×D]`
   embeddings without holding all frames (resolves review P1-3). Chunk size =
   `ScoringPolicy.visionChunkFor(modelId, backend)`, pinned to the phase-2 CPU_EP batches (base 16,
   large 8, so400m 4) under `CPU_EP`, and 16 (decode/release granularity; encode stays batch=1) under
   XNNPACK — resolves review P3-6, replacing the single `VISION_CHUNK` const. The existing all-at-once
   `scoreFrames` remains the default for `Benchmark`/tests; the chunked entry is new and used by the UI.
4. **`Manifest.kt` extension** — additionally read the **already-present** v1 fields `score_semantics`,
   per-file `bytes`, and `sha256` (verified in `manifest_base256.json`). Still `schema_version == 1`;
   purely additive parsing. `ManifestTest` updated to assert the new fields.

---

## 8. Model provisioning (`ModelRepository`) — resolves review P2-9 & P2-10

- **Bundle root:** `context.filesDir/models/<model_id>/` (Phase 2 verified `getExternalFilesDir(null)`
  is **null** on this device).
- **`scan()`** → for each `models/*/manifest.json`: `ModelBundleManifest.parse(...)`, then readiness:
  1. all referenced files (`visionFile`, `textFile`, `tokenizerFile`, any `*DataFile`) **exist**;
  2. **size-match the ONNX `vision`/`text` files only** against their manifest `bytes` (instant
     integrity check). The v1 manifest carries `bytes` for vision/text but **not** for the tokenizer
     (verified in `manifest_base256.json` — resolves review P1-1), so the tokenizer is
     existence-checked and may be **sha256-verified** (it is small, so hashing is cheap). External
     `.onnx_data` files have no `bytes` in v1 → existence-checked here; their size/hash gate ships with
     the downloader;
  3. `score_semantics == "siglip2_pairwise_sigmoid"` (else flagged "unsupported semantics" and not
     runnable — guards the app-side `ScoringPolicy` against a mismatched bundle).
  → `ModelInfo(id, displayName, resolution, precision, scoreSemantics, ready, reason)`.
- **sha256:** the manifest carries per-file hashes, but full-file hashing of 0.4–4 GB on-device is slow
  (tens of seconds → minutes for so400m). Full verification is therefore **lazy / deferred**: it ships
  with the network downloader (where corruption is a real risk) using a cached `.validated` marker like
  the Python side. The adb-push path relies on size-match, since the bytes come from a trusted host
  export, not a flaky download. (This is the explicit, documented readiness contract — resolving the
  apparent conflict with parent §5.0, which lists sha256 as *present in the manifest*, not as
  *verified on every scan*.)
- **One active model at a time**; `Engine` is built per run with that bundle's `modelDir` and a
  conservative per-model `visionBatch` (so400m ≤ 8).
- **Dev provisioning recipe** (the supported path this plan):
  ```
  adb push <bundle_dir> /data/local/tmp/clipcc_models/
  adb shell run-as com.example.clipcc cp -r /data/local/tmp/clipcc_models/<id> files/models/
  ```
  (App is debuggable, so `run-as` can write into the app's internal `filesDir`.)

---

## 9. Error handling, cancellation, lifecycle — resolves review P1-4 & P1-5

### Error handling
- **Validation** (no model/video, <1 label, empty contrast group, blank/duplicate labels) is inline;
  Run is disabled with per-field reasons — never thrown.
- **Runtime failures** caught in the ViewModel → `RunState.Error(message)` with Retry / Back actions:
  - decode failure / 0 frames / unhandled HDR → "Couldn't decode video: …" — plus, for `content://`
    providers Media3 can't seek/decode, a **guarded copy-to-cache fallback** (decode the cached copy);
    only if that also fails do we surface the error (resolves review P2-6).
  - catchable `OrtException` / model-load failure → message surfaced.
  - missing/corrupt/size-mismatch/semantics-mismatch bundles → caught by `ModelRepository` readiness
    before a run.
- **Native OOM caveat (documented, not pretended-handled):** a hard native ORT OOM kills the process
  and cannot be caught. Mitigated by conservative per-model batch + chunked encode (§4.2) bounding peak
  bitmaps + one resident model. Stated in an on-screen note for the largest model.

### Cancellation
- Cooperative checkpoints at **model-load, text-encode, each decoded frame, and each vision chunk**
  (§4.2). `Cancel` → flag flips + `Job.cancel()` → `Cancelling…` → worker unwinds → `Cancelled` →
  Setup; partial work discarded. Mirrors Python `cancel_event`.

### Lifecycle (attended / foreground-only — consistent with the no-service decision)
- The run is **explicitly attended**: it runs while the app is foregrounded (or briefly backgrounded)
  and the screen is held awake (`FLAG_KEEP_SCREEN_ON` while `keepAwake`). If the OS kills the process
  mid-run, the run is **lost by design** (no foreground service this plan) — on relaunch the UI shows
  `Idle` and the restored Setup, never a stale half-run.
- **Setup is restored after process death** via `SavedStateHandle` (selected model id, backend, labels
  + groups, mode, mode options) plus the persisted video URI grant (resolves review P1-4). The ETA
  (§5.1) sets expectations before a long run starts.
- The ViewModel survives configuration changes; an in-flight run continues and the `StateFlow` is
  re-collected. A single shared `OrtEnvironment.getEnvironment()` is reused across runs.

---

## 10. Testing

**Gate (parent §10.3):** all four modes render from real engine output; the benchmark panel shows the
CPU timed lanes + NNAPI capability-only lanes with coverage + experimental badges.

**Seam:** the ViewModel depends on the injected `Classifier`; the real impl wires
`FrameSampler`+`Engine`+`Scoring` (chunked); tests inject a fake → ORT/Media3/device stay out of unit
tests. Setup state is platform-light (`videoUriString`, not `Uri`; thumbnails are pass-through), so
`ClassifyViewModelTest` runs on the plain JVM with no Android type constructed (Robolectric only if a
specific test asserts on an Android type — resolves review P2-4).

- **JVM unit (the bulk, fast):**
  - `ScoringPolicyTest` — every constant equals its pinned Python source value (drift guard vs
    `temporal_policy.py` / `response.py` / `config.py`); `visionChunkFor` returns the pinned phase-2
    batches (base 16 / large 8 / so400m 4 for CPU_EP; 16 for XNNPACK).
  - `ManifestExtensionTest` — parses `score_semantics`, `bytes`, `sha256` from the real manifest
    fixture.
  - `ModelRepositoryTest` — readiness logic: files-exist + size-match + semantics gate (using temp
    dirs / fakes); each failure produces the right `reason`.
  - `LabelValidationTest` — trim, blank rejection, duplicate within/across groups, case preserved.
  - `BenchmarkDataTest` — fixture → CPU timed rows + NNAPI capability-only rows + node-coverage join +
    experimental flag.
  - `ClassifyViewModelTest` — validation gating; `Idle→Running→(Cancelling)→Success/Error/Cancelled`;
    chunked-progress emission; cancel-at-checkpoint; mode→aggregation dispatch with canned matrices;
    contrast pos/neg → `posCount` + concat; `SavedStateHandle` round-trip of Setup; ETA computation.
  - `ChartDataTest` — separate-scale bar prep (incl. negative cosine baseline) + timeline series /
    segment-band prep.
- **Instrumented smoke (1, device):** provision base-256, run the real test clip through the
  `RealClassifier` on-device → `Success` with non-empty scores + best match + only peak thumbnails
  retained (confirms wiring + the chunked release; numerical parity already covered by Plan 1).
  Optional Compose `createComposeRule` test driving Setup→Run→Results-nodes with a fake fast classifier.
- **Manual acceptance:** Pixel 7a screenshots of each mode's Results + the benchmark panel.

---

## 11. Scope

**In scope:** the 2-tab app; Setup (with validation + ETA); chunked, cancellable live run across all
four modes with honest per-chunk progress; Results + separate-scale Canvas charts + a11y summaries; the
read-only Benchmark panel (CPU timed + NNAPI capability-only); `ModelRepository` (readiness =
parse + files-exist + size-match + semantics gate); `ScoringPolicy`; the four engine touches (§7);
`SavedStateHandle` Setup restore; the tests in §10.

**Out of scope (deferred, explicit):**
- the §9 network downloader (HF Xet / resume / free-space / eviction) **and its foreground service** —
  full sha256 verification ships here;
- foreground-service-backed runs (runs are attended this plan);
- manifest **schema-v2** (retired by Decision 4 — v1 already carries the per-model fields);
- model eviction UI;
- live benchmark re-run from the app (the matrix OOMs a single process and takes ~30 min — captured
  data only);
- fp16/fp32 precision toggle (the app uses each bundle's provisioned precision);
- longer / multi-clip benchmark capture.

---

## 12. Items resolved in `writing-plans` before any execution task starts

These were the §12 open items; per review P1-2 they are settled in the plan (not left for mid-build):

- **Task breakdown & order** for subagent-driven-development. Expected order, each its own gate:
  1. Gradle deps + theme/`ClipCCApp` 2-tab scaffold (replaces the `Greeting` template).
  2. `ScoringPolicy` (+ `ScoringPolicyTest`) and `Manifest.kt` extension (+ test).
  3. `ModelRepository` (+ test).
  4. Engine touches §7.1–7.3 (FrameSampler Uri/callback, progress/cancel, chunked encode) with the
     existing tests kept green.
  5. `Classifier`/`RealClassifier` + `ClassifyViewModel` (+ tests, fake classifier).
  6. Setup screen (model/backend/video/labels/mode/options/ETA/validation).
  7. Results + charts (separate-scale bars, timeline, mode extras, a11y).
  8. Benchmark asset ingestion + panel.
  9. Instrumented smoke + manual screenshots.
- **DTO shapes** — frozen in §4.1.
- **Chart scale** — decided (separate stacked scales, §2/§6).
- **`visionChunkFor(modelId, backend)`** — decided (CPU_EP: base 16 / large 8 / so400m 4; XNNPACK: 16
  decode-granularity, encode batch=1), mirroring the phase-2 benchmark batches (§7.3 / P3-6).
- **Whether the optional Compose `createComposeRule` test is in the gate** or manual-only — decided in
  the plan.
- **Empty-state / provisioning card copy** — drafted in the plan.
