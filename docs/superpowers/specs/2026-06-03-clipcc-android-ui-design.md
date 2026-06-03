# clipCC-Android — Plan 3 (Compose UI) Design

**Date:** 2026-06-03
**Status:** Approved design (pre-plan)
**Parent spec:** `docs/superpowers/specs/2026-06-02-clipcc-android-design.md` (§6 UI, §10 phase 3)
**Handoff:** `docs/superpowers/plans/phase3-handoff.md`
**Target device:** Pixel 7a (Tensor G2), Android 16 / API 36
**Predecessors:** Plan 0 (assets/spikes) ✅ · Plan 1 (headless engine) ✅ · Plan 2 (benchmark harness) ✅

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
2. The benchmark panel shows per-model/backend timings (load / vision / ms-per-frame / fps) + actual
   backend + node-coverage % + experimental badges, from the captured `BackendCapabilityReport` data.
3. A long run is cancellable and shows per-frame progress; the screen stays awake during a run.
4. The UI reuses the existing engine unchanged except for two additive, parity-neutral touches (§7).

---

## 2. Decisions locked in brainstorming

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | UI emphasis | **Live classification is the star**; benchmark = read-only view of the JSON | Matches the Python web UI's spirit; benchmark data already captured in Plan 2 |
| 2 | Model provisioning | **adb-push / local import; defer the §9 network downloader** | "adb-push sufficed" (Plan 2); downloader is a large orthogonal subsystem |
| 3 | Charting | **Custom Compose Canvas** (no library) | Charts are simple (grouped bars + one timeline line); zero-dep; simplicity-first |
| 4 | Temporal/contrast defaults | **App-side `ScoringPolicy` constants; no schema-v2 bump** | These are model-*independent* policy constants; a per-model manifest bump would duplicate them 4× |
| 5 | Run lifecycle | **ViewModel coroutine + keep-screen-on + cooperative cancel** | Fits a watched bench run; foreground service is over-engineering here |

Decision 4 **retires the parent spec's "schema-v2 required in Plan 3" open item** (§5.0/§12): the
deferred `ScoringPolicySpec` values (gap 2.0, min-dur 1.0, threshold 0.5, contrast defaults) are
global policy constants, not per-model data, so they live in one Kotlin object instead of four
manifests. The only genuinely per-model scoring values (`logit_scale` / `logit_bias`) are already in
manifest v1.

Decision 1's honesty refinement to parent §6: the backend selector exposes the **real engine lanes**
— **CPU·XNNPACK** (default), **CPU·EP** (batched), **NNAPI (experimental — 0 % delegated on Tensor
G2)** — not the aspirational "CPU / GPU / NPU", because GPU/NPU don't exist as separate lanes on this
device (verified Plans 0/2). Same attempt-and-report honesty as the benchmark.

---

## 3. Architecture

Single-Activity Compose app. Two top-level tabs (`TabRow`): **Classify** | **Benchmark**. No
navigation library, no chart library, no image library.

```
com.example.clipcc/
  MainActivity.kt                 # 2-tab Scaffold; applies FLAG_KEEP_SCREEN_ON during a run
  ui/
    app/ClipCCApp.kt              # top scaffold + TabRow (Classify | Benchmark)
    classify/
      ClassifyViewModel.kt        # single state holder (StateFlow<ClassifyUiState>)
      ClassifyUiState.kt          # SetupState + sealed RunState + ModelInfo
      Classifier.kt               # interface seam (real = FrameSampler+Engine+Scoring; fake in tests)
      SetupCard.kt                # model/backend/video/labels/mode + options, Run/Cancel
      RunStatus.kt                # stage + per-frame progress, Cancel
      ResultsSection.kt           # best-match card + bar chart + mode-extra dispatch
      ModeExtras.kt               # MaxExtras / TemporalExtras / ContrastExtras
    benchmark/
      BenchmarkScreen.kt          # grouped table + experimental badges + device header
      BenchmarkData.kt            # parse bundled JSON → rows + capability join (pure, JVM-testable)
    charts/
      BarChart.kt                 # grouped confidence/raw-similarity bars (Canvas)
      TimelineChart.kt            # temporal per-frame line + segment overlay (Canvas)
      ChartData.kt                # pure data-prep helpers (JVM-testable)
    theme/                        # exists (Color/Theme/Type)
  data/
    ModelRepository.kt            # scan filesDir/models/<id>/manifest.json → ready models
  engine/
    ScoringPolicy.kt              # NEW: model-independent defaults (mirror Python)
    …existing engine (unchanged except the two §7 touches)…
app/src/main/assets/phase2-benchmark-result.json   # bundled, read-only
```

**New dependencies** (via the Gradle version catalog): `androidx.lifecycle:lifecycle-viewmodel-compose`
and `androidx.lifecycle:lifecycle-runtime-compose` (for `viewModel()` and
`collectAsStateWithLifecycle`). Everything else (Compose BOM, Material 3, activity-compose,
lifecycle-runtime-ktx, ONNX Runtime, Media3) is already wired in `app/build.gradle.kts`.

### Module boundaries

- **`ClassifyViewModel`** owns all Classify state and the run lifecycle; depends on a `Classifier`
  interface (the seam) and `ModelRepository`. No Compose, no Android-UI imports → unit-testable on the
  JVM.
- **`Classifier`** (interface): `suspend fun classify(req: ClassifyRequest, onProgress, isCancelled): RunResult`.
  Real implementation wires `FrameSampler` → `Engine.scoreFrames` → `Scoring.aggregate*`. Tests inject
  a fake returning canned `ScoreMatrices`/`AggregationResult`.
- **`ModelRepository`** owns model discovery (filesystem scan + manifest parse + readiness); no UI.
- **`BenchmarkData`** is a pure parser (JSON string → row models); the screen only renders.
- **`charts/`** composables are dumb renderers; all numeric prep is in `ChartData` (pure).

---

## 4. State model (`ClassifyViewModel`)

```kotlin
data class SetupState(
    val availableModels: List<ModelInfo>,
    val selectedModelId: String?,
    val backend: Backend = Backend.CPU_XNNPACK,
    val videoUri: Uri?, val videoName: String?,
    val labels: List<String> = ScoringPolicy.DEFAULT_LABELS,   // mean/max/temporal
    val posLabels: List<String>, val negLabels: List<String>,  // contrast groups
    val mode: AggMode = AggMode.MEAN,
    val temporal: TemporalOptions = TemporalOptions(),         // threshold/gap/minDur defaults
    val contrast: ContrastOptions = ContrastOptions(),         // reduce/threshold defaults
)

enum class AggMode { MEAN, MAX, TEMPORAL, CONTRAST }

sealed interface RunState {
    data object Idle : RunState
    data class Running(val stage: Stage, val frameDone: Int, val frameTotal: Int) : RunState
    data class Success(val result: AggregationResult, val frames: List<SampledFrame>, val meta: RunMeta) : RunState
    data class Error(val message: String) : RunState
    data object Cancelled : RunState
}
enum class Stage { DECODING, ENCODING, AGGREGATING }
data class RunMeta(val modelId: String, val backend: Backend, val frameCount: Int, val elapsedMs: Long)

data class ClassifyUiState(val setup: SetupState, val run: RunState, val keepAwake: Boolean)
```

`StateFlow<ClassifyUiState>`, collected with `collectAsStateWithLifecycle`. The run executes in
`viewModelScope` on `Dispatchers.Default`; the active `Job` is retained so Cancel can call
`Job.cancel()`. `keepAwake = run is Running`.

### Data flow on **Run**

The ViewModel validates, then launches the run and owns the `RunState` transitions; steps 2–4 are
performed *inside the real `Classifier` implementation* (the seam of §3), which emits progress back to
the ViewModel via the `onProgress` callback. Tests swap in a fake `Classifier` and assert the same
transitions.

1. **Validate** (ViewModel) — model ready · video chosen · ≥1 label; contrast requires ≥1 positive
   **and** ≥1 negative. Failures keep `Idle` and disable Run with an inline reason.
2. `Running(DECODING, 0, 0)` → `FrameSampler.sample(uri, ScoringPolicy.FPS, ScoringPolicy.MAX_FRAMES)`
   → frames + timestamps + `VideoMeta`.
3. `Running(ENCODING, i, N)` → `Engine(modelDir, manifest, env, backend, visionBatch, config)
   .scoreFrames(bitmaps, labels, onProgress = { d, n -> emit }, isCancelled = { job cancelled })`
   → `ScoreMatrices(cosine, confidence)`.
4. `Running(AGGREGATING, N, N)` → dispatch by `mode`:
   - `MEAN` → `Scoring.aggregateMean(confidence, cosine, labels)`
   - `MAX` → `Scoring.aggregateMax(confidence, cosine, labels, frameTimestamps)`
   - `TEMPORAL` → `Scoring.aggregateTemporal(confidence, cosine, labels, threshold, gap, minDur,
     FrameTimeline(timestamps, FPS, duration), thresholdWasDefaulted)`
   - `CONTRAST` → labels = `posLabels + negLabels`; `Scoring.aggregateContrast(confidence, cosine,
     labels, posCount = posLabels.size, reduce, threshold, thresholdWasDefaulted, ...)`
5. `Success(result, frames, meta)`.

`videoDuration` for `FrameTimeline` = last timestamp + frame interval (or `VideoMeta`-derived);
clamped exactly as the engine's `FrameTimeline` already does.

---

## 5. Screens

### 5.1 Setup (`SetupCard`)

- **Model dropdown** — `ExposedDropdownMenuBox` over `ModelRepository` ready bundles, labelled
  `"<id> · <res>px · <precision>"`. Not-provisioned bundles greyed with a "not provisioned" hint. If
  none are ready, an empty-state card shows the adb-push provisioning recipe (§8).
- **Backend** — `SingleChoiceSegmentedButtonRow`: CPU·XNNPACK / CPU·EP / NNAPI(exp). The NNAPI option
  carries an "experimental — no acceleration on Tensor G2" caption.
- **Video picker** — button launching `ActivityResultContracts.OpenDocument` (`arrayOf("video/*")`),
  `takePersistableUriPermission` on the result; shows the picked filename.
- **Label editor** — add/remove list of text fields, default = `ScoringPolicy.DEFAULT_LABELS`. In
  **contrast** mode it switches to two grouped lists (Positive / Negative).
- **Mode selector** — segmented control MEAN / MAX / TEMPORAL / CONTRAST, plus a mode-options panel:
  - temporal: threshold (default 0.5), gap (2.0), min-duration (1.0).
  - contrast: reduce mode (`mean` / `top_k_mean` / `max` / `quantile`, default `mean`) + threshold
    (default 0.15).
- **Run** (enabled when valid) / **Cancel** (while running, in `RunStatus`).

### 5.2 Run status (`RunStatus`)

Shown while `Running`: stage label (Decoding / Encoding frame i of N / Aggregating), a linear
progress indicator (indeterminate for decode, determinate `i/N` for encode), and a **Cancel** button.

### 5.3 Results (`ResultsSection`, on `Success`)

- **BestMatchCard** — best-match label + confidence, with a meta line (model · backend · N frames ·
  elapsed ms).
- **Per-label `BarChart`** — grouped bars: confidence (0..1) and raw similarity (cosine, may be
  negative → zero baseline). 0.5 threshold guide line; `siglip2_pairwise_sigmoid` caption.
- **Mode extras** (`ModeExtras.kt`):
  - **MEAN** — bar chart only.
  - **MAX** — per-label peak-frame thumbnail (`Image(bitmap.asImageBitmap())` from the retained
    `SampledFrame` at `peakFrameIndex`) + approx timestamp.
  - **TEMPORAL** — `TimelineChart` (per-label confidence over time + threshold line + shaded
    segments) → segment list (label, start–end, duration, active-avg, peak) → label summaries
    (segment count, total active duration, duration-weighted confidence); best segment highlighted.
  - **CONTRAST** — colored verdict banner (positive / negative / uncertain) + margin (`difference`)
    + positive/negative group means + per-label bars + dominant label.

The run's `SampledFrame` list is retained in `RunState.Success` so MAX thumbnails render. Acceptable
at these frame counts (test clip = 7; cap = 300); thumbnails are drawn downscaled.

### 5.4 Benchmark (`BenchmarkScreen`)

- Loads `assets/phase2-benchmark-result.json` once (`remember`), parsed by `BenchmarkData`.
- **Device header:** *Pixel 7a · Tensor G2 · median-of-3, 1 warm-up discarded · CPU-only · captured
  2026-06-03* — labelled explicitly as captured data, not a live re-run.
- **Per-model groups**, each with rows per lane:
  - **CPU·XNNPACK** and **CPU·EP** (the timed lanes): load ms · vision ms (median) · ms/frame · fps ·
    vision node-coverage % (XNNPACK-delegated vs CPU fallback, joined from `capabilities`).
  - **NNAPI** lanes badged **experimental** with their 0 %-delegated coverage.
- Optional compact bars comparing ms/frame across the 4 models.

---

## 6. Charts (`charts/`, Canvas)

All numeric preparation lives in `ChartData` (pure functions, JVM-tested); the composables only draw.

- **`BarChart`** — input: list of `(label, values: FloatArray, colors)`; draws grouped bars over a
  zero baseline (handles negative raw similarity), an optional threshold guide line, axis ticks, and
  value labels.
- **`TimelineChart`** — input: timestamps, per-label score series, threshold, segments; draws time-x /
  score-y axes, one polyline per label, a dashed threshold line, and shaded segment bands. ≤300
  points and a handful of labels → trivial.

---

## 7. Engine touches (additive, parity-neutral)

Both default to current behavior, so `Benchmark` and all existing Plan-1/2 tests stay unchanged.

1. **`FrameSampler.sample(uri: Uri, fps, maxFrames)` overload** → `MediaItem.fromUri(uri)`. The
   existing `sample(videoPath: String, …)` delegates to it via `Uri.fromFile`. Avoids copying picked
   `content://` videos to cache. Existing `FrameSamplerTest` remains green.
2. **`Engine.scoreFrames(..., onProgress: ((Int, Int) -> Unit)? = null, isCancelled: (() -> Boolean)?
   = null)`** — threaded into `OrtTower.encodeVision`'s per-item / per-batch loop. Progress
   granularity: per-frame under XNNPACK (batch=1), per-batch under CPU·EP. `isCancelled` is checked
   between frames/batches and aborts cleanly (mirrors Python `cancel_event`). Both parameters default
   `null` ⇒ no change to existing callers.

---

## 8. Model provisioning (`ModelRepository`)

- **Bundle root:** `context.filesDir/models/<model_id>/` — Phase 2 verified `getExternalFilesDir(null)`
  is **null** on this device, so internal `filesDir` is used.
- **`scan()`** → for each `models/*/manifest.json`: `ModelBundleManifest.parse(...)` + verify
  `visionFile`, `textFile`, `tokenizerFile`, and any `*DataFile` exist on disk →
  `ModelInfo(id, resolution, precision, ready, missing)`. Readiness = manifest parses + all files
  present. (sha256 verification is deferred — the v1 manifest parser does not expose per-file hashes;
  noted as future work, consistent with the deferred downloader.)
- **One active model at a time**; `Engine` is constructed per run with that bundle's `modelDir` and a
  conservative `visionBatch` per model (so400m ≤ 8 per Spike 0d).
- **Dev provisioning recipe** (the supported path this plan; documented for the operator):
  ```
  adb push <bundle_dir> /data/local/tmp/clipcc_models/
  adb shell run-as com.example.clipcc cp -r /data/local/tmp/clipcc_models/<id> files/models/
  ```
  (The app is debuggable, so `run-as` can write into the app's internal `filesDir`.)

---

## 9. Error handling, cancellation, lifecycle

### Error handling
- **Validation** (no model/video, <1 label, empty contrast group) is inline; Run is disabled with the
  reason — not raised as exceptions.
- **Runtime failures** are caught in the ViewModel and mapped to `RunState.Error(message)` with Retry
  / Back-to-setup actions:
  - decode failure / 0 frames / unhandled HDR → "Couldn't decode video: …".
  - catchable `OrtException` / model-load failure → the message surfaced.
  - missing/corrupt files → caught earlier by `ModelRepository` readiness (never reaches a run).
- **Native OOM caveat (documented, not pretended-handled):** a hard native ORT OOM kills the process
  and cannot be caught in Kotlin. Mitigated by conservative per-model batch (so400m ≤ 8; the live
  XNNPACK path is batch=1 anyway) and one resident model at a time. Stated honestly in the spec and
  in an on-screen note for the largest model.

### Cancellation
- **Cancel** → `Job.cancel()` and the `isCancelled` flag flips true → the engine aborts between
  frames → `RunState.Cancelled` → back to Setup; partial work discarded. Mirrors Python `cancel_event`.

### Lifecycle
- The ViewModel survives configuration changes (rotation); an in-flight run continues and the
  `StateFlow` is re-collected.
- `FLAG_KEEP_SCREEN_ON` is applied while `keepAwake` (i.e. `Running`) and cleared otherwise; the
  Activity observes the flag from the collected state.
- The persistable video URI survives process death (so Setup is restorable); an in-flight multi-minute
  run does **not** survive process death — accepted, since there is no foreground service (lifecycle
  decision). A single shared `OrtEnvironment.getEnvironment()` is reused across runs.

---

## 10. Testing

**Gate (parent §10.3):** all four modes render from real engine output; the benchmark panel shows
timings + coverage + experimental badges.

**Seam:** the ViewModel depends on the injected `Classifier` interface; the real implementation wires
`FrameSampler` + `Engine` + `Scoring`, and tests inject a fake returning canned results. This keeps
ORT/Media3/device out of unit tests.

- **JVM unit (the bulk, fast):**
  - `ScoringPolicyTest` — every constant equals its pinned Python source value (drift guard against
    `temporal_policy.py` / `response.py` / `config.py`).
  - `BenchmarkDataTest` — parse a `phase2-benchmark-result.json` fixture → expected rows, the
    capability join (vision node-coverage %), and the NNAPI experimental flag.
  - `ClassifyViewModelTest` — validation gating; `Idle → Running → Success/Error/Cancelled`
    transitions; `mode → aggregation` dispatch with canned `ScoreMatrices`; contrast pos/neg → correct
    `posCount` + pos-then-neg concatenation.
  - `ChartDataTest` — bar group values (incl. negative raw similarity baseline) and timeline
    series/segment-band prep.
- **Instrumented smoke (1, device):** provision base-256, run the real test clip through the
  ViewModel's real `Classifier` on-device → assert `Success` with non-empty scores and a best match
  (confirms end-to-end wiring; numerical parity is already covered by Plan 1). Optionally a Compose
  `createComposeRule` test driving Setup → Run → asserting Results nodes appear, using a fake fast
  classifier to avoid a 25 s real run.
- **Manual acceptance:** Pixel 7a screenshots of each mode's Results + the benchmark panel.

---

## 11. Scope

**In scope:** the 2-tab app; Setup; live run across all four modes with per-frame progress + cancel;
Results + Canvas charts; the read-only Benchmark panel; `ModelRepository`; `ScoringPolicy`; the two
engine touches (§7); the tests in §10.

**Out of scope (deferred, explicit):**
- the §9 network downloader (HF Xet / resume / free-space / eviction / foreground service);
- foreground service for runs;
- manifest **schema-v2** (retired by Decision 4);
- in-app sha256 verification of bundles;
- model eviction UI;
- live benchmark re-run from the app (the matrix OOMs a single process and takes ~30 min — captured
  data only);
- fp16/fp32 precision toggle (the app uses whatever precision each bundle is provisioned at);
- longer / multi-clip benchmark capture.

---

## 12. Open items to settle in `writing-plans`

- **Task breakdown & order** for subagent-driven-development (likely: deps + theme scaffold →
  `ScoringPolicy` + `ModelRepository` → engine touches (§7) → `Classifier` + ViewModel → Setup →
  Results + charts → Benchmark panel → instrumented smoke), each with its own gate.
- **`Classifier` request/result data shapes** — the exact `ClassifyRequest` / `RunResult` fields.
- **Bar chart raw-similarity rendering** — shared 0..1 axis with a separate similarity scale, or two
  stacked mini-charts (confidence vs cosine) — pick during planning.
- **Wording of the empty-state / provisioning card** copy.
- **Whether the optional Compose `createComposeRule` test is in the gate** or manual-only.
