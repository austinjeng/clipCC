# clipCC-Android — Plan 3 (UI) HANDOFF / START HERE

**Read this first**, then the linked artifacts. Plans 0/1/2 are COMPLETE; Plan 3 (Compose UI) is unstarted.

## Read on session start
- Memory: `clipcc-android-port` + `clipcc-android-review-discipline` (auto-loaded).
- `docs/superpowers/specs/2026-06-02-clipcc-android-design.md` §6 (UI) + §10 phase 3 — the UI design.
- `docs/superpowers/plans/phase1-report.md` (engine) + `phase2-report.md` (benchmark) — what's built + hard-won facts.
- `phase2-benchmark-result.json` — the benchmark data the UI panel will display.

## Status
- **Plan 0** (assets + spikes 0a–0d) ✅ · **Plan 1** (headless engine) ✅ · **Plan 2** (benchmark harness) ✅.
- Branch `feat/clipcc-android-phase0` (host repo). Android project `/Users/austin/AndroidStudioProjects/ClipCC` is **NOT git** ("commit" = save file).

## What Plan 3 is (from master spec §6)
Compose / Material 3 UI on top of the existing engine: **Setup** (model dropdown, backend selector, video picker, label editor, aggregation-mode options) → **Results** (best-match card, per-label confidence/raw-similarity chart, mode extras for max/temporal/contrast) → **Benchmark panel** (per-model/backend table: load/total/ms-per-frame/fps + actual-backend + node-coverage % + experimental badges, from the BackendCapabilityReport). Use the `mobile-android-design` + `frontend-design` skills. Likely needs manifest **schema-v2** (temporal/contrast defaults) and a charting choice (Vico vs Canvas — §12).

**Process:** this project uses superpowers brainstorm → writing-plans → subagent-driven-development (implementer + spec/quality review per task). That discipline caught real bugs in Plans 1–2 — keep it.

## Engine API already available (src/main/.../engine/)
- `Engine(modelDir, manifest, env, backend, visionBatch, config).scoreFrames(bitmaps, labels): ScoreMatrices` (cosine/confidence [F×L]).
- `Scoring.aggregateMean/Max/Temporal/Contrast(...)` + `FrameTimeline` + response data classes (ScoreItem, Segment, LabelSummary, ContrastResult…).
- `Benchmark` (FramePrep + TimedRun + RunMetadata), `BackendCapabilityReport`, `FrameSampler` (Media3), `OrtTower` (4 backends), `ModelBundleManifest.parse`, `HfTokenizer`, `Preprocess`, `Resampler`.
- Tests: JVM unit (`testDebugUnitTest`: Resampler 3, Manifest 1, Scoring 27) + instrumented (Tokenizer/Preprocess/OrtBackend/EndToEndParity/BackendCapability/FrameSampler/BenchmarkSmoke/BenchmarkMatrix/Fp16Consistency).

## Device state (Pixel 7a, adb 36161JEHN16600, API 36)
- 4 model bundles at `/data/local/tmp/clipcc_models/<model_id>/` (base-256/384 fp32, large-384/so400m-384 fp16; all single-file).
- Test video `/data/local/tmp/clipcc_bench/test.mp4` (720×1280 SDR, 5.94 s).
- App `com.example.clipcc` installed; instrumented results retrieved via `adb shell run-as` (external files dir is null on this device).

## Hard-won facts — do NOT relearn (see phase reports for detail)
- ORT embedding = `pooler_output` by NAME (not `res[0]`). **XNNPACK collapses batch→1** → per-item; CPU_EP batches.
- **NNAPI 0% delegated** on Tensor G2 (all 4 models) — no GPU/NPU accel; CPU only. Report it honestly.
- CPU_EP ~2× faster than XNNPACK on small fp32; ~even on large fp16. so400m ≈ 18 s/frame.
- **so400m OOMs a single multi-model process** → benchmark runs one `am instrument` per model (fresh process), merged host-side.
- Build deps: `onnxruntime-android:1.26.0` + `media3-inspector-frame:1.10.1` (both `implementation`); `org.json:json` (testImplementation, JVM stubs). minSdk-24 rewind gotcha: `(buf as java.nio.Buffer).rewind()`.
- JSON hand-serialization needs `Locale.US` floats + `JSONObject.quote()` for free-text.
- env: `JAVA_HOME=/Applications/Android Studio.app/Contents/jbr/Contents/Home`; adb `~/Library/Android/sdk/platform-tools/adb`. Host venv `.venv-export` (transformers 4.57.6) for fixtures.

## To start Plan 3 in a fresh session
"Continue clipCC-Android — start Plan 3 (UI). Read docs/superpowers/plans/phase3-handoff.md." Then brainstorm the UI scope (charting lib, schema-v2, offline-model UX, which screens first) before writing the plan.
