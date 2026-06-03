# clipCC-Android Phase 3 — Compose UI — COMPLETION REPORT

**Status: COMPLETE & verified on-device (Pixel 7a, Android 16 / API 36). 2026-06-03.**
Executed subagent-driven (fresh implementer per task + controller review + final holistic review).
Spec: `2026-06-03-clipcc-android-ui-design.md` (Rev 3). Plan: `2026-06-03-clipcc-android-ui.md` (17 tasks).
Android project: `/Users/austin/AndroidStudioProjects/ClipCC` (not git; "commit" = save).

## Definition-of-Done results

| DoD item (spec §1 / §10) | Result |
|---|---|
| All four aggregation modes render from real engine output | ✅ Engine path verified on-device (smoke); per-mode aggregation unit-tested (Scoring 27) + ViewModel dispatch test; UI composables build + launch-verified. Per-mode *screenshots* = 1 manual step (SAF pick) |
| Benchmark panel: CPU timed lanes + NNAPI capability-only + coverage % + experimental badges | ✅ `BenchmarkData` (2 tests) + `BenchmarkScreen`; renders 4 model groups on device |
| Long run: per-chunk progress, cancellable, screen stays awake | ✅ `ClassifyViewModel` state machine (8 tests incl. cancel→Cancelled), `RunStatus`, `FLAG_KEEP_SCREEN_ON` via `keepAwake` |
| Engine reused with only additive, parity-neutral touches | ✅ 4 touches; `FrameSamplerTest` + `EndToEndParityTest` (0 flips) + `OrtBackendTest` green AFTER changes |

## Verification evidence

- **JVM unit suite: 59 tests, 0 failures** (`testDebugUnitTest`). Plan-1/2 retained (Resampler 3, Manifest 1, Scoring 27) + 7 new classes: ScoringPolicy 2, ManifestExtension 1, ModelRepository 5, LabelValidation 6, ChartData 3, BenchmarkData 2, ClassifyViewModel 8.
- **Instrumented (device):**
  - `FrameSamplerTest` ✅ (Task 8 — streaming overload additive).
  - `EndToEndParityTest` ✅ + `OrtBackendTest` ✅ (Task 9 — engine chunked primitives additive; cosine within tolerance, 0 best-match flips).
  - `ClassifyEndToEndSmokeTest` ✅ (Task 17 — `RealClassifier` runs real ONNX inference over the test clip: non-empty scores, best-match ∈ [0,1], thumbnails retained, frameCount > 0).
  - `EndToEndParityTest` re-confirmed ✅ after all UI work (still 0 flips).
- **App launches** on device, both tabs render; screenshots: Classify-setup (model + ETA "≈2372 ms/frame"), model dropdown, Benchmark panel.

## What was built (file inventory, `app/src/main/java/com/example/clipcc/`)

- `ui/app/ClipCCApp.kt` — 2-tab scaffold + inline saved-state ViewModel factory (`createSavedStateHandle`).
- `ui/classify/`: `UiBackend`, `ClassifyModels` (DTOs), `Classifier` (seam), `RealClassifier` (chunked pipeline), `ClassifyViewModel`, `LabelValidation`, `SetupCard`, `RunStatus`, `ResultsSection`, `ModeExtras`.
- `ui/benchmark/`: `BenchmarkData` (parser), `BenchmarkScreen`.
- `ui/charts/`: `ChartData`, `BarChart`, `TimelineChart` (Canvas, separate scales + a11y semantics).
- `data/ModelRepository.kt` — scan `filesDir/models/<id>/` → readiness (parse + files-exist + vision/text size-match + semantics gate).
- `engine/ScoringPolicy.kt` — NEW model-independent constants (pinned to Python) + `visionChunkFor`.
- **Engine touches (additive):** `Manifest.kt` (+display_name/score_semantics/bytes/sha256), `FrameSampler.kt` (streaming `sample(uri,fps,max,onFrame)`), `OrtTower.kt` (`encodeVision` onItem/isCancelled), `Engine.kt` (`RunCancelledException` + `encodeTextEmbeddings`/`withVisionEncoder`/`VisionEncoder`).
- `MainActivity.kt` — hosts `ClipCCApp`, keep-screen-on. `assets/phase2-benchmark-result.json` — bundled.
- Deps added: lifecycle viewmodel-ktx/compose/runtime-compose/savedstate + kotlinx-coroutines-test.

## Deviations from plan (all evidence-driven)

1. **`ModelInfo.id = dir.name`** (Task 4), not `manifest.model_id`. The plan's test fixture used dir `m1` ≠ manifest id, and asserted `id=="m1"`; the implementer resolved the inconsistency to `dir.name`. Correct in production (bundle dirs are named after the model id, so `dir.name == model_id`), so `visionChunkFor`/benchmark-ETA lookups still receive the real id string.
2. **lifecycle resolved to 2.8.3**, not the pinned 2.6.1 — transitively bumped by the Compose BOM/activity-compose. Harmless; all `createSavedStateHandle`/viewmodel-compose APIs present.
3. **`@RunWith(AndroidJUnit4::class)`** added to `ClassifyEndToEndSmokeTest` to match repo convention / runner resolution.
4. **`visionChunkFor` takes engine `Backend`** (not `UiBackend`) — keeps the `engine` package UI-agnostic; mapping in `RealClassifier`. Documented in the plan.
5. Toybox `run-as cp -r SRC files/models/` needed an explicit dest dir name; `connectedDebugAndroidTest` uninstalls the app (wipes `filesDir`) → re-provision before screenshots.

## Final holistic review outcome

0 Critical. Confirmed safe: Media3 returns fresh bitmaps per frame (mid-decode recycle safe), ORT sessions close on throw (`.use`), cancel maps to `Cancelled` not `Error`, MAX/temporal/contrast dispatch unambiguous, ViewModel factory caches (no per-recompose scan). **3 findings fixed + re-verified** (build + 59 unit + device smoke):
- Bounded bitmap leak on the partial-decode error path → chunk recycle in `finally`.
- Empty-decode (0 frames) → clean `RunState.Error` instead of NaN.
- Contrast threshold display formatted `%.3f`.

## Carried forward / pending

- **Manual (1 step):** per-mode RESULTS screenshots — pick `/sdcard/Download/clipcc_test.mp4` via the in-app SAF picker, Run each mode (MEAN/MAX/TEMPORAL/CONTRAST). The SAF picker is deliberately not scriptable.
- Deferred (out of scope, spec §11): network downloader (HF Xet/resume/eviction) + its foreground service + full sha256 verify; foreground-service-backed runs; model eviction UI; live benchmark re-run; fp16/fp32 toggle; multi-clip benchmark capture.
- UX nit (non-blocking): Classify model selector does not auto-select the only provisioned model on first launch (it is selectable from the dropdown).
- Residual untested path: mid-decode chunk recycle (chunk fills before EOF) is unexercised on-device because the test clip is 7 frames < chunk 16; argued safe (distinct fresh bitmaps) + now recycle-on-error hardened. A >16-frame clip would exercise it.
