# clipCC-Android Phase 1 — Headless Engine — COMPLETION REPORT

**Status: COMPLETE & verified on-device (Pixel 7a, Android 16 / API 36).** 2026-06-03.
Executed via subagent-driven-development (implementer + spec/quality review per task). All gates passed offline.

## Definition-of-Done results

| DoD item | Result | Evidence |
|---|---|---|
| Tokenizer `.so` builds; on-device byte-exact vs Python golden | ✅ | Task 1 (prior session) — `TokenizerParityTest` green |
| Resampler JVM tests pass | ✅ | `ResamplerTest` 3/3 (incl. 4→1 antialias guard = 0.833) |
| Preprocess CHW matches PIL golden ≤ 0.016 on device | ✅ | `PreprocessParityTest` **max_abs_diff = 0.007828** (~2× margin) |
| Manifest parse (schema v1) | ✅ | `ManifestTest` 1/1 (byte-identical to real manifest; `isNull` null-handling) |
| ORT towers L2-normalized embeddings; batched==single | ✅* | `OrtTowerTest` 2/2 — dim 768, ‖v‖≈1; row→frame mapping bit-exact (0.0); distinct frames differ 0.0117 (host truth 0.0115). *See XNNPACK caveat below.* |
| Scoring port passes all `test_scoring.py`-equiv invariants | ✅ | `ScoringTest` 27/27 (mean/max/temporal end-clamp/contrast/reduce/DWC) |
| End-to-end on-device scores match `scores_golden.json` ≤ 0.02, no flips | ✅ | `EndToEndParityTest` — **cosine_max = 9.09e-5** (≤0.01), **confidence_max = 6.14e-7** (≤0.02), **best_match_flips = 0** |

Full suite green offline: **unit 31/31** (Resampler 3, Manifest 1, Scoring 27) + **instrumented 4 engine classes** (Tokenizer, Preprocess, OrtTower×2, EndToEnd), 0 failures.

## ⚠️ Resolved-facts ERRATA (corrected during execution — supersede the plan/spec)

Two plan "verified-correct, do not refactor" claims were **wrong** and were corrected with hard evidence:

1. **Embedding output = `pooler_output` (by NAME), NOT `res[0]`.** Both ONNX towers output `[last_hidden_state, pooler_output]` in that order, so `res[0]` is `last_hidden_state` ([B,256,768]/[B,seq,768]) — the wrong tensor. Host check: normalizing **`pooler_output`** reproduces the golden cosine to **1.96e-7**; `res[0]` is the wrong shape entirely. `OrtTower.runEmbed` now selects `res.get("pooler_output")`.

2. **XNNPACK EP collapses the dynamic batch dim to 1 — for BOTH towers.** This model's `pooler_output` batch dim is symbolic (`floor(batch*…)/256`); with `addXnnpack`, an N>1 input returns `[1,768]` even in a fresh session. CPU EP resolves it correctly (host-verified bit-exact batch==single). So the plan's "ORT batched output is correct" holds **only for CPU EP**. Mitigation: run **batch=1 per item** and stack — `encodeVision` loops per-frame, `Engine` loops per-label for text. `OrtTower.runEmbed` has a `check(rows)` guard that makes silent shape-collapse impossible (it caught the text-tower case during Task 7). Real-batch throughput (if wanted) requires the CPU EP — a perf-vs-EP decision for Plan 2.

3. **Build wiring:** ONNX moved from `androidTestImplementation` → `implementation` so `OrtTower`/`Engine` live in `src/main` (the plan's intended location) and the app bundles the ORT native libs.

## Per-device color caveat
On the Pixel 7a, `BitmapFactory` PNG decode + PIL-bilinear preprocess parity is well within tolerance (max 0.0078), and end-to-end cosine drift is 9e-5 (XNNPACK fp rounding). No per-device color anomaly observed. Real-video decode (Media3) color parity is still deferred to Plan 2.

## File inventory (Android project, `app/src/`)
- `main/.../engine/`: HfTokenizer.kt, Resampler.kt, Preprocess.kt, Manifest.kt, Scoring.kt, **OrtTower.kt**, **Engine.kt**
- `test/.../engine/`: ResamplerTest, ManifestTest, ScoringTest  (+ `test/resources/manifest_base256.json`)
- `androidTest/.../engine/`: TokenizerParityTest, PreprocessParityTest, OrtTowerTest, EndToEndParityTest
- `androidTest/assets/fixtures/`: tokenizer_golden.json, frame_000/001.png, preprocess_golden.json, scores_golden.json
- `build.gradle.kts`: `testImplementation("org.json:json:20240303")` (JVM unit tests), `implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")`
- device bundle `/data/local/tmp/clipcc_models/siglip2-base-patch16-256/`: manifest.json, tokenizer.json, vision_model.onnx, text_model.onnx

## Carried to Plan 2 / housekeeping
- **Housekeeping (non-blocking):** three Phase-0 `androidTest/.../spike/` test classes remain (DownloaderSpikeTest, OrtCoverageSpikeTest, So400mMemorySpikeTest). They pad an *unfiltered* `connectedAndroidTest` run by ~4.5 min (so400m memory spike alone = 259 s) and would error if `/data/local/tmp/clipcc_spike/` is removed. Recommend deleting now that Phase 0/1 are closed.
- Consider pushing the per-item batch loop into `OrtTower.encodeText` (mirror `encodeVision`) for API symmetry.
- so400m text-first/release wiring, real network downloader, FrameSampler (Media3) + real-video color parity, CPU-vs-XNNPACK batch perf — all per the plan's open items.
