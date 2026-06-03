# clipCC-Android Phase 2 — Benchmark Harness — COMPLETION REPORT

**Status: COMPLETE & verified on-device (Pixel 7a, Android 16 / API 36).** 2026-06-03.
Executed subagent-driven (implementer + spec/quality review per task). Raw merged results:
`docs/superpowers/plans/phase2-benchmark-result.json`.

## Definition-of-Done results

| DoD item | Result |
|---|---|
| `OrtTower` backend-aware (4 configs); CPU_XNNPACK per-frame + CPU_EP batched both correct & mutually consistent | ✅ Task 1 — XNNPACK-vs-CPU_EP embeddings agree to 6.8e-7 (vision) / 1.8e-7 (text) |
| `Engine` backend-parameterized; base-256 E2E cosine ≤0.01 on BOTH lanes vs golden | ✅ Task 2 — XNNPACK 9.09e-5, CPU_EP 9.08e-5 |
| `BackendCapabilityReport` per-(model,tower,backend) provider coverage from untimed profiling; NNAPI outcomes honest | ✅ Tasks 3,7 — 24 reports; numbers independently reproduced from on-device profile files |
| `FrameSampler` decodes real clip single-threaded (count/dims/rotation) | ✅ Task 4 — Media3 `FrameExtractor` 1.10.1 (real API javap-verified); 7 frames @720×1280 SDR |
| `Benchmark` FramePrep-once + per-lane TimedRun, warm-up + median-of-3 + cool-down + thermal/battery metadata | ✅ Tasks 5,7 |
| Results retrieved off-device as JSON (`am instrument` + `run-as` fallback) | ✅ Task 6 |
| Full matrix 4 models × 2 CPU lanes + 4×4 vision + 4×2 text capability; fp16 consistency; report | ✅ Task 7 — prep=4, runs=8, capabilities=24 |

## Headline numbers (median-of-3, 1 warm-up discarded, real Media3-decoded frames)

Vision-encode only (`vision_ms` isolated; decode+preprocess timed separately, see below). frames = 7 (so400m = 4).

| Model | precision | CPU_XNNPACK (per-frame) | CPU_EP (batched) | faster lane |
|---|---|---|---|---|
| base-256 | fp32 | 2372 ms/frame (0.42 fps) | **1202 ms/frame (0.83 fps)** | CPU_EP **1.97×** |
| base-384 | fp32 | 5520 ms/frame (0.18 fps) | **2966 ms/frame (0.34 fps)** | CPU_EP **1.86×** |
| large-384 | fp16 | **10678 ms/frame (0.094 fps)** | 11179 ms/frame (0.089 fps) | XNNPACK 1.05× |
| so400m-384 | fp16 | **17880 ms/frame (0.056 fps)** | 19190 ms/frame (0.052 fps) | XNNPACK 1.07× |

**Key finding — batching helps small fp32, not large fp16.** CPU_EP true batching ≈ **2× faster** than
XNNPACK per-frame on the small fp32 models, but is marginally **slower** on the large fp16 models
(large/so400m), where per-frame XNNPACK edges it out. Median spread was tight (min/max within ~1–2%);
**no run thermally throttled** (`thermalThrottled=false` throughout; fresh process per model + force-stop
between kept heat down). so400m ≈ **18 s/frame** on CPU — matches Spike 0d's ~16 s estimate.

Decode+preprocess (once per model, Media3): decode ≈ 2.1–2.6 s, preprocess ≈ 0.58–0.86 s for the frame set.
`end_to_end_ms` in the JSON is a flagged synthetic sum (decode+preprocess were not re-run inside timed runs).

## Backend capability (the honest "attempt-and-report")

| Backend | Applied | Vision delegated (non-CPU) | Verdict |
|---|---|---|---|
| CPU_XNNPACK | XNNPACK EP | base 11.84% / large·so400m 9.3% to `XnnpackExecutionProvider` (rest CPU) | real, partial accel |
| CPU_EP | ORT CPU EP | 0% (all `CPUExecutionProvider`) | baseline |
| NNAPI_DEFAULT | NNAPI EP added OK, session created OK | **0% delegated** — all nodes fall to `CPUExecutionProvider` | no acceleration |
| NNAPI_CPU_DISABLED | NNAPI EP added OK, session created OK | **0% delegated** | no acceleration; never relabeled |

Text tower XNNPACK delegated 13.3% (base) / 7.5% (large·so400m). **NNAPI provides zero acceleration** on
Tensor G2 for these models — `addNnapi`/`createSession` succeed but the profiler shows 0 nodes on any
NNAPI provider (the run silently falls back to CPU). Reported honestly via the node-coverage profile, not
relabeled. Confirms Spike 0b (NNAPI 0%) across all 4 models, both precisions.

## fp16 consistency (XNNPACK vs CPU_EP cosine, per model)
base-256 1.23e-7 · base-384 1.23e-7 · large-384 **0.00242** · so400m-384 **0.00970** — all ≤ 1e-2.
fp32 models are essentially identical across EPs; fp16 models diverge more (so400m closest to the bound at
0.0097), a genuine fp16 cross-EP numerical difference — within tolerance but worth tracking.

## Notable deviations from the plan (all evidence-driven)
1. **Matrix redesigned to one process PER MODEL.** The plan's single-process all-4-model matrix
   **OOM-crashed** (`INSTRUMENTATION_RESULT: shortMsg=Process crashed`) on so400m after ~30 min of
   accumulated native ORT memory — a native death no Kotlin try/catch can catch. Fix: `BenchmarkMatrixTest`
   now benchmarks ONE model per `am instrument` invocation (`-e model <id>`, fresh process), writing a
   per-model `benchmark_<id>.json`; the host driver runs it 4× and merges. `Benchmark.writeResults` gained a
   `fileName` param. Each model gets a clean ~7.6 GB → no OOM.
2. **`getExternalFilesDir(null)` is null on this device** → results written to internal `filesDir`, retrieved
   via `adb shell run-as` (the plan's sanctioned fallback). `adb pull` of the internal path is denied.
3. **Media3 `FrameExtractor` real API** differs from the plan skeleton: `close()` (AutoCloseable), not
   `release()`; `getFrame(long): ListenableFuture<Frame>`; `Frame.bitmap`/`presentationTimeMs` public fields.
   PTS-based EOF termination added (clip is 5.94 s → 7 frames at 1 fps).
4. **JSON serializer hardening** (caught in review before the matrix): `Locale.US` float formatting + escape
   exception strings via `JSONObject.quote()` — required because NNAPI capability rows can carry quoted
   native error text and non-US locales use comma decimals (either → invalid JSON).
5. **Timed-region hygiene:** the per-run pixel-buffer allocation was hoisted out of the timed window so
   `vision_ms` reflects inference, not buffer marshalling.

## Test clip / device caveats
- Clip `/data/local/tmp/clipcc_bench/test.mp4` = 720×1280, h264, 5.94 s, `color_transfer=bt709` (SDR), no
  rotation tag → 7 frames at 1 fps (so400m capped at 4 by design). A longer clip would give more samples;
  7/4 frames is sufficient for stable ms/frame here.
- Numbers are CPU-only (XNNPACK + ORT CPU EP). No GPU/NPU acceleration exists on Tensor G2 for these models.

## Carried to Plan 3 (UI) / later
- Compose UI + benchmark panel consuming the JSON; NNAPI rows badged experimental.
- Full network downloader (Xet/resume) — adb-push sufficed here.
- Optional fp32 transformers goldens for base-384/large/so400m to measure true fp16 drift vs the reference
  (this phase verified XNNPACK-vs-CPU_EP consistency, not vs a fp32 golden, for the fp16 models).
- Longer/multiple benchmark clips; CPU clock-locking (needs userdebug/root) for lower variance.
- Manifest schema-v2 (temporal/contrast defaults) when the UI consumes those modes.
