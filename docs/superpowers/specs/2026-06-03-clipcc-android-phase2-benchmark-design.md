# clipCC-Android Phase 2 — Benchmark Harness — Design (delta)

**Date:** 2026-06-03 · **Status:** Approved (pre-plan), tightened after two design-review rounds · **Target:** Pixel 7a (Tensor G2, 8 GB)
**Builds on:** master design `2026-06-02-clipcc-android-design.md` (§3, §5.4, §5.5, §10 phase 2) and the
completed Plan 1 headless engine (`phase1-report.md`). This is a **delta**; where it differs from the
master spec for Phase 2, this doc wins.

## Goal
A **headless** benchmark harness measuring on-device SigLIP2 **inference** speed for the 4
`benchmark-v1` models across ORT backend configs, over a real decoded video, with honest
backend-capability evidence and benchmark hygiene. No UI (Phase 3). Output = structured metrics
retrievable off-device + instrumented tests that run the matrix.

## Model facts (verified from manifests)
base-256 = **fp32** (vision 372 MB / text 1.13 GB) · base-384 = **fp32** (373 MB / 1.13 GB) ·
large-384 = **fp16** (633 MB / 1.13 GB) · so400m-384 = **fp16** (857 MB / 1.42 GB). All **single-file**
(no `.onnx_data`). Two lanes (large, so400m) are fp16 → fp16 coverage is in scope (see Gate).

## Why a reframing (Plan-0/1 evidence supersedes master "4×3, batch 32, 300 frames")
- **NNAPI delegates 0%; XNNPACK ~12% of nodes** (Spike 0b); GPU/NPU structurally unavailable to
  third-party custom models on Tensor G2 (master §3).
- **XNNPACK EP collapses the ONNX symbolic batch dim to 1** for both towers (Plan-1 ERRATA): batching is
  a no-op under XNNPACK; the **CPU EP batches correctly**. Real signal = *XNNPACK per-frame vs CPU-EP batched*.
- **so400m ≈ 16 s/frame** (Spike 0d) → 300 frames impractical for a repeated protocol.

## Decisions (this phase)

### D1 — Backends are ORT EP/flag configs, not GPU-vs-NPU
ORT-Java exposes NNAPI as ONE EP with `NNAPIFlags`; there is no GPU/NPU selection (the NNAPI runtime
picks hardware opaquely). The four configs:
- `CPU_XNNPACK` — `addXnnpack`, **per-frame** vision (batch-collapse). **Timed lane.**
- `CPU_EP` — no EP (pure ORT CPU), **batched** vision. **Timed lane.**
- `NNAPI_DEFAULT` — `addNnapi(EnumSet.noneOf(NNAPIFlags))` (CPU fallback allowed). **Capability probe only.**
- `NNAPI_CPU_DISABLED` — `addNnapi(EnumSet.of(CPU_DISABLED))`. This disables NNAPI's *CPU device* — it is
  **NOT "hardware-only"**: unsupported nodes still fall back to ORT CPU kernels, and session-create may
  fail outright. **Capability probe only;** never a timed lane unless profiling proves real HW coverage.

Every backend result records: `requested`, `addNnapiOutcome` (added/threw+reason), `sessionCreateOutcome`
(ok/threw+reason), `appliedEp`, `providerCounts` (nodes per provider), `delegatedNodes` (count + %), and
— for NNAPI — the note "actual hardware target is opaque to ORT." **Never relabel** a CPU run as GPU/NPU.

### D2 — Capability evidence = separate, bounded probe mode
A **separate runnable mode** (its own instrumented test), with per-session **timeout + progress logging**,
not bundled into the timed benchmark (it churns large so400m sessions). Each probe is **one-frame,
profiling-ON, UNTIMED**. To keep it cheap:
- **CPU lanes** (`CPU_XNNPACK`, `CPU_EP`): probe per (model, **both towers**).
- **NNAPI lanes** (`NNAPI_DEFAULT`, `NNAPI_CPU_DISABLED`): probe per model on the **vision tower only**
  (text-tower NNAPI coverage is not decision-critical; documented as such).
Emits a per-(model, tower, backend) `BackendCapabilityReport`. Timed runs (D3) have **profiling OFF**.

### D3 — Timing model: prep once, time per backend
Split cleanly so the EP comparison isolates compute and there is no decode/preprocess double-count:
- **`FramePrepResult`** — produced **once per (model, resolution)**: Media3 `decode_ms` + `preprocess_ms`,
  yielding the cached pixel tensors for the fixed frame set. Reused across both timed CPU lanes.
- **`TimedRun`** — produced **per backend lane** over the cached tensors: `text_ms`, `vision_ms`,
  `scoring_ms`. (Vision is the headline; profiling OFF.)
- **`end_to_end_ms`** = **composed/synthetic** = `decode_ms + preprocess_ms + text_ms + vision_ms +
  scoring_ms`, explicitly flagged synthetic (decode/preprocess were not re-run inside the timed run).

### D4 — Benchmark hygiene
AndroidX Microbenchmark is unsuitable (sub-ms-op design; so400m is ~16 s/frame), so adopt its *principles*:
**1 warm-up (discarded) + median of 3 timed** per (model, lane); **cool-down sleep between runs**; record
`RunMetadata` = thermal status (`PowerManager.getCurrentThermalStatus`), battery %, charging state,
run-order index, wall-clock, **pinned dependency versions**; **flag any run with thermal status ≥ MODERATE**.
Report median + min/max. A fast **smoke test** (base-256, CPU_EP, 2 frames, 1 run) is separate from the
long full-matrix test.

### D5 — Exact, fixed batch map (no auto-shrink-as-protocol)
CPU_EP vision batch is an **exact map**: `base-256=16`, `base-384=16`, `large-384=8`, `so400m-384=4`. Run
fixed-size chunks; record `effective_batch_size`. **Any deviation from the map is a recorded
failed/adapted run**, not a silently re-measured one. Auto-shrink exists only as a last-resort safety that
**fails loudly**. (CPU_XNNPACK is always per-frame = batch 1.)

### D6 — Pinned `BackendConfig` (fair comparison)
`intraOpThreads = 4` (both CPU lanes), `interOpThreads`, `graphOptLevel`, `memoryPatternOptimization`.
Recorded in every `BenchmarkResult`.

### D7 — Frame set + clip
Fixed frame set: **16 frames** (so400m: **4**) decoded from a **pinned SDR clip**.

### D8 — Architecture: refactor, not "reuse unchanged"
`Preprocess` / `Scoring` reused unchanged. `OrtTower` **refactored** → backend-aware session factory +
explicit encode strategies (per-item vs batched). `Engine` **parameterized by backend** (its current
hardcoded XNNPACK per-label workaround moves into the strategy). New: `Benchmark`, `FrameSampler`,
`BackendCapabilityReport`, `BackendConfig`, `FramePrepResult`, `TimedRun`, `BenchmarkResult`/`RunMetadata`.

## Components
- **`OrtTower` (refactored).** `OrtTower.open(path, env, backend, config)`; SessionOptions per backend
  (XNNPACK / none / NNAPI-with-flags inside try/catch → catch, log reason, fall back, never relabel).
  Encode strategy: `BatchedVision` (CPU_EP, fixed chunk per D5) vs `PerFrameVision` (XNNPACK; existing
  `check(rows)` guard). Text: per-item under XNNPACK, batched under CPU_EP.
- **`BackendCapabilityReport`** (untimed, D2). requested, `addNnapiOutcome`, `sessionCreateOutcome`,
  applied EP, `providerCounts` + `delegatedNodes` (count + %, parsed from ORT `enable_profiling` JSON —
  Spike 0b proved readable, per tower), NNAPI flag set, fallback reason.
- **`FrameSampler`** (Media3 `androidx.media3:media3-inspector-frame:1.10.1` `FrameExtractor`). **Single
  dedicated thread** (FrameExtractor must be accessed from one thread); fps=1; `approx_timestamp = i/fps`;
  frame cap; apply rotation metadata; **SDR / limited-range color policy** recorded. `@UnstableApi` →
  exact pinned version behind our interface. Records video color metadata, rotation, seek policy.
- **`Benchmark`** runner. Drives the matrix with the D3/D4 protocol; memory order per master §5.5 (text
  first, release text session, then vision; one large session resident at a time). Collects results.
- **Output / retrieval (robust).** `connectedAndroidTest` auto-uninstalls (wipes app data), so run via
  **`adb shell am instrument`** (install app+test, run, no auto-uninstall). The instrumentation status
  **`Bundle` carries the absolute result-file path**; the run script **`adb pull`s it immediately**, with
  a fallback **`adb shell run-as <pkg> cat <cacheDir>/benchmark_result.json`**. Logcat = human summary only.

## Acceptance gate (instrumented, on device)
- **Timed (full-matrix test):** all 4 models complete both CPU lanes (`CPU_XNNPACK` per-frame + `CPU_EP`
  batched per the D5 map) with `FramePrepResult` (once per model) + per-lane `TimedRun`
  (`text_ms`/`vision_ms`/`scoring_ms`), `BackendConfig`, `effective_batch_size`, `RunMetadata`
  (incl. thermal flag), median + variance.
- **Capability (probe test):** per-model `BackendCapabilityReport`s — CPU lanes per both towers; NNAPI
  (`NNAPI_DEFAULT` + `NNAPI_CPU_DISABLED`) per model on the vision tower — with `addNnapiOutcome` /
  `sessionCreateOutcome` / `delegatedNodes`; no relabeling.
- **Reproducibility:** warm-up + median-of-3; thermally-throttled runs flagged.
- **Parity not broken:** base-256 end-to-end cosine still matches `scores_golden.json` within Plan-1
  tolerance on `CPU_EP` (refactor must not change scores).
- **fp16 coverage:** per model, `CPU_XNNPACK` vs `CPU_EP` cosines agree within tolerance (catches
  EP/precision bugs on fp16 large/so400m). *(Optional host stretch: fp32 transformers goldens for
  base-384/large/so400m to measure true fp16 drift — deferred unless wanted.)*
- **Retrieval:** result JSON pulled off-device (am instrument + Bundle path; run-as fallback verified).
- **Smoke test** (base-256/CPU_EP/2 frames) passes fast, separate from the full matrix.

## Staging (device, offline OK — onnxruntime cached)
- adb-push the 3 not-yet-staged bundles → `/data/local/tmp/clipcc_models/<model_id>/` (base-384, large-384,
  so400m-384; base-256 already there). ~7 GB total.
- Pin SDR clip: pull `/sdcard/Movies/FlexibilityCC/FlexibilityCC_20260513_091459.mp4`, **verify SDR /
  color range**, push to `/data/local/tmp/clipcc_bench/test.mp4`.

## Prerequisites / notes
- **Media3 pinned EXACTLY `androidx.media3:media3-inspector-frame:1.10.1`** (verified latest on Google
  Maven; the artifact only ships 1.10.x, so the master spec's "≥1.9.0" is superseded). Add to the Gradle
  version catalog; record the resolved version in `RunMetadata`. Not in the Gradle cache → first build
  needs network (available 2026-06-03; Google Maven reachable). Everything else builds offline.
- so400m/large are fp16; treat fp16 numerics per the fp16-coverage gate.

## Out of scope (→ Phase 3 / later)
- Compose UI + benchmark panel; full network downloader (Xet/resume — adb-push suffices); manifest
  schema-v2 temporal/contrast fields; GPU via non-NNAPI paths; locking CPU clocks (needs userdebug/root).
