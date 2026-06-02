# clipCC-Android Phase 2 — Benchmark Harness — Design (delta)

**Date:** 2026-06-03 · **Status:** Approved (pre-plan), tightened after design review · **Target:** Pixel 7a (Tensor G2, 8 GB)
**Builds on:** master design `2026-06-02-clipcc-android-design.md` (§3, §5.4, §5.5, §10 phase 2) and the
completed Plan 1 headless engine (`phase1-report.md`). This is a **delta**; where it differs from the
master spec for Phase 2, this doc wins.

## Goal
A **headless** benchmark harness measuring on-device SigLIP2 **inference** speed for the 4
`benchmark-v1` models across backends, over a real decoded video, with honest backend-capability
evidence and benchmark hygiene. No UI (Phase 3). Output = structured metrics retrievable off-device +
an instrumented test that runs the matrix.

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
1. **Backends are ORT EP/flag configs, not GPU-vs-NPU.** ORT-Java exposes NNAPI as ONE EP with
   `NNAPIFlags`; there is no GPU/NPU selection (the NNAPI runtime picks hardware opaquely). Lanes:
   - `CPU_XNNPACK` — `addXnnpack`, **per-frame** vision (batch-collapse).
   - `CPU_EP` — no EP (pure ORT CPU), **batched** vision.
   - `NNAPI_DEFAULT` — `addNnapi(EnumSet.noneOf)` (CPU fallback allowed).
   - `NNAPI_NO_CPU` — `addNnapi(EnumSet.of(CPU_DISABLED))` (forces HW; expected to fail/partition→fallback).
   Every result records `requested` + `applied EP` + (for NNAPI) "actual hardware target is opaque." **Never relabel.**
2. **Capability evidence = per-model untimed probe, not "base-256 representative."** fp32 base and fp16
   large/so400m can partition differently, so run a **cheap one-frame, profiling-ON, UNTIMED** session
   per (model, tower, backend) to emit a per-model `BackendCapabilityReport`. Timed runs (below) have
   **profiling OFF** so it never perturbs timings.
3. **Component timing breakdown** (the goal is inference speed, not blended end-to-end). Per run record:
   `decode_ms`, `preprocess_ms`, `text_ms`, `vision_ms`, `scoring_ms`, `end_to_end_ms`. **Decode +
   preprocess the fixed frame set ONCE**, cache the pixel tensors, and reuse them across CPU_XNNPACK vs
   CPU_EP so the EP comparison isolates `vision_ms`. (Media3 decode is measured once per model, reported
   separately, not re-run per lane.)
4. **Benchmark hygiene** (phone clocks throttle; median-of-3 alone is not enough). AndroidX Microbenchmark
   is unsuitable here (designed for sub-ms ops with many iterations; so400m is ~16 s/frame), so adopt its
   *principles* in our harness: **1 warm-up (discarded) + median of 3 timed** per (model, lane); **cool-down
   sleep between runs**; record `RunMetadata` = thermal status (`PowerManager.getCurrentThermalStatus`),
   battery %, charging state, run-order index, wall-clock; **flag any run whose thermal status ≥ MODERATE**.
   Report median + min/max (variance). **Separate a fast smoke test** (base-256, CPU_EP, 2 frames, 1 run)
   from the long full-matrix test.
5. **Fixed preselected batches, not auto-shrink-as-protocol.** Auto-shrink-on-OOM alters the measured
   protocol and risks LMK process kills. Preselect per-model batch for CPU_EP (base ≤16, large ≤8, so400m
   ≤4), run **fixed-size chunks**, and record `effective_batch_size`. Auto-shrink remains only a last-resort
   safety that **fails the run loudly** (not silently re-measured).
6. **Pinned `BackendConfig`** for fair comparison: `intraOpThreads` (=4 for both CPU lanes), `interOpThreads`,
   `graphOptLevel`, `memoryPatternOptimization`. Recorded in every `BenchmarkResult`.
7. **Frame set + protocol numbers.** Fixed frame set ~16 (so400m ~4) decoded from a pinned SDR clip.
8. **Architecture: refactor, not "reuse unchanged."** `Preprocess` / `Scoring` are reused unchanged.
   `OrtTower` is **refactored** to a backend-aware session factory + explicit encode strategies
   (per-item vs batched); `Engine` is **parameterized by backend** (its current hardcoded XNNPACK per-label
   workaround moves into the strategy). New classes: `Benchmark`, `FrameSampler`, `BackendCapabilityReport`,
   `BackendConfig`, `BenchmarkResult`/`RunMetadata`.

## Components
- **`OrtTower` (refactored).** `OrtTower.open(path, env, backend, config)`; SessionOptions per backend
  (XNNPACK / none / NNAPI-with-flags inside try/catch → catch, log reason, fall back to CPU, never relabel).
  Encode strategy: `BatchedVision` (CPU_EP, fixed chunk) vs `PerFrameVision` (XNNPACK; the existing
  `check(rows)` guard enforces it). Text uses the per-item rule under XNNPACK, batched under CPU_EP.
- **`BackendCapabilityReport`** (untimed). requested backend, applied EP, **node coverage** (count + % of
  graph nodes per provider, parsed from ORT `enable_profiling` JSON — Spike 0b proved readable, per tower),
  NNAPI flag set, fallback reason. Produced once per (model, tower, backend) in the untimed probe pass.
- **`FrameSampler`** (Media3 `androidx.media3:media3-inspector-frame` `FrameExtractor`). **Single dedicated
  thread** (FrameExtractor must be accessed from one thread); fps=1; `approx_timestamp = i/fps`; frame cap;
  apply rotation metadata; **SDR / limited-range color policy**, recorded. `@UnstableApi` → pinned version
  behind our interface. Records video color metadata, rotation, seek policy in the result.
- **`Benchmark`** runner. Drives the lane matrix with the protocol; memory order per master §5.5 (text first,
  release text session, then vision; one large session resident at a time). Collects `BenchmarkResult`.
- **Output / retrieval.** `connectedAndroidTest` auto-uninstalls the app (wipes app data), so the benchmark
  runs via **`adb shell am instrument`** (install test+app, run, do NOT auto-uninstall): write the
  `BenchmarkResult` JSON to the app external files dir, **`adb pull`** it after the run; ALSO emit a compact
  summary via the instrumentation status `Bundle`. Logcat = human summary only (rotation-safe), not the
  source of record.

## Acceptance gate (instrumented, on device)
- All 4 models complete both CPU lanes (`CPU_XNNPACK` per-frame + `CPU_EP` batched) with component timing
  (`vision_ms` isolated via shared cached tensors), `BackendConfig`, `effective_batch_size`, and `RunMetadata`.
- Per-model `BackendCapabilityReport`s from the untimed probe for all 4 lanes incl. `NNAPI_DEFAULT` +
  `NNAPI_NO_CPU` (applied EP + node coverage + fallback reason); no relabeling.
- Reproducibility: warm-up + median-of-3 + variance reported; any thermally-throttled run flagged.
- **Parity not broken:** base-256 end-to-end cosine still matches `scores_golden.json` within Plan-1
  tolerance on the `CPU_EP` lane (EP/batch refactor must not change scores).
- **fp16 coverage:** per model, `CPU_XNNPACK` vs `CPU_EP` cosines agree within tolerance (catches EP/precision
  bugs on the fp16 large/so400m lanes). *(Optional host stretch: generate fp32 transformers goldens for
  base-384/large/so400m to measure true fp16 drift — internet available; deferred unless drift validation wanted.)*
- Results retrieved off-device as JSON (survives the test lifecycle).

## Staging (device, offline OK — onnxruntime cached)
- adb-push the 3 not-yet-staged bundles → `/data/local/tmp/clipcc_models/<model_id>/` (base-384, large-384,
  so400m-384; base-256 already there). ~7 GB total on /data/local/tmp.
- Pin an SDR clip: pull `/sdcard/Movies/FlexibilityCC/FlexibilityCC_20260513_091459.mp4`, **verify SDR /
  color range**, push to `/data/local/tmp/clipcc_bench/test.mp4`.

## Prerequisites / notes
- **Media3** (`media3-inspector-frame`, pin ≥1.9.0; verify exact artifact/version resolves) is NOT in the
  Gradle cache → its build needs network (internet available as of 2026-06-03; confirmed Google Maven reachable).
  Everything else builds offline (onnxruntime cached).
- so400m/large are fp16; treat fp16 numerics per the fp16-coverage gate above.

## Out of scope (→ Phase 3 / later)
- Compose UI + benchmark panel; full network downloader (Xet/resume — adb-push suffices here); manifest
  schema-v2 temporal/contrast fields; GPU via non-NNAPI paths; locking CPU clocks (needs userdebug/root).
