# clipCC-Android Phase 2 — Benchmark Harness — Design (delta)

**Date:** 2026-06-03 · **Status:** Approved (pre-plan) · **Target:** Pixel 7a (Tensor G2, 8 GB)
**Builds on:** master design `2026-06-02-clipcc-android-design.md` (§3, §5.4, §5.5, §10 phase 2) and the
completed Plan 1 headless engine (`phase1-report.md`). This is a **delta** — it records the
benchmark-phase decisions and the reframing forced by Plan-0/1 evidence; the master spec covers the
rest. Where they differ, this doc wins for Phase 2.

## Goal
A **headless** benchmark harness that measures on-device SigLIP2 inference speed for the 4
`benchmark-v1` models across backends, over a real decoded video, with honest backend-capability
evidence. No UI (Phase 3). Output = structured metrics + an instrumented test that runs the matrix.

## Why a reframing (Plan-0/1 evidence supersedes the master spec's "4×3, batch 32, 300 frames")
- **NNAPI delegates 0%; XNNPACK ~12% of nodes** (Spike 0b). GPU/NPU are structurally unavailable to
  third-party custom models on Tensor G2 (master §3) — confirmed.
- **XNNPACK EP collapses the ONNX symbolic batch dim to 1** for both towers (Plan-1 ERRATA): batching
  is a no-op under XNNPACK. The **CPU EP batches correctly** (host-verified). So the benchmark's real
  signal is *XNNPACK per-frame vs CPU-EP batched*, not "batch 32 everywhere."
- **so400m ≈ 16 s/frame** (Spike 0d) → 300 frames ≈ 80 min/run, impractical for a repeated protocol.

## Decisions (this phase)
1. **Backend lanes.** Headline = 4 models × **{CPU_XNNPACK (per-frame), CPU_EP (batched)}**. Plus a
   single **GPU_NNAPI** attempt and a single **NPU_NNAPI** attempt on base-256 to produce the honest
   `BackendCapabilityReport` (applied EP, node coverage %, fallback reason), documented as
   representative ("all models fall back identically"). No wasteful 4×dead-backend rows.
2. **Frame source.** Real video via the **Media3 `FrameExtractor`** `FrameSampler` (deferred from
   Plan 1): fps=1, rotation + color/SDR-HDR handling, frame cap, behind our own interface. Benchmark
   times decode + preprocess + encode end-to-end.
3. **Protocol.** Per (model, lane): **1 warm-up run (discarded) + median of 3 timed runs** over a
   small fixed frame set (~16 frames; so400m capped lower, ~4). Report load ms, total ms, ms/frame,
   fps, peak mem, and variance. Reproducible, not single-shot.
4. **Model set.** All 4 `benchmark-v1` (base-256, base-384, large-384, so400m-384) — all already
   exported on host; adb-push to device. so400m runs fewer frames to stay tractable.
5. **Architecture: extend, don't re-abstract.** Add a `Backend` enum + encode strategy to the existing
   `OrtTower`; add small new classes `Benchmark`, `FrameSampler`, `BackendCapabilityReport`. Reuse
   `Engine` / `Scoring` / `Preprocess` unchanged.

## Components
- **`OrtTower` extension.** `enum Backend { CPU_XNNPACK, CPU_EP, GPU_NNAPI, NPU_NNAPI }`. SessionOptions
  per backend (XNNPACK = `addXnnpack`; CPU_EP = no EP; NNAPI legs = `addNnapi` inside try/catch →
  catch, log reason, fall back to CPU; **never relabel**). Encode strategy: **CPU_EP = real batched**
  vision encode (auto-shrink 32→16→8→1 on alloc failure; so400m ceiling ≤8); **XNNPACK = per-frame**
  (the existing `check(rows)` guard enforces it). Text uses the same per-item rule under XNNPACK.
- **`BackendCapabilityReport`** (runtime). Requested backend, applied EP/delegate, **node coverage**
  (count + % of graph nodes on target EP vs CPU fallback, parsed from ORT `enable_profiling` JSON —
  Spike 0b proved readable), fallback reason. The honest deliverable from master §3.
- **`FrameSampler`** (Media3). `androidx.media3:media3-inspector-frame` `FrameExtractor`; fps=1;
  `approx_timestamp_seconds = i/fps`; cap frames; apply rotation metadata; SDR/limited-range color
  policy recorded. `@UnstableApi` → pinned version, wrapped behind our interface.
- **`Benchmark`** runner. Drives the lane matrix with the protocol above; collects per-run
  `BenchmarkResult` (model, backend, applied EP, frames, load ms, median total ms, ms/frame, fps,
  peak mem, variance, BackendCapabilityReport). Memory order per master §5.5 (text-first, release
  text session, then vision; one large session resident at a time).
- **Output.** `BenchmarkResult` list → JSON in app `cacheDir` + a logcat summary table. (Phase-3 UI
  consumes it later.)

## Acceptance gate (instrumented, on device)
- All 4 models complete both CPU lanes (XNNPACK + CPU_EP) with correct backend labels and a node-
  coverage number; GPU_NNAPI + NPU_NNAPI evidence runs on base-256 emit honest
  `BackendCapabilityReport`s (applied EP + fallback reason), never relabeled.
- Numbers reproducible (median + variance reported); CPU_EP batched vs XNNPACK per-frame is quantified.
- **Parity not broken:** end-to-end cosine still matches `scores_golden.json` within Plan-1 tolerance
  on at least the CPU_EP lane (EP/batch changes must not change scores).

## Staging (device)
- adb-push all 4 bundles → `/data/local/tmp/clipcc_models/<model_id>/` (offline OK; onnxruntime cached).
- Pull `/sdcard/Movies/FlexibilityCC/FlexibilityCC_20260513_091459.mp4`, push to
  `/data/local/tmp/clipcc_bench/test.mp4` (or another short clip) for FrameSampler input.

## Prerequisites / notes
- **Media3 dependency** (`media3-inspector-frame`, pin a version ≥1.9.0; master/Plan-1 referenced
  1.10.1) is **not yet in the Gradle cache** → its task needs network on first build (internet is
  available as of 2026-06-03). Verify the exact artifact/version resolves during execution. Everything
  else (backend/EP, batched encode, report, runner) builds offline (onnxruntime cached).
- so400m fp16 vs fp32 parity (master risk 3) is empirical per model; out of this phase's gate unless a
  lane uses fp16.

## Out of scope (→ Phase 3 / later)
- Compose UI + benchmark panel; full network downloader (Xet/resume — adb-push suffices here);
  manifest schema-v2 temporal/contrast fields; GPU via non-NNAPI paths.
