# clipCC-Android Phase 0 — Spike Results

**Date:** 2026-06-03
**Device:** Pixel 7a, **Android 16**, Google Tensor G2, 8 GB RAM (`adb` id `36161JEHN16600`)
**Android project:** `/Users/austin/AndroidStudioProjects/ClipCC` — module `:app`, package
`com.example.clipcc`, minSdk 24 / target+compile 36, AGP 9.0.0-rc01, Kotlin 2.0.21. **Not a git
repo** (spike test files live in `app/src/androidTest/java/com/example/clipcc/spike/`, not
committed; results captured here).
**Runtime:** `com.microsoft.onnxruntime:onnxruntime-android:1.26.0` (added as
`androidTestImplementation`). Test APK + all spikes compiled clean against the ORT Java API on
AGP 9 — the API surface used by the plan (`addXnnpack`, `addNnapi`, `enableProfiling`,
`endProfiling`, `OnnxTensor.createTensor(FloatBuffer, long[])`, `session.inputNames`) is valid.

## On-device test-data staging (lesson for Plan 1/2 instrumented tests)
- `/sdcard/Android/data/<pkg>/files/` is **not usable**: adb-pushed files there gave **EACCES**
  to the app process (FUSE/scoped-storage), and `connectedAndroidTest` **uninstalls the app
  afterward, wiping that dir**.
- **Push model files to `/data/local/tmp/clipcc_spike/`** — the app can **read** them there.
- The app **cannot write** to `/data/local/tmp` (SELinux; `enableProfiling`/log writes → ENOENT).
  Write ORT profiles + logs to the app's `cacheDir`
  (`InstrumentationRegistry.getInstrumentation().targetContext.cacheDir`).
- Capture spike output from **logcat** (`System.out` tag); for long runs (>~5 min) logcat
  rotates, so keep on-device runs short or stream logcat live.

---

## Spike 0a — Tokenizer lowercasing  ✅ RESOLVED
**Question:** does the Rust `tokenizers` path (on the shipped `tokenizer.json`) match the
Python `AutoProcessor`, and does anything lowercase?

**Result:** `DECISION lowercase_applied_by = tokenizer_json`, `parity_ok = True` (9/9 cases incl.
`Car`/`CAR`/`ALLCAPS PHRASE`). `rust_lowercases = False`, `normalizer_lowercases = False`,
`matches_raw = True`, `matches_lower = False`.

**Conclusion:** SigLIP2's fast tokenizer (`GemmaTokenizerFast`) is **case-sensitive** — no
Lowercase normalizer in `tokenizer.json`, and `AutoProcessor` does not lowercase
(`AutoProcessor("Car")`=`[3726,1]`, `("car")`=`[2269,1]`, `("CAR")`=`[15547,1]`, each matching
`rust.encode(text)`). **The Android tokenizer wrapper must NOT lowercase** (it would break
parity); it only truncates to 64 + pads with 0. Corroborated by `app/models/siglip2_model.py:82`.
*(Note: run in `.venv-export` transformers 4.57.6; the shipped `tokenizer.json` is version-
independent, so the conclusion holds for the v5 server too — both use the fast path.)*

## Spike 0b — ORT node coverage / BackendCapabilityReport  ✅ RESOLVED
**Question:** can we read per-node EP coverage from ORT's profiling JSON on Android?

**Result (base-256 vision tower):**
- `XNNPACK`: available=true, **760 nodes, 90 on XNNPACK (11.8%), 670 on CPUExecutionProvider**.
- `NNAPI`: available=true, **475 nodes, ALL on CPUExecutionProvider, 0.0% delegated**.

**Conclusion:**
- The profiling JSON **does** expose a per-node `args.provider` field → `BackendCapabilityReport`
  (% delegated) is implementable by parsing it (dedupe `_kernel_time`/`_fence_*` to one record
  per node). This is the mechanism Plan 2 should use.
- **NNAPI delivers ZERO acceleration** for a custom SigLIP2 ViT on Tensor G2 / Android 16 —
  the whole graph falls back to CPU. Empirically confirms the research/spec "NPU unavailable"
  conclusion. Even XNNPACK only delegates ~12% of nodes (rest on the ORT CPU EP).
- Node counts differ per EP (760 vs 475) because partitioning/fusion reshapes the graph; treat
  NNAPI's node-count coverage as a lower bound (it fuses), XNNPACK's per-op count as meaningful.

## Spike 0c — External-data / large-tower load + integrity  ✅ RESOLVED (scoped)
**Result:** so400m fp16 **vision (857 MB)** and **text (1416 MB)** both load in ORT on-device;
**sha256 asserted equal to the manifest** (gated, not just printed); 93.5 GB free. Both
single-file (no `.onnx_data`). The optional fp32-external-data probe was **skipped** (not
staged) — true external-data load + resumable/Xet download remain a **Plan 2** item.

## Spike 0d — so400m memory + batch ceiling  ✅ RESOLVED
**Result (text-first → release → vision, ascending single-batch probes):**
- text loaded+run: peak PSS **~2.94 GB**.
- **both towers resident (overlap): peak PSS ~3.19 GB** — the binding moment; well within 8 GB.
- vision batches **1, 2, 4, 8 all OK**, peak flat at ~3.19 GB.
- (Prior run) **batch 32 thrashed the device into the low-memory-killer** (killed gms/camera/…)
  and the test process died after ~10 min — so the **so400m batch ceiling is ≤ 8**.
- **Timing (dummy tensors, XNNPACK 4 threads): ~16–17 s PER FRAME** for so400m vision
  (batch 1 ≈ 17 s incl. warmup, batch 8 ≈ 135 s). CPU-only big-model inference is
  impractically slow for real video (300 frames ≈ 85 min) — a headline benchmark result.

**Conclusions for Plan 1/2:**
- text-first/cache/**release** strategy is validated; so400m fits 8 GB.
- Default vision batch for so400m ≤ 8 (memory) — but note batch barely changes peak PSS here;
  the batch-32 failure was an activation spike. base/large will tolerate larger batches.
- **ONNX shape-reuse warning** at batch > 1 (`{1,32,1152} != {1,1,1152}`): the prebuilt towers'
  buffers are batch-1-shaped, so ORT reallocates per batch. Plan 1 must verify batched outputs
  are correct (not just non-null) and consider batch=1 if reuse hurts; validate against the
  scores golden fixture.

---

## Updates folded into the spec (`2026-06-02-clipcc-android-design.md`)
- §5.2 tokenizer → RESOLVED (case-sensitive, `tokenizer_json`, no Android lowercasing).
- §5.3 + risks → resampler is **BILINEAR** (resample=2), not bicubic — but Android
  `createScaledBitmap` (plain 2×2 bilinear, no antialias prefilter) does **NOT** match PIL's
  antialiased bilinear on downscale → Plan 1 ports a custom separable-triangle resampler in
  Kotlin. Residual slow-PIL(host) vs fast-torchvision(server) tolerance = Plan-1 item.
- §3 / §5.5 → NNAPI 0% delegated (CPU-only); so400m peak ~3.19 GB, batch ≤ 8, ~16 s/frame CPU.

## Carried into Plan 1/2 as open items
- Resampler tolerance: pin whether fixtures are generated with slow-PIL or fast-torchvision
  bilinear, and set the Android-bilinear-vs-reference score tolerance.
- Real network downloader (resume / free-space preflight / HF-Xet range-GET) + true
  external-data model load — not de-risked in Phase 0.
- Batched-inference correctness for the prebuilt towers (shape-reuse) + per-model batch defaults.
- so400m CPU latency (~16 s/frame) makes it a "does-it-run" benchmark point more than a usable
  real-time path; the benchmark UI should surface frames/sec honestly.

## Device state
Models remain at `/data/local/tmp/clipcc_spike/` (~2.6 GB: base-256 vision; so400m vision+text+
manifest). Remove with `adb shell rm -rf /data/local/tmp/clipcc_spike` when no longer needed.
