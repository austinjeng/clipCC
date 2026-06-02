# clipCC-Android — On-Device SigLIP2 Inference & Benchmark App

**Date:** 2026-06-02
**Status:** Approved design (pre-plan)
**Target device:** Pixel 7a (Google Tensor G2, Mali-G710 GPU, 8 GB RAM, EdgeTPU)
**Parent project:** clipCC (FastAPI SigLIP2 video classifier)

---

## 1. Goal

A native Android app (Kotlin + Jetpack Compose, Material 3) that loads a video, runs
SigLIP2 **entirely on-device**, and reproduces the Python clipCC results — best match,
per-label confidence + raw similarity, and all four aggregation modes
(`mean` / `max` / `temporal` / `contrast`) with charts — **plus** a benchmark panel that
times the four models on the phone.

Primary purpose: **measure on-device inference speed of the 4 default SigLIP2 models**.
Secondary purpose: **full functional parity** with the Python web UI.

### Success criteria

1. On-device fp32 scores match the Python reference within a documented float tolerance
   (golden fixtures pass in an instrumented test).
2. All 4 models run to completion on CPU; GPU/NPU are attempted with honest
   "actual-backend-used + fallback-reason" labeling (never silently relabeled).
3. Per-model timing reported per backend: model-load ms, total ms, ms/frame, frames/sec.
4. All four aggregation modes produce the same shaped results as the Python API and
   render in the UI.

### Non-goals (v1)

- No NaFlex (variable-resolution) SigLIP2 variants — fixed-resolution (FixRes) only.
- No server/API; the app is fully standalone.
- No on-device model fine-tuning or training.
- No int8 quantization (moot — there is no usable NPU target; see §3).

---

## 2. Benchmark model profile & numerical contract

**Naming:** these four are a **named benchmark profile** (`benchmark-v1`), *not* "the defaults".
They match neither the Python `default_model_id` (`siglip2-base-patch16-256` only) nor the
`scripts/download_models.py` `DEV_MODELS` preset
(`base-256` / `large-512` / `so400m-384` / `giant-384`). The profile below is the
user-selected set for this app and is pinned by HF revision in the manifest (§5.0).

| Model id | HF repo | Params | Resolution | Device precision |
|---|---|---|---|---|
| `siglip2-base-patch16-256` | `google/siglip2-base-patch16-256` | 0.4B | 256 | fp32 (fp16 optional) |
| `siglip2-base-patch16-384` | `google/siglip2-base-patch16-384` | 0.4B | 384 | fp32 (fp16 optional) |
| `siglip2-large-patch16-384` | `google/siglip2-large-patch16-384` | 0.9B | 384 | fp16 preferred (fp32 ~3.3 GB) |
| `siglip2-so400m-patch14-384` | `google/siglip2-so400m-patch14-384` | 1.0B | 384 | **fp16 required** (fp32 ~4.2 GB OOM-risky) |

### Scoring contract (identical to Python, derived from `logits_per_image`)

SigLIP2 normalizes both embeddings before the matmul, so the model's `logits_per_image`
and a separately-computed normalized cosine are consistent. On-device we compute both from
the **same normalized embeddings**:

```
img = l2_normalize(vision_tower(pixel_values))     # [F, D]
txt = l2_normalize(text_tower(input_ids))          # [L, D]
cosine[f, l]     = dot(img[f], txt[l])             # == raw_similarity (Python)
logit[f, l]      = cosine[f, l] * exp(logit_scale) + logit_bias
confidence[f, l] = sigmoid(logit[f, l])            # per-label, independent (not softmax)
```

- `logit_scale` (learned scalar) and `logit_bias` (learned scalar) are extracted per model
  during export and **baked into the app's per-model config** (do not hardcode a single value).
- Score semantics tag = `siglip2_pairwise_sigmoid` (matches the Python `score_semantics`).
- Decoupled towers: the **vision tower runs once per frame-set** (the benchmark hot path),
  the **text tower runs once per label-set**. This is both efficient and the cleanest
  mapping for the benchmark.

---

## 3. Runtime decision (research-backed)

**Chosen stack:** **ONNX Runtime Mobile** (`com.microsoft.onnxruntime:onnxruntime-android`,
latest stable 1.26.0 as of 2026-05) running **prebuilt SigLIP2 ONNX towers** on the
**XNNPACK CPU** execution provider.

**Why ORT over LiteRT / ExecuTorch:**
- SigLIP2 ONNX export is **already done and published**: `onnx-community/siglip2-*-ONNX`
  repos ship `vision_model.onnx` + `text_model.onnx` + `model.onnx` + quantized variants,
  built with HuggingFace Optimum. Optimum registers `siglip`, `siglip-text`,
  `siglip_vision_model` ONNX configs, so both towers export independently.
- LiteRT (`ai-edge-torch` / `litert-torch`) has **no turnkey SigLIP2** — you must re-author
  the ViT tower from the PaliGemma blueprint, remap HF weights, and verify numerically.
  That is the single largest schedule risk, taken on only to chase an uncertain GPU number.
- XNNPACK CPU is mature, ARM/KleidiAI-optimized, fp32-parity-safe, and on a Pixel 7a is the
  realistic fastest *reliable* path for a transformer anyway.

**Backends on Tensor G2 — one benchmark lane + two experimental attempts (the honest truth):**
> **VERIFIED on-device (Spike 0b, Pixel 7a / Android 16):** XNNPACK delegates only **~12%** of
> the base-256 vision nodes (rest on the ORT CPU EP); **NNAPI delegates 0%** (full CPU fallback).
> Per-node `provider` coverage is readable from ORT's profiling JSON → `BackendCapabilityReport`
> is implementable. See `phase0-spike-results.md`.
- **CPU (XNNPACK) — the benchmark lane.** Real, reliable, fp32-parity-safe. Even here,
  XNNPACK only accelerates *supported* nodes (~12% measured); unsupported nodes fall back to the
  ORT CPU EP. The report must therefore state node coverage, not just "XNNPACK".
- **GPU — experimental attempt, not a peer lane.** ORT has **no first-class Android GPU
  execution provider**; GPU is only reachable via NNAPI, which on Tensor commonly falls back
  to CPU/EdgeTPU rather than Mali, and the ViT graph is documented to fail GPU delegation.
- **NPU/EdgeTPU — experimental attempt, expected to be unavailable.** **Structurally
  unavailable** to third-party custom models on Tensor G2. NNAPI deprecated (Android 15/API
  35); LiteRT NPU delegate is Qualcomm/Intel only; Google's Tensor ML SDK is private-beta and
  Pixel 10 / Tensor G5 only. Any "NPU" run resolves to a CPU/GPU fallback. Strongest,
  adversarially-confirmed research finding.

### Benchmark UX contract: attempt-and-report with provider evidence

Three backend modes `CPU` / `GPU` / `NPU`. CPU is the benchmark; GPU/NPU are labeled
**experimental**. Each run:
1. Attempts to acquire its delegate/EP.
2. On unavailability or apply-failure, **catches it, logs the reason, and runs on the actual
   fallback backend**.
3. Emits a **`BackendCapabilityReport`** (§5.0): requested backend; applied EP/delegate;
   **node coverage** (count + % of graph nodes assigned to the target EP vs CPU fallback,
   read from ORT partitioning / `enable_profiling`); fallback reason. This is the deeper
   truth the benchmark reports — not merely "NNAPI attempted".
4. **Never** relabels a CPU run as "GPU" or "NPU".

Deliverable: **CPU (real, with node-coverage evidence) vs GPU (experimental, real-or-failed-
with-reason + coverage) vs NPU (experimental, always fallback-with-reason)** — defensible and
evidenced, not a fabricated three-way win.

---

## 4. Module architecture (mirrors the Python separation)

| Module | Responsibility | Python analog |
|---|---|---|
| `tools/android_assets/export_models.py` (host) | Per model: download prebuilt onnx-community towers at target precision (fp32 base; fp16 large/so400m) + any `.onnx_data` sibling + `tokenizer.json`; extract `logit_scale`/`logit_bias`; emit `manifest.json` (incl. derived `resample_contract`); optimum-cli fallback only if no prebuilt ONNX. Fixtures via `gen_fixtures.py`. | `scripts/download_models.py` |
| `ModelStore` / `ModelManager` (Kotlin) | One active model at a time, hot-swap; holds ORT vision+text `OrtSession`s; **download-on-demand to app-specific storage** (adb-push fallback); per-model metadata + constants | `models/model_manager.py` |
| `OrtBackend` | Build ORT `SessionOptions` per backend (CPU=XNNPACK EP; GPU/NPU=NNAPI EP attempt); run sessions; surface actual backend + fallback reason | `models/siglip2_model.py` |
| `Tokenizer` (Rust `tokenizers` JNI `.so`) | Load `tokenizer.json`; encode labels → `input_ids[64]`, pad token 0, truncate at 64; byte-exact with HF | processor tokenize path |
| `Preprocess` | Bitmap → RGB → **bilinear** stretch-to-square at model resolution → ×(1/255) → (x−0.5)/0.5 → CHW float tensor | SigLIP image processor |
| `FrameSampler` | Media3 `FrameExtractor`; sample at fps=1.0; cap `max_frames=300`; emit frames + approx timestamps | `services/video.py` + `frame_timeline.py` |
| `InferenceRunner` + `Benchmark` | Batched vision encode (timed), text encode, build `[F × L]` cosine/confidence matrices; collect per-backend metrics | `inference_runner.py` |
| `Scoring` | Port `aggregate_mean` / `aggregate_max` / `aggregate_temporal` / `aggregate_contrast` + `FrameTimeline` + temporal policy + contrast policy | `services/scoring.py`, `frame_timeline.py`, `temporal_policy.py` |
| `ui/` (Compose, Material 3) | Setup → Results → Benchmark screens (see §6) | `static/index.html` |

---

## 5. Sub-system specifications

### 5.0 Generated contract manifest (the parity boundary)

The host pipeline (§5.1) emits **one generated `ModelBundleManifest`** per model that Android
consumes and golden fixtures validate against. This is the single boundary across which
Android may vary implementation but must satisfy a generated contract. Kept lean — every
field maps to a real parity/benchmark need; no speculative fields.

- **`ModelBundle`** — model id, HF repo + pinned revision, precision (fp32/fp16), ONNX file
  names (vision/text) + any external-data files, byte sizes, sha256, resolution, RAM budget.
- **`TokenizerSpec`** — `tokenizer.json` ref + `tokenizer_config.json`-derived settings
  (lowercasing, `padding_side`, special tokens, pad id), `max_length=64`, padding mode,
  truncation. (Lowercasing is in the fast-tokenizer normalizer; captured here for completeness
  and to detect drift.)
- **`FramePipelineSpec`** — sampling fps, timestamp policy, rotation policy, color-range /
  SDR-HDR / color-space policy, and the **pre-scale + JPEG-compatibility choice** (§5.4).
- **`PreprocessSpec`** — resize mode (stretch-to-square), resample kernel (bilinear),
  normalization (mean/std 0.5 → [−1,1]), tensor layout (CHW).
- **`ScoringPolicySpec`** — `siglip2_pairwise_sigmoid` semantics, `logit_scale`/`logit_bias`,
  rounding (6 dp, matching Python), temporal defaults (gap 2.0 / min-dur 1.0), contrast
  defaults + valid `contrast_reduce` modes, exact response shape.
- **`BackendCapabilityReport`** (runtime, not generated) — requested backend, applied
  EP/delegate, node coverage / profiling evidence, fallback reason (§3).

**Schema versioning (resolved in Phase 0 plan).** `manifest.json` carries `schema_version`.
**v1** includes the fields above plus provenance (`hf_revision` for the `google/` source +
`onnx_source_repo`/`onnx_source_revision` for the actual ONNX bytes), `ram_budget_mb`, per-file
`sha256`/`data_sha256`, tokenizer `padding_side`, and an exact `resample_contract` (the real
`AutoProcessor` image-processor params, not the raw `preprocessor_config.json`). v1
**intentionally defers** `FramePipelineSpec` rotation/color-range/SDR-HDR/color-space/timestamp
policy and the full `ScoringPolicySpec` (rounding, temporal gap=2.0/min-dur=1.0, contrast
defaults, response shape) to a **schema v2 bump in Plan 3** — consuming temporal/contrast
defaults requires it. Recorded so the boundary is explicit, not silently incomplete.

### 5.1 Model asset pipeline (host, Phase 0)
- For each of the 4 models: prefer downloading the prebuilt `onnx-community/siglip2-*-ONNX`
  artifacts; if the exact resolution repo does not exist, run
  `optimum-cli export onnx --model google/siglip2-<variant> ...` to produce
  `vision_model.onnx` and `text_model.onnx`.
- Produce **fp32** (parity reference) and **fp16** (so400m mandatory; large preferred).
- Extract `logit_scale` and `logit_bias` scalars from the checkpoint; emit the
  **`ModelBundleManifest`** (§5.0) per model — the generated contract Android + tests consume.
- Generate **golden fixtures** from the exact pinned `transformers` version + checkpoint:
  - `tokenizer_golden.json`: list of `(text, input_ids[64])`.
  - `preprocess_golden.npz`: sample images → expected CHW tensors.
  - `scores_golden.json`: sample (frames, labels) → expected cosine + confidence (fp32).

### 5.2 Tokenizer (byte-exact parity)
- Cross-compile the HuggingFace `tokenizers` Rust crate to an `arm64-v8a` `cdylib` `.so`;
  call via JNI from Kotlin.
- Load **`tokenizer.json`** (the fast artifact `AutoProcessor` loads), **not** `tokenizer.model`.
  Rationale: `Siglip2Tokenizer` subclasses `GemmaTokenizer` (Rust-`tokenizers` BPE,
  `byte_fallback`, normalizer `Sequence([Lowercase(), Replace(" ", "▁")])`). A raw
  SentencePiece `.model` does not reproduce lowercasing / disabled add_dummy_prefix.
- **Parity target is the `AutoProcessor` text path, not the raw tokenizer.** The residual
  processor behavior the Rust library does *not* do — `truncation=True`,
  `padding="max_length"`, `max_length=64`, pad id = 0 — is implemented in Kotlin per
  `TokenizerSpec` (§5.0).
- **RESOLVED (Phase 0 Spike 0a, 2026-06-03):** SigLIP2's fast tokenizer is **case-sensitive**.
  The shipped `tokenizer.json` has **no** `Lowercase` normalizer, and the default
  `AutoProcessor` (`GemmaTokenizerFast`) does **not** lowercase — `AutoProcessor("Car")`,
  `("car")`, `("CAR")` produce distinct ids, each byte-matching `rust.encode(text)`. So
  `lowercase_applied_by = "tokenizer_json"`: **the Android wrapper must NOT lowercase labels**
  (doing so would break parity with the Python reference, which is case-sensitive —
  cf. `app/models/siglip2_model.py:82`). The Kotlin wrapper only does truncate-to-64 + pad-0.
- Android build gotcha: `pthread_cond_clockwait` — fix with
  `CXXFLAGS='-lpthread -D__ANDROID_API__=<level>'`, API ≥ 21.
- **Gate:** instrumented test asserts byte-exact equality against `tokenizer_golden.json`
  (generated from the exact pinned `AutoProcessor` + checkpoint). Pin `transformers`;
  regenerate fixtures on bump.

### 5.3 Preprocessing (SigLIP exact)
From `preprocessor_config.json`:
1. Convert frame to RGB.
2. **Resize to the square model resolution as a non-aspect-preserving stretch** (256×256 or
   384×384), `resample = BICUBIC`. **No** center crop, **no** letterbox.
3. Rescale by `1/255` (`rescale_factor = 0.00392156862745098`).
4. Normalize `(x − 0.5) / 0.5` (mean = std = [0.5, 0.5, 0.5]) → range **[−1, 1]**.
5. Channel-first CHW.

**Resampler (RESOLVED — Spike 0a/Task 8 + Plan-1 grounding, 2026-06-03):** SigLIP2's
`preprocessor_config.json` specifies **`resample=2` (PIL BILINEAR)** for all 4 profile models —
**not bicubic** (an earlier assumption). BUT a **custom Kotlin resampler is still required**:
PIL bilinear is **convolution-based and antialiases on downscale** (and video frames downscale
to 256/384), whereas Android's `Bitmap.createScaledBitmap(filter=true)` is plain 2×2 bilinear
with **no prefilter** → tens-of-LSB drift → **label flips near the 0.5 sigmoid threshold**. So
`createScaledBitmap` is **insufficient**; Plan 1 ports PIL's separable-triangle resize to Kotlin
(per-axis support; antialiased). (The `antialias: null` in `resample_contract.json` is a red
herring — it gated the legacy LANCZOS path, not PIL's always-on triangle prefilter.) Residual
parity risk: host fixtures use **slow PIL** bilinear; the server `.venv` has torchvision 0.27.0
and may use **fast** `SiglipImageProcessorFast` (≈1–2 LSB delta vs PIL). Decision: generate
fixtures with **slow PIL** (deterministic, version-stable); set tolerance accordingly.
`resample_contract.json` carries the authoritative per-model value.

**Parity gotcha (the full Python pixel path is lossy and aspect-preserving):** the Python
reference does *not* feed raw decoded frames into this step. `services/video.py` runs
`ffmpeg ... -vf "fps=N,scale='min(512,iw)':'min(512,ih)':force_original_aspect_ratio=decrease"
-q:v 2` — i.e. it (a) downscales to fit within 512×512 **preserving aspect ratio** and then
(b) re-encodes each frame to **lossy JPEG (q:v 2)** before SigLIP's own bilinear square resize.
So there are two pixel-altering stages upstream of §5.3. Byte-for-byte replication of ffmpeg
swscale + JPEG q:v 2 on Android is impractical. Therefore parity is defined in **two layers**:

- **Model-math parity (must be exact, gated):** generate `preprocess_golden` and
  `scores_golden` by feeding **identical lossless frames** (e.g. pre-extracted PNGs) through
  *both* the Python `__call__`-equivalent (bilinear stretch → normalize) and the Android
  pipeline. This isolates and locks the tensor + model math from frame-decode noise.
- **End-to-end decode parity (documented tolerance, not exact):** the Android `FrameSampler`
  (§5.4) decodes via Media3, not ffmpeg+JPEG, so end-to-end scores carry a documented,
  measured tolerance band.

**DECISION (locked): lossless reference pipeline.** The parity contract is defined against a
**lossless** reference — frames extracted as **PNG with no 512 pre-scale**, fed straight into
SigLIP bilinear stretch-to-square + normalize. The fixture-generation script (host) uses this
lossless path; the Android `FrameSampler` decodes via Media3 and resizes bilinear with **no
pre-scale and no JPEG round-trip**. We do **not** replicate the production `video.py` lossy
ffmpeg+JPEG path. Changing the production `video.py` to also drop the pre-scale/JPEG is
**out of scope** for this app (optional future cleanup on the Python side).
`FramePipelineSpec` records: `prescale=none`, `intermediate_codec=none (lossless)`,
`resample=bilinear`.

### 5.4 Frame extraction
- Use **Media3 `androidx.media3.inspector.frame.FrameExtractor`** (media3-inspector,
  ≥ 1.9.0) — Google's replacement for `MediaMetadataRetriever`.
- Sample at `fps = 1.0`; `approx_timestamp_seconds = sample_index / fps` (matches Python).
- Cap at `max_frames = 300`.
- Do **not** use `MediaMetadataRetriever.getScaledFrameAtTime` (keyframe-snapping +
  slow) or `ffmpeg-kit` (retired, binaries pulled from Maven Central 2025-04).
- Fall back to `MediaExtractor` + `MediaCodec` + OpenGL only if `FrameExtractor` is too slow.

**Caveats to handle explicitly (ffmpeg masks these; Media3 does not):**
- `FrameExtractor` is `@UnstableApi` — pin the Media3 version and wrap it behind our own
  `FrameSampler` interface so an API break is contained to one class.
- It is **single-threaded per instance** — for a 300-frame benchmark, decode sequentially on
  one worker (the cost is part of what we measure) or shard across instances deliberately.
- **Rotation:** apply the video's rotation metadata so portrait clips aren't scored sideways
  (ffmpeg auto-applies display matrix; Media3 must be told).
- **Color:** handle SDR vs **HDR** (tone-map to SDR), color **range** (limited vs full), and
  color space so decoded RGB matches the reference. Record the chosen policy in
  `FramePipelineSpec` (§5.0).
- These decode differences are exactly why end-to-end parity is a documented tolerance, not
  byte-exact (§5.3).

### 5.5 Inference runner & benchmark
- **Memory strategy (two towers + ORT overhead on so400m):** run the **text tower first**
  over the label batch, cache the L2-normalized text embeddings, then **release the text
  `OrtSession`** before creating the vision session. Only one large session is resident at a
  time. Vision **batch size auto-shrinks** on allocation failure (32 → 16 → 8 → 1).
  **OOM fallback:** on `OrtException`/OOM, drop precision toward fp16 (if not already) and/or
  batch 1; if still failing, surface a clear "model too large for this device" error rather
  than crashing. so400m is the binding case — **VERIFIED (Spike 0d)**: peak PSS ~3.19 GB with
  both towers resident (fits 8 GB); vision batches 1–8 OK, **batch 32 thrashes the
  low-memory-killer** → so400m batch ceiling ≤ 8; ~16 s/frame on CPU (see
  `phase0-spike-results.md`).
- Vision: batch frames through `vision_model.onnx` (default batch 32 like Python); time the
  vision pass — the dominant cost and the benchmark's headline.
- Text: run `text_model.onnx` once over the label batch (before vision, per above).
- Build `[F × L]` cosine + confidence matrices per the §2 contract.
- Metrics per (model, backend): model-load ms, total inference ms, ms/frame, frames/sec,
  peak memory (best-effort), plus the `BackendCapabilityReport` (§5.0) — actual EP, **node
  coverage** (% delegated vs CPU-fallback from ORT profiling), and fallback reason.

### 5.6 Scoring port
Port from the **`ScoreBatch` sigmoid tensors only**. **Do NOT port `compute_frame_scores`**
(`services/scoring.py:74`) — it is a stale CLIP-style **softmax** helper used by tests only,
not by any production aggregation (all aggregations read `ctx.confidence` from `ScoreBatch`).
Porting it would inject wrong (softmax, sum-to-1) semantics that contradict SigLIP2's
independent per-label sigmoid. (Pre-existing in the Python repo; left as-is there.)

Port the Python aggregation semantics 1:1, with unit tests mirroring `tests/test_scoring.py`
(minus the `compute_frame_scores` tests):
- `mean` (default): per-label mean confidence + mean raw_similarity; best = argmax confidence.
- `max`: per-label max confidence + peak frame index + approx timestamp.
- `temporal`: detection scores → threshold → segments (gap-merge, min-duration) → segment
  stats (active_avg, interval_avg, coverage_ratio, active_duration) → label summaries
  (segment_count, total_active_duration, duration_weighted_confidence) → best_segment +
  timeline. Honor `threshold` / `gap_tolerance` / `min_duration` with the same defaults
  (2.0 / 1.0) and temporal-policy threshold behavior.
- `contrast`: pos/neg label groups → per-frame group means → frame margins → `contrast_reduce`
  (`mean` / `top_k_mean` / `max` / `quantile`) → video margin → verdict
  (positive / negative / uncertain by ±threshold) → group results + dominant label.

---

## 6. UI (Jetpack Compose / Material 3)

Use the `mobile-android-design` skill during implementation.

- **Setup screen:** model dropdown (params / resolution / precision); backend segmented
  control (CPU / GPU / NPU); video picker; label editor (default = the 3 driving-behavior
  labels); aggregation-mode selector with mode-specific options
  (temporal: threshold / gap / min-duration; contrast: positive/negative label groups +
  reduce mode); Run button.
- **Results screen:** best-match card; per-label confidence + raw-similarity bar chart;
  mode extras — `max`: peak-frame thumbnail + timestamp; `temporal`: timeline line chart +
  segment list + label summaries; `contrast`: verdict banner + group scores + dominant label.
- **Benchmark panel:** per-backend table (load / total / ms-per-frame / fps) with **actual
  backend used + node coverage % + fallback reason** (from `BackendCapabilityReport`); GPU/NPU
  rows badged **experimental**; cross-model comparison view.

---

## 7. Data flow

```
pick model → pick backend → pick video + labels (+ mode options)
  → FrameSampler (fps=1, cap 300)
  → Preprocess (bilinear → CHW [-1,1])
  → vision encode (TIMED)  ──┐
  → text encode (labels)  ───┤
                             ├→ cosine/confidence [F×L]
                             → aggregate(mode)
                             → render results + benchmark metrics
```

---

## 8. Parity validation (acceptance gate)

Two-layer parity (§5.3): **model-math = exact/gated**, **end-to-end decode = documented
tolerance**. Instrumented (on-device) + unit tests:
1. **Tokenizer** — byte-exact `text → input_ids` vs `tokenizer_golden.json` (from `AutoProcessor`).
2. **Preprocess (model-math layer)** — CHW tensor within tolerance vs `preprocess_golden`,
   built from **lossless** frames through both pipelines (validates the bilinear resampler,
   isolated from decode noise).
3. **End-to-end (model-math layer)** — fp32 cosine + confidence within tolerance vs
   `scores_golden.json`, also from lossless frames.
4. **End-to-end (decode layer)** — measured tolerance band for Media3-decoded frames vs the
   ffmpeg+JPEG reference; documented, not gated to byte-exactness.
5. **Aggregation** — Kotlin unit tests ported from `tests/test_scoring.py`.
6. **fp16 models** — validated empirically against their fp32 reference (cosine-sim drift
   within tolerance); hybrid-fp32 sensitive layers if drift is excessive.

---

## 9. Model provisioning (downloader spec)

Models are 1.4–4.2 GB and **cannot be bundled in the APK/AAB**; download-on-demand to
app-specific external storage, mirroring the Python `CLIP_CACHE_DIR` cache pattern. Concrete
requirements (planned before implementation, not left abstract):

- **External ONNX data:** `large-384` (fp32 ~3.3 GB) and `so400m` (fp16 ~2.1 GB) exceed the
  **2 GB ONNX protobuf limit**, so they are exported as `model.onnx` + `model.onnx_data`.
  The downloader must fetch **both** and place the external-data file **beside** the `.onnx`
  with the exact expected filename, or ORT load fails. The manifest (§5.0) lists every file.
- **Resumable downloads:** HTTP range / resume for multi-GB files; survive process death and
  network drops; do not restart from zero.
- **Free-space preflight:** check available bytes ≥ (download size + unpacked size + margin)
  before starting; surface a clear error otherwise.
- **Integrity:** sha256 each file against the manifest after download; reject/redownload on
  mismatch; only mark a bundle "ready" when all files verify.
- **Cache & eviction:** one resident model at a time on disk is *not* required, but provide
  eviction (LRU or manual "remove model") so 4 bundles (~9 GB total) don't wedge the device.
- **Source:** primary = HuggingFace (note: HF now serves large files via **Xet**, not plain
  LFS — verify the Android HTTP client handles the resolve/redirect, or mirror the exported
  bundles to our own object storage with plain range-GET). Decide in planning.
- **adb-push fallback (dev phone):** push exported bundles directly to the app's files dir,
  bypassing network — the fast path for this benchmark phone.
- **Failure recovery:** partial/corrupt downloads are quarantined and retried; the UI never
  shows a half-downloaded model as runnable.

Runtime residency: active model loads one at a time; the text session is released before the
vision pass to bound peak memory (§5.5).

---

## 10. Build order (phases)

**Phase 0 — model/fixture prep + de-risking spikes.** Each spike is a hard gate; the engine
is not "implementation-ready" until all four pass. Spikes are small, throwaway, and answer a
single unknown:

- **0a — Tokenizer spike:** resolve the lowercasing question (§5.2). Compare `AutoProcessor`
  vs Rust `tokenizers` (loading the real `tokenizer.json`) + the Kotlin padding wrapper on a
  mixed-case set (`Car`/`car`, punctuation, multi-word). *Gate: byte-exact, or the exact
  extra normalization step is identified.*
- **0b — ORT coverage spike:** on-device, run one tiny ONNX and one real SigLIP2 vision tower
  under XNNPACK + NNAPI; confirm how to read **node coverage / EP assignment** from the ORT
  **Java** API (profiling JSON or session logs). *Gate: a reliable "% delegated" number, or a
  documented fallback parse.*
- **0c — Downloader spike:** fetch one model **with external `.onnx_data`** (e.g. `so400m`
  fp16) end-to-end: resume, free-space check, sha256, correct co-located placement, ORT load.
  *Gate: a >2 GB external-data model loads in ORT on-device.*
- **0d — Memory spike:** load + run `so400m` fp16 vision tower on the Pixel 7a at batch 32;
  measure peak RSS; confirm the text-first/release strategy (§5.5) keeps it within budget.
  *Gate: so400m completes a 300-frame run without OOM, or the auto-shrink/fallback path is
  proven.*
- **0e — Model + fixtures:** export/obtain the 4 ONNX bundles + emit manifests + golden
  fixtures (lossless-frame based, §8). *Gate: 4 bundles + fixtures exist and verify.*

1. **Headless engine**: ORT sessions + tokenizer JNI + preprocess + frame sampler + scoring
   port. *Gate: parity tests (§8.1–8.3, 8.5) green.*
2. **Benchmark harness**: backend switching + timing + `BackendCapabilityReport`. *Gate: 4
   models × 3 backends produce metrics with correct backend labeling + node coverage.*
3. **Compose UI**: setup → results → benchmark panel wired to the engine. *Gate: all 4 modes
   render; benchmark panel shows per-model/backend timings + coverage + experimental badges.*

---

## 11. Top risks & mitigations

1. **ONNX availability** — `onnx-community` shipped `-224`; the exact 256/384/so400m repos
   must be verified. *Mitigation:* Phase 0 gate; self-export via Optimum if missing.
2. **Resampler parity** — SigLIP2 uses **bilinear** (resample=2), but PIL bilinear
   **antialiases on downscale** and Android `createScaledBitmap` does not → label flips near 0.5.
   *Mitigation:* Plan 1 ports PIL's separable-triangle resize to Kotlin (NOT `createScaledBitmap`);
   fixtures use slow-PIL; residual slow-PIL vs fast-torchvision delta is a tolerance to set;
   validated by fixture #2.
3. **so400m fp16 parity** — no SigLIP2-specific fp16 data exists. *Mitigation:* validate
   empirically per model; hybrid-fp32 sensitive layers if cosine drift exceeds tolerance.
4. **GPU/NPU don't accelerate** — accepted up front. *Mitigation:* attempt-and-report (§3).
5. **Tokenizer version drift** — fast tokenizer has historically diverged. *Mitigation:* pin
   `transformers`; CI gate on golden `(text → input_ids)` set; regenerate on bump.
6. **NNAPI EP deprecation** — building the GPU/NPU attempt on NNAPI is legacy. *Mitigation:*
   it's only the best-effort leg; CPU/XNNPACK is the durable baseline.
7. **Upstream pixel-path mismatch** — Python pre-scales to 512 (aspect-preserving) + lossy
   JPEG before SigLIP resize; Media3 decode differs. *Mitigation:* two-layer parity (§5.3,
   §8) — exact model-math gate on lossless frames; documented tolerance for decode; record
   the chosen pre-scale/JPEG policy in `FramePipelineSpec`.
8. **Media3 `FrameExtractor` is `@UnstableApi` + single-threaded + color/rotation pitfalls.**
   *Mitigation:* pin Media3, wrap behind our `FrameSampler` interface, handle rotation /
   color-range / SDR-HDR explicitly (§5.4).

---

## 12. Open items to resolve during planning

- Charting library for Compose (Vico vs custom Canvas) — decide in the UI phase.
- ONNX batch-axis: fixed batch vs dynamic — fixed/static favored for predictability and any
  future GPU-delegate attempt.
- Whether to ship fp16 for base/large by default or keep fp32 as the shipped precision with
  fp16 as a benchmark toggle.
- ~~**Frame-pipeline parity policy** (§5.3)~~ — **DECIDED: lossless reference** (PNG, no
  pre-scale, bilinear; production `video.py` unchanged, out of scope). See §5.3.

**Planning decisions to settle in `writing-plans` (before the relevant phase):**
- **Project layout:** where the Android project lives (new repo vs `android/` dir in this
  repo) and module structure (`:engine` headless lib, `:app` UI, `:tokenizer` JNI).
- **Manifest JSON schema + versioning:** exact field types + a `schema_version` so Android
  rejects incompatible bundles (mirrors the Python `.validated` marker `schema_version`).
- **Benchmark protocol:** warm-up runs, number of timed iterations, discard-first, mean/median
  + variance reporting, fixed input set — so numbers are reproducible, not single-shot.
- **Long-op lifecycle:** foreground service + notification for multi-GB downloads and long
  inference; cooperative cancellation (mirrors Python's `cancel_event`/timeout) so the user
  can abort a run.
- **Video input:** `content://` URI handling from the picker (persistable permissions,
  copy-to-cache vs stream), and the constraint set (max duration/frames) mirroring `video.py`.
- **Golden-fixture size budget:** keep `tokenizer_golden`/`preprocess_golden`/`scores_golden`
  small enough for the repo + CI (a handful of frames/labels), not full videos.
