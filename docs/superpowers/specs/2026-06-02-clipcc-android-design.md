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

## 2. The 4 models & numerical contract

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

**Backends available on Tensor G2 (the honest truth):**
- **CPU (XNNPACK):** real, reliable, fp32-parity-safe. The baseline and only "real" leg.
- **GPU:** ORT has **no first-class Android GPU execution provider**; GPU is only reachable
  via NNAPI, which on Tensor commonly falls back to CPU/EdgeTPU rather than Mali. Treat as
  best-effort/attempt-and-report.
- **NPU/EdgeTPU:** **structurally unavailable** to third-party custom models on Tensor G2.
  NNAPI deprecated (Android 15/API 35); LiteRT NPU delegate is Qualcomm/Intel only; Google's
  Tensor ML SDK is private-beta and Pixel 10 / Tensor G5 only. Any "NPU" run resolves to a
  CPU/GPU fallback. This is the strongest, adversarially-confirmed research finding.

### Benchmark UX contract: attempt-and-report

Three backend modes `CPU` / `GPU` / `NPU`. Each:
1. Attempts to acquire its delegate/EP.
2. On unavailability or apply-failure, **catches it, logs the reason, and runs on the actual
   fallback backend**.
3. Reports the **actual backend used** plus the reason, e.g.
   `"NPU unavailable on Tensor G2 (no public delegate) — ran on CPU"` or
   `"GPU delegate failed to apply on ViT graph — ran on CPU"`.
4. **Never** relabels a CPU run as "GPU" or "NPU".

Deliverable: **CPU (real) vs GPU (real-or-failed-with-reason) vs NPU (always
fallback-with-reason)** — a defensible, accurate result, not a fabricated three-way win.

---

## 4. Module architecture (mirrors the Python separation)

| Module | Responsibility | Python analog |
|---|---|---|
| `tools/export_models.py` (host) | Per model: obtain ONNX (download `onnx-community/...` else `optimum-cli export onnx`); emit vision+text `.onnx` (fp32; fp16 for large/so400m), `tokenizer.json`, `preprocessor_config.json`, extracted `logit_scale`/`logit_bias`; generate golden parity fixtures | `scripts/download_models.py` |
| `ModelStore` / `ModelManager` (Kotlin) | One active model at a time, hot-swap; holds ORT vision+text `OrtSession`s; **download-on-demand to app-specific storage** (adb-push fallback); per-model metadata + constants | `models/model_manager.py` |
| `OrtBackend` | Build ORT `SessionOptions` per backend (CPU=XNNPACK EP; GPU/NPU=NNAPI EP attempt); run sessions; surface actual backend + fallback reason | `models/siglip2_model.py` |
| `Tokenizer` (Rust `tokenizers` JNI `.so`) | Load `tokenizer.json`; encode labels → `input_ids[64]`, pad token 0, truncate at 64; byte-exact with HF | processor tokenize path |
| `Preprocess` | Bitmap → RGB → **bicubic** stretch-to-square at model resolution → ×(1/255) → (x−0.5)/0.5 → CHW float tensor | SigLIP image processor |
| `FrameSampler` | Media3 `FrameExtractor`; sample at fps=1.0; cap `max_frames=300`; emit frames + approx timestamps | `services/video.py` + `frame_timeline.py` |
| `InferenceRunner` + `Benchmark` | Batched vision encode (timed), text encode, build `[F × L]` cosine/confidence matrices; collect per-backend metrics | `inference_runner.py` |
| `Scoring` | Port `aggregate_mean` / `aggregate_max` / `aggregate_temporal` / `aggregate_contrast` + `FrameTimeline` + temporal policy + contrast policy | `services/scoring.py`, `frame_timeline.py`, `temporal_policy.py` |
| `ui/` (Compose, Material 3) | Setup → Results → Benchmark screens (see §6) | `static/index.html` |

---

## 5. Sub-system specifications

### 5.1 Model asset pipeline (host, Phase 0)
- For each of the 4 models: prefer downloading the prebuilt `onnx-community/siglip2-*-ONNX`
  artifacts; if the exact resolution repo does not exist, run
  `optimum-cli export onnx --model google/siglip2-<variant> ...` to produce
  `vision_model.onnx` and `text_model.onnx`.
- Produce **fp32** (parity reference) and **fp16** (so400m mandatory; large preferred).
- Extract `logit_scale` and `logit_bias` scalars from the checkpoint; write a per-model
  `model.json` (id, display name, params, resolution, precision, logit_scale, logit_bias,
  vision/text file names + sizes + sha256).
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
- The library only returns subword IDs. Implement in Kotlin: `truncation=True`,
  `padding="max_length"`, `max_length=64`, pad id = 0.
- Android build gotcha: `pthread_cond_clockwait` — fix with
  `CXXFLAGS='-lpthread -D__ANDROID_API__=<level>'`, API ≥ 21.
- **Gate:** instrumented test asserts byte-exact equality against `tokenizer_golden.json`.
  Pin `transformers`; regenerate fixtures on bump.

### 5.3 Preprocessing (SigLIP exact)
From `preprocessor_config.json`:
1. Convert frame to RGB.
2. **Resize to the square model resolution as a non-aspect-preserving stretch** (256×256 or
   384×384), `resample = BICUBIC`. **No** center crop, **no** letterbox.
3. Rescale by `1/255` (`rescale_factor = 0.00392156862745098`).
4. Normalize `(x − 0.5) / 0.5` (mean = std = [0.5, 0.5, 0.5]) → range **[−1, 1]**.
5. Channel-first CHW.

**Parity gotcha:** Android `Bitmap.createScaledBitmap` is bilinear/nearest only — no bicubic.
Exact bicubic parity requires a **native resampler** (e.g. bundled libswscale, or a custom
bicubic kernel) or scores will drift per frame. Validated by `preprocess_golden`.

### 5.4 Frame extraction
- Use **Media3 `androidx.media3.inspector.frame.FrameExtractor`** (media3-inspector,
  stable ≥ 1.9.0) — Google's replacement for `MediaMetadataRetriever`.
- Sample at `fps = 1.0`; `approx_timestamp_seconds = sample_index / fps` (matches Python).
- Cap at `max_frames = 300`.
- Do **not** use `MediaMetadataRetriever.getScaledFrameAtTime` (keyframe-snapping +
  slow) or `ffmpeg-kit` (retired, binaries pulled from Maven Central 2025-04).
- Fall back to `MediaExtractor` + `MediaCodec` + OpenGL only if `FrameExtractor` is too slow.

### 5.5 Inference runner & benchmark
- Vision: batch frames through `vision_model.onnx` (batch size configurable, default 32 like
  Python); time the vision pass — this is the dominant cost and the benchmark's headline.
- Text: run `text_model.onnx` once over the label batch.
- Build `[F × L]` cosine + confidence matrices per the §2 contract.
- Metrics per (model, backend): model-load ms, total inference ms, ms/frame, frames/sec,
  actual backend, fallback reason, peak memory (best-effort).

### 5.6 Scoring port
Port the Python aggregation semantics 1:1, with unit tests mirroring `tests/test_scoring.py`:
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
  backend used + fallback reason**; cross-model comparison view.

---

## 7. Data flow

```
pick model → pick backend → pick video + labels (+ mode options)
  → FrameSampler (fps=1, cap 300)
  → Preprocess (bicubic → CHW [-1,1])
  → vision encode (TIMED)  ──┐
  → text encode (labels)  ───┤
                             ├→ cosine/confidence [F×L]
                             → aggregate(mode)
                             → render results + benchmark metrics
```

---

## 8. Parity validation (acceptance gate)

Instrumented (on-device) + unit tests:
1. **Tokenizer** — byte-exact `text → input_ids` vs `tokenizer_golden.json`.
2. **Preprocess** — CHW tensor within tolerance vs `preprocess_golden` (validates bicubic).
3. **End-to-end** — fp32 cosine + confidence within tolerance vs `scores_golden.json`.
4. **Aggregation** — Kotlin unit tests ported from `tests/test_scoring.py`.
5. **fp16 models** — validated empirically against their fp32 reference (cosine-sim drift
   within tolerance); hybrid-fp32 sensitive layers if drift is excessive.

---

## 9. Model provisioning

Models are 1.4–4.2 GB and **cannot be bundled in the APK/AAB**. Strategy:
**download-on-demand** to app-specific external storage (with progress UI + sha256 verify),
mirroring the Python `CLIP_CACHE_DIR` cache pattern; **adb-push fallback** for a dev phone
(push the exported assets directly to the app's files dir). Active model loads one at a time;
no two large models resident simultaneously.

---

## 10. Build order (phases)

0. **Model + fixture prep** (host tooling). *Gate: 4 ONNX model sets + golden fixtures exist.*
1. **Headless engine**: ORT sessions + tokenizer JNI + preprocess + frame sampler + scoring
   port. *Gate: parity tests (§8.1–8.4) green.*
2. **Benchmark harness**: backend switching + timing + attempt-and-report. *Gate: 4 models ×
   3 backends produce metrics with correct backend labeling.*
3. **Compose UI**: setup → results → benchmark panel wired to the engine. *Gate: all 4 modes
   render; benchmark panel shows per-model/backend timings.*

---

## 11. Top risks & mitigations

1. **ONNX availability** — `onnx-community` shipped `-224`; the exact 256/384/so400m repos
   must be verified. *Mitigation:* Phase 0 gate; self-export via Optimum if missing.
2. **Bicubic parity** — Android lacks bicubic; bilinear shifts scores. *Mitigation:* native
   bicubic resampler; validated by fixture #2.
3. **so400m fp16 parity** — no SigLIP2-specific fp16 data exists. *Mitigation:* validate
   empirically per model; hybrid-fp32 sensitive layers if cosine drift exceeds tolerance.
4. **GPU/NPU don't accelerate** — accepted up front. *Mitigation:* attempt-and-report (§3).
5. **Tokenizer version drift** — fast tokenizer has historically diverged. *Mitigation:* pin
   `transformers`; CI gate on golden `(text → input_ids)` set; regenerate on bump.
6. **NNAPI EP deprecation** — building the GPU/NPU attempt on NNAPI is legacy. *Mitigation:*
   it's only the best-effort leg; CPU/XNNPACK is the durable baseline.

---

## 12. Open items to resolve during planning

- Charting library for Compose (Vico vs custom Canvas) — decide in the UI phase.
- ONNX batch-axis: fixed batch vs dynamic — fixed/static favored for predictability and any
  future GPU-delegate attempt.
- Whether to ship fp16 for base/large by default or keep fp32 as the shipped precision with
  fp16 as a benchmark toggle.
