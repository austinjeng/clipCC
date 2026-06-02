# clipCC-Android Phase 1 — RESUME / Handoff

**Read this first, then `2026-06-03-clipcc-android-phase1-engine.md` (the executable plan).**
Also load memory: `clipcc-android-port` + `clipcc-android-review-discipline`.

## Status (2026-06-03)
- **Phase 0 COMPLETE** (assets + spikes 0a–0d) — see `phase0-spike-results.md`.
- **Plan 1 (headless engine) COMPLETE — all 7 tasks done, all gates passed on Pixel 7a.** See `phase1-report.md`.
  Acceptance gate (Task 7 end-to-end): cosine_max **9.09e-5** (≤0.01), confidence_max **6.14e-7** (≤0.02), **0 best-match flips**.
  Full suite green offline: unit 31/31 + 4 instrumented engine classes.
- **TWO plan resolved-facts were WRONG and corrected (see `phase1-report.md` ERRATA):**
  (1) embedding = ONNX `pooler_output` selected BY NAME, not `res[0]` (=`last_hidden_state`); host cosine match to 1.96e-7.
  (2) XNNPACK EP collapses the symbolic batch dim to 1 for BOTH towers → run batch=1 per item (vision per-frame, text per-label); CPU EP batches fine. ONNX dep moved to `implementation`; OrtTower/Engine live in `src/main`.
- **Next: Plan 2 (benchmark)** — plus optional housekeeping: delete the three Phase-0 `androidTest/.../spike/` test classes.

## Toolchain (already installed — do NOT reinstall)
- Rust 1.96 + `cargo-ndk` 4.1.2 + targets `aarch64-linux-android`, `x86_64-linux-android` (rustup binary in `~/.cargo/bin`; `~/.bash_profile` is NOT writable, so set PATH explicitly).
- NDK r28c at `~/Library/Android/sdk/ndk/28.2.13676358`.
- `libhftokenizer.so` already built into `AndroidStudioProjects/ClipCC/app/src/main/jniLibs/{arm64-v8a,x86_64}/`. Rebuild ONLY if `cpp-tokenizer/{Cargo.toml,src/lib.rs}` changes:
  ```bash
  cd /Users/austin/AndroidStudioProjects/ClipCC/app/src/main/cpp-tokenizer
  export PATH="$HOME/.cargo/bin:$PATH" ANDROID_HOME="$HOME/Library/Android/sdk" \
         ANDROID_NDK_HOME="$HOME/Library/Android/sdk/ndk/28.2.13676358"
  cargo ndk -t arm64-v8a -t x86_64 --platform 24 -o ../jniLibs build --release
  ```

## Paths
- clipCC repo (this repo, git branch `feat/clipcc-android-phase0`): host tooling `tools/android_assets/`; model bundles `build/android_assets/<model_id>/` (git-ignored); fixtures `build/android_assets/fixtures/`. Host venv `.venv-export` (transformers 4.57.6).
- Android project (NOT a git repo): `/Users/austin/AndroidStudioProjects/ClipCC`, module `:app`, package `com.example.clipcc`, minSdk 24, AGP 9.0.0-rc01.
- Engine code goes in `app/src/main/java/com/example/clipcc/engine/`; JVM tests `app/src/test/...`; device tests `app/src/androidTest/...`; fixtures shipped in `app/src/androidTest/assets/fixtures/`.

## Build/run commands (env matters)
```bash
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
ADB="$HOME/Library/Android/sdk/platform-tools/adb"
cd /Users/austin/AndroidStudioProjects/ClipCC
# JVM unit tests (fast, Tasks 2/4/6):
./gradlew :app:testDebugUnitTest --tests "*ResamplerTest*"
# Device instrumented test (Tasks 3/5/7), one class:
./gradlew :app:connectedDebugAndroidTest --console=plain \
  -Pandroid.testInstrumentationRunnerArguments.class=com.example.clipcc.engine.<TestClass>
# Capture test stdout via logcat (System.out tag); clear first with `$ADB logcat -c`.
```

## Device state (Pixel 7a, Android 16, adb id 36161JEHN16600)
- Read models from `/data/local/tmp/...` (app can READ there; CANNOT write — write ORT profiles/logs to the app `cacheDir`). `connectedAndroidTest` uninstalls the app after, wiping `/sdcard/Android/data/<pkg>` — never stage there.
- Already staged: `tokenizer.json` at `/data/local/tmp/clipcc_models/siglip2-base-patch16-256/`; spike models at `/data/local/tmp/clipcc_spike/` (base-256 vision; so400m vision+text+manifest). For Tasks 3/5/7 push the base-256 `vision_model.onnx` + `text_model.onnx` into `/data/local/tmp/clipcc_models/siglip2-base-patch16-256/` and `chmod -R a+rX`.

## Hard-won facts (do not re-litigate — see spec §5.2/§5.3, plan resolved-facts)
- Tokenizer **case-sensitive** → Android must NOT lowercase (proven byte-exact).
- Resize = **PIL antialiased bilinear** (resample=2). `Bitmap.createScaledBitmap` is NOT acceptable (no antialias prefilter) → Task 2 ports a separable-triangle resampler.
- `rewind() as FloatBuffer/LongBuffer` crashes minSdk<33 → cast receiver to `java.nio.Buffer`, rewind as a statement.
- ORT: embedding = `pooler_output` (BY NAME), NOT `res[0]`. **XNNPACK EP collapses dynamic batch→1 for BOTH towers** → run batch=1 per item (CPU EP batches fine). so400m: peak ~3.19 GB, ~16 s/frame; Engine encodes text → releases text session → opens vision. (CORRECTED — see `phase1-report.md` ERRATA; supersedes the old "batched output is correct / {1,32,1152} benign" note.)
- "Verified-correct, do not refactor" list is in the plan's resolved-facts section.

## Acceptance gate (Task 7)
On-device end-to-end scores vs `scores_golden.json`: cosine max-abs ≤ 0.01, confidence ≤ 0.02, no best-match label flips.
