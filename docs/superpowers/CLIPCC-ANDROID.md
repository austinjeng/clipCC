# clipCC-Android (moved to its own repo)

The native **Android port** of clipCC — an on-device SigLIP2 inference + benchmark app
(Kotlin / Jetpack Compose) — now lives in its own repository, together with its design
specs, implementation plans, and phase reports:

**→ https://github.com/austinjeng/ClipCC-Android**
(see `docs/specs/` and `docs/plans/` there)

The host-side tooling that exports the Android model bundles + golden fixtures from the
Python source models stays in **this** repo, under [`tools/android_assets/`](../../tools/android_assets/).
