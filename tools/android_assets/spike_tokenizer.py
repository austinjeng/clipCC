from __future__ import annotations

import json
import sys
from pathlib import Path

CASES = ["Car", "car", "CAR", "Texting While Driving", "texting while driving",
         "EATING", "a dog.", "two words", "ALLCAPS PHRASE"]


def _pad_trunc(ids: list[int], max_length: int = 64, pad_id: int = 0) -> list[int]:
    """Reproduce the Kotlin-side wrapper: truncate to max_length, pad right with pad_id."""
    ids = list(ids[:max_length])
    return ids + [pad_id] * (max_length - len(ids))


def run(hf_repo: str, bundle_dir: Path) -> dict:
    from transformers import AutoProcessor
    from tokenizers import Tokenizer

    # Reference = HF AutoProcessor (torch side). Candidate = the shipped tokenizer.json read by
    # the SAME Rust engine the Android JNI uses.
    proc = AutoProcessor.from_pretrained(hf_repo)
    tok_json_path = bundle_dir / "tokenizer.json"
    rust = Tokenizer.from_file(str(tok_json_path))

    # (1) Structural ground truth: does the serialized normalizer contain a Lowercase step?
    norm = json.loads(tok_json_path.read_text()).get("normalizer")
    normalizer_lowercases = "Lowercase" in json.dumps(norm)

    # (2) Behavioral test: does the Rust engine lowercase on its own? Compare encode(X) to
    #     encode(X.lower()) only for inputs that actually change under lower().
    rust_lowercases = all(
        rust.encode(t).ids == rust.encode(t.lower()).ids
        for t in CASES if t != t.lower()
    )

    # Decision: only "tokenizer_json" when BOTH structural and behavioral evidence agree.
    decision = "tokenizer_json" if (rust_lowercases and normalizer_lowercases) else "kotlin_wrapper"

    # (3) The actual §10 0a GATE: the FULL candidate pipeline must equal AutoProcessor's
    #     max_length=64 output byte-for-byte. Candidate = (lowercase iff kotlin_wrapper) ->
    #     rust.encode -> _pad_trunc. This also surfaces any special-token / EOS differences.
    rows = []
    parity_ok = True
    for text in CASES:
        ref = proc(text=[text], padding="max_length", max_length=64,
                   truncation=True, return_tensors="np")["input_ids"][0].tolist()
        candidate_text = text if decision == "tokenizer_json" else text.lower()
        cand = _pad_trunc(rust.encode(candidate_text).ids)
        ok = cand == ref
        parity_ok = parity_ok and ok
        rows.append((text, ok))

    return {
        "decision": decision,
        "rust_lowercases": rust_lowercases,
        "normalizer_lowercases": normalizer_lowercases,
        "parity_ok": parity_ok,
        "rows": rows,
    }


if __name__ == "__main__":
    # argv: <hf_repo> <bundle_dir>
    out = run(sys.argv[1], Path(sys.argv[2]))
    for text, ok in out["rows"]:
        print(f"{text!r:24} full_pipeline_parity:{ok}")
    print("rust_lowercases =", out["rust_lowercases"],
          " normalizer_lowercases =", out["normalizer_lowercases"])
    print("DECISION lowercase_applied_by =", out["decision"])
    print("GATE 0a parity_ok =", out["parity_ok"])
    raise SystemExit(0 if out["parity_ok"] else 1)
