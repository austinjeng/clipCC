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

    # (2) Behavioral diagnostic: does the Rust engine lowercase on its own?
    rust_lowercases = all(
        rust.encode(t).ids == rust.encode(t.lower()).ids
        for t in CASES if t != t.lower()
    )

    # (3) The §10 0a GATE — decide from what AutoProcessor ACTUALLY emits, not from a
    #     structural guess. The candidate pipeline (rust.encode -> _pad_trunc) must equal the
    #     AutoProcessor max_length=64 output. Test BOTH: encode raw text vs encode lowered text.
    #     Whichever matches AutoProcessor for ALL cases is the correct decision:
    #       - raw matches   -> "tokenizer_json"  (Android must NOT lowercase; case-sensitive)
    #       - lower matches -> "kotlin_wrapper"  (Android must .lowercase() before encoding)
    def ref_ids(text: str) -> list[int]:
        return proc(text=[text], padding="max_length", max_length=64,
                    truncation=True, return_tensors="np")["input_ids"][0].tolist()

    def cand_ids(text: str) -> list[int]:
        return _pad_trunc(rust.encode(text).ids)

    matches_raw = all(cand_ids(t) == ref_ids(t) for t in CASES)
    matches_lower = all(cand_ids(t.lower()) == ref_ids(t) for t in CASES)
    if matches_raw:
        decision = "tokenizer_json"
    elif matches_lower:
        decision = "kotlin_wrapper"
    else:
        decision = "needs_investigation"
    parity_ok = matches_raw or matches_lower

    # Per-case parity under the chosen decision (for the printed table).
    lower_first = decision == "kotlin_wrapper"
    rows = [(text, cand_ids(text.lower() if lower_first else text) == ref_ids(text))
            for text in CASES]

    return {
        "decision": decision,
        "rust_lowercases": rust_lowercases,
        "normalizer_lowercases": normalizer_lowercases,
        "matches_raw": matches_raw,
        "matches_lower": matches_lower,
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
    print("matches_raw =", out["matches_raw"], " matches_lower =", out["matches_lower"])
    print("DECISION lowercase_applied_by =", out["decision"])
    print("GATE 0a parity_ok =", out["parity_ok"])
    raise SystemExit(0 if out["parity_ok"] else 1)
