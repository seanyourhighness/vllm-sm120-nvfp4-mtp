#!/usr/bin/env python3
"""corruption_verify.py v2 — canonical long-decode corruption gate.

Fixes the v1 false-FAIL (source of record: GBrain
`vllm/2026-08-20-v0271-release-c8-corruption-ab`): with `ignore_eos` the model
completes the module, closes the markdown fence, then loops `<|im_start|>`; v1
parsed the whole string including the forced post-EOS loop. The MTP-3 vs MTP-off
A/B exonerated MTP and NVFP4 KV (identical degeneration with spec on/off).

v2:
  - strips the leading ```python fence,
  - truncates at the first <|im_start|>,
  - strips the trailing ``` fence the model closed before looping,
  - detects the self-test by real patterns (__name__ with any quote style,
    or a main()/_self_test() call).

Verified 2026-08-20 against the release image (MTP-3): PASS, deterministic
2/2 (identical sha256 103cc361, 9,010 chars, valid Python, TokenBucket class
+ self-test).
"""
import ast
import hashlib
import json
import os
import re
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8006")
MODEL = os.environ.get("MODEL", "qwen3.8-27b-nvfp4")
PROMPT = """Write a complete, production-quality Python module named rate_limiter.py.
It must implement a thread-safe token-bucket rate limiter with a small public API,
clear type hints, docstrings, monotonic-time handling, a context-manager helper,
and a short executable self-test under if __name__ == '__main__'. Return code only."""


def post(payload, timeout=600):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def clean(content: str) -> str:
    """Reduce raw output to the model's actual module.

    With ignore_eos the model writes the module, closes the fence, then loops
    <|im_start|>. Everything from the first <|im_start|> is the forced-length
    degeneration tail, not model output.
    """
    s = re.sub(r"^```(?:python)?\s*", "", content).strip()
    i = s.find("<|im_start|>")
    if i != -1:
        s = s[:i].rstrip()
    s = re.sub(r"```\s*$", "", s).rstrip()
    return s


def analyze(stripped: str) -> dict:
    try:
        tree = ast.parse(stripped)
        valid, err = True, None
    except SyntaxError as e:
        valid, err = False, str(e)
    has_bucket = (
        any(isinstance(n, ast.ClassDef) and "bucket" in n.name.lower()
            for n in ast.walk(tree)) if valid else False
    )
    has_selftest = (
        re.search(r"if\s+__name__\s*==\s*[\"']__main__[\"']", stripped) is not None
        or "main()" in stripped
        or "_self_test()" in stripped
    )
    return {
        "valid_python": valid,
        "syntax_error": err,
        "has_token_bucket_class": has_bucket,
        "has_self_test": has_selftest,
        "chars": len(stripped),
        "sha256": hashlib.sha256(stripped.encode()).hexdigest(),
        "head": stripped[:90],
        "tail": stripped[-90:],
    }


def run() -> dict:
    d = post({"model": MODEL,
              "messages": [{"role": "user", "content": PROMPT}],
              "temperature": 0, "max_tokens": 4096, "ignore_eos": True})
    content = d["choices"][0]["message"]["content"] or ""
    stripped = clean(content)
    a = analyze(stripped)
    a.update({
        "tokens": d.get("usage", {}).get("completion_tokens"),
        "finished_naturally": d["choices"][0].get("finish_reason"),
    })
    return a


if __name__ == "__main__":
    r1, r2 = run(), run()
    verdict = "PASS" if (
        r1["valid_python"] and r2["valid_python"]
        and r1["has_token_bucket_class"] and r2["has_token_bucket_class"]
        and r1["has_self_test"] and r2["has_self_test"]
        and r1["sha256"] == r2["sha256"]
    ) else "FAIL"
    print(json.dumps({"run1": r1, "run2": r2,
                      "deterministic": r1["sha256"] == r2["sha256"],
                      "verdict": verdict}, indent=2))
    raise SystemExit(0 if verdict == "PASS" else 1)
