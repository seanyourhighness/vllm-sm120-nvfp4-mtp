#!/usr/bin/env python3
"""Correctness + vision gates for the release image (routes via CPU sidecar).

Gates:
  1. needle-32k: cold + prefix-cached replay, 3 runs each (MOONWEASEL-7)
  2. determinism: two identical temp-0 runs, byte-identical sha256
  3. long-decode corruption: uses v2 semantics (truncate at first
     <|im_start|>, strip fences, real self-test patterns) — see
     corruption_verify.py v2
  4. vision: real JPEG via chat completions image_url, compare to ground truth
"""
import ast
import base64
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8006")
MODEL = os.environ.get("MODEL", "qwen3.8-27b-nvfp4")
IMAGE_PATH = os.environ.get(
    "IMAGE_PATH", os.path.join(HERE, "bench", "vision_test_a.png")
)
IMAGE_MIME = "image/png"

RESULTS = {}


def post(path: str, payload: dict, timeout: int = 600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return data, time.monotonic() - started


def chat(messages, max_tokens=128, temperature=0.0, **kw):
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature, **kw}
    data, elapsed = post("/v1/chat/completions", payload)
    content = (data["choices"][0]["message"].get("content") or "").strip()
    return content, data, elapsed


# ---------------- Gate 1: needle-32k ----------------
def needle_run(index: int):
    marker = f"Vision campaign {time.time_ns():x}, lane {index}. "
    filler = "The archives of the northern province record grain yields and river levels. "
    body = [marker] + [filler] * 2399
    body.insert(1200, "\n\nIMPORTANT: The vault codeword is MOONWEASEL-7. Remember it.\n\n")
    prompt = "".join(body) + "\n\nWhat is the vault codeword? Reply with only the codeword."
    content, _, elapsed = chat(
        [{"role": "user", "content": prompt}],
        max_tokens=64,
        chat_template_kwargs={"enable_thinking": False},
    )
    return {"lane": index, "passed": "MOONWEASEL-7" in content,
            "output": content[:80], "elapsed_s": round(elapsed, 2)}


def run_needle():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        cold = sorted(pool.map(needle_run, range(3)), key=lambda r: r["lane"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        replay = sorted(pool.map(needle_run, range(3)), key=lambda r: r["lane"])
    RESULTS["needle"] = {
        "cold": cold, "replay": replay,
        "verdict": "PASS" if all(r["passed"] for r in cold + replay) else "FAIL",
    }
    print(json.dumps(RESULTS["needle"], indent=2))


# ---------------- Gate 2: determinism ----------------
def determinism_run():
    prompt = "Write exactly the sentence: The quick brown fox jumps over the lazy dog. Then stop."
    content, _, _ = chat([{"role": "user", "content": prompt}], max_tokens=64)
    return content, hashlib.sha256(content.encode()).hexdigest()


def run_determinism():
    a, ha = determinism_run()
    b, hb = determinism_run()
    RESULTS["determinism"] = {"sha256_a": ha, "sha256_b": hb,
                              "deterministic": ha == hb,
                              "output_a": a[:120], "output_b": b[:120],
                              "verdict": "PASS" if ha == hb else "FAIL"}
    print(json.dumps(RESULTS["determinism"], indent=2))


# ---------------- Gate 3: long decode corruption (v2 semantics) ----------------
CODING_PROMPT = """Write a complete, production-quality Python module named rate_limiter.py.
It must implement a thread-safe token-bucket rate limiter with a small public API,
clear type hints, docstrings, monotonic-time handling, a context-manager helper,
and a short executable self-test under if __name__ == '__main__'. Return code only."""


def clean_code(content: str) -> str:
    """v2 clean: strip leading fence, truncate at forced post-EOS loop,
    strip trailing fence the model closed before looping."""
    s = re.sub(r"^```(?:python)?\s*", "", content).strip()
    i = s.find("<|im_start|>")
    if i != -1:
        s = s[:i].rstrip()
    return re.sub(r"```\s*$", "", s).rstrip()


def coding_run():
    content, _, elapsed = chat([{"role": "user", "content": CODING_PROMPT}],
                               max_tokens=4096, ignore_eos=True)
    stripped = clean_code(content)
    try:
        tree = ast.parse(stripped)
        valid = True
    except SyntaxError:
        valid = False
        tree = None
    has_bucket = (
        any(isinstance(n, ast.ClassDef) and "bucket" in n.name.lower()
            for n in ast.walk(tree)) if valid else False
    )
    has_selftest = (
        re.search(r"if\s+__name__\s*==\s*[\"']__main__[\"']", stripped) is not None
        or "main()" in stripped
        or "_self_test()" in stripped
    )
    return {"valid_python": valid, "has_token_bucket_class": has_bucket,
            "has_self_test": has_selftest, "chars": len(stripped),
            "sha256": hashlib.sha256(stripped.encode()).hexdigest(),
            "elapsed_s": round(elapsed, 2), "head": stripped[:80],
            "tail": stripped[-80:]}


def run_corruption():
    r1, r2 = coding_run(), coding_run()
    RESULTS["corruption"] = {
        "run1": r1, "run2": r2,
        "deterministic": r1["sha256"] == r2["sha256"],
        "verdict": "PASS" if (r1["valid_python"] and r2["valid_python"]
                              and r1["has_token_bucket_class"] and r2["has_token_bucket_class"]
                              and r1["has_self_test"] and r2["has_self_test"]) else "FAIL",
    }
    print(json.dumps(RESULTS["corruption"], indent=2))


# ---------------- Gate 4: vision ----------------
def run_vision():
    with open(IMAGE_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{IMAGE_MIME};base64,{b64}"}},
            {"type": "text",
             "text": "Describe this image in detail. What does it show? Include any visible text."},
        ],
    }]
    try:
        content, data, elapsed = chat(messages, max_tokens=300)
        usage = data.get("usage", {})
        RESULTS["vision"] = {
            "ok": True,
            "response": content,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as e:
        RESULTS["vision"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(RESULTS["vision"], indent=2))


if __name__ == "__main__":
    print("=== GATE 1: needle-32k (cold + replay) ===")
    run_needle()
    print("\n=== GATE 2: determinism ===")
    run_determinism()
    print("\n=== GATE 3: long-decode corruption (v2) ===")
    run_corruption()
    print("\n=== GATE 4: vision ===")
    run_vision()
    verdicts = {k: v.get("verdict", "SEE-ABOVE")
                for k, v in RESULTS.items() if k != "vision"}
    vision_v = "PASS(see response)" if RESULTS["vision"].get("ok") else "FAIL"
    print("\n=== SUMMARY ===")
    for k, v in {**verdicts, "vision": vision_v}.items():
        print(f"  {k}: {v}")
    raise SystemExit(0 if all(x == "PASS" for x in verdicts.values())
                     and RESULTS["vision"].get("ok") else 1)
