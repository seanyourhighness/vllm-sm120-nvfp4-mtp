#!/usr/bin/env python3
"""Multi-image + vision concurrency gates against :18079.

Test 1: TWO images in ONE request -> must read both (color/shape/label).
Test 2: concurrency sweep c=1/2/4/8 with single-image requests ->
        aggregate tok/s, per-request latency, and correctness.
"""
import base64
import concurrent.futures
import json
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8006")
MODEL = os.environ.get("MODEL", "qwen3.8-27b-nvfp4")
IMG_A = os.path.join(HERE, "vision_test_a.png")
IMG_B = os.path.join(HERE, "vision_test_b.png")


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def post(path, payload, timeout=900):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return data, time.monotonic() - t


def chat(content_parts, max_tokens=200, **kw):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content_parts}],
               "max_tokens": max_tokens, "temperature": 0.0, **kw}
    data, elapsed = post("/v1/chat/completions", payload)
    text = (data["choices"][0]["message"].get("content") or "").strip()
    usage = data.get("usage", {})
    return text, usage, elapsed


# ---------- Test 1: multi-image ----------
print("=== TEST 1: multi-image (2 images, one request) ===")
parts = [
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(IMG_A)}"}},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(IMG_B)}"}},
    {"type": "text", "text": (
        "Two images are provided. For EACH image, state: (1) background color, "
        "(2) shape drawn, (3) the exact text label. Be precise and short.")},
]
text, usage, elapsed = chat(parts, max_tokens=250,
                            chat_template_kwargs={"enable_thinking": False})
print(json.dumps({"response": text, "prompt_tokens": usage.get("prompt_tokens"),
                  "completion_tokens": usage.get("completion_tokens"),
                  "elapsed_s": round(elapsed, 2)}, indent=2))
low = text.lower()
checks = {
    "sees_A_red": "red" in low,
    "sees_A_circle": "circle" in low or "round" in low or "disc" in low,
    "sees_A_label": "alpha" in low,
    "sees_B_blue": "blue" in low,
    "sees_B_square": "square" in low or "rectang" in low,
    "sees_B_label": "bravo" in low,
}
multi_verdict = "PASS" if all(checks.values()) else "FAIL"
print(json.dumps({"multi_image_checks": checks, "verdict": multi_verdict}, indent=2))

# ---------- Test 2: concurrency sweep ----------
print("\n=== TEST 2: vision concurrency sweep ===")


def vision_lane(i):
    img, expect = (IMG_A, ["red", "circle", "alpha"]) if i % 2 == 0 else (IMG_B, ["blue", "square", "bravo"])
    parts = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(img)}"}},
        {"type": "text", "text": "What is the background color, the shape, and the exact text label in this image? Be short."},
    ]
    try:
        t, usage, el = chat(parts, max_tokens=150,
                            chat_template_kwargs={"enable_thinking": False})
        low = t.lower()
        correct = all(exp in low for exp in expect)
        ct = usage.get("completion_tokens") or 0
        return {"lane": i, "ok": True, "correct": correct, "out": t[:60],
                "prompt_tok": usage.get("prompt_tokens"), "comp_tok": ct,
                "tok_s": round(ct / el, 2), "elapsed_s": round(el, 2)}
    except Exception as e:
        return {"lane": i, "ok": False, "correct": False,
                "error": f"{type(e).__name__}: {e}"}


for conc in (1, 2, 4, 8):
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
        results = list(pool.map(vision_lane, range(conc)))
    wall = time.monotonic() - t0
    comp_total = sum(r.get("comp_tok", 0) for r in results)
    all_ok = all(r["ok"] for r in results)
    all_correct = all(r["correct"] for r in results)
    print(json.dumps({
        "concurrency": conc,
        "wall_s": round(wall, 2),
        "agg_tok_s": round(comp_total / wall, 2),
        "correct": f"{sum(r['correct'] for r in results)}/{conc}",
        "per_req": [{"lane": r["lane"], "tok_s": r.get("tok_s"), "s": r.get("elapsed_s"),
                     "correct": r.get("correct")} for r in results],
        "verdict": "PASS" if (all_ok and all_correct) else "FAIL",
    }, indent=2))
