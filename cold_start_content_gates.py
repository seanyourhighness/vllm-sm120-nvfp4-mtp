#!/usr/bin/env python3
"""Content gates for the cold-start boot (needle + determinism).

Same semantics as correctness_gate.py gates 1-2, trimmed to what a cold-boot
regression must prove: the model that just downloaded and loaded produces
correct, deterministic output through the strict MTP loader.
"""
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:18080")
MODEL = os.environ.get("MODEL", "qwen38-nvfp4-armb-mtp1-g1-fullpw-vision")


def chat(messages, max_tokens=128, temperature=0.0):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def needle_run(index: int):
    marker = f"Cold-gate campaign {time.time_ns():x}, lane {index}. "
    filler = (
        "The archives of the northern province record grain yields and river levels. "
    )
    body = [marker] + [filler] * 2399
    body.insert(1200, "\n\nIMPORTANT: The vault codeword is MOONWEASEL-7. Remember it.\n\n")
    prompt = "".join(body) + "\n\nWhat is the vault codeword? Reply with only the codeword."
    data = chat([{"role": "user", "content": prompt}], max_tokens=16)
    content = (data["choices"][0]["message"].get("content") or "").strip()
    return {"lane": index, "content": content, "pass": "MOONWEASEL-7" in content}


failures = []

# Gate A: needle-in-haystack, 2 lanes in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    lanes = sorted(pool.map(needle_run, range(2)), key=lambda r: r["lane"])
ok = all(r["pass"] for r in lanes)
print(json.dumps({"gate": "needle-32k", "lanes": lanes, "pass": ok}, indent=2))
if not ok:
    failures.append("needle")

# Gate B: temp-0 determinism
def det():
    data = chat(
        [{"role": "user", "content": "Explain speculative decoding in one paragraph."}],
        max_tokens=128,
    )
    content = (data["choices"][0]["message"].get("content") or "").strip()
    return hashlib.sha256(content.encode()).hexdigest()

ha, hb = det(), det()
ok = ha == hb
print(json.dumps({"gate": "determinism", "sha256_a": ha, "sha256_b": hb, "pass": ok}, indent=2))
if not ok:
    failures.append("determinism")

if failures:
    raise SystemExit(f"CONTENT GATES FAILED: {', '.join(failures)}")
print("PASS: needle-32k (2/2) + determinism")
