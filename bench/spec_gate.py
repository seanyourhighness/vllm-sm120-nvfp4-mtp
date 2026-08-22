#!/usr/bin/env python3
"""MTP-profile verify gates: tool-call, non-repetition, spec acceptance.

Catches the Trap-122 class of failure where the server stays healthy while
generations silently collapse under a FULL CUDA-graph capture: empty content,
missing tool calls, or repetitive garbage.

Gates (all against the OpenAI-compatible endpoint):
  1. tool_call: a get_weather request must return finish_reason=tool_calls
     with a callable function and a Paris argument.
  2. non_repetition: a 256-token temp-0 completion must be non-empty,
     sufficiently diverse, and free of repeated 6-gram loops.
  3. spec_acceptance (EXPECT_SPEC=1 only): vllm:spec_decode_* counters must
     advance with accepted > 0 over two generations.

Exit 0 on PASS, 1 on FAIL.
"""
import json
import os
import sys
import urllib.request
from collections import Counter

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:18079")
MODEL = os.environ.get("MODEL", "qwen3.8-27b-nvfp4")
EXPECT_SPEC = os.environ.get("EXPECT_SPEC", "1") == "1"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


def post(payload, timeout=300):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def spec_counters():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {}
    for name in ("vllm:spec_decode_num_draft_tokens_total",
                 "vllm:spec_decode_num_accepted_tokens_total"):
        for line in text.splitlines():
            if line.startswith(name + "{"):
                out[name] = float(line.rsplit(" ", 1)[-1])
                break
    return out


def gate_tool_call():
    data = post({
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": "What is the weather in Paris? Use the get_weather tool."}],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 128,
    })
    msg = data["choices"][0]["message"]
    finish = data["choices"][0].get("finish_reason")
    calls = msg.get("tool_calls") or []
    if finish != "tool_calls" or not calls:
        return {"ok": False, "finish_reason": finish,
                "content": (msg.get("content") or "")[:120]}
    fn = calls[0].get("function") or {}
    raw_args = fn.get("arguments") or ""
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        args = {"_raw": raw_args}
    city = str(args.get("city", ""))
    ok = fn.get("name") == "get_weather" and "paris" in city.lower()
    return {"ok": ok, "finish_reason": finish, "name": fn.get("name"),
            "args": args, "city": city}


def gate_non_repetition():
    data = post({
        "model": MODEL,
        "messages": [{"role": "user", "content":
                      "Write a detailed essay about the history of urban water "
                      "systems, covering engineering, public health, and modern "
                      "infrastructure."}],
        "temperature": 0,
        "max_tokens": 256,
    })
    text = (data["choices"][0]["message"].get("content") or "").strip()
    words = text.split()
    if not text or len(words) < 40:
        return {"ok": False,
                "reason": f"degenerate short/empty output ({len(words)} words)",
                "head": text[:120]}
    uniq_ratio = len(set(words)) / len(words)
    grams = Counter(tuple(words[i:i + 6]) for i in range(len(words) - 5))
    top_ratio = grams.most_common(1)[0][1] / max(len(words) - 5, 1)
    ok = uniq_ratio >= 0.15 and top_ratio <= 0.25
    return {"ok": ok, "words": len(words), "uniq_ratio": round(uniq_ratio, 3),
            "top_6gram_ratio": round(top_ratio, 3),
            "head": text[:100]}


def gate_spec_acceptance():
    m0 = spec_counters()
    for _ in range(2):
        post({"model": MODEL,
              "messages": [{"role": "user", "content":
                            "Explain how speculative decoding works in large "
                            "language models."}],
              "temperature": 0, "max_tokens": 256})
    m1 = spec_counters()
    if not m0 or not m1:
        return {"ok": False, "reason": "spec counters not found in /metrics"}
    drafts = m1.get("vllm:spec_decode_num_draft_tokens_total", 0) - m0.get(
        "vllm:spec_decode_num_draft_tokens_total", 0)
    accepted = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get(
        "vllm:spec_decode_num_accepted_tokens_total", 0)
    return {"ok": drafts > 0 and accepted > 0,
            "drafts": int(drafts), "accepted": int(accepted),
            "accept_rate": round(accepted / drafts, 3) if drafts else None}


def main():
    results = {}
    print("=== SPEC GATE: tool call ===")
    results["tool_call"] = gate_tool_call()
    print(json.dumps(results["tool_call"], indent=2))
    print("\n=== SPEC GATE: non-repetition ===")
    results["non_repetition"] = gate_non_repetition()
    print(json.dumps(results["non_repetition"], indent=2))
    print("\n=== SPEC GATE: spec acceptance ===")
    if EXPECT_SPEC:
        results["spec_acceptance"] = gate_spec_acceptance()
        print(json.dumps(results["spec_acceptance"], indent=2))
    else:
        print("skipped (EXPECT_SPEC=0: no-MTP profile)")
    print("\n=== SUMMARY ===")
    for name, result in results.items():
        print(f"  {name}: {'PASS' if result['ok'] else 'FAIL'}")
    raise SystemExit(0 if results and all(r["ok"] for r in results.values()) else 1)


if __name__ == "__main__":
    main()
