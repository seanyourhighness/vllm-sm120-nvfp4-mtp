"""Raw c8 decode proof: depth curve with per-stream first-to-last-token rates.

For each concurrency arm, runs N warmed reps of 512-token matched streams and
reports aggregate tok/s plus each stream's own rate (tokens / first-to-last
token interval), so the >=75 tok/s per-stream gate is measured directly.
"""

import concurrent.futures
import json
import os
import statistics
import sys
import time

import requests

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:18079")
URL = f"{BASE}/v1/chat/completions"
MODEL = os.environ.get("MODEL", "qwen3.8-27b-nvfp4")
METRICS = f"{BASE}/metrics"

PROMPTS = [
    "Write a detailed engineering design note about KV cache memory management "
    "in vLLM, covering pinning, pooling, and concurrency tradeoffs.",
    "Explain the architecture of a speculative decoding system, including the "
    "drafter, verification, and acceptance criteria, with concrete examples.",
    "Describe how 4-bit KV cache quantization works for hybrid linear-attention "
    "models, including the GatedDeltaNet recurrent state.",
    "Discuss how CUDA graph capture interacts with Mamba draft models in a "
    "hybrid decoder, and what the capture ladder should look like.",
    "Write about the tradeoffs between synchronous and asynchronous scheduling "
    "in LLM serving engines under speculative decoding.",
]


def metrics():
    out = {}
    text = requests.get(METRICS, timeout=10).text
    for name in ("vllm:spec_decode_num_draft_tokens_total",
                 "vllm:spec_decode_num_accepted_tokens_total"):
        for line in text.splitlines():
            if line.startswith(name + "{"):
                out[name] = float(line.split(" ")[-1])
                break
    return out


def one(prompt: str) -> dict:
    t0 = time.monotonic()
    first = None
    last = None
    done = 0
    with requests.post(URL, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            text = raw[6:]
            if text == "[DONE]":
                break
            ev = json.loads(text)
            if ev.get("usage"):
                done = ev["usage"].get("completion_tokens", 0)
            ch = ev.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                now = time.monotonic()
                if first is None:
                    first = now
                last = now
    wall = time.monotonic() - t0
    n = done or 0
    active = (last - first) if (first and last and done > 1) else wall
    return {"tokens": n, "active_s": active, "tok_s": n / active if active else 0.0,
            "wall_s": wall}


def run_arm(concurrency: int, reps: int) -> dict:
    m0 = metrics()
    all_streams = []
    agg_tokens = 0
    wall_start = time.monotonic()
    for rep in range(reps):
        prompts = PROMPTS * ((concurrency + len(PROMPTS) - 1) // len(PROMPTS))
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            results = list(ex.map(one, prompts[:concurrency]))
        all_streams.extend(results)
        agg_tokens += sum(r["tokens"] for r in results)
    wall = time.monotonic() - wall_start
    m1 = metrics()
    drafts = m1.get("vllm:spec_decode_num_draft_tokens_total", 0) - m0.get(
        "vllm:spec_decode_num_draft_tokens_total", 0)
    accepted = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get(
        "vllm:spec_decode_num_accepted_tokens_total", 0)
    rates = sorted(r["tok_s"] for r in all_streams)
    return {
        "c": concurrency,
        "reps": reps,
        "streams": len(all_streams),
        "agg_tok_s": round(agg_tokens / wall, 1),
        "per_stream_median": round(statistics.median(rates), 1),
        "per_stream_min": round(rates[0], 1),
        "per_stream_p25": round(rates[int(len(rates) * 0.25)], 1),
        "below_70": sum(1 for r in rates if r < 70),
        "below_75": sum(1 for r in rates if r < 75),
        "drafts": int(drafts),
        "accepted": int(accepted),
        "accept_rate": round(accepted / drafts, 3) if drafts else None,
        "wall_s": round(wall, 1),
    }


def main() -> None:
    arms = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 2, 4, 6, 8]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out = []
    for c in arms:
        print(f"--- c={c} reps={reps} ---", flush=True)
        r = run_arm(c, reps)
        print(json.dumps(r))
        out.append(r)
    output = os.environ.get("OUTPUT", "c8-proof.json")
    with open(output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
