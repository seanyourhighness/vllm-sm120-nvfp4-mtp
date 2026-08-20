# Evidence — vLLM v0.27.1 SM120 NVFP4-KV + MTP-3 build (RTX 5090)

Compiled 2026-08-20 from the verified local runs (2026-08-16/19/20) and the
upstream research sweep. Sources: `GOAL-PLAN-hermes-vllm-final-2026-08-18.md`,
`FINAL-PROFILE-README.md`, `armB-final-*.log`, `8PACK-TESTING-NOTES.md`
(research run `vllm-qwen38-nvfp4-mtp-concurrency-20260816`).

## What makes this build unique

No public container, repo, or blog ships a working **NVFP4 weights + NVFP4 KV +
MTP + concurrency** stack on a 5090 (4-subject research sweep 2026-08-20; 4
vLLM subagent streams; final-synthesis in the container-scan run
`vllm-nvfp4-mtp-container-scan-20260820`). Every public 5090 recipe is 2-of-4
at best — the universal pattern is NVFP4 weights + FP8 KV. Upstream NVFP4-KV
PRs are unmerged (#46329, #44851, #49818 closed-unmerged; #49891 open and
active 2026-08-20). This overlay is the working 4/4 artifact.

## Overlay contents (16 files, +514/−99 vs v0.27.1)

| Area | Files | What it does |
|---|---|---|
| NVFP4 KV kernels | `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu` | SM120 NVFP4 KV store/load |
| NVFP4 KV backend | `vllm/v1/attention/backends/flashinfer.py`, `gdn_attn.py`, `single_type_kv_cache_manager.py`, `vllm/v1/worker/gpu/attn_utils.py` | FlashInfer wiring, XQA + V-scale path |
| GDN ops | `vllm/third_party/flash_linear_attention/ops/fused_recurrent.py`, `fused_sigmoid_gating.py`, `vllm/utils/flashinfer.py`, `vllm/v1/worker/mamba_utils.py` | fused recurrent / sigmoid-gating for the GDN model family |
| MTP × tool-call correctness | `vllm/v1/core/sched/scheduler.py`, `vllm/v1/structured_output/*`, `vllm/v1/worker/gpu_model_runner.py` | structured-output requests skip spec decode; MTP decode-bubble trims; cudagraph safety |
| Vision offload plumbing | `vllm/config/vllm.py`, `vllm/parser/engine/adapters.py`, `vllm/reasoning/abs_reasoning_parsers.py` | `--enable-mm-embeds` image_embeds path |

## Verified measurements

| Item | Value | Source |
|---|---|---|
| KV pool @ 8 GiB pin, 262k boot, MTP-3 | 394,488 (N1) / 373,797 (N3) / 364,618 (N4) tok | plan doc §3 (2026-08-18) |
| MTP-3 clean single-stream long decode | 98 tok/s vs 63.6 no-MTP (+54%) | armB A/B logs |
| MTP-3 full8 ladder, 512-token matched | c1 100.8 → c8 686.5 tok/s | 8PACK-TESTING-NOTES |
| C8 agentic aggregate | 16.5 tok/s (tool loops, prefix-heavy) | 2026-08-20 run |
| Agentic A/B (2026-08-19 soak) | no-MTP: ITL 25 ms, prefix-hit 81.8%, +52% aggregate; MTP-3: ITL 75–200 ms, prefix-hit 55.5% | FINAL-PROFILE-README |
| Vision CPU sidecar parity | 7/7 exact-match (image tests) | Joshua8 playbook + local port |
| Vision gate, containerized build (2026-08-20) | multi-image 6/6 checks; concurrency sweep 1/2/4/8 → **15/15 correct** via CPU sidecar (:8006 → image_embeds → server); MTP active on vision lanes (mean accept 2.68) | `bench/vision_gate.py` |

## Gate results — containerized build (2026-08-20)

| Gate | Result |
|---|---|
| Boot + config (NVFP4 KV, MTP-3, full8 ladder, vision offloaded) | PASS — pool 373,797 tokens @ 8 GiB pin |
| Determinism (temp 0) | PASS (two independent gates, byte-identical) |
| Needle-32k (cold + prefix-cached replay) | PASS — 9/9 (MOONWEASEL-7, RABBIT-42) |
| Tool battery + 4-way tool concurrency | PASS — 8/8 + 4/4 with correct structured args |
| C8 agentic proof | PASS — aggregate 104.8 → 704.4 tok/s (c1→c8); every stream ≥ 92.8 t/s; MTP accept 0.57–0.62 |
| Vision (multi-image, concurrency, JPEG) | PASS — 21/21 lanes correct |
| Long-decode corruption gate (v2) | **PASS** — deterministic 2/2 (see below) |

**Long-decode corruption gate — PASS (v2, 2026-08-20).** The v1 `ignore_eos`
parser produced a **false FAIL** by parsing the forced post-EOS `<|im_start|>`
tail. A direct **MTP-3 vs MTP-off A/B** (same release image, same flags, spec
toggled) exonerated MTP and NVFP4 KV: the degeneration was identical with spec
on and off, and with fences stripped both arms parse as valid Python. The
corrected gate (`bench/corruption_verify.py` v2) truncates at the first
`<|im_start|>`, strips the real fences, and detects the self-test by actual
patterns; it passes **deterministically 2/2** (identical sha256
`103cc361…`, 9,010 chars, valid Python, TokenBucket + self-test). A/B control
launcher: `launch-server-nomtp.sh`. Sources of record: GBrain
`vllm/2026-08-20-v0271-release-c8-corruption-ab` and
`ai/sm120-nvfp4-mtp-cudagraph-corruption` UPDATE 4.

Honest caveat: under concurrent *agentic* load the A/B favored no-MTP (both
launchers are in this release); MTP-3 wins clean single-stream long decode.
The shipped defaults are the user's choice: NVFP4 KV + MTP-3 + 8-stream
concurrency at the 8 GiB pin (~373k pool).

## Licensing

- vLLM v0.27.1: Apache-2.0. The overlay derives from it → Apache-2.0.
- Base model `Qwen/Qwen3.8-27B`: Apache-2.0 (HF metadata).
- NVFP4 quant used in verification (`Qwen3.8-27B-NVFP4-RTX5090`, modelopt
  export): Apache-2.0 (checkpoint LICENSE checked 2026-08-20).

## Reproduction

```bash
./build.sh          # fresh v0.27.1 clone + git apply + docker build
./launch-server.sh  # canonical flags
```

The patch was apply-tested against a fresh `v0.27.1` clone and matches the
verified working tree byte-for-byte (16/16 files, 2026-08-20).
