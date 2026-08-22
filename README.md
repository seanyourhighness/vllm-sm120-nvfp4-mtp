# vLLM NVFP4-KV + MTP for RTX 5090

Community vLLM build for a single GeForce RTX 5090: ModelOpt NVFP4 weights,
NVFP4 KV cache, MTP-3 speculative decoding, 262K context, eight concurrent
streams, tool calling, and optional CPU-offloaded vision.

> This is not an official vLLM or NVIDIA image. It is a pinned community
> overlay on vLLM v0.27.1, validated on an RTX 5090 (SM120, 32 GB).

## Deploy in two commands

Prerequisites: Linux or WSL2, an RTX 5090, a working NVIDIA driver,
[Docker Engine with the Compose plugin](https://docs.docker.com/engine/install/),
and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Allow roughly 30 GB of downloads for the 8.55 GB runtime image and 20.6 GB
model checkpoint.

```bash
git clone https://github.com/seanyourhighness/vllm-sm120-nvfp4-mtp.git
cd vllm-sm120-nvfp4-mtp && ./start.sh
```

`start.sh` checks the GPU, VRAM, Docker/Compose, disk space, and ports; pulls
the image; downloads the exact pinned model into a Docker volume; starts vLLM
and the CPU vision sidecar; waits for health; and sends a real chat completion.

On hosts with more than one GPU, set `GPU_DEVICE` in `.env` to the RTX 5090's
index or UUID (default `0`). `start.sh` probes only that device and the compose
stack exposes only that device to the container (`NVIDIA_VISIBLE_DEVICES`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`), so an older second card cannot interfere with
device selection or enumeration.

Endpoints after startup:

- OpenAI-compatible vLLM API: `http://127.0.0.1:18079/v1`
- Vision-capable proxy: `http://127.0.0.1:8006/v1`
- Served model: `qwen3.8-27b-nvfp4`

First startup is dominated by the two downloads and CUDA/FlashInfer warmup.
Subsequent starts reuse the Docker image and the named model cache.

## Exact pinned stack

| Component | Pinned artifact |
|---|---|
| Release | [`v0.27.1-sm120.2`](https://github.com/seanyourhighness/vllm-sm120-nvfp4-mtp/releases/tag/v0.27.1-sm120.2) |
| Runtime | [`ghcr.io/seanyourhighness/vllm-sm120-nvfp4-mtp`](https://github.com/users/seanyourhighness/packages/container/package/vllm-sm120-nvfp4-mtp)`@sha256:8a7fcf235fccac6da98c08b6077b9cdd4b4a974822a39eee42c8cc07f83198ae` |
| Model + revision | [`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090@69274a0`](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090/tree/69274a0d8dff5dd35bcee8290612f71e03b6e981) |
| vLLM base | [v0.27.1 commit `6e448d0ea`](https://github.com/vllm-project/vllm/commit/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac) |
| FlashInfer | 0.6.16.post3 |
| CUDA build | 13.0.3, SM120 only |
| Overlay | [`0001-v0271-sm120-nvfp4-kv-mtp-toolcall.patch`](https://github.com/seanyourhighness/vllm-sm120-nvfp4-mtp/releases/download/v0.27.1-sm120.2/0001-v0271-sm120-nvfp4-kv-mtp-toolcall.patch) (`sha256:55f127c2…`) |
| Overlay (issue #4) | `0002-v0271-mtp-lmhead-nvfp4-dequant.patch` (`sha256:db9fe07e…`) — MTP drafter loader accepts NVFP4-packed *and* BF16 lm_head; dequantizes packed heads at load, drops scale tensors, guards shape/missing-scale |
| Chat template | [`chat-template.jinja`](https://github.com/seanyourhighness/vllm-sm120-nvfp4-mtp/releases/download/v0.27.1-sm120.2/chat-template.jinja) (`sha256:398edf5b…`) |
| Checksums | [`SHA256SUMS`](https://github.com/seanyourhighness/vllm-sm120-nvfp4-mtp/releases/download/v0.27.1-sm120.2/SHA256SUMS) |

The image and model are pinned by immutable digests/revisions, not floating
tags. Compose passes the pinned model revision to vLLM and mounts the shipped
release template with `--chat-template`; this intentionally overrides the
different template bundled with the model. The model weights are not
redistributed in the runtime image.

Release `.2` is a metadata-only successor to `.1`: it preserves the exact
runtime filesystem while correcting the OCI source/release labels and embedding
the model revision plus patch/template checksums in the image configuration.

## Common operations

```bash
./status.sh                 # containers, health, GPU, and model-cache status
./verify.sh                 # fast health/model/chat checks
./verify.sh --full          # needle, determinism, long-decode, vision, tool-call,
                            # non-repetition, and (MTP) spec-acceptance gates
./start.sh --no-mtp         # same server with speculative decoding disabled
./stop.sh                   # stop services; preserve the downloaded model cache
./stop.sh --purge-cache     # also delete the ~20.6 GB model cache
docker compose logs -f server
```

To override ports or binding, copy `.env.example` to `.env` and edit it. The
safe default binds both APIs to `127.0.0.1`. Do not expose an unauthenticated
vLLM endpoint to the public internet.

Notes from community testing:

- **Agent workloads need headroom.** A healthy chat smoke is not enough for
  agent tasks: on this stack, an independent tester scored 0/502 on a bounded
  agent task with thinking off and `max_tokens=8192`, versus 303/502 with
  thinking on (medium) and `max_tokens=16384`. Keep `enable_thinking=true`
  and give agents a generous `max_tokens`.
- **Cold-boot Hub rate limits.** The model is public and ungated, but
  unauthenticated downloads can hit Hub rate limits on a cold start; set
  `HF_TOKEN` in `.env` if you see warnings.
- **Validation order.** While
  [issue #4](https://github.com/seanyourhighness/vllm-sm120-nvfp4-mtp/issues/4)
  is open, `./start.sh --no-mtp` is the documented first-validation arm.
  `./verify.sh --full` gates the MTP profile on content quality (tool call +
  non-repetition) and live spec-acceptance counters, not `/health` alone.

## What starts

```text
client ──► :8006 vision proxy ──► :8000 vLLM container ──► RTX 5090
             │                         │
             └── CPU vision tower      └── NVFP4 weights/KV + MTP-3
```

Text-only clients may call port 18079 directly. Clients sending `image_url`
content should call port 8006; the sidecar computes image embeddings on CPU and
forwards them to vLLM, keeping the vision tower out of VRAM.

## Verified release gates

- Boot/config: NVFP4 KV, MTP-3, 8 GiB KV pin, 262K context, full8 graph ladder
- KV pool: 373,797 tokens with MTP-3
- Needle-32K: 9/9 cold and prefix-cached replay
- Determinism: byte-identical temperature-zero output
- Tool calls: 8/8 plus 4/4 concurrent structured arguments
- C8 decode proof: 704.4 aggregate tok/s; all 24 streams at least 92.8 tok/s
- MTP verify gates: tool-call, non-repetition, and spec-acceptance counters
  (draft→accept > 0) via `./verify.sh --full`
- Vision: 21/21 lanes through the CPU sidecar
- Long-decode gate v2: deterministic 2/2, valid complete Python

The earlier long-decode failure was a harness artifact: `ignore_eos` forced the
model past its completed code fence into a special-token tail, which the old
parser then treated as Python. Direct MTP-on/off comparison reproduced the tail
in both arms; the corrected gate validates only the completed module.

See [EVIDENCE.md](EVIDENCE.md) for the full matrix, provenance, and caveats.

## Hardware scope

The supplied production profile requires a 32 GB RTX 5090. SM120 compilation
also covers other GeForce Blackwell cards, but the 27B model plus the configured
8 GiB KV pool will not fit on 16 GB cards. SM121/DGX Spark is compile-supported
by the patch but has not been device-validated by this release.

## Build from source

Most users should pull the pinned prebuilt image. To rebuild:

```bash
./build.sh
```

The build script checks out the exact upstream commit, verifies the overlay
checksum, applies the 16-file patch, and uses vLLM's official CUDA Dockerfile.
This is source-reproducible; byte-for-byte image output can still vary with
BuildKit and network package repositories.

## License and attribution

Apache-2.0. The overlay modifies vLLM v0.27.1 and retains upstream licensing.
The pinned Qwen checkpoint is also Apache-2.0. See [NOTICE](NOTICE) and
[EVIDENCE.md](EVIDENCE.md).
