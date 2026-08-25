#!/usr/bin/env bash
# launch-server-nomtp.sh — A/B control arm: same canonical flags as
# launch-server.sh minus MTP spec decode (NVFP4 KV + 8 streams + 262k + 8 GiB
# pin, vision offloaded). Use for the MTP-off arm of the corruption A/B
# (source of record: GBrain vllm/2026-08-20-v0271-release-c8-corruption-ab).
set -euo pipefail

IMAGE="${1:-${IMAGE:-ghcr.io/seanyourhighness/vllm-sm120-nvfp4-mtp@sha256:506d9a1ab859f4accb75f3fdb881644917a8704e3bda39a10df8de65ffc61297}}"
MODEL_DIR="${MODEL_DIR:-/home/sean/Models/Qwen3.8-27B-NVFP4-RTX5090}"
PORT="${PORT:-18079}"
CTR_NAME="${CTR_NAME:-sm120-nvfp4-mtp-nomtp}"
TEMPLATE="$(cd "$(dirname "$0")" && pwd)/chat-template.jinja"

docker rm -f "$CTR_NAME" >/dev/null 2>&1 || true

docker run -d --gpus all -p "$PORT":8000 --name "$CTR_NAME" \
  -v "$MODEL_DIR":/models/model \
  -v "$TEMPLATE":/chat-template.jinja \
  "$IMAGE" \
  --model /models/model \
  --served-model-name qwen38-nvfp4-armb-mtp1-g1-fullpw-vision qwen38-nvfp4-armb-mtp1-g1-fullpw-vision-64k \
  --quantization modelopt --trust-remote-code --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false, "reasoning_effort": "medium"}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --chat-template /chat-template.jinja \
  --enable-mm-embeds --limit-mm-per-prompt '{"image":0,"video":0}' \
  --kv-cache-dtype nvfp4 --kv-cache-memory-bytes 8589934592 \
  --mamba-ssm-cache-dtype bfloat16 \
  --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --long-prefill-token-threshold 2048 --scheduling-policy priority \
  --enable-prefix-caching --enable-chunked-prefill \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32]}' \
  --attention-config '{"flash_attn_version":2}'

echo "started $CTR_NAME on :$PORT (no-MTP control; logs: docker logs -f $CTR_NAME)"
