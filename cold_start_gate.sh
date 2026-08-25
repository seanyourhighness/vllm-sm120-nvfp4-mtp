#!/usr/bin/env bash
# cold_start_gate.sh — cold-cache boot regression gate for the pinned Hub
# revision through the strict MTP loader (issue #4 hardening).
#
# What it proves, end to end:
#   1. The pinned model revision downloads into an EMPTY HF cache and boots
#      under the strict MTP loader (0002 dequant shim active for NVFP4-packed
#      lm_head; BF16 fast path untouched).
#   2. /health comes up.
#   3. Content gates pass: needle-in-haystack + temp-0 determinism
#      (same semantics as correctness_gate.py).
#   4. MTP-3 is actually serving (spec decode metrics present).
#
# This gate intentionally uses a SCRATCH HF cache volume so "verified" always
# means a true first-download cold start — exactly what a fresh user gets.
#
# Usage:
#   ./cold_start_gate.sh <image>            # e.g. vllm-sm120-nvfp4-mtp:local-v0271-...
# Env:
#   MODEL_REPO    (default gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090)
#   MODEL_REVISION(default 69274a0d8dff5dd35bcee8290612f71e03b6e981)
#   HF_TOKEN      required if the repo is gated
#   PORT          (default 18080) host port for the server
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${1:?usage: ./cold_start_gate.sh <image>}"
MODEL_REPO="${MODEL_REPO:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
MODEL_REVISION="${MODEL_REVISION:-69274a0d8dff5dd35bcee8290612f71e03b6e981}"
PORT="${PORT:-18080}"
CTR="cold-start-gate-$$"
SCRATCH="hf-cold-gate-$$"   # scratch cache: guarantees a cold download
SERVED="qwen38-nvfp4-armb-mtp1-g1-fullpw-vision"
TEMPLATE="$HERE/chat-template.jinja"

cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; docker volume rm -f "$SCRATCH" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "== [1/4] cold boot from pinned revision $MODEL_REVISION (scratch cache: $SCRATCH)"

HF_TOKEN_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_ARGS=(-e "HF_TOKEN=${HF_TOKEN}")
fi

docker run -d --gpus all -p "$PORT":8000 --name "$CTR" \
  -v "$SCRATCH":/home/vllm/.cache/huggingface \
  -v "$TEMPLATE":/chat-template.jinja \
  "${HF_TOKEN_ARGS[@]}" \
  "$IMAGE" \
  --model "$MODEL_REPO" \
  --revision "$MODEL_REVISION" \
  --served-model-name "$SERVED" \
  --quantization modelopt --trust-remote-code --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --kv-cache-dtype nvfp4 --kv-cache-memory-bytes 8589934592 \
  --mamba-ssm-cache-dtype bfloat16 \
  --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --enable-prefix-caching --enable-chunked-prefill \
  --spec-method mtp --spec-tokens 3 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8]}' \
  --attention-config '{"flash_attn_version":2}' >/dev/null
echo "container $CTR up; first-boot download + load in progress"

# Poll /health with a generous timeout: cold download of ~28 GB + CUDA warmup.
BASE="http://127.0.0.1:$PORT"
DEADLINE=$(( $(date +%s) + ${BOOT_TIMEOUT:-3600} ))
until curl -fsS "$BASE/health" >/dev/null 2>&1; do
  if [[ $(date +%s) -ge $DEADLINE ]]; then
    echo "FAIL: /health not ready within ${BOOT_TIMEOUT:-3600}s" >&2
    docker logs --tail 80 "$CTR" >&2 || true
    exit 1
  fi
  sleep 15
done
echo "PASS: /health up after cold download+load"

echo "== [2/4] strict-loader sanity: no NVFP4 lm_head crash in logs"
if docker logs "$CTR" 2>&1 | grep -qiE "lm_head.*(uint8|packed|dequantiz)"; then
  echo "NOTE: 0002 dequant path engaged (NVFP4-packed head detected in this export)"
fi
if docker logs "$CTR" 2>&1 | grep -qiE "(size mismatch|RuntimeError).*(lm_head|weight_scale)"; then
  echo "FAIL: lm_head load error present in server log" >&2
  docker logs --tail 40 "$CTR" >&2
  exit 1
fi
echo "PASS: no lm_head loader errors"

echo "== [3/4] content gates (needle + determinism), same semantics as correctness_gate.py"
BASE_URL="$BASE" MODEL="$SERVED" python3 "$HERE/cold_start_content_gates.py"

echo "== [4/4] MTP serving check"
METRICS="$(curl -fsS "$BASE/metrics")"
if echo "$METRICS" | grep -qE "spec_decode_num_accepted_tokens_total"; then
  echo "PASS: spec-decode metrics present (MTP active)"
else
  echo "FAIL: no spec_decode metrics — MTP is not serving" >&2
  exit 1
fi

echo
echo "COLD-START GATE: ALL PASS"
echo "  image:     $IMAGE"
echo "  revision:  $MODEL_REVISION"
echo "  scratch:   docker volume $SCRATCH (removed on exit; rerun = another cold start)"
