#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -f .env ]] || cp .env.example .env
set -a
# shellcheck disable=SC1091
source .env
set +a

for cmd in docker nvidia-smi curl python3; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "error: required command not found: $cmd" >&2; exit 1; }
done
docker compose version >/dev/null 2>&1 || { echo "error: Docker Compose plugin is required" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "error: Docker daemon is not reachable by this user" >&2; exit 1; }

# Probe a single device by index/UUID (default 0). No pipe: under
# `set -euo pipefail`, `nvidia-smi ... | head -n1` dies with SIGPIPE (141)
# on multi-GPU hosts when nvidia-smi is still writing after head exits.
gpu_line="$(nvidia-smi -i "${GPU_DEVICE:-0}" --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits)"
[[ -n "$gpu_line" ]] || {
  echo "error: nvidia-smi returned no data for GPU ${GPU_DEVICE:-0}; set GPU_DEVICE in .env to your RTX 5090's index or UUID" >&2
  exit 1
}
gpu_name="${gpu_line%%,*}"
gpu_rest="${gpu_line#*, }"
gpu_mem="${gpu_rest%%,*}"
gpu_cap="${gpu_line##*, }"
[[ "$gpu_cap" == "12.0" ]] || { echo "error: requires SM120; found $gpu_name (compute $gpu_cap) on GPU ${GPU_DEVICE:-0}" >&2; exit 1; }
(( gpu_mem >= 30000 )) || { echo "error: canonical profile requires ~32 GB VRAM; found ${gpu_mem} MiB" >&2; exit 1; }

free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
(( free_kb >= 45 * 1024 * 1024 )) || echo "warning: less than 45 GB free; downloads may exhaust disk" >&2

files=(-f compose.yaml)
mode="MTP-3"
if [[ "${1:-}" == "--no-mtp" ]]; then
  files+=(-f compose.nomtp.yaml)
  mode="no-MTP"
elif [[ -n "${1:-}" ]]; then
  echo "usage: ./start.sh [--no-mtp]" >&2
  exit 2
fi

echo "GPU: $gpu_name, ${gpu_mem} MiB, compute $gpu_cap"
echo "Pulling pinned runtime and starting $mode profile..."
docker compose "${files[@]}" pull
docker run --rm \
  -e NVIDIA_VISIBLE_DEVICES="${GPU_DEVICE:-0}" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --gpus all \
  --entrypoint nvidia-smi "$IMAGE" -L >/dev/null || {
  echo "error: Docker cannot access the GPU; configure NVIDIA Container Toolkit" >&2
  exit 1
}
docker compose "${files[@]}" up -d --remove-orphans

deadline=$((SECONDS + START_TIMEOUT_SECONDS))
echo "Waiting for vLLM and vision sidecar (first boot downloads ~20.6 GB)..."
while (( SECONDS < deadline )); do
  if curl -fsS "http://${BIND_ADDRESS}:${VLLM_PORT}/health" >/dev/null 2>&1 \
      && curl -fsS "http://${BIND_ADDRESS}:${VISION_PORT}/health" >/dev/null 2>&1; then
    echo "Services are healthy."
    ./verify.sh --smoke
    echo "OpenAI API: http://${BIND_ADDRESS}:${VLLM_PORT}/v1"
    echo "Vision proxy: http://${BIND_ADDRESS}:${VISION_PORT}/v1"
    exit 0
  fi
  if (( SECONDS % 30 < 10 )); then
    docker compose "${files[@]}" ps --format 'table {{.Service}}\t{{.Status}}' || true
  fi
  sleep 10
done

echo "error: startup timed out after ${START_TIMEOUT_SECONDS}s" >&2
docker compose "${files[@]}" ps >&2 || true
docker compose "${files[@]}" logs --tail=120 server vision >&2 || true
exit 1
