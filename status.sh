#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
docker compose ps
echo
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo
docker volume inspect vllm-sm120-qwen38-model-cache --format 'model cache: {{.Mountpoint}}' 2>/dev/null || echo "model cache: not created"
