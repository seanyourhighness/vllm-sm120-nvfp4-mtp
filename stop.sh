#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
docker compose -f compose.yaml -f compose.nomtp.yaml down --remove-orphans
if [[ "${1:-}" == "--purge-cache" ]]; then
  docker volume rm vllm-sm120-qwen38-model-cache
elif [[ -n "${1:-}" ]]; then
  echo "usage: ./stop.sh [--purge-cache]" >&2
  exit 2
fi
