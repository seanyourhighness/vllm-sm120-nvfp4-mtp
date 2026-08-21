#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

sha256sum --check SHA256SUMS
bash -n build.sh launch-server.sh launch-server-nomtp.sh start.sh status.sh stop.sh verify.sh
python3 -m py_compile sidecar.py correctness_gate.py bench/*.py

release_image="$(sed -n 's/^IMAGE=//p' .env.example)"
[[ -n "$release_image" ]]
grep -Fq "$release_image" launch-server.sh
grep -Fq "$release_image" launch-server-nomtp.sh
grep -Fq '      - ${MODEL_ID}' compose.yaml
grep -Fq '      - ${MODEL_REVISION}' compose.yaml
grep -Fq './chat-template.jinja:/opt/vllm-release/chat-template.jinja:ro' compose.yaml
grep -Fq '      - /opt/vllm-release/chat-template.jinja' compose.yaml
grep -Fq 'io.github.seanyourhighness.vllm.model-revision=' Dockerfile.release-metadata
grep -Fq 'io.github.seanyourhighness.vllm.patch-sha256=' Dockerfile.release-metadata
grep -Fq 'io.github.seanyourhighness.vllm.chat-template-sha256=' Dockerfile.release-metadata

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  docker compose --env-file .env.example -f compose.yaml config --quiet
  docker compose --env-file .env.example -f compose.yaml -f compose.nomtp.yaml config --quiet
fi

if [[ "${CHECK_PATCH_APPLY:-0}" == "1" ]]; then
  work="$(mktemp -d /tmp/vllm-release-check.XXXXXX)"
  trap 'rm -rf "$work"' EXIT
  git clone --filter=blob:none https://github.com/vllm-project/vllm.git "$work/vllm"
  git -C "$work/vllm" checkout --detach 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
  git -C "$work/vllm" apply --check "$ROOT/0001-v0271-sm120-nvfp4-kv-mtp-toolcall.patch"
fi

echo "release integrity: PASS"
