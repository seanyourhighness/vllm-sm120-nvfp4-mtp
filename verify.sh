#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
set -a
# shellcheck disable=SC1091
source "${ROOT}/.env"
set +a

base="http://${BIND_ADDRESS}:${VLLM_PORT}"
vision="http://${BIND_ADDRESS}:${VISION_PORT}"
curl -fsS "$base/health" >/dev/null
curl -fsS "$vision/health" >/dev/null
response="$(curl -fsS "$base/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"temperature\":0,\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: READY\"}]}")"
python3 - "$response" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
text = payload["choices"][0]["message"].get("content") or ""
if "READY" not in text.upper():
    raise SystemExit(f"smoke completion did not contain READY: {text!r}")
print("PASS: health, vision health, model routing, and chat completion")
PY

if [[ "${1:-}" == "--full" ]]; then
  BASE_URL="$vision" MODEL="$SERVED_MODEL_NAME" \
    IMAGE_PATH="$ROOT/bench/vision_test_a.png" python3 correctness_gate.py
elif [[ "${1:-}" != "--smoke" && -n "${1:-}" ]]; then
  echo "usage: ./verify.sh [--smoke|--full]" >&2
  exit 2
fi
