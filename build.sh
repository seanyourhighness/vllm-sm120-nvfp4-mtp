#!/usr/bin/env bash
# build.sh — reproducible build of the SM120 NVFP4-KV + MTP-3 vLLM image
#
# Recipe (matches the verified local build, 2026-08-16/19/20):
#   upstream vllm v0.27.1 source + 0001-v0271-...patch (16 files, +514/-99)
#   + official vLLM Dockerfile (CUDA 13.0.3, FlashInfer 0.6.16.post3 from
#     the tree's requirements/cuda.txt)
#
# Prereqs: docker, git, ~15 GB disk for the tree, ~1 h build time, GPU node
#          for the smoke test.
# Usage:   ./build.sh          # build only
#          ./build.sh --push   # build + tag + push to ghcr.io (needs a token
#                              # with write:packages: GHCR_PAT=... or
#                              # GHCR_PAT="$(gh auth token)" after
#                              # `gh auth refresh -s write:packages`)
set -euo pipefail

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
VLLM_VERSION=v0.27.1
VLLM_COMMIT=6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
PATCH_SHA256=55f127c2ef353dda5dbf1af7d1efae6a792bb152ca93eda6d0609a029af284ae
SOURCE_URL=https://github.com/seanyourhighness/vllm-sm120-nvfp4-mtp
# GHCR_OWNER only matters for --push. Default to the account the release is
# published from; a local build never needs it.
GHCR_OWNER="${GHCR_OWNER:-seanyourhighness}"
TAG="v0271-fi616-$(date +%Y%m%d)"
LOCAL_IMG="vllm-sm120-nvfp4-mtp:local-${TAG}"
PUSH_REPO="ghcr.io/${GHCR_OWNER}/vllm-sm120-nvfp4-mtp"
PUSH_IMG="${PUSH_REPO}:${TAG}"

# Target: consumer Blackwell (RTX 50-series, SM120, x86_64) by default.
# For an NVIDIA DGX Spark / GB10 (ARM64, SM121), run with SPARK=1. Cross-
# building arm64 on an x86 host needs buildx + QEMU emulation (slow); the
# same script run natively on the Spark is the fast path.
if [[ "${SPARK:-0}" == "1" ]]; then
  BUILD_PLATFORM="linux/arm64"
  TORCH_ARCH_LIST="12.1"
else
  BUILD_PLATFORM="linux/amd64"
  TORCH_ARCH_LIST="12.0"
fi

WORK="$(mktemp -d /tmp/vllm-sm120-build.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

echo "==> cloning vllm ${VLLM_VERSION} (${VLLM_COMMIT})"
git clone --filter=blob:none https://github.com/vllm-project/vllm.git "$WORK/vllm"
git -C "$WORK/vllm" checkout --detach "$VLLM_COMMIT"

echo "==> applying sm120-nvfp4-mtp overlay"
actual_patch_sha="$(sha256sum "$SCRIPT_DIR/0001-v0271-sm120-nvfp4-kv-mtp-toolcall.patch" | cut -d' ' -f1)"
[[ "$actual_patch_sha" == "$PATCH_SHA256" ]] || {
  echo "patch checksum mismatch: expected $PATCH_SHA256, got $actual_patch_sha" >&2
  exit 1
}
# Fresh /tmp clone has no committer identity — needed only for cleanliness.
git -C "$WORK/vllm" config user.name  "${GIT_USER_NAME:-vllm-sm120-nvfp4-mtp-build}"
git -C "$WORK/vllm" config user.email "${GIT_USER_EMAIL:-build@local}"
git -C "$WORK/vllm" apply --index "$SCRIPT_DIR/0001-v0271-sm120-nvfp4-kv-mtp-toolcall.patch"

echo "==> building image (official vLLM Dockerfile, CUDA 13.0.3, SM120-only)"
# Parallelism: vLLM's setup.py computes num_jobs = MAX_JOBS // NVCC_THREADS,
# so to get N parallel nvcc invocations we pass MAX_JOBS = N * NVCC_THREADS.
# MAX_JOBS here means "parallel nvcc processes". With SM120-only archs each
# nvcc stays modest; 3 jobs x 2 threads is a safe default on a 32-64 GB host.
# On 128 GB hosts, MAX_JOBS=6 NVCC_THREADS=8 works well.
PARALLEL_JOBS="${MAX_JOBS:-3}"
NVCC_THREADS="${NVCC_THREADS:-2}"
docker build \
  --platform "$BUILD_PLATFORM" \
  -f "$WORK/vllm/docker/Dockerfile" \
  --build-arg CUDA_VERSION=13.0.3 \
  --build-arg torch_cuda_arch_list="$TORCH_ARCH_LIST" \
  --build-arg max_jobs="$((PARALLEL_JOBS * NVCC_THREADS))" \
  --build-arg nvcc_threads="$NVCC_THREADS" \
  --label "org.opencontainers.image.source=${SOURCE_URL}" \
  --label "org.opencontainers.image.revision=${VLLM_COMMIT}" \
  --label "org.opencontainers.image.version=${TAG}" \
  --label "org.opencontainers.image.licenses=Apache-2.0" \
  -t "$LOCAL_IMG" \
  "$WORK/vllm"

echo "==> smoke test (on a GPU node):"
echo "  ./launch-server.sh $LOCAL_IMG"
echo "  # expect boot OK + MTP-3 serving, KV pool ~373k tokens at the 8 GiB pin"

if [[ "${1:-}" == "--push" ]]; then
  : "${GHCR_PAT:?set GHCR_PAT (GitHub token with write:packages)}"
  docker tag "$LOCAL_IMG" "$PUSH_IMG"
  docker tag "$LOCAL_IMG" "${PUSH_REPO}:v0271"
  echo "==> logging into ghcr.io"
  echo "$GHCR_PAT" | docker login ghcr.io --username "$GHCR_OWNER" --password-stdin
  echo "==> pushing ${PUSH_IMG} (~10 GB)"
  docker push "$PUSH_IMG"
  docker push "${PUSH_REPO}:v0271"
  echo "done: ${PUSH_IMG} and ${PUSH_REPO}:v0271"
fi
