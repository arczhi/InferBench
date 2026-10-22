#!/usr/bin/env bash
# Pre-fetch the model into the shared HF cache volume.
#
#   ./scripts/fetch_model.sh
#   MODEL=Qwen/Qwen3-4B-Instruct-2507 HF_ENDPOINT=https://hf-mirror.com ./scripts/fetch_model.sh
#   MODEL_IMAGE=docker.m.daocloud.io/lmsysorg/sglang:latest ./scripts/fetch_model.sh
#
# Uses a one-off container from an image that is already present. Never pulls.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_IMAGE="${MODEL_IMAGE:-docker.m.daocloud.io/vllm/vllm-openai:latest}"
VOLUME="${HF_CACHE_VOLUME:-inference-exp-hf}"
HF_ENDPOINT="${HF_ENDPOINT:-}"

command -v docker >/dev/null 2>&1 || { echo "[FAIL] docker not found" >&2; exit 1; }
docker image inspect "${MODEL_IMAGE}" >/dev/null 2>&1 || {
  echo "[FAIL] image not present: ${MODEL_IMAGE}" >&2
  echo "       run ./scripts/pull_images.sh first" >&2
  exit 1
}

ARGS=(
  run --rm
  --entrypoint python3
  -v "${VOLUME}:/root/.cache/huggingface"
  -e HF_HOME=/root/.cache/huggingface
  -e HF_HUB_DISABLE_TELEMETRY=1
)
if [[ -n "${HF_ENDPOINT}" ]]; then
  ARGS+=(-e "HF_ENDPOINT=${HF_ENDPOINT}")
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  ARGS+=(-e HF_TOKEN)
fi

echo "[fetch] model=${MODEL} image=${MODEL_IMAGE} endpoint=${HF_ENDPOINT:-default}"
docker "${ARGS[@]}" "${MODEL_IMAGE}" - "${MODEL}" <<'PY'
import sys
from huggingface_hub import snapshot_download

model = sys.argv[1]
path = snapshot_download(repo_id=model)
print(f"[fetch] ready: {path}")
PY

echo "[fetch] cache volume contents:"
docker run --rm -v "${VOLUME}:/cache" --entrypoint sh "${MODEL_IMAGE}" -c \
  'du -sh /cache 2>/dev/null || true; ls -1 /cache/hub 2>/dev/null | head -20 || true'
