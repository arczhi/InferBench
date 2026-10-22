#!/usr/bin/env bash
# Download the model from ModelScope to a local directory.
# Use this when huggingface.co / hf-mirror.com are unreachable but
# www.modelscope.cn is reachable (common on mainland-China GPU hosts).
#
#   python3 -m pip install -U modelscope \
#       -i https://pypi.tuna.tsinghua.edu.cn/simple
#   ./scripts/fetch_model_modelscope.sh
#   MODEL=Qwen/Qwen3-8B DEST=/data/models/Qwen3-8B ./scripts/fetch_model_modelscope.sh
#
# The harness can bind-mount an absolute model path read-only into the
# container, so after this finishes set in configs/lab.json:
#   "model": "/data/models/Qwen3-4B-Instruct-2507"

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DEST="${DEST:-/data/models/$(basename "${MODEL}")}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MIN_FREE_GB="${MIN_FREE_GB:-15}"

log() { printf '[fetch] %s\n' "$*"; }
die() { printf '[fetch][FAIL] %s\n' "$*" >&2; exit 1; }

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "python3 not found"

if ! "${PYTHON_BIN}" -c "import modelscope" >/dev/null 2>&1; then
  cat >&2 <<EOF
[fetch][FAIL] the 'modelscope' package is not installed.
Install it into the current Python environment, for example:
  ${PYTHON_BIN} -m pip install -U modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
Then rerun this script.
EOF
  exit 1
fi

mkdir -p "$(dirname "${DEST}")"
FREE_GB="$(df -Pk "$(dirname "${DEST}")" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
log "model=${MODEL}"
log "dest=${DEST} (${FREE_GB} GiB free on the target filesystem)"
if [[ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]]; then
  die "need at least ${MIN_FREE_GB} GiB free; found ${FREE_GB} GiB"
fi

"${PYTHON_BIN}" - "${MODEL}" "${DEST}" <<'PY'
import sys
from modelscope import snapshot_download

model, dest = sys.argv[1], sys.argv[2]
path = snapshot_download(model, local_dir=dest)
print(f"[fetch] ready: {path}")
PY

[[ -f "${DEST}/config.json" ]] || die "download finished but ${DEST}/config.json is missing"

log "model files:"
du -sh "${DEST}" 2>/dev/null || true
ls -1 "${DEST}" | head -20

cat <<EOF

[fetch] done.
Set this in configs/lab.json (the harness mounts the absolute path read-only):
  "model": "${DEST}"

Then run:
  python3 -m exp_kv.run    --profile smoke --engine vllm
  python3 -m exp_cliff.run --profile smoke --engine vllm
EOF
