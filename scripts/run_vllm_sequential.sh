#!/usr/bin/env bash
# Run the two vLLM experiments sequentially in one process tree.
# The harness lock forbids running them concurrently on the same GPU.
#
#   nohup ./scripts/run_vllm_sequential.sh quick > logs/vllm-quick.log 2>&1 &
#   nohup ./scripts/run_vllm_sequential.sh full  > logs/vllm-full.log  2>&1 &
#
# Stop (from another shell):
#   pkill -f 'run_vllm_sequential.sh|exp_kv.run|exp_cliff.run'
#   ./scripts/cleanup.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PROFILE="${1:-quick}"
export PYTHONUNBUFFERED=1

if [[ -z "${EXP_MODEL:-}" ]]; then
  if [[ -f /data/models/Qwen3-4B-Instruct-2507/config.json ]]; then
    export EXP_MODEL=/data/models/Qwen3-4B-Instruct-2507
  else
    echo "[FAIL] EXP_MODEL is not set and the default local model was not found" >&2
    echo "       export EXP_MODEL=/data/models/Qwen3-4B-Instruct-2507" >&2
    exit 2
  fi
fi
if [[ ! -f "${EXP_MODEL}/config.json" ]]; then
  echo "[FAIL] no config.json under EXP_MODEL=${EXP_MODEL}" >&2
  exit 2
fi

echo "=== sequential vLLM experiments ==="
echo "profile : ${PROFILE}"
echo "model   : ${EXP_MODEL}"
echo "started : $(date -Is)"
echo

python3 -u -m exp_kv.run --profile "${PROFILE}" --engine vllm
KV_RC=$?
echo
echo "kv exit=${KV_RC} at $(date -Is)"
echo

python3 -u -m exp_cliff.run --profile "${PROFILE}" --engine vllm
CLIFF_RC=$?
echo
echo "cliff exit=${CLIFF_RC} at $(date -Is)"
echo "finished: $(date -Is)"

if [[ "${KV_RC}" -ne 0 && "${CLIFF_RC}" -ne 0 ]]; then
  exit 1
fi
exit 0
