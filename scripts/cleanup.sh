#!/usr/bin/env bash
# Remove only resources owned by these experiments.
#
#   ./scripts/cleanup.sh                 # stop/remove labeled containers
#   ./scripts/cleanup.sh --runs-keep 5   # also keep the newest 5 run dirs
#   ./scripts/cleanup.sh --volume        # also remove the HF cache volume (re-download)
#   ./scripts/cleanup.sh --dry-run

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEEP=0
REMOVE_VOLUME=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs-keep) KEEP="${2:?missing N}"; shift 2 ;;
    --volume) REMOVE_VOLUME=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "[cleanup] containers labeled inference-exp=1"
NAMES=()
while IFS= read -r name; do
  [[ -n "${name}" ]] && NAMES+=("${name}")
done < <(docker ps -a --filter "label=inference-exp=1" --format '{{.Names}}' 2>/dev/null || true)
if [[ "${#NAMES[@]}" -eq 0 ]]; then
  echo "  none"
else
  for name in "${NAMES[@]}"; do
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      echo "  would remove ${name}"
    else
      echo "  removing ${name}"
      docker rm -f "${name}" >/dev/null
    fi
  done
fi

if [[ "${KEEP}" -gt 0 ]]; then
  echo "[cleanup] keeping newest ${KEEP} run directories"
  python3 - "${ROOT}/runs" "${KEEP}" "${DRY_RUN}" <<'PY'
import os
import shutil
import sys

runs_dir, keep, dry = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
if not os.path.isdir(runs_dir):
    raise SystemExit(0)
entries = []
for name in os.listdir(runs_dir):
    if name in ("_dryrun", "_engine_help", ".lock"):
        continue
    path = os.path.join(runs_dir, name)
    if os.path.isdir(path):
        entries.append((os.path.getmtime(path), path))
entries.sort(reverse=True)
for _mtime, path in entries[keep:]:
    if dry:
        print(f"  would remove {path}")
    else:
        print(f"  removing {path}")
        shutil.rmtree(path, ignore_errors=True)
PY
fi

if [[ "${REMOVE_VOLUME}" -eq 1 ]]; then
  if docker volume inspect inference-exp-hf >/dev/null 2>&1; then
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      echo "[cleanup] would remove volume inference-exp-hf"
    else
      echo "[cleanup] removing volume inference-exp-hf (model cache will be re-downloaded)"
      docker volume rm inference-exp-hf >/dev/null
    fi
  fi
fi

echo "[cleanup] done. Images are never removed."
