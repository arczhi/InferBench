#!/usr/bin/env bash
# Pull vLLM and SGLang images through the DaoCloud Docker Hub mirror.
#
#   ./scripts/pull_images.sh                 # pull both, tag=latest
#   VLLM_TAG=v0.10.1 SGLANG_TAG=v0.4.10 ./scripts/pull_images.sh
#   ./scripts/pull_images.sh --check         # connectivity + disk only
#   ./scripts/pull_images.sh --no-sglang
#   ./scripts/pull_images.sh --dry-run
#   ALLOW_DOCKERHUB_FALLBACK=1 ./scripts/pull_images.sh
#
# No image is deleted. Digests are written to docker/images.lock.json so the
# experiment results can be tied to exact images.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIRROR="${MIRROR:-docker.m.daocloud.io}"
VLLM_REPO="${VLLM_REPO:-vllm/vllm-openai}"
SGLANG_REPO="${SGLANG_REPO:-lmsysorg/sglang}"
VLLM_TAG="${VLLM_TAG:-latest}"
SGLANG_TAG="${SGLANG_TAG:-latest}"
VLLM_IMAGE="${VLLM_IMAGE:-${MIRROR}/${VLLM_REPO}:${VLLM_TAG}}"
SGLANG_IMAGE="${SGLANG_IMAGE:-${MIRROR}/${SGLANG_REPO}:${SGLANG_TAG}}"
LOCK_FILE="${ROOT}/docker/images.lock.json"
MIN_FREE_GB="${MIN_FREE_GB:-35}"

PULL_VLLM=1
PULL_SGLANG=1
CHECK_ONLY=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1 ;;
    --no-vllm) PULL_VLLM=0 ;;
    --no-sglang) PULL_SGLANG=0 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[FAIL] %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker CLI not found"
docker info >/dev/null 2>&1 || die "docker daemon not reachable (permission or not running)"

DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
FREE_GB="$(df -Pk "${DOCKER_ROOT}" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024/1024}' || echo 0)"
log "docker root: ${DOCKER_ROOT} (${FREE_GB} GiB free)"

if [[ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]]; then
  echo "[WARN] less than ${MIN_FREE_GB} GiB free; two engine images + model need ~50 GiB." >&2
  if [[ "${FREE_GB}" -lt 15 ]]; then
    die "not enough disk space (${FREE_GB} GiB)"
  fi
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  log "checking mirror connectivity..."
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 "https://${MIRROR}/v2/" || true)"
  case "${code}" in
    200|401|403) log "mirror reachable (HTTP ${code})" ;;
    *) die "mirror not reachable (HTTP ${code:-none})" ;;
  esac
  log "check-only: not pulling"
  exit 0
fi

ensure_image() {
  local image="$1"
  local label="$2"
  local fallback="$3"

  if docker image inspect "${image}" >/dev/null 2>&1; then
    if [[ "${image}" != *":latest" ]]; then
      log "${label}: already present: ${image}"
      return 0
    fi
    log "${label}: refreshing mutable :latest tag"
  fi

  local attempt
  for attempt in 1 2 3; do
    log "${label}: pull attempt ${attempt}/3: ${image}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      echo "  docker pull ${image}"
      return 0
    fi
    if docker pull "${image}"; then
      log "${label}: pulled"
      return 0
    fi
    if [[ "${attempt}" -lt 3 ]]; then
      log "${label}: pull failed, sleeping $((attempt * 10))s"
      sleep $((attempt * 10))
    fi
  done

  if [[ -n "${fallback}" && "${ALLOW_DOCKERHUB_FALLBACK:-0}" == "1" ]]; then
    log "${label}: mirror failed, falling back to ${fallback}"
    if docker pull "${fallback}"; then
      docker tag "${fallback}" "${image}"
      log "${label}: pulled from Docker Hub and tagged ${image}"
      return 0
    fi
  fi
  return 1
}

FAILED=0
if [[ "${PULL_VLLM}" -eq 1 ]]; then
  ensure_image "${VLLM_IMAGE}" "vllm" "docker.io/${VLLM_REPO}:${VLLM_TAG}" || FAILED=1
fi
if [[ "${PULL_SGLANG}" -eq 1 ]]; then
  ensure_image "${SGLANG_IMAGE}" "sglang" "docker.io/${SGLANG_REPO}:${SGLANG_TAG}" || FAILED=1
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "dry-run: nothing pulled"
  exit 0
fi

mkdir -p "$(dirname "${LOCK_FILE}")"
python3 - "$LOCK_FILE" "${VLLM_IMAGE}" "${SGLANG_IMAGE}" "${PULL_VLLM}" "${PULL_SGLANG}" <<'PY'
import datetime
import json
import subprocess
import sys

lock_path, vllm_image, sglang_image, pull_vllm, pull_sglang = sys.argv[1:6]
images = []
for enabled, name, image in ((pull_vllm, "vllm", vllm_image), (pull_sglang, "sglang", sglang_image)):
    if enabled != "1":
        continue
    try:
        raw = subprocess.run(
            ["docker", "image", "inspect", image],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout
        data = json.loads(raw)[0]
        images.append({
            "name": name,
            "image": image,
            "id": data.get("Id"),
            "repo_digests": data.get("RepoDigests"),
            "created": data.get("Created"),
            "size_bytes": data.get("Size"),
        })
    except Exception as exc:
        images.append({"name": name, "image": image, "error": str(exc)})

payload = {
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "images": images,
}
with open(lock_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] wrote {lock_path}")
for item in images:
    print(f"  {item['name']:7s} {item.get('id', '?')[:19]}  {item.get('image')}")
PY

if [[ "${FAILED}" -ne 0 ]]; then
  die "one or more images failed to pull"
fi

log "done. Recorded digests in ${LOCK_FILE}"
log "Tip: pin tags for a paper-grade rerun, e.g. VLLM_TAG=<version> SGLANG_TAG=<version>."
