# Runbook

## 0. Prerequisites on the target host

- 1 × NVIDIA L4 (24 GB), driver exposing `nvidia-smi`
- Docker 24+ with the `nvidia` runtime (`nvidia-container-toolkit`)
- Python 3.8+ (stdlib only; no pip install)
- ~50 GiB free disk for two images + HF cache + logs
- ~16 GiB host RAM (24 GiB recommended); the survey tells you the real number
- Outbound access to `docker.m.daocloud.io` (or a local mirror) and to a model
  source (Hugging Face or `hf-mirror.com`)

## 1. Survey the host (read-only)

```bash
cd inference_exp
python3 scripts/survey_l4.py --json-out survey.json
```

Read the `[FAIL]` / `[WARN]` lines and the recommendation block. The JSON is the
record of the machine state at experiment time. If `docker.m.daocloud.io` is not
reachable, fix DNS/proxy first; the pull script will fail the same way.

## 2. Configure

Open `configs/lab.json`. Usually only these need to change:

| key | why |
|---|---|
| `model` | if you want a different Qwen model or a local path |
| `hf_endpoint` | `https://hf-mirror.com` when Hugging Face is slow/blocked |
| `docker.container_memory_gb` | clamp to ~45 % of host RAM |
| `docker.cpus` | leave at least half the cores for the web services |
| `engine.vllm_port` / `engine.sglang_port` | if 8000 / 30000 are busy |
| `engine.gpu_memory_utilization` | 0.85 if the survey shows foreign GPU memory |
| `slo` | fix this **before** the run; the safe region is defined by it |

The harness also clamps container memory to 45 % of host RAM at load time and
prints a warning. It does not silently raise limits.

## 2.5 Model access (check this before pulling)

The survey probes Hugging Face, hf-mirror, and ModelScope. If Hugging Face is
unreachable, pick one of these:

**A. Local model directory (recommended for offline hosts).** Download the model
from ModelScope on the host (or copy it from another machine), then set an
absolute path in `configs/lab.json`:

```bash
python3 -m pip install -U modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
./scripts/fetch_model_modelscope.sh
# writes /data/models/Qwen3-4B-Instruct-2507
```

```json
"model": "/data/models/Qwen3-4B-Instruct-2507"
```

The harness bind-mounts that directory read-only into the container at the same
absolute path. No Hugging Face access is needed at runtime.

**B. Internal HF proxy.** Set `hf_endpoint` in `configs/lab.json` to your proxy
and keep the repo id as `model`.

**C. Pre-populate the HF cache volume.** Use `scripts/fetch_model.sh` on a machine
with access, then copy the volume; this keeps the repo id as `model`.

Do not start a run until one of these works; otherwise the engine startup will
spend the full timeout retrying downloads.

## 3. Pull images

```bash
./scripts/pull_images.sh --check     # connectivity + disk only
./scripts/pull_images.sh             # pull both, retry 3×, record digests
```

The script writes `docker/images.lock.json`. Keep that file with the results.
To pin versions later:

```bash
VLLM_TAG=<version> SGLANG_TAG=<version> ./scripts/pull_images.sh
```

## 4. Verify the harness (no GPU work)

```bash
make test
make dry-run
```

`make test` uses a local mock OpenAI server; it should pass on any OS. The dry run
prints the exact `docker run` command for each arm and the level plan. Check that
the model name, context, ports and memory limits are what you expect.

## 5. Smoke run (model download + real engine)

Use `tmux` or `screen`; the first model download can take minutes.

```bash
tmux new -s exp
python3 -m exp_kv.run    --profile smoke --engine vllm
python3 -m exp_cliff.run --profile smoke --engine vllm
```

If the engine starts and the report has data, the environment is ready. If an arm
is skipped, read `arm_summary.json` → `skipped_reason` (usually an unsupported
engine flag).

## 6. Main runs

Quick (roughly 30–60 min depending on host and model download):

```bash
python3 -m exp_kv.run    --profile quick --engine all
python3 -m exp_cliff.run --profile quick --engine all
```

Full (several hours; run one experiment at a time):

```bash
python3 -m exp_kv.run    --profile full --engine vllm --include-hicache
python3 -m exp_cliff.run --profile full --engine all
```

Why one at a time: the lock file, the single GPU, and the 0.9 watermark. Two
engines would compete for the same 24 GB and invalidate both runs.

## 7. Collect results

Each run writes to `runs/<experiment>/<run_id>/`. Fetch at least:

```bash
scp -r host:inference_exp/runs/cliff/<run_id> ./results/
```

Copy `report.html`, `levels.csv`, `summary.json`, `image.json`, `guard.jsonl`,
`engine.log`, and the original `survey.json`. If a run aborted, `abort.json` is
the first file to read.

## 8. Clean up

```bash
./scripts/cleanup.sh                 # experiment containers only
./scripts/cleanup.sh --runs-keep 10  # keep the newest 10 run dirs
./scripts/cleanup.sh --volume        # also delete the HF cache (re-download later)
```

Images are never removed automatically.

## Troubleshooting

### `image ... is not present`

Run `./scripts/pull_images.sh`, or set `VLLM_IMAGE` / `SGLANG_IMAGE` to an image
you already have. The harness refuses to pull implicitly so that a run never
blocks on a surprise multi-GB download.

### Pull fails through the mirror

```bash
curl -sS -o /dev/null -w '%{http_code}\n' https://docker.m.daocloud.io/v2/
```

`401` or `403` means reachable (auth challenge). `000` means DNS/TLS/proxy. As a
last resort, `ALLOW_DOCKERHUB_FALLBACK=1 ./scripts/pull_images.sh` will pull from
Docker Hub and retag. Record which path was used.

### `port 8000 is already in use`

Something else is listening. Change `engine.vllm_port` (and `sglang_port`) in the
config, or stop the other service. The survey lists busy ports.

### Container exits during startup

Read `runs/.../engine.log` and `container_state.json`. Common causes:

- **OOM killed**: lower `gpu_memory_utilization` to 0.85, lower `max_model_len`,
  or remove the foreign GPU process. Check `container_state.json.oom_killed`.
- **Unsupported flag**: the image is older/newer than this harness. The arm's
  `skipped_reason` lists missing flags; remove or update the arm.
- **CUDA/driver mismatch**: `nvidia-smi` works but the container cannot see the
  GPU. Check `docker run --rm --gpus device=0 <image> nvidia-smi`. `latest` images
  may require a newer driver than the host has; pin an older tag compatible with
  the driver shown in the survey.
- **`No module named sglang` / `No module named vllm`**: the image keeps its
  Python in a venv that is not first on `PATH`. Set `docker.python_bin` in
  `configs/lab.json` to the image's interpreter (for example
  `/opt/venv/bin/python3`), or run
  `docker run --rm --entrypoint which <image> python3` to find it.
- **Model download failed**: 401/403 (gated model or missing `HF_TOKEN`), DNS, or
  disk full. Try `EXP_HF_ENDPOINT=https://hf-mirror.com`.
- **Context too large**: `max_model_len` exceeds what the model/config supports.
  Qwen3-4B supports 32K natively; do not raise it without a YaRN config.

### Engine ready, requests return 400

- `model` in the payload must equal `--served-model-name`. The harness sends
  `cfg.served_model_name`; if you attach to an external engine, set
  `EXP_SERVED_MODEL_NAME` to its actual id.
- Some engines reject `stream_options`. The client retries without it and marks
  `usage_estimated=true`.
- The prompt may exceed `max_model_len`; the builder reserves 256 tokens and
  drops old turns, but an unusually long system prompt can still overflow.

### `--enable-prefix-caching` is not listed by the image

vLLM v1 enables APC by default. The runner drops that one flag and continues; the
control arm `--no-enable-prefix-caching` will still be skipped if the image does
not know it. Check the image's `--help` output saved under
`runs/_engine_help/`.

### SGLang FP8 / HiCache arm fails

SGLang flags and hardware support change between releases. The harness validates
the CLI, not the kernel. A clean startup failure is recorded as an arm error; do
not treat it as a harness bug. Pin a known-good SGLang tag for a comparable run.

### Guard aborts on temperature

Improve airflow, lower the ambient temperature, or run a shorter profile. Do not
raise `gpu_temp_abort_c` above the card's rated maximum. Check `power_w` and
`sm_clock_mhz` in `gpu.jsonl` to see whether the card was power-limited or
thermally limited.

### Guard aborts on host memory

Lower `docker.container_memory_gb`, stop other memory-heavy processes, or reduce
`max_model_len` (a smaller KV budget means less host-side bookkeeping, though the
big win is the container cap). The web services should never be the ones
sacrificed.

### Throughput or TTFT looks wrong at N=1

The first level includes any remaining CUDA graph capture and prefix-cache warmup.
The harness performs a short and a long warmup before the sweep and excludes both
from statistics. If N=1 is still anomalous, compare `gpu.jsonl` clocks across
runs; a thermally throttled card produces a different baseline.

### `another experiment is running`

A previous process still holds `runs/.lock`. Check `ps` and the PID recorded in
that file. If the process is gone, remove the stale lock file manually; the lock
is advisory and does not survive a reboot.

### Report has no charts

Charts need at least one completed level with a numeric metric. Check
`summary.json` → `arms[].status` and `error`. Skipped arms have no levels by
design.

## Experiment hygiene

- Do not change the SLO, context lengths, or model between arms of the same run.
- Record the image digest (`docker/images.lock.json`) and the survey JSON.
- Run quick profiles first; a full profile is expensive and should be a deliberate
  choice, not a default.
- If you change the workload builder, rerun `make test`; the budget/determinism
  tests are the guardrail.
