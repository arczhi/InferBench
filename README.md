# L4 inference lab: KV cache strategy + concurrency cliff

Two reproducible experiments for a **single NVIDIA L4 (24 GB, sm89)** shared with
lightweight web services. Both engines run in Docker, both use a **0.9 GPU memory
watermark**, and both are wrapped in an independent watchdog that can stop the
experiment before it disturbs the host.

The harness is **Python stdlib only**. No pip install is required on the host.
Only the vLLM/SGLang containers need their own dependencies.

## Run this first (on the target host)

```bash
python3 scripts/survey_l4.py
```

It is read-only: no pulls, no containers, no writes outside the JSON report.
It prints a machine fact sheet (GPU free memory, foreign GPU processes, host RAM,
Docker/NVIDIA runtime, disk headroom, firewall/registry/HF reachability, busy
ports) and a recommended config block. Save the JSON; it is the input to the
config decisions below.

## What the two experiments answer

### `exp_kv/` — KV cache management

Fixed 24 GB, fixed 0.9 watermark, long contexts (8K / 16K / 22.7K), sweep session
concurrency. Compare:

| arm | engine | KV strategy |
|---|---|---|
| `vllm-fp16-apc` | vLLM | bf16 KV + automatic prefix caching |
| `vllm-fp8-apc` | vLLM | FP8 KV + APC |
| `vllm-fp16-noapc` | vLLM | APC disabled (control) |
| `sglang-fp16-radix` | SGLang | bf16 KV + RadixAttention |
| `sglang-fp8-radix` | SGLang | FP8 KV + RadixAttention |
| `sglang-fp16-noapc` | SGLang | Radix cache disabled (control) |
| `sglang-fp8-hicache` | SGLang | FP8 KV + host-memory hierarchical cache (opt-in) |

Outputs: max safe concurrent sessions per context, cache hit rate, peak cache
occupancy, throughput, and a **decode break-even probe** (TPOT at batch 1 vs 8)
that tests whether FP8 KV actually lowers per-token decode cost on L4.

The Proxima/STAR-KV numbers are motivation, not a reproduced arm. That project is
an out-of-tree vLLM plugin; this harness deliberately uses native engine switches
first. SGLang HiCache is the native host-tier KV path included here.

### `exp_cliff/` — concurrency cliff and admission control

Agent-shaped, growing-context workload (tool-call request → injected tool result →
final answer, repeated for N turns, ~22.7K context in the full profile). Sweep
session concurrency and record TTFT p50/p95/p99, TPOT, throughput, cache
occupancy, queue depth, preemptions and prefix-cache hit rate.

Arms: default admission, an admission cap (`--max-num-seqs` /
`--max-running-requests`), FP8 KV, no-APC control. The output is the **cliff
concurrency** and the **safe operating region** per arm, plus a phase breakdown
showing how much of agent latency is the tool round-trip versus the final answer.

## Quickstart on the target

```bash
# 0. copy this directory to the host, then:
python3 scripts/survey_l4.py

# 1. inspect the survey, edit configs/lab.json only if the host differs
#    (container_memory_gb, cpus, hf_endpoint, ports)

# 2. pull images through the DaoCloud mirror (retries, disk check, digest lock)
./scripts/pull_images.sh
# optional: pre-fetch the model into the shared volume
./scripts/fetch_model.sh

# 3. verify the harness without touching the GPU
make test
make dry-run

# 4. a short real run to validate model download + engine flags
python3 -m exp_kv.run    --profile smoke --engine vllm
python3 -m exp_cliff.run --profile smoke --engine vllm

# 5. main runs (use tmux; full profiles are long)
python3 -m exp_kv.run    --profile quick --engine all
python3 -m exp_cliff.run --profile quick --engine all

# 6. full profiles overnight
python3 -m exp_kv.run    --profile full --engine vllm --include-hicache
python3 -m exp_cliff.run --profile full --engine vllm
```

Every run produces `runs/<experiment>/<run_id>/` with:

```
report.html          # summary + dependency-free SVG curves
report.md            # same content in markdown
levels.csv           # one row per (arm, context, concurrency)
summary.json         # machine-readable, includes guard abort reason
requests.jsonl       # every HTTP request: ttft, tpot, usage, phase
sessions.jsonl       # every synthetic session: turns completed, wall time
gpu.jsonl            # 1 Hz nvidia-smi
host.jsonl           # 1 Hz /proc load, memory, disk
engine_metrics.jsonl # 2 Hz Prometheus scrape
guard.jsonl          # warnings, heartbeats, abort evidence
engine.log           # container logs (tail is preserved on failure)
image.json           # image digest used by the arm
```

## Safety model (short version)

- Engine watermark is `0.9` (`--gpu-memory-utilization` / `--mem-fraction-static`).
  On a 24 GB card that leaves ~2.4 GB for CUDA context and the host.
- Container: memory cap, CPU cap, `cpu-shares=512` (below default web services),
  no restart, no privileged mode, no host network, port bound to `127.0.0.1`.
- Watchdog aborts on: GPU ≥87 °C, GPU memory ≥97 %, host available RAM ≤1.5 GiB,
  disk ≤5 GiB, load >2× cores for 60 s, container OOM/exit, request error rate
  >50 %, or wall-clock budget. Abort stops the **experiment container only**.
- Between arms the harness waits for the GPU to cool to ≤68 °C and for memory to
  be >85 % free.
- A lock file prevents two experiment processes from running at once.
- Only containers labeled `inference-exp=1` are ever touched.

Details and tuning: [docs/SAFETY.md](docs/SAFETY.md).

## Repository layout

```
scripts/survey_l4.py      read-only host survey (stdlib)
scripts/pull_images.sh    mirror pull + digest lock
scripts/fetch_model.sh    pre-download the model into the shared volume
scripts/fetch_model_modelscope.sh  download via ModelScope for offline hosts
scripts/cleanup.sh        remove only experiment containers / old runs
common/                   client, engine lifecycle, workload, guard, reports
exp_kv/                   KV strategy arms, levels, analysis
exp_cliff/                cliff arms, levels, analysis
configs/lab.json          single config (env + CLI overrides)
docs/                     design, safety, runbook
tests/                    stdlib unit + integration tests with a mock server
```

## Configuration

Precedence: `configs/lab.json` < environment < CLI flags.

Useful env overrides: `EXP_MODEL`, `EXP_HF_ENDPOINT`, `EXP_GPU_MEM_UTIL`,
`EXP_MAX_MODEL_LEN`, `EXP_CONTAINER_MEM_GB`, `EXP_CPUS`, `EXP_VLLM_PORT`,
`EXP_SGLANG_PORT`, `EXP_MAX_WALL_S`, `VLLM_IMAGE`, `SGLANG_IMAGE`.
`HF_TOKEN` is passed through to the container only if it is set on the host; it
is never written to disk.

## Model and image notes

- Default model: `Qwen/Qwen3-4B-Instruct-2507` (~8 GB bf16). It leaves roughly
  11–12 GB of KV budget at 0.9 on a 24 GB L4, which is enough to expose a cliff
  but not so small that the engine cannot start.
- `latest` images are pulled by default **and their digests are recorded**. For a
  paper-grade rerun, pin `VLLM_TAG` / `SGLANG_TAG` and keep
  `docker/images.lock.json`.
- vLLM on L4 uses an Ada-compatible attention backend (FA2/FlashInfer); results
  from FA3/Hopper papers do not transfer directly.
- FP8 KV support is hardware- and version-dependent. The harness validates CLI
  flags but cannot validate kernel support before startup; an unsupported arm
  fails cleanly and is recorded in `arm_summary.json`.
- If Hugging Face is unreachable, set `hf_endpoint` to `https://hf-mirror.com` or
  pre-populate the `inference-exp-hf` volume. On hosts where only ModelScope is
  reachable, use `scripts/fetch_model_modelscope.sh`, then set `model` to the
  absolute local path; the harness bind-mounts it read-only into the container.

## Known limitations

- This is a performance experiment. It does **not** evaluate answer quality under
  FP8/INT4 KV or quantization.
- The "agent" workload is synthetic: tool calls are forced text, results are
  injected, and no native tool parser is exercised. This isolates context shape
  and round-trip cost from parser behavior.
- vLLM PagedAttention + APC and SGLang RadixAttention are not identical designs;
  cross-engine numbers are directional, within-engine deltas are the strong claim.
- Planned token counts use a chars/3.6 estimate; the report also records exact
  `usage` from the server, which is the number to trust.
- The guard protects the host from the experiment, not from a broken driver or a
  pre-existing hardware fault. Run the survey first.

See [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md) for hypotheses and
metric definitions, and [docs/RUNBOOK.md](docs/RUNBOOK.md) for step-by-step
operation and troubleshooting.
