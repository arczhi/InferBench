# Safety model

The target host runs lightweight web services and one L4. The experiment must be
a good citizen: bounded memory, bounded CPU, bounded time, no host-network
exposure, and a watchdog that stops the experiment before the host becomes
unhappy. This document describes what is enforced and what is not.

## Memory math on a 24 GB L4

A 0.9 watermark asks the engine to reserve ~22.1 GiB for weights, activations,
CUDA graphs and KV cache. With Qwen3-4B bf16 (~8.5 GiB) and ~1.5 GiB of runtime
overhead, roughly **11–12 GiB** remains for KV. Other processes on the GPU reduce
that number directly. The survey reports foreign GPU memory; if it is non-trivial,
use 0.85 or 0.8 instead of 0.9.

The guard aborts if total GPU memory usage reaches **97 %** for two consecutive
samples. At that point the engine is at the edge of an OOM; stopping it is safer
than letting the driver kill a co-located process.

## Guard thresholds

All values are in `configs/lab.json` → `guard`.

| signal | warn | abort | rationale |
|---|---|---|---|
| GPU temperature | ≥80 °C | ≥87 °C (2 samples) | L4 throttles near its maximum operating temperature; leave headroom for the host |
| GPU memory used | — | ≥97 % (2 samples) | engine watermark is 90 %; 97 % means something else is consuming memory |
| Host available RAM | ≤3 GiB | ≤1.5 GiB (3 samples) | protects the web services and the page cache |
| Disk free | <50 GiB at survey | ≤5 GiB (runtime) | images, model cache, JSONL, container logs |
| Host load1 | >1.2× cores | >2× cores for 60 s | CPU starvation of the web tier |
| Request error rate | — | ≥50 % over 20 requests | engine overload, crash, or broken flag |
| Container | — | OOM-killed or exited | fail fast instead of measuring nothing |
| Wall clock | — | `max_wall_s` (7200 quick, 28800 full) | no runaway overnight job |

Every abort writes `abort.json` and a `guard.jsonl` record with the elapsed time
and the evidence. The runner saves the arm summary and any completed levels, so a
tripped run is still analyzable.

## What the guard does on abort

1. Sets an abort event.
2. A watcher thread calls `docker stop -t 30` and `docker rm -f` on the
   experiment container.
3. In-flight HTTP requests fail; the runner records the errors and stops
   scheduling new turns.
4. The guard, samplers, and engine log are flushed; `summary.json` is written with
   `"aborted": true` and the reason.

It does **not** reboot the host, reset the GPU, kill unrelated containers, or
change driver settings.

## Container isolation

- `--gpus device=0` (only GPU 0)
- `--shm-size 8g` instead of host IPC (clamped to half the container memory)
- `--memory 24g --memory-swap 24g` (no container swap)
- `--cpus 6 --cpu-shares 512` (web services keep the default 1024 shares and win
  under contention)
- `--pids-limit 4096`
- `--ulimit nofile=65535:65535`
- `--restart no`
- port published only on `127.0.0.1`
- json-file logs capped at 200 MB × 2
- label `inference-exp=1` on every container the harness may touch

The harness refuses to remove a container whose name collides but whose label is
missing. `scripts/cleanup.sh` filters by the same label.

## Host-level protection

- **One experiment at a time**: `runs/.lock` is an exclusive `flock`. A second
  process exits immediately.
- **Port check**: the engine refuses to start if 8000 (vLLM) or 30000 (SGLang) is
  already listening. Change `engine.vllm_port` / `engine.sglang_port` in the
  config if the web tier uses those ports.
- **Cooldown**: after each arm, wait until GPU ≤68 °C and >85 % of memory is free,
  up to 180 s.
- **Bounded request time**: 900 s per request, 1800–2400 s per concurrency level.
  A stuck engine is stopped, not waited on forever.
- **No image pruning**: the pull script never deletes images. Disk pressure is
  checked before pulling.

## External engine mode

`--external-url http://127.0.0.1:8000` attaches to an engine you started
yourself. The guard still samples and records, but **it cannot stop a server it
did not start**. Use this mode only when you are watching the machine, or accept
that an abort is advisory. The default Docker mode is the safe path.

## If the host is more sensitive than expected

Start with:

```json
{
  "engine": {"gpu_memory_utilization": 0.85},
  "docker": {"container_memory_gb": 16, "cpus": 4, "cpu_shares": 256},
  "guard": {"max_wall_s": 3600, "gpu_temp_abort_c": 84}
}
```

Then run the smoke profile and inspect `gpu.jsonl` (clocks, temperature, power)
and `host.jsonl` (available RAM, load) before committing to a full sweep. If the
web tier is latency-sensitive during the run, lower `cpus` further; prefill is
GPU-bound, but tokenization, scheduling and JSON serialization are not.

## Thermal notes

The L4 is passively cooled in many servers. Sustained prefill at 0.9 can raise
the card to its power limit; the guard records `power_w`, `sm_count` clocks and
`pstate` at 1 Hz. If you see repeated 80 °C warnings, improve chassis airflow
before raising any threshold. Do not raise the abort temperature above the card's
rated maximum.

## Data safety

- `HF_TOKEN` is read from the host environment and passed as a Docker env var; it
  is never written to the repo or the run directory.
- Run directories contain prompts and model outputs. They are synthetic, but
  treat them as logs: they can be large. `scripts/cleanup.sh --runs-keep N` removes
  old runs; it never touches the HF cache unless `--volume` is given explicitly.
