# Experiment design

## Common setup

| item | value |
|---|---|
| GPU | 1 × NVIDIA L4, 24 GB, compute capability 8.9 (Ada) |
| GPU watermark | `0.9` (`--gpu-memory-utilization` for vLLM, `--mem-fraction-static` for SGLang) |
| Model | `Qwen/Qwen3-4B-Instruct-2507`, bf16 weights |
| Max model length | 32768 (fixed across arms) |
| Engine | one container at a time, latest vLLM or SGLang image from `docker.m.daocloud.io` |
| Host | shared with lightweight web services; container gets capped CPU/RAM and lower CPU shares |
| Timing | TTFT and TPOT measured client-side from the streaming HTTP response |
| Sampling | GPU/host at 1 Hz, engine `/metrics` at 2 Hz |

`max_model_len` is held at 32768 for every arm so that the number of KV blocks is
determined by the KV dtype and the engine, not by our workload. The workload then
varies the *actual* context length inside that budget.

### Token accounting

- Planned token counts use `len(text)/3.6` for English prompts. This is only used
  to stay inside the context budget.
- Every report and CSV uses the server's `usage.prompt_tokens`,
  `usage.completion_tokens`, and `usage.prompt_tokens_details.cached_tokens` when
  present. If the server omits usage, the record is marked `usage_estimated=true`.
- Prefix-cache hit rate is `cached_tokens / prompt_tokens` from usage. Engine
  counters (`vllm:prefix_cache_hits_total` / `queries_total`,
  `sglang:cached_tokens_total` / `prompt_tokens_total`) are recorded as a
  secondary, cross-checking signal.

### Metric definitions

- **TTFT**: time from sending the request to the first streamed chunk that
  contains content or a tool call. Role-only chunks do not count.
- **TPOT**: `(e2e - TTFT) / (completion_tokens - 1)`. For very short completions
  this is noisy; the decode probe uses 256 output tokens on purpose.
- **Throughput**: output tokens divided by the wall time from the first request
  start to the last request end at that concurrency level.
- **Error rate**: failed HTTP/stream requests divided by all requests at that
  level. A request with no `finish_reason` is a failure.
- **Safe**: `TTFT p99 ≤ 5000 ms`, `TPOT p99 ≤ 150 ms`, `error_rate ≤ 5 %`.
  These are the defaults in `configs/lab.json`; change them before comparing
  runs, not after.
- **Cliff**: the first concurrency level that is unsafe, or whose TTFT p99 is
  more than 5× the N=1 baseline, whichever comes first.
- **Eviction signature**: preemption counter increases, cache usage near 1.0,
  and a non-zero waiting queue at the same level.

## Experiment 1 — KV cache strategy (`exp_kv`)

### Question

At a fixed 24 GB and a fixed 0.9 watermark, how much concurrency does each KV
strategy buy at long context, and what does it cost per token? Where is the
break-even batch size between "more resident sessions" and "more expensive
decode"?

### Workload

Each concurrency level uses a unique cohort salt in the system prompt, so the
level starts with a cold prefix cache. This is intentional: it exposes the cliff
instead of hiding it behind a warm cache. Steady-state hit rates will be higher
than the level averages in the report.

Single-turn, long-context requests. Each session sends a system prompt sized to
~90 % of the target context (8K / 16K / 22.7K), then generates up to
`max_tokens` (128–256 depending on profile). Before the sessions start, one
excluded request warms the shared system prefix so APC arms measure warm-prefix
capacity rather than a thundering herd of identical cold prefills. The warm
request's cached blocks stay resident, which is the realistic agent-service
state. APC-off arms pay the full prefill for every session. Multi-turn growth is
the job of experiment 2.

At each (context, concurrency) point we record:

- TTFT and TPOT percentiles
- peak KV cache occupancy
- prefix-cache hit rate
- actual prompt tokens
- output throughput
- preemption delta

### Decode break-even probe

A cheap second phase on the same engine: short prompt (512 tokens), 256 output
tokens, concurrency 1 and 8. If FP8 KV is slower at batch 1 but faster or equal
at batch 8, the compressed path has a real break-even; if it is slower at both,
the extra capacity must be justified by concurrency alone. This is the L4-specific
part of the experiment: Ada has limited memory bandwidth, and decode is
bandwidth-bound at small batches.

### Expected outputs

For each arm and context:

- `max_safe_concurrency`
- `max_safe_ttft_p99_ms`
- `peak_cache_usage`
- `mean_cache_hit_rate`
- `actual_prompt_tokens`

Plus a decode table per arm (`N1`, `N8`: TPOT, TTFT, output tok/s) and charts:

- TTFT p99 vs concurrency per context
- output throughput vs concurrency per context
- max safe concurrency vs context
- decode TPOT p50 at N=1 vs N=8
- prefix hit rate vs concurrency

### What would support the hypothesis

- FP8 KV roughly doubles the number of safe sessions at the same context, and
  the gain grows with context length.
- APC-on has a much higher hit rate than APC-off, and the capacity gap between
  them is largest when the system prefix is a large fraction of the prompt.
- FP8 TPOT is worse at N=1 but converges or wins by N=8. If not, the "compression
  buys concurrency at a decode cost" story is incomplete on L4.

### Tiering scope

The included tiering arm is SGLang HiCache (GPU → host memory). An SSD tier (the
`llama-tierkv` direction) needs either SGLang's storage backend or an LMCache /
Proxima-style connector built into a derived image. Add it as another arm with
the same level plan; the harness does not need changes. The report will then
compare GPU-only, host-tier, and storage-tier capacity and per-token cost on the
same L4 workload.

### Reference point, not a reproduction

Proxima/STAR-KV reports, on L4 24 GB, 8K-context concurrency 1 → 4, 16K from
startup failure to 2 concurrent sessions, and 3.11× KV block capacity, with a
decode penalty at batch 1 that reverses by batch 8–16. Those numbers are useful
as a shape to look for, but this harness uses native engine switches, a different
model, and a different block size. Do not expect identical values; compare the
direction, the break-even batch, and the cost curve.

### What this experiment does not do

- No output-quality evaluation. FP8 KV changes numerics; a correctness/quality
  pass is a separate study.
- No out-of-the-box out-of-tree KV connector (Proxima/STAR-KV, LMCache). Native
  switches only, plus SGLang HiCache as the host-tier native path.
- No multi-node, no TP. One L4, one process.

## Experiment 2 — Concurrency cliff (`exp_cliff`)

### Question

At ~22.7K agent context, where does the L4 fall off the cliff, and which knob
moves the cliff: more KV capacity (FP8), prefix reuse (APC), or admission control
(queue instead of evict)?

### Workload

Agent-shaped multi-turn sessions:

1. a tool-call request: "reply only with the lookup JSON";
2. a final-answer request containing the synthetic tool result (~600–1000 tokens).

Each turn adds the user question, the tool call, the tool result, and the answer
to the context. The system prompt is ~45 % of the target context, and the growth
per turn is sized so that the final turn reaches ~90 % of the target. When the
budget is exceeded, the builder drops the oldest complete turns, never the system
prompt.

This is deliberately **not** a native tool-parser test. The tool call is text and
the result is injected, which makes the workload deterministic and engine-neutral.
The cost being measured is the extra HTTP round trip plus the growing context,
not the parser.

### Arms

| arm | mechanism | expected effect |
|---|---|---|
| `vllm-baseline` | default admission | reference cliff |
| `vllm-cap12` | `--max-num-seqs 12` | queue instead of evict; lower throughput, higher TTFT, no preemption storm |
| `vllm-fp8` | FP8 KV | shifts cliff right if capacity-bound |
| `vllm-noapc` | APC off | shows how much of the cliff is prefix-eviction driven |
| `sglang-baseline` | RadixAttention | cross-engine reference |
| `sglang-fp8` | FP8 KV + Radix | cross-engine capacity test |
| `sglang-cap12` | `--max-running-requests 12` | SGLang admission control |

### Level plan

- smoke: 8K context, 3 turns, N = 1, 2, 4
- quick: 16K context, 6 turns, N = 1, 2, 4, 8, 12, 16
- full: 22.7K context, 12 turns, N = 1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48

The cap defaults to 12. In the quick profile, the cap arm sees queueing at N=16;
in the full profile it sees queueing from N=14 onward. That is the intended
contrast: a queue is not free, but it may avoid a cascade of evictions and
re-prefills.

### Expected outputs

- `cliff_concurrency`, `max_safe_concurrency`, throughput peak, throughput
  collapse point per arm
- TTFT p50/p95/p99 curves
- agent phase breakdown: tool-call request, final-answer request, round-trip e2e
- cache occupancy and queue depth at the cliff
- preemption deltas and APC hit rate as evidence for the mechanism

### Interpretation guide

- **Cliff with preemptions and cache ≈ 1.0**: capacity-bound. FP8 or a larger
  card helps; a cap only converts eviction into waiting.
- **Cliff with queue depth but no preemptions**: admission-bound. The cap is
  working; the question is whether throughput is stable enough to justify the
  TTFT.
- **Cliff without either**: look at CPU, host memory, or client-side connection
  limits. The guard records host load and memory next to the engine metrics.
- **APC-off cliff earlier than APC-on**: prefix recomputation is consuming the
  capacity; the shared system prompt is the lever.

## Threats to validity

1. **Version drift.** `latest` images change defaults (chunked prefill, scheduler,
   metrics names). Digests are recorded; for a rerun, pin tags.
2. **Tokenizer mismatch.** Planned context uses a character heuristic. The
   report's actual prompt tokens are the source of truth.
3. **Cross-engine comparison.** PagedAttention + APC and RadixAttention have
   different block granularity and eviction. Within-engine deltas are stronger
   than between-engine deltas.
4. **Synthetic agent shape.** Real tools return variable-size results and may
   fail; this workload is deterministic.
5. **Thermal and background load.** The guard aborts on temperature and host
   pressure, and the cooldown is recorded, but a shared host is never perfectly
   quiet. Compare runs taken at similar times, and check `gpu.jsonl` clocks.
6. **FP8 kernel support.** CLI acceptance does not prove the kernel is fast or
   even correct on sm89. A failed or anomalously slow arm is a result, not a bug
   to hide.
7. **SLO choice.** The safe region is only as meaningful as the SLO. State the
   SLO next to every chart; do not compare a 5 s run with a 2 s run.

## Adding a new KV strategy

An arm is just an engine argument list in `exp_kv/arms.py` (or
`exp_cliff/arms.py`). Add the flags, run `--dry-run`, then a smoke profile. If the
image does not list a flag, the harness skips the arm with the missing flags in
`arm_summary.json` rather than guessing. For out-of-tree connectors, build a
derived image and set `VLLM_IMAGE` / `SGLANG_IMAGE`; the rest of the harness is
unchanged.
