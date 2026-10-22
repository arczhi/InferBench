# Concurrency cliff experiment

Question: at ~22.7K agent context, where does a single L4 fall off the cliff, and
which knob moves it — more KV capacity, prefix reuse, or admission control?

```bash
python3 -m exp_cliff.run --profile smoke --engine vllm --dry-run
python3 -m exp_cliff.run --profile quick --engine all
python3 -m exp_cliff.run --profile full  --engine all

# focused sweep
python3 -m exp_cliff.run --profile quick --engine vllm \
  --arms vllm-baseline,vllm-cap12,vllm-fp8 \
  --context 22700 --turns 12 --levels 1,2,4,8,12,14,16
```

Profiles:

| profile | context | turns | concurrency | arms |
|---|---|---|---|---|
| smoke | 8192 | 3 | 1,2,4 | vLLM + SGLang baseline |
| quick | 16000 | 6 | 1,2,4,8,12,16 | baseline, cap, FP8, SGLang baseline/FP8 |
| full | 22700 | 12 | 1,2,4,6,8,10,12,14,16,20,24,32,48 | all including no-APC |

Agent workload: each turn sends a forced tool-call request, then a final-answer
request containing a synthetic tool result. The context grows turn by turn; the
builder drops the oldest complete turns if the budget is exceeded.

The report gives:

- `cliff_concurrency` and `max_safe_concurrency` per arm
- TTFT p50/p95/p99 curves and output throughput curves
- tool-call vs final-answer vs round-trip latency
- cache occupancy, queue depth and preemption deltas
- `comparison.json` with the cliff shift relative to `vllm-baseline`

A cap arm (`--max-num-seqs` / `--max-running-requests`) is expected to queue
rather than evict. Judge it by stable throughput and bounded p99, not by the
absolute TTFT at the highest concurrency.
