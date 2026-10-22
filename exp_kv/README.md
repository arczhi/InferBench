# KV cache strategy experiment

Question: at 24 GB and a 0.9 watermark, how many long-context sessions does each
KV strategy sustain, and what does the extra concurrency cost per decoded token?

```bash
python3 -m exp_kv.run --profile smoke --engine vllm --dry-run
python3 -m exp_kv.run --profile quick --engine all
python3 -m exp_kv.run --profile full  --engine vllm --include-hicache

# subset
python3 -m exp_kv.run --profile quick --engine vllm \
  --arms vllm-fp16-apc,vllm-fp8-apc \
  --contexts 8192,16384 --concurrency 1,2,4,8
```

Profiles:

| profile | contexts | concurrency | max tokens | decode probe |
|---|---|---|---|---|
| smoke | 8192 | 1,2 | 128 | no |
| quick | 8192,16384 | 1,2,4,8 | 192 | yes |
| full | 8192,16384,22700 | 1,2,4,6,8,12,16 | 256 | yes |

The workload is single-turn long context: the system prompt is ~90 % of the
target context, then a short question and up to `max_tokens` of output. All
sessions at a level start together. See `docs/EXPERIMENT_DESIGN.md` for the full
rationale.

Read the result as:

- `max_safe_concurrency` per context = the capacity claim.
- `cache_hit_rate_usage` = how much of the prompt was actually reused.
- `decode` / `decode_break_even` = whether FP8 KV pays for itself at batch 1 or
  only at batch 8.
- `peak_cache_usage` = whether the run was actually near the KV limit, or just
  limited by the SLO.

Note: FP8 KV support on sm89 depends on the engine build. The harness validates
CLI flags, not kernels; a failed arm is recorded, not hidden.
