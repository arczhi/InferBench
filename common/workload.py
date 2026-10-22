from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

CHARS_PER_TOKEN = 3.6

FILLER = [
    "The service keeps a per-tenant ledger of retrieval events, and each event records the tenant, the query class, the shard identifier, and the wall-clock second at which the event was admitted. Operators use the ledger to reconstruct why a particular request was routed to a cold replica instead of a warm one.",
    "Capacity reviews distinguish between resident state that can be reconstructed cheaply and resident state that cannot. Reconstructing a prefix costs a full prefill pass, while reconstructing a routing decision costs only a dictionary lookup. The distinction matters when the eviction policy has to choose a victim under pressure.",
    "A scheduler should expose its queue depth, its running batch size, and the fraction of the cache that is currently pinned. Without those three signals, an operator cannot tell whether latency is caused by admission, by computation, or by a cache miss that forced a recomputation.",
    "When a request is preempted, the engine may swap its state to host memory or discard it and recompute later. Recompute avoids host bandwidth but spends GPU cycles again. The correct choice depends on the ratio of prefill throughput to decode throughput for the model and sequence length in question.",
    "Prefix caching works best when many requests share a long, stable preamble. Agent workloads often share a tool description, a policy block, and a system prompt, so the shared fraction can be large even when the final user question differs. Measuring the shared fraction is a prerequisite for predicting capacity.",
    "A safe operating envelope is not a single number. It is a region bounded by a latency percentile, an error budget, a thermal limit, and a host-memory floor. The region shrinks when the machine has background work, because the background work competes for the same memory bus and the same page cache.",
    "Decode is memory-bandwidth bound for small batches. Each generated token requires reading the model weights and the KV cache, so quantization can help only when it reduces bytes moved per token without adding expensive dequantization work. Kernel launch overhead and poor vectorization can erase the theoretical gain.",
    "Prefill is compute bound for long prompts. Chunked prefill interleaves prefill chunks with decode steps so that a long prompt does not monopolize the device and inflate inter-token latency for other requests. The trade-off is a longer time to first token for the chunked request.",
    "Admission control is often more effective than eviction. If the scheduler admits fewer sessions than the cache can hold, requests wait in a queue but retain their state. If the scheduler admits too many, it evicts state and later pays for recomputation, which can be far more expensive than waiting.",
    "Telemetry should be sampled at a fixed cadence and stored with monotonic timestamps. Averages hide cliffs. The useful quantities are tail percentiles, cache occupancy, queue depth, and the rate of preemptions per completed request, all aligned on the same clock.",
    "A reproducible experiment pins the model revision, the container digest, the random seed, the prompt construction, and the measurement definitions. Otherwise a difference in time to first token may come from a changed chat template rather than from the mechanism under study.",
    "Host memory is part of the inference system even when the GPU has enough capacity. Model loading, tokenizer work, metrics aggregation, and log buffering all consume host pages. A container limit that is too tight can kill a run during startup, while a limit that is too loose can starve co-located web services.",
    "Thermal headroom matters on a shared workstation. A 24GB Ada card can sustain a long prefill burst, but sustained load raises clocks and fan behavior. A watchdog that stops the engine before the card reaches its throttle point protects both the experiment and the other users of the machine.",
    "The interesting comparison is not compression versus no compression. It is the break-even batch size: the point at which the extra concurrency unlocked by compression outweighs the per-token cost of the compressed path. Below that point, compression loses; above it, compression wins on aggregate throughput.",
    "Tool-using agents create a distinctive trace shape. A short planning request is followed by a tool call, then by a much larger observation, then by another short request. The observation dominates the prompt and is highly repetitive across sessions, which makes it a good candidate for prefix reuse.",
    "A routing layer can protect a large executor from overload. The router classifies an incoming task with a small model and sends only the tasks that need the executor. The router must be cheap enough that its own latency does not dominate, and its memory footprint must be reserved before the executor claims the device.",
]


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def estimate_messages_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += 4
        content = message.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += estimate_tokens(str(part.get("text", "")))
        if message.get("tool_calls"):
            total += estimate_tokens(json.dumps(message["tool_calls"], separators=(",", ":")))
    return total


def build_system_prompt(target_tokens: int, seed: int, salt: str, agent: bool) -> str:
    target_chars = max(2000, int(target_tokens * CHARS_PER_TOKEN))
    rng = random.Random(seed)
    header = [
        "You are an operations assistant inside a capacity-planning lab.",
        f"Cohort: {salt}.",
        "Follow the operator's instruction exactly and keep answers short.",
        "The reference material below is noisy but authoritative for this session.",
    ]
    if agent:
        header.append(
            "When asked to look up a value, reply with one JSON object of the form "
            '{"tool":"lookup","args":{"metric":"...","window":"..."}} and nothing else.'
        )
    chunks: List[str] = ["\n".join(header), ""]
    used = len(chunks[0]) + 2
    index = 1
    while used < target_chars:
        base = FILLER[rng.randrange(len(FILLER))]
        block = f"[section {index:04d}] {base} Note {rng.randrange(10**6):06d}."
        chunks.append(block)
        used += len(block) + 2
        index += 1
        if index > 4000:
            break
    return "\n".join(chunks)


def build_tool_result(target_tokens: int, turn: int, seed: int) -> str:
    rng = random.Random(seed * 1000 + turn)
    target_chars = max(400, int(target_tokens * CHARS_PER_TOKEN))
    records: List[Dict[str, Any]] = []
    used = 0
    while used < target_chars and len(records) < 500:
        record = {
            "id": f"rec-{rng.randrange(10**9):09d}",
            "shard": rng.randrange(64),
            "metric": f"latency.p{rng.choice([50, 90, 95, 99])}",
            "value_ms": round(rng.uniform(1.0, 900.0), 3),
            "window_s": rng.choice([60, 300, 900]),
            "admitted": rng.randrange(10**6),
            "recomputed": rng.randrange(0, 5000),
        }
        text = json.dumps(record, separators=(",", ":"))
        records.append(record)
        used += len(text) + 2
    payload = {
        "tool": "lookup",
        "status": "ok",
        "turn": turn,
        "nonce": f"{rng.randrange(10**12):012d}",
        "records": records,
    }
    return json.dumps(payload, separators=(",", ":"))


USER_TEMPLATES = [
    "Look up the p99 admission latency for tenant t-{tid} over the last {window} seconds, then summarize the trend in one sentence.",
    "Check the recompute count for shard {shard} in the current cohort. If it rose, say by how much in one sentence.",
    "Find the cache occupancy reported for tenant t-{tid} and tell me whether it is closer to the warning or the critical threshold.",
    "Look up the decode latency p50 for the {shard} shard group and state whether it is stable compared with the previous window.",
    "Query the eviction count for tenant t-{tid} over the last {window} seconds and report the single largest contributor.",
    "Check the tool-call success rate for cohort {cohort} and answer with the rate and one possible cause.",
]


@dataclass
class TurnSpec:
    user_text: str
    tool_result: str


@dataclass
class SessionSpec:
    session_id: int
    level: int
    seed: int
    salt: str
    system_prompt: str
    task: str
    turns: List[TurnSpec] = field(default_factory=list)
    mode: str = "agent"

    def _base(self) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.task},
        ]

    def build_messages(
        self,
        turn_index: int,
        phase: str,
        history: Sequence[Sequence[Dict[str, Any]]],
        budget_tokens: int,
        tool_call_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Assemble messages for one request, dropping oldest turns if needed.

        `history` is a list of completed turn blocks, each a list of messages.
        Only whole turns are dropped so the conversation stays well formed.
        """
        current: List[Dict[str, Any]] = []
        turn = self.turns[turn_index]
        if self.mode == "agent" and phase == "tool":
            current.append(
                {
                    "role": "user",
                    "content": turn.user_text
                    + " Reply only with the lookup JSON object; do not answer the question yet.",
                }
            )
        elif self.mode == "agent" and phase == "final":
            current.append({"role": "user", "content": turn.user_text})
            current.append(
                {
                    "role": "assistant",
                    "content": tool_call_text or self._placeholder_tool_call(turn_index),
                }
            )
            current.append(
                {
                    "role": "user",
                    "content": "<tool_result>\n" + turn.tool_result + "\n</tool_result>\n"
                    "Answer the original question in one short sentence using only this result.",
                }
            )
        else:
            current.append({"role": "user", "content": turn.user_text})

        kept: List[Sequence[Dict[str, Any]]] = list(history)
        while True:
            messages = self._base()
            for block in kept:
                messages.extend(block)
            messages.extend(current)
            if estimate_messages_tokens(messages) <= budget_tokens or not kept:
                break
            kept.pop(0)
        if estimate_messages_tokens(messages) > budget_tokens:
            messages = self._truncate_last(messages, budget_tokens)
        return messages

    def _placeholder_tool_call(self, turn_index: int) -> str:
        return json.dumps(
            {
                "tool": "lookup",
                "args": {
                    "metric": f"latency.p99.turn{turn_index}",
                    "window": "900",
                    "session": self.session_id,
                },
            },
            separators=(",", ":"),
        )

    def _truncate_last(self, messages: List[Dict[str, Any]], budget_tokens: int) -> List[Dict[str, Any]]:
        if not messages:
            return messages
        overflow = estimate_messages_tokens(messages) - budget_tokens
        if overflow <= 0:
            return messages
        last = dict(messages[-1])
        content = last.get("content")
        if isinstance(content, str) and content:
            keep_chars = max(400, len(content) - int(overflow * CHARS_PER_TOKEN))
            last["content"] = content[:keep_chars] + "\n[truncated]"
            messages = messages[:-1] + [last]
        return messages


def build_session(
    level: int,
    session_id: int,
    salt: str,
    mode: str,
    system_tokens: int,
    turns: int,
    tool_result_tokens: int,
    seed: int = 7,
) -> SessionSpec:
    session_seed = (seed * 1000003) ^ (level * 7919) ^ (session_id * 104729)
    rng = random.Random(session_seed)
    system_prompt = build_system_prompt(system_tokens, session_seed, salt, agent=(mode == "agent"))
    task = (
        f"You are session {session_id} in cohort {salt}. "
        "The operator will ask a sequence of operational questions. "
        "Keep every answer to one sentence unless asked for JSON."
    )
    spec = SessionSpec(
        session_id=session_id,
        level=level,
        seed=session_seed,
        salt=salt,
        system_prompt=system_prompt,
        task=task,
        mode=mode,
    )
    for turn_index in range(turns):
        template = USER_TEMPLATES[rng.randrange(len(USER_TEMPLATES))]
        user_text = template.format(
            tid=rng.randrange(1, 10**5),
            shard=rng.randrange(64),
            cohort=salt,
            window=rng.choice([60, 300, 900]),
        )
        user_text += f" Request tag: r-{rng.randrange(10**9):09d}."
        spec.turns.append(
            TurnSpec(
                user_text=user_text,
                tool_result=build_tool_result(tool_result_tokens, turn_index, session_seed),
            )
        )
    return spec


def build_history_block(
    turn: TurnSpec,
    turn_index: int,
    assistant_text: str,
    mode: str,
    tool_call_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Messages appended after a completed turn, used as context for later turns."""
    block: List[Dict[str, Any]] = [{"role": "user", "content": turn.user_text}]
    if mode == "agent":
        block.append(
            {
                "role": "assistant",
                "content": tool_call_text
                or json.dumps(
                    {
                        "tool": "lookup",
                        "args": {
                            "metric": f"latency.p99.turn{turn_index}",
                            "window": "900",
                        },
                    },
                    separators=(",", ":"),
                ),
            }
        )
        block.append(
            {
                "role": "user",
                "content": "<tool_result>\n" + turn.tool_result + "\n</tool_result>",
            }
        )
    block.append({"role": "assistant", "content": assistant_text or "(no answer)"})
    return block
