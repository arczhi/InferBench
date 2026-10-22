from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass
class ChatResult:
    ok: bool = False
    status: Optional[int] = None
    error: Optional[str] = None
    ttft_event_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    e2e_ms: Optional[float] = None
    itl_ms: List[float] = field(default_factory=list)
    text: str = ""
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    usage_estimated: bool = False
    chunks: int = 0
    http_attempts: int = 1

    def to_record(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "error": self.error,
            "ttft_event_ms": self.ttft_event_ms,
            "ttft_ms": self.ttft_ms,
            "e2e_ms": self.e2e_ms,
            "itl_ms": self.itl_ms,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "usage_estimated": self.usage_estimated,
            "chunks": self.chunks,
            "http_attempts": self.http_attempts,
            "text_chars": len(self.text),
        }


def build_payload(
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    stream: bool = True,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _parse_usage(obj: Dict[str, Any]) -> Dict[str, Optional[int]]:
    usage = obj.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        cached = usage.get("prompt_cache_hit_tokens")
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": cached,
    }


def _content_delta(delta: Dict[str, Any]) -> str:
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
        return "".join(out)
    return ""


def chat_stream(
    base_url: str,
    payload: Dict[str, Any],
    timeout_s: float = 900.0,
    retry_without_usage: bool = True,
) -> ChatResult:
    result = ChatResult()
    body = json.dumps(payload).encode("utf-8")
    url = base_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.monotonic()
    try:
        resp = _OPENER.open(req, timeout=timeout_s)
    except urllib.error.HTTPError as exc:
        text = ""
        try:
            text = exc.read().decode("utf-8", "replace")[:1500]
        except Exception:  # noqa: BLE001
            pass
        result.status = exc.code
        if (
            retry_without_usage
            and exc.code == 400
            and "stream_options" in (text or "").lower()
            and "stream_options" in payload
        ):
            retry_payload = dict(payload)
            retry_payload.pop("stream_options", None)
            result = chat_stream(base_url, retry_payload, timeout_s, retry_without_usage=False)
            result.http_attempts = 2
            return result
        result.error = f"http {exc.code}: {text or exc.reason}"
        return result
    except Exception as exc:  # noqa: BLE001
        result.error = f"connect: {exc}"
        return result

    first_event_at: Optional[float] = None
    first_content_at: Optional[float] = None
    last_content_at: Optional[float] = None
    text_parts: List[str] = []
    tool_seen = False
    try:
        with resp:
            for raw in resp:
                if not raw:
                    continue
                line = raw.decode("utf-8", "replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                now = time.monotonic()
                if first_event_at is None:
                    first_event_at = now
                if obj.get("usage"):
                    parsed = _parse_usage(obj)
                    result.prompt_tokens = parsed["prompt_tokens"]
                    result.completion_tokens = parsed["completion_tokens"]
                    result.cached_tokens = parsed["cached_tokens"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                piece = _content_delta(delta)
                has_tool = bool(delta.get("tool_calls"))
                if (piece or has_tool) and first_content_at is None:
                    first_content_at = now
                if piece:
                    if last_content_at is not None:
                        result.itl_ms.append(round(1000 * (now - last_content_at), 2))
                    last_content_at = now
                    text_parts.append(piece)
                if has_tool:
                    tool_seen = True
                    if last_content_at is not None:
                        result.itl_ms.append(round(1000 * (now - last_content_at), 2))
                    last_content_at = now
                if piece or has_tool:
                    result.chunks += 1
                if choice.get("finish_reason"):
                    result.finish_reason = choice.get("finish_reason")
    except Exception as exc:  # noqa: BLE001
        result.error = f"stream: {exc}"
        result.e2e_ms = round(1000 * (time.monotonic() - started), 2)
        result.text = "".join(text_parts)
        return result

    ended = time.monotonic()
    result.e2e_ms = round(1000 * (ended - started), 2)
    result.text = "".join(text_parts)
    if first_event_at is not None:
        result.ttft_event_ms = round(1000 * (first_event_at - started), 2)
    if first_content_at is not None:
        result.ttft_ms = round(1000 * (first_content_at - started), 2)
    elif first_event_at is not None:
        result.ttft_ms = result.ttft_event_ms
    if result.completion_tokens is None:
        result.completion_tokens = max(result.chunks, len(result.itl_ms) + (1 if result.chunks else 0))
        result.usage_estimated = True
    result.ok = (result.error is None) and bool(result.finish_reason)
    if result.error is None and not result.finish_reason:
        result.error = "no finish_reason (stream truncated?)"
        result.ok = False
    if tool_seen and not result.text:
        result.text = "[tool_call]"
    return result


def chat_nonstream(base_url: str, payload: Dict[str, Any], timeout_s: float = 900.0) -> ChatResult:
    result = ChatResult()
    payload = dict(payload)
    payload["stream"] = False
    payload.pop("stream_options", None)
    url = base_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with _OPENER.open(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        text = ""
        try:
            text = exc.read().decode("utf-8", "replace")[:1500]
        except Exception:  # noqa: BLE001
            pass
        result.status = exc.code
        result.error = f"http {exc.code}: {text or exc.reason}"
        return result
    except Exception as exc:  # noqa: BLE001
        result.error = f"connect: {exc}"
        return result
    result.status = status
    result.e2e_ms = round(1000 * (time.monotonic() - started), 2)
    result.ttft_ms = result.e2e_ms
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        result.error = f"bad json: {raw[:300]}"
        return result
    parsed = _parse_usage(obj)
    result.prompt_tokens = parsed["prompt_tokens"]
    result.completion_tokens = parsed["completion_tokens"]
    result.cached_tokens = parsed["cached_tokens"]
    choices = obj.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        result.text = _content_delta(msg)
        result.finish_reason = choices[0].get("finish_reason")
    result.ok = bool(obj.get("choices")) and result.error is None
    if not result.ok and result.error is None:
        result.error = "no choices"
    return result
