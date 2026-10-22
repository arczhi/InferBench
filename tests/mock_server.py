from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


class MockState:
    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.reject_stream_options = False
        self.delay_s = 0.012
        self.tokens_per_response = 6
        self.lock = threading.Lock()


def make_server(state: Optional[MockState] = None) -> tuple[ThreadingHTTPServer, MockState, str]:
    state = state or MockState()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            pass

        def _json(self, status: int, payload: Dict[str, Any]) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": "exp-model", "object": "model"}]})
            elif self.path == "/version":
                self.send_response(200)
                body = b"0.0.0-mock"
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/metrics":
                body = (
                    "# HELP vllm:gpu_cache_usage_perc cache usage\n"
                    "# TYPE vllm:gpu_cache_usage_perc gauge\n"
                    "vllm:gpu_cache_usage_perc 0.42\n"
                    "vllm:num_requests_running 2\n"
                    "vllm:num_requests_waiting 0\n"
                    "vllm:num_preemptions_total 0\n"
                    "vllm:prefix_cache_queries_total 1000\n"
                    "vllm:prefix_cache_hits_total 800\n"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            with state.lock:
                state.requests.append(payload)
            if state.reject_stream_options and "stream_options" in payload:
                self._json(400, {"error": "stream_options is not supported"})
                return
            prompt_chars = 0
            for message in payload.get("messages") or []:
                content = message.get("content")
                if isinstance(content, str):
                    prompt_chars += len(content)
            prompt_tokens = max(1, prompt_chars // 4)
            cached = int(prompt_tokens * 0.8)
            max_tokens = int(payload.get("max_tokens") or state.tokens_per_response)
            count = min(state.tokens_per_response, max_tokens)
            if not payload.get("stream"):
                self._json(
                    200,
                    {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion",
                        "model": payload.get("model", "exp-model"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "mock answer"},
                                "finish_reason": "length",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": count,
                            "total_tokens": prompt_tokens + count,
                            "prompt_tokens_details": {"cached_tokens": cached},
                        },
                    },
                )
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            time.sleep(state.delay_s)
            for index in range(count):
                chunk = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "model": payload.get("model", "exp-model"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"t{index} "},
                            "finish_reason": "length" if index == count - 1 else None,
                        }
                    ],
                }
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
                time.sleep(0.004)
            if (payload.get("stream_options") or {}).get("include_usage"):
                usage = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "model": payload.get("model", "exp-model"),
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": count,
                        "total_tokens": prompt_tokens + count,
                        "prompt_tokens_details": {"cached_tokens": cached},
                    },
                }
                self.wfile.write(("data: " + json.dumps(usage) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    return server, state, url
