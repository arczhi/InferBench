from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.client import build_payload, chat_nonstream, chat_stream
from common.config import Config
from common.engine import DockerEngine, looks_like_help
from common.guard import Guard
from common.plot import line_chart_svg
from common.report import write_reports
from common.runner import Arm, ExperimentRunner, LevelPlan, cliff_analysis
from common.stats import percentile, summarize
from common.workload import build_session, estimate_messages_tokens
from tests.mock_server import make_server


class ClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.state, self.url = make_server()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def test_stream_usage_and_ttft(self) -> None:
        session = build_session(1, 0, "t", "chat", 200, 1, 0)
        payload = build_payload("exp-model", session.build_messages(0, "chat", [], 4000), 8)
        result = chat_stream(self.url, payload, timeout_s=30)
        self.assertTrue(result.ok, result.error)
        self.assertIsNotNone(result.ttft_ms)
        self.assertGreater(result.ttft_ms or 0, 0)
        self.assertEqual(result.completion_tokens, 6)
        self.assertIsNotNone(result.cached_tokens)
        self.assertGreater(len(result.itl_ms), 0)
        self.assertEqual(result.finish_reason, "length")

    def test_stream_options_retry(self) -> None:
        self.state.reject_stream_options = True
        session = build_session(1, 0, "t", "chat", 200, 1, 0)
        payload = build_payload("exp-model", session.build_messages(0, "chat", [], 4000), 8)
        result = chat_stream(self.url, payload, timeout_s=30)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.http_attempts, 2)
        self.assertTrue(result.usage_estimated)

    def test_nonstream(self) -> None:
        session = build_session(1, 0, "t", "chat", 100, 1, 0)
        payload = build_payload("exp-model", session.build_messages(0, "chat", [], 4000), 8, stream=False)
        result = chat_nonstream(self.url, payload, timeout_s=30)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.completion_tokens, 6)


class WorkloadTests(unittest.TestCase):
    def test_deterministic_and_salt_sensitive(self) -> None:
        a = build_session(4, 1, "salt-a", "agent", 500, 3, 120)
        b = build_session(4, 1, "salt-a", "agent", 500, 3, 120)
        c = build_session(4, 1, "salt-b", "agent", 500, 3, 120)
        self.assertEqual(a.system_prompt, b.system_prompt)
        self.assertEqual([t.user_text for t in a.turns], [t.user_text for t in b.turns])
        self.assertNotEqual(a.system_prompt, c.system_prompt)

    def test_budget_drops_old_turns(self) -> None:
        session = build_session(8, 2, "budget", "agent", 1000, 12, 500)
        history = []
        for index, turn in enumerate(session.turns[:11]):
            history.append(
                [
                    {"role": "user", "content": turn.user_text},
                    {"role": "assistant", "content": "ok"},
                ]
            )
        messages = session.build_messages(11, "final", history, budget_tokens=1800)
        self.assertLessEqual(estimate_messages_tokens(messages), 1850)
        self.assertEqual(messages[0]["role"], "system")

    def test_agent_phases(self) -> None:
        session = build_session(2, 0, "phase", "agent", 300, 2, 100)
        tool_messages = session.build_messages(0, "tool", [], 4000)
        final_messages = session.build_messages(0, "final", [], 4000, tool_call_text='{"tool":"lookup"}')
        self.assertEqual(len(tool_messages), 3)
        self.assertEqual(len(final_messages), 5)
        self.assertEqual(final_messages[-2]["role"], "assistant")


class StatsTests(unittest.TestCase):
    def test_percentile(self) -> None:
        values = [1, 2, 3, 4, 5]
        self.assertEqual(percentile(values, 50), 3)
        self.assertEqual(percentile(values, 0), 1)
        self.assertEqual(percentile(values, 100), 5)
        summary = summarize(values)
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["p99"], 4.96)


class GuardTests(unittest.TestCase):
    class FakeSampler:
        def __init__(self, value):
            self.value = value

        def snapshot(self):
            return dict(self.value)

    def _guard(self, gpu=None, host=None):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        gpu = gpu or {"temp_c": 50, "mem_used_pct": 50}
        host = host or {"mem_avail_mib": 8000, "load1": 1.0, "disk_free_gb": 100}
        guard = Guard(
            Config(),
            tmp,
            self.FakeSampler(gpu),
            self.FakeSampler(host),
            lambda: {"status": "running", "oom_killed": False},
            time.monotonic(),
        )
        self.addCleanup(guard._writer.close)
        return guard

    def test_temperature_trip(self) -> None:
        guard = self._guard(gpu={"temp_c": 90, "mem_used_pct": 50})
        guard._check()
        self.assertFalse(guard.abort_event.is_set())
        guard._check()
        self.assertTrue(guard.abort_event.is_set())
        self.assertEqual(guard.reason, "gpu_temperature")

    def test_host_memory_trip(self) -> None:
        guard = self._guard(host={"mem_avail_mib": 500, "load1": 1.0, "disk_free_gb": 100})
        for _ in range(3):
            guard._check()
        self.assertTrue(guard.abort_event.is_set())
        self.assertEqual(guard.reason, "host_memory_low")

    def test_temperature_counter_resets(self) -> None:
        guard = self._guard(gpu={"temp_c": 90, "mem_used_pct": 50})
        guard._check()
        guard.gpu = self.FakeSampler({"temp_c": 50, "mem_used_pct": 50})
        guard._check()
        guard.gpu = self.FakeSampler({"temp_c": 90, "mem_used_pct": 50})
        guard._check()
        self.assertFalse(guard.abort_event.is_set())
        guard._check()
        self.assertTrue(guard.abort_event.is_set())

    def test_error_rate_trip(self) -> None:
        guard = self._guard()
        for _ in range(20):
            guard.note_request(False)
        guard._check()
        self.assertTrue(guard.abort_event.is_set())
        self.assertEqual(guard.reason, "request_error_rate")

    def test_wall_time_trip(self) -> None:
        guard = self._guard()
        guard.run_started = time.monotonic() - 100
        guard.thresholds.max_wall_s = 10
        guard._check()
        self.assertTrue(guard.abort_event.is_set())
        self.assertEqual(guard.reason, "wall_time_exceeded")


class HelpParsingTests(unittest.TestCase):
    def test_looks_like_help(self) -> None:
        self.assertTrue(looks_like_help("usage: api_server.py [-h] " + "x" * 300))
        self.assertFalse(looks_like_help("Traceback: something failed"))


class CommandTests(unittest.TestCase):
    def test_vllm_command(self) -> None:
        cfg = Config()
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        engine = DockerEngine(cfg, "vllm", "arm", "run", tmp, ["--kv-cache-dtype", "fp8"])
        cmd = engine.command()
        self.assertIn("--gpu-memory-utilization", cmd)
        self.assertEqual(cmd[cmd.index("--gpu-memory-utilization") + 1], "0.9")
        docker = engine.docker_run_cmd()
        self.assertIn("127.0.0.1:8000:8000", docker)
        self.assertIn("--memory", docker)
        self.assertIn("inference-exp-vllm", docker)

    def test_flag_fallback(self) -> None:
        cfg = Config()
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        engine = DockerEngine(cfg, "vllm", "arm", "run", tmp, ["--max-num-seqs", "12"])
        self.assertIn("--served-model-name", engine.command())
        remaining = engine.apply_flag_fallbacks(["--served-model-name", "--max-num-seqs"])
        self.assertEqual(remaining, ["--max-num-seqs"])
        self.assertNotIn("--served-model-name", engine.command())

    def test_local_model_mount(self) -> None:
        cfg = Config()
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as fh:
            fh.write("{}")
        cfg.model = tmp
        engine = DockerEngine(cfg, "vllm", "arm", "run", tmp, [])
        cmd = engine.docker_run_cmd()
        self.assertIn(f"{tmp}:{tmp}:ro", cmd)
        self.assertIn(tmp, engine.command())

    def test_sglang_command(self) -> None:
        cfg = Config()
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        engine = DockerEngine(cfg, "sglang", "arm", "run", tmp, ["--max-running-requests", "12"])
        cmd = engine.command()
        self.assertIn("--mem-fraction-static", cmd)
        self.assertEqual(cmd[cmd.index("--mem-fraction-static") + 1], "0.9")
        self.assertIn("--max-running-requests", cmd)


class RunnerLevelTests(unittest.TestCase):
    def test_run_level_agent(self) -> None:
        server, state, url = make_server()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        cfg = Config()
        cfg.engine.max_model_len = 4096
        cfg.workload.max_tokens = 8
        runner = ExperimentRunner(cfg, "unit", [], [], out_root=tmp, level_timeout_s=30)
        engine = SimpleNamespace(base_url=url, run_dir=tmp, metrics=lambda: {})
        plan = LevelPlan(
            concurrency=2,
            turns=2,
            system_tokens=200,
            tool_result_tokens=100,
            max_tokens=8,
            mode="agent",
            label="unit",
            context_hint=1000,
        )
        guard = Guard(
            cfg,
            tmp,
            GuardTests.FakeSampler({}),
            GuardTests.FakeSampler({}),
            lambda: {"status": "running", "oom_killed": False},
            time.monotonic(),
        )
        self.addCleanup(guard._writer.close)
        row = runner._run_level(engine, Arm("a", "vllm"), plan, 0, guard)
        self.assertEqual(row["requests"], 8)
        self.assertEqual(row["requests_ok"], 8)
        self.assertEqual(row["phases"]["agent_roundtrip_e2e_ms"]["count"], 4)
        self.assertTrue(row["safe"])
        path = os.path.join(tmp, "requests.jsonl")
        with open(path, "r", encoding="utf-8") as fh:
            self.assertEqual(len(fh.readlines()), 8)


class CliffTests(unittest.TestCase):
    def test_cliff_detection(self) -> None:
        rows = [
            {
                "concurrency": 1,
                "ttft_ms": {"p99": 100},
                "tpot_ms": {"p99": 10},
                "error_rate": 0,
                "engine": {},
                "output_tok_s": 100,
                "safe": True,
            },
            {
                "concurrency": 8,
                "ttft_ms": {"p99": 20000},
                "tpot_ms": {"p99": 10},
                "error_rate": 0,
                "engine": {"waiting_peak": 5, "cache_usage_peak": 0.99, "preemptions_delta": 3},
                "output_tok_s": 50,
                "safe": False,
            },
        ]
        result = cliff_analysis(rows, Config().slo)
        self.assertEqual(result["cliff_concurrency"], 8)
        self.assertEqual(result["max_safe_concurrency"], 1)
        self.assertEqual(result["throughput_peak"]["concurrency"], 1)


class ReportTests(unittest.TestCase):
    def test_report_files(self) -> None:
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        summary = {
            "run_id": "unit",
            "started_at": "a",
            "finished_at": "b",
            "duration_s": 1,
            "aborted": False,
            "config": {},
            "arms": [
                {
                    "name": "vllm-baseline",
                    "engine": "vllm",
                    "status": "completed",
                    "levels": [
                        {
                            "concurrency": 1,
                            "requests": 2,
                            "requests_ok": 2,
                            "error_rate": 0,
                            "ttft_ms": {"p50": 100, "p99": 200},
                            "tpot_ms": {"p99": 10},
                            "output_tok_s": 50,
                            "prefill_tok_s": 1000,
                            "cache_hit_rate_usage": 0.8,
                            "engine": {"cache_usage_peak": 0.5, "waiting_peak": 0, "preemptions_delta": 0},
                            "safe": True,
                        }
                    ],
                    "cliff": {"max_safe_concurrency": 1, "cliff_concurrency": None},
                }
            ],
        }
        svg = line_chart_svg("t", [("a", [(1, 100)])], "ms")
        paths = write_reports(tmp, summary, [("chart", svg)], "Unit report")
        self.assertTrue(os.path.exists(paths["html"]))
        self.assertTrue(os.path.exists(paths["md"]))
        self.assertTrue(os.path.exists(paths["csv"]))
        with open(paths["html"], "r", encoding="utf-8") as fh:
            body = fh.read()
        self.assertIn("Unit report", body)
        self.assertIn("<svg", body)


class SurveyTests(unittest.TestCase):
    def test_survey_runs_on_any_platform(self) -> None:
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        out = os.path.join(tmp, "survey.json")
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "survey_l4.py"), "--no-network", "--json-out", out],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(out, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["schema"], "inference-exp.survey/v1")
        self.assertIn("recommendation", data)


if __name__ == "__main__":
    unittest.main()


class RunnerExternalTests(unittest.TestCase):
    def test_external_engine_end_to_end(self) -> None:
        server, _state, url = make_server()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        cfg = Config()
        cfg.engine.max_model_len = 4096
        cfg.guard.max_wall_s = 60
        cfg.guard.cooldown_max_s = 1
        cfg.slo.ttft_p99_ms = 10000
        arms = [Arm("mock-arm", "vllm", [])]
        levels = [
            LevelPlan(1, 1, 200, 0, 8, "chat", "unit", 512),
            LevelPlan(2, 1, 200, 0, 8, "chat", "unit", 512),
        ]
        runner = ExperimentRunner(
            cfg,
            "unit-external",
            arms,
            levels,
            out_root=tmp,
            external_url=url,
            level_timeout_s=30,
        )
        summary = runner.run()
        self.assertFalse(summary["aborted"])
        self.assertEqual(summary["arms"][0]["status"], "completed")
        self.assertEqual(len(summary["arms"][0]["levels"]), 2)
        self.assertTrue(os.path.exists(os.path.join(runner.run_dir, "mock-arm", "requests.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(runner.run_dir, "run_meta.json")))
        self.assertTrue(os.path.exists(os.path.join(runner.run_dir, "summary.json")))
