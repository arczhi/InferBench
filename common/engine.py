from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from .config import Config
from .util import ROOT, atomic_write_json, ensure_dir, run_cmd, utc_now

VLLM_MODULE = "vllm.entrypoints.openai.api_server"
SGLANG_MODULE = "sglang.launch_server"
HELP_CACHE = os.path.join(ROOT, "runs", "_engine_help")


def looks_like_help(text: str) -> bool:
    lowered = text.lower()
    return "usage:" in lowered and len(text) > 200


class EngineError(RuntimeError):
    pass


class EngineStartError(EngineError):
    pass


def image_for(cfg: Config, kind: str) -> str:
    return cfg.docker.vllm_image if kind == "vllm" else cfg.docker.sglang_image


def module_for(kind: str) -> str:
    return VLLM_MODULE if kind == "vllm" else SGLANG_MODULE


def _port_free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        return sock.connect_ex(("127.0.0.1", port)) != 0
    finally:
        sock.close()


def _http(url: str, timeout: float = 5.0, method: str = "GET", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with opener.open(req, timeout=timeout) as resp:
            return {"ok": True, "status": resp.status, "body": resp.read().decode("utf-8", "replace")}
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:2000]
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "status": exc.code, "body": body, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": None, "body": "", "error": str(exc)}


def _parse_prometheus(text: str, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "_bucket{" in line or line.startswith("_bucket"):
            continue
        try:
            name_part, value_part = line.rsplit(" ", 1)
            value = float(value_part)
        except ValueError:
            continue
        name = name_part.split("{", 1)[0].strip()
        if prefix and not name.startswith(prefix):
            continue
        out[name] = out.get(name, 0.0) + value
    return out


class DockerEngine:
    def __init__(
        self,
        cfg: Config,
        kind: str,
        arm_name: str,
        run_id: str,
        run_dir: str,
        engine_args: Sequence[str],
    ) -> None:
        if kind not in ("vllm", "sglang"):
            raise ValueError(f"unknown engine kind: {kind}")
        self.cfg = cfg
        self.kind = kind
        self.arm_name = arm_name
        self.run_id = run_id
        self.run_dir = run_dir
        self.engine_args = list(engine_args)
        self.image = image_for(cfg, kind)
        self.port = cfg.engine.vllm_port if kind == "vllm" else cfg.engine.sglang_port
        self.container_name = f"inference-exp-{kind}"
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._started = False
        self._help_text: Optional[str] = None
        self.drop_served_model_name = False

    # ------------------------------------------------------------------ docker
    def _docker(self, args: Sequence[str], timeout: float = 60.0, check: bool = False) -> Dict[str, Any]:
        return run_cmd(["docker"] + list(args), timeout=timeout, check=check)

    def image_id(self) -> Optional[str]:
        res = self._docker(["image", "inspect", self.image, "--format", "{{.Id}}"], timeout=20)
        return res["out"] if res["ok"] else None

    def image_digest(self) -> Dict[str, Any]:
        res = self._docker(["image", "inspect", self.image], timeout=20)
        if not res["ok"]:
            return {"image": self.image, "error": res["err"]}
        try:
            data = json.loads(res["out"])[0]
            return {
                "image": self.image,
                "id": data.get("Id"),
                "repo_digests": data.get("RepoDigests"),
                "created": data.get("Created"),
                "size_bytes": data.get("Size"),
            }
        except (json.JSONDecodeError, IndexError, KeyError) as exc:
            return {"image": self.image, "error": str(exc)}

    def ensure_image(self) -> None:
        if self._docker(["image", "inspect", self.image], timeout=20)["ok"]:
            return
        raise EngineStartError(
            f"image {self.image} is not present. Pull it first:\n"
            f"  ./scripts/pull_images.sh"
        )

    def ensure_volume(self) -> None:
        name = self.cfg.docker.hf_cache_volume
        res = self._docker(["volume", "inspect", name], timeout=15)
        if not res["ok"]:
            created = self._docker(["volume", "create", name], timeout=15)
            if not created["ok"]:
                raise EngineStartError(f"cannot create volume {name}: {created['err']}")

    def _remove_own_container(self) -> None:
        res = self._docker(["container", "inspect", self.container_name], timeout=15)
        if not res["ok"]:
            return
        try:
            data = json.loads(res["out"])[0]
            labels = (data.get("Config") or {}).get("Labels") or {}
            state = (data.get("State") or {}).get("Status")
        except (json.JSONDecodeError, IndexError):
            raise EngineStartError(f"container {self.container_name} exists but cannot be inspected")
        if labels.get("inference-exp") != "1":
            raise EngineStartError(
                f"container {self.container_name} exists and is not owned by inference-exp; refusing to touch it"
            )
        print(f"[{utc_now()}] removing stale experiment container {self.container_name} ({state})")
        self._docker(["rm", "-f", self.container_name], timeout=60)

    # ------------------------------------------------------------------- flags
    def get_help_text(self) -> Optional[str]:
        if self._help_text is not None:
            return self._help_text
        ensure_dir(HELP_CACHE)
        image_id = self.image_id() or self.image.replace("/", "_").replace(":", "_")
        cache_path = os.path.join(HELP_CACHE, f"{self.kind}-{image_id[:24]}.txt")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8", errors="replace") as fh:
                self._help_text = fh.read()
            return self._help_text
        if not self._docker(["image", "inspect", self.image], timeout=20)["ok"]:
            print(f"[WARN] cannot probe flags; image is not present: {self.image}")
            self._help_text = ""
            return self._help_text
        print(f"[{utc_now()}] probing engine flags in image (no GPU): {self.image}")
        res = self._docker(
            [
                "run",
                "--rm",
                "--pull=never",
                "--entrypoint",
                self.cfg.docker.python_bin,
                self.image,
                "-m",
                module_for(self.kind),
                "--help",
            ],
            timeout=300,
        )
        text = (res["out"] + "\n" + res["err"]).strip()
        if not text or not looks_like_help(text):
            print(f"[WARN] could not read --help from {self.image}: {res['err'][:300]}")
            self._help_text = ""
            return self._help_text
        if not res["ok"]:
            print(
                f"[{utc_now()}] --help exited {res['rc']} but printed usage; "
                "using it for flag validation"
            )
        self._help_text = text
        try:
            with open(cache_path, "w", encoding="utf-8") as fh:
                fh.write(self._help_text)
        except OSError:
            pass
        return self._help_text

    def check_flags(self, args: Sequence[str]) -> List[str]:
        """Return requested long flags that the installed engine does not know."""
        help_text = self.get_help_text()
        if not help_text:
            return []
        missing: List[str] = []
        for arg in args:
            if not arg.startswith("--"):
                continue
            flag = arg.split("=", 1)[0]
            if flag not in help_text:
                missing.append(flag)
        return missing

    # ----------------------------------------------------------------- commands
    def command(self) -> List[str]:
        c = self.cfg
        common = [
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--dtype",
            c.engine.dtype,
        ]
        if self.kind == "vllm":
            base = [
                "--model",
                c.model,
                "--gpu-memory-utilization",
                str(c.engine.gpu_memory_utilization),
                "--max-model-len",
                str(c.engine.max_model_len),
                "--seed",
                "0",
            ]
            if not self.drop_served_model_name:
                base += ["--served-model-name", c.served_model_name]
            return base + common + self.engine_args
        base = [
            "--model-path",
            c.model,
            "--mem-fraction-static",
            str(c.engine.gpu_memory_utilization),
            "--context-length",
            str(c.engine.max_model_len),
        ]
        if not self.drop_served_model_name:
            base += ["--served-model-name", c.served_model_name]
        return base + common + self.engine_args

    def apply_flag_fallbacks(self, missing: Sequence[str]) -> List[str]:
        """Drop optional flags the image no longer exposes; return the rest."""
        remaining: List[str] = []
        for flag in missing:
            if flag == "--served-model-name":
                if not self.drop_served_model_name:
                    print(f"[WARN] {self.kind} image has no --served-model-name; using /v1/models id")
                self.drop_served_model_name = True
            else:
                remaining.append(flag)
        return remaining

    def docker_run_cmd(self) -> List[str]:
        c = self.cfg.docker
        cmd = [
            "docker",
            "run",
            "-d",
            "--pull=never",
            "--name",
            self.container_name,
            "--label",
            "inference-exp=1",
            "--label",
            f"inference-exp.run={self.run_id}",
            "--label",
            f"inference-exp.kind={self.kind}",
            "--gpus",
            f"device={c.gpu_device}",
            "--shm-size",
            f"{c.shm_size_gb}g",
            "--memory",
            f"{c.container_memory_gb}g",
            "--memory-swap",
            f"{c.container_memory_gb}g",
            "--cpus",
            str(c.cpus),
            "--cpu-shares",
            str(c.cpu_shares),
            "--pids-limit",
            str(c.pids_limit),
            "--ulimit",
            "nofile=65535:65535",
            "--init",
            "--log-driver",
            "json-file",
            "--log-opt",
            f"max-size={c.log_max_size}",
            "--log-opt",
            f"max-file={c.log_max_file}",
            "--restart",
            "no",
            "-p",
            f"127.0.0.1:{self.port}:{self.port}",
            "-v",
            f"{c.hf_cache_volume}:/root/.cache/huggingface",
            "-e",
            "HF_HOME=/root/.cache/huggingface",
            "-e",
            "HF_HUB_DISABLE_TELEMETRY=1",
        ]
        if self.cfg.hf_endpoint:
            cmd += ["-e", f"HF_ENDPOINT={self.cfg.hf_endpoint}"]
        model = self.cfg.model
        if os.path.isabs(model) and os.path.isdir(model):
            cmd += ["-v", f"{model}:{model}:ro"]
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            cmd += ["-e", "HF_TOKEN"]
        cmd += ["--entrypoint", c.python_bin, self.image, "-m", module_for(self.kind)]
        cmd += self.command()
        return cmd

    # ------------------------------------------------------------------- start
    def start(self) -> None:
        if not _port_free(self.port):
            raise EngineStartError(
                f"port {self.port} is already in use. Stop the other process or change the port in configs/lab.json"
            )
        self.ensure_image()
        self.ensure_volume()
        self._remove_own_container()
        ensure_dir(self.run_dir)
        atomic_write_json(
            os.path.join(self.run_dir, "engine_command.json"),
            {
                "kind": self.kind,
                "arm": self.arm_name,
                "image": self.image,
                "image_digest": self.image_digest(),
                "docker_run": self.docker_run_cmd(),
                "started_at": utc_now(),
            },
        )
        print(f"[{utc_now()}] starting {self.kind} arm={self.arm_name} port={self.port}")
        res = self._docker(self.docker_run_cmd()[1:], timeout=120)
        if not res["ok"]:
            raise EngineStartError(f"docker run failed: {res['err'][:1000]}")
        self._started = True
        try:
            self.wait_ready()
        except Exception:
            self.save_logs()
            self.stop()
            raise

    def wait_ready(self) -> None:
        timeout = self.cfg.engine.startup_timeout_s
        started = time.time()
        last_note = 0.0
        while time.time() - started < timeout:
            status = self.container_state()
            state = status.get("status")
            if state in ("exited", "dead"):
                tail = self.logs_tail(80)
                raise EngineStartError(
                    f"{self.kind} container exited during startup "
                    f"(exit={status.get('exit_code')}, oom={status.get('oom_killed')})\n{tail}"
                )
            health = _http(self.base_url + "/health", timeout=3)
            models = _http(self.base_url + "/v1/models", timeout=3)
            healthy = bool(health.get("ok") and health.get("status") == 200)
            models_up = bool(models.get("ok") and models.get("status") == 200)
            if healthy or models_up:
                print(
                    f"[{utc_now()}] {self.kind} ready after {time.time() - started:.0f}s "
                    f"(health={healthy}, models={models_up})"
                )
                time.sleep(2.0)
                return
            now = time.time()
            if now - last_note > 30:
                last_note = now
                tail = self.logs_tail(3).replace("\n", " | ")[-240:]
                print(
                    f"[{utc_now()}] waiting for {self.kind} ({now - started:.0f}s) "
                    f"state={state} last_log={tail}"
                )
            time.sleep(self.cfg.engine.health_interval_s)
        raise EngineStartError(
            f"{self.kind} not ready after {timeout}s; last logs:\n{self.logs_tail(100)}"
        )

    # -------------------------------------------------------------------- state
    def container_state(self) -> Dict[str, Any]:
        res = self._docker(
            [
                "container",
                "inspect",
                self.container_name,
                "--format",
                "{{.State.Status}}|{{.State.OOMKilled}}|{{.State.ExitCode}}|{{.State.StartedAt}}|{{.State.FinishedAt}}",
            ],
            timeout=15,
        )
        if not res["ok"]:
            return {"status": "missing", "error": res["err"]}
        parts = res["out"].split("|")
        return {
            "status": parts[0] if parts else "unknown",
            "oom_killed": (parts[1].lower() == "true") if len(parts) > 1 else None,
            "exit_code": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None,
            "started_at": parts[3] if len(parts) > 3 else None,
            "finished_at": parts[4] if len(parts) > 4 else None,
        }

    def logs_tail(self, n: int = 50) -> str:
        res = self._docker(["logs", "--tail", str(n), self.container_name], timeout=20)
        return (res["out"] + "\n" + res["err"]).strip()

    def save_logs(self) -> Optional[str]:
        path = os.path.join(self.run_dir, "engine.log")
        ensure_dir(self.run_dir)
        try:
            with open(path, "w", encoding="utf-8", errors="replace") as fh:
                proc = subprocess.run(
                    ["docker", "logs", self.container_name],
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                )
            return path if proc.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None

    def stop(self) -> None:
        state = self.container_state()
        if state.get("status") not in (None, "missing"):
            atomic_write_json(os.path.join(self.run_dir, "container_state.json"), state)
            self.save_logs()
        self._docker(["stop", "-t", "30", self.container_name], timeout=90)
        self._docker(["rm", "-f", self.container_name], timeout=60)
        self._started = False

    # ------------------------------------------------------------------ metrics
    def metrics(self) -> Dict[str, float]:
        res = _http(self.base_url + "/metrics", timeout=10)
        if not res.get("ok"):
            return {}
        return _parse_prometheus(res.get("body") or "")

    def server_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"kind": self.kind, "base_url": self.base_url}
        if self.kind == "vllm":
            res = _http(self.base_url + "/version", timeout=5)
            info["version"] = (res.get("body") or "").strip() if res.get("ok") else None
        else:
            res = _http(self.base_url + "/get_server_info", timeout=10)
            if res.get("ok"):
                try:
                    info["server_info"] = json.loads(res.get("body") or "{}")
                except json.JSONDecodeError:
                    info["server_info_raw"] = (res.get("body") or "")[:4000]
        models = _http(self.base_url + "/v1/models", timeout=10)
        if models.get("ok"):
            try:
                info["models"] = json.loads(models.get("body") or "{}")
            except json.JSONDecodeError:
                pass
        info["collected_at"] = utc_now()
        return info

    @property
    def started(self) -> bool:
        return self._started


def cleanup_orphans(run_id: str) -> None:
    """Remove only containers labeled with this run id; never touches others."""
    res = run_cmd(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label=inference-exp.run={run_id}",
            "--format",
            "{{.Names}}",
        ],
        timeout=20,
    )
    if not res["ok"]:
        return
    for name in res["out"].splitlines():
        name = name.strip()
        if not name.startswith("inference-exp-"):
            continue
        print(f"[{utc_now()}] cleaning orphan container {name}")
        run_cmd(["docker", "rm", "-f", name], timeout=60)


class ExternalEngine:
    """Attach to an already-running OpenAI-compatible server (no container)."""

    def __init__(self, base_url: str, kind: str, cfg: Config, run_dir: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.kind = kind
        self.cfg = cfg
        self.run_dir = run_dir
        self.container_name = "external"
        self.port = int(self.base_url.rsplit(":", 1)[-1].split("/")[0]) if ":" in self.base_url else 0
        self.image = "external"
        self.engine_args: List[str] = []
        self._started = False

    def command(self) -> List[str]:
        return []

    def check_flags(self, args: Sequence[str]) -> List[str]:
        return []

    def apply_flag_fallbacks(self, missing: Sequence[str]) -> List[str]:
        return list(missing)

    def image_digest(self) -> Dict[str, Any]:
        return {"image": "external", "base_url": self.base_url}

    def start(self) -> None:
        res = _http(self.base_url + "/v1/models", timeout=10)
        if not res.get("ok"):
            raise EngineStartError(f"external engine not reachable at {self.base_url}: {res.get('error')}")
        self._started = True

    def stop(self) -> None:
        self._started = False

    def container_state(self) -> Dict[str, Any]:
        res = _http(self.base_url + "/health", timeout=5)
        return {"status": "external", "healthy": bool(res.get("ok")), "oom_killed": False}

    def metrics(self) -> Dict[str, float]:
        res = _http(self.base_url + "/metrics", timeout=10)
        if not res.get("ok"):
            return {}
        return _parse_prometheus(res.get("body") or "")

    def server_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"kind": self.kind, "base_url": self.base_url, "external": True}
        if self.kind == "vllm":
            res = _http(self.base_url + "/version", timeout=5)
            info["version"] = (res.get("body") or "").strip() if res.get("ok") else None
        else:
            res = _http(self.base_url + "/get_server_info", timeout=10)
            if res.get("ok"):
                try:
                    info["server_info"] = json.loads(res.get("body") or "{}")
                except json.JSONDecodeError:
                    pass
        return info

    def logs_tail(self, n: int = 50) -> str:
        return ""

    def save_logs(self) -> Optional[str]:
        return None

    @property
    def started(self) -> bool:
        return self._started
