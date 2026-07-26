"""Run a resumable multi-model browser-use collection campaign.

Local models are served one at a time through vLLM or SGLang. Cloud models are
run in the configured order. The existing orchestrator remains responsible for
episode-level parallelism, trace resume, and browser cleanup.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import yaml

import orchestrator


PROJECT_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG = PROJECT_DIR / "browser_use_campaign.yaml"


class CampaignError(RuntimeError):
    pass


class OutOfCredit(CampaignError):
    pass


class ResourceBlocked(CampaignError):
    pass


class CampaignRunner:
    def __init__(self, config_path: Path, dry_run: bool = False) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        self.settings = self.config.get("campaign", {})
        experiment_path = Path(
            self.settings.get("experiment_config", "multi_harness_config.yaml")
        )
        self.experiment_path = (
            experiment_path
            if experiment_path.is_absolute()
            else PROJECT_DIR / experiment_path
        ).resolve()
        self.experiment = orchestrator.load_experiment(self.experiment_path)
        self.trace_output_dir = orchestrator.resolve_output_dir(self.experiment)
        self.registry = orchestrator.load_registry()
        self.datasets = orchestrator.resolve_datasets(self.experiment)
        self.episodes_per_combo = self.experiment.get("run", {}).get(
            "episodes_per_combo", 1
        )
        self.harness = self.settings.get("harness", "browser_use")
        self.dry_run = dry_run
        self.run_id = (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid.uuid4().hex[:8]
        )
        run_root = Path(self.settings.get("run_root", "campaign_runs"))
        self.run_dir = (
            run_root if run_root.is_absolute() else PROJECT_DIR / run_root
        ) / self.run_id
        self.manifest_path = self.run_dir / "manifest.json"
        self.manifest: dict[str, Any] = {
            "campaign_id": self.run_id,
            "config": str(self.config_path),
            "experiment_config": str(self.experiment_path),
            "started_at": self._utc_now(),
            "status": "running",
            "models": {},
        }
        self.server_process: subprocess.Popen | None = None
        self.orchestrator_process: subprocess.Popen | None = None
        self.server_log_handle = None
        self.openrouter_model_ids: set[str] | None = None

        resolved = orchestrator.resolve_agents(self.experiment, self.registry)
        self.agent_env = {
            agent["agent_id"]: agent["env"]
            for agent in resolved
            if agent["harness"] == self.harness
        }

    @staticmethod
    def _utc_now() -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _save_manifest(self) -> None:
        if self.dry_run:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2))
        temporary.replace(self.manifest_path)

    def _set_model_state(self, agent_id: str, **updates: Any) -> None:
        state = self.manifest["models"].setdefault(agent_id, {})
        state.update(updates)
        state["updated_at"] = self._utc_now()
        self._save_manifest()

    def _question(self, item: dict | str) -> str:
        return item["question"] if isinstance(item, dict) else item

    def collection_progress(self, agent_id: str) -> tuple[int, int]:
        collected = 0
        expected = 0
        for dataset in self.datasets:
            counts = orchestrator.find_completed_counts(
                agent_id,
                dataset["name"],
                self.harness,
                self.trace_output_dir,
            )
            for item in dataset["questions"]:
                expected += self.episodes_per_combo
                collected += min(
                    counts.get(self._question(item), 0),
                    self.episodes_per_combo,
                )
        return collected, expected

    def _require_registered_agent(self, agent_id: str) -> dict[str, str]:
        if agent_id not in self.agent_env:
            raise CampaignError(
                f"{agent_id} is not configured for {self.harness} in "
                f"{self.experiment_path.name}"
            )
        return self.agent_env[agent_id]

    @staticmethod
    def _local_engine(
        spec: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        engines = [engine for engine in ("vllm", "sglang") if engine in spec]
        if len(engines) != 1:
            raise CampaignError(
                f"{spec.get('agent_id', '<unknown>')} must configure exactly "
                "one local engine: vllm or sglang"
            )
        engine = engines[0]
        engine_config = spec[engine]
        if not isinstance(engine_config, dict):
            raise CampaignError(
                f"{spec.get('agent_id', '<unknown>')}.{engine} must be a mapping"
            )
        return engine, engine_config

    def _gpu_memory_used(self) -> dict[int, int]:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        usage: dict[int, int] = {}
        for line in result.stdout.splitlines():
            index, used = (part.strip() for part in line.split(",", 1))
            usage[int(index)] = int(used)
        return usage

    def _assert_gpus_available(self, engine_config: dict[str, Any]) -> None:
        requested = [int(gpu) for gpu in engine_config.get("gpus", [])]
        maximum_used = int(
            engine_config.get(
                "max_existing_gpu_memory_mib",
                self.settings.get("max_existing_gpu_memory_mib", 2048),
            )
        )
        usage = self._gpu_memory_used()
        busy = {
            gpu: usage.get(gpu)
            for gpu in requested
            if usage.get(gpu, maximum_used + 1) > maximum_used
        }
        if busy:
            details = ", ".join(
                f"GPU {gpu}: {used} MiB used" for gpu, used in busy.items()
            )
            raise ResourceBlocked(f"requested GPUs are not free ({details})")

    @staticmethod
    def _port_is_open(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    @staticmethod
    def _json_request(
        url: str,
        *,
        api_key: str | None = None,
        payload: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        data = None
        method = "GET"
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if payload is not None:
            method = "POST"
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode()
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    def _server_api_key(self, engine: str) -> str | None:
        if engine == "vllm":
            return os.getenv("VLLM_API_KEY") or "not-needed"
        return None

    def _server_command(self, engine: str, engine_config: dict[str, Any]) -> list[str]:
        model = str(engine_config["model"])
        port = str(engine_config["port"])
        args = [str(value) for value in engine_config.get("args", [])]
        if engine == "vllm":
            executable = str(
                engine_config.get(
                    "executable",
                    self.settings.get("vllm_executable", "vllm"),
                )
            )
            return [executable, "serve", model, "--port", port, *args]
        if engine == "sglang":
            executable = str(
                engine_config.get(
                    "executable",
                    self.settings.get("sglang_executable", sys.executable),
                )
            )
            command = [
                executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                model,
                "--port",
                port,
            ]
            served_model = engine_config.get("served_model_name")
            if served_model and "--served-model-name" not in args:
                command.extend(["--served-model-name", str(served_model)])
            return [*command, *args]
        raise CampaignError(f"unsupported local engine: {engine}")

    def _wait_for_server(
        self,
        engine: str,
        port: int,
        served_model: str,
        startup_timeout_s: float,
    ) -> None:
        deadline = time.monotonic() + startup_timeout_s
        last_error = "server has not answered"
        api_key = self._server_api_key(engine)
        while time.monotonic() < deadline:
            if (
                self.server_process is not None
                and self.server_process.poll() is not None
            ):
                raise CampaignError(
                    f"{engine} exited during startup with code "
                    f"{self.server_process.returncode}"
                )
            try:
                response = self._json_request(
                    f"http://127.0.0.1:{port}/v1/models",
                    api_key=api_key,
                    timeout=5,
                )
                model_ids = {item.get("id") for item in response.get("data", [])}
                if served_model in model_ids:
                    return
                last_error = f"served models were {sorted(model_ids)}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(5)
        raise CampaignError(
            f"{engine} did not become ready within {startup_timeout_s:g}s: "
            f"{last_error}"
        )

    def _smoke_test_server(self, engine: str, port: int, served_model: str) -> None:
        self._json_request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            api_key=self._server_api_key(engine),
            payload={
                "model": served_model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 8,
            },
            timeout=float(
                self.settings.get(
                    f"{engine}_smoke_timeout_s",
                    self.settings.get("server_smoke_timeout_s", 120),
                )
            ),
        )

    def start_server(self, agent_id: str, spec: dict[str, Any]) -> None:
        engine, engine_config = self._local_engine(spec)
        self._assert_gpus_available(engine_config)
        port = int(engine_config["port"])
        if self._port_is_open(port):
            raise ResourceBlocked(
                f"port {port} is already occupied; refusing to stop or reuse "
                "a server the campaign did not start"
            )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu) for gpu in engine_config.get("gpus", [])
        )
        command = self._server_command(engine, engine_config)
        log_path = self.run_dir / "logs" / f"{agent_id}.{engine}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.server_log_handle = log_path.open("w")
        self.server_process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            env=env,
            stdout=self.server_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self._set_model_state(
            agent_id,
            server_engine=engine,
            server_pid=self.server_process.pid,
            server_command=command,
            server_log=str(log_path),
        )
        self._wait_for_server(
            engine,
            port,
            str(engine_config.get("served_model_name", engine_config["model"])),
            float(engine_config.get("startup_timeout_s", 1800)),
        )
        if engine_config.get("smoke_test", True):
            self._smoke_test_server(
                engine,
                port,
                str(engine_config.get("served_model_name", engine_config["model"])),
            )

    @staticmethod
    def _terminate_process(
        process: subprocess.Popen | None, grace_s: float = 20
    ) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def stop_server(self) -> None:
        self._terminate_process(self.server_process)
        self.server_process = None
        if self.server_log_handle is not None:
            self.server_log_handle.close()
            self.server_log_handle = None

    def _run_orchestrator_once(self, agent_id: str, workers: int) -> int:
        command = [
            sys.executable,
            str(PROJECT_DIR / "orchestrator.py"),
            "--config",
            str(self.experiment_path),
            "--agents",
            agent_id,
            "--harnesses",
            self.harness,
            "--workers",
            str(workers),
        ]
        log_path = self.run_dir / "logs" / f"{agent_id}.orchestrator.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        process_env = os.environ.copy()
        process_env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("a") as log:
            self.orchestrator_process = subprocess.Popen(
                command,
                cwd=PROJECT_DIR,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert self.orchestrator_process.stdout is not None
            for line in self.orchestrator_process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            returncode = self.orchestrator_process.wait()
        self.orchestrator_process = None
        return returncode

    def collect_model(self, agent_id: str, workers: int) -> None:
        maximum_rounds = int(self.settings.get("max_collection_rounds", 2))
        for round_number in range(1, maximum_rounds + 1):
            before, expected = self.collection_progress(agent_id)
            self._set_model_state(
                agent_id,
                collected=before,
                expected=expected,
                collection_round=round_number,
            )
            if before >= expected:
                self._set_model_state(agent_id, status="complete")
                return
            returncode = self._run_orchestrator_once(agent_id, workers)
            after, _ = self.collection_progress(agent_id)
            self._set_model_state(
                agent_id,
                collected=after,
                orchestrator_returncode=returncode,
            )
            if returncode == 2:
                raise OutOfCredit(
                    f"{agent_id} stopped after a fatal API or credit error"
                )
            if returncode != 0:
                raise CampaignError(
                    f"{agent_id} orchestrator exited with code {returncode}"
                )
            if after >= expected:
                self._set_model_state(agent_id, status="complete")
                return
            if after <= before:
                raise CampaignError(
                    f"{agent_id} made no collection progress "
                    f"({after}/{expected} valid traces)"
                )
        collected, expected = self.collection_progress(agent_id)
        raise CampaignError(
            f"{agent_id} remains incomplete after {maximum_rounds} rounds "
            f"({collected}/{expected})"
        )

    def _usage_summary(self, agent_id: str) -> dict[str, Any]:
        totals = {
            "episodes_with_usage": 0,
            "prompt_tokens": 0,
            "cached_prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "invocations": 0,
        }
        trace_root = self.trace_output_dir / agent_id
        for path in trace_root.glob(f"*/{self.harness}/*/*.json"):
            try:
                episode = json.loads(path.read_text())
                usage = (episode.get("browser_use_log") or {}).get("usage")
                if not usage:
                    continue
                totals["episodes_with_usage"] += 1
                totals["prompt_tokens"] += int(usage.get("total_prompt_tokens", 0))
                totals["cached_prompt_tokens"] += int(
                    usage.get("total_prompt_cached_tokens", 0)
                )
                totals["completion_tokens"] += int(
                    usage.get("total_completion_tokens", 0)
                )
                totals["total_tokens"] += int(usage.get("total_tokens", 0))
                totals["invocations"] += int(usage.get("entry_count", 0))
            except Exception:
                continue
        return totals

    def _openrouter_key(self) -> str:
        key = os.getenv("OPEN_ROUTER_API_KEY", "")
        if not key:
            raise CampaignError("OPEN_ROUTER_API_KEY is not set")
        return key

    def openrouter_credit_remaining(self) -> float | None:
        response = self._json_request(
            "https://openrouter.ai/api/v1/key",
            api_key=self._openrouter_key(),
            timeout=30,
        )
        data = response.get("data", response)
        remaining = data.get("limit_remaining")
        return float(remaining) if remaining is not None else None

    def _check_openrouter_credit(self) -> float | None:
        remaining = self.openrouter_credit_remaining()
        minimum = float(self.settings.get("openrouter_minimum_credit", 5))
        if remaining is not None and remaining < minimum:
            raise OutOfCredit(
                f"OpenRouter key has {remaining:.2f} credits remaining; "
                f"minimum is {minimum:.2f}"
            )
        return remaining

    def _load_openrouter_models(self) -> set[str]:
        response = self._json_request("https://openrouter.ai/api/v1/models", timeout=30)
        return {item["id"] for item in response.get("data", [])}

    def validate_openrouter_model(self, model_name: str) -> None:
        if self.openrouter_model_ids is None:
            self.openrouter_model_ids = self._load_openrouter_models()
        if model_name not in self.openrouter_model_ids:
            raise CampaignError(
                f"OpenRouter model {model_name!r} is not currently available"
            )

    def run_local(self, spec: dict[str, Any]) -> None:
        agent_id = spec["agent_id"]
        engine, engine_config = self._local_engine(spec)
        env = self._require_registered_agent(agent_id)
        collected, expected = self.collection_progress(agent_id)
        print(f"[LOCAL:{engine}] {agent_id}: " f"{collected}/{expected} valid traces")
        if collected >= expected:
            self._set_model_state(
                agent_id,
                kind="local",
                engine=engine,
                status="complete",
                collected=collected,
                expected=expected,
                accounting=self._usage_summary(agent_id),
            )
            return
        if self.dry_run:
            print(
                f"[DRY-RUN] {shlex.join(self._server_command(engine, engine_config))}\n"
                f"[DRY-RUN] GPUs {engine_config.get('gpus')}; collect with "
                f"{self.settings.get('local_workers', 10)} workers"
            )
            return
        configured_url = env.get("MIDSCENE_MODEL_BASE_URL", "")
        expected_port = f":{engine_config['port']}/"
        if expected_port not in configured_url:
            raise CampaignError(
                f"{agent_id} registry URL {configured_url!r} does not match "
                f"campaign port {engine_config['port']}"
            )
        self._set_model_state(
            agent_id,
            kind="local",
            engine=engine,
            status=f"starting_{engine}",
        )
        try:
            self.start_server(agent_id, spec)
            self._set_model_state(agent_id, status="collecting")
            self.collect_model(agent_id, int(self.settings.get("local_workers", 10)))
            self._set_model_state(
                agent_id,
                accounting=self._usage_summary(agent_id),
            )
        finally:
            self.stop_server()

    def run_openrouter(self, spec: dict[str, Any]) -> None:
        agent_id = spec["agent_id"]
        env = self._require_registered_agent(agent_id)
        model_name = env.get("MIDSCENE_MODEL_NAME", "")
        if "openrouter.ai" not in env.get("MIDSCENE_MODEL_BASE_URL", ""):
            raise CampaignError(f"{agent_id} is not configured for OpenRouter")
        if model_name != spec["model"]:
            raise CampaignError(
                f"{agent_id} registry model {model_name!r} does not match "
                f"campaign model {spec['model']!r}"
            )
        collected, expected = self.collection_progress(agent_id)
        print(f"[OPENROUTER] {agent_id}: {collected}/{expected} valid traces")
        if collected >= expected:
            self._set_model_state(
                agent_id,
                kind="openrouter",
                status="complete",
                collected=collected,
                expected=expected,
                accounting=self._usage_summary(agent_id),
            )
            return
        if self.dry_run:
            print(
                f"[DRY-RUN] would collect {model_name} with "
                f"{self.settings.get('openrouter_workers', 3)} workers"
            )
            return
        self.validate_openrouter_model(model_name)
        before_credit = self._check_openrouter_credit()
        self._set_model_state(
            agent_id,
            kind="openrouter",
            status="collecting",
            credit_before=before_credit,
        )
        try:
            self.collect_model(
                agent_id, int(self.settings.get("openrouter_workers", 3))
            )
        except OutOfCredit:
            try:
                remaining = self.openrouter_credit_remaining()
            except Exception:
                remaining = None
            self._set_model_state(
                agent_id,
                status="stopped_out_of_credit",
                credit_after=remaining,
                accounting=self._usage_summary(agent_id),
            )
            raise
        after_credit = self.openrouter_credit_remaining()
        spend = (
            before_credit - after_credit
            if before_credit is not None and after_credit is not None
            else None
        )
        self._set_model_state(
            agent_id,
            credit_after=after_credit,
            credit_spent=spend,
            accounting=self._usage_summary(agent_id),
        )

    def stop_owned_processes(self) -> None:
        self._terminate_process(self.orchestrator_process, grace_s=5)
        self.orchestrator_process = None
        self.stop_server()

    def run(
        self,
        *,
        only: set[str] | None = None,
        skip_local: bool = False,
        skip_openrouter: bool = False,
    ) -> int:
        if not self.dry_run:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._save_manifest()
        blocked: list[str] = []
        try:
            if not skip_local:
                for spec in self.config.get("local_models", []):
                    agent_id = spec["agent_id"]
                    if only and agent_id not in only:
                        continue
                    try:
                        self.run_local(spec)
                    except ResourceBlocked as exc:
                        blocked.append(agent_id)
                        self._set_model_state(
                            agent_id,
                            kind="local",
                            status="resource_blocked",
                            reason=str(exc),
                        )
                        print(f"[BLOCKED] {agent_id}: {exc}")
                        if not self.settings.get("continue_on_resource_blocked", True):
                            raise

            if not skip_openrouter:
                for spec in self.config.get("openrouter_models", []):
                    agent_id = spec["agent_id"]
                    if only and agent_id not in only:
                        continue
                    self.run_openrouter(spec)

            self.manifest["status"] = (
                "complete_with_resource_blocks" if blocked else "complete"
            )
            self.manifest["resource_blocked_models"] = blocked
            self.manifest["finished_at"] = self._utc_now()
            self._save_manifest()
            return 3 if blocked else 0
        except OutOfCredit as exc:
            self.manifest["status"] = "stopped_out_of_credit"
            self.manifest["reason"] = str(exc)
            self.manifest["finished_at"] = self._utc_now()
            self._save_manifest()
            print(f"[OUT OF CREDIT] {exc}")
            return 2
        except KeyboardInterrupt:
            self.manifest["status"] = "interrupted"
            self.manifest["finished_at"] = self._utc_now()
            self._save_manifest()
            print("[INTERRUPTED] Campaign stopped; completed traces are resumable.")
            return 130
        except Exception as exc:
            self.manifest["status"] = "failed"
            self.manifest["reason"] = f"{type(exc).__name__}: {exc}"
            self.manifest["finished_at"] = self._utc_now()
            self._save_manifest()
            print(f"[FAILED] {type(exc).__name__}: {exc}")
            return 1
        finally:
            self.stop_owned_processes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable browser-use multi-model campaign."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--skip-openrouter", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = CampaignRunner(args.config, dry_run=args.dry_run)

    def handle_signal(_signum, _frame):
        runner.stop_owned_processes()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    return runner.run(
        only=set(args.only) if args.only else None,
        skip_local=args.skip_local,
        skip_openrouter=args.skip_openrouter,
    )


if __name__ == "__main__":
    raise SystemExit(main())
