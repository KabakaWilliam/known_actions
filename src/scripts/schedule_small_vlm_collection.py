#!/usr/bin/env python3
"""Resource-aware collection queue for the small-VLM engine experiment."""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
QUEUE_LOCK = Path("/tmp/known-actions-small-vlm-engine-queue.lock")


@dataclass(frozen=True)
class Job:
    agent_id: str
    engine: str
    config: str


PHASES = [
    [
        Job(
            "glm_4.6v_flash",
            "vllm",
            "experiments/inference_engine/configs/"
            "webshop_small_vlm_vllm_campaign.yaml",
        ),
        Job(
            "glm_4.6v_flash",
            "sglang",
            "experiments/inference_engine/configs/"
            "webshop_small_vlm_sglang_campaign.yaml",
        ),
    ],
    [
        Job(
            "qwen3vl_8b",
            "vllm",
            "experiments/inference_engine/configs/"
            "webshop_small_vlm_vllm_campaign.yaml",
        ),
        Job(
            "qwen3vl_8b",
            "sglang",
            "experiments/inference_engine/configs/"
            "webshop_small_vlm_sglang_campaign.yaml",
        ),
    ],
]


def gpu_state() -> dict[int, tuple[int, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = csv.reader(result.stdout.splitlines())
    return {
        int(index.strip()): (int(used.strip()), int(utilization.strip()))
        for index, used, utilization in rows
    }


def command(job: Job, gpu: int, python: str) -> list[str]:
    return [
        python,
        "browser_use_campaign.py",
        "--config",
        job.config,
        "--only",
        job.agent_id,
        "--gpus",
        str(gpu),
        "--skip-openrouter",
    ]


def run_phase(
    jobs: list[Job],
    *,
    candidate_gpus: list[int],
    maximum_used_mib: int,
    maximum_utilization: int,
    poll_seconds: int,
    python: str,
) -> None:
    pending = list(jobs)
    active: dict[Job, tuple[subprocess.Popen, int]] = {}
    while pending or active:
        for job, (process, gpu) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            del active[job]
            if returncode == 3:
                pending.insert(0, job)
                print(
                    f"[QUEUE RETRY] {job.agent_id}@{job.engine} lost its GPU "
                    "allocation race; returning it to the queue",
                    flush=True,
                )
                continue
            if returncode != 0:
                raise RuntimeError(
                    f"{job.agent_id}@{job.engine} failed with exit code "
                    f"{returncode}; completed traces remain resumable"
                )
            print(
                f"[QUEUE COMPLETE] {job.agent_id}@{job.engine} on GPU {gpu}",
                flush=True,
            )

        reserved = {gpu for _process, gpu in active.values()}
        try:
            state = gpu_state()
        except Exception as exc:
            print(f"[QUEUE] GPU check failed: {exc}; retrying", flush=True)
            time.sleep(poll_seconds)
            continue
        available = [
            gpu
            for gpu in candidate_gpus
            if gpu not in reserved
            and state.get(
                gpu,
                (maximum_used_mib + 1, maximum_utilization + 1),
            )[0]
            <= maximum_used_mib
            and state.get(
                gpu,
                (maximum_used_mib + 1, maximum_utilization + 1),
            )[1]
            <= maximum_utilization
        ]
        while pending and available:
            job = pending.pop(0)
            gpu = available.pop(0)
            print(
                f"[QUEUE START] {job.agent_id}@{job.engine} on GPU {gpu}",
                flush=True,
            )
            process = subprocess.Popen(
                command(job, gpu, python),
                cwd=PROJECT_DIR,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            active[job] = (process, gpu)

        if pending or active:
            if pending and not available:
                details = ", ".join(
                    (
                        f"GPU {gpu}={state[gpu][0]} MiB/"
                        f"{state[gpu][1]}%"
                        if gpu in state
                        else f"GPU {gpu}=unknown"
                    )
                    for gpu in candidate_gpus
                )
                print(
                    f"[QUEUE WAIT] pending={len(pending)}; {details}; "
                    f"checking again in {poll_seconds}s",
                    flush=True,
                )
            time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--maximum-used-mib", type=int, default=2048)
    parser.add_argument("--maximum-utilization", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--python",
        default="/opt/anaconda/envs/dispatch/bin/python",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    QUEUE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_LOCK.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another small-VLM collection queue holds {QUEUE_LOCK}"
            ) from exc
        lock.write(f"{os.getpid()}\n")
        lock.flush()
        for index, jobs in enumerate(PHASES, start=1):
            label = "GLM-Flash" if index == 1 else "Qwen3-VL-8B"
            print(f"[QUEUE PHASE {index}/{len(PHASES)}] {label}", flush=True)
            run_phase(
                jobs,
                candidate_gpus=args.gpus,
                maximum_used_mib=args.maximum_used_mib,
                maximum_utilization=args.maximum_utilization,
                poll_seconds=args.poll_seconds,
                python=args.python,
            )
        print("[QUEUE COMPLETE] all engine/model conditions", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
