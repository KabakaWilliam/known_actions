# reusable worker
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

# Get the directory where this module is located
_MODULE_DIR = Path(__file__).parent.absolute()

DEFAULT_SANDBOX = os.environ.get(
    "PLAYWRIGHT_SANDBOX_DIR",
    str(_MODULE_DIR / "playwright-sandbox"),
)


class SandboxBrowserError(RuntimeError):
    pass


def run_browser_worker(
    payload: Mapping[str, Any],
    worker_script: str = "browser_worker.py",
    sandbox_dir: str = DEFAULT_SANDBOX,
    workdir: Optional[str] = None,
    headed: bool = True,
    extra_env: Optional[Mapping[str, str]] = None,
    capture_output: bool = True,
    check: bool = True,
    target_artifacts_dir: Optional[Path] = None,
) -> dict[str, Any]:
    workdir = os.path.abspath(workdir or os.getcwd())
    sandbox_dir = os.path.abspath(sandbox_dir)
    worker_path = os.path.join(workdir, worker_script)

    if not os.path.isdir(sandbox_dir):
        raise SandboxBrowserError(
            f"Sandbox not found at {sandbox_dir}. Run setup_browser_image.sh first."
        )
    if not os.path.isfile(worker_path):
        raise SandboxBrowserError(f"Worker script not found: {worker_path}")

    # Determine artifacts directory
    if target_artifacts_dir is not None:
        # Use provided directory for this eval run
        target_artifacts_dir = Path(target_artifacts_dir)
        if not target_artifacts_dir.is_absolute():
            # Relative path - combine with workdir
            artifacts_dir = Path(workdir) / target_artifacts_dir
            artifacts_relative = target_artifacts_dir
        else:
            # Absolute path - compute relative
            artifacts_dir = target_artifacts_dir
            artifacts_relative = artifacts_dir.relative_to(Path(workdir))
    else:
        # Create session timestamp directory for artifacts (default behavior)
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        artifacts_dir = Path(workdir) / "artifacts" / session_timestamp
        artifacts_relative = Path("artifacts") / session_timestamp
    
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload_path = artifacts_dir / "browser_payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    host_env = os.environ.copy()
    extra_env = dict(extra_env or {})
    host_env.update(extra_env)

    inner_cmd = ["python", f"/work/{Path(worker_script).name}", "--payload", f"/work/{artifacts_relative}/browser_payload.json"]
    if headed:
        inner_cmd = ["xvfb-run", "-a", *inner_cmd]

    env_flags: list[str] = []
    passthrough_keys = ["OPENAI_API_KEY"]
    for key in passthrough_keys:
        value = host_env.get(key)
        if value:
            env_flags.extend(["--env", f"{key}={value}"])
    for key, value in extra_env.items():
        env_flags.extend(["--env", f"{key}={value}"])

    quoted = " ".join(_shell_quote(part) for part in inner_cmd)
    cmd = [
        "apptainer",
        "exec",
        "--cleanenv",
        "--bind",
        f"{workdir}:/work",
        "--pwd",
        "/work",
        *env_flags,
        sandbox_dir,
        "/bin/bash",
        "-lc",
        quoted,
    ]

    result = subprocess.run(
        cmd,
        env=host_env,
        capture_output=capture_output,
        text=True,
    )

    if check and result.returncode != 0:
        raise SandboxBrowserError(
            "Sandboxed browser worker failed.\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:{result.stderr}"
        )

    stdout = (result.stdout or "").strip()
    if not stdout:
        return {"status": "ok", "stdout": "", "returncode": result.returncode}

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "status": "ok" if result.returncode == 0 else "error",
            "stdout": stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"
