import argparse, datetime, itertools, json, os, random, re, shlex, shutil, signal, subprocess, sys, threading, time, uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from qa_dataset import QA_QUESTIONS

try:
    import yaml
except ImportError:
    print("PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not installed. Run: pip install tqdm")
    sys.exit(1)

PROJECT_DIR   = Path(__file__).parent.resolve()
REGISTRY      = PROJECT_DIR / "config.yaml"           # agent + loader registry
DEFAULT_CFG   = PROJECT_DIR / "custom_config.yaml"    # default experiment spec
DEFAULT_SIF_PATH = PROJECT_DIR / "agent.sif"
SIF_PATH      = DEFAULT_SIF_PATH
OUTPUT_DIR    = PROJECT_DIR / "traces"
TEMPLATES_DIR = PROJECT_DIR / "task_prompt_templates"
UNKNOWN_ANSWER_SENTINELS = {"", "na", "n/a", "none", "null", "unknown"}
ACTIVE_PROCESSES: set[subprocess.Popen] = set()
ACTIVE_PROCESSES_LOCK = threading.Lock()


def has_known_answer(value) -> bool:
    return (
        value is not None
        and str(value).strip().lower() not in UNKNOWN_ANSWER_SENTINELS
    )


def load_template(name_or_inline: str | None, task_type: str) -> str | None:
    """Resolve a prompt template string.

    Priority:
      1. name_or_inline is a multiline string → use inline as-is (backwards compat)
      2. name_or_inline is a short name → load task_prompt_templates/{name}.txt
      3. name_or_inline is None → try task_prompt_templates/{task_type}.txt
      4. No matching file → return None (agent_runner uses its own fallback)
    """
    if name_or_inline is not None and "\n" in name_or_inline:
        return name_or_inline
    name = name_or_inline or task_type
    path = TEMPLATES_DIR / f"{name}.txt"
    return path.read_text() if path.exists() else None

load_dotenv(PROJECT_DIR / ".env")


def _expand_env(value: str) -> str:
    return re.sub(r'\$([A-Z_][A-Z0-9_]*)', lambda m: os.environ.get(m.group(1), ""), value)


def load_registry() -> dict:
    with open(REGISTRY) as f:
        return yaml.safe_load(f)


def load_experiment(cfg_path: Path) -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve_agents(exp: dict, registry: dict) -> list[dict]:
    """Merge experiment agent list with registry definitions and harnesses.

    For each agent_id in the experiment, look up its env from the registry,
    apply harness defaults, then apply any per-agent env overrides from the
    experiment config. Experiments without a harness list remain MidScene-only.
    """
    reg_by_id   = {a["agent_id"]: a for a in registry.get("agents", [])}
    midscene_defaults = {k: str(v) for k, v in registry.get("midscene_defaults", {}).items()}
    browser_use_defaults = {k: str(v) for k, v in registry.get("browser_use_defaults", {}).items()}
    global_harnesses = exp.get("harnesses", ["midscene"])
    valid_harnesses = {"midscene", "browser_use"}

    agents = []
    for entry in exp.get("agents", []):
        agent_id = entry["agent_id"]
        if agent_id not in reg_by_id:
            print(f"[WARN] agent_id '{agent_id}' not found in registry ({REGISTRY.name}). Skipping.")
            continue

        harnesses = entry.get("harnesses", global_harnesses)
        for harness in harnesses:
            if harness not in valid_harnesses:
                print(f"[WARN] Unknown harness '{harness}' for '{agent_id}'. Skipping.")
                continue

            # 1. harness defaults  2. registry env  3. experiment overrides
            env = {**midscene_defaults}
            if harness == "browser_use":
                env.update(browser_use_defaults)
            for k, v in reg_by_id[agent_id].get("env", {}).items():
                env[k] = _expand_env(str(v))
            for k, v in entry.get("env", {}).items():
                env[k] = _expand_env(str(v))

            agents.append({"agent_id": agent_id, "harness": harness, "env": env})

    return agents


def resolve_datasets(exp: dict) -> list[dict]:
    """Load question lists for each dataset entry in the experiment config."""
    datasets = []
    for dcfg in exp.get("datasets", []):
        name   = dcfg["name"]
        source = dcfg.get("source", "builtin")

        if source == "builtin":
            questions = list(QA_QUESTIONS)
        else:
            path = PROJECT_DIR / source
            if not path.exists():
                print(f"[WARN] Dataset file not found: {path}")
                print(f"       Run: python prep_datasets.py --config <your_config.yaml>")
                continue
            with open(path) as f:
                questions = json.load(f)

        n    = dcfg.get("n_questions")
        seed = dcfg.get("seed", 42)
        if n is not None and len(questions) > n:
            questions = random.Random(seed).sample(questions, n)

        task_type = dcfg.get("task_type", "qa")
        datasets.append({
            "name":                  name,
            "questions":             questions,
            "start_url":             dcfg.get("start_url", "https://en.wikipedia.org"),
            "task_prompt_template":  load_template(dcfg.get("task_prompt_template"), task_type),
            "task_type":             task_type,
        })

    return datasets


def build_apptainer_cmd(
    agent,
    q: dict,
    episode_id,
    output_dir: str,
    dataset: dict,
    task_timeout_s: int | float | None = None,
    episode_tmp_dir: Path | None = None,
) -> list[str]:
    question = q["question"]
    # Per-question start_url (e.g. WebGames) overrides dataset-level start_url
    start_url = q.get("start_url") or dataset["start_url"]
    cmd = [
        "apptainer", "exec",
        "--bind", f"{PROJECT_DIR}:/app/workspace",
    ]
    if agent["harness"] == "browser_use":
        if episode_tmp_dir is None:
            raise ValueError("browser-use requires an episode temp directory")
        cmd += ["--bind", f"{episode_tmp_dir}:/browser-use-tmp"]
    cmd += ["--no-home", "--pwd", "/app"]
    for key, value in agent["env"].items():
        cmd += ["--env", f"{key}={value}"]
    if agent["harness"] == "browser_use":
        for key, value in {
            "TMPDIR": "/browser-use-tmp",
            "TMP": "/browser-use-tmp",
            "TEMP": "/browser-use-tmp",
            "BROWSER_USE_EPISODE_TMP": "/browser-use-tmp",
            "BROWSER_USE_CONFIG_DIR": "/browser-use-tmp/config",
            "XDG_CONFIG_HOME": "/browser-use-tmp/xdg",
        }.items():
            cmd += ["--env", f"{key}={value}"]
    runner = (
        ["npx", "tsx", "/app/workspace/agent_runner.ts"]
        if agent["harness"] == "midscene"
        else ["python3", "/app/workspace/browser_use_runner.py"]
    )
    cmd += [
        str(SIF_PATH),
        *runner,
        "--question",   question,
        "--agent_id",   agent["agent_id"],
        "--episode_id", episode_id,
        "--output_dir", output_dir,
        "--start_url",  start_url,
        "--task_type",  dataset["task_type"],
    ]
    template = dataset.get("task_prompt_template")
    if template:
        prompt = template.replace("{question}", question).replace("{start_url}", start_url)
        cmd += ["--task_prompt", prompt]
    expected_answer = q.get("answer")
    if has_known_answer(expected_answer):
        cmd += ["--expected-answer", str(expected_answer)]
    if agent["harness"] == "browser_use" and task_timeout_s is not None:
        cmd += ["--task-timeout", str(task_timeout_s)]
    return cmd


def format_cmd_for_display(cmd: list[str]) -> str:
    """Shell-quote a dry-run command while masking credential environment values."""
    displayed = list(cmd)
    for index, token in enumerate(displayed[:-1]):
        if token != "--env":
            continue
        key, separator, _value = displayed[index + 1].partition("=")
        if separator and any(marker in key.upper() for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
            displayed[index + 1] = f"{key}=***"
    return shlex.join(displayed)


# ─── Resume helpers ───────────────────────────────────────────────────────────

def find_completed_counts(agent_id: str, dataset_name: str, harness: str) -> dict[str, int]:
    """Return {question: n_valid_traces} for already-collected episodes.

    Counts traces that are not API-level failures and have DOM events.
    Task-level failures (replanning limit, unclear instructions) are counted
    as valid behavioral traces — only fatal API errors are excluded.
    """
    _fatal = [p.lower() for p in FATAL_API_PATTERNS]
    dataset_dir = OUTPUT_DIR / agent_id / dataset_name
    if not dataset_dir.exists():
        return {}
    counts: dict[str, int] = defaultdict(int)
    trace_files = list((dataset_dir / harness).glob("*/*.json"))
    # Traces collected before harness metadata existed are MidScene episodes.
    if harness == "midscene":
        trace_files.extend(dataset_dir.glob("*/*.json"))
    for trace_file in trace_files:
        try:
            with open(trace_file) as f:
                trace = json.load(f)
            err = (trace.get("error") or "").lower()
            if any(p in err for p in _fatal):
                continue
            if not (trace.get("dom_trace") or {}).get("events"):
                continue
            q = (trace.get("meta") or {}).get("question")
            if q:
                counts[q] += 1
        except Exception:
            pass
    return dict(counts)


# ─── Fatal error detection ────────────────────────────────────────────────────

# Errors that mean every subsequent episode will also fail — abort immediately.
FATAL_API_PATTERNS = [
    "credit balance is too low",
    "insufficient_quota",
    "invalid_api_key",
    "401 Unauthorized",
    "402 Payment",
    "402 This request requires more credits",
]


def check_trace_for_fatal_error(host_out_dir: Path) -> str | None:
    """Scan the newest JSON in host_out_dir for a fatal API error.

    Returns the error string if fatal, None otherwise.
    """
    try:
        traces = sorted(host_out_dir.glob("*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        if not traces:
            return None
        with open(traces[0]) as f:
            trace = json.load(f)
        error = trace.get("error") or ""
        for pattern in FATAL_API_PATTERNS:
            if pattern.lower() in error.lower():
                return error
    except Exception:
        pass
    return None


# ─── Post-run validation ──────────────────────────────────────────────────────

def summarise_run_errors(agents: list, datasets: list, run_ts: str) -> int:
    """Print any traces from this run that have a non-null error. Returns error count."""
    error_count = 0
    for agent, dataset in itertools.product(agents, datasets):
        out_dir = OUTPUT_DIR / agent["agent_id"] / dataset["name"] / agent["harness"] / run_ts
        for trace_file in sorted(out_dir.glob("*.json")):
            try:
                with open(trace_file) as f:
                    trace = json.load(f)
                if trace.get("error"):
                    error_count += 1
                    short = str(trace["error"]).split("\n")[0][:120]
                    print(
                        f"  [ERROR] {agent['agent_id']}/{dataset['name']}/"
                        f"{agent['harness']}/{trace_file.name}: {short}"
                    )
            except Exception:
                pass
    return error_count


def run_episode(
    agent,
    q: dict,
    episode_id,
    output_dir,
    task_timeout_s,
    cleanup_grace_s,
    dataset,
    browser_tmp_run_dir,
    dry_run=False,
) -> tuple[bool, list[str]]:
    episode_tmp_dir = (
        browser_tmp_run_dir / episode_id
        if agent["harness"] == "browser_use"
        else None
    )
    if episode_tmp_dir is not None and not dry_run:
        try:
            episode_tmp_dir.mkdir(parents=True, mode=0o700)
            for child in ("config", "downloads", "profile", "xdg"):
                (episode_tmp_dir / child).mkdir(mode=0o700)
        except OSError as exc:
            return False, [
                f"[✗] {episode_id}",
                f"Unable to prepare browser tmpfs directory: {exc}",
            ]

    cmd = build_apptainer_cmd(
        agent,
        q,
        episode_id,
        output_dir,
        dataset,
        task_timeout_s,
        episode_tmp_dir,
    )
    if dry_run:
        return True, [format_cmd_for_display(cmd)]
    process_timeout_s = task_timeout_s
    if agent["harness"] == "browser_use":
        process_timeout_s += cleanup_grace_s
    process: subprocess.Popen | None = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        with ACTIVE_PROCESSES_LOCK:
            ACTIVE_PROCESSES.add(process)
        try:
            stdout, stderr = process.communicate(timeout=process_timeout_s)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            stdout, stderr = process.communicate()
            return False, [
                f"[TIMEOUT] {episode_id} process exceeded {process_timeout_s}s "
                f"({task_timeout_s}s task + {cleanup_grace_s}s cleanup grace)",
                stderr[-800:],
            ]
        if process.returncode != 0:
            return False, [f"[✗] {episode_id}", stderr[-800:]]
        lines = [line for line in stdout.splitlines() if line.startswith("[")]
        return True, lines
    finally:
        if process is not None:
            with ACTIVE_PROCESSES_LOCK:
                ACTIVE_PROCESSES.discard(process)
        if episode_tmp_dir is not None:
            shutil.rmtree(episode_tmp_dir, ignore_errors=True)


def _terminate_process_group(process: subprocess.Popen, grace_s: float = 5) -> None:
    """Terminate one episode and every descendant that retained its process group."""
    if process.poll() is not None:
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


def _terminate_active_processes(grace_s: float = 5) -> None:
    """Stop all episode process groups, used when the orchestrator is interrupted."""
    with ACTIVE_PROCESSES_LOCK:
        processes = list(ACTIVE_PROCESSES)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in processes):
            return
        time.sleep(0.1)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main():
    global SIF_PATH
    parser = argparse.ArgumentParser(
        description="Run agent tracing experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python orchestrator.py\n"
            "  python orchestrator.py --config my_experiment.yaml\n"
            "  python orchestrator.py --config custom_config.yaml --episodes-per-combo 1 --dry-run\n"
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG,
                        help=f"Experiment config file (default: {DEFAULT_CFG.name})")
    parser.add_argument("--episodes-per-combo", type=int, default=None,
                        help="Override run.episodes_per_combo from the experiment config")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override run.workers from the experiment config (parallel episodes)")
    parser.add_argument("--agents", nargs="+", default=None,
                        help="Run only these agent_ids from the experiment config")
    parser.add_argument("--harnesses", nargs="+", choices=["midscene", "browser_use"], default=None,
                        help="Run only these harnesses from the experiment config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print apptainer commands without executing")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        print(f"Create one based on custom_config.yaml or specify --config <path>")
        sys.exit(1)

    registry = load_registry()
    exp      = load_experiment(args.config)
    run_cfg  = exp.get("run", {})
    configured_sif = Path(run_cfg.get("sif_path", DEFAULT_SIF_PATH.name))
    SIF_PATH = configured_sif if configured_sif.is_absolute() else PROJECT_DIR / configured_sif
    SIF_PATH = SIF_PATH.resolve()

    agents   = resolve_agents(exp, registry)
    datasets = resolve_datasets(exp)
    if args.agents:
        agents = [agent for agent in agents if agent["agent_id"] in set(args.agents)]
    if args.harnesses:
        agents = [agent for agent in agents if agent["harness"] in set(args.harnesses)]
    episodes_per_combo = args.episodes_per_combo or run_cfg.get("episodes_per_combo", 3)
    timeout_s          = run_cfg.get("task_timeout_s", run_cfg.get("timeout_s", 300))
    cleanup_grace_s    = run_cfg.get("cleanup_grace_s", 0)
    workers            = args.workers or run_cfg.get("workers", 1)

    if not agents:
        print("No valid agents in experiment config. Check agent_ids match registry.")
        sys.exit(1)
    if not datasets:
        print("No datasets loaded. Check dataset sources exist (run prep_datasets.py).")
        sys.exit(1)

    if not args.dry_run and not SIF_PATH.exists():
        print(f"{SIF_PATH.name} not found. Build it first:\n"
              f"  apptainer build {SIF_PATH.name} agent.def")
        sys.exit(1)

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    has_browser_use = any(agent["harness"] == "browser_use" for agent in agents)
    browser_tmp_run_dir: Path | None = None
    if has_browser_use:
        configured_tmp_root = Path(
            run_cfg.get(
                "browser_use_tmp_root",
                f"/dev/shm/known-actions-browser-use-{os.getuid()}",
            )
        )
        browser_tmp_root = (
            configured_tmp_root
            if configured_tmp_root.is_absolute()
            else (PROJECT_DIR / configured_tmp_root).resolve()
        )
        browser_tmp_run_dir = browser_tmp_root / (
            f"{run_ts}_{uuid.uuid4().hex[:8]}"
        )
        if not args.dry_run:
            try:
                browser_tmp_root.mkdir(parents=True, mode=0o700, exist_ok=True)
                available_bytes = shutil.disk_usage(browser_tmp_root).free
                minimum_bytes = int(
                    float(run_cfg.get("browser_use_tmp_min_free_gb", 10))
                    * 1024**3
                )
                if available_bytes < minimum_bytes:
                    print(
                        "Insufficient browser-use temporary space: "
                        f"{available_bytes / 1024**3:.1f} GiB available at "
                        f"{browser_tmp_root}, "
                        f"{minimum_bytes / 1024**3:.1f} GiB required."
                    )
                    sys.exit(1)
                browser_tmp_run_dir.mkdir(mode=0o700)
            except OSError as exc:
                print(
                    f"Unable to prepare browser-use temporary storage at "
                    f"{browser_tmp_root}: {exc}"
                )
                sys.exit(1)

    # ── Resume: scan existing traces to find already-completed questions ───────
    print("Scanning existing traces for resume state...")
    completed_counts: dict[tuple, dict[str, int]] = {}
    for agent, dataset in itertools.product(agents, datasets):
        key = (agent["agent_id"], agent["harness"], dataset["name"])
        completed_counts[key] = find_completed_counts(
            agent["agent_id"], dataset["name"], agent["harness"]
        )

    total_q        = sum(len(d["questions"]) for d in datasets)
    total_possible = len(agents) * total_q * episodes_per_combo
    ds_summary     = ", ".join(f"{d['name']}({len(d['questions'])}q)" for d in datasets)
    workers_label  = f"{workers} (parallel)" if workers > 1 else "1 (serial)"

    # ── Build flat job list, skipping already-completed episodes ──────────────
    jobs    = []
    skipped = 0
    for agent, dataset in itertools.product(agents, datasets):
        key           = (agent["agent_id"], agent["harness"], dataset["name"])
        counts        = completed_counts[key]
        container_out = (
            f"/app/workspace/traces/{agent['agent_id']}/{dataset['name']}/"
            f"{agent['harness']}/{run_ts}"
        )
        host_out      = (
            OUTPUT_DIR / agent["agent_id"] / dataset["name"] / agent["harness"] / run_ts
        )
        for q in dataset["questions"]:
            question_str = q["question"] if isinstance(q, dict) else q
            done = counts.get(question_str, 0)
            remaining = episodes_per_combo - done
            if remaining <= 0:
                skipped += episodes_per_combo
                continue
            if done > 0:
                skipped += done
            for _ in range(remaining):
                ep_id = f"{agent['agent_id']}_{agent['harness']}_{uuid.uuid4().hex[:8]}"
                label = (
                    f"{agent['agent_id']} | {agent['harness']} | "
                    f"{dataset['name']} | {question_str[:45]}"
                )
                jobs.append((agent, q, ep_id, container_out, host_out, dataset, label))

    print(f"Config:   {args.config.name}")
    agent_labels = [f"{a['agent_id']}@{a['harness']}" for a in agents]
    print(f"Agents:   {agent_labels}")
    print(f"Datasets: [{ds_summary}]")
    print(f"Episodes: {len(jobs)} to run  ({skipped} skipped — already collected)")
    print(f"Workers:  {workers_label}")
    print(f"Budget:   {timeout_s}s task + {cleanup_grace_s}s browser-use cleanup grace")
    if browser_tmp_run_dir is not None:
        print(f"Temp:     {browser_tmp_run_dir}")
    print()

    if not jobs:
        print("All episodes already collected. Nothing to do.")
        return

    succeeded = failed = 0
    lock        = threading.Lock()
    abort_event = threading.Event()

    def _handle_interrupt(_signum, _frame):
        abort_event.set()
        _terminate_active_processes()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    def _handle_result(ok, lines, label, agent, dataset, host_out):
        nonlocal succeeded, failed
        with lock:
            if ok:
                succeeded += 1
            else:
                failed += 1
        status = "✓" if ok else "✗"
        tqdm.write(f"[{status}] {label}")
        for line in lines:
            tqdm.write(f"    {line}")
        # Fatal API errors can be written alongside a non-zero runner exit.
        if not args.dry_run:
            fatal = check_trace_for_fatal_error(host_out)
            if fatal:
                short = fatal.split("\n")[0][:120]
                tqdm.write(f"\n[FATAL] API error — aborting run:\n  {short}\n")
                abort_event.set()

    def _run_with_abort(
        agent, q, ep_id, out, host_out, timeout_s, cleanup_grace_s, dataset
    ):
        if abort_event.is_set():
            return False, ["[SKIP] Aborted due to fatal error"]
        return run_episode(
            agent,
            q,
            ep_id,
            out,
            timeout_s,
            cleanup_grace_s,
            dataset,
            browser_tmp_run_dir,
            args.dry_run,
        )

    if args.dry_run or workers == 1:
        bar = tqdm(jobs, total=len(jobs), unit="ep", dynamic_ncols=True)
        for agent, q, ep_id, out, host_out, dataset, label in bar:
            if abort_event.is_set():
                break
            bar.set_description(label[:55])
            ok, lines = _run_with_abort(
                agent,
                q,
                ep_id,
                out,
                host_out,
                timeout_s,
                cleanup_grace_s,
                dataset,
            )
            _handle_result(ok, lines, label, agent, dataset, host_out)
            bar.set_postfix(ok=succeeded, fail=failed)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_meta = {
                pool.submit(
                    _run_with_abort,
                    agent,
                    q,
                    ep_id,
                    out,
                    host_out,
                    timeout_s,
                    cleanup_grace_s,
                    ds,
                ):
                    (label, agent, ds, host_out)
                for agent, q, ep_id, out, host_out, ds, label in jobs
            }
            bar = tqdm(as_completed(future_to_meta), total=len(jobs), unit="ep", dynamic_ncols=True)
            for future in bar:
                label, agent, dataset, host_out = future_to_meta[future]
                ok, lines = future.result()
                _handle_result(ok, lines, label, agent, dataset, host_out)
                with lock:
                    bar.set_postfix(ok=succeeded, fail=failed)

    print(f"\nDone: {succeeded} succeeded, {failed} failed")

    if browser_tmp_run_dir is not None and not args.dry_run:
        try:
            browser_tmp_run_dir.rmdir()
            browser_tmp_run_dir.parent.rmdir()
        except OSError:
            # Non-empty run directories are intentionally retained for explicit
            # inspection/cleanup after abnormal episode shutdown.
            pass

    # ── Post-run: report any traces with errors ────────────────────────────────
    if not args.dry_run:
        print("\nValidating collected traces...")
        n_errors = summarise_run_errors(agents, datasets, run_ts)
        if n_errors:
            print(f"  {n_errors} trace(s) recorded errors (see above).")
        else:
            print("  All traces clean.")


if __name__ == "__main__":
    main()
