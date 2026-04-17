import argparse, datetime, itertools, json, os, random, re, subprocess, sys, threading, uuid
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
SIF_PATH      = PROJECT_DIR / "agent.sif"
OUTPUT_DIR    = PROJECT_DIR / "traces"
TEMPLATES_DIR = PROJECT_DIR / "task_prompt_templates"


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
    """Merge experiment agent list with registry definitions.

    For each agent_id in the experiment, look up its env from the registry,
    apply global midscene_defaults, then apply any per-agent env overrides
    from the experiment config.
    """
    reg_by_id   = {a["agent_id"]: a for a in registry.get("agents", [])}
    defaults    = {k: str(v) for k, v in registry.get("midscene_defaults", {}).items()}

    agents = []
    for entry in exp.get("agents", []):
        agent_id = entry["agent_id"]
        if agent_id not in reg_by_id:
            print(f"[WARN] agent_id '{agent_id}' not found in registry ({REGISTRY.name}). Skipping.")
            continue

        # 1. global defaults  2. registry env  3. experiment overrides
        env = {**defaults}
        for k, v in reg_by_id[agent_id].get("env", {}).items():
            env[k] = _expand_env(str(v))
        for k, v in entry.get("env", {}).items():
            env[k] = _expand_env(str(v))

        agents.append({"agent_id": agent_id, "env": env})

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


def build_apptainer_cmd(agent, q: dict, episode_id, output_dir: str, dataset: dict) -> list[str]:
    question = q["question"]
    # Per-question start_url (e.g. WebGames) overrides dataset-level start_url
    start_url = q.get("start_url") or dataset["start_url"]
    cmd = [
        "apptainer", "exec",
        "--bind", f"{PROJECT_DIR}:/app/workspace",
        "--no-home",
        "--pwd", "/app",
    ]
    for key, value in agent["env"].items():
        cmd += ["--env", f"{key}={value}"]
    cmd += [
        str(SIF_PATH),
        "npx", "tsx", "/app/workspace/agent_runner.ts",
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
    if expected_answer:
        cmd += ["--expected-answer", expected_answer]
    return cmd


# ─── Resume helpers ───────────────────────────────────────────────────────────

def find_completed_counts(agent_id: str, dataset_name: str) -> dict[str, int]:
    """Return {question: n_valid_traces} for already-collected episodes.

    Only counts traces where error is null (successful collection).
    Used to skip questions that already have enough reps.
    """
    dataset_dir = OUTPUT_DIR / agent_id / dataset_name
    if not dataset_dir.exists():
        return {}
    counts: dict[str, int] = defaultdict(int)
    for trace_file in dataset_dir.glob("*/*.json"):
        try:
            with open(trace_file) as f:
                trace = json.load(f)
            if trace.get("error") is None:
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
        out_dir = OUTPUT_DIR / agent["agent_id"] / dataset["name"] / run_ts
        for trace_file in sorted(out_dir.glob("*.json")):
            try:
                with open(trace_file) as f:
                    trace = json.load(f)
                if trace.get("error"):
                    error_count += 1
                    short = str(trace["error"]).split("\n")[0][:120]
                    print(f"  [ERROR] {agent['agent_id']}/{dataset['name']}/{trace_file.name}: {short}")
            except Exception:
                pass
    return error_count


def run_episode(agent, q: dict, episode_id, output_dir, timeout_s, dataset, dry_run=False) -> tuple[bool, list[str]]:
    cmd = build_apptainer_cmd(agent, q, episode_id, output_dir, dataset)
    if dry_run:
        return True, [" ".join(cmd)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if result.returncode != 0:
            return False, [f"[✗] {episode_id}", result.stderr[-800:]]
        lines = [l for l in result.stdout.splitlines() if l.startswith("[")]
        return True, lines
    except subprocess.TimeoutExpired:
        return False, [f"[TIMEOUT] {episode_id} after {timeout_s}s"]


def main():
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

    agents   = resolve_agents(exp, registry)
    datasets = resolve_datasets(exp)
    episodes_per_combo = args.episodes_per_combo or run_cfg.get("episodes_per_combo", 3)
    timeout_s          = run_cfg.get("timeout_s", 300)
    workers            = args.workers or run_cfg.get("workers", 1)

    if not agents:
        print("No valid agents in experiment config. Check agent_ids match registry.")
        sys.exit(1)
    if not datasets:
        print("No datasets loaded. Check dataset sources exist (run prep_datasets.py).")
        sys.exit(1)

    if not args.dry_run and not SIF_PATH.exists():
        print("agent.sif not found. Build it first:\n  apptainer build agent.sif agent.def")
        sys.exit(1)

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resume: scan existing traces to find already-completed questions ───────
    print("Scanning existing traces for resume state...")
    completed_counts: dict[tuple, dict[str, int]] = {}
    for agent, dataset in itertools.product(agents, datasets):
        key = (agent["agent_id"], dataset["name"])
        completed_counts[key] = find_completed_counts(agent["agent_id"], dataset["name"])

    total_q        = sum(len(d["questions"]) for d in datasets)
    total_possible = len(agents) * total_q * episodes_per_combo
    ds_summary     = ", ".join(f"{d['name']}({len(d['questions'])}q)" for d in datasets)
    workers_label  = f"{workers} (parallel)" if workers > 1 else "1 (serial)"

    # ── Build flat job list, skipping already-completed episodes ──────────────
    jobs    = []
    skipped = 0
    for agent, dataset in itertools.product(agents, datasets):
        key           = (agent["agent_id"], dataset["name"])
        counts        = completed_counts[key]
        container_out = f"/app/workspace/traces/{agent['agent_id']}/{dataset['name']}/{run_ts}"
        host_out      = OUTPUT_DIR / agent["agent_id"] / dataset["name"] / run_ts
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
                ep_id = f"{agent['agent_id']}_{uuid.uuid4().hex[:8]}"
                label = f"{agent['agent_id']} | {dataset['name']} | {question_str[:45]}"
                jobs.append((agent, q, ep_id, container_out, host_out, dataset, label))

    print(f"Config:   {args.config.name}")
    print(f"Agents:   {[a['agent_id'] for a in agents]}")
    print(f"Datasets: [{ds_summary}]")
    print(f"Episodes: {len(jobs)} to run  ({skipped} skipped — already collected)")
    print(f"Workers:  {workers_label}")
    print()

    if not jobs:
        print("All episodes already collected. Nothing to do.")
        return

    succeeded = failed = 0
    lock        = threading.Lock()
    abort_event = threading.Event()

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
        # Fatal error check — fires after every successful subprocess call
        if ok and not args.dry_run:
            fatal = check_trace_for_fatal_error(host_out)
            if fatal:
                short = fatal.split("\n")[0][:120]
                tqdm.write(f"\n[FATAL] API error — aborting run:\n  {short}\n")
                abort_event.set()

    def _run_with_abort(agent, q, ep_id, out, host_out, timeout_s, dataset):
        if abort_event.is_set():
            return False, ["[SKIP] Aborted due to fatal error"]
        return run_episode(agent, q, ep_id, out, timeout_s, dataset, args.dry_run)

    if args.dry_run or workers == 1:
        bar = tqdm(jobs, total=len(jobs), unit="ep", dynamic_ncols=True)
        for agent, q, ep_id, out, host_out, dataset, label in bar:
            if abort_event.is_set():
                break
            bar.set_description(label[:55])
            ok, lines = _run_with_abort(agent, q, ep_id, out, host_out, timeout_s, dataset)
            _handle_result(ok, lines, label, agent, dataset, host_out)
            bar.set_postfix(ok=succeeded, fail=failed)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_meta = {
                pool.submit(_run_with_abort, agent, q, ep_id, out, host_out, timeout_s, ds):
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
