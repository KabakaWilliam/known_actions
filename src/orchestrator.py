import argparse, datetime, itertools, json, os, random, re, subprocess, sys, threading, uuid
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

PROJECT_DIR  = Path(__file__).parent.resolve()
REGISTRY     = PROJECT_DIR / "config.yaml"           # agent + loader registry
DEFAULT_CFG  = PROJECT_DIR / "custom_config.yaml"    # default experiment spec
SIF_PATH     = PROJECT_DIR / "agent.sif"
OUTPUT_DIR   = PROJECT_DIR / "traces"

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

        datasets.append({
            "name":                  name,
            "questions":             questions,
            "start_url":             dcfg.get("start_url", "https://en.wikipedia.org"),
            "task_prompt_template":  dcfg.get("task_prompt_template"),  # None → agent_runner uses default
            "task_type":             dcfg.get("task_type", "qa"),
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
        cmd += ["--task_prompt", template.replace("{question}", question)]
    expected_answer = q.get("answer")
    if expected_answer:
        cmd += ["--expected-answer", expected_answer]
    return cmd


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

    total_q       = sum(len(d["questions"]) for d in datasets)
    total_episodes = len(agents) * total_q * episodes_per_combo
    ds_summary    = ", ".join(f"{d['name']}({len(d['questions'])}q)" for d in datasets)
    workers_label = f"{workers} (parallel)" if workers > 1 else "1 (serial)"
    print(f"Config:   {args.config.name}")
    print(f"Agents:   {[a['agent_id'] for a in agents]}")
    print(f"Datasets: [{ds_summary}]")
    print(f"Episodes: {total_episodes}  ({len(agents)} agents × {total_q} questions × {episodes_per_combo} reps)")
    print(f"Workers:  {workers_label}")
    print()

    # Pre-build flat job list
    jobs = []
    for agent, dataset in itertools.product(agents, datasets):
        container_out = f"/app/workspace/traces/{agent['agent_id']}/{dataset['name']}/{run_ts}"
        for q in dataset["questions"]:
            for _ in range(episodes_per_combo):
                ep_id = f"{agent['agent_id']}_{uuid.uuid4().hex[:8]}"
                label = f"{agent['agent_id']} | {dataset['name']} | {q['question'][:45]}"
                jobs.append((agent, q, ep_id, container_out, dataset, label))

    succeeded = failed = 0
    lock = threading.Lock()

    if args.dry_run or workers == 1:
        bar = tqdm(jobs, total=len(jobs), unit="ep", dynamic_ncols=True)
        for agent, q, ep_id, out, dataset, label in bar:
            bar.set_description(label[:55])
            ok, lines = run_episode(agent, q, ep_id, out, timeout_s, dataset, args.dry_run)
            if ok:
                succeeded += 1
            else:
                failed += 1
            bar.set_postfix(ok=succeeded, fail=failed)
            for line in lines:
                tqdm.write(line)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_label = {
                pool.submit(run_episode, agent, q, ep_id, out, timeout_s, ds): label
                for agent, q, ep_id, out, ds, label in jobs  # q is now the full question dict
            }
            bar = tqdm(as_completed(future_to_label), total=len(jobs), unit="ep", dynamic_ncols=True)
            for future in bar:
                label = future_to_label[future]
                ok, lines = future.result()
                with lock:
                    if ok:
                        succeeded += 1
                    else:
                        failed += 1
                    bar.set_postfix(ok=succeeded, fail=failed)
                status = "✓" if ok else "✗"
                tqdm.write(f"[{status}] {label}")
                for line in lines:
                    tqdm.write(f"    {line}")

    print(f"\nDone: {succeeded} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
