import argparse, datetime, itertools, json, os, random, re, subprocess, sys, uuid
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

        datasets.append({"name": name, "questions": questions})

    return datasets


def build_apptainer_cmd(agent, question, episode_id, output_dir: str) -> list[str]:
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
    ]
    return cmd


def run_episode(agent, question, episode_id, output_dir, timeout_s, dry_run=False) -> bool:
    cmd = build_apptainer_cmd(agent, question, episode_id, output_dir)
    if dry_run:
        print(" ".join(cmd))
        return True
    print(f"[→] {agent['agent_id']} | {episode_id} | {question[:55]}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if result.returncode != 0:
            print(result.stderr[-800:])
            return False
        for line in result.stdout.splitlines():
            if line.startswith("["):
                print(line)
        return True
    except subprocess.TimeoutExpired:
        print(f"[ERROR] {episode_id} timed out after {timeout_s}s")
        return False


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
    print(f"Config:   {args.config.name}")
    print(f"Agents:   {[a['agent_id'] for a in agents]}")
    print(f"Datasets: [{ds_summary}]")
    print(f"Episodes: {total_episodes}  ({len(agents)} agents × {total_q} questions × {episodes_per_combo} reps)")
    print()

    succeeded = failed = 0
    combos = list(itertools.product(agents, datasets))
    bar = tqdm(total=total_episodes, unit="ep", dynamic_ncols=True)

    for agent, dataset in combos:
        container_out = f"/app/workspace/traces/{agent['agent_id']}/{dataset['name']}/{run_ts}"
        for q in dataset["questions"]:
            for _ in range(episodes_per_combo):
                ep_id = f"{agent['agent_id']}_{uuid.uuid4().hex[:8]}"
                bar.set_description(f"{agent['agent_id']} | {dataset['name']} | {q['question'][:40]}")
                ok = run_episode(agent, q["question"], ep_id, container_out, timeout_s, args.dry_run)
                if ok:
                    succeeded += 1
                else:
                    failed += 1
                bar.set_postfix(ok=succeeded, fail=failed)
                bar.update(1)

    bar.close()
    print(f"\nDone: {succeeded} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
