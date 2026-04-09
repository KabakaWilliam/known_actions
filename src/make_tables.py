"""
make_tables.py — Generate publication-ready LaTeX table from trace_analyzer results.

Usage:
    python make_tables.py [traces_dir] [--agents agent_id ...]

Examples:
    python make_tables.py                                      # all agents with display_name
    python make_tables.py --agents gpt_5_4 qwen3vl_8b uitars_7b # subset only
    python make_tables.py ./traces --agents gpt_5_4 glm_4.6v_flash

Requires in LaTeX preamble:
    \\usepackage[table]{xcolor}
    \\usepackage{booktabs,makecell,colortbl}
    \\definecolor{headergreen}{RGB}{198,224,180}
    \\definecolor{headerblue}{RGB}{189,215,238}

Outputs:
    traces/models/table_main.tex
"""

import argparse, json, sys
from pathlib import Path

try:
    import yaml as _yaml
    _YAML = True
except ImportError:
    _YAML = False

CONFIG_PATH = Path(__file__).parent / "config.yaml"

# ─── Column definitions ───────────────────────────────────────────────────────
# Built dynamically from --wiki-tag / --amazon-tag at runtime.
# Defaults match the most common experiment tag names.
WIKI_TAG_DEFAULT   = "wiki_ood_amazon"
AMAZON_TAG_DEFAULT = "webshop_ood_deepshop"

def build_column_defs(wiki_tag: str, amazon_tag: str) -> list:
    """Return COLUMN_DEFS with the given experiment tags substituted in."""
    return [
        (wiki_tag,   "test", None,
            r"\makecell{\textbf{2Wiki} \\ \textit{(in-dom.)}}"),
        (wiki_tag,   "ood",  "webshop",
            r"\makecell{\textbf{2Wiki$\to$Webshop} \\ \textit{(OOD)}}"),
        (wiki_tag,   "ood",  "deepshop",
            r"\makecell{\textbf{2Wiki$\to$DeepShop} \\ \textit{(hard OOD)}}"),
        (amazon_tag, "test", None,
            r"\makecell{\textbf{Webshop} \\ \textit{(in-dom.)}}"),
        (amazon_tag, "ood",  "2wikimultihop",
            r"\makecell{\textbf{Webshop$\to$2Wiki} \\ \textit{(OOD)}}"),
        (amazon_tag, "ood",  "deepshop",
            r"\makecell{\textbf{Webshop$\to$DeepShop} \\ \textit{(hard OOD)}}"),
    ]

N_COND_COLS  = 6
N_TOTAL_COLS = N_COND_COLS + 2  # + Model + Clf.

# Classifiers shown per agent group (key in results.json → display label)
CLASSIFIERS = [
    ("RandomForest", "RF"),
    ("XGBoost",      "XGB"),
    ("LSTM",         "LSTM"),
]


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_results(traces_dir: Path) -> dict:
    """Load all results.json files found under traces_dir/models/."""
    results_map = {}
    for path in (traces_dir / "models").glob("*/results.json"):
        tag = path.parent.name
        try:
            with open(path) as f:
                results_map[tag] = json.load(f)
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")
    return results_map


def load_agents(config_path: Path) -> list:
    """
    Read agents from config.yaml.
    Returns list of {agent_id, display_name, source} for agents that have
    display_name set, in config order (proprietary first, then open).
    """
    if not _YAML or not config_path.exists():
        return []
    with open(config_path) as f:
        cfg = _yaml.safe_load(f)
    agents = []
    for a in cfg.get("agents", []):
        if "display_name" not in a:
            continue
        agents.append({
            "agent_id":     a["agent_id"],
            "display_name": a["display_name"],
            "source":       a.get("source", "open"),
        })
    # Stable sort: proprietary before open, preserving config order within each group
    agents.sort(key=lambda a: 0 if a["source"] == "proprietary" else 1)
    return agents


# ─── Report accessors ─────────────────────────────────────────────────────────

def get_report(results_map: dict, tag: str, clf_key: str,
               split: str, ood_key: str | None = None):
    """Return the classification report dict, or None if not available."""
    if tag not in results_map:
        return None
    m = results_map[tag].get("models", {}).get(clf_key)
    if m is None:
        return None
    if split == "ood":
        ood = m.get("ood_reports") or {}
        return ood.get(ood_key) if ood_key else next(iter(ood.values()), None)
    return m.get(f"{split}_report")


def macro_f1(report) -> float | None:
    if not report:
        return None
    return (report.get("macro avg") or {}).get("f1-score")


def filtered_macro_f1(report, agent_ids: list) -> float | None:
    """Recompute macro F1 from only the specified agents' per-class F1 scores."""
    if not report or not agent_ids:
        return macro_f1(report)
    f1s = [
        report[aid]["f1-score"]
        for aid in agent_ids
        if isinstance(report.get(aid), dict) and "f1-score" in report[aid]
    ]
    return sum(f1s) / len(f1s) if f1s else None


_ZERO_WITH_SUPPORT = object()  # sentinel: agent ran but classifier got 0 correct


def agent_f1(report, agent_id: str):
    if not report:
        return None
    entry = report.get(agent_id)
    if not isinstance(entry, dict):
        return None
    if entry.get("support", 1) == 0:
        return None  # agent has no episodes in this split → not a real result
    f1 = entry.get("f1-score", 0.0)
    if f1 == 0.0:
        return _ZERO_WITH_SUPPORT  # ran but completely misclassified → dagger
    return f1


def fmt(x) -> str:
    if x is None:
        return "--"
    if x is _ZERO_WITH_SUPPORT:
        return r"0.00$^\dagger$"
    return f"{x * 100:.2f}"


# ─── Bolding ──────────────────────────────────────────────────────────────────

def bold_best_per_col(rows: list) -> list:
    """Bold the maximum numeric value per column across classifier rows."""
    if not rows:
        return rows
    result = [list(r) for r in rows]
    for c in range(len(rows[0])):
        nums = []
        for r in rows:
            try:
                nums.append(float(r[c]))
            except (ValueError, TypeError):
                nums.append(None)
        valid = [v for v in nums if v is not None]
        if not valid:
            continue
        best = max(valid)
        for ri, num in enumerate(nums):
            if num is not None and num >= best - 1e-6:
                result[ri][c] = r"\textbf{" + result[ri][c] + "}"
    return result


# ─── Table building ───────────────────────────────────────────────────────────

def clf_rows(results_map: dict, get_f1, column_defs: list) -> list:
    """Return one formatted row per classifier, with bold applied per column."""
    raw = []
    for clf_key, _ in CLASSIFIERS:
        row = []
        for tag, split, ood_key, _ in column_defs:
            rep = get_report(results_map, tag, clf_key, split, ood_key)
            row.append(fmt(get_f1(rep)))
        raw.append(row)
    return bold_best_per_col(raw)


def render_group(model_label: str, rows: list, lines: list):
    """Append LaTeX rows for a model group (RF / XGB / LSTM)."""
    for i, ((_, clf_disp), row) in enumerate(zip(CLASSIFIERS, rows)):
        prefix = f"{model_label:<16}" if i == 0 else " " * 16
        cells  = " & ".join(row)
        lines.append(f"{prefix} & {clf_disp:<4} & {cells} \\\\")


def make_table(results_map: dict, agents: list, column_defs: list) -> str:
    lines: list = []

    lines += [
        r"% Requires in LaTeX preamble:",
        r"% \usepackage[table]{xcolor}",
        r"% \usepackage{booktabs,makecell,colortbl}",
        r"% \definecolor{headergreen}{RGB}{198,224,180}",
        r"% \definecolor{headerblue}{RGB}{189,215,238}",
        r"",
        r"\begin{table}[t]",
        r"\caption{Agent identification macro F1 (\%) across datasets and classifiers."
        r" Best F1 per model group in \textbf{bold}.}",
        r"\label{tab:main}",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{7pt}",
        r"\resizebox{0.97\textwidth}{!}{%",
        r"\begin{tabular}{l l " + "c " * N_COND_COLS + "}",
        r"\toprule",
    ]

    # Header
    col_headers = "\n& ".join(h for _, _, _, h in column_defs)
    lines.append(r"\textbf{Model} & \textbf{Clf.}" + "\n& " + col_headers + r" \\")
    lines.append(r"\midrule")

    # Per-source agent groups
    proprietary = [a for a in agents if a["source"] == "proprietary"]
    open_src    = [a for a in agents if a["source"] == "open"]

    for grp_label, color, grp_agents in [
        ("Proprietary Models", "headergreen", proprietary),
        ("Open-Source Models", "headerblue",  open_src),
    ]:
        if not grp_agents:
            continue
        lines.append(rf"\rowcolor{{{color}}}")
        lines.append(
            rf"\multicolumn{{{N_TOTAL_COLS}}}{{l}}{{\textbf{{\textit{{{grp_label}}}}}}}" + r" \\"
        )
        lines.append(r"\midrule")

        for agent in grp_agents:
            aid, disp = agent["agent_id"], agent["display_name"]
            rows = clf_rows(results_map, lambda rep, a=aid: agent_f1(rep, a), column_defs)
            render_group(disp, rows, lines)
            lines.append(r"\midrule")

    # All Models section — macro F1 recomputed from included agents only
    agent_ids = [a["agent_id"] for a in agents]
    n_str = f" ({len(agent_ids)} models)" if agent_ids else ""
    lines.append(r"\rowcolor{headergreen}")
    lines.append(
        rf"\multicolumn{{{N_TOTAL_COLS}}}{{l}}{{\textbf{{\textit{{All Models{n_str}}}}}}}" + r" \\"
    )
    lines.append(r"\midrule")
    rows = clf_rows(results_map, lambda rep: filtered_macro_f1(rep, agent_ids), column_defs)
    render_group("All", rows, lines)

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"{\footnotesize $^\dagger$ Agent has traces in this domain but the classifier"
        r" assigns zero correct predictions (complete cross-domain failure).}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table from trace_analyzer results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python make_tables.py\n"
            "  python make_tables.py --agents gpt_5_4 qwen3vl_8b uitars_7b\n"
            "  python make_tables.py ./traces --agents gpt_5_4 glm_4.6v_flash"
        ),
    )
    parser.add_argument("traces_dir", nargs="?", type=Path, default=Path("./traces"),
                        help="Root traces directory (default: ./traces)")
    parser.add_argument("--agents", nargs="+", default=None, metavar="AGENT_ID",
                        help="Agent IDs to include (default: all agents with display_name in config.yaml). "
                             "Filters per-agent rows and recomputes All Models macro F1 accordingly.")
    parser.add_argument("--wiki-tag", default=WIKI_TAG_DEFAULT, metavar="TAG",
                        help=f"Experiment tag for Wikipedia-trained results (default: {WIKI_TAG_DEFAULT})")
    parser.add_argument("--amazon-tag", default=AMAZON_TAG_DEFAULT, metavar="TAG",
                        help=f"Experiment tag for Amazon-trained results (default: {AMAZON_TAG_DEFAULT})")
    cli = parser.parse_args()

    column_defs = build_column_defs(cli.wiki_tag, cli.amazon_tag)

    results_map = load_results(cli.traces_dir)
    if not results_map:
        print(f"No results.json files found under {cli.traces_dir}/models/")
        sys.exit(1)

    agents = load_agents(CONFIG_PATH)
    if cli.agents is not None:
        requested = set(cli.agents)
        agents = [a for a in agents if a["agent_id"] in requested]
        missing = requested - {a["agent_id"] for a in agents}
        if missing:
            print(f"Warning: agents not found in config.yaml with display_name: {sorted(missing)}")
    if not agents:
        print("Warning: no agents to include — per-agent rows will be omitted.")

    table = make_table(results_map, agents, column_defs)

    out = cli.traces_dir / "models" / "table_main.tex"
    out.write_text(table + "\n")
    print(table)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
