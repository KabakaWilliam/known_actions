"""
make_tables.py — Generate publication-ready LaTeX table from trace_analyzer results.

Usage:
    python make_tables.py
    python make_tables.py --agents gpt_5_4 qwen3vl_8b uitars_7b
    python make_tables.py --wiki-tag wiki_ood_amazon_deep
    python make_tables.py --in-domain-only --wiki-tag wiki ##preferred with all 9 models


Column sources (defaults):
    2Wiki columns        : wiki_ood_amazon_deep  (test + webshop OOD + deepshop OOD)
    Webshop columns      : webshop_ood_deepshop  (test + deepshop OOD)
    Webshop→2Wiki column : webshop_ood_deepshop_wiki (OOD 2wiki key)

Claude Opus notes:
    - Has NO deepshop traces → marked † in 2Wiki→DeepShop and Webshop→DeepShop columns
    - Was excluded from the webshop experiments (pre-dates its addition) →
      marked † in Webshop, Webshop→2Wiki, Webshop→DeepShop columns

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

# ─── Agent ID aliases ─────────────────────────────────────────────────────────
# Some older experiments stored agent IDs under different names.
AGENT_ALIASES: dict[str, list[str]] = {
    "gpt_5_4": ["gpt54", "gpt_5_4"],
}

# Agents with no deepshop traces (excluded from deepshop-related experiments).
# claude_opus_4_6 has deepshop_ood traces — no longer absent.
DEEPSHOP_ABSENT = frozenset()
# Agents absent from webshop experiments.
# claude_opus_4_6 has webshop train/val/test traces — no longer absent.
WEBSHOP_ABSENT  = frozenset()

# ─── Default experiment tags ──────────────────────────────────────────────────
WIKI_TAG_DEFAULT          = "wiki_ood_amazon_deep"
AMAZON_TAG_DEFAULT        = "webshop_ood_deepshop"
AMAZON_WIKI_TAG_DEFAULT   = "webshop_ood_deepshop_wiki"
FRAMES_TAG_DEFAULT        = "frames_2_wiki"
WIKI_FRAMES_TAG_DEFAULT   = "wiki_2_frames"
DEEPSHOP_TAG_DEFAULT      = "deepshop_2_webshop"
WEBGAMES_TAG_DEFAULT      = "webgames_all_ood"

# Short keys for OOD columns — used by --ood-cols
OOD_COL_KEYS = [
    "wiki_webshop", "wiki_deepshop", "wiki_frames",
    "webshop_wiki", "webshop_deepshop",
    "frames_wiki",
    "deepshop_webshop",
]

# ─── Classifiers shown per agent group ────────────────────────────────────────
CLASSIFIERS = [
    ("RandomForest", "RF"),
    ("XGBoost",      "XGB"),
    ("LSTM",         "LSTM"),
]

# ─── Sentinels ────────────────────────────────────────────────────────────────
_ZERO_WITH_SUPPORT = object()   # agent in experiment but classifier got 0 F1
_DAGGER            = object()   # agent excluded / has no data in this domain


def build_column_defs(wiki_tag: str, amazon_tag: str, amazon_wiki_tag: str,
                      frames_tag: str, wiki_frames_tag: str,
                      deepshop_tag: str, webgames_tag: str) -> list:
    """
    Return COLUMN_DEFS as 6-tuples:
        (col_key, tag, split, ood_key, latex_header, absent_agents)

    col_key: short string identifier used by --ood-cols to filter OOD columns.
             In-domain columns always use col_key="test_*".

    absent_agents: set of agent_ids that show '--‡' when their result is None.
    """
    return [
        # ── In-domain (col_key="test_*", always included unless --in-domain-only
        #    filters to only these) ────────────────────────────────────────────
        ("test_wiki",     wiki_tag,     "test", None,
            r"\makecell{\textbf{2Wiki} \\ \textit{(in-dom.)}}",
            frozenset()),
        ("test_frames",   frames_tag,   "test", None,
            r"\makecell{\textbf{FRAMES} \\ \textit{(in-dom.)}}",
            frozenset()),
        ("test_webshop",  amazon_tag,   "test", None,
            r"\makecell{\textbf{Webshop} \\ \textit{(in-dom.)}}",
            WEBSHOP_ABSENT),
        ("test_deepshop", deepshop_tag, "test", None,
            r"\makecell{\textbf{DeepShop} \\ \textit{(in-dom.)}}",
            DEEPSHOP_ABSENT),
        ("test_webgames", webgames_tag, "test", None,
            r"\makecell{\textbf{WebGames} \\ \textit{(in-dom.)}}",
            frozenset()),
        # ── OOD ───────────────────────────────────────────────────────────────
        ("wiki_webshop",    wiki_tag,        "ood", "webshop",
            r"\makecell{\textbf{2Wiki$\to$Webshop} \\ \textit{(OOD)}}",
            frozenset()),
        ("wiki_deepshop",   wiki_tag,        "ood", "deepshop",
            r"\makecell{\textbf{2Wiki$\to$DeepShop} \\ \textit{(OOD)}}",
            DEEPSHOP_ABSENT),
        ("wiki_frames",     wiki_frames_tag, "ood", "frames",
            r"\makecell{\textbf{2Wiki$\to$FRAMES} \\ \textit{(OOD)}}",
            frozenset()),
        ("webshop_wiki",    amazon_wiki_tag, "ood", "2wikimultihop",
            r"\makecell{\textbf{Webshop$\to$2Wiki} \\ \textit{(OOD)}}",
            WEBSHOP_ABSENT),
        ("webshop_deepshop", amazon_tag,     "ood", "deepshop",
            r"\makecell{\textbf{Webshop$\to$DeepShop} \\ \textit{(OOD)}}",
            WEBSHOP_ABSENT | DEEPSHOP_ABSENT),
        ("frames_wiki",     frames_tag,      "ood", "2wikimultihop",
            r"\makecell{\textbf{FRAMES$\to$2Wiki} \\ \textit{(OOD)}}",
            frozenset()),
        ("deepshop_webshop", deepshop_tag,   "ood", "webshop",
            r"\makecell{\textbf{DeepShop$\to$Webshop} \\ \textit{(OOD)}}",
            DEEPSHOP_ABSENT),
    ]


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_results(traces_dir: Path) -> dict:
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


def _resolve_agent_f1(report, agent_id: str) -> float | None:
    """Look up agent_id in report, trying AGENT_ALIASES if the primary key is missing."""
    for key in AGENT_ALIASES.get(agent_id, [agent_id]):
        entry = report.get(key)
        if isinstance(entry, dict):
            return entry
    return None


def agent_f1(report, agent_id: str, absent_agents: frozenset = frozenset()):
    """
    Return the F1 value for agent_id in report, or a sentinel:
      _DAGGER            — agent is in absent_agents and has no result (excluded by design)
      _ZERO_WITH_SUPPORT — agent ran but got 0 F1 (classifier failure)
      None               — no data, unknown reason
    """
    if not report:
        return _DAGGER if agent_id in absent_agents else None
    entry = _resolve_agent_f1(report, agent_id)
    if entry is None:
        return _DAGGER if agent_id in absent_agents else None
    if not isinstance(entry, dict):
        return _DAGGER if agent_id in absent_agents else None
    if entry.get("support", 1) == 0:
        return None
    f1 = entry.get("f1-score", 0.0)
    if f1 == 0.0:
        return _ZERO_WITH_SUPPORT
    return f1


def fmt(x) -> str:
    if x is None:
        return "--"
    if x is _DAGGER:
        return r"--$^{\ddagger}$"
    if x is _ZERO_WITH_SUPPORT:
        return r"0.00$^{\dagger}$"
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
            raw = r[c]
            # strip any LaTeX markup to get the number
            stripped = raw.replace(r"\textbf{", "").replace("}", "").split("$")[0]
            try:
                nums.append(float(stripped))
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

def clf_rows(results_map: dict, get_f1_fn, column_defs: list) -> list:
    """Return one formatted row per classifier, with bold applied per column."""
    raw = []
    for clf_key, _ in CLASSIFIERS:
        row = []
        for _, tag, split, ood_key, _, absent_agents in column_defs:
            rep = get_report(results_map, tag, clf_key, split, ood_key)
            row.append(fmt(get_f1_fn(rep, absent_agents)))
        raw.append(row)
    return bold_best_per_col(raw)


def agent_has_results(results_map: dict, agent_id: str, column_defs: list) -> bool:
    """Return True if the agent has at least one non-None F1 value across all columns."""
    for clf_key, _ in CLASSIFIERS:
        for _, tag, split, ood_key, _, absent_agents in column_defs:
            if agent_id in absent_agents:
                continue
            rep = get_report(results_map, tag, clf_key, split, ood_key)
            v = agent_f1(rep, agent_id, absent_agents)
            if v is not None and v is not _DAGGER:
                return True
    return False


def render_group(model_label: str, rows: list, lines: list):
    for i, ((_, clf_disp), row) in enumerate(zip(CLASSIFIERS, rows)):
        prefix = f"{model_label:<20}" if i == 0 else " " * 20
        cells  = " & ".join(row)
        lines.append(f"{prefix} & {clf_disp:<4} & {cells} \\\\")


def make_table(results_map: dict, agents: list, column_defs: list) -> str:
    n_cond   = len(column_defs)
    n_total  = n_cond + 2   # + Model + Clf.

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
        r" Best F1 per model group in \textbf{bold}."
        r" $^\dagger$~zero F1 despite training presence;"
        r" $^\ddagger$~agent excluded (no traces in this domain).}",
        r"\label{tab:main}",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l l " + "c " * n_cond + "}",
        r"\toprule",
    ]

    # Header
    col_headers = " & ".join(h for _, _, _, _, h, _ in column_defs)
    lines.append(r"\textbf{Model} & \textbf{Clf.} & " + col_headers + r" \\")
    lines.append(r"\midrule")

    proprietary = [a for a in agents if a["source"] == "proprietary"]
    open_src    = [a for a in agents if a["source"] == "open"]

    for grp_label, color, grp_agents in [
        ("Proprietary Models", "headergreen", proprietary),
        ("Open-Source Models", "headerblue",  open_src),
    ]:
        active = [a for a in grp_agents
                  if agent_has_results(results_map, a["agent_id"], column_defs)]
        if not active:
            continue
        lines.append(rf"\rowcolor{{{color}}}")
        lines.append(
            rf"\multicolumn{{{n_total}}}{{l}}{{\textbf{{\textit{{{grp_label}}}}}}}" + r" \\"
        )
        lines.append(r"\midrule")

        for agent in active:
            aid, disp = agent["agent_id"], agent["display_name"]
            rows = clf_rows(
                results_map,
                lambda rep, absent, a=aid: agent_f1(rep, a, absent),
                column_defs,
            )
            render_group(disp, rows, lines)
            lines.append(r"\midrule")

    # All Models section — only include agents with actual results
    agent_ids = [a["agent_id"] for a in agents
                 if agent_has_results(results_map, a["agent_id"], column_defs)]
    n_str = f" ({len(agent_ids)} models)" if agent_ids else ""
    lines.append(r"\rowcolor{headergreen}")
    lines.append(
        rf"\multicolumn{{{n_total}}}{{l}}{{\textbf{{\textit{{All Models{n_str}}}}}}}" + r" \\"
    )
    lines.append(r"\midrule")
    rows = clf_rows(
        results_map,
        lambda rep, absent: filtered_macro_f1(rep, agent_ids),
        column_defs,
    )
    render_group("All", rows, lines)

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table from trace_analyzer results.",
    )
    parser.add_argument("traces_dir", nargs="?", type=Path, default=Path("./traces"))
    parser.add_argument("--agents", nargs="+", default=None, metavar="AGENT_ID")
    parser.add_argument("--wiki-tag",         default=WIKI_TAG_DEFAULT)
    parser.add_argument("--amazon-tag",       default=AMAZON_TAG_DEFAULT)
    parser.add_argument("--amazon-wiki-tag",  default=AMAZON_WIKI_TAG_DEFAULT,
                        help="Tag for the Webshop→2Wiki column "
                             f"(default: {AMAZON_WIKI_TAG_DEFAULT})")
    parser.add_argument("--frames-tag",       default=FRAMES_TAG_DEFAULT,
                        help=f"Tag for FRAMES in-domain + frames_wiki OOD columns (default: {FRAMES_TAG_DEFAULT})")
    parser.add_argument("--wiki-frames-tag",  default=WIKI_FRAMES_TAG_DEFAULT,
                        help=f"Tag for 2Wiki→FRAMES OOD column (default: {WIKI_FRAMES_TAG_DEFAULT})")
    parser.add_argument("--deepshop-tag",     default=DEEPSHOP_TAG_DEFAULT,
                        help=f"Tag for DeepShop in-domain + deepshop_webshop OOD columns (default: {DEEPSHOP_TAG_DEFAULT})")
    parser.add_argument("--webgames-tag",     default=WEBGAMES_TAG_DEFAULT,
                        help=f"Tag for the WebGames in-domain column (default: {WEBGAMES_TAG_DEFAULT})")
    parser.add_argument("--in-domain-only", action="store_true", default=False,
                        help="Only include in-domain (test split) columns — drops all OOD columns.")
    parser.add_argument("--ood-cols", nargs="+", default=None, metavar="COL_KEY",
                        choices=OOD_COL_KEYS,
                        help="OOD columns to include. Choices: " + ", ".join(OOD_COL_KEYS) +
                             ". Default: all. Ignored when --in-domain-only is set.")
    cli = parser.parse_args()

    column_defs = build_column_defs(
        cli.wiki_tag, cli.amazon_tag, cli.amazon_wiki_tag,
        cli.frames_tag, cli.wiki_frames_tag,
        cli.deepshop_tag, cli.webgames_tag,
    )
    if cli.in_domain_only:
        column_defs = [c for c in column_defs if c[2] == "test"]
    elif cli.ood_cols is not None:
        col_map = {c[0]: c for c in column_defs}
        column_defs = [col_map[k] for k in cli.ood_cols if k in col_map]

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
            print(f"Warning: agents not in config.yaml: {sorted(missing)}")
    if not agents:
        print("Warning: no agents to include — per-agent rows will be omitted.")

    table = make_table(results_map, agents, column_defs)

    suffix = "_in_domain" if cli.in_domain_only else "_main"
    out = cli.traces_dir / "models" / f"table{suffix}.tex"
    out.write_text(table + "\n")
    print(table)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
