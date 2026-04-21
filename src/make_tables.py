"""
make_tables.py — Generate publication-ready LaTeX table from trace_analyzer results.

OOD columns are specified as source→target pairs.  The experiment tag is always
derived as  {source}_ood_all  (overridable per-dataset with --tag-override).

Usage:
    python make_tables.py
    python make_tables.py --in-domain-only
    python make_tables.py --ood-pairs wiki:frames wiki:webshop webshop:wiki
    python make_tables.py --classifiers RF XGB LSTM Lasso LR
    python make_tables.py --tag-override wiki=wiki_custom_tag

Requires in LaTeX preamble:
    \\usepackage[table]{xcolor}
    \\usepackage{booktabs,makecell,colortbl}
    \\definecolor{headergreen}{RGB}{198,224,180}
    \\definecolor{headerblue}{RGB}{189,215,238}

Outputs:
    traces/classifiers/table_main.tex  (or table_in_domain.tex)
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
AGENT_ALIASES: dict[str, list[str]] = {
    "gpt_5_4": ["gpt54", "gpt_5_4"],
}

# ─── Dataset registry ─────────────────────────────────────────────────────────
# short_name → (latex_display, ood_report_key_in_results_json, absent_agents)
#
# ood_report_key is the key used inside results.json ood_reports{}:
#   wiki uses "2wikimultihop" but the experiment tag uses "wiki_ood_all"
DATASETS: dict[str, tuple[str, str, frozenset]] = {
    "wiki":     ("2Wiki",    "2wikimultihop", frozenset()),
    "frames":   ("FRAMES",   "frames",        frozenset()),
    "webshop":  ("Webshop",  "webshop",       frozenset()),
    "deepshop": ("DeepShop", "deepshop",      frozenset()),
    "webgames": ("WebGames", "webgames",      frozenset()),
}

# Experiment tag suffix — tag = {dataset}{_TAG_SUFFIX}
_TAG_SUFFIX = "_ood_all"

# Default OOD pairs shown in the full table
_DEFAULT_OOD_PAIRS = [
    ("wiki",     "webshop"),
    ("wiki",     "deepshop"),
    ("wiki",     "frames"),
    ("webshop",  "wiki"),
    ("webshop",  "deepshop"),
    ("frames",   "wiki"),
    ("deepshop", "webshop"),
]

# ─── Classifiers registry ─────────────────────────────────────────────────────
ALL_CLASSIFIERS: dict[str, tuple[str, str]] = {
    "RF":    ("RandomForest", "RF"),
    "XGB":   ("XGBoost",      "XGB"),
    "LSTM":  ("LSTM",         "LSTM"),
    "Lasso": ("LR_Lasso",     "Lasso"),
    "LR":    ("LR_L2",        "LR"),
}
_DEFAULT_CLASSIFIERS = ["RF", "XGB", "LSTM"]

# ─── Sentinels ────────────────────────────────────────────────────────────────
_ZERO_WITH_SUPPORT = object()
_DAGGER            = object()


# ─── Column definition builder ────────────────────────────────────────────────

def _experiment_tag(dataset: str, tag_overrides: dict) -> str:
    return tag_overrides.get(dataset, f"{dataset}{_TAG_SUFFIX}")


def build_column_defs(
    indomain: list[str] | None = None,
    ood_pairs: list[tuple[str, str]] | None = None,
    tag_overrides: dict | None = None,
) -> list:
    """
    Return column_defs as 6-tuples:
        (col_key, tag, split, ood_key, latex_header, absent_agents)

    indomain:   dataset short names for in-domain test columns
                (default: all five datasets in DATASETS order)
    ood_pairs:  list of (source, target) short names
                (default: _DEFAULT_OOD_PAIRS)
    tag_overrides: {dataset_short_name: custom_experiment_tag}
    """
    overrides = tag_overrides or {}
    cols = []

    # In-domain columns
    for ds in (indomain if indomain is not None else list(DATASETS)):
        disp, _, absent = DATASETS[ds]
        tag = _experiment_tag(ds, overrides)
        cols.append((
            f"test_{ds}", tag, "test", None,
            rf"\makecell{{\textbf{{{disp}}} \\ \textit{{(in-dom.)}}}}",
            absent,
        ))

    # OOD columns
    for src, tgt in (ood_pairs if ood_pairs is not None else _DEFAULT_OOD_PAIRS):
        src_disp, _, src_absent = DATASETS[src]
        tgt_disp, tgt_key, tgt_absent = DATASETS[tgt]
        tag = _experiment_tag(src, overrides)
        cols.append((
            f"{src}_{tgt}", tag, "ood", tgt_key,
            rf"\makecell{{\textbf{{{src_disp}$\to${tgt_disp}}} \\ \textit{{(OOD)}}}}",
            src_absent | tgt_absent,
        ))

    return cols


def parse_ood_pairs(specs: list[str]) -> list[tuple[str, str]]:
    """Parse ['wiki:frames', 'webshop:wiki', ...] into [(src, tgt), ...]."""
    pairs = []
    for s in specs:
        if ":" not in s:
            print(f"Warning: invalid --ood-pairs entry '{s}' (expected src:tgt) — skipped")
            continue
        src, tgt = s.split(":", 1)
        if src not in DATASETS:
            print(f"Warning: unknown source dataset '{src}' in '{s}' — skipped")
            continue
        if tgt not in DATASETS:
            print(f"Warning: unknown target dataset '{tgt}' in '{s}' — skipped")
            continue
        pairs.append((src, tgt))
    return pairs


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_results(traces_dir: Path) -> dict:
    results_map = {}
    for path in (traces_dir / "classifiers").glob("*/results.json"):
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


def weighted_f1(report) -> float | None:
    if not report:
        return None
    return (report.get("weighted avg") or {}).get("f1-score")


def _resolve_agent_entry(report, agent_id: str):
    for key in AGENT_ALIASES.get(agent_id, [agent_id]):
        entry = report.get(key)
        if isinstance(entry, dict):
            return entry
    return None


def agent_f1(report, agent_id: str, absent_agents: frozenset = frozenset()):
    """
    Return per-class F1 for agent_id, or a sentinel:
      _DAGGER            — agent excluded by design (absent_agents)
      _ZERO_WITH_SUPPORT — classifier ran but predicted F1 = 0
      None               — no data
    """
    if not report:
        return _DAGGER if agent_id in absent_agents else None
    entry = _resolve_agent_entry(report, agent_id)
    if entry is None:
        return _DAGGER if agent_id in absent_agents else None
    if not isinstance(entry, dict):
        return _DAGGER if agent_id in absent_agents else None
    if entry.get("support", 1) == 0:
        return None
    f1 = entry.get("f1-score", 0.0)
    return _ZERO_WITH_SUPPORT if f1 == 0.0 else f1


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
            stripped = r[c].replace(r"\textbf{", "").replace("}", "").split("$")[0]
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

def best_weighted_f1_row(results_map: dict, column_defs: list,
                         classifiers: list) -> list[str]:
    """For each column, return weighted avg F1 of the best classifier."""
    row = []
    for _, tag, split, ood_key, _, _ in column_defs:
        best: float | None = None
        for clf_key, _ in classifiers:
            rep = get_report(results_map, tag, clf_key, split, ood_key)
            val = weighted_f1(rep)
            if val is not None and (best is None or val > best):
                best = val
        row.append(fmt(best))
    return row

def clf_rows(results_map: dict, get_f1_fn, column_defs: list,
             classifiers: list) -> list:
    raw = []
    for clf_key, _ in classifiers:
        row = []
        for _, tag, split, ood_key, _, absent_agents in column_defs:
            rep = get_report(results_map, tag, clf_key, split, ood_key)
            row.append(fmt(get_f1_fn(rep, absent_agents)))
        raw.append(row)
    return bold_best_per_col(raw)


def agent_has_results(results_map: dict, agent_id: str, column_defs: list,
                      classifiers: list) -> bool:
    for clf_key, _ in classifiers:
        for _, tag, split, ood_key, _, absent_agents in column_defs:
            if agent_id in absent_agents:
                continue
            rep = get_report(results_map, tag, clf_key, split, ood_key)
            v = agent_f1(rep, agent_id, absent_agents)
            if v is not None and v is not _DAGGER:
                return True
    return False


def render_group(model_label: str, rows: list, lines: list, classifiers: list):
    for i, ((_, clf_disp), row) in enumerate(zip(classifiers, rows)):
        prefix = f"{model_label:<20}" if i == 0 else " " * 20
        lines.append(f"{prefix} & {clf_disp:<4} & {' & '.join(row)} \\\\")


def make_table(results_map: dict, agents: list, column_defs: list,
               classifiers: list) -> str:
    n_cond  = len(column_defs)
    n_total = n_cond + 2

    lines: list = [
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
                  if agent_has_results(results_map, a["agent_id"], column_defs, classifiers)]
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
                column_defs, classifiers,
            )
            render_group(disp, rows, lines, classifiers)
            lines.append(r"\midrule")

    # All Models row — weighted avg F1 of best classifier per column
    n_active = sum(1 for a in agents
                   if agent_has_results(results_map, a["agent_id"], column_defs, classifiers))
    n_str = f" ({n_active} models)" if n_active else ""
    lines.append(r"\rowcolor{headergreen}")
    lines.append(
        rf"\multicolumn{{{n_total}}}{{l}}{{\textbf{{\textit{{All Models{n_str}}}}}}}" + r" \\"
    )
    lines.append(r"\midrule")
    all_row = best_weighted_f1_row(results_map, column_defs, classifiers)
    prefix  = f"{'All':<20}"
    lines.append(f"{prefix} & {'Best':<4} & {' & '.join(all_row)} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ds_names = list(DATASETS)
    pair_examples = "wiki:frames webshop:wiki deepshop:webshop"

    parser = argparse.ArgumentParser(
        description="Generate LaTeX table from trace_analyzer results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python make_tables.py\n"
            "  python make_tables.py --in-domain-only\n"
            f"  python make_tables.py --ood-pairs {pair_examples}\n"
            "  python make_tables.py --classifiers RF XGB LSTM Lasso LR\n"
            "  python make_tables.py --tag-override wiki=wiki_custom frames=frames_v2\n"
        ),
    )
    parser.add_argument("traces_dir", nargs="?", type=Path, default=Path("./traces"))
    parser.add_argument("--agents", nargs="+", default=None, metavar="AGENT_ID",
                        help="Restrict to these agent IDs.")
    parser.add_argument("--classifiers", nargs="+", default=_DEFAULT_CLASSIFIERS,
                        choices=list(ALL_CLASSIFIERS), metavar="CLF",
                        help="Classifiers to show. Choices: "
                             + ", ".join(ALL_CLASSIFIERS)
                             + f".  Default: {' '.join(_DEFAULT_CLASSIFIERS)}.")
    parser.add_argument("--in-domain-only", action="store_true", default=False,
                        help="Include only in-domain (test split) columns.")
    parser.add_argument("--ood-only", action="store_true", default=False,
                        help="Include only OOD columns — drops all in-domain columns.")
    parser.add_argument("--indomain", nargs="+", default=None, metavar="DS",
                        choices=ds_names,
                        help="Which datasets to show as in-domain columns "
                             f"(default: all).  Choices: {', '.join(ds_names)}.")
    parser.add_argument("--ood-pairs", nargs="+", default=None, metavar="SRC:TGT",
                        help="OOD columns as source:target pairs, e.g. "
                             f"{pair_examples}.  "
                             f"Source/target must be one of: {', '.join(ds_names)}.  "
                             "Default: the standard 7-pair set.")
    parser.add_argument("--tag-override", nargs="+", default=None, metavar="DS=TAG",
                        help="Override the experiment tag for a dataset, e.g. "
                             "wiki=wiki_ood_v2  frames=frames_ood_v2.")
    cli = parser.parse_args()

    # Resolve tag overrides
    tag_overrides: dict[str, str] = {}
    for spec in (cli.tag_override or []):
        if "=" not in spec:
            print(f"Warning: invalid --tag-override '{spec}' (expected DS=TAG) — skipped")
            continue
        ds, tag = spec.split("=", 1)
        if ds not in DATASETS:
            print(f"Warning: unknown dataset '{ds}' in --tag-override — skipped")
            continue
        tag_overrides[ds] = tag

    # Resolve OOD pairs
    ood_pairs: list[tuple[str, str]] | None = None
    if cli.ood_pairs is not None:
        ood_pairs = parse_ood_pairs(cli.ood_pairs)
    if cli.in_domain_only:
        ood_pairs = []
    indomain = [] if cli.ood_only else cli.indomain

    classifiers = [ALL_CLASSIFIERS[name] for name in cli.classifiers]
    column_defs = build_column_defs(
        indomain=indomain,
        ood_pairs=ood_pairs,
        tag_overrides=tag_overrides,
    )

    results_map = load_results(cli.traces_dir)
    if not results_map:
        print(f"No results.json files found under {cli.traces_dir}/classifiers/")
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

    table = make_table(results_map, agents, column_defs, classifiers)

    suffix = "_in_domain" if cli.in_domain_only else "_ood" if cli.ood_only else "_main"
    out = cli.traces_dir / "classifiers" / f"table{suffix}.tex"
    out.write_text(table + "\n")
    print(table)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
