"""
make_tables.py — Generate publication-ready LaTeX tables from trace_analyzer results.

Usage:
    python make_tables.py                    # reads ./traces/models/
    python make_tables.py /path/to/traces    # custom traces dir

Outputs:
    traces/models/table_main.tex       — main accuracy / macro-F1 table
    traces/models/table_per_class.tex  — per-agent F1 breakdown table
"""

import json, sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — maps experiment tag to human-readable column labels
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "wiki_ood_amazon": {
        "test_label": "Wiki (in-domain)",
        "ood_label":  r"Wiki $\to$ Amazon (OOD)",
    },
    "webshop": {
        "test_label": "Amazon (in-domain)",
        "ood_label":  r"Amazon $\to$ DeepShop (OOD)",
    },
}

MODEL_KEYS    = ["RandomForest", "GradientBoosting", "LSTM"]
MODEL_DISPLAY = {
    "RandomForest":     "Random Forest",
    "GradientBoosting": "Gradient Boosting",
    "LSTM":             "LSTM",
}

# Column order for both tables
CONDITION_ORDER = [
    ("wiki_ood_amazon", "test"),
    ("wiki_ood_amazon", "ood"),
    ("webshop",         "test"),
    ("webshop",         "ood"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(traces_dir: Path, tag: str) -> dict:
    path = traces_dir / "models" / tag / "results.json"
    with open(path) as f:
        return json.load(f)


def fmt(x) -> str:
    """Float 0–1 → percentage string with one decimal place."""
    if x is None:
        return "--"
    return f"{x * 100:.1f}"


def bold_max(values: list) -> list[str]:
    """Return formatted strings; bold the maximum (ignoring '--')."""
    floats = [float(v) if v != "--" else None for v in values]
    valid  = [v for v in floats if v is not None]
    if not valid:
        return values
    best = max(valid)
    out  = []
    for v, f in zip(values, floats):
        if f is not None and abs(f - best) < 1e-9:
            out.append(r"\textbf{" + v + "}")
        else:
            out.append(v)
    return out


def get_metric(report, key: str):
    """Safely extract a scalar from a classification report dict."""
    if report is None:
        return None
    entry = report.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("f1-score")
    return entry   # scalar (accuracy)


def get_report(results: dict, model: str, split: str):
    """Return the report dict for a given model and split name."""
    return results.get("models", {}).get(model, {}).get(f"{split}_report")


# ---------------------------------------------------------------------------
# Table 1 — main results: accuracy + macro F1
# ---------------------------------------------------------------------------

def make_main_table(results_map: dict) -> str:
    # Column headers
    cond_labels = []
    for tag, split in CONDITION_ORDER:
        label_key = "test_label" if split == "test" else "ood_label"
        cond_labels.append(EXPERIMENTS[tag][label_key])

    n_conds = len(CONDITION_ORDER)   # 4

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\caption{Agent identification accuracy (\%) and macro F1 (\%) across domains. "
                 r"In-domain = held-out test split from training distribution. "
                 r"OOD = domain not seen during training.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{l " + " cc" * n_conds + "}")
    lines.append(r"\toprule")

    # Multi-column header row
    header_cols = " & ".join(
        r"\multicolumn{2}{c}{\textbf{" + lbl + "}}" for lbl in cond_labels
    )
    lines.append(r" & " + header_cols + r" \\")

    # cmidrule under each pair
    cmidrules = []
    for i in range(n_conds):
        lo = 2 + i * 2
        hi = lo + 1
        cmidrules.append(rf"\cmidrule(lr){{{lo}-{hi}}}")
    lines.append("".join(cmidrules))

    # Sub-header: Acc / F1 pairs
    sub = r"\textbf{Model}" + " & Acc & F1" * n_conds + r" \\"
    lines.append(sub)
    lines.append(r"\midrule")

    # Data rows — collect all values first for per-column bolding
    # Shape: [model_idx][condition_idx * 2 + {0=acc, 1=f1}]
    all_vals = []
    for mkey in MODEL_KEYS:
        row = []
        for tag, split in CONDITION_ORDER:
            report = get_report(results_map[tag], mkey, split)
            acc = fmt(get_metric(report, "accuracy"))
            f1  = fmt(get_metric(report.get("macro avg") if report else None, None)
                      if report and isinstance(report.get("macro avg"), dict)
                      else None)
            # cleaner extraction:
            if report:
                acc = fmt(report.get("accuracy"))
                ma  = report.get("macro avg", {})
                f1  = fmt(ma.get("f1-score") if isinstance(ma, dict) else None)
            else:
                acc, f1 = "--", "--"
            row.extend([acc, f1])
        all_vals.append(row)

    # Bold per column
    n_cols = len(all_vals[0])
    for col_i in range(n_cols):
        col_vals = [all_vals[r][col_i] for r in range(len(MODEL_KEYS))]
        bolded   = bold_max(col_vals)
        for r in range(len(MODEL_KEYS)):
            all_vals[r][col_i] = bolded[r]

    for mkey, row in zip(MODEL_KEYS, all_vals):
        cells = " & ".join(row)
        lines.append(f"{MODEL_DISPLAY[mkey]:20s} & {cells} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2 — per-agent F1 breakdown
# ---------------------------------------------------------------------------

def make_per_class_table(results_map: dict) -> str:
    # Collect all agent names across experiments
    agents: list[str] = []
    for tag in EXPERIMENTS:
        for name in results_map[tag].get("class_names", []):
            if name not in agents:
                agents.append(name)

    n_conds = len(CONDITION_ORDER)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Per-agent F1 score (\%) by training/evaluation condition.}")
    lines.append(r"\label{tab:per-class}")
    lines.append(r"\centering")
    # Need makecell for line breaks in header
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{l " + "c " * n_conds + "}")
    lines.append(r"\toprule")

    # Column headers (abbreviated, two-line)
    abbrev = {
        ("wiki_ood_amazon", "test"): r"\makecell{Wiki\\(in-dom.)}",
        ("wiki_ood_amazon", "ood"):  r"\makecell{Wiki$\to$\\Amazon}",
        ("webshop",         "test"): r"\makecell{Amazon\\(in-dom.)}",
        ("webshop",         "ood"):  r"\makecell{Amazon$\to$\\DeepShop}",
    }
    header = r"\textbf{Agent}" + " & " + " & ".join(abbrev[c] for c in CONDITION_ORDER)
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for mkey in MODEL_KEYS:
        lines.append(rf"\multicolumn{{{n_conds + 1}}}{{l}}{{\textit{{{MODEL_DISPLAY[mkey]}}}}}" + r" \\")

        # Collect per-agent per-condition F1
        agent_vals: dict[str, list[str]] = {a: [] for a in agents}
        for tag, split in CONDITION_ORDER:
            report = get_report(results_map[tag], mkey, split)
            for agent in agents:
                if report and agent in report and isinstance(report[agent], dict):
                    v = fmt(report[agent].get("f1-score"))
                else:
                    v = "--"
                agent_vals[agent].append(v)

        # Bold max per column
        for col_i in range(n_conds):
            col_vals = [agent_vals[a][col_i] for a in agents]
            bolded   = bold_max(col_vals)
            for a, bv in zip(agents, bolded):
                agent_vals[a][col_i] = bv

        for agent in agents:
            label = r"\quad " + agent.replace("_", r"\_")
            cells = " & ".join(agent_vals[agent])
            lines.append(f"{label} & {cells} \\\\")

        if mkey != MODEL_KEYS[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    traces_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./traces")

    missing = [tag for tag in EXPERIMENTS if not (traces_dir / "models" / tag / "results.json").exists()]
    if missing:
        print(f"Missing results for experiments: {missing}")
        print(f"Expected at: {traces_dir}/models/<tag>/results.json")
        sys.exit(1)

    results_map = {tag: load_results(traces_dir, tag) for tag in EXPERIMENTS}

    table1 = make_main_table(results_map)
    table2 = make_per_class_table(results_map)

    out_dir = traces_dir / "models"
    out1 = out_dir / "table_main.tex"
    out2 = out_dir / "table_per_class.tex"
    out1.write_text(table1 + "\n")
    out2.write_text(table2 + "\n")

    print(table1)
    print()
    print(table2)
    print(f"\nSaved: {out1}  {out2}")


if __name__ == "__main__":
    main()
