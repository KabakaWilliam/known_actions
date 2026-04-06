"""
make_tables.py — Generate publication-ready LaTeX tables from multi-domain trace_analyzer results.

Usage:
    python make_tables.py                    # reads ./traces/models/(wiki_ood_amazon|webshop)/results.json
    python make_tables.py /path/to/traces    # custom traces dir

Outputs:
    traces/models/table_main.tex       — main accuracy / F1 table across domains
    traces/models/table_per_class.tex  — per-agent F1 breakdown by domain
"""

import json, sys
from pathlib import Path

# Experiment tags and domain labels
EXPERIMENTS = {
    "wiki_ood_amazon": {
        "test_label": r"\textbf{Wiki (in-dom.)}",
        "ood_label":  r"\textbf{Wiki $\to$ Amazon}",
    },
    "webshop": {
        "test_label": r"\textbf{Amazon (in-dom.)}",
        "ood_label":  r"\textbf{Amazon $\to$ DeepShop}",
    },
}

# Column order: (experiment_tag, split_type)
CONDITION_ORDER = [
    ("wiki_ood_amazon", "test"),
    ("wiki_ood_amazon", "ood"),
    ("webshop",         "test"),
    ("webshop",         "ood"),
]

MODEL_KEYS    = ["RandomForest", "GradientBoosting", "LSTM"]
MODEL_DISPLAY = {
    "RandomForest":     "Random Forest",
    "GradientBoosting": "Gradient Boosting",
    "LSTM":             "LSTM",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(results_path: Path) -> dict:
    with open(results_path) as f:
        return json.load(f)


def fmt(x) -> str:
    """Float 0–1 → percentage string with one decimal place."""
    if x is None:
        return "--"
    return f"{x * 100:.1f}"


def bold_max(values: list) -> list[str]:
    """Bold the maximum value in a list."""
    vals_clean = []
    for v in values:
        v_str = v.replace(r"\textbf{", "").replace("}", "")
        try:
            vals_clean.append(float(v_str))
        except:
            vals_clean.append(None)
    
    valid = [v for v in vals_clean if v is not None]
    if not valid:
        return values
    
    best = max(valid)
    out = []
    for v_raw, v_num in zip(values, vals_clean):
        v_str = v_raw.replace(r"\textbf{", "").replace("}", "")
        if v_num is not None and abs(v_num - best) < 1e-6:
            out.append(r"\textbf{" + v_str + "}")
        else:
            out.append(v_str)
    return out


def get_report(results_map: dict, tag: str, model: str, split: str):
    """Get test or ood report for a model."""
    if tag not in results_map or model not in results_map[tag].get("models", {}):
        return None
    report_key = f"{split}_report"
    return results_map[tag]["models"][model].get(report_key)


# ---------------------------------------------------------------------------
# Table 1 — main results: accuracy + macro F1 across domains
# ---------------------------------------------------------------------------

def make_main_table(results_map: dict) -> str:
    # Column headers
    cond_labels = []
    for tag, split in CONDITION_ORDER:
        if split == "test":
            cond_labels.append(EXPERIMENTS[tag]["test_label"])
        else:
            cond_labels.append(EXPERIMENTS[tag]["ood_label"])

    n_conds = len(CONDITION_ORDER)

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\caption{Agent identification accuracy (\%) and macro F1 (\%) across domains. "
                 r"In-domain = held-out test split from training distribution. "
                 r"OOD = domain not seen during training.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l " + " cc" * n_conds + "}")
    lines.append(r"\toprule")

    # Multi-column header
    header_cols = " & ".join(
        r"\multicolumn{2}{c}{" + lbl + "}" for lbl in cond_labels
    )
    lines.append(" & " + header_cols + r" \\")

    # cmidrule under each pair
    cmidrules = []
    for i in range(n_conds):
        lo = 2 + i * 2
        hi = lo + 1
        cmidrules.append(rf"\cmidrule(lr){{{lo}-{hi}}}")
    lines.append("".join(cmidrules))

    # Sub-header
    sub = r"\textbf{Model}" + " & Acc & F1" * n_conds + r" \\"
    lines.append(sub)
    lines.append(r"\midrule")

    # Collect all data first for per-column bolding
    all_vals = []
    for mkey in MODEL_KEYS:
        row = []
        for tag, split in CONDITION_ORDER:
            report = get_report(results_map, tag, mkey, split)
            if report:
                acc = fmt(report.get("accuracy"))
                ma = report.get("macro avg", {})
                f1 = fmt(ma.get("f1-score") if isinstance(ma, dict) else None)
            else:
                acc, f1 = "--", "--"
            row.extend([acc, f1])
        all_vals.append(row)

    # Bold per column
    n_cols = len(all_vals[0])
    for col_i in range(n_cols):
        col_vals = [all_vals[r][col_i] for r in range(len(MODEL_KEYS))]
        bolded = bold_max(col_vals)
        for r in range(len(MODEL_KEYS)):
            all_vals[r][col_i] = bolded[r]

    # Data rows
    for mkey, row in zip(MODEL_KEYS, all_vals):
        cells = " & ".join(row)
        lines.append(f"{MODEL_DISPLAY[mkey]:20s} & {cells} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2 — per-agent F1 breakdown
# ---------------------------------------------------------------------------

def make_per_class_table(results_map: dict) -> str:
    # Collect all unique agents across experiments
    agents = []
    for tag in EXPERIMENTS:
        if tag in results_map:
            for name in results_map[tag].get("class_names", []):
                if name not in agents:
                    agents.append(name)

    # Column abbreviations with line breaks
    abbrev = {
        ("wiki_ood_amazon", "test"): r"\makecell{\textbf{Wiki}\\(in-dom.)}",
        ("wiki_ood_amazon", "ood"):  r"\makecell{\textbf{Wiki}$\to$\\\textbf{Amazon}}",
        ("webshop",         "test"): r"\makecell{\textbf{Amazon}\\(in-dom.)}",
        ("webshop",         "ood"):  r"\makecell{\textbf{Amazon}$\to$\\\textbf{DeepShop}}",
    }

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\caption{Per-agent F1 score (\%) by training/evaluation condition.}")
    lines.append(r"\label{tab:per-class}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{l c c c c}")
    lines.append(r"\toprule")

    # Header
    header = r"\textbf{Agent}"
    for tag, split in CONDITION_ORDER:
        header += " & " + abbrev[(tag, split)]
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    # For each model, print its agents' F1 scores
    for mkey in MODEL_KEYS:
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{MODEL_DISPLAY[mkey]}}}}}" + r" \\")

        # Collect F1 per agent per condition
        agent_vals = {a: [] for a in agents}
        for tag, split in CONDITION_ORDER:
            report = get_report(results_map, tag, mkey, split)
            for agent in agents:
                if report and agent in report and isinstance(report[agent], dict):
                    f1 = fmt(report[agent].get("f1-score"))
                else:
                    f1 = "--"
                agent_vals[agent].append(f1)

        # Bold max per column
        for col_i in range(len(CONDITION_ORDER)):
            col_vals = [agent_vals[a][col_i] for a in agents]
            bolded = bold_max(col_vals)
            for a, bv in zip(agents, bolded):
                agent_vals[a][col_i] = bv

        # Print agents
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

    # Load all experiment results
    results_map = {}
    for tag in EXPERIMENTS:
        path = traces_dir / "models" / tag / "results.json"
        if path.exists():
            results_map[tag] = load_results(path)
        else:
            print(f"Warning: {path} not found")

    if not results_map:
        print(f"No results found in {traces_dir}/models/")
        sys.exit(1)

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
