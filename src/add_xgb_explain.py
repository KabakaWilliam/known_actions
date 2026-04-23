"""
Compute SHAP-based feature importance for the XGBoost classifier in a
trained experiment and append it to that experiment's results.json.

Usage:
    python add_xgb_explain.py --tag deepshop_family_ood_all [--traces-dir src/traces] [--overwrite]

The script expects classifier.pkl to contain an 'X_test' key (saved by
trace_analyzer.py >= the version that added this field). If the key is
missing, re-run trace_analyzer.py to retrain the experiment.

Output appended to results.json:
    results["models"]["XGBoost"]["shap_importances"] = {
        "feature_name": mean_abs_shap_value,
        ...   # sorted descending
    }
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


def load_classifier(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_shap_importances(xgb_model, X: np.ndarray, feat_names: list) -> dict:
    try:
        import shap
    except ImportError:
        sys.exit("ERROR: 'shap' is not installed. Run: pip install shap")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)

    # Multi-class: shap_values shape is (n_samples, n_features, n_classes)
    # Binary:      shap_values shape is (n_samples, n_features)
    sv = np.array(shap_values)
    mean_abs = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)

    importance = dict(zip(feat_names, mean_abs.tolist()))
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def main():
    parser = argparse.ArgumentParser(description="Append SHAP importances to results.json for an XGBoost experiment.")
    parser.add_argument("--tag", required=True, help="Experiment tag (subdirectory under classifiers/)")
    parser.add_argument("--traces-dir", default=None,
                        help="Root traces directory (default: <script_dir>/traces)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing shap_importances if present")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    traces_dir = Path(args.traces_dir) if args.traces_dir else script_dir / "traces"
    models_dir = traces_dir / "classifiers" / args.tag

    pkl_path = models_dir / "classifier.pkl"
    results_path = models_dir / "results.json"

    if not pkl_path.exists():
        sys.exit(f"ERROR: classifier.pkl not found at {pkl_path}")
    if not results_path.exists():
        sys.exit(f"ERROR: results.json not found at {results_path}")

    print(f"Loading {pkl_path} ...")
    data = load_classifier(pkl_path)

    if "X_test" not in data:
        sys.exit(
            "ERROR: classifier.pkl does not contain 'X_test'.\n"
            "Re-run trace_analyzer.py for this experiment to regenerate the pkl."
        )

    xgb_model  = data["xgb"]
    feat_names = data["feat_names"]
    X_test     = data["X_test"]

    if X_test is None or len(X_test) == 0:
        sys.exit("ERROR: X_test is empty — no test samples available for this experiment.")

    print(f"Computing SHAP values over {len(X_test)} test samples × {len(feat_names)} features ...")
    importances = compute_shap_importances(xgb_model, X_test, feat_names)

    with open(results_path) as f:
        results = json.load(f)

    xgb_entry = results.get("models", {}).get("XGBoost", {})
    if "shap_importances" in xgb_entry:
        if not args.overwrite:
            sys.exit(
                "ERROR: shap_importances already exists in results.json. "
                "Use --overwrite to replace it."
            )
        print("Overwriting existing shap_importances.")

    xgb_entry["shap_importances"] = importances
    results.setdefault("models", {})["XGBoost"] = xgb_entry

    # Atomic write
    tmp_path = results_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, results_path)

    print(f"Written shap_importances ({len(importances)} features) to {results_path}")
    print("\nTop 10 features by mean |SHAP|:")
    for i, (feat, val) in enumerate(list(importances.items())[:10], 1):
        print(f"  {i:2d}. {feat:<30s}  {val:.4f}")


if __name__ == "__main__":
    main()
