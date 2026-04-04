import json, pickle, warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

TRACE_DIR = Path("./traces")

EVENT_VOCAB = {
    "<pad>":        0,   # padding token — never appears in real data
    "<unk>":        1,   # unknown event type (forward-compatible)
    "click":        2,
    "keydown":      3,
    "scroll":       4,
    "navigate":     5,
    "beforeunload": 6,
    "focus":        7,   # input/textarea focus — search box interactions
}
VOCAB_SIZE = len(EVENT_VOCAB)


def _infer_split(dataset_name: str) -> str | None:
    if dataset_name.endswith("_train"): return "train"
    if dataset_name.endswith("_val"):   return "val"
    if dataset_name.endswith("_test"):  return "test"
    return None


def extract_features(episode) -> dict:
    dom    = episode.get("dom_trace", {})
    events = dom.get("events", [])
    mlog   = episode.get("midscene_log", [])

    clicks       = [e for e in events if e["type"] == "click"]
    scrolls      = [e for e in events if e["type"] == "scroll"]
    navs         = [e for e in events if e["type"] == "navigate"]
    keydowns     = [e for e in events if e["type"] == "keydown"]
    beforeunload = [e for e in events if e["type"] == "beforeunload"]
    focuses      = [e for e in events if e["type"] == "focus"]

    page_count = dom.get("pageCount") or len({e.get("url", "") for e in events})

    ts   = [e.get("t_episode", e["t"]) for e in events]
    ieis = np.diff(ts).tolist() if len(ts) > 1 else [0]

    # Timing
    total_duration_s = (ts[-1] - ts[0]) / 1000 if len(ts) >= 2 else 0
    mean_iei_ms      = float(np.mean(ieis))
    std_iei_ms       = float(np.std(ieis))
    median_iei_ms    = float(np.median(ieis))
    p10_iei_ms       = float(np.percentile(ieis, 10))
    p90_iei_ms       = float(np.percentile(ieis, 90))

    # Scroll
    max_scroll_pct  = max((e.get("pct", 0) for e in scrolls), default=0) or 0
    mean_scroll_pct = float(np.mean([e.get("pct", 0) for e in scrolls])) if scrolls else 0.0
    n_deep_scrolls  = sum(1 for e in scrolls if e.get("pct", 0) > 60)

    pcts  = [e.get("pct", 0) for e in scrolls]
    diffs = np.diff(pcts)
    scroll_reversals = int(np.sum((diffs[:-1] * diffs[1:]) < 0)) if len(diffs) > 1 else 0

    # Clicks
    click_xs = [e.get("x", 0) for e in clicks]
    click_ys = [e.get("y", 0) for e in clicks]
    click_x_std = float(np.std(click_xs)) if click_xs else 0.0
    click_y_std = float(np.std(click_ys)) if click_ys else 0.0

    n_link_clicks    = sum(1 for e in clicks if e.get("href"))
    link_click_ratio = n_link_clicks / max(len(clicks), 1)

    # Navigation / volume
    n_clicks           = len(clicks)
    n_scrolls          = len(scrolls)
    n_navigations      = len(navs)
    n_keydowns         = len(keydowns)
    n_beforeunload     = len(beforeunload)
    n_focus            = len(focuses)
    n_events_total     = len(events)
    n_midscene_actions = len(mlog)

    actions_per_page    = n_events_total / max(page_count, 1)
    nav_to_click_ratio  = n_navigations / max(n_clicks, 1)
    keydowns_per_page   = n_keydowns / max(page_count, 1)
    midscene_per_page   = n_midscene_actions / max(page_count, 1)
    focus_per_page      = n_focus / max(page_count, 1)

    # Scroll depth at beforeunload — how deep was the agent when it left each page
    bu_pcts = [e.get("pct", 0) for e in beforeunload]
    mean_exit_scroll_pct = float(np.mean(bu_pcts)) if bu_pcts else 0.0

    return {
        # Volume
        "n_clicks":             n_clicks,
        "n_scrolls":            n_scrolls,
        "n_navigations":        n_navigations,
        "n_keydowns":           n_keydowns,
        "n_beforeunload":       n_beforeunload,
        "n_focus":              n_focus,
        "n_events_total":       n_events_total,
        "n_midscene_actions":   n_midscene_actions,
        "page_count":           page_count,
        # Timing
        "total_duration_s":     total_duration_s,
        "mean_iei_ms":          mean_iei_ms,
        "std_iei_ms":           std_iei_ms,
        "median_iei_ms":        median_iei_ms,
        "p10_iei_ms":           p10_iei_ms,
        "p90_iei_ms":           p90_iei_ms,
        # Scroll
        "max_scroll_pct":       max_scroll_pct,
        "mean_scroll_pct":      mean_scroll_pct,
        "n_deep_scrolls":       n_deep_scrolls,
        "scroll_reversals":     scroll_reversals,
        # Clicks
        "click_x_std":          click_x_std,
        "click_y_std":          click_y_std,
        "n_link_clicks":        n_link_clicks,
        "link_click_ratio":     link_click_ratio,
        # Navigation
        "actions_per_page":       actions_per_page,
        "nav_to_click_ratio":     nav_to_click_ratio,
        "keydowns_per_page":      keydowns_per_page,
        "midscene_per_page":      midscene_per_page,
        "focus_per_page":         focus_per_page,
        "mean_exit_scroll_pct":   mean_exit_scroll_pct,
    }


def extract_sequence(episode) -> list:
    events = episode.get("dom_trace", {}).get("events", [])
    return [EVENT_VOCAB.get(e["type"], EVENT_VOCAB["<unk>"]) for e in events]


def load_dataset(trace_dir: Path) -> dict[str, tuple]:
    """Load all episode traces, bucketed by split inferred from the path.

    Path pattern: traces/{agent_id}/{dataset_name}/{timestamp}/{episode_id}.json
    Split is determined by the suffix of dataset_name:
      _train → "train",  _val → "val",  _test → "test"

    Returns {"train": (features, sequences, labels), "val": ..., "test": ...}
    with empty lists for any split that has no data.
    """
    buckets: dict[str, tuple[list, list, list]] = {
        "train": ([], [], []),
        "val":   ([], [], []),
        "test":  ([], [], []),
    }
    for path in sorted(trace_dir.rglob("*.json")):
        rel_parts = path.relative_to(trace_dir).parts
        if rel_parts[0] == "models":
            continue  # skip results.json / classifier artefacts
        if len(rel_parts) < 2:
            warnings.warn(f"Skipping {path}: unexpected path depth")
            continue
        split = _infer_split(rel_parts[1])
        if split is None:
            warnings.warn(f"Skipping {path}: unrecognised dataset '{rel_parts[1]}'")
            continue
        try:
            with open(path) as f:
                episode = json.load(f)
            buckets[split][0].append(extract_features(episode))
            buckets[split][1].append(extract_sequence(episode))
            buckets[split][2].append(episode["meta"]["agent_id"])
        except Exception as e:
            warnings.warn(f"Skipping {path.name}: {e}")
    return {s: tuple(lists) for s, lists in buckets.items()}


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels_encoded):
        # sequences: list of 1-D torch.LongTensor
        # labels_encoded: 1-D torch.LongTensor
        self.sequences      = sequences
        self.labels_encoded = labels_encoded

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels_encoded[idx]


def collate_fn(batch):
    seqs, lbls = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    padded  = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, lengths, torch.stack(lbls)


class AgentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, n_classes,
                 dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, lengths):
        # x: (batch, seq_len) — padded token IDs
        # lengths: (batch,) — true sequence lengths
        emb    = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n: (n_layers, batch, hidden_dim) — take the last layer
        out = self.dropout(h_n[-1])
        return self.head(out)


_LSTM_EMBED_DIM  = 16
_LSTM_N_LAYERS   = 2
_LSTM_BATCH_SIZE = 16
_LSTM_LR         = 1e-3
_LSTM_GRID = [
    {"hidden_dim": 64,  "dropout": 0.2},
    {"hidden_dim": 64,  "dropout": 0.4},
    {"hidden_dim": 128, "dropout": 0.2},
    {"hidden_dim": 128, "dropout": 0.4},
]


def _fit_lstm(seq_train, y_train, n_classes, hyperparams, n_epochs=30):
    """Train one AgentLSTM config on seq_train/y_train. Returns the trained model."""
    device = torch.device("cpu")
    hidden_dim = hyperparams["hidden_dim"]
    dropout    = hyperparams["dropout"]

    seq_tensors = [torch.tensor(s, dtype=torch.long) for s in seq_train]
    lbl_tensor  = torch.tensor(y_train, dtype=torch.long)
    dl = DataLoader(SequenceDataset(seq_tensors, lbl_tensor),
                    batch_size=_LSTM_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model     = AgentLSTM(VOCAB_SIZE, _LSTM_EMBED_DIM, hidden_dim, _LSTM_N_LAYERS,
                          n_classes, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=_LSTM_LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(n_epochs):
        for padded, lengths, lbls in dl:
            padded, lengths, lbls = padded.to(device), lengths.to(device), lbls.to(device)
            optimizer.zero_grad()
            criterion(model(padded, lengths), lbls).backward()
            optimizer.step()
    return model


def _eval_lstm(model, seq_eval, y_eval):
    """Evaluate a trained AgentLSTM. Returns (accuracy, predictions_list)."""
    device = torch.device("cpu")
    seq_tensors = [torch.tensor(s, dtype=torch.long) for s in seq_eval]
    lbl_tensor  = torch.tensor(y_eval, dtype=torch.long)
    dl = DataLoader(SequenceDataset(seq_tensors, lbl_tensor),
                    batch_size=_LSTM_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for padded, lengths, lbls in dl:
            padded, lengths, lbls = padded.to(device), lengths.to(device), lbls.to(device)
            preds = model(padded, lengths).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_trues.extend(lbls.cpu().tolist())

    acc = sum(p == t for p, t in zip(all_preds, all_trues)) / len(all_trues)
    return float(acc), all_preds


def train_lstm(seq_train, y_train, seq_val, y_val, seq_test, y_test,
               n_classes, n_epochs=30) -> dict:
    """Grid search over hidden_dim × dropout, pick best by val accuracy.

    Fits final model on train only. Returns best_params, val_report, test_report.
    test_report is None when seq_test is empty.
    """
    best_val_acc = -1.0
    best_params  = _LSTM_GRID[0]

    print("  LSTM grid search:")
    for params in _LSTM_GRID:
        model = _fit_lstm(seq_train, y_train, n_classes, params, n_epochs=n_epochs)
        val_acc, _ = _eval_lstm(model, seq_val, y_val)
        print(f"    hidden_dim={params['hidden_dim']}  dropout={params['dropout']}  "
              f"val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params  = params

    print(f"  Best: {best_params}  val_acc={best_val_acc:.3f}")

    final_model      = _fit_lstm(seq_train, y_train, n_classes, best_params, n_epochs=n_epochs)
    _, val_preds     = _eval_lstm(final_model, seq_val, y_val)
    val_report       = classification_report(y_val, val_preds, output_dict=True)
    test_report      = None
    if seq_test:
        _, test_preds = _eval_lstm(final_model, seq_test, y_test)
        test_report   = classification_report(y_test, test_preds, output_dict=True)

    torch.save(final_model.state_dict(), TRACE_DIR / "models" / "lstm_model.pt")
    return {"best_params": best_params, "val_report": val_report, "test_report": test_report}


RF_PARAM_GRID = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5],
}
GB_PARAM_GRID = {
    "n_estimators":  [100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth":     [3, 5],
}


def train(trace_dir: Path) -> None:
    splits = load_dataset(trace_dir)
    feat_train, seq_train, lbl_train = splits["train"]
    feat_val,   seq_val,   lbl_val   = splits["val"]
    feat_test,  seq_test,  lbl_test  = splits["test"]

    # --- guards ---
    if not feat_train:
        print("ERROR: No training episodes found. Run more episodes first.")
        return
    if len(feat_train) < 10:
        print(f"WARNING: Only {len(feat_train)} training episodes — results will not be meaningful.")
    has_val  = bool(feat_val)
    has_test = bool(feat_test)
    if not has_val:
        warnings.warn("No val episodes found — skipping hyperparameter tuning, using defaults.")
    if not has_test:
        warnings.warn("No test episodes found — test_report will be null.")

    # --- label encoder fitted on all labels across splits ---
    le = LabelEncoder()
    le.fit(lbl_train + lbl_val + lbl_test)
    y_train = le.transform(lbl_train)
    y_val   = le.transform(lbl_val)  if lbl_val  else np.array([], dtype=int)
    y_test  = le.transform(lbl_test) if lbl_test else np.array([], dtype=int)

    # --- feature matrices ---
    feat_names = list(feat_train[0].keys())
    def to_X(feats):
        return np.array([[ep[k] for k in feat_names] for ep in feats]) if feats \
               else np.empty((0, len(feat_names)))
    X_train = to_X(feat_train)
    X_val   = to_X(feat_val)
    X_test  = to_X(feat_test)

    # --- RF / GradientBoosting ---
    clf_results = {}
    best_rf     = None

    for name, Clf, param_grid, defaults in [
        ("RandomForest",
         RandomForestClassifier,
         RF_PARAM_GRID,
         {"n_estimators": 200, "random_state": 42}),
        ("GradientBoosting",
         GradientBoostingClassifier,
         GB_PARAM_GRID,
         {"n_estimators": 100, "random_state": 42}),
    ]:
        if has_val:
            gs = GridSearchCV(Clf(random_state=42), param_grid,
                              cv=3, scoring="accuracy", n_jobs=-1, refit=True)
            gs.fit(X_train, y_train)
            best_clf    = gs.best_estimator_
            best_params = gs.best_params_
            val_preds   = best_clf.predict(X_val)
            val_report  = classification_report(
                y_val, val_preds, target_names=le.classes_, output_dict=True)
            print(f"{name:20s}  best={best_params}  "
                  f"val_acc={val_report['accuracy']:.3f}")
        else:
            best_clf = Clf(**defaults).fit(X_train, y_train)
            best_params, val_report = defaults, None

        test_report = None
        if has_test:
            test_preds  = best_clf.predict(X_test)
            test_report = classification_report(
                y_test, test_preds, target_names=le.classes_, output_dict=True)
            print(f"{name:20s}  test_acc={test_report['accuracy']:.3f}")

        clf_results[name] = {
            "best_params": best_params,
            "val_report":  val_report,
            "test_report": test_report,
        }
        if name == "RandomForest":
            best_rf = best_clf

    importances = sorted(zip(feat_names, best_rf.feature_importances_), key=lambda x: -x[1])
    print("\nTop 10 features (Random Forest):")
    for fname, imp in importances[:10]:
        print(f"  {fname:<30} {imp:.4f}")
    clf_results["RandomForest"]["feature_importances"] = {
        fname: float(imp) for fname, imp in importances
    }

    # --- LSTM ---
    def remap(report):
        if report is None:
            return None
        out = {}
        for k, v in report.items():
            try:
                out[le.classes_[int(k)]] = v
            except (ValueError, IndexError):
                out[k] = v
        return out

    print("\nTraining LSTM ...")
    n_classes = len(le.classes_)
    if has_val:
        lstm_result = train_lstm(
            seq_train, list(y_train),
            seq_val,   list(y_val),
            seq_test,  list(y_test),
            n_classes,
        )
        lstm_result["val_report"]  = remap(lstm_result["val_report"])
        lstm_result["test_report"] = remap(lstm_result["test_report"])
        if lstm_result["val_report"]:
            print(f"{'LSTM':20s}  val_acc={lstm_result['val_report']['accuracy']:.3f}")
        if lstm_result["test_report"]:
            print(f"{'LSTM':20s}  test_acc={lstm_result['test_report']['accuracy']:.3f}")
    else:
        default_params = {"hidden_dim": 64, "dropout": 0.3}
        model          = _fit_lstm(seq_train, list(y_train), n_classes, default_params)
        test_report    = None
        if has_test:
            _, test_preds = _eval_lstm(model, seq_test, list(y_test))
            test_report   = remap(classification_report(list(y_test), test_preds, output_dict=True))
        torch.save(model.state_dict(), TRACE_DIR / "models" / "lstm_model.pt")
        lstm_result = {"best_params": default_params, "val_report": None, "test_report": test_report}

    clf_results["LSTM"] = lstm_result

    # --- save artefacts ---
    models_dir = trace_dir / "models"
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "classifier.pkl", "wb") as f:
        pickle.dump({"rf": best_rf, "le": le, "feat_names": feat_names}, f)

    results = {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "n_episodes":  {"train": len(feat_train), "val": len(feat_val), "test": len(feat_test)},
        "class_names": list(le.classes_),
        "models":      clf_results,
    }
    results_path = models_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}  traces/models/classifier.pkl  traces/models/lstm_model.pt")


if __name__ == "__main__":
    train(TRACE_DIR)
