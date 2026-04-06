import argparse, json, pickle, warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    if dataset_name.endswith("_ood"):   return "ood"
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

    ts   = [e.get("t_episode") or e.get("t") or 0 for e in events]
    ieis = np.diff(ts).tolist() if len(ts) > 1 else [0]

    # Timing
    total_duration_s = (ts[-1] - ts[0]) / 1000 if len(ts) >= 2 else 0
    mean_iei_ms      = float(np.mean(ieis))
    std_iei_ms       = float(np.std(ieis))
    median_iei_ms    = float(np.median(ieis))
    p10_iei_ms       = float(np.percentile(ieis, 10))
    p90_iei_ms       = float(np.percentile(ieis, 90))

    # Scroll
    max_scroll_pct  = max((e.get("pct") or 0 for e in scrolls), default=0)
    mean_scroll_pct = float(np.mean([e.get("pct") or 0 for e in scrolls])) if scrolls else 0.0
    n_deep_scrolls  = sum(1 for e in scrolls if (e.get("pct") or 0) > 60)

    pcts  = [e.get("pct") or 0 for e in scrolls]
    diffs = np.diff(pcts)
    scroll_reversals = int(np.sum((diffs[:-1] * diffs[1:]) < 0)) if len(diffs) > 1 else 0

    # Clicks
    click_xs = [e.get("x") or 0 for e in clicks]
    click_ys = [e.get("y") or 0 for e in clicks]
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
    bu_pcts = [e.get("pct") or 0 for e in beforeunload]
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


def extract_sequence(episode) -> list[tuple[int, float, float, float, float]]:
    """Return a list of (token_id, f0, f1, f2, f3) per event.

    f0 = log1p(delta_t_ms)   — inter-event gap
    f1 = log1p(t_episode_ms) — absolute position in session
    f2, f3 = event-specific spatial/depth scalars:
        scroll → (pct/100, 0)
        click  → (x/1280, y/768)
        other  → (0, 0)
    """
    events = episode.get("dom_trace", {}).get("events", [])
    result = []
    prev_t = None
    for e in events:
        token = EVENT_VOCAB.get(e["type"], EVENT_VOCAB["<unk>"])
        t     = e.get("t_episode") or e.get("t") or 0
        delta = (t - prev_t) if prev_t is not None else 0.0
        f0    = float(np.log1p(max(delta, 0)))
        f1    = float(np.log1p(max(t, 0)))
        if e["type"] == "scroll":
            f2, f3 = (e.get("pct") or 0) / 100.0, 0.0
        elif e["type"] == "click":
            f2, f3 = (e.get("x") or 0) / 1280.0, (e.get("y") or 0) / 768.0
        else:
            f2, f3 = 0.0, 0.0
        result.append((token, f0, f1, f2, f3))
        prev_t = t
    return result


def load_dataset(trace_dir: Path,
                 train_datasets: list[str] | None = None,
                 ood_datasets: list[str] | None = None,
                 ) -> tuple[dict[str, tuple], dict[str, set]]:
    """Load all episode traces, bucketed by split.

    Path pattern: traces/{agent_id}/{dataset_name}/{timestamp}/{episode_id}.json

    train_datasets: base names (suffix stripped) whose _train/_val/_test traces
      go into the train/val/test buckets. e.g. ["2wikimultihop"] loads only
      2wikimultihop_* traces for training.

    ood_datasets: base names whose traces all go into the OOD bucket regardless
      of suffix. e.g. ["webshop"] loads webshop_train/val/test all as OOD.

    If both are None: legacy mode — all datasets, split by suffix.

    Returns (splits, ds_names) where:
      splits   = {"train": (features, sequences, labels), "val": ..., ...}
      ds_names = {"train": {"2wikimultihop_train", ...}, "val": ..., ...}
    """
    buckets: dict[str, tuple[list, list, list]] = {
        "train": ([], [], []),
        "val":   ([], [], []),
        "test":  ([], [], []),
        "ood":   ([], [], []),
    }
    ds_names: dict[str, set] = {"train": set(), "val": set(), "test": set(), "ood": set()}
    for path in sorted(trace_dir.rglob("*.json")):
        rel_parts = path.relative_to(trace_dir).parts
        if rel_parts[0].startswith("models"):
            continue  # skip results.json / classifier artefacts (models/, models_v1/, etc.)
        if len(rel_parts) < 2:
            warnings.warn(f"Skipping {path}: unexpected path depth")
            continue
        dataset_name = rel_parts[1]
        base = dataset_name.rsplit("_", 1)[0]

        # Determine which bucket this trace belongs to
        if ood_datasets is not None and base in ood_datasets:
            split = "ood"
        elif train_datasets is not None and base in train_datasets:
            split = _infer_split(dataset_name)
            if split is None or split == "ood":
                continue  # skip _ood-suffixed dirs when using explicit train list
        elif train_datasets is None and ood_datasets is None:
            # Legacy: accept all datasets, bucket by suffix
            split = _infer_split(dataset_name)
            if split is None:
                warnings.warn(f"Skipping {path}: unrecognised dataset '{dataset_name}'")
                continue
        else:
            continue  # not in either explicit list — skip
        try:
            with open(path) as f:
                episode = json.load(f)
            buckets[split][0].append(extract_features(episode))
            buckets[split][1].append(extract_sequence(episode))
            buckets[split][2].append(episode["meta"]["agent_id"])
            ds_names[split].add(dataset_name)
        except Exception as e:
            warnings.warn(f"Skipping {path.name}: {e}")
    return {s: tuple(lists) for s, lists in buckets.items()}, ds_names


class SequenceDataset(Dataset):
    def __init__(self, tok_seqs, time_seqs, rf_feats, labels_encoded):
        # tok_seqs:  list of 1-D LongTensor  (token IDs)
        # time_seqs: list of 2-D FloatTensor (seq_len, _N_CONTINUOUS)
        # rf_feats:  list of 1-D FloatTensor (n_rf_features,)
        # labels_encoded: 1-D LongTensor
        self.tok_seqs       = tok_seqs
        self.time_seqs      = time_seqs
        self.rf_feats       = rf_feats
        self.labels_encoded = labels_encoded

    def __len__(self):
        return len(self.tok_seqs)

    def __getitem__(self, idx):
        return self.tok_seqs[idx], self.time_seqs[idx], self.rf_feats[idx], self.labels_encoded[idx]


def collate_fn(batch):
    toks, times, rfs, lbls = zip(*batch)
    lengths      = torch.tensor([t.size(0) for t in toks], dtype=torch.long)
    padded_toks  = pad_sequence(toks,  batch_first=True, padding_value=0)
    padded_times = pad_sequence(times, batch_first=True, padding_value=0.0)  # (B, T, _N_CONTINUOUS)
    rf_batch     = torch.stack(rfs)                                           # (B, n_rf)
    return padded_toks, padded_times, lengths, rf_batch, torch.stack(lbls)


_N_CONTINUOUS = 4   # number of continuous scalars per event (f0..f3)


class AgentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, n_classes,
                 n_rf_features=0, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # LSTM input = token embedding + _N_CONTINUOUS continuous scalars per event
        self.lstm = nn.LSTM(embed_dim + _N_CONTINUOUS, hidden_dim, num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        # Head concatenates LSTM final state with pre-computed RF aggregate features
        self.head = nn.Linear(hidden_dim + n_rf_features, n_classes)

    def forward(self, toks, times, lengths, rf_feats):
        # toks:     (B, T)         — padded token IDs
        # times:    (B, T, _N_CONTINUOUS) — padded continuous scalars
        # lengths:  (B,)           — true sequence lengths
        # rf_feats: (B, n_rf)      — pre-computed aggregate features
        emb    = self.dropout(self.embedding(toks))      # (B, T, embed_dim)
        inp    = torch.cat([emb, times], dim=-1)         # (B, T, embed_dim+_N_CONTINUOUS)
        packed = pack_padded_sequence(inp, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        ctx      = self.dropout(h_n[-1])                 # (B, hidden_dim)
        combined = torch.cat([ctx, rf_feats], dim=-1)    # (B, hidden_dim + n_rf)
        return self.head(combined)


_LSTM_EMBED_DIM   = 16
_LSTM_N_LAYERS    = 2
_LSTM_BATCH_SIZE  = 16
_LSTM_LR          = 1e-3
_LSTM_WEIGHT_DECAY = 1e-4
_LSTM_N_EPOCHS    = 50
_LSTM_GRID = [
    {"hidden_dim": 64,  "dropout": 0.2},
    {"hidden_dim": 64,  "dropout": 0.4},
    {"hidden_dim": 128, "dropout": 0.2},
    {"hidden_dim": 128, "dropout": 0.4},
]


def _make_tensors(sequences):
    """Unpack list of (token, f0, f1, f2, f3) tuples into tok and time tensors."""
    tok_tensors  = [torch.tensor([e[0] for e in s], dtype=torch.long) for s in sequences]
    time_tensors = [torch.tensor([[e[1], e[2], e[3], e[4]] for e in s], dtype=torch.float)
                    for s in sequences]
    return tok_tensors, time_tensors


def _fit_lstm(seq_train, X_train, y_train, n_classes, hyperparams, n_epochs=_LSTM_N_EPOCHS):
    """Train one AgentLSTM config. X_train is a numpy array (N, n_rf_features)."""
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = hyperparams["hidden_dim"]
    dropout    = hyperparams["dropout"]
    n_rf       = X_train.shape[1]

    tok_tensors, time_tensors = _make_tensors(seq_train)
    rf_tensors = [torch.tensor(X_train[i], dtype=torch.float) for i in range(len(X_train))]
    lbl_tensor = torch.tensor(y_train, dtype=torch.long)
    dl = DataLoader(SequenceDataset(tok_tensors, time_tensors, rf_tensors, lbl_tensor),
                    batch_size=_LSTM_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model     = AgentLSTM(VOCAB_SIZE, _LSTM_EMBED_DIM, hidden_dim, _LSTM_N_LAYERS,
                          n_classes, n_rf_features=n_rf, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=_LSTM_LR, weight_decay=_LSTM_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(n_epochs):
        for p_toks, p_times, lengths, rf_batch, lbls in dl:
            p_toks   = p_toks.to(device)
            p_times  = p_times.to(device)
            lengths  = lengths.to(device)
            rf_batch = rf_batch.to(device)
            lbls     = lbls.to(device)
            optimizer.zero_grad()
            criterion(model(p_toks, p_times, lengths, rf_batch), lbls).backward()
            optimizer.step()
    return model


def _eval_lstm(model, seq_eval, X_eval, y_eval):
    """Evaluate a trained AgentLSTM. Returns (accuracy, predictions_list)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_tensors, time_tensors = _make_tensors(seq_eval)
    rf_tensors = [torch.tensor(X_eval[i], dtype=torch.float) for i in range(len(X_eval))]
    lbl_tensor = torch.tensor(y_eval, dtype=torch.long)
    dl = DataLoader(SequenceDataset(tok_tensors, time_tensors, rf_tensors, lbl_tensor),
                    batch_size=_LSTM_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for p_toks, p_times, lengths, rf_batch, lbls in dl:
            p_toks   = p_toks.to(device)
            p_times  = p_times.to(device)
            lengths  = lengths.to(device)
            rf_batch = rf_batch.to(device)
            lbls     = lbls.to(device)
            preds = model(p_toks, p_times, lengths, rf_batch).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_trues.extend(lbls.cpu().tolist())

    acc = sum(p == t for p, t in zip(all_preds, all_trues)) / len(all_trues)
    return float(acc), all_preds


def train_lstm(seq_train, X_train, y_train,
               seq_val,   X_val,   y_val,
               seq_test,  X_test,  y_test,
               n_classes, models_dir: Path = TRACE_DIR / "models",
               n_epochs=_LSTM_N_EPOCHS) -> dict:
    """Grid search over hidden_dim × dropout, pick best by val accuracy.

    Fits final model on train only. Returns best_params, val_report, test_report.
    test_report is None when seq_test is empty.
    """
    best_val_acc = -1.0
    best_params  = _LSTM_GRID[0]

    print("  LSTM grid search:")
    for params in _LSTM_GRID:
        model = _fit_lstm(seq_train, X_train, y_train, n_classes, params, n_epochs=n_epochs)
        val_acc, _ = _eval_lstm(model, seq_val, X_val, y_val)
        print(f"    hidden_dim={params['hidden_dim']}  dropout={params['dropout']}  "
              f"val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params  = params

    print(f"  Best: {best_params}  val_acc={best_val_acc:.3f}")

    final_model      = _fit_lstm(seq_train, X_train, y_train, n_classes, best_params, n_epochs=n_epochs)
    _, val_preds     = _eval_lstm(final_model, seq_val, X_val, y_val)
    val_report       = classification_report(y_val, val_preds, output_dict=True)
    test_report      = None
    if seq_test:
        _, test_preds = _eval_lstm(final_model, seq_test, X_test, y_test)
        test_report   = classification_report(y_test, test_preds, output_dict=True)

    torch.save(final_model.state_dict(), models_dir / "lstm_model.pt")
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


def train(trace_dir: Path, tag: str | None = None,
          train_datasets: list[str] | None = None,
          ood_datasets: list[str] | None = None) -> None:
    splits, ds_names = load_dataset(trace_dir, train_datasets=train_datasets,
                                    ood_datasets=ood_datasets)
    feat_train, seq_train, lbl_train = splits["train"]
    feat_val,   seq_val,   lbl_val   = splits["val"]
    feat_test,  seq_test,  lbl_test  = splits["test"]
    feat_ood,   seq_ood,   lbl_ood   = splits["ood"]

    # --- guards ---
    if not feat_train:
        print("ERROR: No training episodes found. Run more episodes first.")
        return
    if len(feat_train) < 10:
        print(f"WARNING: Only {len(feat_train)} training episodes — results will not be meaningful.")
    has_val  = bool(feat_val)
    has_test = bool(feat_test)
    has_ood  = bool(feat_ood)
    if not has_val:
        warnings.warn("No val episodes found — skipping hyperparameter tuning, using defaults.")
    if not has_test:
        warnings.warn("No test episodes found — test_report will be null.")

    # --- label encoder fitted on all labels across splits ---
    le = LabelEncoder()
    le.fit(lbl_train + lbl_val + lbl_test + lbl_ood)
    y_train = le.transform(lbl_train)
    y_val   = le.transform(lbl_val)  if lbl_val  else np.array([], dtype=int)
    y_test  = le.transform(lbl_test) if lbl_test else np.array([], dtype=int)
    y_ood   = le.transform(lbl_ood)  if lbl_ood  else np.array([], dtype=int)

    # --- feature matrices ---
    feat_names = list(feat_train[0].keys())
    def to_X(feats):
        return np.array([[ep[k] for k in feat_names] for ep in feats]) if feats \
               else np.empty((0, len(feat_names)))
    X_train = to_X(feat_train)
    X_val   = to_X(feat_val)
    X_test  = to_X(feat_test)
    X_ood   = to_X(feat_ood)

    # Scaled copies for the LSTM head — RF/GB are scale-invariant so they use raw X
    scaler    = StandardScaler().fit(X_train)
    Xs_train  = scaler.transform(X_train)
    Xs_val    = scaler.transform(X_val)  if has_val  else X_val
    Xs_test   = scaler.transform(X_test) if has_test else X_test
    Xs_ood    = scaler.transform(X_ood)  if has_ood  else X_ood

    # --- derive models_dir from train-split dataset base names (or --tag override) ---
    if tag is None:
        base_names = sorted({n.rsplit("_", 1)[0] for n in ds_names["train"]})
        tag = "_".join(base_names) if base_names else "unknown"
    models_dir = trace_dir / "models" / tag
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- RF / GradientBoosting ---
    clf_results = {}
    best_rf     = None
    best_gb     = None

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

        ood_report = None
        if has_ood:
            ood_preds  = best_clf.predict(X_ood)
            ood_report = classification_report(
                y_ood, ood_preds, target_names=le.classes_, output_dict=True)
            print(f"{name:20s}  ood_acc={ood_report['accuracy']:.3f}")  # type: ignore[index]

        clf_results[name] = {
            "best_params": best_params,
            "val_report":  val_report,
            "test_report": test_report,
            "ood_report":  ood_report,
        }
        if name == "RandomForest":    best_rf = best_clf
        if name == "GradientBoosting": best_gb = best_clf

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
            seq_train, Xs_train, list(y_train),
            seq_val,   Xs_val,   list(y_val),
            seq_test,  Xs_test,  list(y_test),
            n_classes, models_dir=models_dir,
        )
        lstm_result["val_report"]  = remap(lstm_result["val_report"])
        lstm_result["test_report"] = remap(lstm_result["test_report"])
        if lstm_result["val_report"]:
            print(f"{'LSTM':20s}  val_acc={lstm_result['val_report']['accuracy']:.3f}")
        if lstm_result["test_report"]:
            print(f"{'LSTM':20s}  test_acc={lstm_result['test_report']['accuracy']:.3f}")
        # OOD eval reuses the final trained model saved in train_lstm
        lstm_result["ood_report"] = None
        if has_ood:
            final_model = _fit_lstm(seq_train, Xs_train, list(y_train), n_classes,
                                    lstm_result["best_params"])
            _, ood_preds      = _eval_lstm(final_model, seq_ood, Xs_ood, list(y_ood))
            lstm_result["ood_report"] = remap(
                classification_report(list(y_ood), ood_preds, output_dict=True))
            print(f"{'LSTM':20s}  ood_acc={lstm_result['ood_report']['accuracy']:.3f}")  # type: ignore[index]
    else:
        default_params = {"hidden_dim": 64, "dropout": 0.3}
        model          = _fit_lstm(seq_train, Xs_train, list(y_train), n_classes, default_params)
        test_report    = None
        if has_test:
            _, test_preds = _eval_lstm(model, seq_test, Xs_test, list(y_test))
            test_report   = remap(classification_report(list(y_test), test_preds, output_dict=True))
        ood_report_lstm = None
        if has_ood:
            _, ood_preds    = _eval_lstm(model, seq_ood, Xs_ood, list(y_ood))
            ood_report_lstm = remap(classification_report(list(y_ood), ood_preds, output_dict=True))
            print(f"{'LSTM':20s}  ood_acc={ood_report_lstm['accuracy']:.3f}")  # type: ignore[index]
        torch.save(model.state_dict(), models_dir / "lstm_model.pt")
        lstm_result = {"best_params": default_params, "val_report": None,
                       "test_report": test_report, "ood_report": ood_report_lstm}

    clf_results["LSTM"] = lstm_result

    # --- save artefacts ---
    with open(models_dir / "classifier.pkl", "wb") as f:
        pickle.dump({"rf": best_rf, "gb": best_gb, "le": le,
                     "feat_names": feat_names, "scaler": scaler}, f)

    results = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "tag":            tag,
        "train_datasets": sorted(ds_names["train"]),
        "val_datasets":   sorted(ds_names["val"]),
        "test_datasets":  sorted(ds_names["test"]),
        "ood_datasets":   sorted(ds_names["ood"]),
        "n_episodes":     {"train": len(feat_train), "val": len(feat_val),
                           "test": len(feat_test), "ood": len(feat_ood)},
        "class_names":    list(le.classes_),
        "models":         clf_results,
    }
    results_path = models_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}  {models_dir}/classifier.pkl  {models_dir}/lstm_model.pt")

    # --- print classification reports ---
    def _print_report(model_name, split_name, report):
        print(f"\n{'─' * 60}")
        print(f"  {model_name}  [{split_name}]")
        print(f"{'─' * 60}")
        print(f"  {'':20}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}  {'support':>7}")
        print()
        for cls in le.classes_:
            if cls not in report: continue
            r = report[cls]
            print(f"  {cls:>20}  {r['precision']:>9.3f}  {r['recall']:>9.3f}"
                  f"  {r['f1-score']:>9.3f}  {int(r['support']):>7d}")
        print()
        for avg in ("macro avg", "weighted avg"):
            if avg not in report: continue
            r = report[avg]
            print(f"  {avg:>20}  {r['precision']:>9.3f}  {r['recall']:>9.3f}"
                  f"  {r['f1-score']:>9.3f}  {int(r['support']):>7d}")
        if "accuracy" in report:
            n = int(sum(report[c]["support"] for c in le.classes_ if c in report))
            print(f"\n  {'accuracy':>20}  {'':>9}  {'':>9}  {report['accuracy']:>9.3f}  {n:>7d}")

    for model_name, res in clf_results.items():
        for split_name in ("val", "test", "ood"):
            report = res.get(f"{split_name}_report")
            if report:
                _print_report(model_name, split_name, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate agent classifiers from collected traces.",
        epilog=(
            "Examples:\n"
            "  python trace_analyzer.py\n"
            "  python trace_analyzer.py --train-datasets 2wikimultihop\n"
            "  python trace_analyzer.py --train-datasets webshop\n"
            "  python trace_analyzer.py --train-datasets webshop --ood-datasets deepshop\n"
            "  python trace_analyzer.py --train-datasets 2wikimultihop --ood-datasets webshop --tag wiki_ood_amazon\n"
            "  python trace_analyzer.py --train-datasets webshop --ood-datasets 2wikimultihop --tag amazon_ood_wiki\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"),
                        help="Root traces directory (default: ./traces)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Models subdirectory name. Default: auto-derived from train dataset names.")
    parser.add_argument("--train-datasets", nargs="+", default=None,
                        metavar="NAME",
                        help="Base dataset names for train/val/test splits, e.g. --train-datasets 2wikimultihop. "
                             "Default: all datasets found in traces-dir.")
    parser.add_argument("--ood-datasets", nargs="+", default=None,
                        metavar="NAME",
                        help="Base dataset names to load as OOD (all suffixes → OOD bucket), "
                             "e.g. --ood-datasets webshop")
    cli = parser.parse_args()
    train(cli.traces_dir, tag=cli.tag, train_datasets=cli.train_datasets,
          ood_datasets=cli.ood_datasets)
