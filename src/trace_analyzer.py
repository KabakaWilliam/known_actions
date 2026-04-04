import json, pickle, warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
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
}
VOCAB_SIZE = len(EVENT_VOCAB)


def extract_features(episode) -> dict:
    dom    = episode.get("dom_trace", {})
    events = dom.get("events", [])
    mlog   = episode.get("midscene_log", [])

    clicks       = [e for e in events if e["type"] == "click"]
    scrolls      = [e for e in events if e["type"] == "scroll"]
    navs         = [e for e in events if e["type"] == "navigate"]
    keydowns     = [e for e in events if e["type"] == "keydown"]
    beforeunload = [e for e in events if e["type"] == "beforeunload"]

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

    # Navigation
    n_clicks           = len(clicks)
    n_scrolls          = len(scrolls)
    n_navigations      = len(navs)
    n_keydowns         = len(keydowns)
    n_beforeunload     = len(beforeunload)
    n_events_total     = len(events)
    n_midscene_actions = len(mlog)

    actions_per_page    = n_events_total / max(page_count, 1)
    nav_to_click_ratio  = n_navigations / max(n_clicks, 1)
    keydowns_per_page   = n_keydowns / max(page_count, 1)
    midscene_per_page   = n_midscene_actions / max(page_count, 1)

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
        "mean_exit_scroll_pct":   mean_exit_scroll_pct,
    }


def extract_sequence(episode) -> list:
    events = episode.get("dom_trace", {}).get("events", [])
    return [EVENT_VOCAB.get(e["type"], EVENT_VOCAB["<unk>"]) for e in events]


def load_dataset(trace_dir) -> tuple:
    features  = []
    sequences = []
    labels    = []
    # traces/{agent_id}/{timestamp}/episode.json — recurse two levels deep
    for path in sorted(trace_dir.rglob("*.json")):
        try:
            with open(path) as f:
                episode = json.load(f)
            features.append(extract_features(episode))
            sequences.append(extract_sequence(episode))
            labels.append(episode["meta"]["agent_id"])
        except Exception as e:
            warnings.warn(f"Skipping {path.name}: {e}")
    return features, sequences, labels


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


def train_lstm(sequences, labels_encoded, n_classes, n_epochs=30):
    EMBED_DIM  = 16
    HIDDEN_DIM = 64
    N_LAYERS   = 2
    BATCH_SIZE = 16
    LR         = 1e-3

    device = torch.device("cpu")

    seq_tensors = [torch.tensor(s, dtype=torch.long) for s in sequences]
    lbl_tensor  = torch.tensor(labels_encoded, dtype=torch.long)

    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(seq_tensors, labels_encoded)):
        train_seqs = [seq_tensors[i] for i in train_idx]
        val_seqs   = [seq_tensors[i] for i in val_idx]
        train_lbls = lbl_tensor[train_idx]
        val_lbls   = lbl_tensor[val_idx]

        train_ds = SequenceDataset(train_seqs, train_lbls)
        val_ds   = SequenceDataset(val_seqs,   val_lbls)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn)
        val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn)

        model     = AgentLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, N_LAYERS,
                              n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(n_epochs):
            for padded, lengths, lbls in train_dl:
                padded, lengths, lbls = padded.to(device), lengths.to(device), lbls.to(device)
                optimizer.zero_grad()
                logits = model(padded, lengths)
                loss   = criterion(logits, lbls)
                loss.backward()
                optimizer.step()

        # Evaluate on validation fold
        model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for padded, lengths, lbls in val_dl:
                padded, lengths, lbls = padded.to(device), lengths.to(device), lbls.to(device)
                logits = model(padded, lengths)
                preds  = logits.argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total   += lbls.size(0)
        fold_accs.append(correct / total if total > 0 else 0.0)

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))

    # Train final model on all data
    full_ds = SequenceDataset(seq_tensors, lbl_tensor)
    full_dl = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True,
                         collate_fn=collate_fn)
    model = AgentLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, N_LAYERS,
                      n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(n_epochs):
        for padded, lengths, lbls in full_dl:
            padded, lengths, lbls = padded.to(device), lengths.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(padded, lengths)
            loss   = criterion(logits, lbls)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), TRACE_DIR / "models" / "lstm_model.pt")

    return mean_acc, std_acc


def train(trace_dir):
    features, sequences, labels = load_dataset(trace_dir)

    if len(features) < 10:
        print(
            f"Only {len(features)} episodes loaded — results will not be "
            "meaningful. Run more episodes first."
        )
        # Continue anyway so the code path is tested.

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    # --- Random Forest ---
    feat_names = list(features[0].keys())
    X = np.array([[ep[k] for k in feat_names] for ep in features])

    rf  = RandomForestClassifier(n_estimators=200, random_state=42)
    gb  = GradientBoostingClassifier(n_estimators=100, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in [("RandomForest", rf), ("GradientBoosting", gb)]:
        scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
        print(f"{name:20s}  CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    rf.fit(X, y)
    importances = sorted(zip(feat_names, rf.feature_importances_),
                         key=lambda x: -x[1])
    print("\nTop 10 features (Random Forest):")
    for fname, imp in importances[:10]:
        print(f"  {fname:<30} {imp:.4f}")

    # --- LSTM ---
    print("\nTraining LSTM (5-fold CV) ...")
    n_classes = len(le.classes_)
    lstm_mean, lstm_std = train_lstm(sequences, y, n_classes)
    print(f"{'LSTM':20s}  CV accuracy: {lstm_mean:.3f} ± {lstm_std:.3f}")

    # Save Random Forest + label encoder for later inference
    models_dir = trace_dir / "models"
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "classifier.pkl", "wb") as f:
        pickle.dump({"rf": rf, "le": le, "feat_names": feat_names}, f)
    print("\nSaved: traces/models/classifier.pkl  traces/models/lstm_model.pt")


if __name__ == "__main__":
    train(TRACE_DIR)
