import argparse, json, pickle, warnings
from datetime import datetime, timezone
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_family_map(config_path: Path = _CONFIG_PATH) -> dict[str, str]:
    """Return {agent_id: family} from config.yaml. Agents without a family field are omitted."""
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        return {a["agent_id"]: a["family"]
                for a in cfg.get("agents", []) if "family" in a}
    except Exception:
        return {}


# Traces with these errors have no usable DOM events — exclude from training.
# Task-level failures (replanning limit, agent stuck) are still valid fingerprints.
_INVALID_TRACE_PATTERNS = [
    "credit balance is too low",
    "insufficient_quota",
    "invalid_api_key",
    "401 unauthorized",
    "402 payment",
    "failed to call ai model service",
]

def _is_valid_trace(episode: dict) -> bool:
    """Return False only for API-level failures that produce empty/useless traces."""
    err = (episode.get("error") or "").lower()
    if any(p in err for p in _INVALID_TRACE_PATTERNS):
        return False
    return bool((episode.get("dom_trace") or {}).get("events"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
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


def _type_ieis(events, event_type) -> list[float]:
    """Inter-event intervals for a single event type, in ms."""
    ts = [e.get("t_episode") or e.get("t") or 0
          for e in events if e["type"] == event_type]
    return np.diff(ts).tolist() if len(ts) > 1 else [0.0]


def extract_features(episode, n_events: int | None = None,
                     t_max_ms: int | None = None) -> dict:
    dom    = episode.get("dom_trace", {})
    events = dom.get("events", [])
    if t_max_ms is not None:
        events = [e for e in events if (e.get("t_episode") or e.get("t") or 0) <= t_max_ms]
    if n_events is not None:
        events = events[:n_events]
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

    # ── Global timing ─────────────────────────────────────────────────────────
    total_duration_s = (ts[-1] - ts[0]) / 1000 if len(ts) >= 2 else 0
    t_first_action_ms = float(ts[0]) if ts else 0.0
    mean_iei_ms      = float(np.mean(ieis))
    std_iei_ms       = float(np.std(ieis))
    median_iei_ms    = float(np.median(ieis))
    p10_iei_ms       = float(np.percentile(ieis, 10))
    p90_iei_ms       = float(np.percentile(ieis, 90))

    # IEI trend: does the agent slow down as context fills?
    # ratio > 1 means second half is slower than first (larger model slow-down)
    if len(ieis) >= 4:
        mid = len(ieis) // 2
        iei_trend = float(np.mean(ieis[mid:])) / max(float(np.mean(ieis[:mid])), 1.0)
    else:
        iei_trend = 1.0

    # ── Per-type IEIs (planning latency per action type) ──────────────────────
    click_ieis = _type_ieis(events, "click")
    mean_click_iei_ms = float(np.mean(click_ieis))
    std_click_iei_ms  = float(np.std(click_ieis))

    nav_ieis = _type_ieis(events, "navigate")
    mean_nav_iei_ms = float(np.mean(nav_ieis))   # ≈ page dwell time
    std_nav_iei_ms  = float(np.std(nav_ieis))
    max_page_dwell_ms = float(np.max(nav_ieis)) if len(nav_ieis) > 1 else 0.0

    key_ieis = _type_ieis(events, "keydown")
    mean_key_iei_ms = float(np.mean(key_ieis))   # ≈ API keystroke latency
    std_key_iei_ms  = float(np.std(key_ieis))

    # ── Scroll ────────────────────────────────────────────────────────────────
    max_scroll_pct  = max((e.get("pct") or 0 for e in scrolls), default=0)
    mean_scroll_pct = float(np.mean([e.get("pct") or 0 for e in scrolls])) if scrolls else 0.0
    n_deep_scrolls  = sum(1 for e in scrolls if (e.get("pct") or 0) > 60)

    pcts  = [e.get("pct") or 0 for e in scrolls]
    diffs = np.diff(pcts)
    scroll_reversals = int(np.sum((diffs[:-1] * diffs[1:]) < 0)) if len(diffs) > 1 else 0

    # ── Clicks ────────────────────────────────────────────────────────────────
    click_xs = [e.get("x") or 0 for e in clicks]
    click_ys = [e.get("y") or 0 for e in clicks]
    click_x_std = float(np.std(click_xs)) if click_xs else 0.0
    click_y_std = float(np.std(click_ys)) if click_ys else 0.0

    # Bounding-box coverage of click positions as fraction of viewport
    if len(click_xs) >= 2:
        click_bbox_area_frac = (
            (max(click_xs) - min(click_xs)) * (max(click_ys) - min(click_ys))
        ) / (1280.0 * 768.0)
    else:
        click_bbox_area_frac = 0.0

    # Fraction of clicks in the top quarter of the viewport (navbar / search bar area)
    click_top_frac = sum(1 for y in click_ys if y < 192) / max(len(click_ys), 1)

    n_link_clicks    = sum(1 for e in clicks if e.get("href"))
    link_click_ratio = n_link_clicks / max(len(clicks), 1)

    # ── Navigation ────────────────────────────────────────────────────────────
    n_clicks           = len(clicks)
    n_scrolls          = len(scrolls)
    n_navigations      = len(navs)
    n_keydowns         = len(keydowns)
    n_beforeunload     = len(beforeunload)
    n_focus            = len(focuses)
    n_events_total     = len(events)

    # Backtracking: popstate navigations vs. total (history.back() pattern)
    n_popstate    = sum(1 for e in navs if e.get("trigger") == "popstate")
    popstate_ratio = n_popstate / max(n_navigations, 1)

    actions_per_page    = n_events_total / max(page_count, 1)
    nav_to_click_ratio  = n_navigations / max(n_clicks, 1)
    keydowns_per_page   = n_keydowns / max(page_count, 1)
    focus_per_page      = n_focus / max(page_count, 1)

    # Structural keys (Enter, Arrow*, Tab, Escape) vs. printable char keys
    _structural = {"Enter", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
                   "Tab", "Escape", "Backspace", "Delete"}
    n_structural_keys    = sum(1 for e in keydowns if e.get("key") in _structural)
    structural_key_ratio = n_structural_keys / max(n_keydowns, 1)

    # Unique domains visited (subdomains count as same if same eTLD+1 is impractical,
    # so we use full hostname — simple and consistent across agents)
    from urllib.parse import urlparse
    nav_urls = [e.get("url") or "" for e in navs]
    n_unique_domains = len({urlparse(u).netloc for u in nav_urls if u})

    # Scroll-to-click ratio
    scroll_to_click_ratio = n_scrolls / max(n_clicks, 1)

    # ── Exit scroll depth ─────────────────────────────────────────────────────
    bu_pcts = [e.get("pct") or 0 for e in beforeunload]
    mean_exit_scroll_pct = float(np.mean(bu_pcts)) if bu_pcts else 0.0

    return {
        # Volume
        "n_clicks":              n_clicks,
        "n_scrolls":             n_scrolls,
        "n_navigations":         n_navigations,
        "n_keydowns":            n_keydowns,
        "n_focus":               n_focus,
        "n_events_total":        n_events_total,
        "page_count":            page_count,
        "n_unique_domains":      n_unique_domains,
        # Global timing
        "total_duration_s":      total_duration_s,
        "t_first_action_ms":     t_first_action_ms,
        "mean_iei_ms":           mean_iei_ms,
        "std_iei_ms":            std_iei_ms,
        "median_iei_ms":         median_iei_ms,
        "p10_iei_ms":            p10_iei_ms,
        "p90_iei_ms":            p90_iei_ms,
        "iei_trend":             iei_trend,
        # Per-type IEIs (planning latency by action type)
        "mean_click_iei_ms":     mean_click_iei_ms,
        "std_click_iei_ms":      std_click_iei_ms,
        "mean_nav_iei_ms":       mean_nav_iei_ms,
        "std_nav_iei_ms":        std_nav_iei_ms,
        "max_page_dwell_ms":     max_page_dwell_ms,
        "mean_key_iei_ms":       mean_key_iei_ms,
        "std_key_iei_ms":        std_key_iei_ms,
        # Scroll
        "max_scroll_pct":        max_scroll_pct,
        "mean_scroll_pct":       mean_scroll_pct,
        "n_deep_scrolls":        n_deep_scrolls,
        "scroll_reversals":      scroll_reversals,
        # Clicks
        "click_x_std":           click_x_std,
        "click_y_std":           click_y_std,
        "click_bbox_area_frac":  click_bbox_area_frac,
        "click_top_frac":        click_top_frac,
        "n_link_clicks":         n_link_clicks,
        "link_click_ratio":      link_click_ratio,
        # Navigation strategy
        "popstate_ratio":        popstate_ratio,
        "scroll_to_click_ratio": scroll_to_click_ratio,
        "actions_per_page":      actions_per_page,
        "nav_to_click_ratio":    nav_to_click_ratio,
        "keydowns_per_page":     keydowns_per_page,
        "focus_per_page":        focus_per_page,
        "structural_key_ratio":  structural_key_ratio,
        # Exit
        "mean_exit_scroll_pct":  mean_exit_scroll_pct,
    }


def extract_sequence(episode, n_events: int | None = None,
                     t_max_ms: int | None = None) -> list[tuple[int, float, float, float, float, float]]:
    """Return a list of (token_id, f0, f1, f2, f3, f4) per event.

    f0 = log1p(delta_t_ms)        — inter-event gap
    f1 = log1p(t_episode_ms)      — absolute position in session
    f2, f3 = event-specific spatial/depth scalars:
        scroll → (pct/100, 0)
        click  → (x/1280, y/768)
        other  → (0, 0)
    f4 = log1p(running_mean_iei_ms for this event type)
        — per-type planning latency signal; lets LSTM learn type-specific timing
          without having to implicitly isolate event types from the mixed sequence
    """
    events = episode.get("dom_trace", {}).get("events", [])
    if t_max_ms is not None:
        events = [e for e in events if (e.get("t_episode") or e.get("t") or 0) <= t_max_ms]
    if n_events is not None:
        events = events[:n_events]
    result  = []
    prev_t  = None
    # Running per-type IEI accumulators: {type: (sum_ieis, count)}
    type_prev_t: dict[str, float] = {}
    type_iei_sum: dict[str, float] = {}
    type_iei_cnt: dict[str, int]   = {}

    for e in events:
        etype = e["type"]
        token = EVENT_VOCAB.get(etype, EVENT_VOCAB["<unk>"])
        t     = e.get("t_episode") or e.get("t") or 0
        delta = (t - prev_t) if prev_t is not None else 0.0
        f0    = float(np.log1p(max(delta, 0)))
        f1    = float(np.log1p(max(t, 0)))
        if etype == "scroll":
            f2, f3 = (e.get("pct") or 0) / 100.0, 0.0
        elif etype == "click":
            f2, f3 = (e.get("x") or 0) / 1280.0, (e.get("y") or 0) / 768.0
        else:
            f2, f3 = 0.0, 0.0

        # Per-type running mean IEI
        if etype in type_prev_t:
            type_iei = t - type_prev_t[etype]
            type_iei_sum[etype] = type_iei_sum.get(etype, 0.0) + type_iei
            type_iei_cnt[etype] = type_iei_cnt.get(etype, 0) + 1
            running_mean = type_iei_sum[etype] / type_iei_cnt[etype]
        else:
            running_mean = 0.0
        type_prev_t[etype] = t
        f4 = float(np.log1p(max(running_mean, 0)))

        result.append((token, f0, f1, f2, f3, f4))
        prev_t = t
    return result


def load_dataset(trace_dir: Path,
                 train_datasets: list[str] | None = None,
                 ood_datasets: list[str] | None = None,
                 agents: list[str] | None = None,
                 open_set_agents: list[str] | None = None,
                 resplit_datasets: list[str] | None = None,
                 resplit_fracs: tuple[float, float, float] = (0.5, 0.25, 0.25),
                 resplit_seed: int = 42,
                 resplit_n_per_agent: int | None = None,
                 return_events: bool = False,
                 label_by: str = "agent",
                 ) -> tuple[dict[str, tuple], dict[str, set]]:
    """Load all episode traces, bucketed by split.

    Path pattern: traces/{agent_id}/{dataset_name}/{timestamp}/{episode_id}.json

    train_datasets: base names (suffix stripped) whose _train/_val/_test traces
      go into the train/val/test buckets. e.g. ["2wikimultihop"] loads only
      2wikimultihop_* traces for training.

    ood_datasets: base names whose traces all go into the OOD bucket regardless
      of suffix. e.g. ["webshop"] loads webshop_train/val/test all as OOD.

    resplit_datasets: subset of train_datasets whose traces lack explicit
      train/val/test directory splits (e.g. "frames" only has frames_test).
      All matching traces are pooled then stratified-split by agent into
      train/val/test using resplit_fracs (default 50/25/25 to match 2wiki).

    resplit_n_per_agent: if set, cap each agent's pool to this many episodes
      before splitting (sampled randomly). Use 300 to match 2wiki's ~150/75/75
      per-agent budget and keep the two experiments comparable.

    If both train_datasets and ood_datasets are None: legacy mode — all
      datasets, split by suffix.

    Returns (splits, ds_names) where:
      splits   = {"train": (features, sequences, labels, ds_bases[, raw_events]), "val": ..., ...}
        ds_bases is a list of base dataset names (one per episode), used for per-OOD-dataset eval
        raw_events is included only when return_events=True
      ds_names = {"train": {"2wikimultihop_train", ...}, "val": ..., ...}
    """
    resplit_datasets  = set(resplit_datasets or [])
    _open_set_agents  = set(open_set_agents or [])
    _family_map       = _load_family_map() if label_by == "family" else {}
    buckets: dict[str, tuple[list, list, list, list, list]] = {
        "train":    ([], [], [], [], []),
        "val":      ([], [], [], [], []),
        "test":     ([], [], [], [], []),
        "ood":      ([], [], [], [], []),
        "open_set": ([], [], [], [], []),
    }
    ds_names: dict[str, set] = {"train": set(), "val": set(), "test": set(),
                                 "ood": set(), "open_set": set()}
    # staging pool for datasets that need a post-hoc stratified split
    resplit_pool: list[tuple] = []  # (feat, seq, lbl, raw_agent_id, base, dataset_name, raw_events)

    for path in sorted(trace_dir.rglob("*.json")):
        rel_parts = path.relative_to(trace_dir).parts
        if rel_parts[0].startswith("classifiers"):
            continue  # skip results.json / classifier artefacts
        if len(rel_parts) < 2:
            warnings.warn(f"Skipping {path}: unexpected path depth")
            continue
        agent_id = rel_parts[0]
        # Open-set agents are routed to the "open_set" bucket; they are never trained on.
        # Apply the same dataset filter as for known agents so open-set traces come from
        # the same domain (e.g. 2wikimultihop only, not frames/webshop/webgames).
        if agent_id in _open_set_agents:
            dataset_name = rel_parts[1]
            base = dataset_name.rsplit("_", 1)[0]
            # Skip if this dataset is outside the configured train/ood scope
            if train_datasets is not None and base not in train_datasets:
                if ood_datasets is None or base not in ood_datasets:
                    continue
            try:
                with open(path) as f:
                    episode = json.load(f)
                feat       = extract_features(episode)
                seq        = extract_sequence(episode)
                lbl        = episode["meta"]["agent_id"]
                lbl        = _family_map.get(lbl, lbl)
                raw_events = episode.get("dom_trace", {}).get("events", [])
                buckets["open_set"][0].append(feat)
                buckets["open_set"][1].append(seq)
                buckets["open_set"][2].append(lbl)
                buckets["open_set"][3].append(base)
                buckets["open_set"][4].append(raw_events)
                ds_names["open_set"].add(dataset_name)
            except Exception as e:
                warnings.warn(f"Skipping {path.name}: {e}")
            continue
        if agents is not None and agent_id not in agents:
            continue
        dataset_name = rel_parts[1]
        base = dataset_name.rsplit("_", 1)[0]

        # Determine which bucket this trace belongs to
        if ood_datasets is not None and base in ood_datasets:
            split = "ood"
        elif train_datasets is not None and base in train_datasets:
            split = _infer_split(dataset_name)
            if split is None or (split == "ood" and base not in resplit_datasets):
                continue  # skip _ood-suffixed dirs unless dataset is being resplit
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
            if not _is_valid_trace(episode):
                continue
            feat         = extract_features(episode)
            seq          = extract_sequence(episode)
            raw_agent_id = episode["meta"]["agent_id"]
            lbl          = _family_map.get(raw_agent_id, raw_agent_id)
            raw_events   = episode.get("dom_trace", {}).get("events", [])
            if base in resplit_datasets:
                # store raw_agent_id separately so resplit caps per-agent, not per-family
                resplit_pool.append((feat, seq, lbl, raw_agent_id, base, dataset_name, raw_events))
            else:
                buckets[split][0].append(feat)
                buckets[split][1].append(seq)
                buckets[split][2].append(lbl)
                buckets[split][3].append(base)
                buckets[split][4].append(raw_events)
                ds_names[split].add(dataset_name)
        except Exception as e:
            warnings.warn(f"Skipping {path.name}: {e}")

    # Stratified split for resplit_datasets — group by raw agent_id (not family label)
    # so resplit_n_per_agent caps per individual checkpoint, not per family class.
    if resplit_pool:
        import random as _random
        rng = _random.Random(resplit_seed)
        by_agent: dict[str, list] = {}
        for item in resplit_pool:
            by_agent.setdefault(item[3], []).append(item)  # item[3] = raw_agent_id
        tr_f, va_f, _ = resplit_fracs
        for agent_items in by_agent.values():
            rng.shuffle(agent_items)
            if resplit_n_per_agent is not None:
                agent_items = agent_items[:resplit_n_per_agent]
            n = len(agent_items)
            n_train = int(n * tr_f)
            n_val   = int(n * va_f)
            assignments = (
                [("train", i) for i in agent_items[:n_train]] +
                [("val",   i) for i in agent_items[n_train:n_train + n_val]] +
                [("test",  i) for i in agent_items[n_train + n_val:]]
            )
            for dest, (feat, seq, lbl, _aid, base, dataset_name, raw_events) in assignments:
                buckets[dest][0].append(feat)
                buckets[dest][1].append(seq)
                buckets[dest][2].append(lbl)
                buckets[dest][3].append(base)
                buckets[dest][4].append(raw_events)
                ds_names[dest].add(dataset_name)

    if return_events:
        return {s: tuple(lists) for s, lists in buckets.items()}, ds_names
    # Always return 4-tuples (drop raw_events), but keep "open_set" in the dict
    return {s: tuple(lists[:4]) for s, lists in buckets.items()}, ds_names


def load_from_hub(
    hf_repo: str,
    train_datasets: list[str] | None = None,
    ood_datasets: list[str] | None = None,
    agents: list[str] | None = None,
    open_set_agents: list[str] | None = None,
    resplit_datasets: list[str] | None = None,
    resplit_fracs: tuple[float, float, float] = (0.5, 0.25, 0.25),
    resplit_seed: int = 42,
    resplit_n_per_agent: int | None = None,
    return_events: bool = False,
    label_by: str = "agent",
    token: str | None = None,
) -> tuple[dict[str, tuple], dict[str, set]]:
    """Load episodes from a HuggingFace dataset repo; returns same structure as load_dataset()."""
    from datasets import load_dataset as hf_load

    _open_set_agents = set(open_set_agents or [])
    _family_map      = _load_family_map() if label_by == "family" else {}
    _resplit         = set(resplit_datasets or [])

    buckets: dict[str, tuple[list, list, list, list, list]] = {
        "train":    ([], [], [], [], []),
        "val":      ([], [], [], [], []),
        "test":     ([], [], [], [], []),
        "ood":      ([], [], [], [], []),
        "open_set": ([], [], [], [], []),
    }
    ds_names: dict[str, set] = {"train": set(), "val": set(), "test": set(),
                                 "ood": set(), "open_set": set()}
    resplit_pool: list[tuple] = []

    all_bases: set[str] = set(train_datasets or []) | set(ood_datasets or [])
    if not all_bases:
        raise ValueError("load_from_hub requires --train-datasets and/or --ood-datasets")

    for base in sorted(all_bases):
        is_ood_only = bool(ood_datasets and base in ood_datasets and
                           (not train_datasets or base not in train_datasets))
        is_resplit  = base in _resplit

        if is_ood_only:
            hf_splits = ["train", "val", "test", "ood"]
        elif is_resplit:
            hf_splits = ["test"]
        else:
            hf_splits = ["train", "val", "test"]

        for hf_split in hf_splits:
            try:
                ds = hf_load(hf_repo, base, split=hf_split, token=token)
            except Exception:
                continue

            for row in ds:
                agent_id     = row["agent_id"]
                dataset_name = row["dataset_name"]
                raw_events   = json.loads(row["dom_events_json"])

                episode = {
                    "meta":         json.loads(row["meta_json"]),
                    "dom_trace":    {"events": raw_events,
                                     "episodeDuration": row["duration_ms"]},
                    "verification": {"correct":      row["correct"],
                                     "ground_truth": row["ground_truth"],
                                     "predicted":    row["predicted_answer"]},
                    "error":        row["error"] or None,
                }

                if agent_id in _open_set_agents:
                    if not _is_valid_trace(episode):
                        continue
                    feat = extract_features(episode)
                    seq  = extract_sequence(episode)
                    lbl  = _family_map.get(agent_id, agent_id)
                    buckets["open_set"][0].append(feat)
                    buckets["open_set"][1].append(seq)
                    buckets["open_set"][2].append(lbl)
                    buckets["open_set"][3].append(base)
                    buckets["open_set"][4].append(raw_events)
                    ds_names["open_set"].add(dataset_name)
                    continue

                if agents is not None and agent_id not in agents:
                    continue
                if not _is_valid_trace(episode):
                    continue

                feat         = extract_features(episode)
                seq          = extract_sequence(episode)
                raw_agent_id = agent_id
                lbl          = _family_map.get(raw_agent_id, raw_agent_id)

                if is_ood_only or (ood_datasets and base in ood_datasets):
                    dest = "ood"
                elif is_resplit:
                    resplit_pool.append((feat, seq, lbl, raw_agent_id, base, dataset_name, raw_events))
                    continue
                else:
                    dest = _infer_split(dataset_name) or hf_split
                    if dest not in buckets:
                        continue

                buckets[dest][0].append(feat)
                buckets[dest][1].append(seq)
                buckets[dest][2].append(lbl)
                buckets[dest][3].append(base)
                buckets[dest][4].append(raw_events)
                ds_names[dest].add(dataset_name)

    if resplit_pool:
        import random as _random
        rng = _random.Random(resplit_seed)
        by_agent: dict[str, list] = {}
        for item in resplit_pool:
            by_agent.setdefault(item[3], []).append(item)
        tr_f, va_f, _ = resplit_fracs
        for agent_items in by_agent.values():
            rng.shuffle(agent_items)
            if resplit_n_per_agent is not None:
                agent_items = agent_items[:resplit_n_per_agent]
            n       = len(agent_items)
            n_train = int(n * tr_f)
            n_val   = int(n * va_f)
            assignments = (
                [("train", i) for i in agent_items[:n_train]] +
                [("val",   i) for i in agent_items[n_train:n_train + n_val]] +
                [("test",  i) for i in agent_items[n_train + n_val:]]
            )
            for dest, (feat, seq, lbl, _aid, base, dataset_name, raw_events) in assignments:
                buckets[dest][0].append(feat)
                buckets[dest][1].append(seq)
                buckets[dest][2].append(lbl)
                buckets[dest][3].append(base)
                buckets[dest][4].append(raw_events)
                ds_names[dest].add(dataset_name)

    if return_events:
        return {s: tuple(lists) for s, lists in buckets.items()}, ds_names
    return {s: tuple(lists[:4]) for s, lists in buckets.items()}, ds_names


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
    # pack_padded_sequence rejects length-0 entries (can occur at tiny prefix sizes);
    # clamp to 1 so empty sequences are treated as a single all-zero event.
    lengths      = lengths.clamp(min=1)
    rf_batch     = torch.stack(rfs)                                           # (B, n_rf)
    return padded_toks, padded_times, lengths, rf_batch, torch.stack(lbls)


_N_CONTINUOUS = 5   # number of continuous scalars per event (f0..f4)


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
    {"hidden_dim": 256, "dropout": 0.2},
    {"hidden_dim": 256, "dropout": 0.4},
    {"hidden_dim": 512, "dropout": 0.2},
    {"hidden_dim": 512, "dropout": 0.4},
]


def _make_tensors(sequences):
    """Unpack list of (token, f0..f4) tuples into tok and time tensors."""
    tok_tensors  = [torch.tensor([e[0] for e in s], dtype=torch.long) for s in sequences]
    # reshape(-1, _N_CONTINUOUS) ensures shape (T, 5) even when T==0 (empty prefix),
    # which pad_sequence requires to infer the feature dimension.
    time_tensors = [torch.tensor([[e[1], e[2], e[3], e[4], e[5]] for e in s], dtype=torch.float
                                 ).reshape(-1, _N_CONTINUOUS)
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


def _eval_lstm(model, seq_eval, X_eval, y_eval, return_proba: bool = False):
    """Evaluate a trained AgentLSTM.

    Returns (accuracy, predictions_list) by default.
    When return_proba=True, returns (accuracy, predictions_list, max_softmax_scores_list).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_tensors, time_tensors = _make_tensors(seq_eval)
    rf_tensors = [torch.tensor(X_eval[i], dtype=torch.float) for i in range(len(X_eval))]
    lbl_tensor = torch.tensor(y_eval, dtype=torch.long)
    dl = DataLoader(SequenceDataset(tok_tensors, time_tensors, rf_tensors, lbl_tensor),
                    batch_size=_LSTM_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_preds, all_trues, all_confs = [], [], []
    with torch.no_grad():
        for p_toks, p_times, lengths, rf_batch, lbls in dl:
            p_toks   = p_toks.to(device)
            p_times  = p_times.to(device)
            lengths  = lengths.to(device)
            rf_batch = rf_batch.to(device)
            lbls     = lbls.to(device)
            logits   = model(p_toks, p_times, lengths, rf_batch)
            preds    = logits.argmax(dim=1)
            if return_proba:
                confs = torch.nn.functional.softmax(logits, dim=1).max(dim=1).values
                all_confs.extend(confs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_trues.extend(lbls.cpu().tolist())

    acc = sum(p == t for p, t in zip(all_preds, all_trues)) / len(all_trues)
    if return_proba:
        return float(acc), all_preds, all_confs
    return float(acc), all_preds


def train_lstm(seq_train, X_train, y_train,
               seq_val,   X_val,   y_val,
               seq_test,  X_test,  y_test,
               n_classes, models_dir: Path = TRACE_DIR / "classifiers",
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
    "n_estimators":      [200, 400],
    "max_depth":         [None, 15, 30],
    "max_features":      ["sqrt", "log2", 0.4],
    "min_samples_split": [2, 5],
}
XGB_PARAM_DIST = {
    "n_estimators":     [100, 200, 300, 400, 500],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2, 0.3],
    "max_depth":        [3, 4, 5, 6, 7, 8],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
    "reg_alpha":        [0, 0.01, 0.1, 1.0],
    "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
}
LR_PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
}

_PREFIX_SIZES_EVENTS = [5, 10, 20, 30, 50, 75, 100, 150, None]
_PREFIX_SIZES_MS     = [1000, 2000, 5000, 10000, 20000, 30000, None]


def eval_prefix_curve(
    trained_clfs: dict,
    lstm_model,
    raw_events_test: list,
    raw_events_ood:  list,
    lbl_test: list,
    lbl_ood:  list,
    ds_bases_ood: list,
    le,
    feat_names: list,
    scaler,
    prefix_sizes_events=None,
    prefix_sizes_ms=None,
) -> dict:
    """Evaluate all classifiers at truncated trace prefixes.

    Trains on full traces (already done); here we only re-extract features and
    sequences at each prefix size and run inference to measure how quickly the
    models can identify each agent.

    Returns a nested dict suitable for JSON serialisation under 'prefix_curve'.
    """
    if prefix_sizes_events is None:
        prefix_sizes_events = _PREFIX_SIZES_EVENTS
    if prefix_sizes_ms is None:
        prefix_sizes_ms = _PREFIX_SIZES_MS

    y_test = le.transform(lbl_test) if lbl_test else np.array([], dtype=int)
    y_ood  = le.transform(lbl_ood)  if lbl_ood  else np.array([], dtype=int)
    ds_bases_ood_arr = np.array(ds_bases_ood)
    ood_base_names   = sorted(set(ds_bases_ood))

    # Classifiers that were trained on unscaled features (RF, XGBoost are scale-invariant)
    _UNSCALED_CLFS = {"RandomForest", "XGBoost"}

    def _fake_ep(raw_ev):
        return {"dom_trace": {"events": raw_ev}}

    def _build_X(raw_ev_list, n_ev=None, t_ms=None):
        """Return unscaled feature matrix."""
        feats = [extract_features(_fake_ep(ev), n_events=n_ev, t_max_ms=t_ms)
                 for ev in raw_ev_list]
        return np.array([[f.get(k, 0.0) for k in feat_names] for f in feats])

    def _build_Xs(raw_ev_list, n_ev=None, t_ms=None):
        """Return scaler-transformed feature matrix (for LR, LSTM)."""
        return scaler.transform(_build_X(raw_ev_list, n_ev=n_ev, t_ms=t_ms))

    def _build_seqs(raw_ev_list, n_ev=None, t_ms=None):
        return [extract_sequence(_fake_ep(ev), n_events=n_ev, t_max_ms=t_ms)
                for ev in raw_ev_list]

    def _per_class_f1(report):
        return {cls: report.get(cls, {}).get("f1-score", 0.0)
                for cls in le.classes_}

    def _macro_f1(per_class: dict) -> float:
        vals = list(per_class.values())
        return float(np.mean(vals)) if vals else 0.0

    def _split_report(y_true, preds):
        rep = classification_report(y_true, preds, target_names=le.classes_,
                                    output_dict=True, zero_division=0)
        pcf = _per_class_f1(rep)
        return {"accuracy": rep["accuracy"],
                "macro_f1": _macro_f1(pcf),
                "per_class_f1": pcf}

    def _eval_one(size_key, X_te, Xs_te, seqs_te, X_od, Xs_od, seqs_od):
        result = {}
        for clf_name, clf in trained_clfs.items():
            if clf is None:
                continue
            # RF/XGBoost trained on raw features; LR trained on scaled features
            Xtr_te = X_te  if clf_name in _UNSCALED_CLFS else Xs_te
            Xtr_od = X_od  if clf_name in _UNSCALED_CLFS else Xs_od
            entry = {}
            if len(Xtr_te):
                entry["test"] = _split_report(y_test, clf.predict(Xtr_te))
            ood_entry = {}
            for oname in ood_base_names:
                mask     = ds_bases_ood_arr == oname
                Xs_sub   = Xtr_od[mask]
                y_sub    = y_ood[mask]
                if not len(Xs_sub):
                    continue
                ood_entry[oname] = _split_report(y_sub, clf.predict(Xs_sub))
            entry["ood"] = ood_entry
            result[clf_name] = entry

        # LSTM uses scaled features
        if lstm_model is not None and len(seqs_te):
            _, preds_te = _eval_lstm(lstm_model, seqs_te, Xs_te, list(y_test))
            lstm_entry  = {"test": _split_report(list(y_test), preds_te)}
            lstm_ood    = {}
            for oname in ood_base_names:
                mask        = ds_bases_ood_arr == oname
                seqs_sub    = [s for s, m in zip(seqs_od, mask) if m]
                Xs_sub      = Xs_od[mask]
                y_sub       = [int(v) for v, m in zip(y_ood.tolist(), mask) if m]
                if not seqs_sub:
                    continue
                _, preds_od = _eval_lstm(lstm_model, seqs_sub, Xs_sub, y_sub)
                lstm_ood[oname] = _split_report(y_sub, preds_od)
            lstm_entry["ood"] = lstm_ood
            result["LSTM"] = lstm_entry
        return result

    curve: dict = {"n_events": {}, "t_ms": {}}

    # ── n_events sweep ──────────────────────────────────────────────────────
    print("  prefix curve — n_events:", end="", flush=True)
    for size in prefix_sizes_events:
        key = str(size) if size is not None else "null"
        print(f" {key}", end="", flush=True)
        n_ev    = size
        X_te    = _build_X (raw_events_test, n_ev=n_ev)
        Xs_te   = _build_Xs(raw_events_test, n_ev=n_ev)
        seqs_te = _build_seqs(raw_events_test, n_ev=n_ev)
        X_od    = _build_X (raw_events_ood, n_ev=n_ev)
        Xs_od   = _build_Xs(raw_events_ood, n_ev=n_ev)
        seqs_od = _build_seqs(raw_events_ood, n_ev=n_ev)
        per_clf = _eval_one(key, X_te, Xs_te, seqs_te, X_od, Xs_od, seqs_od)
        for clf_name, entry in per_clf.items():
            curve["n_events"].setdefault(clf_name, {})[key] = entry
    print()

    # ── t_ms sweep ──────────────────────────────────────────────────────────
    print("  prefix curve — t_ms:", end="", flush=True)
    for size in prefix_sizes_ms:
        key = str(size) if size is not None else "null"
        print(f" {key}", end="", flush=True)
        t_ms    = size
        X_te    = _build_X (raw_events_test, t_ms=t_ms)
        Xs_te   = _build_Xs(raw_events_test, t_ms=t_ms)
        seqs_te = _build_seqs(raw_events_test, t_ms=t_ms)
        X_od    = _build_X (raw_events_ood, t_ms=t_ms)
        Xs_od   = _build_Xs(raw_events_ood, t_ms=t_ms)
        seqs_od = _build_seqs(raw_events_ood, t_ms=t_ms)
        per_clf = _eval_one(key, X_te, Xs_te, seqs_te, X_od, Xs_od, seqs_od)
        for clf_name, entry in per_clf.items():
            curve["t_ms"].setdefault(clf_name, {})[key] = entry
    print()

    return curve


def eval_open_set(
    trained_clfs: dict,
    lstm_model,
    X_known: np.ndarray,    # unscaled (for RF/XGBoost)
    Xs_known: np.ndarray,   # scaler-transformed (for LR/LSTM)
    X_unknown: np.ndarray,
    Xs_unknown: np.ndarray,
    seqs_known: list,
    seqs_unknown: list,
    y_known: np.ndarray,
    le,
    open_set_agent_names: list[str],
) -> dict:
    """Measure how well each classifier can distinguish known agents from unknown ones.

    Classifiers are trained on known agents only. At eval time we mix their test
    traces with traces from the held-out unknown agents and report:
      - AUROC  — area under the ROC curve for in-set (1) vs out-of-set (0); threshold-independent
      - FPR95  — false positive rate at TPR=0.95 (standard OOD benchmark)
      - binary_report — precision/recall/F1 at the Youden threshold (maximises TPR−FPR)
      - closed_set_accuracy — accuracy on known agents (unknowns excluded)
      - open_set_accuracy   — accuracy when samples below the FPR95 threshold are rejected

    Confidence signal: max(softmax) / max(predict_proba) — higher = more confident
    the model believes the trace belongs to a known class.
    """
    from sklearn.metrics import roc_curve

    n_known   = len(X_known)
    n_unknown = len(X_unknown)
    if n_known == 0 or n_unknown == 0:
        return {}

    # Binary ground truth: 1 = known (in-set), 0 = unknown (out-of-set)
    in_labels = np.array([1] * n_known + [0] * n_unknown)

    def _fpr_at_tpr(fprs, tprs, target_tpr=0.95):
        for fpr, tpr in zip(fprs, tprs):
            if tpr >= target_tpr:
                return float(fpr)
        return 1.0

    def _youden_threshold(fprs, tprs, thresholds):
        """Threshold that maximises Youden's J = TPR − FPR (balanced binary F1 proxy)."""
        j_scores = tprs - fprs
        best_idx = int(np.argmax(j_scores))
        return float(thresholds[best_idx])

    def _binary_report(conf_known, conf_unknown, threshold):
        """Binary classification report: known (1) vs unknown (0) at a given threshold."""
        y_true = [1] * len(conf_known) + [0] * len(conf_unknown)
        y_pred = [1 if s >= threshold else 0 for s in list(conf_known) + list(conf_unknown)]
        rep = classification_report(y_true, y_pred,
                                    target_names=["unknown", "known"],
                                    output_dict=True, zero_division=0)
        return {
            "accuracy":       rep["accuracy"],
            "known_precision":   rep["known"]["precision"],
            "known_recall":      rep["known"]["recall"],
            "known_f1":          rep["known"]["f1-score"],
            "unknown_precision": rep["unknown"]["precision"],
            "unknown_recall":    rep["unknown"]["recall"],
            "unknown_f1":        rep["unknown"]["f1-score"],
            "macro_f1":          rep["macro avg"]["f1-score"],
        }

    result = {}

    def _package(auroc, fpr95, threshold, bin_rep):
        return {
            "auroc":            auroc,
            "fpr95":            fpr95,
            "youden_threshold": threshold,
            "binary_report":    bin_rep,
            "n_known":          n_known,
            "n_unknown":        n_unknown,
        }

    # ── sklearn classifiers ───────────────────────────────────────────────────
    # RF/XGBoost trained on unscaled features; LR trained on scaled features.
    _UNSCALED = {"RandomForest", "XGBoost"}
    for clf_name, clf in trained_clfs.items():
        if clf is None:
            continue
        Xk = X_known   if clf_name in _UNSCALED else Xs_known
        Xu = X_unknown if clf_name in _UNSCALED else Xs_unknown
        conf_known   = clf.predict_proba(Xk).max(axis=1)
        conf_unknown = clf.predict_proba(Xu).max(axis=1)
        scores       = np.concatenate([conf_known, conf_unknown])

        auroc                    = float(roc_auc_score(in_labels, scores))
        fprs, tprs, thresholds   = roc_curve(in_labels, scores, pos_label=1)
        fpr95                    = _fpr_at_tpr(fprs, tprs)
        threshold                = _youden_threshold(fprs, tprs, thresholds)
        result[clf_name]         = _package(auroc, fpr95, threshold,
                                            _binary_report(conf_known, conf_unknown, threshold))

    # ── LSTM ──────────────────────────────────────────────────────────────────
    if lstm_model is not None:
        dummy_y = np.zeros(n_unknown, dtype=int)
        _, _, conf_known   = _eval_lstm(lstm_model, seqs_known,   Xs_known,   list(y_known), return_proba=True)
        _, _, conf_unknown = _eval_lstm(lstm_model, seqs_unknown, Xs_unknown, list(dummy_y), return_proba=True)

        conf_known_arr   = np.array(conf_known)
        conf_unknown_arr = np.array(conf_unknown)
        scores           = np.concatenate([conf_known_arr, conf_unknown_arr])

        auroc                  = float(roc_auc_score(in_labels, scores))
        fprs, tprs, thresholds = roc_curve(in_labels, scores, pos_label=1)
        fpr95                  = _fpr_at_tpr(fprs, tprs)
        threshold              = _youden_threshold(fprs, tprs, thresholds)
        result["LSTM"]         = _package(auroc, fpr95, threshold,
                                          _binary_report(conf_known_arr, conf_unknown_arr, threshold))

    return result


def train(trace_dir: Path, tag: str | None = None,
          train_datasets: list[str] | None = None,
          ood_datasets: list[str] | None = None,
          agents: list[str] | None = None,
          open_set_agents: list[str] | None = None,
          resplit_datasets: list[str] | None = None,
          resplit_n_per_agent: int | None = None,
          prefix_eval: bool = False,
          label_by: str = "agent",
          hf_repo: str | None = None,
          hf_token: str | None = None) -> None:
    if hf_repo:
        splits, ds_names = load_from_hub(hf_repo, token=hf_token,
                                         train_datasets=train_datasets,
                                         ood_datasets=ood_datasets, agents=agents,
                                         open_set_agents=open_set_agents,
                                         resplit_datasets=resplit_datasets,
                                         resplit_n_per_agent=resplit_n_per_agent,
                                         return_events=prefix_eval,
                                         label_by=label_by)
    else:
        splits, ds_names = load_dataset(trace_dir, train_datasets=train_datasets,
                                        ood_datasets=ood_datasets, agents=agents,
                                        open_set_agents=open_set_agents,
                                        resplit_datasets=resplit_datasets,
                                        resplit_n_per_agent=resplit_n_per_agent,
                                        return_events=prefix_eval,
                                        label_by=label_by)
    if prefix_eval:
        feat_train, seq_train, lbl_train, _,            *_ = splits["train"]
        feat_val,   seq_val,   lbl_val,   _,            *_ = splits["val"]
        feat_test, seq_test, lbl_test, _,            raw_events_test = splits["test"]
        feat_ood,  seq_ood,  lbl_ood,  ds_bases_ood, raw_events_ood  = splits["ood"]
    else:
        feat_train, seq_train, lbl_train, _            = splits["train"]
        feat_val,   seq_val,   lbl_val,   _            = splits["val"]
        feat_test, seq_test, lbl_test, _             = splits["test"]
        feat_ood,  seq_ood,  lbl_ood,  ds_bases_ood  = splits["ood"]
        raw_events_test = raw_events_ood = None
    # Open-set split: slice [:4] so this is safe whether return_events is on or off
    feat_os, seq_os, lbl_os, _ = splits["open_set"][:4]

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

    # Unique OOD dataset base names in order — used for per-dataset OOD evaluation
    ood_base_names = sorted(set(ds_bases_ood)) if ds_bases_ood else []
    ds_bases_ood_arr = np.array(ds_bases_ood)  # for numpy boolean indexing

    # Scaled copies for the LSTM head — RF/GB are scale-invariant so they use raw X
    scaler    = StandardScaler().fit(X_train)
    Xs_train  = scaler.transform(X_train)
    Xs_val    = scaler.transform(X_val)  if has_val  else X_val
    Xs_test   = scaler.transform(X_test) if has_test else X_test
    Xs_ood    = scaler.transform(X_ood)  if has_ood  else X_ood

    # Pre-compute open-set arrays so we can print per-classifier AUROC inline
    _os_ready = bool(open_set_agents and feat_os and feat_test)
    if _os_ready:
        _X_os_raw  = to_X(feat_os)
        _X_os_scl  = scaler.transform(_X_os_raw)
        _X_te_raw  = to_X(feat_test)
        _X_te_scl  = scaler.transform(_X_te_raw)
        _y_te_enc  = le.transform(lbl_test)
        # Binary labels: known=1, unknown=0
        _in_labels = np.concatenate([np.ones(len(_y_te_enc)),
                                     np.zeros(len(feat_os))])

    # --- derive models_dir from train-split dataset base names (or --tag override) ---
    if tag is None:
        base_names = sorted({n.rsplit("_", 1)[0] for n in ds_names["train"]})
        tag = "_".join(base_names) if base_names else "unknown"
    models_dir = trace_dir / "classifiers" / tag
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Sklearn classifiers ---
    clf_results  = {}
    best_rf      = None
    best_xgb     = None
    best_lr_l2   = None

    # (name, estimator, param_dist, uses_scaled_features, use_random_search)
    # RF: exhaustive grid — small grid, parallelises well with n_jobs=-1
    # XGBoost: randomised — large search space, GPU-accelerated per fit
    # LR: exhaustive — only 4 C values
    _clf_specs = [
        ("RandomForest",
         RandomForestClassifier(n_estimators=200, random_state=42),
         RF_PARAM_GRID, False, False),
        # LR variants use StandardScaler-normalized features
        # sklearn 1.8+: penalty deprecated; use l1_ratio (0=L2, 1=L1)
        ("LR_L2",
         LogisticRegression(solver="lbfgs", l1_ratio=0.0, max_iter=5000, random_state=42),
         LR_PARAM_GRID, True, False),
        ("LR_Lasso",
         LogisticRegression(solver="saga",  l1_ratio=1.0, max_iter=5000, random_state=42),
         LR_PARAM_GRID, True, False),
    ]
    if _XGBOOST_AVAILABLE:
        _clf_specs.append((
            "XGBoost",
            XGBClassifier(tree_method="hist", device="cpu",
                          eval_metric="mlogloss", random_state=42, verbosity=0),
            XGB_PARAM_DIST, False, True,   # randomised search
        ))

    for name, base_estimator, param_dist, scaled, use_random in _clf_specs:
        Xtr = Xs_train if scaled else X_train
        Xvl = Xs_val   if scaled else X_val
        Xte = Xs_test  if scaled else X_test
        Xod = Xs_ood   if scaled else X_ood

        if has_val:
            if use_random:
                gs = RandomizedSearchCV(base_estimator, param_dist, n_iter=40,
                                        cv=3, scoring="accuracy",
                                        n_jobs=-1, refit=True, random_state=42)
            else:
                gs = GridSearchCV(base_estimator, param_dist,
                                  cv=3, scoring="accuracy", n_jobs=-1, refit=True)
            gs.fit(Xtr, y_train)
            best_clf    = gs.best_estimator_
            best_params = gs.best_params_
            val_preds   = best_clf.predict(Xvl)
            val_report  = classification_report(
                y_val, val_preds, target_names=le.classes_, output_dict=True)
            print(f"{name:20s}  best={best_params}  "
                  f"val_acc={val_report['accuracy']:.3f}")
        else:
            best_clf = base_estimator.fit(Xtr, y_train)
            best_params, val_report = {}, None

        test_report = None
        if has_test:
            test_preds  = best_clf.predict(Xte)
            test_report = classification_report(
                y_test, test_preds, target_names=le.classes_, output_dict=True)
            _auroc_str = ""
            if _os_ready and hasattr(best_clf, "predict_proba"):
                _Xte_os = _X_te_scl if scaled else _X_te_raw
                _Xos_os = _X_os_scl if scaled else _X_os_raw
                _scores  = np.concatenate([
                    best_clf.predict_proba(_Xte_os).max(axis=1),
                    best_clf.predict_proba(_Xos_os).max(axis=1),
                ])
                _auroc_str = f"  auroc={roc_auc_score(_in_labels, _scores):.3f}"
            print(f"{name:20s}  test_acc={test_report['accuracy']:.3f}{_auroc_str}")

        ood_reports = {}
        for oname in ood_base_names:
            mask = ds_bases_ood_arr == oname
            X_ood_sub = Xod[mask]
            y_ood_sub = y_ood[mask]
            preds = best_clf.predict(X_ood_sub)
            ood_reports[oname] = classification_report(
                y_ood_sub, preds, target_names=le.classes_, output_dict=True)
            print(f"{name:20s}  ood[{oname}]_acc={ood_reports[oname]['accuracy']:.3f}")

        clf_results[name] = {
            "best_params": best_params,
            "val_report":  val_report,
            "test_report": test_report,
            "ood_reports": ood_reports,
        }
        if name == "RandomForest": best_rf       = best_clf
        if name == "XGBoost":      best_xgb      = best_clf
        if name == "LR_L2":        best_lr_l2    = best_clf
        if name == "LR_Lasso":     best_lr_lasso = best_clf

    importances = sorted(zip(feat_names, best_rf.feature_importances_), key=lambda x: -x[1])
    print("\nTop 10 features (Random Forest):")
    for fname, imp in importances[:10]:
        print(f"  {fname:<30} {imp:.4f}")
    clf_results["RandomForest"]["feature_importances"] = {
        fname: float(imp) for fname, imp in importances
    }

    if best_lr_l2 is not None and hasattr(best_lr_l2, "coef_"):
        coef_abs = np.abs(best_lr_l2.coef_).mean(axis=0)  # mean abs coef across classes
        nonzero = [(feat_names[i], float(coef_abs[i])) for i in range(len(feat_names))
                   if coef_abs[i] > 0]
        nonzero.sort(key=lambda x: -x[1])
        print(f"\nLR_L2 top features ({len(nonzero)}/{len(feat_names)} non-zero):")
        for fname, c in nonzero[:10]:
            print(f"  {fname:<30} {c:.4f}")

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

    final_lstm_model = None   # set in both has_val and else branches below
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
            _lstm_auroc_str = ""
            if _os_ready and final_lstm_model is not None:
                _, _, _lstm_confs_te = _eval_lstm(
                    final_lstm_model, seq_test, _X_te_scl, _y_te_enc, return_proba=True)
                _, _, _lstm_confs_os = _eval_lstm(
                    final_lstm_model, seq_os, _X_os_scl,
                    np.zeros(len(seq_os), dtype=int), return_proba=True)
                _lstm_scores = np.concatenate([_lstm_confs_te, _lstm_confs_os])
                _lstm_auroc_str = f"  auroc={roc_auc_score(_in_labels, _lstm_scores):.3f}"
            print(f"{'LSTM':20s}  test_acc={lstm_result['test_report']['accuracy']:.3f}{_lstm_auroc_str}")
        # OOD / open-set eval needs a final model trained on full train set (not val-split model)
        lstm_ood_reports = {}
        if has_ood or (open_set_agents and feat_os):
            final_model = _fit_lstm(seq_train, Xs_train, list(y_train), n_classes,
                                    lstm_result["best_params"])
            final_lstm_model = final_model
            for oname in ood_base_names:
                mask = ds_bases_ood_arr == oname
                seq_ood_sub = [s for s, m in zip(seq_ood, mask) if m]
                Xs_ood_sub  = Xs_ood[mask]
                y_ood_sub   = [l for l, m in zip(list(y_ood), mask) if m]
                _, ood_preds = _eval_lstm(final_model, seq_ood_sub, Xs_ood_sub, y_ood_sub)
                lstm_ood_reports[oname] = remap(
                    classification_report(y_ood_sub, ood_preds, output_dict=True))
                print(f"{'LSTM':20s}  ood[{oname}]_acc={lstm_ood_reports[oname]['accuracy']:.3f}")
        lstm_result["ood_reports"] = lstm_ood_reports
    else:
        default_params   = {"hidden_dim": 64, "dropout": 0.3}
        model            = _fit_lstm(seq_train, Xs_train, list(y_train), n_classes, default_params)
        final_lstm_model = model
        test_report    = None
        if has_test:
            _, test_preds = _eval_lstm(model, seq_test, Xs_test, list(y_test))
            test_report   = remap(classification_report(list(y_test), test_preds, output_dict=True))
        lstm_ood_reports = {}
        if has_ood:
            for oname in ood_base_names:
                mask = ds_bases_ood_arr == oname
                seq_ood_sub = [s for s, m in zip(seq_ood, mask) if m]
                Xs_ood_sub  = Xs_ood[mask]
                y_ood_sub   = [l for l, m in zip(list(y_ood), mask) if m]
                _, ood_preds = _eval_lstm(model, seq_ood_sub, Xs_ood_sub, y_ood_sub)
                lstm_ood_reports[oname] = remap(
                    classification_report(y_ood_sub, ood_preds, output_dict=True))
                print(f"{'LSTM':20s}  ood[{oname}]_acc={lstm_ood_reports[oname]['accuracy']:.3f}")
        torch.save(model.state_dict(), models_dir / "lstm_model.pt")
        lstm_result = {"best_params": default_params, "val_report": None,
                       "test_report": test_report, "ood_reports": lstm_ood_reports}

    clf_results["LSTM"] = lstm_result

    # --- prefix curve (early identification analysis) ---
    prefix_curve = None
    if prefix_eval and raw_events_test:
        print("\nRunning early-identification prefix curve analysis ...")
        prefix_curve = eval_prefix_curve(
            trained_clfs={
                "RandomForest": best_rf,
                "XGBoost":      best_xgb,
                "LR_L2":        best_lr_l2,
                "LR_Lasso":     best_lr_lasso,
            },
            lstm_model=final_lstm_model,
            raw_events_test=list(raw_events_test),
            raw_events_ood=list(raw_events_ood) if raw_events_ood else [],
            lbl_test=lbl_test,
            lbl_ood=lbl_ood,
            ds_bases_ood=list(ds_bases_ood),
            le=le,
            feat_names=feat_names,
            scaler=scaler,
        )

    # --- open-set evaluation ---
    open_set_result = None
    if open_set_agents and feat_os:
        print("\nRunning open-set recognition evaluation ...")
        # Raw (unscaled) features for RF/XGBoost; scaled for LR/LSTM
        X_os_raw  = to_X(feat_os)
        X_os_scl  = scaler.transform(X_os_raw)
        X_te_raw  = to_X(feat_test)  if feat_test else np.empty((0, len(feat_names)))
        X_te_scl  = scaler.transform(X_te_raw) if feat_test else X_te_raw
        y_te_os   = le.transform(lbl_test) if lbl_test else np.array([], dtype=int)
        open_set_result = eval_open_set(
            trained_clfs={
                "RandomForest": best_rf,
                "XGBoost":      best_xgb,
                "LR_L2":        best_lr_l2,
                "LR_Lasso":     best_lr_lasso,
            },
            lstm_model=final_lstm_model,
            X_known=X_te_raw,
            Xs_known=X_te_scl,
            X_unknown=X_os_raw,
            Xs_unknown=X_os_scl,
            seqs_known=seq_test,
            seqs_unknown=seq_os,
            y_known=y_te_os,
            le=le,
            open_set_agent_names=lbl_os,
        )
        # Print open-set summary — this is the primary result in this mode
        unknown_str = ", ".join(sorted(set(lbl_os)))
        print(f"\n{'─' * 62}")
        print(f"  Open-set recognition  [unknown: {unknown_str}]")
        print(f"  known n={len(y_te_os)}  unknown n={len(lbl_os)}")
        print(f"{'─' * 62}")
        print(f"  {'Classifier':20s}  {'test-acc':>8}  {'AUROC':>6}  {'FPR95':>6}  "
              f"{'known-F1':>9}  {'unk-F1':>7}  {'macro-F1':>9}")
        print()
        for clf_name, res in open_set_result.items():
            br = res.get("binary_report", {})
            # Retrieve test accuracy from the closed-set report (LSTM lives in lstm_result)
            if clf_name == "LSTM":
                _tr = (lstm_result.get("test_report") or {})
            else:
                _tr = (clf_results.get(clf_name, {}).get("test_report") or {})
            test_acc_str = f"{_tr.get('accuracy', float('nan')):8.3f}" if _tr else "     n/a"
            print(f"  {clf_name:20s}  {test_acc_str}  {res['auroc']:6.3f}  {res['fpr95']:6.3f}  "
                  f"  {br.get('known_f1', 0):7.3f}  {br.get('unknown_f1', 0):7.3f}"
                  f"  {br.get('macro_f1', 0):9.3f}")
        print()

    # --- save artefacts ---
    with open(models_dir / "classifier.pkl", "wb") as f:
        pickle.dump({
            "rf":         best_rf,
            "xgb":        best_xgb,
            "lr_l2":      best_lr_l2,
            "le":         le,
            "feat_names": feat_names,
            "scaler":     scaler,
        }, f)

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
        # mean trace length (n DOM events) per split — used by plot_early_id.py
        "mean_n_events":  {
            "test": float(np.mean([f["n_events_total"] for f in feat_test])) if feat_test else None,
            "ood":  float(np.mean([f["n_events_total"] for f in feat_ood]))  if feat_ood  else None,
        },
        "models":         clf_results,
        "prefix_curve":   prefix_curve,
        "open_set":       open_set_result,
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
        report = res.get("test_report")
        if report:
            _print_report(model_name, "test", report)
        for oname, report in (res.get("ood_reports") or {}).items():
            _print_report(model_name, f"ood[{oname}]", report)


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
            "  python trace_analyzer.py --train-datasets 2wikimultihop --agents gpt_5_4 qwen3vl_8b --tag wiki_no_uitars\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--traces-dir", type=Path, default=Path("./traces"),
                        help="Root traces directory (default: ./traces)")
    parser.add_argument("--hf-repo", default=None,
                        help="HuggingFace dataset repo to load traces from instead of --traces-dir, "
                             "e.g. your-org/known-actions-traces. Requires --train-datasets.")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token for private repos (default: reads HF_TOKEN env var)")
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
    parser.add_argument("--agents", nargs="+", default=None,
                        metavar="AGENT_ID",
                        help="Agent IDs to include (e.g. --agents gpt_5_4 qwen3vl_8b). "
                             "Default: all agents found in traces-dir.")
    parser.add_argument("--families", nargs="+", default=None,
                        metavar="FAMILY",
                        help="Select agents by model family (e.g. --families qwen3vl glm gpt). "
                             "Expands to all agent_ids with that family in config.yaml. "
                             "Combined with --agents via union if both are specified. "
                             "Use with --label-by family for family-level classification.")
    parser.add_argument("--resplit-datasets", nargs="+", default=None,
                        metavar="NAME",
                        help="Subset of --train-datasets that lack explicit train/val/test "
                             "directory splits (e.g. --resplit-datasets frames). All matching "
                             "traces are pooled and stratified-split 50/25/25 by agent.")
    parser.add_argument("--resplit-n-per-agent", type=int, default=None,
                        metavar="N",
                        help="Cap each agent's pool to N episodes before splitting "
                             "(e.g. --resplit-n-per-agent 300 → 150/75/75 per agent, "
                             "matching 2wikimultihop's budget for a fair comparison).")
    parser.add_argument("--prefix-eval", action="store_true", default=False,
                        help="Run early-identification prefix curve analysis at eval time. "
                             "Evaluates each classifier on truncated prefixes (first N events "
                             "and first T ms). Adds 'prefix_curve' key to results.json.")
    parser.add_argument("--open-set-agents", nargs="+", default=None,
                        metavar="AGENT_ID",
                        help="Agents to treat as unknown at eval time (excluded from training). "
                             "Enables open-set AUROC/FPR95 evaluation. Adds 'open_set' key to "
                             "results.json. Example: --open-set-agents gpt_5_4")
    parser.add_argument("--label-by", choices=["agent", "family"], default="agent",
                        help="Classification target: 'agent' (default) uses individual checkpoint "
                             "IDs as class labels; 'family' remaps each agent to its model family "
                             "(defined in config.yaml) — e.g. qwen3vl_8b and qwen3vl_30b_a3b both "
                             "become 'qwen3vl'. Useful for family-level fingerprinting experiments.")
    cli = parser.parse_args()

    # Resolve --families into agent IDs and merge with --agents
    agents = cli.agents
    if cli.families:
        family_map = _load_family_map()
        requested  = set(cli.families)
        unknown    = requested - set(family_map.values())
        if unknown:
            print(f"[WARN] Unknown families (not in config.yaml): {sorted(unknown)}")
        family_agents = [aid for aid, fam in family_map.items() if fam in requested]
        agents = sorted(set(family_agents) | set(agents or []))

    import os
    hf_token = cli.hf_token or os.environ.get("HF_TOKEN")

    train(cli.traces_dir, tag=cli.tag, train_datasets=cli.train_datasets,
          ood_datasets=cli.ood_datasets, agents=agents,
          open_set_agents=cli.open_set_agents,
          resplit_datasets=cli.resplit_datasets,
          resplit_n_per_agent=cli.resplit_n_per_agent,
          prefix_eval=cli.prefix_eval,
          label_by=cli.label_by,
          hf_repo=cli.hf_repo,
          hf_token=hf_token)
