#!/usr/bin/env python3
"""
Final Preprocessed Dataset Builder
===================================
Produces READY-TO-USE parquet files for Kaggle training.
One file per split, containing: IPs + timestamps + scaled features + label.
No merging, scaling, or splitting needed at runtime.

Output (6 files total):
  datasets/final/
    cic17_train.parquet    — benign only, scaled, with IPs
    cic17_val.parquet      — mixed, scaled, with IPs
    cic17_test.parquet     — mixed, scaled, with IPs
    unsw_train.parquet
    unsw_val.parquet
    unsw_test.parquet

Target: LAPTOP (runs once, upload output to Kaggle)
"""

import sys, os, json, time
from pathlib import Path
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from nids import set_seed, SEED
set_seed()

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "final"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ── Config ──
MAX_TRAIN_FLOWS = 500_000  # Max training flows (benign-only, chronological first N)
VAL_SIZE = 75_000
TEST_SIZE = 75_000


def load_native(path: Path) -> pd.DataFrame:
    """Load native track parquet."""
    return pd.read_parquet(path)


def load_common(path: Path) -> pd.DataFrame:
    """Load common track parquet."""
    return pd.read_parquet(path)


def get_ip_columns(native: pd.DataFrame) -> Tuple[str, str]:
    """Find source and destination IP columns in native track."""
    src_candidates = ["Source IP", "srcip"]
    dst_candidates = ["Destination IP", "dstip"]
    src = next((c for c in src_candidates if c in native.columns), None)
    dst = next((c for c in dst_candidates if c in native.columns), None)
    if src is None or dst is None:
        raise ValueError(f"Cannot find IP columns. Available: {list(native.columns)}")
    return src, dst


def get_time_column(native: pd.DataFrame) -> Optional[str]:
    """Find timestamp column."""
    for c in ["Timestamp", "Stime"]:
        if c in native.columns:
            return c
    return None


def get_feature_columns(common: pd.DataFrame) -> list[str]:
    """Get numeric feature columns (exclude label, metadata, non-numeric).

    Also excludes features marked 'dropped' in feature_provenance.json:
      - std_iat: jitter proxy, wrong signal
      - syn_count: constant 0 for UNSW = dataset-identity leakage
      - ack_count: constant 0 for UNSW = dataset-identity leakage
      - syn_count_imputed, ack_count_imputed: companion flags (no longer needed)
    """
    exclude = {
        "label", "label_str", "attack_cat",
        "srcip", "dstip", "sport", "dsport",
        "Source IP", "Destination IP", "Source Port", "Destination Port",
        "Flow ID", "Timestamp", "Stime", "Ltime",
        "timestamp", "stime", "ltime",
    }
    # Dropped features (per architectural review — see feature_provenance.json)
    dropped = {"std_iat", "syn_count", "ack_count",
               "syn_count_imputed", "ack_count_imputed"}
    exclude.update(dropped)

    feats = [c for c in common.columns
             if c not in exclude and pd.api.types.is_numeric_dtype(common[c])]
    return feats


def parse_timestamps(native: pd.DataFrame, time_col: Optional[str]) -> np.ndarray:
    """Parse timestamps into seconds-from-start float array."""
    if time_col and time_col in native.columns:
        try:
            ts = pd.to_datetime(native[time_col], errors="coerce")
            ts = (ts - ts.min()).dt.total_seconds().fillna(0).values
            return ts.astype(np.float64)
        except Exception:
            pass
    # Fallback: row index as pseudo-time (flows are captured in order)
    return np.arange(len(native), dtype=np.float64) * 0.001


def build_dataset(dataset_name: str, native_path: Path, common_path: Path,
                  max_train: int = MAX_TRAIN_FLOWS):
    """Build final preprocessed dataset for one source.

    Steps:
      1. Load native + common
      2. Extract IPs, timestamps, features, labels
      3. Chronological split (benign-only train, balanced val/test)
      4. Fit StandardScaler on train features
      5. Apply scaler to all splits
      6. Save as single parquet per split
    """
    print(f"\n{'='*60}")
    print(f"BUILDING: {dataset_name}")
    print(f"{'='*60}")

    # ── Load ──
    t0 = time.time()
    native = load_native(native_path)
    common = load_common(common_path)
    print(f"Native: {len(native):,} rows x {len(native.columns)} cols")
    print(f"Common: {len(common):,} rows x {len(common.columns)} cols")

    # Align lengths
    n = min(len(native), len(common))
    native = native.iloc[:n]
    common = common.iloc[:n]

    # ── Extract columns ──
    src_col, dst_col = get_ip_columns(native)
    time_col = get_time_column(native)
    feat_cols = get_feature_columns(common)
    label_col = "label" if "label" in common.columns else "Label"

    print(f"IP cols: {src_col}, {dst_col}")
    print(f"Time col: {time_col}")
    print(f"Feature cols: {len(feat_cols)}")
    print(f"Label col: {label_col}")

    src_ips = native[src_col].astype(str).values
    dst_ips = native[dst_col].astype(str).values
    timestamps = parse_timestamps(native, time_col)
    labels = common[label_col].values.astype(np.int32)

    # Pre-extract feature matrix
    X = common[feat_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Data extracted: {time.time()-t0:.1f}s")

    # ── Chronological split ──
    # Sort by timestamp if available
    sort_idx = np.argsort(timestamps) if time_col else np.arange(n)
    src_ips = src_ips[sort_idx]
    dst_ips = dst_ips[sort_idx]
    timestamps = timestamps[sort_idx]
    labels = labels[sort_idx]
    X = X[sort_idx]

    benign_mask = labels == 0
    attack_mask = labels == 1

    benign_idx = np.where(benign_mask)[0]
    attack_idx = np.where(attack_mask)[0]

    # Train: first N benign flows (chronological, benign only)
    n_train = min(max_train, len(benign_idx))
    train_idx = benign_idx[:n_train]

    # Val: next benign + some attack (balanced ~50/50)
    remaining_benign = benign_idx[n_train:]
    n_val = min(VAL_SIZE, len(remaining_benign) + len(attack_idx))
    n_val_attack = min(len(attack_idx) // 3, n_val // 2)
    n_val_benign = n_val - n_val_attack
    val_idx = np.concatenate([
        remaining_benign[:n_val_benign],
        attack_idx[:n_val_attack]
    ])
    np.random.RandomState(SEED).shuffle(val_idx)

    # Test: remaining benign + remaining attack (balanced)
    remaining_b = remaining_benign[n_val_benign:]
    remaining_a = attack_idx[n_val_attack:]
    n_test = min(TEST_SIZE, len(remaining_b) + len(remaining_a))
    n_test_attack = min(len(remaining_a), n_test // 2)
    n_test_benign = n_test - n_test_attack
    test_idx = np.concatenate([
        remaining_b[:n_test_benign],
        remaining_a[:n_test_attack]
    ])
    np.random.RandomState(SEED).shuffle(test_idx)

    print(f"Train: {len(train_idx):,} (benign only)")
    print(f"Val:   {len(val_idx):,} (attack: {labels[val_idx].sum():,})")
    print(f"Test:  {len(test_idx):,} (attack: {labels[test_idx].sum():,})")

    # ── Scale features ──
    scaler = StandardScaler()
    X_train = X[train_idx]
    scaler.fit(X_train)

    # Apply to all splits
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X[val_idx]).astype(np.float32)
    X_test_s = scaler.transform(X[test_idx]).astype(np.float32)

    # Save scaler
    prefix = dataset_name.lower().replace("-", "")
    joblib.dump(scaler, OUTPUT_DIR / f"{prefix}_scaler.joblib")
    scaler_meta = {
        "dataset": dataset_name,
        "n_features": len(feat_cols),
        "feature_names": feat_cols,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_train_samples": len(train_idx),
    }
    with open(OUTPUT_DIR / f"{prefix}_scaler_meta.json", "w") as f:
        json.dump(scaler_meta, f, indent=2)

    # ── Build output DataFrames ──
    def build_df(indices: np.ndarray, X_scaled: np.ndarray) -> pd.DataFrame:
        """Build a single DataFrame with IPs, timestamps, features, label."""
        data = {
            "src_ip": src_ips[indices],
            "dst_ip": dst_ips[indices],
            "timestamp": timestamps[indices],
            "label": labels[indices],
        }
        for i, col in enumerate(feat_cols):
            data[col] = X_scaled[:, i]
        return pd.DataFrame(data)

    df_train = build_df(train_idx, X_train_s)
    df_val = build_df(val_idx, X_val_s)
    df_test = build_df(test_idx, X_test_s)

    # ── Save ──
    train_path = OUTPUT_DIR / f"{prefix}_train.parquet"
    val_path = OUTPUT_DIR / f"{prefix}_val.parquet"
    test_path = OUTPUT_DIR / f"{prefix}_test.parquet"

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    mb = lambda p: p.stat().st_size / 1e6
    print(f"\nSaved:")
    print(f"  {train_path.name} — {len(df_train):,} rows, {len(df_train.columns)} cols, {mb(train_path):.1f} MB")
    print(f"  {val_path.name}   — {len(df_val):,} rows, {len(df_val.columns)} cols, {mb(val_path):.1f} MB")
    print(f"  {test_path.name}  — {len(df_test):,} rows, {len(df_test.columns)} cols, {mb(test_path):.1f} MB")
    print(f"  {prefix}_scaler.joblib — {len(feat_cols)} features")

    # Quick sanity
    for name, ddf in [("train", df_train), ("val", df_val), ("test", df_test)]:
        assert ddf["label"].isna().sum() == 0, f"{name}: NaN labels"
        feat_cols_present = [c for c in feat_cols if c in ddf.columns]
        feat_data = ddf[feat_cols_present].values
        assert not np.isinf(feat_data).any(), f"{name}: Inf in features"
        assert not np.isnan(feat_data).any(), f"{name}: NaN in features"
    print("Sanity checks: ALL PASSED")

    return {
        "dataset": dataset_name,
        "train": len(train_idx), "val": len(val_idx), "test": len(test_idx),
        "n_features": len(feat_cols),
        "feature_names": feat_cols,
    }


def main():
    parser = argparse.ArgumentParser(description="Build final preprocessed datasets")
    parser.add_argument("--max-train", type=int, default=MAX_TRAIN_FLOWS)
    parser.add_argument("--target", default="laptop", choices=["laptop", "kaggle"])
    args = parser.parse_args()

    if args.target != "laptop":
        raise RuntimeError("Dataset building runs on laptop only.")

    print("=" * 60)
    print("FINAL PREPROCESSED DATASET BUILDER")
    print(f"Max train flows: {args.max_train:,}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    summaries = []

    # ── CIC-IDS2017 ──
    s = build_dataset(
        "CIC-IDS2017",
        PROCESSED_DIR / "cic17_native.parquet",
        PROCESSED_DIR / "cic17_common.parquet",
        max_train=args.max_train,
    )
    summaries.append(s)

    # ── UNSW-NB15 ──
    s = build_dataset(
        "UNSW-NB15",
        PROCESSED_DIR / "unsw_native.parquet",
        PROCESSED_DIR / "unsw_common.parquet",
        max_train=args.max_train,
    )
    summaries.append(s)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("DONE — Upload datasets/final/ to Kaggle as 'nids-final'")
    print(f"{'='*60}")
    for s in summaries:
        print(f"  {s['dataset']}: train={s['train']:,} val={s['val']:,} "
              f"test={s['test']:,} features={s['n_features']}")


if __name__ == "__main__":
    main()
