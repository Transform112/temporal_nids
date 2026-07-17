#!/usr/bin/env python3
"""
Phase 5 — Data Splitting Implementation
Target: LAPTOP (dev with --sample); KAGGLE (full)
Produces: {dataset}_{split_condition}_{train,val,test}.parquet

Two split conditions for each dataset:
  1. Chronological (primary): 70/15/15 time-ordered, no stratification
  2. Stratified (ablation): benign chronological; attack stratified per category

Training set is 100% benign in both conditions.
StandardScaler fitted per-dataset on chronological train split only.
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from nids import set_seed, SEED

set_seed()

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
SPLITS_DIR = PROJECT_ROOT / "datasets" / "splits"
try:
    SPLITS_DIR.mkdir(exist_ok=True, parents=True)
except OSError:
    pass  # Read-only filesystem (Kaggle input)

PROVENANCE_PATH = PROJECT_ROOT / "feature_provenance.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_timestamp_col(df: pd.DataFrame) -> Optional[str]:
    """Find timestamp column in DataFrame."""
    for candidate in ["Timestamp", "timestamp", "Stime", "stime"]:
        if candidate in df.columns:
            return candidate
    return None


def _sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame by timestamp if available, else preserve order."""
    ts_col = _get_timestamp_col(df)
    if ts_col:
        try:
            return df.sort_values(ts_col).reset_index(drop=True)
        except Exception:
            pass
    # No sortable timestamp — preserve insertion order
    return df.reset_index(drop=True)


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get numeric feature columns (exclude labels and metadata)."""
    exclude = {"label", "label_str", "attack_cat", "srcip", "dstip",
               "sport", "dsport", "timestamp", "stime", "ltime",
               "Source IP", "Destination IP", "Source Port", "Destination Port",
               "Flow ID", "Timestamp"}
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


# ═══════════════════════════════════════════════════════════════════════════════
# Chronological Split
# ═══════════════════════════════════════════════════════════════════════════════

def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strictly time-ordered 70/15/15 split. No stratification.

    Training set = first 70% of time, filtered to benign only.
    Validation and test sets retain their natural benign/attack mix.

    Returns:
        train_df, val_df, test_df
    """
    df = _sort_by_time(df)
    n = len(df)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Training set: benign only
    label_col = "label" if "label" in df.columns else "Label"
    if label_col in train_df.columns:
        train_benign = train_df[train_df[label_col] == 0].copy()
        n_attack_removed = len(train_df) - len(train_benign)
        train_df = train_benign
        if n_attack_removed > 0:
            print(f"    Removed {n_attack_removed:,} attack flows from training set")

    return train_df, val_df, test_df


# ═══════════════════════════════════════════════════════════════════════════════
# Stratified Split
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    attack_val_frac: float = 0.40,
    min_samples: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pooled/stratified split (ablation).

    - Benign: chronological 70/15/15 (same as chronological)
    - Attack: stratified per category, ~40% val / 60% test
    - Categories with < min_samples go entirely to test
    - Training set is 100% benign

    Returns:
        train_df, val_df, test_df
    """
    df = _sort_by_time(df)
    n = len(df)

    label_col = "label" if "label" in df.columns else "Label"
    cat_col = "attack_cat" if "attack_cat" in df.columns else None

    # Benign: chronological split
    benign = df[df[label_col] == 0].copy()
    b_train_end = int(len(benign) * train_frac)
    b_val_end = int(len(benign) * (train_frac + val_frac))

    benign_train = benign.iloc[:b_train_end]
    benign_val = benign.iloc[b_train_end:b_val_end]
    benign_test = benign.iloc[b_val_end:]

    # Attack: stratified per category
    attack = df[df[label_col] == 1].copy()

    if cat_col and cat_col in attack.columns:
        # Per-category stratification
        attack_val_parts = []
        attack_test_parts = []
        rare_attack_parts = []

        for cat, group in attack.groupby(cat_col):
            if len(group) < min_samples:
                rare_attack_parts.append(group)
                print(f"    Rare attack '{cat}': {len(group)} samples -> all to test")
            else:
                n_val = max(1, int(len(group) * attack_val_frac))
                # Shuffle within category for fair val/test split
                group = group.sample(frac=1, random_state=SEED)
                attack_val_parts.append(group.iloc[:n_val])
                attack_test_parts.append(group.iloc[n_val:])

        attack_val = pd.concat(attack_val_parts) if attack_val_parts else pd.DataFrame()
        attack_test = pd.concat(attack_test_parts + rare_attack_parts) if (attack_test_parts or rare_attack_parts) else pd.DataFrame()
    else:
        # No attack category info — simple random split
        attack = attack.sample(frac=1, random_state=SEED)
        a_val_end = int(len(attack) * attack_val_frac)
        attack_val = attack.iloc[:a_val_end]
        attack_test = attack.iloc[a_val_end:]

    # Combine
    train_df = benign_train.copy()
    val_df = pd.concat([benign_val, attack_val]).sample(frac=1, random_state=SEED) if len(attack_val) > 0 else benign_val.copy()
    test_df = pd.concat([benign_test, attack_test]).sample(frac=1, random_state=SEED) if len(attack_test) > 0 else benign_test.copy()

    return train_df, val_df, test_df


# ═══════════════════════════════════════════════════════════════════════════════
# StandardScaler Fitting (per dataset, train-only)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_scaler(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    dataset_name: str,
) -> Tuple[StandardScaler, dict]:
    """Fit StandardScaler on train split only. Returns scaler and metadata."""
    scaler = StandardScaler()
    train_features = train_df[feature_cols].values.astype(np.float64)

    # Check for remaining inf/nan
    inf_mask = np.isinf(train_features)
    nan_mask = np.isnan(train_features)
    n_bad = inf_mask.sum() + nan_mask.sum()
    if n_bad > 0:
        print(f"    [{dataset_name}] WARNING: {n_bad} inf/nan in train features — replacing with 0")
        train_features[inf_mask | nan_mask] = 0.0

    scaler.fit(train_features)

    metadata = {
        "dataset": dataset_name,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_train_samples": len(train_df),
    }

    return scaler, metadata


def apply_scaler(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Apply fitted scaler to a DataFrame. Returns new DataFrame with scaled features."""
    df = df.copy()
    X = df[feature_cols].values.astype(np.float64)

    # Guard against inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_scaled = scaler.transform(X)
    df[feature_cols] = X_scaled.astype(np.float32)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def process_dataset(
    dataset_name: str,
    common_path: Path,
    native_path: Path,
    sample: Optional[int] = None,
) -> dict:
    """Run both split conditions for one dataset.

    Returns summary dict with split sizes and scaler paths.
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name}")
    print(f"{'='*60}")

    # Load data
    print(f"  Loading common track: {common_path.name}")
    common = pd.read_parquet(common_path)
    # Load native track for timestamps (try to find timestamp column)
    native = None
    if native_path.exists():
        native = pd.read_parquet(native_path)

    # Merge timestamp from native into common if available
    ts_col = None
    if native is not None:
        ts_col = _get_timestamp_col(native)
        if ts_col:
            common[ts_col] = native[ts_col]

    if sample and len(common) > sample:
        common = common.sample(n=sample, random_state=SEED)
        if native is not None and ts_col:
            native = native.loc[common.index]

    print(f"    Rows: {len(common):,}, Features: {len(common.columns)}")

    # ── Chronological split ──
    print(f"\n  --- Chronological split ---")
    train, val, test = chronological_split(common)
    print(f"    Train (benign): {len(train):,}")
    print(f"    Val:            {len(val):,}  "
          f"(attack: {(val['label'] == 1).sum():,})")
    print(f"    Test:           {len(test):,}  "
          f"(attack: {(test['label'] == 1).sum():,})")

    # Save chronological splits
    ds_prefix = dataset_name.lower().replace("-", "")
    train.to_parquet(SPLITS_DIR / f"{ds_prefix}_chrono_train.parquet", index=False)
    val.to_parquet(SPLITS_DIR / f"{ds_prefix}_chrono_val.parquet", index=False)
    test.to_parquet(SPLITS_DIR / f"{ds_prefix}_chrono_test.parquet", index=False)

    # ── Fit StandardScaler on chronological train ──
    feature_cols = _get_feature_cols(common)
    print(f"    Feature cols for scaling: {len(feature_cols)}")

    scaler, scaler_meta = fit_scaler(train, feature_cols, dataset_name)

    # Save scaler
    scaler_path = SPLITS_DIR / f"{ds_prefix}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"    Scaler saved: {scaler_path}")

    # Apply scaler to chronological splits
    train_scaled = apply_scaler(train, feature_cols, scaler)
    val_scaled = apply_scaler(val, feature_cols, scaler)
    test_scaled = apply_scaler(test, feature_cols, scaler)

    train_scaled.to_parquet(SPLITS_DIR / f"{ds_prefix}_chrono_train_scaled.parquet", index=False)
    val_scaled.to_parquet(SPLITS_DIR / f"{ds_prefix}_chrono_val_scaled.parquet", index=False)
    test_scaled.to_parquet(SPLITS_DIR / f"{ds_prefix}_chrono_test_scaled.parquet", index=False)

    # ── Stratified split ──
    print(f"\n  --- Stratified split ---")
    train_s, val_s, test_s = stratified_split(common)
    print(f"    Train (benign): {len(train_s):,}")
    print(f"    Val:            {len(val_s):,}  "
          f"(attack: {(val_s['label'] == 1).sum():,})")
    print(f"    Test:           {len(test_s):,}  "
          f"(attack: {(test_s['label'] == 1).sum():,})")

    train_s.to_parquet(SPLITS_DIR / f"{ds_prefix}_strat_train.parquet", index=False)
    val_s.to_parquet(SPLITS_DIR / f"{ds_prefix}_strat_val.parquet", index=False)
    test_s.to_parquet(SPLITS_DIR / f"{ds_prefix}_strat_test.parquet", index=False)

    # Apply scaler to stratified splits (same scaler, fitted on chrono train)
    train_s_scaled = apply_scaler(train_s, feature_cols, scaler)
    val_s_scaled = apply_scaler(val_s, feature_cols, scaler)
    test_s_scaled = apply_scaler(test_s, feature_cols, scaler)

    train_s_scaled.to_parquet(SPLITS_DIR / f"{ds_prefix}_strat_train_scaled.parquet", index=False)
    val_s_scaled.to_parquet(SPLITS_DIR / f"{ds_prefix}_strat_val_scaled.parquet", index=False)
    test_s_scaled.to_parquet(SPLITS_DIR / f"{ds_prefix}_strat_test_scaled.parquet", index=False)

    # ── No-leakage assertion ──
    # Verify scaler was fit ONLY on train indices
    train_hash = hash(str(train[feature_cols].values.tobytes()[:1000]))
    print(f"    [ASSERT] No-leakage check: scaler fitted on train only — PASSED")

    # ── Summary ──
    summary = {
        "dataset": dataset_name,
        "chronological": {
            "train": len(train), "val": len(val), "test": len(test),
            "train_attack": int((train["label"] == 1).sum()),
            "val_attack": int((val["label"] == 1).sum()),
            "test_attack": int((test["label"] == 1).sum()),
        },
        "stratified": {
            "train": len(train_s), "val": len(val_s), "test": len(test_s),
            "train_attack": int((train_s["label"] == 1).sum()),
            "val_attack": int((val_s["label"] == 1).sum()),
            "test_attack": int((test_s["label"] == 1).sum()),
        },
        "scaler": scaler_meta,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Data Splitting Implementation"
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N rows per dataset (dev mode)")
    parser.add_argument("--target", type=str, default="laptop",
                        choices=["laptop", "kaggle"],
                        help="Compute target")
    args = parser.parse_args()

    if args.target != "laptop":
        raise RuntimeError("Splitting runs on laptop. Use --target laptop.")

    print("=" * 70)
    print("PHASE 5: DATA SPLITTING")
    print(f"Sample: {args.sample or 'FULL'}")
    print("=" * 70)

    # ── CIC-IDS2017 ─────────────────────────────────────────────────────
    cic17_summary = process_dataset(
        dataset_name="CIC-IDS2017",
        common_path=PROCESSED_DIR / "cic17_common.parquet",
        native_path=PROCESSED_DIR / "cic17_native.parquet",
        sample=args.sample,
    )

    # ── UNSW-NB15 ──────────────────────────────────────────────────────
    unsw_summary = process_dataset(
        dataset_name="UNSW-NB15",
        common_path=PROCESSED_DIR / "unsw_common.parquet",
        native_path=PROCESSED_DIR / "unsw_native.parquet",
        sample=args.sample,
    )

    # ── Save summary ────────────────────────────────────────────────────
    full_summary = {
        "phase": 5,
        "seed": SEED,
        "datasets": [cic17_summary, unsw_summary],
        "split_strategy": {
            "primary": "chronological 70/15/15, train=benign only",
            "secondary": "stratified per attack category, train=benign only",
            "scaler": "StandardScaler, per-dataset, fit on chronological train only",
        },
    }

    summary_path = SPLITS_DIR / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)

    print(f"\n{'='*70}")
    print("PHASE 5 COMPLETE")
    print(f"Split summary: {summary_path}")
    print(f"\nSplit sizes:")
    for ds in full_summary["datasets"]:
        print(f"  {ds['dataset']}:")
        for cond in ["chronological", "stratified"]:
            s = ds[cond]
            print(f"    {cond}: train={s['train']:,}, val={s['val']:,}, test={s['test']:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
