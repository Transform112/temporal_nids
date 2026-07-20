"""
L03 — Chronological Train/Val/Test Split
===========================================
RUNS ON LAPTOP (CPU only). Reads cleaned parquet files.
Computes time-respecting 70/15/15 split and persists indices.

CRITICAL: This produces the SINGLE SOURCE OF TRUTH split indices.
Every downstream Kaggle script loads these — never recomputed.

Memory: Only reads FLOW_START_MILLISECONDS + labels for sorting.
Output: dataset/splits/{dataset}_{split}_index.parquet + split_metadata.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import yaml
from datetime import datetime, timezone
import sys

# ---- CONFIG ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = PROJECT_ROOT / 'dataset' / 'cleaned'
SPLITS_DIR = PROJECT_ROOT / 'dataset' / 'splits'
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test
SEED = 42

DATASETS = ['NF-CICIDS2018', 'NF-UNSW-NB15']
TIME_COL = 'FLOW_START_MILLISECONDS'


def compute_split(dataset_name):
    """Compute chronological split for one dataset."""
    path = CLEANED_DIR / f'{dataset_name}_cleaned'
    if not path.exists():
        print(f"  SKIPPED: {path} not found. Run L01 first.")
        return None

    print(f"\n{'='*60}")
    print(f"  {dataset_name}")
    print(f"{'='*60}")

    # Read ONLY the time column (memory efficient)
    print(f"  Reading timestamps...")
    df_times = pd.read_parquet(path, columns=[TIME_COL, 'unified_label'])

    # Sort by time
    print(f"  Sorting {len(df_times):,} flows by {TIME_COL}...")
    df_times = df_times.sort_values(TIME_COL).reset_index(drop=True)

    n = len(df_times)
    train_end = int(n * SPLIT_RATIOS[0])
    val_end = int(n * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))

    # Build split masks
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    # Time boundaries
    t_train_min = df_times.loc[train_mask, TIME_COL].min()
    t_train_max = df_times.loc[train_mask, TIME_COL].max()
    t_val_min = df_times.loc[val_mask, TIME_COL].min()
    t_val_max = df_times.loc[val_mask, TIME_COL].max()
    t_test_min = df_times.loc[test_mask, TIME_COL].min()
    t_test_max = df_times.loc[test_mask, TIME_COL].max()

    # Verify chronological ordering
    assert t_train_max <= t_val_min, \
        f"TIME LEAKAGE: train max ({t_train_max}) > val min ({t_val_min})"
    assert t_val_max <= t_test_min, \
        f"TIME LEAKAGE: val max ({t_val_max}) > test min ({t_test_min})"

    # Class distribution per split
    for split_name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
        n_rows = mask.sum()
        class_dist = df_times.loc[mask, 'unified_label'].value_counts().to_dict()
        print(f"  {split_name:6s}: {n_rows:>10,} flows "
              f"({n_rows/n*100:5.1f}%) — "
              f"time: [{df_times.loc[mask, TIME_COL].min():.0f}, {df_times.loc[mask, TIME_COL].max():.0f}] — "
              f"classes: {len(class_dist)}")

    # Save split indices
    for split_name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
        split_df = pd.DataFrame({
            'row_index': np.where(mask)[0],
            'is_train': split_name == 'train',
            'is_val': split_name == 'val',
            'is_test': split_name == 'test',
        })
        out_path = SPLITS_DIR / f'{dataset_name}_{split_name}_index.parquet'
        split_df.to_parquet(out_path, index=False)
        print(f"  Saved: {out_path}")

    return {
        'n_train': int(train_mask.sum()),
        'n_val': int(val_mask.sum()),
        'n_test': int(test_mask.sum()),
        't_train_range': [float(t_train_min), float(t_train_max)],
        't_val_range': [float(t_val_min), float(t_val_max)],
        't_test_range': [float(t_test_min), float(t_test_max)],
    }


# ---- MAIN ----
if __name__ == '__main__':
    print("L03 — Chronological Train/Val/Test Split")
    print(f"Split ratios: {SPLIT_RATIOS}")
    print(f"Time column: {TIME_COL}")

    metadata = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'seed': SEED,
        'split_method': 'chronological',
        'split_column': TIME_COL,
        'split_ratios': list(SPLIT_RATIOS),
        'datasets': {},
    }

    all_ok = True
    for ds_name in DATASETS:
        result = compute_split(ds_name)
        if result:
            metadata['datasets'][ds_name] = result
        else:
            all_ok = False

    # Save metadata
    with open(SPLITS_DIR / 'split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n{'All splits computed. Metadata saved.' if all_ok else 'Some datasets failed -- check errors.'}")
    if all_ok:
        print("Next: Run L04_graph_construction.py")
