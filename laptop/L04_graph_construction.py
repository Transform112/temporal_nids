"""
L04 — Windowed Graph Construction & Feature Scaling
======================================================
RUNS ON LAPTOP (CPU only, memory-efficient).
Builds 120s-windowed PyG Data objects from cleaned data.
Fits StandardScaler on E_train, applies frozen to val/test.
Saves graphs as .pt files for direct Kaggle loading.

Memory optimizations:
  - Processes one dataset + one split at a time
  - Builds one 120s window at a time, saves immediately
  - Uses float32 for all numeric data
  - Uses hashlib for node ID hashing (no global node map explosion)
  - Deletes intermediate DataFrames after each window

Output: dataset/graphs/{dataset}_{split}_list.pt + scaler.pkl
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import hashlib
import pickle
import json
from pathlib import Path
import sys
import gc
import warnings
warnings.filterwarnings('ignore')

# ---- CONFIG ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = PROJECT_ROOT / 'dataset' / 'cleaned'
SPLITS_DIR = PROJECT_ROOT / 'dataset' / 'splits'
GRAPHS_DIR = PROJECT_ROOT / 'dataset' / 'graphs'
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE_SEC = 120  # seconds
SEED = 42
np.random.seed(SEED)

# Load feature manifest
import yaml
with open(PROJECT_ROOT / 'feature_manifest.yaml', 'r') as f:
    fm = yaml.safe_load(f)
with open(PROJECT_ROOT / 'label_map.yaml', 'r') as f:
    label_map = yaml.safe_load(f)

KEPT_FEATURES = fm['kept_features']
UNIFIED_CLASSES = label_map['unified_classes']
LABEL_TO_IDX = {name: i for i, name in enumerate(UNIFIED_CLASSES)}

DATASETS = ['NF-CICIDS2018', 'NF-UNSW-NB15']
SPLITS = ['train', 'val', 'test']


def hash_endpoint(ip, port):
    """Stable integer node ID from IP:port."""
    s = f"{ip}:{port}"
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % (2**31)


def build_windowed_graphs(df_split, split_name, dataset_name):
    """
    Build list of PyG Data objects for one split, partitioned into 120s windows.

    Args:
        df_split: DataFrame for one split (already filtered)
        split_name: 'train', 'val', or 'test'
        dataset_name: dataset identifier string

    Returns:
        list of Data objects
    """
    # Normalize time to seconds from min
    time_ms = df_split['FLOW_START_MILLISECONDS'].values.astype(np.float64)
    t_min = time_ms.min()
    time_sec = (time_ms - t_min) / 1000.0
    window_indices = (time_sec // WINDOW_SIZE_SEC).astype(np.int32)

    # Prepare feature matrix (float32 for memory)
    feature_cols = [c for c in KEPT_FEATURES if c in df_split.columns]
    X = df_split[feature_cols].values.astype(np.float32)

    # Label mapping
    labels = df_split['unified_label'].map(LABEL_TO_IDX).values.astype(np.int64)

    # Node identity columns
    src_ips = df_split['IPV4_SRC_ADDR'].values
    src_ports = df_split['L4_SRC_PORT'].values
    dst_ips = df_split['IPV4_DST_ADDR'].values
    dst_ports = df_split['L4_DST_PORT'].values

    # Binary labels
    is_attack = (df_split['unified_label'] != 'Benign').values.astype(np.int64)

    n_windows = window_indices.max() + 1
    graphs = []
    total_edges = 0

    print(f"    Building {n_windows} windows...")

    for w in range(n_windows):
        w_mask = window_indices == w
        n_edges = w_mask.sum()
        if n_edges == 0:
            continue

        w_indices = np.where(w_mask)[0]

        # Build local node mapping for THIS window only (memory efficient)
        node_ids = {}
        src_nodes = np.empty(n_edges, dtype=np.int64)
        dst_nodes = np.empty(n_edges, dtype=np.int64)

        for i, idx in enumerate(w_indices):
            src_key = f"{src_ips[idx]}:{src_ports[idx]}"
            dst_key = f"{dst_ips[idx]}:{dst_ports[idx]}"

            if src_key not in node_ids:
                node_ids[src_key] = len(node_ids)
            if dst_key not in node_ids:
                node_ids[dst_key] = len(node_ids)

            src_nodes[i] = node_ids[src_key]
            dst_nodes[i] = node_ids[dst_key]

        # Build PyG Data object
        data = Data(
            edge_index=torch.tensor(np.stack([src_nodes, dst_nodes]), dtype=torch.long),
            edge_attr=torch.tensor(X[w_indices], dtype=torch.float32),
            y=torch.tensor(labels[w_indices], dtype=torch.long),
            y_binary=torch.tensor(is_attack[w_indices], dtype=torch.long),
            edge_time=torch.tensor(time_ms[w_indices], dtype=torch.float64),
            num_nodes=len(node_ids),
            window_idx=w,
        )

        graphs.append(data)
        total_edges += n_edges

        # GC every 20 windows
        if w % 20 == 0:
            gc.collect()

    print(f"    Built {len(graphs)} non-empty windows ({total_edges:,} edges)")

    return graphs


def fit_global_scaler():
    print(f"\n{'='*60}\nFitting Global StandardScaler on E_train\n{'='*60}")
    scaler = StandardScaler()
    for name in DATASETS:
        split_path = SPLITS_DIR / f'{name}_train_index.parquet'
        cleaned_path = CLEANED_DIR / f'{name}_cleaned'
        if not split_path.exists() or not cleaned_path.exists():
            print(f"  SKIPPED: Missing files for {name}")
            return None
        df = pd.read_parquet(cleaned_path)
        split_df = pd.read_parquet(split_path)
        train_features = df.iloc[split_df['row_index'].values][KEPT_FEATURES].values.astype(np.float32)
        scaler.partial_fit(train_features)
        print(f"  Partial fit on {name}: {train_features.shape[0]:,} samples")
        del df, split_df, train_features; gc.collect()
    
    scaler_path = GRAPHS_DIR / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Global scaler saved: {scaler_path}")
    return scaler

def process_dataset(name, scaler):
    """Process one dataset: load splits, build graphs, apply global scaler, save."""
    cleaned_path = CLEANED_DIR / f'{name}_cleaned'
    if not cleaned_path.exists():
        print(f"  SKIPPED: {cleaned_path} not found. Run L01 first.")
        return False

    print(f"\n{'='*60}")
    print(f"Graph construction: {name}")
    print(f"{'='*60}")

    # Load dataset in chunks to reduce memory
    print(f"  Loading cleaned data...")
    df = pd.read_parquet(cleaned_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Load split indices
    split_data = {}
    for split_name in SPLITS:
        split_path = SPLITS_DIR / f'{name}_{split_name}_index.parquet'
        if not split_path.exists():
            print(f"  SKIPPED: split file {split_path} not found. Run L03 first.")
            return False
        split_df = pd.read_parquet(split_path)
        split_indices = split_df['row_index'].values
        split_data[split_name] = df.iloc[split_indices].copy()
        print(f"  {split_name}: {len(split_data[split_name]):,} flows")

    # Free full df
    del df; gc.collect()

    # ---- Build and save graphs per split ----
    for split_name in SPLITS:
        print(f"\n  Building {split_name} graphs...")

        # Apply global scaler and clamp
        df_split = split_data[split_name]
        feature_cols = [c for c in KEPT_FEATURES if c in df_split.columns]
        scaled = scaler.transform(df_split[feature_cols].values.astype(np.float32))
        scaled = np.clip(scaled, -10.0, 10.0)  # Stop outlier explosion
        df_split[feature_cols] = scaled

        # Build windowed graphs
        graphs = build_windowed_graphs(df_split, split_name, name)

        # Save
        out_path = GRAPHS_DIR / f'{name}_{split_name}_list.pt'
        torch.save(graphs, out_path)
        print(f"  Saved: {out_path} ({len(graphs)} windows)")

        # Free memory
        del df_split, graphs; gc.collect()

    # Free split data
    del split_data; gc.collect()

    return True


# ---- MAIN ----
if __name__ == '__main__':
    print("L04 — Windowed Graph Construction & Feature Scaling")
    print(f"Window size: {WINDOW_SIZE_SEC}s")
    print(f"Datasets: {DATASETS}")

    global_scaler = fit_global_scaler()
    if global_scaler is None:
        print("Failed to fit global scaler. Exiting.")
        sys.exit(1)

    results = {}
    for ds_name in DATASETS:
        ok = process_dataset(ds_name, global_scaler)
        results[ds_name] = ok

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAILED'}] {name}")

    all_ok = all(results.values())

    # List output sizes
    if all_ok:
        print(f"\nOutput files in {GRAPHS_DIR}:")
        total_size = 0
        for f in sorted(GRAPHS_DIR.glob('*')):
            size_mb = f.stat().st_size / (1024*1024)
            total_size += size_mb
            print(f"  {f.name}: {size_mb:.1f} MB")
        print(f"  Total: {total_size:.1f} MB")

    print(f"\n{'All graphs built. Ready for Kaggle upload.' if all_ok else 'Some datasets failed -- check errors.'}")
    if all_ok:
        print("\nLAPTOP PHASE COMPLETE.")
        print("Upload to Kaggle: dataset/graphs/*.pt + dataset/graphs/scaler.pkl")
        print("Also upload: label_map.yaml, feature_manifest.yaml")
        print("\nThen run Kaggle scripts: K01 -> K02 -> K03 -> K04 -> K05 -> K06")
