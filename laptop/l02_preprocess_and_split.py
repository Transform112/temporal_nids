"""
L02 — Full Data Preprocessing Pipeline (CPU, Memory-Efficient)
===============================================================
Runs LOCALLY. Produces all artifacts that Kaggle GPU scripts need.
Chunked CSV reading, NaN/Inf cleaning, label mapping, chronological
split, windowed graph construction, scaler fitting.

Edge input: 41 raw features + 17 Time2Vec = 58-dim

Outputs (in laptop/processed/):
  - cleaned_{dataset}.parquet          (cleaned + labeled data)
  - split_indices/{dataset}_{split}_index.parquet
  - graphs/{dataset}_{split}_list.pt   (windowed PyG Data objects)
  - scaler.pkl
  - preprocessing_report.json
"""

import pandas as pd
import numpy as np
import yaml
import json
import pickle
import hashlib
import time
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'
PROCESSED_DIR = PROJECT_ROOT / 'laptop' / 'processed'
SPLIT_DIR = PROCESSED_DIR / 'split_indices'
GRAPH_DIR = PROCESSED_DIR / 'graphs'
SCALER_PATH = PROCESSED_DIR / 'scaler.pkl'

for d in [PROCESSED_DIR, SPLIT_DIR, GRAPH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500_000
WINDOW_SIZE_SEC = 120

# Load manifests
with open(PROJECT_ROOT / 'label_map.yaml', 'r') as f:
    label_map = yaml.safe_load(f)
with open(PROJECT_ROOT / 'feature_manifest.yaml', 'r') as f:
    fm = yaml.safe_load(f)

KEPT_FEATURES = fm['kept_features']           # 41 features
DROPPED_FIELDS = fm['dropped_fields']          # 9 fields
UNIFIED_CLASSES = label_map['unified_classes'] # 11 classes
EDGE_INPUT_DIM = fm['final_edge_input_dim']    # 58
TIME_FIELD = fm['time_signal_field']           # FLOW_START_MILLISECONDS

assert EDGE_INPUT_DIM == len(KEPT_FEATURES) + 17, \
    f"Dimension mismatch: {EDGE_INPUT_DIM} != {len(KEPT_FEATURES)} + 17"

print("=" * 60)
print("L02 — FULL DATA PREPROCESSING PIPELINE")
print("=" * 60)
print(f"Features kept: {len(KEPT_FEATURES)}")
print(f"Features dropped: {len(DROPPED_FIELDS)}")
print(f"Edge input dim: {EDGE_INPUT_DIM} (41 raw + 17 Time2Vec)")
print(f"Window size: {WINDOW_SIZE_SEC}s")
print(f"Chunk size: {CHUNK_SIZE:,}")

# ── Dataset config ──────────────────────────────────────────
TRAIN_DATASETS = {
    'NF-CICIDS2018': {
        'file': DATASET_DIR / 'NF-CICIDS2018-v3.csv',
        'label_key': 'NF-CSE-CIC-IDS2018',
    },
    'NF-UNSW-NB15': {
        'file': DATASET_DIR / 'NF-UNSW-NB15-v3.csv',
        'label_key': 'NF-UNSW-NB15',
    },
}
BLIND_DATASETS = {
    'NF-ToN-IoT': {
        'file': DATASET_DIR / 'NF-ToN-IoT-v3.csv',
        'label_key': 'NF-ToN-IoT',
    },
    'NF-BoT-IoT': {
        'file': DATASET_DIR / 'NF-BoT-IoT-v3.csv',
        'label_key': 'NF-BoT-IoT',
    },
}

# ── STEP 1: Clean, Label, and Save Each Dataset ─────────────
print("\n" + "=" * 60)
print("STEP 1: CLEAN + LABEL + SAVE")
print("=" * 60)

# NaN/Inf features to clean (discovered in sanity check)
INF_NAN_FEATURES = ['SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES']

def clean_chunk(chunk):
    """Clean NaN/Inf from a data chunk."""
    chunk.columns = chunk.columns.str.strip()

    # Fill protocol-conditional fields with 0
    for field in ['ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER']:
        if field in chunk.columns:
            chunk[field] = chunk[field].fillna(0)

    # Fix NaN in throughput fields (div by zero when no bytes sent)
    for feat in INF_NAN_FEATURES:
        if feat in chunk.columns:
            chunk[feat] = chunk[feat].fillna(0)
            chunk[feat] = chunk[feat].replace([np.inf, -np.inf], 0)

    # Generic NaN catch — fill remaining with 0
    for feat in KEPT_FEATURES:
        if feat in chunk.columns:
            chunk[feat] = chunk[feat].fillna(0)
            chunk[feat] = chunk[feat].replace([np.inf, -np.inf], 0)

    return chunk

def apply_taxonomy(chunk, label_key):
    """Map raw Attack labels to unified 11-class taxonomy."""
    mapping = label_map[label_key]
    chunk['unified_label'] = chunk['Attack'].map(mapping)
    unmapped = chunk['unified_label'].isna()
    if unmapped.any():
        n_unmapped = unmapped.sum()
        unmapped_labels = chunk.loc[unmapped, 'Attack'].unique()
        if n_unmapped > 0:
            print(f"    WARNING: {n_unmapped} rows with unmapped labels: {list(unmapped_labels)[:5]}")
        chunk = chunk[~unmapped]
    return chunk

cleaned_stats = {}
for ds_name, ds_cfg in {**TRAIN_DATASETS, **BLIND_DATASETS}.items():
    fpath = ds_cfg['file']
    if not fpath.exists():
        print(f"\n  SKIP {ds_name}: file not found")
        continue

    print(f"\n  Processing {ds_name}...")
    t0 = time.time()

    total_rows = 0
    chunks_written = 0
    first_chunk = True

    for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = clean_chunk(chunk)
        chunk = apply_taxonomy(chunk, ds_cfg['label_key'])

        # Keep only needed columns: node IDs, edge typing, features, time, labels
        needed_cols = (
            ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL'] +
            KEPT_FEATURES + [TIME_FIELD] + ['Label', 'Attack', 'unified_label']
        )
        needed_cols = [c for c in needed_cols if c in chunk.columns]
        chunk = chunk[needed_cols]

        # Write to parquet (append)
        out_path = PROCESSED_DIR / f'cleaned_{ds_name}.parquet'
        if first_chunk:
            chunk.to_parquet(out_path, index=False)
            first_chunk = False
        else:
            chunk.to_parquet(out_path, index=False, append=True)

        total_rows += len(chunk)
        chunks_written += 1
        if chunks_written % 20 == 0:
            print(f"    ... {total_rows:>12,} rows ({chunks_written} chunks)")

    elapsed = time.time() - t0
    cleaned_stats[ds_name] = {'rows': total_rows, 'chunks': chunks_written, 'time_s': elapsed}
    print(f"    [OK] {total_rows:,} rows in {chunks_written} chunks ({elapsed:.1f}s)")
    print(f"    Saved: cleaned_{ds_name}.parquet")

# ── STEP 2: Chronological Split ─────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: CHRONOLOGICAL SPLIT")
print("=" * 60)

label_to_idx = {name: i for i, name in enumerate(UNIFIED_CLASSES)}

for ds_name, ds_cfg in TRAIN_DATASETS.items():
    fpath = PROCESSED_DIR / f'cleaned_{ds_name}.parquet'
    if not fpath.exists():
        print(f"  SKIP {ds_name}: cleaned file not found")
        continue

    print(f"\n  {ds_name}:")
    # Read time column only for split
    df_time = pd.read_parquet(fpath, columns=[TIME_FIELD])
    n = len(df_time)

    # Sort by time and compute split boundaries
    sorted_idx = df_time[TIME_FIELD].argsort().values
    train_end = int(n * 0.70)
    val_end = int(n * (0.70 + 0.15))

    for split_name, (start, end) in {
        'train': (0, train_end),
        'val': (train_end, val_end),
        'test': (val_end, n),
    }.items():
        split_indices = sorted_idx[start:end]
        idx_df = pd.DataFrame({'original_index': split_indices})
        out = SPLIT_DIR / f'{ds_name}_{split_name}_index.parquet'
        idx_df.to_parquet(out, index=False)
        print(f"    {split_name}: {len(split_indices):>11,} flows -> {out.name}")

    # Log time boundaries
    train_times = df_time.iloc[sorted_idx[:train_end]][TIME_FIELD]
    val_times = df_time.iloc[sorted_idx[train_end:val_end]][TIME_FIELD]
    test_times = df_time.iloc[sorted_idx[val_end:]][TIME_FIELD]
    print(f"    Train time: [{train_times.min():.0f}, {train_times.max():.0f}] ms")
    print(f"    Val time:   [{val_times.min():.0f}, {val_times.max():.0f}] ms")
    print(f"    Test time:  [{test_times.min():.0f}, {test_times.max():.0f}] ms")
    assert train_times.max() <= val_times.min(), "TIME LEAKAGE: train overlaps val!"
    assert val_times.max() <= test_times.min(), "TIME LEAKAGE: val overlaps test!"
    print(f"    [OK] No time overlap between splits")

# ── STEP 3: Graph Construction ──────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: WINDOWED GRAPH CONSTRUCTION")
print("=" * 60)

def build_graphs_for_dataset(ds_name, split_name):
    """Build windowed PyG Data objects for a dataset split."""
    cleaned_path = PROCESSED_DIR / f'cleaned_{ds_name}.parquet'
    idx_path = SPLIT_DIR / f'{ds_name}_{split_name}_index.parquet'

    if not cleaned_path.exists() or not idx_path.exists():
        print(f"    SKIP: missing files")
        return []

    # Read split indices
    idx_df = pd.read_parquet(idx_path)
    split_indices = set(idx_df['original_index'].values)

    # Read data for this split
    graphs = []
    current_window = -1
    window_edges = []  # collect edges for current window

    # We need: node IDs, features, time, labels
    read_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT',
                 'PROTOCOL'] + KEPT_FEATURES + [TIME_FIELD, 'unified_label']
    read_cols = [c for c in read_cols if c in pd.read_parquet(cleaned_path, nrows=1).columns]

    # Node hashing cache (local to this function, cleared per window)
    node_id_cache = {}

    def hash_endpoint(ip, port):
        key = f"{ip}:{port}"
        if key not in node_id_cache:
            node_id_cache[key] = len(node_id_cache)
        return node_id_cache[key]

    # Process chunked
    for chunk in pd.read_parquet(cleaned_path, columns=read_cols):
        # Filter to split indices
        chunk = chunk[chunk.index.isin(split_indices)]
        if len(chunk) == 0:
            continue

        # Compute window index from time
        chunk_time_sec = chunk[TIME_FIELD].values / 1000.0

        if current_window < 0:
            t0 = chunk_time_sec.min()

        chunk['_window'] = ((chunk_time_sec - t0) // WINDOW_SIZE_SEC).astype(int)

        for w in chunk['_window'].unique():
            w_mask = chunk['_window'] == w
            w_data = chunk[w_mask]

            if len(w_data) == 0:
                continue

            # Build graph for this window
            src_ips = w_data['IPV4_SRC_ADDR'].values
            src_ports = w_data['L4_SRC_PORT'].values
            dst_ips = w_data['IPV4_DST_ADDR'].values
            dst_ports = w_data['L4_DST_PORT'].values

            src_nodes = [hash_endpoint(ip, port) for ip, port in zip(src_ips, src_ports)]
            dst_nodes = [hash_endpoint(ip, port) for ip, port in zip(dst_ips, dst_ports)]
            
            node_id_cache = {}  # PREP-1 fix: Reset per window
            src_nodes = [hash_endpoint(ip, port) for ip, port in zip(src_ips, src_ports)]
            dst_nodes = [hash_endpoint(ip, port) for ip, port in zip(dst_ips, dst_ports)]

            edge_index = np.stack([src_nodes, dst_nodes], axis=0)  # (2, E)

            # Edge features (41 kept features)
            feat_cols_in_chunk = [f for f in KEPT_FEATURES if f in w_data.columns]
            edge_attr = w_data[feat_cols_in_chunk].values.astype(np.float32)  # (E, 41)

            # Edge times
            edge_time = w_data[TIME_FIELD].values.astype(np.float64)  # (E,)

            # Labels
            unified_labels = w_data['unified_label'].values
            y = np.array([label_to_idx.get(l, 0) for l in unified_labels], dtype=np.int64)

            # Save as PyG Data
            import torch
            from torch_geometric.data import Data

            g = Data(
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                edge_time=torch.tensor(edge_time, dtype=torch.float32),
                y=torch.tensor(y, dtype=torch.long),
                num_nodes=len(node_id_cache),
                window_idx=int(w),
            )
            graphs.append(g)

    print(f"    Built {len(graphs)} windows")
    return graphs

for ds_name in TRAIN_DATASETS:
    cleaned_path = PROCESSED_DIR / f'cleaned_{ds_name}.parquet'
    if not cleaned_path.exists():
        continue
    print(f"\n  {ds_name}:")
    for split_name in ['train', 'val', 'test']:
        print(f"    {split_name}...", end=' ', flush=True)
        t0 = time.time()
        graphs = build_graphs_for_dataset(ds_name, split_name)

        if graphs:
            out_path = GRAPH_DIR / f'{ds_name}_{split_name}_list.pt'
            import torch
            torch.save(graphs, out_path)
            total_edges = sum(g.edge_index.shape[1] for g in graphs)
            print(f"{len(graphs)} windows, {total_edges:,} edges ({time.time()-t0:.1f}s)")

# ── STEP 4: Fit Scaler on E_train Only ──────────────────────
print("\n" + "=" * 60)
print("STEP 4: FEATURE SCALING (fit on E_train only)")
print("=" * 60)

from sklearn.preprocessing import StandardScaler

# Collect E_train features
all_train_features = []
for ds_name in TRAIN_DATASETS:
    graph_path = GRAPH_DIR / f'{ds_name}_train_list.pt'
    if graph_path.exists():
        import torch
        graphs = torch.load(graph_path, weights_only=False)
        for g in graphs:
            all_train_features.append(g.edge_attr.numpy())

all_train_features = np.concatenate(all_train_features, axis=0)
assert all_train_features.shape[0] > 10_000_000, "Scaler should be fit on both datasets"
print(f"  E_train samples for scaler: {all_train_features.shape[0]:,} x {all_train_features.shape[1]}")

scaler = StandardScaler()
scaler.fit(all_train_features)
print(f"  Feature means range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
print(f"  Feature stds range:  [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  Saved: {SCALER_PATH}")

# Apply scaler to all graphs
print("\n  Applying scaler to all graphs...")
import torch
for ds_name in TRAIN_DATASETS:
    for split_name in ['train', 'val', 'test']:
        graph_path = GRAPH_DIR / f'{ds_name}_{split_name}_list.pt'
        if graph_path.exists():
            graphs = torch.load(graph_path, weights_only=False)
            for g in graphs:
                g.edge_attr = torch.clamp(torch.tensor(
                    scaler.transform(g.edge_attr.numpy()),
                    dtype=torch.float32
                ), -10.0, 10.0)
            torch.save(graphs, graph_path)
            print(f"    {ds_name}_{split_name} — normalized")

# ── STEP 5: Preprocessing Report ────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: PREPROCESSING REPORT")
print("=" * 60)

report = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'edge_input_dim': EDGE_INPUT_DIM,
    'raw_features': len(KEPT_FEATURES),
    'time2vec_dim': 17,
    'window_size_sec': WINDOW_SIZE_SEC,
    'split_method': 'chronological',
    'split_column': TIME_FIELD,
    'split_ratio': '70/15/15',
    'cleaning': {
        'nan_filled_features': INF_NAN_FEATURES,
        'protocol_conditional_filled': ['ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER'],
        'inf_replaced': INF_NAN_FEATURES,
    },
    'datasets': {},
    'class_distribution_combined': {},
}

# Combined class distribution from training datasets
combined_counts = Counter()
for ds_name in TRAIN_DATASETS:
    cleaned_path = PROCESSED_DIR / f'cleaned_{ds_name}.parquet'
    if cleaned_path.exists():
        # Read just the unified_label column
        labels_df = pd.read_parquet(cleaned_path, columns=['unified_label'])
        ds_counts = Counter(labels_df['unified_label'].values)
        combined_counts.update(ds_counts)
        report['datasets'][ds_name] = {
            'total': int(len(labels_df)),
            'class_counts': {k: int(v) for k, v in ds_counts.items()},
        }

total_all = sum(combined_counts.values())
for cls in UNIFIED_CLASSES:
    c = combined_counts.get(cls, 0)
    report['class_distribution_combined'][cls] = {
        'count': int(c),
        'pct': round(c / total_all * 100, 3) if total_all > 0 else 0,
        'below_median': c < np.median(list(combined_counts.values())),
    }

report['imbalance_ratio'] = round(max(combined_counts.values()) / max(min(combined_counts.values()), 1), 1)
report['minority_classes'] = [cls for cls in UNIFIED_CLASSES
                              if combined_counts.get(cls, 0) < np.median(list(combined_counts.values()))]

with open(PROCESSED_DIR / 'preprocessing_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"  Report saved: preprocessing_report.json")
print(f"  Processed directory: {PROCESSED_DIR}")
print(f"\n  Files ready for Kaggle upload:")
for f in sorted(PROCESSED_DIR.rglob('*')):
    if f.is_file():
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.relative_to(PROCESSED_DIR)} ({size_mb:.1f} MB)")

print("\n" + "=" * 60)
print("L02 COMPLETE — Ready for Kaggle upload.")
print("=" * 60)
print("\nUpload to Kaggle:")
print("  1. All files from laptop/processed/graphs/")
print("  2. laptop/processed/scaler.pkl")
print("  3. laptop/processed/split_indices/")
print("  4. label_map.yaml, feature_manifest.yaml")
print("\nThen run Kaggle scripts: k01_time2vec_mae.py, k02_cvae.py, ...")
