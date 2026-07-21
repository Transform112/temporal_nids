"""
PREPROCESSING PIPELINE — Prepare data for Kaggle upload.
==========================================================
1. Clean NaN/Inf, apply labels → cleaned_{ds}.parquet
2. Chronological split (70/15/15) → split_indices/
3. Windowed graph construction (120s) → graphs/  (training datasets only)
4. Fit StandardScaler on E_train → scaler.pkl
5. Apply scaler to all graphs
6. Generate preprocessing report → preprocessing_report.json

Runs on CPU. Memory-efficient chunked processing.
Output: laptop/processed/  →  upload to Kaggle as a dataset
"""
import pandas as pd, numpy as np, yaml, json, pickle, os, time, sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'
PROCESSED = PROJECT_ROOT / 'laptop' / 'processed'
SPLIT_DIR = PROCESSED / 'split_indices'
GRAPH_DIR = PROCESSED / 'graphs'
SCALER_PATH = PROCESSED / 'scaler.pkl'
REPORT_PATH = PROCESSED / 'preprocessing_report.json'

for d in [PROCESSED, SPLIT_DIR, GRAPH_DIR]: d.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT/'label_map.yaml') as f: label_map = yaml.safe_load(f)
with open(PROJECT_ROOT/'feature_manifest.yaml') as f: fm = yaml.safe_load(f)

KEPT = fm['kept_features']       # 41
UNIFIED = label_map['unified_classes']
TIME_FIELD = fm['time_signal_field']
LABEL_TO_IDX = {n: i for i, n in enumerate(UNIFIED)}
WINDOW_SEC = 120
CHUNK = 500_000

INF_NAN_FEATURES = ['SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES']
PROTOCOL_FILL = ['ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER']

DATASETS = {
    'NF-CICIDS2018': {'file': 'NF-CICIDS2018-v3.csv', 'lk': 'NF-CSE-CIC-IDS2018', 'train': True},
    'NF-UNSW-NB15':  {'file': 'NF-UNSW-NB15-v3.csv',  'lk': 'NF-UNSW-NB15',       'train': True},
    'NF-ToN-IoT':    {'file': 'NF-ToN-IoT-v3.csv',     'lk': 'NF-ToN-IoT',         'train': False},
    'NF-BoT-IoT':    {'file': 'NF-BoT-IoT-v3.csv',     'lk': 'NF-BoT-IoT',         'train': False},
}

print("="*60)
print("PREPROCESSING PIPELINE — Preparing data for Kaggle")
print("="*60)
print(f"Features: {len(KEPT)} kept (41)")
print(f"Window size: {WINDOW_SEC}s")
print(f"Output: {PROCESSED}")

# ═══════════════════════════════════════════════════════════════
# STEP 1: Clean + Label + Save all datasets
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1: CLEAN + LABEL")
print("="*60)

all_clean_stats = {}

for ds_name, cfg in DATASETS.items():
    fpath = DATASET_DIR / cfg['file']
    if not fpath.exists():
        print(f"  SKIP {ds_name}: not found")
        continue

    out_dir = PROCESSED / f'cleaned_{ds_name}'
    if out_dir.exists() and any(out_dir.glob('*.parquet')):
        n_files = len(list(out_dir.glob('*.parquet')))
        print(f"\n  {ds_name}: SKIP ({n_files} chunks already exist)")
        continue

    fs_gb = fpath.stat().st_size/1e9
    print(f"\n  {ds_name} ({fs_gb:.2f} GB) ...")
    t0 = time.time()
    mapping = label_map[cfg['lk']]
    total = 0; chunks_done = 0
    out_dir = PROCESSED / f'cleaned_{ds_name}'
    out_dir.mkdir(parents=True, exist_ok=True)

    for chunk in pd.read_csv(fpath, chunksize=CHUNK, low_memory=False):
        chunk.columns = chunk.columns.str.strip()

        # Clean NaN/Inf
        for feat in PROTOCOL_FILL:
            if feat in chunk.columns:
                chunk[feat] = chunk[feat].fillna(0)
        for feat in INF_NAN_FEATURES:
            if feat in chunk.columns:
                chunk[feat] = chunk[feat].replace([np.inf, -np.inf], np.nan).fillna(0)
        for feat in KEPT:
            if feat in chunk.columns:
                chunk[feat] = chunk[feat].fillna(0).replace([np.inf, -np.inf], 0)

        # Apply label mapping
        chunk['unified_label'] = chunk['Attack'].map(mapping)
        unmapped = chunk['unified_label'].isna()
        if unmapped.any():
            bad = chunk.loc[unmapped, 'Attack'].unique()
            print(f"    WARNING: {unmapped.sum()} rows unmapped: {list(bad)[:5]}")
            chunk = chunk[~unmapped]

        # Keep only needed columns
        needed = []
        seen = set()
        for c in (['IPV4_SRC_ADDR','IPV4_DST_ADDR','L4_SRC_PORT','L4_DST_PORT'] +
                  KEPT + [TIME_FIELD, 'unified_label']):
            if c in chunk.columns and c not in seen:
                needed.append(c); seen.add(c)
        chunk = chunk[needed]

        # Write parquet chunk
        chunk_path = out_dir / f'part_{chunks_done:05d}.parquet'
        chunk.to_parquet(chunk_path, index=False)

        total += len(chunk); chunks_done += 1
        if chunks_done % 20 == 0:
            elapsed = time.time()-t0
            print(f"    {total:>12,} rows | {total/elapsed:,.0f} rows/s")

    elapsed = time.time()-t0
    all_clean_stats[ds_name] = {'rows': total, 'time_s': round(elapsed, 1)}
    total_mb = sum(f.stat().st_size for f in out_dir.glob('*.parquet')) / 1e6
    print(f"    [OK] {total:,} rows in {elapsed:.0f}s -> {total_mb:.0f} MB ({chunks_done} chunks)")

# ═══════════════════════════════════════════════════════════════
# STEP 2: Chronological Split
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2: CHRONOLOGICAL SPLIT (70/15/15)")
print("="*60)

split_info = {}

for ds_name, cfg in DATASETS.items():
    cleaned_path = PROCESSED / f'cleaned_{ds_name}'
    if not cleaned_path.exists(): continue

    print(f"\n  {ds_name}:")
    df_time = pd.read_parquet(cleaned_path, columns=[TIME_FIELD])
    n = len(df_time)
    sort_idx = df_time[TIME_FIELD].argsort().values
    train_end = int(n * 0.70); val_end = int(n * (0.70 + 0.15))

    split_info[ds_name] = {}
    for sp, (start, end) in [('train',(0,train_end)),('val',(train_end,val_end)),('test',(val_end,n))]:
        sp_idx = sort_idx[start:end]
        pd.DataFrame({'original_index': sp_idx}).to_parquet(
            SPLIT_DIR / f'{ds_name}_{sp}_index.parquet', index=False)

        sp_times = df_time.iloc[sp_idx][TIME_FIELD]
        split_info[ds_name][sp] = {
            'n': end-start, 't_min': sp_times.min(), 't_max': sp_times.max()}

        print(f"    {sp}: {end-start:>11,} flows  "
              f"[{sp_times.min():.0f}, {sp_times.max():.0f}]")

    # Verify no overlap
    assert split_info[ds_name]['train']['t_max'] <= split_info[ds_name]['val']['t_min'], \
        f"TIME LEAK in {ds_name}!"
    assert split_info[ds_name]['val']['t_max'] <= split_info[ds_name]['test']['t_min'], \
        f"TIME LEAK in {ds_name}!"
    print(f"    [OK] Chronological split verified")

# ═══════════════════════════════════════════════════════════════
# STEP 3: Graph Construction (training datasets only)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3: WINDOWED GRAPH CONSTRUCTION (training datasets)")
print("="*60)

for ds_name, cfg in DATASETS.items():
    if not cfg['train']:  # skip blind test datasets
        print(f"\n  {ds_name}: SKIP (blind test — graphs built on Kaggle if needed)")
        continue

    cleaned_path = PROCESSED / f'cleaned_{ds_name}'
    if not cleaned_path.exists(): continue

    print(f"\n  {ds_name}:")

    for sp in ['train', 'val', 'test']:
        idx_path = SPLIT_DIR / f'{ds_name}_{sp}_index.parquet'
        out_path = GRAPH_DIR / f'{ds_name}_{sp}_list.pt'

        if out_path.exists():
            graphs = torch.load(out_path, weights_only=False)
            print(f"    {sp}: {len(graphs)} windows (already exists, skip)")
            continue

        print(f"    {sp}: building graphs...", end=' ', flush=True)
        t0 = time.time()

        # Load split indices
        split_idx = set(pd.read_parquet(idx_path)['original_index'].values)

        # Read columns needed for graph construction
        read_cols = []
        seen2 = set()
        for c in (['IPV4_SRC_ADDR','IPV4_DST_ADDR','L4_SRC_PORT','L4_DST_PORT'] +
                  KEPT + [TIME_FIELD, 'unified_label']):
            if c not in seen2:
                read_cols.append(c); seen2.add(c)
        # Filter to columns that exist in the cleaned data
        first_file = next(cleaned_path.glob('*.parquet'))
        valid_cols = pd.read_parquet(first_file).columns
        read_cols = [c for c in read_cols if c in valid_cols]

        graphs = []
        node_cache = {}
        t0_ref = None

        # Read parquet files one at a time
        # Track global position since parquet files have independent RangeIndex
        parquet_files = sorted(cleaned_path.glob('*.parquet'))
        global_pos = 0
        for pf in parquet_files:
            df_chunk = pd.read_parquet(pf, columns=read_cols)
            n_local = len(df_chunk)

            # Check which global positions are in split_idx
            local_in_split = np.array([(global_pos + i) in split_idx for i in range(n_local)])
            df_chunk = df_chunk[local_in_split]
            global_pos += n_local
            if len(df_chunk) == 0: continue

            # Window assignment
            chunk_time_s = df_chunk[TIME_FIELD].values / 1000.0
            if t0_ref is None:
                t0_ref = chunk_time_s.min()
            df_chunk['_w'] = ((chunk_time_s - t0_ref) // WINDOW_SEC).astype(int)

            for w in df_chunk['_w'].unique():
                w_mask = df_chunk['_w'] == w
                w_data = df_chunk[w_mask]
                if len(w_data) < 2: continue

                # Hash endpoints
                src_keys = [f"{ip}:{port}" for ip, port in
                           zip(w_data['IPV4_SRC_ADDR'].values, w_data['L4_SRC_PORT'].values)]
                dst_keys = [f"{ip}:{port}" for ip, port in
                           zip(w_data['IPV4_DST_ADDR'].values, w_data['L4_DST_PORT'].values)]

                for key in src_keys + dst_keys:
                    if key not in node_cache:
                        node_cache[key] = len(node_cache)

                src_nodes = [node_cache[k] for k in src_keys]
                dst_nodes = [node_cache[k] for k in dst_keys]

                feat_cols = [f for f in KEPT if f in w_data.columns]
                ea = torch.tensor(w_data[feat_cols].values.astype(np.float32))
                et = torch.tensor(w_data[TIME_FIELD].values.astype(np.float32))
                y = torch.tensor([LABEL_TO_IDX.get(l, 0) for l in w_data['unified_label'].values],
                                 dtype=torch.long)

                g = Data(
                    edge_index=torch.tensor([src_nodes, dst_nodes], dtype=torch.long),
                    edge_attr=ea, edge_time=et, y=y,
                    num_nodes=len(node_cache), window_idx=int(w),
                )
                graphs.append(g)

        torch.save(graphs, out_path)
        total_e = sum(g.edge_index.shape[1] for g in graphs)
        print(f"{len(graphs)} windows, {total_e:,} edges ({time.time()-t0:.0f}s)")

# ═══════════════════════════════════════════════════════════════
# STEP 4: Fit Scaler on E_train
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 4: SCALER (fit on E_train only)")
print("="*60)

from sklearn.preprocessing import StandardScaler

all_train_ea = []
for ds_name, cfg in DATASETS.items():
    if not cfg['train']: continue
    gpath = GRAPH_DIR / f'{ds_name}_train_list.pt'
    if gpath.exists():
        graphs = torch.load(gpath, weights_only=False)
        for g in graphs:
            all_train_ea.append(g.edge_attr.numpy())
        print(f"  {ds_name}: {sum(g.edge_index.shape[1] for g in graphs):,} train edges")

all_train_ea = np.concatenate(all_train_ea, axis=0)
print(f"  Total E_train samples for scaler: {all_train_ea.shape[0]:,} x {all_train_ea.shape[1]}")

scaler = StandardScaler()
scaler.fit(all_train_ea)
print(f"  Mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
print(f"  Std range:  [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

# Apply to all graphs
print("\n  Applying scaler to all graphs...")
for ds_name, cfg in DATASETS.items():
    for sp in ['train', 'val', 'test']:
        gpath = GRAPH_DIR / f'{ds_name}_{sp}_list.pt'
        if gpath.exists():
            graphs = torch.load(gpath, weights_only=False)
            for g in graphs:
                g.edge_attr = torch.tensor(scaler.transform(g.edge_attr.numpy()), dtype=torch.float32)
            torch.save(graphs, gpath)
            print(f"    {ds_name}_{sp}: normalized")

# ═══════════════════════════════════════════════════════════════
# STEP 5: Report
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 5: REPORT")
print("="*60)

report = {
    'created': datetime.now(timezone.utc).isoformat(),
    'features_kept': len(KEPT),
    'edge_input_dim': len(KEPT) + 17,  # 41 + 17 = 58
    'window_sec': WINDOW_SEC,
    'split': 'chronological 70/15/15',
    'split_column': TIME_FIELD,
    'datasets': {},
    'scaler': {
        'n_samples': int(all_train_ea.shape[0]),
        'n_features': int(all_train_ea.shape[1]),
        'mean_range': [float(scaler.mean_.min()), float(scaler.mean_.max())],
        'std_range': [float(scaler.scale_.min()), float(scaler.scale_.max())],
    },
}

for ds_name, cfg in DATASETS.items():
    gpath = GRAPH_DIR / f'{ds_name}_train_list.pt'
    if gpath.exists():
        graphs = torch.load(gpath, weights_only=False)
        report['datasets'][ds_name] = {
            'train_windows': len(graphs),
            'train_edges': sum(g.edge_index.shape[1] for g in graphs),
            'is_training': cfg['train'],
        }

with open(REPORT_PATH, 'w') as f:
    json.dump(report, f, indent=2, default=str)

# Inventory
print(f"\n  Processed directory ready: {PROCESSED}")
print(f"\n  FILES FOR KAGGLE UPLOAD:")
total_mb = 0
for f in sorted(PROCESSED.rglob('*')):
    if f.is_file():
        mb = f.stat().st_size/1e6; total_mb += mb
        print(f"    {f.relative_to(PROCESSED)} ({mb:.1f} MB)")
print(f"\n  TOTAL: {total_mb:.0f} MB")
print(f"\n  Upload these files as a Kaggle dataset named 'ids-nf3-processed'")
print(f"  Then run k01 → k02 → k03 → k04 → k05 → k06 on Kaggle T4x2.")
print("\n" + "="*60)
print("PREPROCESSING COMPLETE")
print("="*60)

