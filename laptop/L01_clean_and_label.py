"""
L01 — Data Cleaning, NaN/Inf Handling & Label Reassignment
============================================================
RUNS ON LAPTOP (CPU only, memory-efficient chunked processing).
Inputs raw NF3 CSVs, outputs cleaned parquet files.

What this does:
  1. Strips whitespace from column names
  2. Detects & handles NaN/inf in feature columns
  3. Fills protocol-conditional fields (ICMP/DNS) with 0 for non-applicable flows
  4. Maps raw attack labels → unified 11-class taxonomy
  5. Drops flows with unmappable labels
  6. Saves cleaned data as parquet per dataset

Memory: chunksize=500,000 rows, never holds full dataset in RAM.
Output: dataset/cleaned/{dataset_name}_cleaned.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys
import warnings
warnings.filterwarnings('ignore')

# ---- CONFIG ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'
OUTPUT_DIR = DATASET_DIR / 'cleaned'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 500_000  # rows per chunk — fits in laptop RAM

# ---- Load YAML manifests ----
with open(PROJECT_ROOT / 'label_map.yaml', 'r') as f:
    label_map = yaml.safe_load(f)
with open(PROJECT_ROOT / 'feature_manifest.yaml', 'r') as f:
    fm = yaml.safe_load(f)

KEPT_FEATURES = fm['kept_features']
DROPPED_FIELDS = fm['dropped_fields']
PROTOCOL_CONDITIONAL = fm.get('protocol_conditional_fields', {
    'ICMP': ['ICMP_TYPE', 'ICMP_IPV4_TYPE'],
    'DNS': ['DNS_QUERY_TYPE', 'DNS_TTL_ANSWER'],
})

# Dataset configs
DATASETS = {
    'NF-CICIDS2018': {
        'file': DATASET_DIR / 'NF-CICIDS2018-v3.csv',
        'label_key': 'NF-CSE-CIC-IDS2018',
    },
    'NF-UNSW-NB15': {
        'file': DATASET_DIR / 'NF-UNSW-NB15-v3.csv',
        'label_key': 'NF-UNSW-NB15',
    },
    'NF-ToN-IoT': {
        'file': DATASET_DIR / 'NF-ToN-IoT-v3.csv',
        'label_key': 'NF-ToN-IoT',
    },
    'NF-BoT-IoT': {
        'file': DATASET_DIR / 'NF-BoT-IoT-v3.csv',
        'label_key': 'NF-BoT-IoT',
    },
}

NAN_FILL_VALUE = 0.0  # Fill NaN in features with 0
INF_REPLACEMENT = 0.0  # Replace inf with 0


def clean_chunk(chunk, dataset_name, label_key):
    """
    Clean a single chunk of data.
    Returns cleaned chunk or None if unrecoverable.
    """
    # 1. Strip whitespace from column names
    chunk.columns = chunk.columns.str.strip()

    # 2. Identify feature columns present in this chunk
    feature_cols = [c for c in KEPT_FEATURES if c in chunk.columns]
    all_feature_cols = [c for c in fm['kept_features'] + DROPPED_FIELDS if c in chunk.columns]

    # 3. Detect NaN in features
    nan_counts = {}
    for col in all_feature_cols:
        if col in chunk.columns:
            n_nan = chunk[col].isna().sum()
            if n_nan > 0:
                nan_counts[col] = int(n_nan)

    if nan_counts:
        total_nan = sum(nan_counts.values())
        if total_nan > chunk.shape[0] * 0.3:  # >30% NaN is suspicious
            print(f"  WARNING: {total_nan} NaN values across {len(nan_counts)} columns")

    # 4. Fill NaN in features with 0 (NOT -1 or separate indicator)
    for col in feature_cols:
        if col in chunk.columns and chunk[col].isna().any():
            chunk[col] = chunk[col].fillna(NAN_FILL_VALUE)

    # 5. Handle inf
    for col in feature_cols:
        if col in chunk.columns:
            inf_mask = np.isinf(chunk[col].values)
            if inf_mask.any():
                n_inf = inf_mask.sum()
                chunk.loc[inf_mask, col] = INF_REPLACEMENT
                print(f"  Fixed {n_inf} inf values in '{col}'")

    # 6. Fill protocol-conditional fields with 0
    for protocol, fields in PROTOCOL_CONDITIONAL.items():
        for field in fields:
            if field in chunk.columns and chunk[field].isna().any():
                chunk[field] = chunk[field].fillna(0)

    # 7. Apply label mapping
    mapping = label_map[label_key]
    chunk['unified_label'] = chunk['Attack'].map(mapping)

    # 8. Drop rows with unmappable labels
    unmapped_mask = chunk['unified_label'].isna()
    n_unmapped = unmapped_mask.sum()
    if n_unmapped > 0:
        unmapped_labels = chunk.loc[unmapped_mask, 'Attack'].unique()
        print(f"  Dropping {n_unmapped} rows with unmapped labels: {list(unmapped_labels)}")
        chunk = chunk[~unmapped_mask]

    # 9. Keep only essential columns to reduce memory (deduplicate)
    essential_cols = list(dict.fromkeys(
        ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT',
         'PROTOCOL', 'FLOW_START_MILLISECONDS'] +
        [c for c in KEPT_FEATURES if c in chunk.columns] +
        ['Label', 'unified_label']
    ))

    return chunk[essential_cols]


def process_dataset(dataset_name, config):
    """Process a dataset end-to-end with chunked reading/writing."""
    filepath = config['file']
    label_key = config['label_key']

    if not filepath.exists():
        print(f"  SKIPPED: file not found at {filepath}")
        return False

    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"  File: {filepath.name}")
    print(f"  Label key: {label_key}")
    print(f"{'='*60}")

    output_dir = OUTPUT_DIR / f'{dataset_name}_cleaned'
    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    total_dropped = 0
    chunk_count = 0

    try:
        for chunk in pd.read_csv(filepath, chunksize=CHUNKSIZE, low_memory=False):
            chunk_count += 1
            n_before = len(chunk)

            cleaned = clean_chunk(chunk, dataset_name, label_key)
            if cleaned is None or len(cleaned) == 0:
                continue

            n_after = len(cleaned)
            total_rows += n_after
            total_dropped += (n_before - n_after)

            # Write each chunk as a separate parquet file
            chunk_path = output_dir / f'chunk_{chunk_count:04d}.parquet'
            cleaned.to_parquet(chunk_path, engine='pyarrow', index=False)

            if chunk_count % 10 == 0:
                print(f"  ... processed {total_rows:,} rows ({chunk_count} chunks)")

        print(f"\n  COMPLETE: {total_rows:,} rows kept, {total_dropped:,} dropped")
        print(f"  Output: {output_dir}/ ({chunk_count} parquet files)")

        # Get total size
        total_size = sum(f.stat().st_size for f in output_dir.glob('*.parquet'))
        print(f"  Size: {total_size / (1024*1024):.1f} MB")

        # Quick validation
        print(f"  Validating...")
        df_check = pd.read_parquet(output_dir)
        print(f"  Columns: {list(df_check.columns)}")
        print(f"  Unified labels: {sorted(df_check['unified_label'].unique())}")
        print(f"  NaN check: {df_check[KEPT_FEATURES[:5]].isna().sum().sum()} NaN in first 5 features")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---- MAIN ----
if __name__ == '__main__':
    print("L01 — Data Cleaning & Label Reassignment")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Chunk size: {CHUNKSIZE:,} rows")
    print(f"Datasets to process: {list(DATASETS.keys())}")

    results = {}
    for name, config in DATASETS.items():
        ok = process_dataset(name, config)
        results[name] = ok

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {name}")

    all_ok = all(results.values())
    print(f"\n{'All datasets processed successfully.' if all_ok else 'Some datasets failed -- check errors above.'}")

    if all_ok:
        print("\nNext: Run L02_feature_analysis.py")
