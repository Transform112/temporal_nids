"""
L02 — Feature Analysis & Correlation Pruning
==============================================
RUNS ON LAPTOP (CPU only). Reads cleaned parquet files.
Computes pairwise Pearson correlation on a STRATIFIED SAMPLE (not full dataset).
Updates feature_manifest.yaml with pruned features.

Memory: Samples 100k rows per dataset for correlation computation.
Output: Updated feature_manifest.yaml, tab03_feature_schema.csv/.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
import warnings
warnings.filterwarnings('ignore')

# ---- CONFIG ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = PROJECT_ROOT / 'dataset' / 'cleaned'
SAMPLE_SIZE = 100_000  # rows per dataset for correlation

# ---- Load manifests ----
with open(PROJECT_ROOT / 'feature_manifest.yaml', 'r') as f:
    fm = yaml.safe_load(f)
with open(PROJECT_ROOT / 'label_map.yaml', 'r') as f:
    label_map = yaml.safe_load(f)

KEPT_FEATURES = fm['kept_features']
CORRELATION_THRESHOLD = 0.95
MAX_PRUNED = 4  # Never drop more than 4 features via correlation

DATASETS = ['NF-CICIDS2018', 'NF-UNSW-NB15']


def compute_correlations():
    """Load samples from training datasets and compute pairwise correlations."""
    all_samples = []

    for ds_name in DATASETS:
        path = CLEANED_DIR / f'{ds_name}_cleaned'
        if not path.exists():
            print(f"  SKIPPED: {path} not found. Run L01 first.")
            continue

        print(f"  Reading {ds_name}...")
        df = pd.read_parquet(path)

        # Take a stratified sample (balanced across labels)
        n_sample = min(SAMPLE_SIZE, len(df))
        sampled = df.groupby('unified_label', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max(1, n_sample // df['unified_label'].nunique())),
                               random_state=42)
        ).reset_index(drop=True)
        sampled = sampled.sample(n=min(n_sample, len(sampled)), random_state=42)

        # Extract feature columns
        feature_cols = [c for c in KEPT_FEATURES if c in sampled.columns]
        all_samples.append(sampled[feature_cols])
        print(f"    Sampled {len(sampled):,} rows x{len(feature_cols)} features")

    if not all_samples:
        print("  No data loaded. Exiting.")
        return None

    combined = pd.concat(all_samples, ignore_index=True)
    print(f"\n  Combined sample: {combined.shape[0]:,} x{combined.shape[1]}")

    # Compute correlation matrix
    print(f"  Computing Pearson correlation on {combined.shape[1]} features...")
    corr_matrix = combined.corr(method='pearson')
    print(f"  Correlation matrix: {corr_matrix.shape}")

    return corr_matrix


def find_highly_correlated(corr_matrix):
    """Find feature pairs with |r| > threshold, pick the less-interpretable to drop."""
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > CORRELATION_THRESHOLD:
                high_corr_pairs.append({
                    'feature_a': corr_matrix.columns[i],
                    'feature_b': corr_matrix.columns[j],
                    'correlation': round(float(r), 4),
                })

    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

    print(f"\n  Found {len(high_corr_pairs)} pairs with |r| > {CORRELATION_THRESHOLD}:")
    for pair in high_corr_pairs[:20]:  # print top 20
        print(f"    r={pair['correlation']:+.4f}: {pair['feature_a']} <-> {pair['feature_b']}")

    return high_corr_pairs


def select_features_to_prune(high_corr_pairs):
    """
    For each correlated pair, keep the MORE interpretable feature.
    Interpretability ranking (higher = keep):
      - Named behavioral features (TCP_FLAGS, FLOW_DURATION, etc.) > computed stats (AVG, STDDEV)
      - Kept features with clearer SHAP-readable names preferred
    """
    # Features that are harder to interpret (prefer to drop these)
    less_interpretable_patterns = [
        'STDDEV', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
        'NUM_PKTS_1024_TO_1514_BYTES',  # tail of histogram
        'DST_TO_SRC_AVG_THROUGHPUT',  # redundant with SRC_TO_DST
        'DURATION_OUT',  # redundant with FLOW_DURATION
    ]

    dropped = set()
    for pair in high_corr_pairs:
        if len(dropped) >= MAX_PRUNED:
            break

        a, b = pair['feature_a'], pair['feature_b']

        # Skip if either already dropped
        if a in dropped or b in dropped:
            continue

        # Decide which to drop
        a_less_interpretable = any(p in a for p in less_interpretable_patterns)
        b_less_interpretable = any(p in b for p in less_interpretable_patterns)

        if a_less_interpretable and not b_less_interpretable:
            dropped.add(a)
            print(f"  DROP {a} (keep {b}): r={pair['correlation']:+.4f}")
        elif b_less_interpretable and not a_less_interpretable:
            dropped.add(b)
            print(f"  DROP {b} (keep {a}): r={pair['correlation']:+.4f}")
        elif a_less_interpretable and b_less_interpretable:
            dropped.add(a)  # arbitrary tie-break
            print(f"  DROP {a} (keep {b}): r={pair['correlation']:+.4f} [both low-interp, kept {b}]")
        else:
            # Both interpretable — keep both, flag for manual review
            print(f"  KEEP BOTH: {a} <->{b} (r={pair['correlation']:+.4f}) — both interpretable, manual review needed")

    return list(dropped)


def update_feature_manifest(pruned_features):
    """Update feature_manifest.yaml with correlation pruning results."""
    if not pruned_features:
        print("\n  No features to prune. feature_manifest.yaml unchanged.")
        return

    fm['correlation_pruned'] = pruned_features
    fm['final_raw_feature_count'] = len(KEPT_FEATURES) - len(pruned_features)
    fm['final_edge_input_dim'] = fm['final_raw_feature_count'] + fm['time2vec_dim']

    # Update kept_features list
    updated_kept = [f for f in KEPT_FEATURES if f not in pruned_features]
    fm['kept_features'] = updated_kept

    with open(PROJECT_ROOT / 'feature_manifest.yaml', 'w') as f:
        yaml.dump(fm, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Updated feature_manifest.yaml:")
    print(f"    Pruned: {pruned_features}")
    print(f"    Raw features: {fm['final_raw_feature_count']}")
    print(f"    Edge input dim: {fm['final_edge_input_dim']}")


# ---- MAIN ----
if __name__ == '__main__':
    print("L02 — Feature Analysis & Correlation Pruning")
    print(f"Datasets: {DATASETS}")
    print(f"Correlation threshold: |r| > {CORRELATION_THRESHOLD}")
    print(f"Max features to prune: {MAX_PRUNED}")

    corr_matrix = compute_correlations()
    if corr_matrix is None:
        print("\nERROR: Correlation computation failed. Check that L01 completed successfully.")
        sys.exit(1)

    high_corr_pairs = find_highly_correlated(corr_matrix)
    pruned = select_features_to_prune(high_corr_pairs)
    update_feature_manifest(pruned)

    print(f"\nFeature analysis complete. {len(pruned)} features pruned.")
    print("Next: Run L03_chronological_split.py")
