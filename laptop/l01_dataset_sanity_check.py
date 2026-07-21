"""
L01 — Dataset Sanity Check & Preliminary Analysis
==================================================
Runs LOCALLY on CPU. Memory-efficient chunked reading.
No GPU required. Produces dataset statistics, class distribution,
feature analysis, and preliminary figures.

Outputs (saved to laptop/outputs/):
  - Dataset statistics table
  - Class distribution plot (pre-augmentation)
  - Feature correlation heatmap (sample-based)
  - NaN/Inf report
  - Split strategy visualization
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# -- Config --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'
OUTPUT_DIR = PROJECT_ROOT / 'laptop' / 'outputs'
FIGURES_DIR = OUTPUT_DIR / 'figures'
TABLES_DIR = OUTPUT_DIR / 'tables'

for d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500_000
SAMPLE_FOR_CORR = 50_000  # rows to sample for correlation matrix

# Load YAML manifests
with open(PROJECT_ROOT / 'label_map.yaml', 'r') as f:
    label_map = yaml.safe_load(f)
with open(PROJECT_ROOT / 'feature_manifest.yaml', 'r') as f:
    fm = yaml.safe_load(f)

KEPT_FEATURES = fm['kept_features']
DROPPED_FIELDS = fm['dropped_fields']
UNIFIED_CLASSES = label_map['unified_classes']
PRUNED = fm.get('correlation_pruned', [])
RAW_DIM = fm['final_raw_feature_count']  # 41
EDGE_DIM = fm['final_edge_input_dim']     # 58

print("=" * 70)
print("L01 — DATASET SANITY CHECK")
print("=" * 70)
print(f"Kept features: {RAW_DIM} (3 pruned: {PRUNED})")
print(f"Edge input dim: {EDGE_DIM} (41 raw + 17 Time2Vec)")
print(f"Unified classes: {len(UNIFIED_CLASSES)}")
print(f"Chunk size: {CHUNK_SIZE:,} rows")

# -- Dataset inventory ---------------------------------------
DATASETS = {
    'NF-CICIDS2018': {
        'file': DATASET_DIR / 'NF-CICIDS2018-v3.csv',
        'label_key': 'NF-CSE-CIC-IDS2018',
        'role': 'Train/Val/Test (primary)',
    },
    'NF-UNSW-NB15': {
        'file': DATASET_DIR / 'NF-UNSW-NB15-v3.csv',
        'label_key': 'NF-UNSW-NB15',
        'role': 'Train/Val/Test (primary)',
    },
    'NF-ToN-IoT': {
        'file': DATASET_DIR / 'NF-ToN-IoT-v3.csv',
        'label_key': 'NF-ToN-IoT',
        'role': 'Cross-dataset blind test (in-schema)',
    },
    'NF-BoT-IoT': {
        'file': DATASET_DIR / 'NF-BoT-IoT-v3.csv',
        'label_key': 'NF-BoT-IoT',
        'role': 'Cross-dataset blind test (in-schema)',
    },
}

# -- STEP 1: Scan all datasets -------------------------------
print("\n" + "=" * 70)
print("STEP 1: DATASET SCAN (chunked, memory-efficient)")
print("=" * 70)

all_stats = {}
all_class_counts = {}  # dataset -> Counter of unified class
nan_reports = {}
inf_reports = {}
label_mapping_issues = defaultdict(list)
column_schemas = {}

for ds_name, ds_cfg in DATASETS.items():
    fpath = ds_cfg['file']
    if not fpath.exists():
        print(f"\n[!!]  {ds_name}: FILE NOT FOUND — {fpath}")
        continue

    file_size_gb = fpath.stat().st_size / 1e9
    print(f"\n{'='*60}")
    print(f"  {ds_name}  ({file_size_gb:.2f} GB)")
    print(f"  Role: {ds_cfg['role']}")
    print(f"{'='*60}")

    total_rows = 0
    class_counter = Counter()
    nan_counts = defaultdict(int)
    inf_counts = defaultdict(int)
    label_issues = 0
    feature_mins = {}
    feature_maxs = {}
    feature_means = defaultdict(float)
    time_min = float('inf')
    time_max = float('-inf')
    first_chunk_cols = None

    n_chunks = 0
    for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=False):
        # Strip column name whitespace
        chunk.columns = chunk.columns.str.strip()

        if first_chunk_cols is None:
            first_chunk_cols = list(chunk.columns)
            # Verify expected columns
            expected = set(fm['kept_features'] + fm['dropped_fields'] + ['Label', 'Attack'])
            actual = set(chunk.columns)
            missing = expected - actual
            extra = actual - expected
            if missing:
                print(f"  [!!]  MISSING columns: {sorted(missing)[:10]}...")
            if extra:
                print(f"  [!!]  EXTRA columns (not in manifest): {sorted(extra)[:10]}...")
            column_schemas[ds_name] = first_chunk_cols

        n_chunks += 1
        n_rows = len(chunk)
        total_rows += n_rows

        # --- NaN/Inf check (on kept features only) ---
        for feat in KEPT_FEATURES:
            if feat in chunk.columns:
                col = chunk[feat]
                nan_c = col.isna().sum()
                inf_c = (np.isinf(col.replace([np.inf, -np.inf], np.nan).dropna()) if col.dtype in ('float64','float32') else 0)
                if nan_c > 0:
                    nan_counts[feat] += nan_c
                if hasattr(inf_c, 'sum'):
                    inf_counts[feat] += int(inf_c.sum()) if hasattr(inf_c, 'sum') else 0

        # --- Feature range (running min/max) ---
        for feat in KEPT_FEATURES:
            if feat in chunk.columns:
                col = chunk[feat].dropna()
                if len(col) > 0:
                    if feat not in feature_mins:
                        feature_mins[feat] = col.min()
                        feature_maxs[feat] = col.max()
                    else:
                        feature_mins[feat] = min(feature_mins[feat], col.min())
                        feature_maxs[feat] = max(feature_maxs[feat], col.max())
                    feature_means[feat] += col.mean() * len(col)

        # --- Time range ---
        if 'FLOW_START_MILLISECONDS' in chunk.columns:
            time_min = min(time_min, chunk['FLOW_START_MILLISECONDS'].min())
            time_max = max(time_max, chunk['FLOW_START_MILLISECONDS'].max())

        # --- Label mapping ---
        label_mapping = label_map[ds_cfg['label_key']]
        if 'Attack' in chunk.columns:
            attack_vals = chunk['Attack'].fillna('UNKNOWN')
            for raw_label in attack_vals.unique():
                if raw_label not in label_mapping and raw_label != 'UNKNOWN':
                    label_mapping_issues[ds_name].append(raw_label)
                    label_issues += 1

            # Map to unified and count
            mapped = attack_vals.map(label_mapping).fillna('UNMAPPED')
            class_counter.update(mapped.values)

        # --- Progress ---
        if n_chunks % 10 == 0:
            print(f"  ... {total_rows:>12,} rows processed ({n_chunks} chunks)")

    # Finalize feature means
    for feat in feature_means:
        feature_means[feat] /= max(total_rows, 1)

    # Store results
    all_stats[ds_name] = {
        'total_rows': total_rows,
        'file_size_gb': round(file_size_gb, 2),
        'role': ds_cfg['role'],
        'time_range_ms': (time_min, time_max),
        'time_range_hours': (time_max - time_min) / 3_600_000 if time_min != float('inf') else 0,
        'n_chunks': n_chunks,
        'label_issues': label_issues,
    }
    all_class_counts[ds_name] = class_counter
    nan_reports[ds_name] = dict(nan_counts)
    inf_reports[ds_name] = dict(inf_counts)

    # Print summary
    benign_c = class_counter.get('Benign', 0)
    attack_c = total_rows - benign_c - class_counter.get('UNMAPPED', 0)
    print(f"\n  [OK] TOTAL: {total_rows:>12,} rows")
    print(f"  [OK] Benign: {benign_c:>11,} ({benign_c/total_rows*100:.1f}%)")
    print(f"  [OK] Attack: {attack_c:>11,} ({attack_c/total_rows*100:.1f}%)")
    print(f"  [OK] Time span: {(time_max-time_min)/3_600_000:.1f} hours")
    if nan_counts:
        print(f"  [!!]  NaN found in: {sorted(nan_counts.keys())[:8]}...")
    else:
        print(f"  [OK] No NaN values found")
    if inf_counts:
        print(f"  [!!]  Inf found in: {sorted(inf_counts.keys())[:8]}...")
    else:
        print(f"  [OK] No Inf values found")
    if label_issues > 0:
        print(f"  [!!]  {label_issues} rows with unmappable labels: {label_mapping_issues[ds_name][:10]}...")
    else:
        print(f"  [OK] All labels mappable")

# -- STEP 2: Split Strategy Report ---------------------------
print("\n" + "=" * 70)
print("STEP 2: SPLIT STRATEGY")
print("=" * 70)
print("""
+-------------------------------------------------------------+
| CHRONOLOGICAL SPLIT (Time-Respecting)                       |
|                                                             |
|  Train (70%)         Val (15%)      Test (15%)              |
|  [earliest flows] -> [middle flows] -> [latest flows]         |
|                                                             |
|  Why: Prevents look-ahead leakage in temporal graphs.       |
|  Random splits let the model see "future" flows during      |
|  training — artificially inflates metrics by 10-30%.        |
|                                                             |
|  Split key: FLOW_START_MILLISECONDS (not random seed)       |
|  Three SEPARATE physical graphs: G_train, G_val, G_test     |
|  Scaler fit on E_train ONLY, applied frozen to val/test     |
|  Time2Vec normalization fit on E_train time range ONLY      |
|  No cross-graph neighbor sampling (structural isolation)    |
+-------------------------------------------------------------+
""")

for ds_name, stats in all_stats.items():
    total = stats['total_rows']
    train_n = int(total * 0.70)
    val_n = int(total * 0.15)
    test_n = total - train_n - val_n
    print(f"  {ds_name}:")
    print(f"    Train: {train_n:>11,} flows (70%) — earliest")
    print(f"    Val:   {val_n:>11,} flows (15%) — middle")
    print(f"    Test:  {test_n:>11,} flows (15%) — latest")

# -- STEP 3: Data Imbalance Report ---------------------------
print("\n" + "=" * 70)
print("STEP 3: DATA IMBALANCE ANALYSIS")
print("=" * 70)

# Combine counts from training datasets
combined_counts = Counter()
for ds_name in ['NF-CICIDS2018', 'NF-UNSW-NB15']:
    if ds_name in all_class_counts:
        combined_counts.update(all_class_counts[ds_name])

# Remove UNMAPPED
combined_counts.pop('UNMAPPED', None)

total_all = sum(combined_counts.values())
majority_cls = max(combined_counts, key=combined_counts.get)
majority_count = combined_counts[majority_cls]
minority_threshold = np.median(list(combined_counts.values()))

print(f"\n  Combined training data: {total_all:,} flows")
print(f"  Majority class: {majority_cls} ({majority_count:,} flows)")
print(f"  Median class count: {minority_threshold:,.0f}")
print(f"  Imbalance ratio (max/min): {majority_count / min(combined_counts.values()):.1f}:1")
print(f"\n  Unified class distribution (combined train datasets):")
print(f"  {'Class':<25s} {'Count':>10s} {'%':>7s} {'Minority?':>10s}")
print(f"  {'-'*52}")

imbalance_report = []
for cls in UNIFIED_CLASSES:
    count = combined_counts.get(cls, 0)
    pct = count / total_all * 100
    is_minority = count < minority_threshold
    marker = '<- CVAE TARGET' if is_minority else ''
    print(f"  {cls:<25s} {count:>10,} {pct:>6.2f}% {'YES' if is_minority else '':>10s}  {marker}")
    imbalance_report.append({
        'class': cls,
        'count': count,
        'pct': round(pct, 2),
        'minority': is_minority,
        'target_40pct_of_majority': int(majority_count * 0.4),
        'synthetic_needed': max(0, int(majority_count * 0.4) - count),
    })

# CVAE strategy overview
print(f"""
  +-------------------------------------------------------------+
  | IMBALANCE MITIGATION STRATEGY                               |
  |                                                             |
  | 1. CVAE augmentation (Stage E):                             |
  |    - Targets minority classes below median count            |
  |    - Generates synthetic 768-dim embeddings                 |
  |    - Target: each minority reaches ~40% of majority ({int(majority_count*0.4):,})|
  |    - NOT full balance — avoids synthetic-dominated training |
  |                                                             |
  | 2. Focal Loss (gamma=2) with effective-number reweighting:      |
  |    - Down-weights easy examples (majority class)            |
  |    - Up-weights hard examples (minority class)              |
  |    - Effective-number alpha handles extreme imbalance better    |
  |      than plain inverse-frequency weighting                 |
  |                                                             |
  | 3. Per-epoch undersampling in Stage F (Binary):             |
  |    - Resample benign:attack to 2:1 each epoch               |
  |    - NOT static — resampled every epoch for diversity       |
  |                                                             |
  | 4. Per-class threshold calibration in Stage G:              |
  |    - Grid search [0.1, 0.9] per class on val set           |
  |    - Accounts for class-specific precision/recall tradeoffs |
  |    - Classes sharing top features get separate thresholds   |
  |    - Justified by XAI cross-check in Stage J               |
  +-------------------------------------------------------------+
""")

# -- STEP 4: Generate Figures ---------------------------------
print("=" * 70)
print("STEP 4: GENERATING PRELIMINARY FIGURES")
print("=" * 70)

# Set consistent style
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# --- Fig A: Combined Class Distribution (log scale) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Per-dataset class distribution
colors = plt.cm.Set2(np.linspace(0, 1, len(DATASETS)))
for idx, (ds_name, counts) in enumerate(all_class_counts.items()):
    if ds_name not in all_stats: continue
    ax = axes[0]
    classes = [c for c in UNIFIED_CLASSES if counts.get(c, 0) > 0]
    values = [counts.get(c, 0) for c in classes]
    x = np.arange(len(classes))
    bars = ax.bar(x + idx*0.2 - 0.2, values, 0.2, label=ds_name, color=colors[idx],
                  edgecolor='black', linewidth=0.3)
ax.set_xticks(range(len(classes)))
ax.set_xticklabels([c[:12] for c in classes], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Flow Count')
ax.set_yscale('log')
ax.set_title('Class Distribution by Dataset (log scale)')
ax.legend(fontsize=7)
ax.grid(axis='y', alpha=0.3)

# Combined training distribution
ax = axes[1]
sorted_classes = sorted(UNIFIED_CLASSES, key=lambda c: combined_counts.get(c, 0), reverse=True)
values = [combined_counts.get(c, 0) for c in sorted_classes]
bar_colors = ['#4CAF50' if combined_counts.get(c, 0) >= minority_threshold else '#FF5722'
              for c in sorted_classes]
bars = ax.bar(range(len(sorted_classes)), values, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(sorted_classes)))
ax.set_xticklabels([c[:12] for c in sorted_classes], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Flow Count')
ax.set_yscale('log')
ax.set_title('Combined Training Distribution\n(Green≥median, Red=minority -> CVAE augmentation)')
ax.grid(axis='y', alpha=0.3)
# Add count labels
for i, (c, v) in enumerate(zip(sorted_classes, values)):
    if v > 0:
        ax.text(i, v*1.1, f'{v:,}', ha='center', fontsize=6, rotation=90)

fig.suptitle('Figure A: Class Distribution — Pre-Augmentation (All Datasets)',
             fontweight='bold', fontsize=13)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'figA_class_distribution_all.png')
fig.savefig(FIGURES_DIR / 'figA_class_distribution_all.svg')
print("  [OK] figA_class_distribution_all")

# --- Fig B: Feature Correlation Heatmap (sample-based) ---
print("  Computing feature correlation matrix (sampled)...")
corr_samples = []
for ds_name in ['NF-CICIDS2018', 'NF-UNSW-NB15']:
    ds_cfg = DATASETS[ds_name]
    if ds_cfg['file'].exists():
        samples_per_ds = SAMPLE_FOR_CORR // 2
        # Read just enough rows
        reader = pd.read_csv(ds_cfg['file'], chunksize=CHUNK_SIZE, low_memory=False)
        sample_chunks = []
        collected = 0
        for chunk in reader:
            chunk.columns = chunk.columns.str.strip()
            feats_in_chunk = [f for f in KEPT_FEATURES if f in chunk.columns]
            sample_chunks.append(chunk[feats_in_chunk].dropna())
            collected += len(chunk)
            if collected >= samples_per_ds * 3:  # oversample then trim
                break
        if sample_chunks:
            sample_df = pd.concat(sample_chunks, ignore_index=True)
            if len(sample_df) > samples_per_ds:
                sample_df = sample_df.sample(samples_per_ds, random_state=42)
            corr_samples.append(sample_df)

if corr_samples:
    combined_sample = pd.concat(corr_samples, ignore_index=True)
    print(f"  Correlation sample: {len(combined_sample):,} rows × {len(combined_sample.columns)} features")

    # Compute correlation
    corr_matrix = combined_sample.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.1,
                xticklabels=[f[:15] for f in corr_matrix.columns],
                yticklabels=[f[:15] for f in corr_matrix.columns],
                cbar_kws={'shrink': 0.6, 'label': 'Pearson r'},
                ax=ax)
    ax.set_title('Figure B: Feature Correlation Matrix (Pearson, Sampled from Training Data)',
                 fontweight='bold', fontsize=12)
    ax.tick_params(axis='both', labelsize=5)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figB_feature_correlation_matrix.png')
    fig.savefig(FIGURES_DIR / 'figB_feature_correlation_matrix.svg')
    print("  [OK] figB_feature_correlation_matrix")

    # Report highly correlated pairs
    print("\n  Highly correlated feature pairs (|r| > 0.90):")
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.90:
                pair = (corr_matrix.columns[i], corr_matrix.columns[j], r)
                high_corr.append(pair)
                print(f"    {pair[0]:<35s} ↔ {pair[1]:<35s}  r = {pair[2]:.4f}")
    if not high_corr:
        print("    (none found above 0.90)")

# --- Fig C: Feature Value Ranges (Normalized, per-class) ---
fig, ax = plt.subplots(figsize=(14, 6))
# Sample features per class for a few key features
sample_features = [
    'FLOW_DURATION_MILLISECONDS', 'IN_BYTES', 'OUT_BYTES',
    'TCP_FLAGS', 'SRC_TO_DST_IAT_AVG', 'NUM_PKTS_UP_TO_128_BYTES'
]
sample_features = [f for f in sample_features if f in KEPT_FEATURES]

# Read a small sample for visualization
viz_samples = []
for ds_name in ['NF-UNSW-NB15']:  # Use smaller dataset for viz
    ds_cfg = DATASETS[ds_name]
    if ds_cfg['file'].exists():
        df_sample = pd.read_csv(ds_cfg['file'], nrows=50000, low_memory=False)
        df_sample.columns = df_sample.columns.str.strip()
        # Map labels
        label_mapping = label_map[ds_cfg['label_key']]
        df_sample['unified'] = df_sample['Attack'].map(label_mapping)
        df_sample = df_sample.dropna(subset=['unified'])
        viz_samples.append(df_sample[['unified'] + sample_features])

if viz_samples:
    viz_df = pd.concat(viz_samples, ignore_index=True)
    # Group by class, compute mean of each feature
    class_means = viz_df.groupby('unified')[sample_features].mean()

    # Plot as grouped bar chart
    x = np.arange(len(sample_features))
    n_classes = len(class_means)
    width = 0.8 / n_classes

    for i, (cls_name, row) in enumerate(class_means.iterrows()):
        # Normalize each feature to [0,1] for visualization
        normalized = (row - row.min()) / (row.max() - row.min() + 1e-10)
        ax.bar(x + i * width, normalized.values, width, label=cls_name[:12],
               edgecolor='black', linewidth=0.2)

    ax.set_xticks(x + width * n_classes / 2)
    ax.set_xticklabels([f[:20] for f in sample_features], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Normalized Mean Value')
    ax.set_title('Figure C: Feature Value Profiles by Attack Class (UNSW-NB15 sample)')
    ax.legend(fontsize=6, ncol=3, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figC_feature_profiles_by_class.png')
    fig.savefig(FIGURES_DIR / 'figC_feature_profiles_by_class.svg')
    print("  [OK] figC_feature_profiles_by_class")

# --- Fig D: Chronological Split Illustration ---
fig, ax = plt.subplots(figsize=(12, 4))

for ds_name in ['NF-UNSW-NB15']:  # use smaller dataset for visualization
    ds_cfg = DATASETS[ds_name]
    if ds_cfg['file'].exists():
        # Read time column only
        time_chunks = []
        for chunk in pd.read_csv(ds_cfg['file'], chunksize=CHUNK_SIZE,
                                 usecols=['FLOW_START_MILLISECONDS'], low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            time_chunks.append(chunk)
        time_df = pd.concat(time_chunks, ignore_index=True)
        times = time_df['FLOW_START_MILLISECONDS'].values
        times = (times - times.min()) / 3_600_000  # ms -> hours

        n = len(times)
        t_train = int(n * 0.70)
        t_val = int(n * 0.15)

        # Plot flow density over time
        ax.hist(times[:t_train], bins=100, alpha=0.7, color='#4CAF50', label=f'Train (70%)', density=True)
        ax.hist(times[t_train:t_train+t_val], bins=100, alpha=0.7, color='#FF9800', label=f'Val (15%)', density=True)
        ax.hist(times[t_train+t_val:], bins=100, alpha=0.7, color='#F44336', label=f'Test (15%)', density=True)

        ax.axvline(x=train_n/n * times[-1] if n > 0 else 0, color='green', linestyle='--', linewidth=2)
        ax.axvline(x=(train_n+val_n)/n * times[-1] if n > 0 else 0, color='orange', linestyle='--', linewidth=2)

        ax.set_xlabel('Time (hours from capture start)')
        ax.set_ylabel('Flow Density')
        ax.set_title(f'Figure D: Chronological Split — {ds_name}\n(Flows ordered by time, no shuffling across boundaries)',
                     fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'figD_chronological_split.png')
        fig.savefig(FIGURES_DIR / 'figD_chronological_split.svg')
        print("  [OK] figD_chronological_split")
        break

# --- Fig E: Imbalance Severity ---
fig, ax = plt.subplots(figsize=(10, 5))
classes = [c for c in UNIFIED_CLASSES if combined_counts.get(c, 0) > 0]
counts = [combined_counts.get(c, 0) for c in classes]
pcts = [c/total_all*100 for c in counts]

bars = ax.barh(range(len(classes)), counts, color=['#F44336' if p < 1 else '#FF9800' if p < 5 else '#4CAF50' for p in pcts],
               edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(classes, fontsize=8)
ax.set_xscale('log')
ax.set_xlabel('Flow Count (log scale)')
ax.axvline(x=majority_count*0.4, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='40% of majority (CVAE target)')
ax.axvline(x=minority_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Median (minority threshold)')

# Annotate bars
for i, (c, v) in enumerate(zip(classes, counts)):
    ax.text(v*1.05, i, f'{v:,} ({c/total_all*100:.1f}%)', va='center', fontsize=7)

ax.set_title('Figure E: Class Imbalance Severity\nRed <1%, Orange 1-5%, Green >5% of total',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'figE_imbalance_severity.png')
fig.savefig(FIGURES_DIR / 'figE_imbalance_severity.svg')
print("  [OK] figE_imbalance_severity")

# -- STEP 5: Summary Report ----------------------------------
print("\n" + "=" * 70)
print("STEP 5: SUMMARY REPORT")
print("=" * 70)

# Consolidate into a single report
report_lines = [
    "# Dataset Sanity Check — Preliminary Analysis",
    f"\n## Datasets Analyzed",
]

for ds_name, stats in all_stats.items():
    report_lines.append(f"\n### {ds_name} ({stats['role']})")
    report_lines.append(f"- **Total flows:** {stats['total_rows']:,}")
    report_lines.append(f"- **File size:** {stats['file_size_gb']} GB")
    report_lines.append(f"- **Time span:** {stats['time_range_hours']:.1f} hours")
    report_lines.append(f"- **NaN issues:** {len(nan_reports.get(ds_name, {}))} features with NaN")
    report_lines.append(f"- **Inf issues:** {len(inf_reports.get(ds_name, {}))} features with Inf")
    if ds_name in label_mapping_issues:
        report_lines.append(f"- **Label issues:** {label_mapping_issues[ds_name]}")

report_lines.append(f"\n## Split Strategy")
report_lines.append(f"- **Method:** Chronological (time-respecting)")
report_lines.append(f"- **Ratios:** 70% Train / 15% Validation / 15% Test")
report_lines.append(f"- **Key:** FLOW_START_MILLISECONDS (no shuffling across boundaries)")
report_lines.append(f"- **Graphs:** Three separate physical graphs (G_train, G_val, G_test)")
report_lines.append(f"- **Scaler:** Fit on E_train ONLY, applied frozen to val/test")

report_lines.append(f"\n## Data Imbalance")
report_lines.append(f"- **Total training flows (combined):** {total_all:,}")
report_lines.append(f"- **Majority class:** {majority_cls} ({majority_count:,}, {majority_count/total_all*100:.1f}%)")
report_lines.append(f"- **Imbalance ratio (max/min):** {majority_count / min(combined_counts.values()):.1f}:1")
report_lines.append(f"- **Minority classes (below median):** {[c for c in UNIFIED_CLASSES if combined_counts.get(c, 0) < minority_threshold]}")

report_lines.append(f"\n## Mitigation Measures")
report_lines.append(f"1. **CVAE augmentation** — generate synthetic 768-dim embeddings for minority classes")
report_lines.append(f"2. **Focal Loss (gamma=2)** — with effective-number reweighting")
report_lines.append(f"3. **Per-epoch undersampling** — benign:attack = 2:1, resampled every epoch")
report_lines.append(f"4. **Per-class threshold calibration** — separate decision threshold per class")

report_lines.append(f"\n## Generated Figures")
for fname in sorted(os.listdir(FIGURES_DIR)):
    report_lines.append(f"- {fname}")

report_lines.append(f"\n## Updated Dimensions (from feature_manifest.yaml)")
report_lines.append(f"- **Raw features kept:** {RAW_DIM}")
report_lines.append(f"- **Pruned (correlation > 0.95):** {PRUNED}")
report_lines.append(f"- **Time2Vec dims:** 17")
report_lines.append(f"- **Final edge input dim:** {EDGE_DIM}")

with open(OUTPUT_DIR / 'DATASET_SANITY_REPORT.md', 'w') as f:
    f.write('\n'.join(report_lines))

print(report_lines[-30:])  # Print last part
print(f"\n  [OK] Full report: {OUTPUT_DIR / 'DATASET_SANITY_REPORT.md'}")
print(f"  [OK] Figures saved in: {FIGURES_DIR}")
print(f"  Files: {sorted(os.listdir(FIGURES_DIR))}")

print("\n" + "=" * 70)
print("L01 COMPLETE — All datasets analyzed, figures generated.")
print("=" * 70)
