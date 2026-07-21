"""
REFINED DATA-DERIVED FIGURES — Publication Quality
====================================================
Focus: charts derived from actual dataset analysis (not architectural diagrams).
Zero visual bugs: no overlapping labels, readable fonts, proper spacing.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import numpy as np
import pandas as pd
import yaml, json
from pathlib import Path
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = PROJECT_ROOT / 'laptop' / 'outputs' / 'figures'
TABS_DIR = PROJECT_ROOT / 'laptop' / 'outputs' / 'tables'
for d in [FIGS_DIR, TABS_DIR]: d.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT/'label_map.yaml') as f: label_map = yaml.safe_load(f)
with open(PROJECT_ROOT/'feature_manifest.yaml') as f: fm = yaml.safe_load(f)

UNIFIED = label_map['unified_classes']
N_CLASSES = len(UNIFIED)

# ── Publication style ───────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ── Color palette ───────────────────────────────────────────
C_BENIGN = '#2E7D32'
C_ATTACK = '#C62828'
C_MAJORITY = '#1565C0'
C_MINORITY = '#E65100'
C_TRAIN = '#2E7D32'
C_VAL = '#F57F17'
C_TEST = '#C62828'
PALETTE_11 = ['#1B5E20','#B71C1C','#0D47A1','#E65100','#4A148C',
              '#004D40','#BF360C','#1A237E','#3E2723','#827717','#311B92']

# ── DATA (exact numbers from full_dataset_analysis.py) ─────
# Combined training distribution
COMBINED_TRAIN = {
    'Benign': 13466429, 'DoS/DDoS': 1630554, 'Brute Force': 575194,
    'Exploits': 39015, 'Generic': 9243, 'Reconnaissance': 8950,
    'Backdoor': 3253, 'Web Attack': 2538, 'Shellcode/Worms': 1490,
    'Bot': 0, 'Infiltration': 0,
}
TOTAL_COMBINED = sum(COMBINED_TRAIN.values())
MAJORITY = max(COMBINED_TRAIN.values())
MEDIAN = np.median(list(COMBINED_TRAIN.values()))

# Per-dataset train split distributions (unified classes)
TRAIN_DISTS = {
    'NF-CICIDS2018': {
        'Benign': int(14080870*0.8434), 'DoS/DDoS': int(14080870*0.1156),
        'Brute Force': int(14080870*0.0408), 'Web Attack': int(14080870*0.0002),
        'Infiltration': 0, 'Bot': 0,
        'Reconnaissance': 0, 'Exploits': 0, 'Backdoor': 0, 'Generic': 0, 'Shellcode/Worms': 0,
    },
    'NF-UNSW-NB15': {
        'Benign': int(1655796*0.9606), 'Exploits': int(1655796*0.0236),
        'Generic': int(1655796*0.0056), 'Reconnaissance': int(1655796*0.0054),
        'DoS/DDoS': int(1655796*0.0020), 'Backdoor': int(1655796*0.0020),
        'Shellcode/Worms': int(1655796*0.0009),
        'Brute Force': 0, 'Web Attack': 0, 'Infiltration': 0, 'Bot': 0,
    },
}

# Split attack percentages
SPLIT_ATTACK_PCT = {
    'NF-CICIDS2018': {'train': 15.66, 'val': 4.47, 'test': 8.65},
    'NF-UNSW-NB15':  {'train': 3.94, 'val': 8.85, 'test': 8.77},
    'NF-ToN-IoT':    {'train': 29.17, 'val': 78.81, 'test': 44.96},
    'NF-BoT-IoT':    {'train': 99.58, 'val': 99.98, 'test': 99.95},
}

SPLIT_CLASS_PCTS = {
    'NF-CICIDS2018': {
        'Benign':         [84.34, 95.53, 91.35],
        'DoS/DDoS':       [11.56,  0.00,  0.00],
        'Reconnaissance': [ 0.00,  0.00,  0.00],
        'Exploits':       [ 0.00,  0.00,  0.00],
        'Backdoor':       [ 0.00,  0.00,  0.00],
        'Bot':            [ 0.00,  0.00,  6.88],
        'Brute Force':    [ 4.08,  0.00,  0.00],
        'Web Attack':     [ 0.02,  0.00,  0.00],
        'Infiltration':   [ 0.00,  4.47,  1.76],
        'Generic':        [ 0.00,  0.00,  0.00],
        'Shellcode/Worms':[ 0.00,  0.00,  0.00],
    },
    'NF-UNSW-NB15': {
        'Benign':         [96.06, 91.15, 91.23],
        'DoS/DDoS':       [ 0.20,  0.43,  0.35],
        'Reconnaissance': [ 0.54,  1.27,  1.36],
        'Exploits':       [ 2.36,  5.45,  5.13],
        'Backdoor':       [ 0.20,  0.04,  0.36],
        'Bot':            [ 0.00,  0.00,  0.00],
        'Brute Force':    [ 0.00,  0.00,  0.00],
        'Web Attack':     [ 0.00,  0.00,  0.00],
        'Infiltration':   [ 0.00,  0.00,  0.00],
        'Generic':        [ 0.56,  1.49,  1.44],
        'Shellcode/Worms':[ 0.09,  0.17,  0.12],
    },
}

# ── FIGURE 1: Combined Training Class Distribution ──────────
# (This is the main imbalance chart — replaces fig08)
print("Generating Figure 1: Combined Training Class Distribution...")
fig, ax = plt.subplots(figsize=(14, 7))

# Sort by count descending
sorted_cls = sorted(UNIFIED, key=lambda c: COMBINED_TRAIN.get(c, 0), reverse=True)
counts = [COMBINED_TRAIN.get(c, 0) for c in sorted_cls]
pcts = [c/TOTAL_COMBINED*100 for c in counts]
short_names = [c[:16].replace('/','/\n') for c in sorted_cls]

# Color: green for majority (>=median), red for minority
bar_colors = []
for c in sorted_cls:
    cnt = COMBINED_TRAIN.get(c, 0)
    if cnt == 0: bar_colors.append('#BDBDBD')
    elif cnt < MEDIAN: bar_colors.append('#E53935')
    elif cnt < MAJORITY*0.1: bar_colors.append('#FB8C00')
    else: bar_colors.append('#43A047')

bars = ax.bar(range(len(sorted_cls)), counts, color=bar_colors, edgecolor='white', linewidth=1.2, width=0.7)

# Annotate bars
for i, (c, v, p) in enumerate(zip(sorted_cls, counts, pcts)):
    if v == 0:
        ax.text(i, MAJORITY*0.02, 'ZERO\ntraining\nsamples', ha='center', va='bottom',
                fontsize=7, color='#9E9E9E', fontweight='bold', fontstyle='italic')
    elif v < MAJORITY*0.01:
        ax.text(i, v*1.3, f'{v:,}', ha='center', fontsize=8, fontweight='bold', color='#333')
    else:
        ax.text(i, v*1.05, f'{v:,}\n({p:.2f}%)', ha='center', fontsize=8, fontweight='bold', color='#333')

# Reference lines
ax.axhline(y=MEDIAN, color='#7B1FA2', linestyle='--', linewidth=1.5, alpha=0.6,
           label=f'Median class count = {MEDIAN:,.0f}')
ax.axhline(y=MAJORITY*0.4, color='#1565C0', linestyle=':', linewidth=1.5, alpha=0.6,
           label=f'40% of majority (CVAE target) = {int(MAJORITY*0.4):,}')

ax.set_xticks(range(len(sorted_cls)))
ax.set_xticklabels(short_names, fontsize=9)
ax.set_ylabel('Number of Training Flows', fontsize=12)
ax.set_yscale('log')
ax.set_ylim(0.5, MAJORITY*2)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='#DDD')
ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)

# Legend for colors
from matplotlib.patches import Patch
legend_patches = [
    Patch(facecolor='#43A047', label='Above median (>50K)'),
    Patch(facecolor='#FB8C00', label='Below median (5K-50K)'),
    Patch(facecolor='#E53935', label='Minority <5K (CVAE target)'),
    Patch(facecolor='#BDBDBD', label='Zero training samples'),
]
leg2 = ax.legend(handles=legend_patches, fontsize=8, loc='center right',
                 framealpha=0.9, edgecolor='#DDD', title='Class Category')
ax.add_artist(ax.legend_)  # keep the first legend

ax.set_title('Combined Training Class Distribution (CICIDS2018 + UNSW-NB15)\n'
             f'15.7M total flows | {MAJORITY/min(max(v for v in COMBINED_TRAIN.values() if v>0),1):.0f}:1 imbalance ratio',
             fontweight='bold', fontsize=14, pad=15)

plt.tight_layout()
fig.savefig(FIGS_DIR/'fig_class_distribution.png', dpi=300, facecolor='white', edgecolor='none')
fig.savefig(FIGS_DIR/'fig_class_distribution.svg', facecolor='white', edgecolor='none')
plt.close()
print("  [OK] fig_class_distribution")

# ── FIGURE 2: Split Attack% Comparison ──────────────────────
print("Generating Figure 2: Attack% Across Splits...")
fig, ax = plt.subplots(figsize=(12, 6))

ds_names = list(SPLIT_ATTACK_PCT.keys())
x = np.arange(len(ds_names))
width = 0.25

for i, (split_name, color) in enumerate([('train', C_TRAIN), ('val', C_VAL), ('test', C_TEST)]):
    vals = [SPLIT_ATTACK_PCT[ds][split_name] for ds in ds_names]
    bars = ax.bar(x + i*width, vals, width, label=split_name.capitalize(),
                  color=color, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold', color='#333')

ax.set_xticks(x + width)
ax.set_xticklabels([d.replace('NF-','') for d in ds_names], fontsize=10)
ax.set_ylabel('Attack Flow Percentage', fontsize=12)
ax.set_title('Attack Flow Distribution Across Chronological Splits\n'
             '(Temporal drift is expected and proves chronological splitting is necessary)',
             fontweight='bold', fontsize=13, pad=15)
ax.legend(fontsize=10, framealpha=0.9, edgecolor='#DDD')
ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)
ax.set_ylim(0, 105)

# Annotation about temporal drift
ax.annotate('CICIDS2018: Attacks concentrated\nin earliest period (DDoS campaign)\n'
            'then val is clean, test has Bot/Infiltration',
            xy=(0, 16), fontsize=8, color='#555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.8, edgecolor='#F9A825'))

ax.annotate('UNSW-NB15: Attacks INCREASE\nover time (3.9% -> 8.8%)',
            xy=(1, 9), fontsize=8, color='#555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.8, edgecolor='#F9A825'))

ax.annotate('BoT-IoT: Essentially\nALL attacks (99.6-100%)',
            xy=(3, 99.8), fontsize=8, color='#555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.8, edgecolor='#F9A825'))

plt.tight_layout()
fig.savefig(FIGS_DIR/'fig_split_attack_pct.png', dpi=300, facecolor='white', edgecolor='none')
fig.savefig(FIGS_DIR/'fig_split_attack_pct.svg', facecolor='white', edgecolor='none')
plt.close()
print("  [OK] fig_split_attack_pct")

# ── FIGURE 3: Per-Class Split Heatmap ───────────────────────
print("Generating Figure 3: Per-Class Split Distribution Heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ds_idx, ds_name in enumerate(['NF-CICIDS2018', 'NF-UNSW-NB15']):
    ax = axes[ds_idx]
    data = SPLIT_CLASS_PCTS[ds_name]

    # Build matrix: rows=classes, cols=split
    active_cls = [c for c in UNIFIED if max(data[c]) > 0.001]
    mat = np.array([data[c] for c in active_cls])

    # Short names
    row_labels = [c[:16] for c in active_cls]

    sns.heatmap(mat, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=['Train (70%)', 'Val (15%)', 'Test (15%)'],
                yticklabels=row_labels, ax=ax,
                cbar_kws={'label': '% of split', 'shrink': 0.7},
                vmin=0, vmax=max(10, mat.max()),
                linewidths=0.5, linecolor='white',
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})

    ax.set_title(f'{ds_name}\nClass Percentage Within Each Chronological Split',
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_ylabel('')
    ax.set_xlabel('')

    # Highlight zeros
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] == 0:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                            edgecolor='#90A4AE', linewidth=2, linestyle='--'))

fig.suptitle('Per-Class Distribution Across Chronological Splits\n'
             '(Dashed = zero samples; temporal shift between train and test is expected behavior)',
             fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(FIGS_DIR/'fig_class_split_heatmap.png', dpi=300, facecolor='white', edgecolor='none')
fig.savefig(FIGS_DIR/'fig_class_split_heatmap.svg', facecolor='white', edgecolor='none')
plt.close()
print("  [OK] fig_class_split_heatmap")

# ── FIGURE 4: Dataset Composition Treemap-style ─────────────
print("Generating Figure 4: Dataset Composition...")
fig, axes = plt.subplots(2, 2, figsize=(18, 16))
ds_configs = [
    ('NF-CICIDS2018', TRAIN_DISTS['NF-CICIDS2018'], 'Train/Val/Test\n20.1M flows | 391.6h span'),
    ('NF-UNSW-NB15', TRAIN_DISTS['NF-UNSW-NB15'], 'Train/Val/Test\n2.4M flows | 648.7h span'),
    ('NF-ToN-IoT', None, 'Blind test (in-schema)\n27.5M flows | 144.6h span\nBenign: 61% | Attack: 39%'),
    ('NF-BoT-IoT', None, 'Blind test (in-schema)\n16.9M flows | 843.8h span\nBenign: 0.3% | Attack: 99.7%'),
]

for ax, (ds_name, dist, info) in zip(axes.flat, ds_configs):
    if dist is not None:
        # Donut chart for training datasets
        nonzero = {k: v for k, v in dist.items() if v > 0}
        labels = [f'{k}\n({v:,})' for k, v in nonzero.items()]
        vals = list(nonzero.values())
        colors_pie = [C_BENIGN if 'Benign' in l else c for l, c in
                      zip(labels, PALETTE_11[:len(labels)])]

        wedges, texts = ax.pie(vals, labels=None, colors=colors_pie,
                                startangle=90, pctdistance=0.85,
                                wedgeprops=dict(width=0.35, edgecolor='white', linewidth=1.5))

        # Center text
        total = sum(vals)
        ax.text(0, 0, f'{total:,}\nflows', ha='center', va='center',
                fontsize=12, fontweight='bold')

        # Legend outside
        ax.legend(wedges, labels, title='Classes', fontsize=7,
                  loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
                  title_fontsize=8)

    else:
        # Simple bar for blind test datasets
        ax.text(0.5, 0.5, info, ha='center', va='center', fontsize=10,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                          edgecolor='#BDBDBD', linewidth=1))
        ax.axis('off')

    ax.set_title(ds_name, fontsize=12, fontweight='bold', pad=10)

fig.suptitle('Dataset Composition — Training Split Distribution',
             fontweight='bold', fontsize=15, y=1.01)
plt.tight_layout()
fig.savefig(FIGS_DIR/'fig_dataset_composition.png', dpi=300, facecolor='white', edgecolor='none')
fig.savefig(FIGS_DIR/'fig_dataset_composition.svg', facecolor='white', edgecolor='none')
plt.close()
print("  [OK] fig_dataset_composition")

# ── FIGURE 5: CVAE Augmentation Strategy ────────────────────
print("Generating Figure 5: CVAE Augmentation Plan...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

minority_cls = [c for c in UNIFIED if COMBINED_TRAIN.get(c, 0) < MEDIAN]
minority_cls_sorted = sorted(minority_cls, key=lambda c: COMBINED_TRAIN.get(c, 0))

# Left: Before augmentation
pre_counts = [COMBINED_TRAIN.get(c, 0) for c in minority_cls_sorted]
axes[0].barh(range(len(minority_cls_sorted)), pre_counts,
             color=['#E53935' if v==0 else '#FF7043' for v in pre_counts],
             edgecolor='white', linewidth=1.2, height=0.6)
for i, (c, v) in enumerate(zip(minority_cls_sorted, pre_counts)):
    label = 'ZERO' if v == 0 else f'{v:,}'
    axes[0].text(max(v, MAJORITY*0.005), i, f'  {label}', va='center', fontsize=9, fontweight='bold')
axes[0].set_yticks(range(len(minority_cls_sorted)))
axes[0].set_yticklabels([c[:18] for c in minority_cls_sorted], fontsize=9)
axes[0].set_xscale('log'); axes[0].set_xlim(0.5, MAJORITY*0.6)
axes[0].set_xlabel('Training Samples (log scale)', fontsize=11)
axes[0].set_title('Before CVAE Augmentation', fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.2, linestyle='--')

# Right: After augmentation (simulated)
target = int(MAJORITY * 0.4)
post_counts = [min(COMBINED_TRAIN.get(c, 0) + max(0, target - COMBINED_TRAIN.get(c, 0)), target)
               for c in minority_cls_sorted]
synthetic = [max(0, target - COMBINED_TRAIN.get(c, 0)) for c in minority_cls_sorted]

for i, (c, real, synth) in enumerate(zip(minority_cls_sorted, pre_counts, synthetic)):
    # Real portion
    axes[1].barh(i, real, color='#43A047', edgecolor='white', linewidth=1.2, height=0.6)
    # Synthetic portion
    if synth > 0:
        axes[1].barh(i, synth, left=real, color='#90CAF9', edgecolor='white',
                     linewidth=1.2, height=0.6, alpha=0.8)
    total = real + synth
    axes[1].text(total + MAJORITY*0.005, i, f'{total:,} (real: {real:,})', va='center', fontsize=8)

# Target line
axes[1].axvline(x=target, color='#1565C0', linestyle=':', linewidth=2, alpha=0.7,
                label=f'Target: {target:,} (40% of majority)')
axes[1].set_yticks(range(len(minority_cls_sorted)))
axes[1].set_yticklabels([c[:18] for c in minority_cls_sorted], fontsize=9)
axes[1].set_xlabel('Total Samples (real + synthetic)', fontsize=11)
axes[1].set_title('After CVAE Augmentation', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9, loc='lower right', framealpha=0.9)
axes[1].grid(axis='x', alpha=0.2, linestyle='--')

# Legend patches
from matplotlib.patches import Patch
legend2 = [Patch(facecolor='#43A047', label='Real samples'),
           Patch(facecolor='#90CAF9', label='CVAE synthetic')]
axes[1].legend(handles=legend2 + [plt.Line2D([0],[0],color='#1565C0',linestyle=':',lw=2)],
               labels=['Real samples','CVAE synthetic',f'Target = {target:,}'],
               fontsize=9, loc='lower right', framealpha=0.9)

fig.suptitle('CVAE Minority-Class Augmentation Strategy\n'
             '(Synthetic 768-dim embeddings generated for classes below median count)',
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(FIGS_DIR/'fig_cvae_augmentation_plan.png', dpi=300, facecolor='white', edgecolor='none')
fig.savefig(FIGS_DIR/'fig_cvae_augmentation_plan.svg', facecolor='white', edgecolor='none')
plt.close()
print("  [OK] fig_cvae_augmentation_plan")

# ── FIGURE 6: Dataset Class Coverage Matrix ─────────────────
print("Generating Figure 6: Class Coverage Matrix...")
# Which classes exist in which datasets
coverage = np.zeros((N_CLASSES, 4))
ds_order = ['NF-CICIDS2018', 'NF-UNSW-NB15', 'NF-ToN-IoT', 'NF-BoT-IoT']

# Approximate coverage (exists > 0.01% in training split)
for j, ds_name in enumerate(ds_order):
    if ds_name in TRAIN_DISTS:
        for i, cls in enumerate(UNIFIED):
            coverage[i, j] = 1 if TRAIN_DISTS[ds_name].get(cls, 0) > 0 else 0

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(coverage, annot=True, fmt='.0f', cmap=['#ECEFF1', '#43A047'],
            xticklabels=ds_order, yticklabels=[c[:16] for c in UNIFIED],
            ax=ax, cbar=False,
            linewidths=1.5, linecolor='white',
            annot_kws={'fontsize': 16, 'fontweight': 'bold'})

# Overlay text: "YES"/"NO" instead of 1/0
for i in range(N_CLASSES):
    for j in range(4):
        val = coverage[i, j]
        color = '#1B5E20' if val else '#B71C1C'
        text = 'YES' if val else 'NO'
        ax.text(j+0.5, i+0.5, text, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)

ax.set_title('Attack Class Coverage by Dataset\n(Green = class present in training data)',
             fontweight='bold', fontsize=13, pad=15)
ax.set_xlabel(''); ax.set_ylabel('')
plt.tight_layout()
fig.savefig(FIGS_DIR/'fig_class_coverage_matrix.png', dpi=300, facecolor='white', edgecolor='none')
fig.savefig(FIGS_DIR/'fig_class_coverage_matrix.svg', facecolor='white', edgecolor='none')
plt.close()
print("  [OK] fig_class_coverage_matrix")

# ── FIGURE 7: Temporal Attack Trend (UNSW-NB15) ─────────────
print("Generating Figure 7: Temporal Attack Trend...")
# Read UNSW-NB15 time data (smallest dataset, fits in memory)
import pandas as pd
fpath = PROJECT_ROOT / 'dataset' / 'NF-UNSW-NB15-v3.csv'
df_unsw = pd.read_csv(fpath, usecols=['FLOW_START_MILLISECONDS', 'Attack'])
df_unsw.columns = df_unsw.columns.str.strip()

times = df_unsw['FLOW_START_MILLISECONDS'].values
is_attack = (df_unsw['Attack'].values != 'Benign')

# Sort by time
sort_idx = np.argsort(times)
times_sorted = times[sort_idx]
attack_sorted = is_attack[sort_idx]

# 50 equal bins
n_bins = 50
bin_edges = np.linspace(times_sorted.min(), times_sorted.max(), n_bins+1)
bin_attack_pct = []
bin_hours = []

for i in range(n_bins):
    mask = (times_sorted >= bin_edges[i]) & (times_sorted < bin_edges[i+1])
    if mask.sum() > 0:
        bin_attack_pct.append(attack_sorted[mask].mean() * 100)
        bin_hours.append((bin_edges[i] - times_sorted.min()) / 3.6e6)

fig, ax = plt.subplots(figsize=(14, 5))

# Plot attack% over time
ax.fill_between(bin_hours, bin_attack_pct, alpha=0.3, color='#EF5350')
ax.plot(bin_hours, bin_attack_pct, 'o-', color='#C62828', linewidth=1.5, markersize=3,
        label='Attack % per time bin')

# Split boundaries
train_h = (times_sorted[int(len(times_sorted)*0.70)] - times_sorted.min()) / 3.6e6
val_h = (times_sorted[int(len(times_sorted)*0.85)] - times_sorted.min()) / 3.6e6

ax.axvline(x=train_h, color='#2E7D32', linestyle='--', linewidth=2.5, alpha=0.8,
           label=f'Train/Val boundary ({train_h:.0f}h)')
ax.axvline(x=val_h, color='#F57F17', linestyle='--', linewidth=2.5, alpha=0.8,
           label=f'Val/Test boundary ({val_h:.0f}h)')

# Shade split regions
ymax = max(bin_attack_pct) * 1.15
ax.fill_between([0, train_h], 0, ymax, alpha=0.06, color='#2E7D32')
ax.fill_between([train_h, val_h], 0, ymax, alpha=0.06, color='#F57F17')
ax.fill_between([val_h, (times_sorted.max()-times_sorted.min())/3.6e6], 0, ymax, alpha=0.06, color='#C62828')

# Region labels
ax.text(train_h/2, ymax*0.95, 'TRAIN\n(70%)', ha='center', fontsize=10, fontweight='bold', color='#2E7D32')
ax.text((train_h+val_h)/2, ymax*0.95, 'VAL\n(15%)', ha='center', fontsize=10, fontweight='bold', color='#F57F17')
ax.text((val_h+(times_sorted.max()-times_sorted.min())/3.6e6)/2, ymax*0.95, 'TEST\n(15%)', ha='center',
        fontsize=10, fontweight='bold', color='#C62828')

ax.set_xlabel('Time Since Capture Start (hours)', fontsize=12)
ax.set_ylabel('Attack Flow %', fontsize=12)
ax.set_title('UNSW-NB15: Temporal Evolution of Attack Traffic\n'
             '(Attack proportion varies significantly over time — chronological split is essential)',
             fontweight='bold', fontsize=13, pad=15)
ax.legend(fontsize=9, framealpha=0.9, loc='upper left')
ax.grid(alpha=0.2, linestyle='--')
ax.set_ylim(0, ymax)

plt.tight_layout()
fig.savefig(FIGS_DIR/'fig_temporal_attack_trend.png', dpi=300, facecolor='white', edgecolor='none')
fig.savefig(FIGS_DIR/'fig_temporal_attack_trend.svg', facecolor='white', edgecolor='none')
plt.close()
print("  [OK] fig_temporal_attack_trend")

# ── SUMMARY ─────────────────────────────────────────────────
print("\n" + "="*60)
print("REFINED FIGURES COMPLETE")
print("="*60)
import os
for f in sorted(os.listdir(FIGS_DIR)):
    if any(f.startswith(p) for p in ['fig_class_distribution','fig_split_attack',
        'fig_class_split','fig_dataset_composition','fig_cvae_augmentation','fig_class_coverage',
        'fig_temporal_attack']):
        size_kb = os.path.getsize(FIGS_DIR/f)/1024
        print(f"  {f} ({size_kb:.0f} KB)")
print(f"\nAll saved to: {FIGS_DIR}")
