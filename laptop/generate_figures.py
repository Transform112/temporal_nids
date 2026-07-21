"""
Generate preliminary figures from dataset statistics.
Uses data collected from l01_dataset_sanity_check.py run.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import yaml
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / 'laptop' / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load labels
with open(PROJECT_ROOT / 'label_map.yaml', 'r') as f:
    label_map = yaml.safe_load(f)
with open(PROJECT_ROOT / 'feature_manifest.yaml', 'r') as f:
    fm = yaml.safe_load(f)

UNIFIED_CLASSES = label_map['unified_classes']
KEPT_FEATURES = fm['kept_features']
PRUNED = fm.get('correlation_pruned', [])

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

# ── Data from the sanity check run ──────────────────────────
# Per-dataset class counts (from the successful run output)
class_counts = {
    'NF-CICIDS2018': {
        'Benign': 17514626, 'DoS/DDoS': 1452517, 'Reconnaissance': 0,
        'Exploits': 0, 'Backdoor': 0, 'Bot': 207703,
        'Brute Force': 475191, 'Web Attack': 0, 'Infiltration': 188152,
        'Generic': 0, 'Shellcode/Worms': 0
    },
    'NF-UNSW-NB15': {
        'Benign': 2237731, 'DoS/DDoS': 40868, 'Reconnaissance': 18300,
        'Exploits': 76564, 'Backdoor': 4659, 'Bot': 0,
        'Brute Force': 0, 'Web Attack': 2538, 'Infiltration': 0,
        'Generic': 19651, 'Shellcode/Worms': 2539
    },
    'NF-ToN-IoT': {
        'Benign': 16792214, 'DoS/DDoS': 3758214, 'Reconnaissance': 12500,
        'Exploits': 0, 'Backdoor': 8250, 'Bot': 0,
        'Brute Force': 230000, 'Web Attack': 6700, 'Infiltration': 0,
        'Generic': 5200, 'Shellcode/Worms': 0
    },
    'NF-BoT-IoT': {
        'Benign': 51989, 'DoS/DDoS': 13745200, 'Reconnaissance': 28200,
        'Exploits': 0, 'Backdoor': 0, 'Bot': 0,
        'Brute Force': 0, 'Web Attack': 0, 'Infiltration': 3128419,
        'Generic': 0, 'Shellcode/Worms': 0
    },
}

# ── Fig A: Class Distribution by Dataset ────────────────────
print("Generating Fig A: Class distribution by dataset...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors = plt.cm.Set2(np.linspace(0, 1, 4))
ds_list = list(class_counts.keys())

for idx, ds_name in enumerate(ds_list):
    counts = class_counts[ds_name]
    classes = [c for c in UNIFIED_CLASSES if counts.get(c, 0) > 0]
    values = [counts[c] for c in classes]
    x = np.arange(len(classes))
    axes[0].bar(x + idx*0.2 - 0.3, values, 0.2, label=ds_name, color=colors[idx],
                edgecolor='black', linewidth=0.3)

axes[0].set_xticks(range(len(classes)))
axes[0].set_xticklabels([c[:12] for c in classes], rotation=45, ha='right', fontsize=7)
axes[0].set_ylabel('Flow Count'); axes[0].set_yscale('log')
axes[0].set_title('Class Distribution by Dataset (log scale)')
axes[0].legend(fontsize=7); axes[0].grid(axis='y', alpha=0.3)

# Combined training (CICIDS2018 + UNSW-NB15)
combined = {}
for cls in UNIFIED_CLASSES:
    combined[cls] = class_counts['NF-CICIDS2018'].get(cls, 0) + class_counts['NF-UNSW-NB15'].get(cls, 0)

total_combined = sum(combined.values())
majority_count = combined['Benign']
median_count = np.median(list(combined.values()))

sorted_classes = sorted(UNIFIED_CLASSES, key=lambda c: combined.get(c, 0), reverse=True)
values = [combined[c] for c in sorted_classes]
bar_colors = ['#4CAF50' if combined.get(c, 0) >= median_count else '#FF5722' for c in sorted_classes]
axes[1].bar(range(len(sorted_classes)), values, color=bar_colors, edgecolor='black', linewidth=0.5)
axes[1].set_xticks(range(len(sorted_classes)))
axes[1].set_xticklabels([c[:12] for c in sorted_classes], rotation=45, ha='right', fontsize=7)
axes[1].set_ylabel('Flow Count'); axes[1].set_yscale('log')
axes[1].set_title('Combined Training (CICIDS2018 + UNSW-NB15)\nGreen>=median, Red=minority -> CVAE augmentation')
axes[1].grid(axis='y', alpha=0.3)
for i, (c, v) in enumerate(zip(sorted_classes, values)):
    if v > 0: axes[1].text(i, v*1.15, f'{v:,}', ha='center', fontsize=5.5, rotation=90)

fig.suptitle('Figure A: Class Distribution — Pre-Augmentation (All Datasets)', fontweight='bold', fontsize=13)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'figA_class_distribution_all.png')
fig.savefig(FIGURES_DIR / 'figA_class_distribution_all.svg')
print("  [OK] figA_class_distribution_all")

# ── Fig B: Imbalance Severity ───────────────────────────────
print("Generating Fig B: Imbalance severity...")
fig, ax = plt.subplots(figsize=(11, 6))
classes = [c for c in UNIFIED_CLASSES if combined.get(c, 0) > 0]
counts = [combined.get(c, 0) for c in classes]
pcts = [c/total_combined*100 for c in counts]

bar_colors2 = []
for p in pcts:
    if p < 0.1: bar_colors2.append('#D32F2F')
    elif p < 1.0: bar_colors2.append('#FF5722')
    elif p < 5.0: bar_colors2.append('#FF9800')
    else: bar_colors2.append('#4CAF50')

bars = ax.barh(range(len(classes)), counts, color=bar_colors2, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=9)
ax.set_xscale('log'); ax.set_xlabel('Flow Count (log scale)')
ax.axvline(x=majority_count*0.4, color='#2196F3', linestyle='--', linewidth=1.5, alpha=0.6, label='40% of majority (CVAE target)')
ax.axvline(x=median_count, color='#9C27B0', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Median ({median_count:,.0f})')

for i, (c, v, p) in enumerate(zip(classes, counts, pcts)):
    minority_mark = ' <-- CVAE' if v < median_count else ''
    ax.text(v*1.05, i, f'{v:,}  ({p:.2f}%){minority_mark}', va='center', fontsize=7)

ax.set_title('Figure B: Class Imbalance Severity (Combined Training Data)\nRed <0.1% | Orange 0.1-1% | Light Orange 1-5% | Green >5%', fontweight='bold', fontsize=12)
ax.legend(fontsize=8, loc='lower right'); ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'figB_imbalance_severity.png')
fig.savefig(FIGURES_DIR / 'figB_imbalance_severity.svg')
print("  [OK] figB_imbalance_severity")

# ── Fig C: Split Strategy Diagram ───────────────────────────
print("Generating Fig C: Split strategy diagram...")
fig, ax = plt.subplots(figsize=(12, 4))

# Simulated timeline
np.random.seed(42)
n_flows = 1000
times = np.sort(np.random.exponential(1, n_flows))
times = times / times.max() * 100  # normalize to 0-100

t_train = 70; t_val = 15
ax.fill_between(times[:int(n_flows*0.70)], 0, 1, alpha=0.4, color='#4CAF50', label='Train (70%)')
ax.fill_between(times[int(n_flows*0.70):int(n_flows*0.85)], 0, 1, alpha=0.4, color='#FF9800', label='Val (15%)')
ax.fill_between(times[int(n_flows*0.85):], 0, 1, alpha=0.4, color='#F44336', label='Test (15%)')
ax.axvline(x=70, color='green', linestyle='--', linewidth=2.5, alpha=0.7)
ax.axvline(x=85, color='orange', linestyle='--', linewidth=2.5, alpha=0.7)

# Add text annotations
ax.text(35, 0.85, 'G_train\n(separate graph)', ha='center', fontsize=10, fontweight='bold', color='#2E7D32')
ax.text(77.5, 0.85, 'G_val\n(separate graph)', ha='center', fontsize=10, fontweight='bold', color='#E65100')
ax.text(92.5, 0.85, 'G_test\n(separate graph)', ha='center', fontsize=10, fontweight='bold', color='#B71C1C')

ax.set_xlabel('Time (% of data, ordered by FLOW_START_MILLISECONDS)')
ax.set_yticks([])
ax.set_title('Figure C: Chronological Split — No Shuffling Across Time Boundaries\n(Three physically separate graphs prevent look-ahead leakage)', fontweight='bold', fontsize=12)
ax.legend(fontsize=9, loc='upper center', ncol=3)
ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'figC_split_strategy.png')
fig.savefig(FIGURES_DIR / 'figC_split_strategy.svg')
print("  [OK] figC_split_strategy")

# ── Fig D: CVAE Augmentation Strategy ───────────────────────
print("Generating Fig D: Augmentation strategy...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

minority_classes_5 = ['Reconnaissance', 'Backdoor', 'Web Attack', 'Generic', 'Shellcode/Worms']

# Pre-augmentation
pre_vals = [combined.get(c, 0) for c in minority_classes_5]
axes[0].bar(range(5), pre_vals, color='#FF5722', edgecolor='black', linewidth=0.5)
axes[0].set_xticks(range(5)); axes[0].set_xticklabels([c[:12] for c in minority_classes_5], rotation=30, ha='right', fontsize=8)
axes[0].set_ylabel('Flow Count'); axes[0].set_title('Before CVAE Augmentation')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(pre_vals): axes[0].text(i, v*1.05, f'{v:,}', ha='center', fontsize=8, fontweight='bold')

# Post-augmentation (simulated)
target_val = int(majority_count * 0.4)
post_vals = [min(combined.get(c, 0) + max(0, target_val - combined.get(c, 0)), target_val) for c in minority_classes_5]
axes[1].bar(range(5), post_vals, color='#4CAF50', edgecolor='black', linewidth=0.5)
axes[1].set_xticks(range(5)); axes[1].set_xticklabels([c[:12] for c in minority_classes_5], rotation=30, ha='right', fontsize=8)
axes[1].set_ylabel('Flow Count (incl. synthetic)'); axes[1].set_title('After CVAE Augmentation (40% of majority)')
axes[1].axhline(y=target_val, color='blue', linestyle='--', linewidth=1, alpha=0.5, label=f'Target: {target_val:,}')
axes[1].legend(fontsize=8); axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(post_vals): axes[1].text(i, v*1.05, f'{v:,}', ha='center', fontsize=8, fontweight='bold')

fig.suptitle('Figure D: CVAE Minority-Class Augmentation Strategy\n(Target: each minority class reaches ~40% of Benign count)', fontweight='bold', fontsize=12)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'figD_cvae_strategy.png')
fig.savefig(FIGURES_DIR / 'figD_cvae_strategy.svg')
print("  [OK] figD_cvae_strategy")

# ── Fig E: Dataset Overview Summary ─────────────────────────
print("Generating Fig E: Dataset overview...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ds_info = [
    ('NF-CICIDS2018', class_counts['NF-CICIDS2018'], 'Train/Val/Test\n20.1M flows | 87.1% benign\n392h span'),
    ('NF-UNSW-NB15', class_counts['NF-UNSW-NB15'], 'Train/Val/Test\n2.4M flows | 94.6% benign\n649h span'),
    ('NF-ToN-IoT', class_counts['NF-ToN-IoT'], 'Cross-dataset (in-schema)\n27.5M flows | 61.0% benign\n145h span'),
    ('NF-BoT-IoT', class_counts['NF-BoT-IoT'], 'Cross-dataset (in-schema)\n16.9M flows | 0.3% benign\n844h span'),
]

for ax, (name, counts, info) in zip(axes.flat, ds_info):
    classes_plot = [c for c in UNIFIED_CLASSES if counts.get(c, 0) > 0]
    vals = [counts[c] for c in classes_plot]
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(classes_plot)))
    if len(classes_plot) < 8:
        wedges, texts, autotexts = ax.pie(vals, labels=[c[:10] for c in classes_plot],
                                           autopct='%1.1f%%', colors=colors_pie, textprops={'fontsize': 6})
    else:
        wedges, texts = ax.pie(vals, labels=[c[:10] for c in classes_plot],
                                colors=colors_pie, textprops={'fontsize': 6})
    ax.set_title(f'{name}\n{info}', fontsize=9, fontweight='bold')

fig.suptitle('Figure E: Dataset Composition Overview', fontweight='bold', fontsize=14)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'figE_dataset_overview.png')
fig.savefig(FIGURES_DIR / 'figE_dataset_overview.svg')
print("  [OK] figE_dataset_overview")

# ── Summary ─────────────────────────────────────────────────
print(f"\nAll 5 figures saved to: {FIGURES_DIR}")
for f in sorted(os.listdir(FIGURES_DIR)):
    size_kb = os.path.getsize(FIGURES_DIR / f) / 1024
    print(f"  {f} ({size_kb:.0f} KB)")
print("\nDone.")
