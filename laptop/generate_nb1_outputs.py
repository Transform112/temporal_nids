import os
import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKING_DIR = PROJECT_ROOT / 'working'
OUTPUT_DIR = WORKING_DIR / 'outputs'
FIGURES_DIR = OUTPUT_DIR / 'figures'
TABLES_DIR = OUTPUT_DIR / 'tables'

for d in [FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load config
with open(PROJECT_ROOT / 'label_map.yaml', 'r') as f:
    label_map = yaml.safe_load(f)
with open(PROJECT_ROOT / 'feature_manifest.yaml', 'r') as f:
    feature_manifest = yaml.safe_load(f)

UNIFIED_CLASSES = label_map['unified_classes']
KEPT_FEATURES = feature_manifest['kept_features']
DROPPED_FIELDS = feature_manifest['dropped_fields']

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

# ----------------- TABLE 01 -----------------
tab01_rows = []
for name, counts in class_counts.items():
    total = sum(counts.values())
    benign = counts.get('Benign', 0)
    attack = total - benign

    row_base = {
        'dataset': name,
        'total_flows': total,
        'benign_flows': benign,
        'attack_flows': attack,
        'benign_pct': round(benign / total * 100, 2),
        'attack_pct': round(attack / total * 100, 2),
    }

    for cls in UNIFIED_CLASSES:
        row_base[f'class_{cls}'] = counts.get(cls, 0)

    tab01_rows.append(row_base)

tab01_df = pd.DataFrame(tab01_rows)
tab01_df.to_csv(TABLES_DIR / 'tab01_dataset_statistics.csv', index=False)
tab01_df.to_markdown(TABLES_DIR / 'tab01_dataset_statistics.md', index=False)
print("Saved: tab01_dataset_statistics")

# ----------------- TABLE 02 -----------------
tab02_rows = []
for dataset_key, mapping in label_map.items():
    if dataset_key in ['unified_classes', 'minority_classes']:
        continue
    for raw_label, unified_label in mapping.items():
        tab02_rows.append({
            'source_dataset': dataset_key,
            'raw_label': raw_label,
            'unified_class': unified_label
        })

tab02_df = pd.DataFrame(tab02_rows)
tab02_df.to_csv(TABLES_DIR / 'tab02_taxonomy_mapping.csv', index=False)
tab02_df.to_markdown(TABLES_DIR / 'tab02_taxonomy_mapping.md', index=False)
print("Saved: tab02_taxonomy_mapping")

# ----------------- TABLE 03 -----------------
tab03_rows = []
feature_groups = {
    'Volume': ['IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS'],
    'Protocol/Flags': ['PROTOCOL', 'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS'],
    'Duration': ['FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT'],
    'TTL': ['MIN_TTL', 'MAX_TTL'],
    'Packet Size': ['LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN'],
    'Throughput (per-second)': ['SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES'],
    'Retransmission': ['RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS'],
    'Avg Throughput': ['SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT'],
    'Packet Histogram': ['NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES'],
    'TCP Window': ['TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT'],
    'ICMP': ['ICMP_TYPE', 'ICMP_IPV4_TYPE'],
    'DNS': ['DNS_QUERY_TYPE', 'DNS_TTL_ANSWER'],
    'Inter-Arrival Time': ['SRC_TO_DST_IAT_MIN', 'SRC_TO_DST_IAT_MAX', 'SRC_TO_DST_IAT_AVG', 'SRC_TO_DST_IAT_STDDEV',
                           'DST_TO_SRC_IAT_MIN', 'DST_TO_SRC_IAT_MAX', 'DST_TO_SRC_IAT_AVG', 'DST_TO_SRC_IAT_STDDEV'],
}
time2vec_group = {'Time2Vec (17-dim)': ['Time2Vec linear term', 'Time2Vec periodic terms (×16)']}

for group, features in feature_groups.items():
    for feat in features:
        status = 'kept' if feat in KEPT_FEATURES else 'dropped'
        tab03_rows.append({
            'category': group,
            'feature': feat,
            'status': status,
            'dimension': 'raw (44)'
        })

for group, features in time2vec_group.items():
    for feat in features:
        tab03_rows.append({
            'category': group,
            'feature': feat,
            'status': 'kept',
            'dimension': 'temporal (17)'
        })

for feat in DROPPED_FIELDS:
    tab03_rows.append({
        'category': 'Dropped',
        'feature': feat,
        'status': 'dropped',
        'dimension': '-'
    })

tab03_df = pd.DataFrame(tab03_rows)
tab03_df.to_csv(TABLES_DIR / 'tab03_feature_schema.csv', index=False)
tab03_df.to_markdown(TABLES_DIR / 'tab03_feature_schema.md', index=False)
print("Saved: tab03_feature_schema")

# ----------------- FIG 01 -----------------
fig, ax = plt.subplots(figsize=(14, 4))
stages = ['A: Graph\nConstruction', 'B: Time2Vec', 'C: E-GATv2\nEncoder',
          'D: MAE\nPretrain', 'E: CVAE\nAugment', 'F: Binary\nClassifier',
          'G: Multiclass\nClassifier', 'H: Prototypical\nFew-Shot', 'I: Evaluation\n& XAI']
colors = ['#e8f5e9', '#fff3e0', '#e3f2fd', '#fce4ec', '#f3e5f5',
          '#e0f2f1', '#fff8e1', '#ede7f6', '#efebe9']

for i, (stage, color) in enumerate(zip(stages, colors)):
    rect = plt.Rectangle((i * 1.5, 0), 1.3, 1.5, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(i * 1.5 + 0.65, 0.75, stage, ha='center', va='center', fontsize=7, fontweight='bold')

    if i < len(stages) - 1:
        ax.annotate('', xy=(i * 1.5 + 1.3, 0.75), xytext=(i * 1.5 + 1.5, 0.75),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

ax.set_xlim(-0.2, len(stages) * 1.5)
ax.set_ylim(-0.3, 2.0)
ax.axis('off')
ax.set_title('Figure 1: 9-Stage Graph-NIDS Pipeline Architecture', fontsize=12, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig01_architecture_diagram.png', dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / 'fig01_architecture_diagram.svg', bbox_inches='tight')
print("Saved: fig01_architecture_diagram")

# ----------------- FIG 02 -----------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
split_names = ['G_train', 'G_val', 'G_test']
split_colors = ['#4caf50', '#ff9800', '#f44336']

for i, (name, color) in enumerate(zip(split_names, split_colors)):
    ax = axes[i]
    np.random.seed(42 + i)
    n_nodes = 12
    positions = np.random.rand(n_nodes, 2)
    ax.scatter(positions[:, 0], positions[:, 1], s=80, c=color, edgecolors='black', linewidth=1, zorder=3)
    n_edges = 18
    for _ in range(n_edges):
        u, v = np.random.choice(n_nodes, 2, replace=False)
        ax.plot([positions[u, 0], positions[v, 0]],
                [positions[u, 1], positions[v, 1]],
                'gray', alpha=0.4, linewidth=0.8)

    ax.set_title(f'{name}\n{["70% (earliest flows)", "15% (middle flows)", "15% (latest flows)"][i]}',
                 fontsize=10, fontweight='bold')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')

fig.suptitle('Figure 2: Chronological Split — Three Separate Physical Graphs', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig02_graph_construction_diagram.png', dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / 'fig02_graph_construction_diagram.svg', bbox_inches='tight')
print("Saved: fig02_graph_construction_diagram")
print("Done generating outputs.")
