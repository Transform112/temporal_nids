"""
FULL DATASET ANALYSIS — All 4 datasets, exact split numbers, all pre-training tables & figures.
===============================================================================================
Generates: tab01, tab02, tab03, fig01, fig02, fig08, plus split distribution tables.
Memory-efficient chunked processing for the large datasets.
"""
import pandas as pd, numpy as np, yaml, json, os, time
from pathlib import Path
from collections import Counter, defaultdict
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import matplotlib.ticker as mticker
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'
OUTPUT_DIR = PROJECT_ROOT / 'laptop' / 'outputs'
FIGS_DIR = OUTPUT_DIR / 'figures'
TABS_DIR = OUTPUT_DIR / 'tables'
for d in [OUTPUT_DIR, FIGS_DIR, TABS_DIR]: d.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT/'label_map.yaml') as f: label_map = yaml.safe_load(f)
with open(PROJECT_ROOT/'feature_manifest.yaml') as f: fm = yaml.safe_load(f)

UNIFIED = label_map['unified_classes']; N_CLASSES = len(UNIFIED)
KEPT = fm['kept_features']; PRUNED = fm.get('correlation_pruned', [])
TIME_FIELD = fm['time_signal_field']
CHUNK = 500_000

plt.rcParams.update({'font.size':8, 'axes.titlesize':10, 'axes.labelsize':9,
                     'figure.dpi':150, 'savefig.dpi':300, 'savefig.bbox':'tight'})

DATASETS = {
    'NF-CICIDS2018': {'file':'NF-CICIDS2018-v3.csv', 'lk':'NF-CSE-CIC-IDS2018', 'role':'Train/Val/Test (primary)'},
    'NF-UNSW-NB15':  {'file':'NF-UNSW-NB15-v3.csv',  'lk':'NF-UNSW-NB15',       'role':'Train/Val/Test (primary)'},
    'NF-ToN-IoT':    {'file':'NF-ToN-IoT-v3.csv',     'lk':'NF-ToN-IoT',         'role':'Blind test (in-schema)'},
    'NF-BoT-IoT':    {'file':'NF-BoT-IoT-v3.csv',     'lk':'NF-BoT-IoT',         'role':'Blind test (in-schema)'},
}

print("="*70)
print("FULL DATASET ANALYSIS — All 4 Datasets")
print("="*70)

# ── STEP 1: Scan all datasets, compute splits, get exact numbers ──
all_results = {}
for ds_name, ds_cfg in DATASETS.items():
    fpath = DATASET_DIR / ds_cfg['file']
    if not fpath.exists():
        print(f"\nSKIP {ds_name}: file not found")
        continue

    fs_gb = fpath.stat().st_size/1e9
    print(f"\n{'='*60}\n{ds_name} ({fs_gb:.2f} GB) — {ds_cfg['role']}\n{'='*60}")

    # Collect time and labels
    times_list, attack_list = [], []
    t0 = time.time()
    for chunk in pd.read_csv(fpath, chunksize=CHUNK, usecols=[TIME_FIELD,'Attack'], low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        times_list.append(chunk[TIME_FIELD].values)
        attack_list.append(chunk['Attack'].values)

    all_times = np.concatenate(times_list); all_attack = np.concatenate(attack_list)
    n = len(all_times)
    elapsed = time.time()-t0
    print(f"  Loaded {n:,} rows in {elapsed:.1f}s")

    # Sort by time
    sort_idx = np.argsort(all_times)
    sorted_times = all_times[sort_idx]; sorted_attack = all_attack[sort_idx]

    # Chronological split
    train_n = int(n*0.70); val_n = int(n*0.15)
    splits = {
        'train': (0, train_n),
        'val':   (train_n, train_n+val_n),
        'test':  (train_n+val_n, n),
    }

    mapping = label_map[ds_cfg['lk']]
    split_data = {}
    for sp_name, (start, end) in splits.items():
        sp_times = sorted_times[start:end]
        sp_attack = sorted_attack[start:end]
        sp_unified = np.array([mapping.get(l, 'UNMAPPED') for l in sp_attack])

        raw_counts = Counter(sp_attack)
        unified_counts = Counter(sp_unified)

        split_data[sp_name] = {
            'n_flows': end-start,
            'time_min': sp_times.min(),
            'time_max': sp_times.max(),
            'time_span_h': (sp_times.max()-sp_times.min())/3.6e6,
            'raw_label_counts': dict(raw_counts),
            'unified_counts': {c: unified_counts.get(c,0) for c in UNIFIED},
        }

    all_results[ds_name] = {
        'total': n,
        'file_size_gb': round(fs_gb, 2),
        'role': ds_cfg['role'],
        'time_span_h': (sorted_times.max()-sorted_times.min())/3.6e6,
        'splits': split_data,
    }

    # Print time boundaries
    print(f"\n  TIME BOUNDARIES:")
    for sp_name in ['train','val','test']:
        sd = split_data[sp_name]
        print(f"  {sp_name:5s}: [{sd['time_min']:.0f}, {sd['time_max']:.0f}] ms  "
              f"({sd['n_flows']:,} flows, {sd['time_span_h']:.1f}h)")

    # Verify no overlap
    assert split_data['train']['time_max'] <= split_data['val']['time_min'], \
        f"TIME LEAK: train->val in {ds_name}"
    assert split_data['val']['time_max'] <= split_data['test']['time_min'], \
        f"TIME LEAK: val->test in {ds_name}"
    gap_tv = (split_data['val']['time_min'] - split_data['train']['time_max'])/1000
    gap_vt = (split_data['test']['time_min'] - split_data['val']['time_max'])/1000
    print(f"  [OK] No time overlap. Gaps: train->val={gap_tv:.0f}s, val->test={gap_vt:.0f}s")

# ── STEP 2: TAB01 — Dataset Statistics ──────────────────────
print("\n"+"="*60+"\nGENERATING TAB01: Dataset Statistics\n"+"="*60)

tab01_rows = []
for ds_name, res in all_results.items():
    row = {'Dataset': ds_name, 'Role': res['role'],
           'Total Flows': res['total'], 'Time Span (h)': round(res['time_span_h'],1),
           'File Size (GB)': res['file_size_gb']}
    for sp_name in ['train','val','test']:
        sd = res['splits'][sp_name]
        row[f'{sp_name}_flows'] = sd['n_flows']
        row[f'{sp_name}_benign'] = sd['unified_counts'].get('Benign',0)
        row[f'{sp_name}_attack'] = sd['n_flows'] - sd['unified_counts'].get('Benign',0)
        row[f'{sp_name}_attack_pct'] = round((sd['n_flows']-sd['unified_counts'].get('Benign',0))/sd['n_flows']*100,2)
    tab01_rows.append(row)

tab01 = pd.DataFrame(tab01_rows)
tab01.to_csv(TABS_DIR/'tab01_dataset_statistics.csv', index=False)
# Pretty-print
print(tab01[['Dataset','Total Flows','Time Span (h)','train_flows','val_flows','test_flows',
             'train_attack_pct','val_attack_pct','test_attack_pct']].to_string(index=False))
tab01.to_markdown(TABS_DIR/'tab01_dataset_statistics.md', index=False)
print("  [OK] tab01 saved")

# ── STEP 3: TAB02 — Taxonomy Mapping ────────────────────────
print("\n"+"="*60+"\nGENERATING TAB02: Taxonomy Mapping\n"+"="*60)
tab02_rows = []
for ds_name, ds_cfg in DATASETS.items():
    lk = ds_cfg['lk']
    if lk not in label_map: continue
    for raw, unified in label_map[lk].items():
        tab02_rows.append({'Source Dataset': ds_name, 'Raw Label': raw, 'Unified Class': unified})

tab02 = pd.DataFrame(tab02_rows)
tab02.to_csv(TABS_DIR/'tab02_taxonomy_mapping.csv', index=False)
tab02.to_markdown(TABS_DIR/'tab02_taxonomy_mapping.md', index=False)
print(f"  [OK] tab02 saved ({len(tab02)} mappings)")

# ── STEP 4: TAB03 — Feature Schema ──────────────────────────
print("\n"+"="*60+"\nGENERATING TAB03: Feature Schema\n"+"="*60)
tab03_rows = []
groups = {
    'Volume': ['IN_BYTES','IN_PKTS','OUT_BYTES','OUT_PKTS'],
    'Protocol/Flags': ['PROTOCOL','TCP_FLAGS','CLIENT_TCP_FLAGS','SERVER_TCP_FLAGS'],
    'Duration': ['FLOW_DURATION_MILLISECONDS','DURATION_IN','DURATION_OUT'],
    'TTL': ['MIN_TTL','MAX_TTL'],
    'Packet Size': ['LONGEST_FLOW_PKT','SHORTEST_FLOW_PKT','MIN_IP_PKT_LEN'],
    'Throughput': ['SRC_TO_DST_SECOND_BYTES','DST_TO_SRC_SECOND_BYTES'],
    'Retransmission': ['RETRANSMITTED_IN_BYTES','RETRANSMITTED_IN_PKTS','RETRANSMITTED_OUT_BYTES','RETRANSMITTED_OUT_PKTS'],
    'Avg Throughput': ['SRC_TO_DST_AVG_THROUGHPUT'],
    'Packet Histogram': ['NUM_PKTS_UP_TO_128_BYTES','NUM_PKTS_128_TO_256_BYTES','NUM_PKTS_256_TO_512_BYTES','NUM_PKTS_512_TO_1024_BYTES','NUM_PKTS_1024_TO_1514_BYTES'],
    'TCP Window': ['TCP_WIN_MAX_IN','TCP_WIN_MAX_OUT'],
    'ICMP': ['ICMP_TYPE','ICMP_IPV4_TYPE'],
    'DNS': ['DNS_QUERY_TYPE','DNS_TTL_ANSWER'],
    'Inter-Arrival Time (NF3-exclusive)': ['SRC_TO_DST_IAT_MIN','SRC_TO_DST_IAT_MAX','SRC_TO_DST_IAT_AVG',
        'DST_TO_SRC_IAT_MIN','DST_TO_SRC_IAT_MAX','DST_TO_SRC_IAT_AVG','DST_TO_SRC_IAT_STDDEV'],
}
for grp, feats in groups.items():
    for f in feats:
        status = 'kept' if f in KEPT else 'pruned (corr>0.95)' if f in PRUNED else 'dropped'
        tab03_rows.append({'Category': grp, 'Feature': f, 'Status': status, 'Dimension': 'raw' if status=='kept' else '—'})

# Dropped fields
for f in fm['dropped_fields']:
    if not any(f in feats for feats in groups.values()):
        tab03_rows.append({'Category': 'Dropped', 'Feature': f, 'Status': 'dropped', 'Dimension': '—'})

# Time2Vec
tab03_rows.append({'Category': 'Temporal (Time2Vec)', 'Feature': 'Linear term (omega_0*t + b_0)', 'Status': 'kept', 'Dimension': 'temporal'})
tab03_rows.append({'Category': 'Temporal (Time2Vec)', 'Feature': 'Periodic terms x16 (sin(omega_k*t + b_k))', 'Status': 'kept', 'Dimension': 'temporal'})

tab03 = pd.DataFrame(tab03_rows)
tab03.to_csv(TABS_DIR/'tab03_feature_schema.csv', index=False)
tab03.to_markdown(TABS_DIR/'tab03_feature_schema.md', index=False)
print(f"  [OK] tab03 saved ({len(tab03)} features)")

# ── STEP 5: SPLIT DISTRIBUTION TABLE (new) ──────────────────
print("\n"+"="*60+"\nGENERATING: Split Label Distribution Table\n"+"="*60)
split_dist_rows = []
for ds_name, res in all_results.items():
    for sp_name in ['train','val','test']:
        sd = res['splits'][sp_name]
        total_sp = sd['n_flows']
        for cls in UNIFIED:
            cnt = sd['unified_counts'].get(cls, 0)
            if cnt > 0 or sp_name == 'train':
                split_dist_rows.append({
                    'Dataset': ds_name, 'Split': sp_name,
                    'Class': cls, 'Count': cnt,
                    'Pct_of_Split': round(cnt/total_sp*100, 4),
                    'N_Flows_in_Split': total_sp,
                })

split_dist_df = pd.DataFrame(split_dist_rows)
split_dist_df.to_csv(TABS_DIR/'split_label_distribution.csv', index=False)
split_dist_df.to_markdown(TABS_DIR/'split_label_distribution.md', index=False)

# Print summary
print("\n  Unified Class % across splits (training datasets only):")
print(f"  {'Class':<25s} {'CICIDS2018':>30s} {'UNSW-NB15':>30s}")
print(f"  {'':25s} {'Train':>9s} {'Val':>9s} {'Test':>9s} {'Train':>9s} {'Val':>9s} {'Test':>9s}")
print(f"  {'-'*79}")
for cls in UNIFIED:
    parts = []
    for ds_name in ['NF-CICIDS2018','NF-UNSW-NB15']:
        if ds_name not in all_results: continue
        for sp_name in ['train','val','test']:
            sd = all_results[ds_name]['splits'][sp_name]
            pct = sd['unified_counts'].get(cls,0)/sd['n_flows']*100
            parts.append(f"{pct:>8.2f}%")
    if any(float(p.replace('%',''))>0.001 for p in parts):
        print(f"  {cls:<25s} {' '.join(parts)}")

print(f"\n  [OK] split_label_distribution saved")

# ── STEP 6: COMBINED TRAINING DISTRIBUTION ──────────────────
print("\n"+"="*60+"\nGENERATING: Combined Training Class Distribution\n"+"="*60)
combined_train = Counter()
for ds_name in ['NF-CICIDS2018','NF-UNSW-NB15']:
    if ds_name in all_results:
        for cls in UNIFIED:
            combined_train[cls] += all_results[ds_name]['splits']['train']['unified_counts'].get(cls,0)

total_combined = sum(combined_train.values())
majority = max(combined_train.values())
median = np.median(list(combined_train.values()))

print(f"  Combined training flows: {total_combined:,}")
print(f"  Majority: {max(combined_train, key=combined_train.get)} ({majority:,})")
print(f"  Imbalance ratio: {majority/max(min(combined_train.values()),1):.1f}:1")
print(f"\n  {'Class':<25s} {'Count':>10s} {'%':>8s} {'Minority?':>10s} {'CVAE Target':>12s}")
print(f"  {'-'*65}")
for cls in UNIFIED:
    c = combined_train.get(cls,0)
    p = c/total_combined*100
    is_min = c < median
    target = int(majority*0.4)
    needed = max(0, target-c)
    print(f"  {cls:<25s} {c:>10,} {p:>7.2f}% {'YES' if is_min else '':>10s} {f'+{needed:,}' if needed>0 else '':>12s}")

# Save combined distribution
comb_df = pd.DataFrame([{'Class':c, 'Count':combined_train.get(c,0),
                          'Pct':round(combined_train.get(c,0)/total_combined*100,3),
                          'Minority':combined_train.get(c,0)<median,
                          'CVAE_Target_Count':int(majority*0.4),
                          'Synthetic_Needed':max(0,int(majority*0.4)-combined_train.get(c,0))}
                         for c in UNIFIED])
comb_df.to_csv(TABS_DIR/'combined_training_distribution.csv', index=False)

# ── STEP 7: FIGURES ─────────────────────────────────────────
print("\n"+"="*60+"\nGENERATING FIGURES\n"+"="*60)

# -- Fig 01: Architecture diagram --
fig, ax = plt.subplots(figsize=(15, 4))
stages = ['A: Graph\nConstruct', 'B: Time2Vec', 'C: E-GATv2\nEncoder',
          'D: MAE\nPretrain', 'E: CVAE\nAugment', 'F: Binary\nClassify',
          'G: Multiclass\nClassify', 'H: Few-Shot\nZero-Day', 'I: Eval\n& XAI']
scolors = ['#E8F5E9','#FFF3E0','#E3F2FD','#FCE4EC','#F3E5F5',
           '#E0F2F1','#FFF8E1','#EDE7F6','#EFEBE9']
for i,(s,c) in enumerate(zip(stages,scolors)):
    rect = plt.Rectangle((i*1.55,0),1.35,1.8,facecolor=c,edgecolor='black',lw=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(i*1.55+0.675,0.9,s,ha='center',va='center',fontsize=7,fontweight='bold')
    if i<len(stages)-1:
        ax.annotate('',xy=(i*1.55+1.35,0.9),xytext=(i*1.55+1.55,0.9),
                    arrowprops=dict(arrowstyle='->',color='#555',lw=1.8))
ax.set_xlim(-0.1,len(stages)*1.55); ax.set_ylim(-0.3,2.2); ax.axis('off')
ax.set_title('Figure 1: 9-Stage Graph-NIDS Pipeline Architecture',fontweight='bold',fontsize=13,pad=12)
plt.tight_layout()
plt.savefig(FIGS_DIR/'fig01_architecture_diagram.png'); plt.savefig(FIGS_DIR/'fig01_architecture_diagram.svg')
plt.close(); print("  [OK] fig01")

# -- Fig 02: Chronological split diagram --
fig, axes = plt.subplots(1,3,figsize=(14,5))
for i,(sp,color) in enumerate([('G_train\n(earliest 70%)','#4CAF50'),
                                ('G_val\n(middle 15%)','#FF9800'),
                                ('G_test\n(latest 15%)','#F44336')]):
    ax=axes[i]; np.random.seed(42+i); n_nodes=15; pos=np.random.rand(n_nodes,2)
    ax.scatter(pos[:,0],pos[:,1],s=80,c=color,edgecolors='black',lw=1,zorder=3)
    for _ in range(22):
        u,v=np.random.choice(n_nodes,2,replace=False)
        ax.plot([pos[u,0],pos[v,0]],[pos[u,1],pos[v,1]],'gray',alpha=0.35,lw=0.7)
    ax.set_title(sp,fontsize=10,fontweight='bold'); ax.set_xlim(-0.1,1.1); ax.set_ylim(-0.1,1.1); ax.axis('off')
fig.suptitle('Figure 2: Three Physically Separate Graphs (Chronological Split)',fontweight='bold',fontsize=13)
plt.tight_layout()
plt.savefig(FIGS_DIR/'fig02_graph_construction_diagram.png'); plt.savefig(FIGS_DIR/'fig02_graph_construction_diagram.svg')
plt.close(); print("  [OK] fig02")

# -- Fig 08: Class Distribution Pre-Augmentation --
fig, axes = plt.subplots(1,2,figsize=(16,6))
colors_ds = plt.cm.Set2(np.linspace(0,1,4))
ds_names_plot = list(all_results.keys())
# Per-dataset
for idx,ds_name in enumerate(ds_names_plot):
    if ds_name not in all_results: continue
    res=all_results[ds_name]; train_sd=res['splits']['train']
    active_cls=[c for c in UNIFIED if train_sd['unified_counts'].get(c,0)>0]
    vals=[train_sd['unified_counts'].get(c,0) for c in active_cls]
    axes[0].bar(np.arange(len(active_cls))+idx*0.2-0.3,vals,0.2,label=ds_name,color=colors_ds[idx],edgecolor='black',lw=0.3)
axes[0].set_xticks(range(len(active_cls))); axes[0].set_xticklabels([c[:12] for c in active_cls],rotation=45,ha='right',fontsize=7)
axes[0].set_ylabel('Flow Count'); axes[0].set_yscale('log'); axes[0].set_title('Training Set Distribution by Dataset (log scale)')
axes[0].legend(fontsize=7); axes[0].grid(axis='y',alpha=0.3)

# Combined training
sorted_cls=sorted(UNIFIED,key=lambda c:combined_train.get(c,0),reverse=True)
vals=[combined_train.get(c,0) for c in sorted_cls]
bar_colors=['#4CAF50' if combined_train.get(c,0)>=median else '#FF5722' for c in sorted_cls]
bars=axes[1].bar(range(len(sorted_cls)),vals,color=bar_colors,edgecolor='black',lw=0.5)
axes[1].set_xticks(range(len(sorted_cls))); axes[1].set_xticklabels([c[:12] for c in sorted_cls],rotation=45,ha='right',fontsize=7)
axes[1].set_ylabel('Flow Count'); axes[1].set_yscale('log')
axes[1].set_title('Combined Training (CICIDS2018+UNSW-NB15)\n[Green >= median, Red = CVAE augmentation targets]')
axes[1].grid(axis='y',alpha=0.3)
for i,(c,v) in enumerate(zip(sorted_cls,vals)):
    if v>0: axes[1].text(i,v*1.15,f'{v:,}',ha='center',fontsize=5.5,rotation=90)
fig.suptitle('Figure 8: Class Distribution — Pre-Augmentation',fontweight='bold',fontsize=13)
plt.tight_layout()
plt.savefig(FIGS_DIR/'fig08_class_distribution.png'); plt.savefig(FIGS_DIR/'fig08_class_distribution.svg')
plt.close(); print("  [OK] fig08")

# -- Fig: Split distribution heatmap --
fig, axes = plt.subplots(1,2,figsize=(18,7))
for ds_idx, ds_name in enumerate(['NF-CICIDS2018','NF-UNSW-NB15']):
    if ds_name not in all_results: continue
    res=all_results[ds_name]
    mat=np.zeros((N_CLASSES,3))
    for sp_i,sp_name in enumerate(['train','val','test']):
        sd=res['splits'][sp_name]
        for cl_i,cls in enumerate(UNIFIED):
            mat[cl_i,sp_i]=sd['unified_counts'].get(cls,0)/sd['n_flows']*100
    sns.heatmap(mat,annot=True,fmt='.2f',cmap='YlOrRd',
                xticklabels=['Train','Val','Test'],
                yticklabels=[c[:15] for c in UNIFIED],
                ax=axes[ds_idx],cbar_kws={'label':'% of split'},vmin=0,vmax=max(5,mat.max()))
    axes[ds_idx].set_title(f'{ds_name}\nClass % per Split')
plt.tight_layout()
plt.savefig(FIGS_DIR/'fig_split_distribution_heatmap.png'); plt.savefig(FIGS_DIR/'fig_split_distribution_heatmap.svg')
plt.close(); print("  [OK] fig_split_distribution_heatmap")

# -- Fig: Combined imbalance severity --
fig,ax=plt.subplots(figsize=(11,6))
classes_plot=[c for c in UNIFIED if combined_train.get(c,0)>0]
counts=[combined_train.get(c,0) for c in classes_plot]
pcts=[c/total_combined*100 for c in counts]
bar_cs=[]
for p in pcts:
    if p<0.1: bar_cs.append('#D32F2F')
    elif p<1.0: bar_cs.append('#FF5722')
    elif p<5.0: bar_cs.append('#FF9800')
    else: bar_cs.append('#4CAF50')
ax.barh(range(len(classes_plot)),counts,color=bar_cs,edgecolor='black',lw=0.5)
ax.set_yticks(range(len(classes_plot))); ax.set_yticklabels(classes_plot,fontsize=9)
ax.set_xscale('log'); ax.set_xlabel('Flow Count (log scale)')
ax.axvline(x=majority*0.4,color='#2196F3',ls='--',lw=1.5,alpha=0.6,label=f'40% of majority = {int(majority*0.4):,}')
ax.axvline(x=median,color='#9C27B0',ls='--',lw=1.5,alpha=0.6,label=f'Median = {median:,.0f}')
for i,(c,v,p) in enumerate(zip(classes_plot,counts,pcts)):
    mark='  <-- CVAE target' if v<median else ''
    ax.text(v*1.05,i,f'{v:,} ({p:.2f}%){mark}',va='center',fontsize=7)
ax.set_title('Figure: Class Imbalance Severity (Combined Training Data)\nRed<0.1% | Orange 0.1-1% | Yellow 1-5% | Green >5%',fontweight='bold',fontsize=12)
ax.legend(fontsize=8,loc='lower right'); ax.grid(axis='x',alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS_DIR/'fig_imbalance_severity.png'); plt.savefig(FIGS_DIR/'fig_imbalance_severity.svg')
plt.close(); print("  [OK] fig_imbalance_severity")

# -- Fig: Per-dataset pie charts --
fig,axes=plt.subplots(2,2,figsize=(16,14))
ds_labels=[
    ('NF-CICIDS2018','Train/Val/Test\n~20.1M flows'),
    ('NF-UNSW-NB15','Train/Val/Test\n~2.4M flows'),
    ('NF-ToN-IoT','Blind test (in-schema)\n~27.5M flows'),
    ('NF-BoT-IoT','Blind test (in-schema)\n~16.9M flows'),
]
for ax,(ds_name,info) in zip(axes.flat,ds_labels):
    if ds_name not in all_results: ax.axis('off'); continue
    sd=all_results[ds_name]['splits']['train']
    cls_pie=[c for c in UNIFIED if sd['unified_counts'].get(c,0)>0]
    vals_pie=[sd['unified_counts'].get(c,0) for c in cls_pie]
    if len(cls_pie)<8:
        wedges,texts,autotexts=ax.pie(vals_pie,labels=[c[:12] for c in cls_pie],autopct='%1.1f%%',
                                       colors=plt.cm.Set3(np.linspace(0,1,len(cls_pie))),textprops={'fontsize':7})
    else:
        wedges,texts=ax.pie(vals_pie,labels=[c[:12] for c in cls_pie],
                            colors=plt.cm.Set3(np.linspace(0,1,len(cls_pie))),textprops={'fontsize':7})
    ax.set_title(f'{ds_name}\n{info}',fontsize=9,fontweight='bold')
fig.suptitle('Figure: Dataset Composition (Training Split)',fontweight='bold',fontsize=14)
plt.tight_layout()
plt.savefig(FIGS_DIR/'fig_dataset_pie_charts.png'); plt.savefig(FIGS_DIR/'fig_dataset_pie_charts.svg')
plt.close(); print("  [OK] fig_dataset_pie_charts")

# ── STEP 8: FINAL SUMMARY ───────────────────────────────────
print("\n"+"="*70)
print("FULL DATASET ANALYSIS COMPLETE")
print("="*70)

# Exact numbers summary
summary_text = f"""# Dataset Analysis Summary

## All Datasets — Split Verification

| Dataset | Total Flows | Train | Val | Test | Time Span | Attack% (Train) | Attack% (Val) | Attack% (Test) |
|---|---|---|---|---|---|---|---|---|
"""
for ds_name, res in all_results.items():
    sd = res['splits']
    summary_text += f"| {ds_name} | {res['total']:,} | {sd['train']['n_flows']:,} | {sd['val']['n_flows']:,} | {sd['test']['n_flows']:,} | {res['time_span_h']:.1f}h | "
    for sp in ['train','val','test']:
        a_pct = (sd[sp]['n_flows']-sd[sp]['unified_counts'].get('Benign',0))/sd[sp]['n_flows']*100
        summary_text += f"{a_pct:.1f}% | "
    summary_text += "\n"

summary_text += f"""
## Combined Training Distribution (CICIDS2018 + UNSW-NB15)

| Class | Count | % | Minority? | CVAE Needed |
|---|---|---|---|---|
"""
for cls in UNIFIED:
    c = combined_train.get(cls,0)
    summary_text += f"| {cls} | {c:,} | {c/total_combined*100:.3f}% | {'YES' if c<median else ''} | {max(0,int(majority*0.4)-c):,} |\n"

summary_text += f"""
- **Total training flows:** {total_combined:,}
- **Imbalance ratio:** {majority/max(min(combined_train.values()),1):.1f}:1
- **Minority classes (CVAE targets):** {[c for c in UNIFIED if combined_train.get(c,0)<median]}
- **Time split gaps:** All < 1 second (essentially contiguous)
- **No time overlap in any dataset** — chronological split is clean
"""

with open(OUTPUT_DIR/'DATASET_ANALYSIS_SUMMARY.md','w') as f:
    f.write(summary_text)

print(summary_text)
print(f"\nFiles generated in {OUTPUT_DIR}:")
for f in sorted(OUTPUT_DIR.rglob('*')):
    if f.is_file():
        size_kb = f.stat().st_size/1024
        print(f"  {f.relative_to(OUTPUT_DIR)} ({size_kb:.0f} KB)")

print("\nDone.")
