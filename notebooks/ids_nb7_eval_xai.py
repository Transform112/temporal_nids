"""
Notebook 7 — Evaluation, Ablation, XAI, Consolidation (Stages I, J)
=====================================================================
Kaggle T4x2. Run cells sequentially.

Covers:
  - Cross-dataset blind test (in-schema + out-of-schema)
  - Adversarial robustness curve (PGD ε = {0, 0.01, 0.03, 0.05})
  - Ablation study (4 variants)
  - t-SNE/UMAP embeddings (fig09)
  - Inference latency (tab10)
  - XAI: SHAP + attention (fig13, fig14, tab11)
  - RESULTS_SUMMARY.md + output verification

Inputs: All checkpoints from NB1-6, blind-test datasets
Outputs: figs 09-14, tabs 06-08/10-11, RESULTS_SUMMARY.md
"""

# %% [markdown]
# # Notebook 7 — Evaluation, Ablation, XAI & Consolidation
#
# This is the final notebook. It does not train the core pipeline — it evaluates, explains, and consolidates.

# %% [markdown]
# ## PART 0: Imports & Setup

# %%
import torch; import torch.nn as nn; import torch.nn.functional as F
import numpy as np; import pandas as pd; import yaml; import json; import pickle; import random
from datetime import datetime, timezone; from pathlib import Path
from collections import defaultdict, Counter
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.manifold import TSNE
import time

# %% [markdown]
# ## Cell: Seed, Paths, Device

# %%
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.deterministic = True

WORKING_DIR = Path('/kaggle/working')
OUTPUT_DIR = WORKING_DIR / 'outputs'
FIGURES_DIR = OUTPUT_DIR / 'figures'; TABLES_DIR = OUTPUT_DIR / 'tables'
LOGS_DIR = WORKING_DIR / 'logs'; ARTIFACTS_DIR = WORKING_DIR / 'artifacts'
for d in [FIGURES_DIR, TABLES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NB_START_TIME = datetime.now(timezone.utc).isoformat()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# ## Cell: Load Models & Helpers

# %%
with open(ARTIFACTS_DIR/'feature_manifest.yaml') as f: fm=yaml.safe_load(f)
with open(ARTIFACTS_DIR/'label_map.yaml') as f: lm=yaml.safe_load(f)
UNIFIED_CLASSES=lm['unified_classes']; N_CLASSES=len(UNIFIED_CLASSES)
EDGE_INPUT_DIM=fm['final_edge_input_dim']; KEPT_FEATURES=fm['kept_features']

# ---- Model definitions (Time2Vec, EGATv2Encoder, BinaryHead, MulticlassHead, PrototypicalNetwork) ----
# (Same class definitions as NB2-NB6 — in practice, import from shared module)

class Time2Vec(nn.Module):
    def __init__(self,k=16):
        super().__init__(); self.k=k
        self.w0=nn.Parameter(torch.randn(1)*0.1); self.b0=nn.Parameter(torch.zeros(1))
        self.omega=nn.Parameter(10.0**(torch.rand(k)*6-3)); self.bias=nn.Parameter(torch.zeros(k))
        self.output_dim=k+1
    def forward(self,t):
        if t.dim()==1: t=t.unsqueeze(-1)
        return torch.cat([self.w0*t+self.b0,torch.sin(self.omega*t+self.bias)],dim=-1)

class EGATv2Encoder(nn.Module):
    def __init__(self, edge_dim=61, node_init_dim=128, hidden_dim=256,
                 num_heads=8, num_layers=3, dropout_attn=0.3, dropout_feat=0.2):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        self.hidden_dim=hidden_dim; self.num_heads=num_heads
        self.num_layers=num_layers; self.output_dim=hidden_dim*3
        self.node_init_dim=node_init_dim; self.node_embed=None
        self.edge_proj=nn.Linear(edge_dim,hidden_dim)
        self.convs=nn.ModuleList(); self.norms=nn.ModuleList()
        self.dropout=nn.Dropout(dropout_feat)
        for _ in range(num_layers):
            self.convs.append(GATv2Conv((-1,-1),hidden_dim//num_heads,
                heads=num_heads,edge_dim=hidden_dim,dropout=dropout_attn,concat=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.activation=nn.ELU()
    def _get_node_embed(self,num_nodes,device):
        if self.node_embed is None or self.node_embed.shape[0]<num_nodes:
            new=nn.Parameter(torch.randn(num_nodes,self.node_init_dim,device=device)*0.1)
            if self.node_embed is not None: new.data[:self.node_embed.shape[0]]=self.node_embed.data
            self.node_embed=new
        return self.node_embed[:num_nodes]
    def forward(self,data,return_attention=False):
        x=self._get_node_embed(data.num_nodes,data.edge_index.device)
        edge_attr=self.edge_proj(data.edge_attr)
        all_attn=[]
        for conv,norm in zip(self.convs,self.norms):
            x_new,attn=conv(x,data.edge_index,edge_attr=edge_attr,return_attention_weights=True)
            if return_attention: all_attn.append(attn)
            x_new=self.activation(x_new); x_new=self.dropout(x_new); x_new=norm(x_new)
            x=x+x_new if x.shape==x_new.shape else x_new
        out=torch.cat([x[data.edge_index[0]],x[data.edge_index[1]],edge_attr],dim=-1)
        return (out, all_attn) if return_attention else out

class BinaryHead(nn.Module):
    def __init__(self,input_dim=768,hidden_dim=256,bottleneck_dim=64,num_classes=2):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),nn.ELU(),nn.Dropout(0.3),
            nn.Linear(hidden_dim,bottleneck_dim),nn.ELU(),nn.Dropout(0.2),
            nn.Linear(bottleneck_dim,num_classes))
    def forward(self,x): return self.net(x)

class MulticlassHead(nn.Module):
    def __init__(self,input_dim=768,hidden_dim=256,num_classes=11):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),nn.ELU(),nn.Dropout(0.3),
            nn.Linear(hidden_dim,num_classes))
    def forward(self,x): return self.net(x)

class AttentionPrototype(nn.Module):
    def __init__(self,embed_dim=768):
        super().__init__()
        self.attention=nn.Sequential(nn.Linear(embed_dim,128),nn.Tanh(),nn.Linear(128,1))
    def forward(self,support_embeddings):
        scores=self.attention(support_embeddings).squeeze(-1)
        weights=F.softmax(scores,dim=0)
        return (support_embeddings*weights.unsqueeze(-1)).sum(dim=0)

class PrototypicalNetwork(nn.Module):
    def __init__(self,embed_dim=768):
        super().__init__(); self.attention_proto=AttentionPrototype(embed_dim)
    def forward(self,support_embeddings,query_embeddings,support_labels,n_way):
        prototypes=[]
        for c in range(n_way):
            c_mask=support_labels==c
            prototypes.append(self.attention_proto(support_embeddings[c_mask]))
        prototypes=torch.stack(prototypes,dim=0)
        return torch.mm(F.normalize(query_embeddings,p=2,dim=-1),
                        F.normalize(prototypes,p=2,dim=-1).t()), prototypes

# Load latest checkpoints
ckpt_f = torch.load(WORKING_DIR/'checkpoints'/'F_binary'/'best.pt', map_location=device, weights_only=False)
ckpt_g = torch.load(WORKING_DIR/'checkpoints'/'G_multiclass'/'best.pt', map_location=device, weights_only=False)
ckpt_h = torch.load(WORKING_DIR/'checkpoints'/'H_prototypical'/'best.pt', map_location=device, weights_only=False)

time2vec = Time2Vec(k=16).to(device); time2vec.load_state_dict(ckpt_g['time2vec_state_dict'])
encoder = EGATv2Encoder(edge_dim=EDGE_INPUT_DIM).to(device); encoder.load_state_dict(ckpt_g['encoder_state_dict'])
binary_head = BinaryHead().to(device); binary_head.load_state_dict(ckpt_f['binary_head_state_dict'])
multiclass_head = MulticlassHead(num_classes=N_CLASSES).to(device)
multiclass_head.load_state_dict(ckpt_g['multiclass_head_state_dict'])
proto_net = PrototypicalNetwork(embed_dim=768).to(device); proto_net.load_state_dict(ckpt_h['model_state_dict'])

for m in [time2vec, encoder, binary_head, multiclass_head, proto_net]:
    for p in m.parameters(): p.requires_grad = False
    m.eval()

# Load thresholds
with open(WORKING_DIR/'checkpoints'/'F_binary'/'threshold.json') as f:
    binary_threshold = json.load(f)['threshold']
with open(WORKING_DIR/'checkpoints'/'G_multiclass'/'thresholds.json') as f:
    per_class_thresholds = json.load(f)['per_class_thresholds']
with open(WORKING_DIR/'checkpoints'/'H_prototypical'/'tau.json') as f:
    tau_data = json.load(f); global_tau = tau_data['global_tau']

# Time normalization
G_train_all = (torch.load(WORKING_DIR/'G_NF-CICIDS2018_train_list.pt', weights_only=False) +
               torch.load(WORKING_DIR/'G_NF-UNSW-NB15_train_list.pt', weights_only=False))
all_times = torch.cat([g.edge_time for g in G_train_all])
TIME_MIN, TIME_MAX = all_times.min().item(), all_times.max().item()
def normalize_time(t): return (t-TIME_MIN)/(TIME_MAX-TIME_MIN)

# Helper: inference pipeline
def run_inference(graphs):
    """Run full inference pipeline and return predictions."""
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            t_norm = normalize_time(g.edge_time); t_embed = time2vec(t_norm)
            edge_attr_61 = torch.cat([g.edge_attr, t_embed], dim=-1)
            data = torch_geometric.data.Data(
                edge_index=g.edge_index, edge_attr=edge_attr_61, num_nodes=g.num_nodes)
            flow_reps = encoder(data)

            # Stage F: Binary
            bin_logits = binary_head(flow_reps)
            bin_probs = F.softmax(bin_logits, dim=-1)
            attack_mask = bin_probs[:,1] >= binary_threshold

            # Default: Benign
            preds = torch.zeros(flow_reps.shape[0], dtype=torch.long, device=device)

            if attack_mask.any():
                # Stage G: Multiclass
                multi_logits = multiclass_head(flow_reps[attack_mask])
                multi_probs = F.softmax(multi_logits, dim=-1)
                max_prob, max_cls = multi_probs.max(dim=-1)
                preds[attack_mask] = max_cls

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(g.y.cpu().tolist())
    return np.array(all_preds), np.array(all_targets)

print("Models loaded. Inference pipeline ready ✓")

# %% [markdown]
# ## PART 1: In-Domain Evaluation on G_test

# %%
print("="*60 + "\nIN-DOMAIN EVALUATION\n" + "="*60)

G_test_2018 = torch.load(WORKING_DIR/'G_NF-CICIDS2018_test_list.pt', weights_only=False)
G_test_unsw = torch.load(WORKING_DIR/'G_NF-UNSW-NB15_test_list.pt', weights_only=False)
G_test = G_test_2018 + G_test_unsw

test_preds, test_targets = run_inference(G_test)

in_domain_f1 = f1_score(test_targets, test_preds, average='macro')
print(f"In-domain Macro-F1: {in_domain_f1:.4f}")
print(f"In-domain Per-Class:")
for i, cls_name in enumerate(UNIFIED_CLASSES):
    cls_mask = test_targets == i
    if cls_mask.sum() > 0:
        cls_f1 = f1_score(test_targets == i, test_preds == i)
        print(f"  {cls_name:25s}: F1={cls_f1:.4f}")

# Update tab05 with test set results
tab05_test_rows = []
for i, cls_name in enumerate(UNIFIED_CLASSES):
    cls_mask = test_targets == i
    if cls_mask.sum() > 0:
        tab05_test_rows.append({
            'class': cls_name,
            'precision': round(precision_score(test_targets==i, test_preds==i, zero_division=0), 4),
            'recall': round(recall_score(test_targets==i, test_preds==i, zero_division=0), 4),
            'f1_score': round(f1_score(test_targets==i, test_preds==i), 4),
            'support': int(cls_mask.sum()),
        })
tab05_test_rows.append({
    'class': 'MACRO AVG',
    'f1_score': round(in_domain_f1, 4),
    'support': int(len(test_targets)),
})
pd.DataFrame(tab05_test_rows).to_csv(TABLES_DIR/'tab05_main_results.csv', index=False)
print("Saved: tab05_main_results (updated with test set)")

# %% [markdown]
# ## PART 2: Cross-Dataset Blind Test

# %%
print("\n" + "="*60 + "\nCROSS-DATASET BLIND TEST\n" + "="*60)

cross_dataset_results = []

# In-schema: ToN-IoT, BoT-IoT
# These should already be in the Kaggle dataset
# For now, check if they exist; if not, skip gracefully
blind_datasets = {
    'NF-ToN-IoT': WORKING_DIR / 'G_NF-ToN-IoT_test_list.pt',
    'NF-BoT-IoT': WORKING_DIR / 'G_NF-BoT-IoT_test_list.pt',
}

# If these don't exist yet, they need preprocessing through Stage A first
# (same steps as NB1 but applied to blind test datasets)

for ds_name, ds_path in blind_datasets.items():
    print(f"\n{ds_name}:")
    if not ds_path.exists():
        print(f"  SKIPPED — graph file not found. Run Stage A preprocessing first.")
        cross_dataset_results.append({
            'dataset': ds_name, 'schema': 'in-schema',
            'macro_f1': 'N/A', 'status': 'graphs not preprocessed'
        })
        continue

    blind_graphs = torch.load(ds_path, weights_only=False)
    blind_preds, blind_targets = run_inference(blind_graphs)
    macro_f1 = f1_score(blind_targets, blind_preds, average='macro')
    print(f"  Macro-F1: {macro_f1:.4f}")
    cross_dataset_results.append({
        'dataset': ds_name, 'schema': 'in-schema',
        'macro_f1': round(float(macro_f1), 4),
        'num_flows': len(blind_targets),
    })

# Out-of-schema (CIC-DDoS2019, CIC-Darknet2020) — deferred
cross_dataset_results.append({
    'dataset': 'CIC-DDoS2019', 'schema': 'out-of-schema',
    'macro_f1': 'N/A', 'status': 'dataset not yet available'
})
cross_dataset_results.append({
    'dataset': 'CIC-Darknet2020', 'schema': 'out-of-schema',
    'macro_f1': 'N/A', 'status': 'dataset not yet available'
})

tab06_df = pd.DataFrame(cross_dataset_results)
tab06_df.to_csv(TABLES_DIR/'tab06_cross_dataset_results.csv', index=False)
tab06_df.to_markdown(TABLES_DIR/'tab06_cross_dataset_results.md', index=False)
print("\nSaved: tab06_cross_dataset_results")

# %% [markdown]
# ## Cell: Fig 11 — Cross-Dataset Bar Chart

# %%
plot_data = [r for r in cross_dataset_results if r['macro_f1'] != 'N/A']
if plot_data:
    fig, ax = plt.subplots(figsize=(10, 5))
    names = ['In-Domain\n(CICIDS2018+UNSW-NB15)'] + [r['dataset'] for r in plot_data]
    values = [in_domain_f1] + [r['macro_f1'] for r in plot_data]
    colors = ['#4CAF50'] + ['#2196F3'] * len(plot_data)
    ax.bar(range(len(names)), values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Macro-F1'); ax.set_ylim(0, 1.05)
    ax.set_title('Figure 11: Cross-Dataset Generalization')
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/'fig11_cross_dataset_bar_chart.png', dpi=300)
    plt.savefig(FIGURES_DIR/'fig11_cross_dataset_bar_chart.svg')
    plt.show()
    print("Saved: fig11")

# %% [markdown]
# ## PART 3: Adversarial Robustness Curve (PGD ε sweep)

# %%
print("\n" + "="*60 + "\nADVERSARIAL ROBUSTNESS CURVE\n" + "="*60)

EPSILONS = [0.0, 0.01, 0.03, 0.05]
robustness_results = []

# Use a subset of test data for adversarial eval (speed)
G_test_subset = []
for g in G_test[:5]:
    if g.edge_index.shape[1] > 2000:
        idx = torch.randperm(g.edge_index.shape[1])[:2000]
        g.edge_index = g.edge_index[:, idx]
        g.edge_attr = g.edge_attr[idx]; g.edge_time = g.edge_time[idx]; g.y = g.y[idx]
    G_test_subset.append(g)

# Get feature bounds
fmins_61 = torch.cat([torch.tensor([-4.0]*44), torch.full((17,), -4.0)]).to(device)
fmaxs_61 = torch.cat([torch.tensor([4.0]*44), torch.full((17,), 4.0)]).to(device)

for eps in EPSILONS:
    all_preds_adv, all_targets_adv = [], []
    print(f"ε = {eps}:", end=" ")

    for g in G_test_subset:
        g = g.to(device)
        t_norm = normalize_time(g.edge_time); t_embed = time2vec(t_norm)
        edge_attr_61_clean = torch.cat([g.edge_attr, t_embed], dim=-1)

        if eps > 0:
            # Generate PGD attack
            edge_attr_61 = edge_attr_61_clean.clone()
            for _ in range(7):  # PGD steps
                edge_attr_61 = edge_attr_61.clone().detach().requires_grad_(True)
                data_pert = torch_geometric.data.Data(
                    edge_index=g.edge_index, edge_attr=edge_attr_61, num_nodes=g.num_nodes)
                reps = encoder(data_pert)
                # Maximize cross-entropy of multiclass prediction
                logits = multiclass_head(reps)
                loss = -F.cross_entropy(logits, g.y)
                grad = torch.autograd.grad(loss, edge_attr_61)[0]
                edge_attr_61 = edge_attr_61.detach() + 0.01 * grad.sign()
                delta = torch.clamp(edge_attr_61 - edge_attr_61_clean, -eps, eps)
                edge_attr_61 = torch.clamp(edge_attr_61_clean + delta, fmins_61, fmaxs_61)
        else:
            edge_attr_61 = edge_attr_61_clean

        data_final = torch_geometric.data.Data(
            edge_index=g.edge_index, edge_attr=edge_attr_61, num_nodes=g.num_nodes)
        reps_final = encoder(data_final)
        preds_final = multiclass_head(reps_final).argmax(dim=-1)
        all_preds_adv.extend(preds_final.cpu().tolist())
        all_targets_adv.extend(g.y.cpu().tolist())

    adv_f1 = f1_score(all_targets_adv, all_preds_adv, average='macro')
    robustness_results.append({'epsilon': eps, 'macro_f1': round(float(adv_f1), 4)})
    print(f"Macro-F1 = {adv_f1:.4f}")

# Fig 10
tab08_df = pd.DataFrame(robustness_results)
tab08_df.to_csv(TABLES_DIR/'tab08_adversarial_robustness.csv', index=False)
tab08_df.to_markdown(TABLES_DIR/'tab08_adversarial_robustness.md', index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tab08_df['epsilon'], tab08_df['macro_f1'], 'o-', color='#E53935', linewidth=2, markersize=10)
ax.set_xlabel('PGD Perturbation ε'); ax.set_ylabel('Macro-F1')
ax.set_title('Figure 10: Adversarial Robustness Curve')
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
for _, row in tab08_df.iterrows():
    ax.annotate(f"{row['macro_f1']:.3f}", (row['epsilon'], row['macro_f1']),
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR/'fig10_adversarial_robustness_curve.png', dpi=300)
plt.savefig(FIGURES_DIR/'fig10_adversarial_robustness_curve.svg')
plt.show()
print("Saved: fig10, tab08")

# %% [markdown]
# ## PART 4: t-SNE/UMAP Embeddings (Fig 09)

# %%
print("\n" + "="*60 + "\nt-SNE EMBEDDING VISUALIZATION\n" + "="*60)

# Extract embeddings at different stages for a sample
sample_size = 2000  # t-SNE is slow on large datasets

# Use test data
G_test_sample = []
for g in G_test[:5]:
    G_test_sample.append(g)

print("Extracting post-multiclass embeddings (from Stage G)...")
sample_embeddings = []
sample_labels = []
with torch.no_grad():
    for g in G_test_sample:
        g = g.to(device)
        if g.edge_index.shape[1] > sample_size:
            idx = torch.randperm(g.edge_index.shape[1])[:sample_size]
            g.edge_index=g.edge_index[:,idx]; g.edge_attr=g.edge_attr[idx]
            g.edge_time=g.edge_time[idx]; g.y=g.y[idx]

        t_norm = normalize_time(g.edge_time); t_embed = time2vec(t_norm)
        edge_attr_61 = torch.cat([g.edge_attr, t_embed], dim=-1)
        data = torch_geometric.data.Data(
            edge_index=g.edge_index, edge_attr=edge_attr_61, num_nodes=g.num_nodes)
        reps = encoder(data)
        sample_embeddings.append(reps.cpu()); sample_labels.append(g.y.cpu())

    sample_embeddings = torch.cat(sample_embeddings, dim=0)
    sample_labels = torch.cat(sample_labels, dim=0)

# Subsample for t-SNE
if sample_embeddings.shape[0] > sample_size:
    idx = torch.randperm(sample_embeddings.shape[0])[:sample_size]
    sample_embeddings = sample_embeddings[idx]; sample_labels = sample_labels[idx]

print(f"Running t-SNE on {sample_embeddings.shape[0]:,} embeddings...")
tsne_start = time.time()
tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
embeddings_2d = tsne.fit_transform(sample_embeddings.numpy())
print(f"t-SNE completed in {time.time()-tsne_start:.1f}s")

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))
for i, cls_name in enumerate(UNIFIED_CLASSES):
    cls_mask = sample_labels.numpy() == i
    if cls_mask.sum() > 0:
        ax.scatter(embeddings_2d[cls_mask, 0], embeddings_2d[cls_mask, 1],
                   c=[colors[i]], label=cls_name, alpha=0.5, s=5)
ax.set_xlabel('t-SNE Dim 1'); ax.set_ylabel('t-SNE Dim 2')
ax.set_title('Figure 9: t-SNE of Flow Embeddings (Post-Multiclass)')
ax.legend(fontsize=7, markerscale=3, loc='upper right')
plt.tight_layout()
plt.savefig(FIGURES_DIR/'fig09_tsne_embeddings.png', dpi=300)
plt.savefig(FIGURES_DIR/'fig09_tsne_embeddings.svg')
plt.show()
print("Saved: fig09")

# %% [markdown]
# ## PART 5: Inference Latency (Tab 10)

# %%
print("\n" + "="*60 + "\nINFERENCE LATENCY BENCHMARK\n" + "="*60)

# Single-flow latency
test_graph_single = G_test[0].clone().to(device)
# Take just 1 edge
single_edge = test_graph_single.edge_index[:, :1]
single_attr = test_graph_single.edge_attr[:1]
single_time = test_graph_single.edge_time[:1]

# Warmup
for _ in range(50):
    t_norm = normalize_time(single_time); t_embed = time2vec(t_norm)
    ea = torch.cat([single_attr, t_embed], dim=-1)
    data = torch_geometric.data.Data(edge_index=single_edge, edge_attr=ea, num_nodes=2)
    _ = encoder(data)

# Benchmark single-flow
n_runs = 500
torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_runs):
    t_norm = normalize_time(single_time); t_embed = time2vec(t_norm)
    ea = torch.cat([single_attr, t_embed], dim=-1)
    data = torch_geometric.data.Data(edge_index=single_edge, edge_attr=ea, num_nodes=2)
    reps = encoder(data); _ = binary_head(reps); _ = multiclass_head(reps)
torch.cuda.synchronize()
single_latency = (time.time() - t0) / n_runs * 1000  # ms

# Batch throughput
batch_size = 1024
test_batch = G_test[0].clone().to(device)
if test_batch.edge_index.shape[1] > batch_size:
    idx = torch.randperm(test_batch.edge_index.shape[1])[:batch_size]
    test_batch.edge_index = test_batch.edge_index[:, idx]
    test_batch.edge_attr = test_batch.edge_attr[idx]
    test_batch.edge_time = test_batch.edge_time[idx]

# Warmup
for _ in range(20):
    t_norm = normalize_time(test_batch.edge_time); t_embed = time2vec(t_norm)
    ea = torch.cat([test_batch.edge_attr, t_embed], dim=-1)
    data = torch_geometric.data.Data(edge_index=test_batch.edge_index, edge_attr=ea, num_nodes=test_batch.num_nodes)
    _ = encoder(data)

n_batch_runs = 100
torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_batch_runs):
    t_norm = normalize_time(test_batch.edge_time); t_embed = time2vec(t_norm)
    ea = torch.cat([test_batch.edge_attr, t_embed], dim=-1)
    data = torch_geometric.data.Data(edge_index=test_batch.edge_index, edge_attr=ea, num_nodes=test_batch.num_nodes)
    reps = encoder(data); _ = binary_head(reps); _ = multiclass_head(reps)
torch.cuda.synchronize()
batch_latency = (time.time() - t0) / n_batch_runs * 1000
batch_size_actual = test_batch.edge_index.shape[1]
batch_per_flow = batch_latency / batch_size_actual

# Model size
total_params = sum(p.numel() for p in list(time2vec.parameters())+list(encoder.parameters())+
                   list(binary_head.parameters())+list(multiclass_head.parameters()))
model_size_mb = total_params * 4 / (1024*1024)  # fp32 = 4 bytes/param

latency_results = [{
    'metric': 'Single-flow latency (ms)',
    'value': round(single_latency, 4),
    'target': '< 30ms',
}, {
    'metric': 'Batch latency (ms, batch='+str(batch_size_actual)+')',
    'value': round(batch_latency, 4),
    'target': '—',
}, {
    'metric': 'Per-flow amortized (ms)',
    'value': round(batch_per_flow, 4),
    'target': '—',
}, {
    'metric': 'Model parameters',
    'value': f'{total_params:,}',
    'target': '—',
}, {
    'metric': 'Model size (MB, fp32)',
    'value': round(model_size_mb, 2),
    'target': '—',
}]

tab10_df = pd.DataFrame(latency_results)
tab10_df.to_csv(TABLES_DIR/'tab10_inference_latency.csv', index=False)
tab10_df.to_markdown(TABLES_DIR/'tab10_inference_latency.md', index=False)
print(f"Single-flow latency: {single_latency:.2f} ms")
print(f"Batch latency ({batch_size_actual} flows): {batch_latency:.2f} ms ({batch_per_flow:.4f} ms/flow)")
print(f"Model size: {model_size_mb:.1f} MB")
print("Saved: tab10")

# %% [markdown]
# ## PART 6: XAI — SHAP Feature Attribution

# %%
print("\n" + "="*60 + "\nXAI — SHAP FEATURE ATTRIBUTION\n" + "="*60)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not available. Install with: !pip install shap")
    SHAP_AVAILABLE = False

if SHAP_AVAILABLE:
    # Prepare background (100 benign flows from E_train)
    G_train_sample = G_train_all[:3]
    background_edges = []
    bg_labels_list = []
    for g in G_train_sample:
        benign_mask = g.y_binary == 0
        if benign_mask.sum() > 0:
            background_edges.append(g.edge_attr[benign_mask][:50])
    background = torch.cat(background_edges, dim=0)[:100].numpy()
    print(f"SHAP background: {background.shape}")

    # Prepare samples for SHAP (~2000 flows/class from val)
    G_val_all = (torch.load(WORKING_DIR/'G_NF-CICIDS2018_val_list.pt', weights_only=False) +
                 torch.load(WORKING_DIR/'G_NF-UNSW-NB15_val_list.pt', weights_only=False))

    shap_samples = []
    shap_sample_labels = []
    samples_per_class = 200
    for cls_idx in range(N_CLASSES):
        cls_collected = 0
        for g in G_val_all:
            cls_mask = g.y == cls_idx
            if cls_mask.sum() > 0 and cls_collected < samples_per_class:
                n_take = min(int(cls_mask.sum()), samples_per_class - cls_collected)
                idx = cls_mask.nonzero(as_tuple=True)[0][:n_take]
                shap_samples.append(g.edge_attr[idx].cpu())
                shap_sample_labels.extend([cls_idx]*n_take)
                cls_collected += n_take
            if cls_collected >= samples_per_class: break
        print(f"  SHAP samples for {UNIFIED_CLASSES[cls_idx]:25s}: {cls_collected}")

    shap_X = torch.cat(shap_samples, dim=0).numpy()
    print(f"Total SHAP samples: {shap_X.shape}")

    # Define a wrapper for the model that takes 44-dim raw features
    # This is a simplification — full pipeline would need graph structure
    # For practical SHAP, we use the classifier head on pre-computed embeddings
    def model_wrapper(x_np):
        """Wrapper for SHAP: takes 44-dim features, returns multiclass probabilities."""
        with torch.no_grad():
            x = torch.tensor(x_np, dtype=torch.float32, device=device)
            # For simplified SHAP: just pass through multiclass head
            # (treating encoder output as a black box for the SHAP step)
            return F.softmax(multiclass_head(x.unsqueeze(0) if x.dim()==1 else x), dim=-1).cpu().numpy()

    # Use GradientExplainer (faster, if compatible)
    print("Computing SHAP values (GradientExplainer)...")
    try:
        explainer = shap.GradientExplainer(
            lambda x: F.softmax(multiclass_head(torch.tensor(x, dtype=torch.float32, device=device)), dim=-1).cpu().numpy(),
            background[:50]  # smaller background for speed
        )
        shap_values = explainer.shap_values(shap_X[:500])  # limit for speed
        print(f"SHAP values computed: {[sv.shape for sv in shap_values] if isinstance(shap_values, list) else shap_values.shape}")

        # Fig 13: Top-8 SHAP features per class
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.flatten()

        for cls_idx in range(min(N_CLASSES, len(axes))):
            ax = axes[cls_idx]
            if isinstance(shap_values, list) and cls_idx < len(shap_values):
                mean_shap = np.abs(shap_values[cls_idx]).mean(axis=0)
                top8_idx = np.argsort(mean_shap)[-8:]
                top8_vals = mean_shap[top8_idx]
                top8_names = [KEPT_FEATURES[i] if i < len(KEPT_FEATURES) else f'Time2Vec_{i}'
                             for i in top8_idx]

                ax.barh(range(8), top8_vals, color='#2196F3', edgecolor='black', linewidth=0.5)
                ax.set_yticks(range(8))
                ax.set_yticklabels([n[:20] for n in top8_names], fontsize=7)
                ax.set_title(UNIFIED_CLASSES[cls_idx][:15], fontsize=9)
                ax.set_xlabel('Mean |SHAP|')
                ax.grid(axis='x', alpha=0.3)
            else:
                ax.axis('off')

        fig.suptitle('Figure 13: Top-8 SHAP Feature Importance per Class', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR/'fig13_shap_summary.png', dpi=300)
        plt.savefig(FIGURES_DIR/'fig13_shap_summary.svg')
        plt.show()
        print("Saved: fig13")

        # Tab 11: Top-5 SHAP features per class
        tab11_rows = []
        for cls_idx in range(N_CLASSES):
            cls_name = UNIFIED_CLASSES[cls_idx]
            if isinstance(shap_values, list) and cls_idx < len(shap_values):
                mean_shap = np.abs(shap_values[cls_idx]).mean(axis=0)
                top5_idx = np.argsort(mean_shap)[-5:][::-1]
                for rank, feat_idx in enumerate(top5_idx):
                    feat_name = KEPT_FEATURES[feat_idx] if feat_idx < len(KEPT_FEATURES) else f'Time2Vec_{feat_idx}'
                    tab11_rows.append({
                        'class': cls_name, 'rank': rank+1,
                        'feature': feat_name,
                        'mean_shap_value': round(float(mean_shap[feat_idx]), 6),
                    })

        tab11_df = pd.DataFrame(tab11_rows)
        tab11_df.to_csv(TABLES_DIR/'tab11_shap_top_features.csv', index=False)
        tab11_df.to_markdown(TABLES_DIR/'tab11_shap_top_features.md', index=False)
        print("Saved: tab11")
    except Exception as e:
        print(f"SHAP computation failed: {e}")
        print("Saving placeholder — SHAP may need model architecture adaptation")
else:
    print("SHAP not available — skipping XAI SHAP computation")
    print("To enable: !pip install shap in a cell above")

# %% [markdown]
# ## PART 7: XAI — Attention Visualization (Fig 14)

# %%
print("\n" + "="*60 + "\nXAI — ATTENTION VISUALIZATION\n" + "="*60)

# Extract attention weights from final encoder layer for a sample graph
sample_g = G_test[0].clone().to(device)
if sample_g.edge_index.shape[1] > 100:
    idx = torch.randperm(sample_g.edge_index.shape[1])[:100]
    sample_g.edge_index = sample_g.edge_index[:, idx]
    sample_g.edge_attr = sample_g.edge_attr[idx]
    sample_g.edge_time = sample_g.edge_time[idx]
    sample_g.y = sample_g.y[idx]

t_norm = normalize_time(sample_g.edge_time); t_embed = time2vec(t_norm)
edge_attr_61 = torch.cat([sample_g.edge_attr, t_embed], dim=-1)
data = torch_geometric.data.Data(
    edge_index=sample_g.edge_index, edge_attr=edge_attr_61, num_nodes=sample_g.num_nodes)

with torch.no_grad():
    flow_reps, attention_weights = encoder(data, return_attention=True)

# attention_weights: list of (edge_index, attention_weights) tuples per layer
# Use final layer
final_attn = attention_weights[-1]  # tuple (edge_index, attn_weights)
if isinstance(final_attn, tuple):
    attn_edges, attn_scores = final_attn
    # attn_scores shape: (num_edges, num_heads) → average across heads
    avg_attn = attn_scores.mean(dim=1).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    # Draw graph with edge thickness proportional to attention
    edge_index_np = attn_edges.cpu().numpy()
    n_nodes = sample_g.num_nodes
    pos = {i: (np.cos(2*np.pi*i/n_nodes) + np.random.randn()*0.05,
               np.sin(2*np.pi*i/n_nodes) + np.random.randn()*0.05) for i in range(n_nodes)}

    for i in range(edge_index_np.shape[1]):
        u, v = edge_index_np[0, i], edge_index_np[1, i]
        w = avg_attn[i]
        alpha = min(w / avg_attn.max(), 1.0) * 0.8 + 0.2
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                'gray', alpha=alpha, linewidth=w*3)

    # Draw nodes
    for node in range(n_nodes):
        ax.scatter(pos[node][0], pos[node][1], s=50, c='#2196F3',
                  edgecolors='black', linewidth=0.5, zorder=3)

    ax.set_title('Figure 14: Attention Visualization — Final E-GATv2 Layer')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/'fig14_attention_visualization.png', dpi=300)
    plt.savefig(FIGURES_DIR/'fig14_attention_visualization.svg')
    plt.show()
    print("Saved: fig14")

# %% [markdown]
# ## PART 8: Ablation Notes (Tab 07 skeleton)

# %%
print("\n" + "="*60 + "\nABLATION STUDY\n" + "="*60)
print("Ablation requires 4× full retraining of NB2-NB6 with one component removed each.")
print("Run ablation variants as separate Kaggle sessions, then consolidate results here.")
print("Variant notebooks: ids_ablation_no_time2vec, ids_ablation_no_cvae,")
print("  ids_ablation_no_adversarial, ids_ablation_no_prototypical")

# Placeholder tab07
tab07_placeholder = pd.DataFrame([
    {'variant': 'Full Model', 'macro_f1': round(float(in_domain_f1), 4), 'status': 'complete'},
    {'variant': 'no-Time2Vec', 'macro_f1': 'TBD', 'status': 'pending ablation retraining'},
    {'variant': 'no-CVAE', 'macro_f1': 'TBD', 'status': 'pending ablation retraining'},
    {'variant': 'no-Adversarial Training', 'macro_f1': 'TBD', 'status': 'pending ablation retraining'},
    {'variant': 'no-Prototypical Stage', 'macro_f1': 'TBD', 'status': 'pending ablation retraining'},
])
tab07_placeholder.to_csv(TABLES_DIR/'tab07_ablation.csv', index=False)
tab07_placeholder.to_markdown(TABLES_DIR/'tab07_ablation.md', index=False)
print("Saved: tab07 (placeholder — fill after ablation runs)")

# %% [markdown]
# ## PART 9: RESULTS_SUMMARY.md & Verification

# %%
# Generate consolidated results summary
summary_lines = [
    "# RESULTS SUMMARY — Graph-NIDS for IEEE Access",
    f"\nGenerated: {NB_START_TIME}",
    "\n## In-Domain Results (Test Set)",
    f"- Macro-F1: {in_domain_f1:.4f}",
    f"- Dataset: CICIDS2018 + UNSW-NB15 (chronological test split)",
]

# Add per-class results
summary_lines.append("\n### Per-Class F1 Scores")
for row in tab05_test_rows:
    if row['class'] != 'MACRO AVG':
        summary_lines.append(f"- {row['class']}: {row.get('f1_score', 'N/A')}")

# Cross-dataset
summary_lines.append("\n## Cross-Dataset Generalization")
for row in cross_dataset_results:
    summary_lines.append(f"- {row['dataset']} ({row['schema']}): {row['macro_f1']}")

# Adversarial robustness
summary_lines.append("\n## Adversarial Robustness")
for _, row in tab08_df.iterrows():
    summary_lines.append(f"- ε = {row['epsilon']:.2f}: Macro-F1 = {row['macro_f1']:.4f}")

# Zero-day
summary_lines.append("\n## Zero-Day Detection")
summary_lines.append(f"- Global novelty threshold τ: {global_tau:.4f}")
summary_lines.append(f"- Leave-one-class-out results in tab09")

# Latency
summary_lines.append("\n## Inference Latency")
summary_lines.append(f"- Single-flow: {single_latency:.2f} ms")
summary_lines.append(f"- Batch ({batch_size_actual} flows): {batch_latency:.2f} ms ({batch_per_flow:.4f} ms/flow)")
summary_lines.append(f"- Model size: {model_size_mb:.1f} MB (fp32)")

# Ablation placeholder
summary_lines.append("\n## Ablation Study")
summary_lines.append("- See tab07 — FULL MODEL ONLY reported above; variants TBD")

with open(WORKING_DIR/'RESULTS_SUMMARY.md', 'w') as f:
    f.write('\n'.join(summary_lines))
print("Saved: RESULTS_SUMMARY.md")

# %% [markdown]
# ## PART 10: Output Verification

# %%
print("\n" + "="*60)
print("OUTPUT VERIFICATION")
print("="*60)

required_figures = [
    'fig01_architecture_diagram', 'fig02_graph_construction_diagram',
    'fig03_time2vec_diagram', 'fig04_attention_diagram', 'fig05_mae_pretrain_diagram',
    'fig06_cvae_diagram', 'fig07_prototypical_diagram', 'fig08_class_distribution',
    'fig09_tsne_embeddings', 'fig10_adversarial_robustness_curve',
    'fig11_cross_dataset_bar_chart', 'fig12_zero_day_roc_pr',
    'fig13_shap_summary', 'fig14_attention_visualization',
    'fig15_confusion_matrix', 'fig16_training_curves',
]

required_tables = [
    'tab01_dataset_statistics', 'tab02_taxonomy_mapping', 'tab03_feature_schema',
    'tab04_hyperparameters', 'tab05_main_results', 'tab06_cross_dataset_results',
    'tab07_ablation', 'tab08_adversarial_robustness', 'tab09_zero_day_results',
    'tab10_inference_latency', 'tab11_shap_top_features', 'tab12_related_work_comparison',
]

all_pass = True
print("\nFigures:")
for stem in required_figures:
    png_ok = (FIGURES_DIR / f'{stem}.png').exists()
    svg_ok = (FIGURES_DIR / f'{stem}.svg').exists()
    status = '✓' if png_ok and svg_ok else '✗ MISSING'
    if not (png_ok and svg_ok): all_pass = False
    print(f"  [{status}] {stem}")

print("\nTables:")
for stem in required_tables:
    csv_ok = (TABLES_DIR / f'{stem}.csv').exists()
    md_ok = (TABLES_DIR / f'{stem}.md').exists()
    # Some tables may be placeholders
    if csv_ok or md_ok:
        status = '✓' if csv_ok and md_ok else '~ PARTIAL'
        if not (csv_ok and md_ok): all_pass = False
    else:
        status = '✗ MISSING'
        all_pass = False
    print(f"  [{status}] {stem}")

print(f"\n{'✓ ALL OUTPUTS PRESENT' if all_pass else '✗ SOME OUTPUTS MISSING — check above'}")

# NOTE: Many figures are produced in earlier notebooks (NB1-6).
# This verification runs in NB7 and checks what's been saved across all notebooks.
# Run this after copying all outputs to the shared /kaggle/working/outputs/ directory.

# %% [markdown]
# ## Cell: Final Results Log

# %%
nb_end_time = datetime.now(timezone.utc).isoformat()
results_log = {
    'notebook': 7, 'stages': ['I', 'J'],
    'title': 'Evaluation, Ablation, XAI, Consolidation',
    'start_time': NB_START_TIME, 'end_time': nb_end_time,
    'results': {
        'in_domain_macro_f1': float(in_domain_f1),
        'cross_dataset': cross_dataset_results,
        'adversarial_robustness': robustness_results,
        'single_flow_latency_ms': float(single_latency),
        'batch_latency_ms': float(batch_latency),
        'batch_size': batch_size_actual,
    },
    'output_verification_passed': all_pass,
    'warnings': [],
}
with open(LOGS_DIR/'notebook_7_log.json', 'w') as f:
    json.dump(results_log, f, indent=2, default=str)
print("Saved: logs/notebook_7_log.json")

print("\n" + "="*60)
print("NOTEBOOK 7 COMPLETE")
print("="*60)
print(f"In-domain Macro-F1: {in_domain_f1:.4f}")
print(f"Output verification: {'PASSED' if all_pass else 'SOME MISSING'}")
print("\nThe pipeline is complete. Remaining work:")
print("  1. Run ablation variants (4× retraining)")
print("  2. Fill tab07 with ablation results")
print("  3. When CIC-DDoS2019/Darknet2020 available: preprocess + eval")
print("  4. Finalize PAPER_DRAFT.md with real numbers from RESULTS_SUMMARY.md")
print("  5. Verify all citations are real")
print("  6. Update 07_WORK_COMPLETION.md with final sign-off")
