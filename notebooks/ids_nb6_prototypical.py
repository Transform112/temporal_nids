"""
Notebook 6 — Prototypical Few-Shot Network, Zero-Day Detection (Stage H)
==========================================================================
Kaggle T4x2. Run cells sequentially.

Inputs:
  - Stage G checkpoint: checkpoints/G_multiclass/best.pt (from NB5)
  - Windowed graphs (from NB1)
  - feature_manifest.yaml, label_map.yaml

Outputs:
  - Prototypical network checkpoint: checkpoints/H_prototypical/best.pt + config.json
  - Novelty threshold τ: checkpoints/H_prototypical/tau.json
  - fig07_prototypical_diagram
  - fig12_zero_day_roc_pr
  - tab09_zero_day_results
  - logs/notebook_6_log.json
"""

# %% [markdown]
# # Notebook 6: Prototypical Few-Shot (Stage H)
#
# Episodic training: 5-way, 5-shot, 15 query/class, attention-weighted prototypes.

# %% [markdown]
# ## Cell 1: Imports & Setup

# %%
import torch; import torch.nn as nn; import torch.nn.functional as F
import torch.optim as optim
import numpy as np; import pandas as pd; import yaml; import json; import random
from datetime import datetime, timezone; from pathlib import Path
from collections import defaultdict, Counter
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# %% [markdown]
# ## Cell 2: Seed & Paths

# %%
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.deterministic = True

WORKING_DIR = Path('/kaggle/working')
CHECKPOINT_DIR = WORKING_DIR / 'checkpoints' / 'H_prototypical'
LOGS_DIR = WORKING_DIR / 'logs'
FIGURES_DIR = WORKING_DIR / 'outputs' / 'figures'
TABLES_DIR = WORKING_DIR / 'outputs' / 'tables'
ARTIFACTS_DIR = WORKING_DIR / 'artifacts'
for d in [CHECKPOINT_DIR, LOGS_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NB_START_TIME = datetime.now(timezone.utc).isoformat()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# %% [markdown]
# ## Cell 3: Load Frozen Encoder (from Stage G) & Extract Embeddings

# %%
with open(ARTIFACTS_DIR/'feature_manifest.yaml') as f: fm=yaml.safe_load(f)
with open(ARTIFACTS_DIR/'label_map.yaml') as f: lm=yaml.safe_load(f)
UNIFIED_CLASSES=lm['unified_classes']; N_CLASSES=len(UNIFIED_CLASSES)
EDGE_INPUT_DIM=fm['final_edge_input_dim']

# Model definitions
class Time2Vec(nn.Module):
    def __init__(self,k=16):
        super().__init__(); self.k=k
        self.w0=nn.Parameter(torch.randn(1)*0.1); self.b0=nn.Parameter(torch.zeros(1))
        self.omega=nn.Parameter(10.0**(torch.rand(k)*6-3)); self.bias=nn.Parameter(torch.zeros(k))
        self.output_dim=k+1
    def forward(self,t):
        if t.dim()==1: t=t.unsqueeze(-1)
        return torch.cat([self.w0*t+self.b0, torch.sin(self.omega*t+self.bias)],dim=-1)

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
    def forward(self,data):
        x=self._get_node_embed(data.num_nodes,data.edge_index.device)
        edge_attr=self.edge_proj(data.edge_attr)
        for conv,norm in zip(self.convs,self.norms):
            x_new,_=conv(x,data.edge_index,edge_attr=edge_attr,return_attention_weights=True)
            x_new=self.activation(x_new); x_new=self.dropout(x_new); x_new=norm(x_new)
            x=x+x_new if x.shape==x_new.shape else x_new
        return torch.cat([x[data.edge_index[0]],x[data.edge_index[1]],edge_attr],dim=-1)

# Load Stage G checkpoint, freeze encoder
ckpt_g = torch.load(WORKING_DIR/'checkpoints'/'G_multiclass'/'best.pt', map_location=device, weights_only=False)

time2vec = Time2Vec(k=16).to(device)
encoder = EGATv2Encoder(edge_dim=EDGE_INPUT_DIM).to(device)
time2vec.load_state_dict(ckpt_g['time2vec_state_dict'])
encoder.load_state_dict(ckpt_g['encoder_state_dict'])

for p in time2vec.parameters(): p.requires_grad = False
for p in encoder.parameters(): p.requires_grad = False
time2vec.eval(); encoder.eval()

# Load training graphs and extract embeddings
G_train_2018 = torch.load(WORKING_DIR/'G_NF-CICIDS2018_train_list.pt', weights_only=False)
G_train_unsw = torch.load(WORKING_DIR/'G_NF-UNSW-NB15_train_list.pt', weights_only=False)
G_train = G_train_2018 + G_train_unsw

all_times = torch.cat([g.edge_time for g in G_train])
TIME_MIN, TIME_MAX = all_times.min().item(), all_times.max().item()
def normalize_time(t): return (t-TIME_MIN)/(TIME_MAX-TIME_MIN)

def get_embeddings(graph_list):
    embeds, labels_list = [], []
    with torch.no_grad():
        for g in graph_list:
            g = g.to(device)
            t_norm = normalize_time(g.edge_time)
            t_embed = time2vec(t_norm)
            edge_attr_61 = torch.cat([g.edge_attr, t_embed], dim=-1)
            data = torch_geometric.data.Data(
                edge_index=g.edge_index, edge_attr=edge_attr_61, num_nodes=g.num_nodes)
            reps = encoder(data)
            embeds.append(reps.cpu()); labels_list.append(g.y.cpu())
    return torch.cat(embeds, dim=0), torch.cat(labels_list, dim=0)

print("Extracting training embeddings...")
train_embeddings, train_labels = get_embeddings(G_train)
print(f"Training embeddings: {train_embeddings.shape}")

# Organize by class
class_embeddings = defaultdict(list)
for i in range(len(train_labels)):
    cls = train_labels[i].item()
    class_embeddings[cls].append(train_embeddings[i])

# Convert to tensors per class
for cls in class_embeddings:
    class_embeddings[cls] = torch.stack(class_embeddings[cls])
    print(f"  Class {UNIFIED_CLASSES[cls]:25s}: {class_embeddings[cls].shape[0]:,} embeddings")

# %% [markdown]
# ## Cell 4: Prototypical Network with Attention-Weighted Prototypes

# %%
class AttentionPrototype(nn.Module):
    """Attention-weighted prototype computation."""
    def __init__(self, embed_dim=768):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )
    def forward(self, support_embeddings):
        """
        Args:
            support_embeddings: (n_support, embed_dim) — support samples for ONE class
        Returns:
            prototype: (embed_dim,) — attention-weighted mean
        """
        scores = self.attention(support_embeddings).squeeze(-1)  # (n_support,)
        weights = F.softmax(scores, dim=0)
        prototype = (support_embeddings * weights.unsqueeze(-1)).sum(dim=0)
        return prototype

class PrototypicalNetwork(nn.Module):
    """Prototypical few-shot network with attention-weighted prototypes."""
    def __init__(self, embed_dim=768):
        super().__init__()
        self.attention_proto = AttentionPrototype(embed_dim)
    def forward(self, support_embeddings, query_embeddings, support_labels, n_way):
        """
        Args:
            support_embeddings: (n_way * n_shot, embed_dim)
            query_embeddings: (n_way * n_query, embed_dim)
            support_labels: (n_way * n_shot,) — labels in [0, n_way-1]
            n_way: number of classes
        Returns:
            logits: (n_way * n_query, n_way) — cosine similarity to each prototype
        """
        prototypes = []
        for c in range(n_way):
            c_mask = support_labels == c
            c_embeddings = support_embeddings[c_mask]
            proto = self.attention_proto(c_embeddings)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes, dim=0)  # (n_way, embed_dim)

        # Cosine similarity
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        query_norm = F.normalize(query_embeddings, p=2, dim=-1)
        logits = torch.mm(query_norm, prototypes_norm.t())  # (n_query*n_way, n_way)
        return logits, prototypes

# Test
proto_net = PrototypicalNetwork(embed_dim=768)
test_support = torch.randn(5*5, 768); test_query = torch.randn(5*15, 768)
test_labels = torch.arange(5).repeat_interleave(5)
logits, prototypes = proto_net(test_support, test_query, test_labels, 5)
print(f"Prototypical test: logits {logits.shape}, prototypes {prototypes.shape}")
print("Prototypical network OK ✓")

# %% [markdown]
# ## Cell 5: Episodic Training

# %%
HP_PROTO = {
    'n_way': 5, 'n_shot': 5, 'n_query': 15,
    'episodes_per_epoch': 200, 'epochs': 30,
    'lr': 1e-4,
}

proto_net = PrototypicalNetwork(embed_dim=768).to(device)
optimizer = optim.Adam(proto_net.parameters(), lr=HP_PROTO['lr'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HP_PROTO['epochs'])

# Use only non-Benign classes for episodic training (attack classes only)
attack_classes = [c for c in range(N_CLASSES) if UNIFIED_CLASSES[c] != 'Benign']
print(f"Attack classes for episodic training: {len(attack_classes)}")

def sample_episode(n_way, n_shot, n_query):
    """Sample one episode from training data (G_train only)."""
    sampled_classes = random.sample(attack_classes, n_way)

    support_embeds = []; support_labels = []
    query_embeds = []; query_labels = []

    for label_idx, cls in enumerate(sampled_classes):
        cls_data = class_embeddings[cls]
        n_total = cls_data.shape[0]
        n_needed = n_shot + n_query

        if n_total < n_needed:
            # Sample with replacement if not enough
            indices = torch.randint(0, n_total, (n_needed,))
        else:
            indices = torch.randperm(n_total)[:n_needed]

        support_embeds.append(cls_data[indices[:n_shot]])
        query_embeds.append(cls_data[indices[n_shot:]])
        support_labels.extend([label_idx] * n_shot)
        query_labels.extend([label_idx] * n_query)

    support = torch.cat(support_embeds, dim=0)
    query = torch.cat(query_embeds, dim=0)
    s_labels = torch.tensor(support_labels, dtype=torch.long)
    q_labels = torch.tensor(query_labels, dtype=torch.long)

    return support, query, s_labels, q_labels

print(f"\nEpisodic training: {HP_PROTO['episodes_per_epoch']} episodes/epoch, {HP_PROTO['epochs']} epochs")
train_accs = []; val_accs = []

# Quick validation episodes (use G_val for val)
G_val_2018 = torch.load(WORKING_DIR/'G_NF-CICIDS2018_val_list.pt', weights_only=False)
G_val_unsw = torch.load(WORKING_DIR/'G_NF-UNSW-NB15_val_list.pt', weights_only=False)
G_val = G_val_2018 + G_val_unsw
print("Extracting validation embeddings...")
val_embeddings, val_labels = get_embeddings(G_val)
val_class_embeds = defaultdict(list)
for i in range(len(val_labels)):
    val_class_embeds[val_labels[i].item()].append(val_embeddings[i])
for cls in val_class_embeds:
    val_class_embeds[cls] = torch.stack(val_class_embeds[cls])

for epoch in range(HP_PROTO['epochs']):
    proto_net.train()
    epoch_acc = 0.0

    for ep in range(HP_PROTO['episodes_per_epoch']):
        support, query, s_labels, q_labels = sample_episode(
            HP_PROTO['n_way'], HP_PROTO['n_shot'], HP_PROTO['n_query'])
        support, query = support.to(device), query.to(device)
        s_labels, q_labels = s_labels.to(device), q_labels.to(device)

        logits, _ = proto_net(support, query, s_labels, HP_PROTO['n_way'])
        loss = F.cross_entropy(logits, q_labels)

        optimizer.zero_grad(); loss.backward(); optimizer.step()

        preds = logits.argmax(dim=1)
        epoch_acc += (preds == q_labels).float().mean().item()

    scheduler.step()
    avg_acc = epoch_acc / HP_PROTO['episodes_per_epoch']
    train_accs.append(avg_acc)

    # Quick validation
    proto_net.eval()
    val_acc = 0.0; n_val_ep = 20
    with torch.no_grad():
        for _ in range(n_val_ep):
            # Sample val episode (same n_way, n_shot, n_query)
            sampled = random.sample(attack_classes, HP_PROTO['n_way'])
            s_embeds, q_embeds = [], []; s_labs, q_labs = [], []
            for li, cls in enumerate(sampled):
                if cls not in val_class_embeds: continue
                cdata = val_class_embeds[cls]
                if cdata.shape[0] < HP_PROTO['n_shot']+HP_PROTO['n_query']: continue
                idx = torch.randperm(cdata.shape[0])[:HP_PROTO['n_shot']+HP_PROTO['n_query']]
                s_embeds.append(cdata[idx[:HP_PROTO['n_shot']]])
                q_embeds.append(cdata[idx[HP_PROTO['n_shot']:]])
                s_labs.extend([li]*HP_PROTO['n_shot']); q_labs.extend([li]*HP_PROTO['n_query'])
            if len(s_embeds) < HP_PROTO['n_way']: continue
            s_all = torch.cat(s_embeds, 0).to(device); q_all = torch.cat(q_embeds, 0).to(device)
            s_l = torch.tensor(s_labs).to(device); q_l = torch.tensor(q_labs).to(device)
            logits_v, _ = proto_net(s_all, q_all, s_l, HP_PROTO['n_way'])
            preds_v = logits_v.argmax(dim=1)
            val_acc += (preds_v == q_l).float().mean().item()

    val_accs.append(val_acc / max(n_val_ep, 1))
    print(f"Epoch {epoch+1:2d}/{HP_PROTO['epochs']}: Train Acc={avg_acc:.4f}, Val Acc={val_accs[-1]:.4f}")

# Save checkpoint
torch.save({
    'epoch': epoch+1, 'model_state_dict': proto_net.state_dict(),
    'train_accs': train_accs, 'val_accs': val_accs, 'config': HP_PROTO,
}, CHECKPOINT_DIR/'best.pt')
with open(CHECKPOINT_DIR/'config.json','w') as f: json.dump(HP_PROTO, f, indent=2)
print(f"Best val acc: {max(val_accs):.4f}")

# %% [markdown]
# ## Cell 6: Novelty Threshold Tuning (Leave-One-Class-Out)

# %%
print("\nLeave-One-Class-Out novelty threshold tuning...")
proto_net.eval()

# Compute prototypes for all known classes using 5 support samples each
all_prototypes = {}
with torch.no_grad():
    for cls in attack_classes:
        cls_data = class_embeddings[cls]
        idx = torch.randperm(cls_data.shape[0])[:5]
        proto = proto_net.attention_proto(cls_data[idx].to(device))
        all_prototypes[cls] = F.normalize(proto, p=2, dim=-1)

locoo_results = []
all_similarities_known = []
all_similarities_novel = []

for held_out_cls in attack_classes:
    # Held-out class = "novel", all others = "known"
    novel_data = val_class_embeds.get(held_out_cls)
    if novel_data is None or novel_data.shape[0] < 5: continue

    known_data = []
    for cls in attack_classes:
        if cls != held_out_cls and cls in val_class_embeds:
            known_data.append(val_class_embeds[cls][:100])  # sample 100 per known class

    if not known_data: continue

    # Compute similarity of each sample to its nearest prototype
    novel_sims = []
    with torch.no_grad():
        for i in range(0, novel_data.shape[0], 100):
            batch = novel_data[i:i+100].to(device)
            batch_norm = F.normalize(batch, p=2, dim=-1)
            max_sim = -1
            for proto in all_prototypes.values():
                sim = torch.mm(batch_norm, proto.unsqueeze(-1)).squeeze()
                max_sim = torch.maximum(max_sim, sim)
            novel_sims.extend(max_sim.cpu().tolist())

    known_sims = []
    with torch.no_grad():
        for kd in known_data:
            for i in range(0, kd.shape[0], 100):
                batch = kd[i:i+100].to(device)
                batch_norm = F.normalize(batch, p=2, dim=-1)
                max_sim = -1
                for proto_cls, proto in all_prototypes.items():
                    sim = torch.mm(batch_norm, proto.unsqueeze(-1)).squeeze()
                    max_sim = torch.maximum(max_sim, sim)
                known_sims.extend(max_sim.cpu().tolist())

    all_similarities_novel.extend(novel_sims)
    all_similarities_known.extend(known_sims)

    # Find optimal τ for this leave-one-out split
    # τ separates known (high similarity) from novel (low)
    y_true = np.array([0]*len(known_sims) + [1]*len(novel_sims))  # 1 = novel
    scores = np.array(known_sims + novel_sims)
    # Use inverted similarity as novelty score
    novelty_scores = -np.array(known_sims + novel_sims)

    prec, rec, thresholds = precision_recall_curve(y_true, novelty_scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-10)
    best_idx = np.argmax(f1s)
    best_tau = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    locoo_results.append({
        'held_out_class': UNIFIED_CLASSES[held_out_cls],
        'tau': float(best_tau),
        'precision': float(prec[best_idx]),
        'recall': float(rec[best_idx]),
        'f1': float(f1s[best_idx]),
    })
    print(f"  Held-out {UNIFIED_CLASSES[held_out_cls]:25s}: τ={best_tau:.3f}, F1={f1s[best_idx]:.4f}")

# Global τ: median across classes
global_tau = float(np.median([r['tau'] for r in locoo_results]))
print(f"\nGlobal τ (median): {global_tau:.4f}")

# Save
with open(CHECKPOINT_DIR/'tau.json','w') as f:
    json.dump({'global_tau': global_tau, 'per_class': locoo_results}, f, indent=2)

# %% [markdown]
# ## Cell 7: Zero-Day ROC/PR Curves (Fig 12)

# %%
# Aggregate novelty scores across all leave-one-out runs
novelty_scores = -np.array(all_similarities_known + all_similarities_novel)
y_true = np.array([0]*len(all_similarities_known) + [1]*len(all_similarities_novel))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ROC
fpr, tpr, _ = roc_curve(y_true, novelty_scores)
roc_auc = auc(fpr, tpr)
ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.3f}')
ax1.plot([0,1], [0,1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
ax1.set_title('Zero-Day Detection ROC'); ax1.legend(); ax1.grid(alpha=0.3)

# Precision-Recall
prec_vals, rec_vals, _ = precision_recall_curve(y_true, novelty_scores)
ap = average_precision_score(y_true, novelty_scores)
ax2.plot(rec_vals, prec_vals, 'r-', linewidth=2, label=f'AP = {ap:.3f}')
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.set_title('Zero-Day Detection PR Curve'); ax2.legend(); ax2.grid(alpha=0.3)

fig.suptitle('Figure 12: Zero-Day Novelty Detection Performance', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR/'fig12_zero_day_roc_pr.png', dpi=300)
plt.savefig(FIGURES_DIR/'fig12_zero_day_roc_pr.svg')
plt.show()

# Tab09
tab09_df = pd.DataFrame(locoo_results)
tab09_df.to_csv(TABLES_DIR/'tab09_zero_day_results.csv', index=False)
tab09_df.to_markdown(TABLES_DIR/'tab09_zero_day_results.md', index=False)
print("Saved: fig12, tab09")

# %% [markdown]
# ## Cell 8: Prototypical Diagram (Fig 07)

# %%
fig, ax = plt.subplots(figsize=(10, 7))

# Illustrate embedding space with support, query, and prototypes
np.random.seed(42)
n_classes = 5
colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

for c in range(n_classes):
    # Support points (clustered)
    center = np.random.randn(2) * 2
    support = center + np.random.randn(5, 2) * 0.3
    query = center + np.random.randn(15, 2) * 0.8

    # Prototype (attention-weighted center)
    proto = support.mean(axis=0)

    ax.scatter(support[:,0], support[:,1], c=[colors[c]], marker='o', s=60,
              edgecolors='black', linewidth=1, label=f'Class {c} Support')
    ax.scatter(query[:,0], query[:,1], c=[colors[c]], marker='x', s=30, alpha=0.5)
    ax.scatter(proto[0], proto[1], c=[colors[c]], marker='D', s=100,
              edgecolors='black', linewidth=2)

    # Circle for novelty threshold
    circle = plt.Circle(proto, 1.5, fill=False, color=colors[c], linestyle='--', alpha=0.3, linewidth=1)
    ax.add_patch(circle)

# Simulated "novel" point outside all circles
novel = np.array([5, 5])
ax.scatter(novel[0], novel[1], c='red', marker='*', s=200, edgecolors='black', linewidth=1.5, label='Novel/Zero-Day')

ax.set_xlabel('Embedding Dim 1'); ax.set_ylabel('Embedding Dim 2')
ax.set_title('Figure 7: Prototypical Few-Shot — Embedding Space with Novelty Thresholds')
ax.legend(fontsize=7, loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR/'fig07_prototypical_diagram.png', dpi=300)
plt.savefig(FIGURES_DIR/'fig07_prototypical_diagram.svg')
plt.show()
print("Saved: fig07")

# %% [markdown]
# ## Cell 9: Results Log

# %%
nb_end_time = datetime.now(timezone.utc).isoformat()
results_log = {
    'notebook': 6, 'stage': 'H',
    'title': 'Prototypical Few-Shot / Zero-Day Detection',
    'start_time': NB_START_TIME, 'end_time': nb_end_time,
    'hyperparameters': HP_PROTO,
    'best_val_acc': float(max(val_accs)),
    'global_tau': float(global_tau),
    'zero_day_auc': float(roc_auc),
    'zero_day_ap': float(ap),
    'leave_one_out_results': locoo_results,
    'warnings': [],
}
with open(LOGS_DIR/'notebook_6_log.json','w') as f:
    json.dump(results_log, f, indent=2, default=str)
print("Saved: logs/notebook_6_log.json")
print(f"\nNOTEBOOK 6 COMPLETE")
print(f"Global τ: {global_tau:.4f}, AUC: {roc_auc:.4f}, AP: {ap:.4f}")
print("Next: Notebook 7 — Evaluation, Ablation, XAI, Consolidation")
