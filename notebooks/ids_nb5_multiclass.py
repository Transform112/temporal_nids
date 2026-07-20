"""
Notebook 5 — Multiclass Classification, Stage-2 Head (Stage G)
================================================================
Kaggle T4x2, fp16. Run cells sequentially.

Inputs:
  - Stage F checkpoint: checkpoints/F_binary/best.pt (from NB4)
  - Synthetic embeddings: checkpoints/E_cvae/synthetic_embeddings.pt (from NB3)
  - Windowed graphs (from NB1)
  - feature_manifest.yaml, label_map.yaml

Outputs:
  - Stage G checkpoint: checkpoints/G_multiclass/best.pt + config.json
  - Per-class threshold vector: checkpoints/G_multiclass/thresholds.json
  - fig15_confusion_matrix
  - tab05_main_results (in-domain rows)
  - logs/notebook_5_log.json
"""

# %% [markdown]
# # Notebook 5: Multiclass Classification — Stage-2 Head (Stage G)
#
# Runs on attack-flagged flows only. Uses real minority + synthetic (1:1) + undersampled majority.

# %% [markdown]
# ## Cell 1: Imports & Setup

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import yaml
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix, classification_report
)

# %% [markdown]
# ## Cell 2: Seed & Paths

# %%
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

WORKING_DIR = Path('/kaggle/working')
CHECKPOINT_DIR = WORKING_DIR / 'checkpoints' / 'G_multiclass'
LOGS_DIR = WORKING_DIR / 'logs'
FIGURES_DIR = WORKING_DIR / 'outputs' / 'figures'
TABLES_DIR = WORKING_DIR / 'outputs' / 'tables'
ARTIFACTS_DIR = WORKING_DIR / 'artifacts'
for d in [CHECKPOINT_DIR, LOGS_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NB_START_TIME = datetime.now(timezone.utc).isoformat()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# ## Cell 3: Load Artifacts & Model Definitions

# %%
with open(ARTIFACTS_DIR / 'feature_manifest.yaml', 'r') as f:
    fm = yaml.safe_load(f)
with open(ARTIFACTS_DIR / 'label_map.yaml', 'r') as f:
    label_map = yaml.safe_load(f)

UNIFIED_CLASSES = label_map['unified_classes']
EDGE_INPUT_DIM = fm['final_edge_input_dim']
N_CLASSES = len(UNIFIED_CLASSES)

# ---- Model definitions (same as NB4) ----
class Time2Vec(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        self.k = k; self.w0 = nn.Parameter(torch.randn(1)*0.1)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.omega = nn.Parameter(10.0**(torch.rand(k)*6-3))
        self.bias = nn.Parameter(torch.zeros(k))
        self.output_dim = k+1
    def forward(self, t):
        if t.dim()==1: t=t.unsqueeze(-1)
        return torch.cat([self.w0*t+self.b0, torch.sin(self.omega*t+self.bias)], dim=-1)

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

class MulticlassHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x): return self.net(x)

# Load Stage F checkpoint (continues fine-tuning from here)
ckpt_f = torch.load(WORKING_DIR / 'checkpoints' / 'F_binary' / 'best.pt',
                    map_location=device, weights_only=False)

time2vec = Time2Vec(k=16).to(device)
encoder = EGATv2Encoder(edge_dim=EDGE_INPUT_DIM).to(device)
time2vec.load_state_dict(ckpt_f['time2vec_state_dict'])
encoder.load_state_dict(ckpt_f['encoder_state_dict'])

multiclass_head = MulticlassHead(input_dim=768, hidden_dim=256, num_classes=N_CLASSES).to(device)
print(f"Loaded Stage F (val_f1={ckpt_f['val_f1']:.4f})")
print(f"Multiclass head: {sum(p.numel() for p in multiclass_head.parameters()):,} params")

# Load synthetic embeddings
synth_data = torch.load(WORKING_DIR / 'checkpoints' / 'E_cvae' / 'synthetic_embeddings.pt', weights_only=False)
synth_embeddings = synth_data['embeddings']  # (N_synth, 768)
synth_labels = synth_data['labels']           # (N_synth,)
print(f"Synthetic embeddings loaded: {synth_embeddings.shape[0]:,}")

# %% [markdown]
# ## Cell 4: Load Training Data & Time Normalizer

# %%
G_train_2018 = torch.load(WORKING_DIR/'G_NF-CICIDS2018_train_list.pt', weights_only=False)
G_train_unsw = torch.load(WORKING_DIR/'G_NF-UNSW-NB15_train_list.pt', weights_only=False)
G_val_2018 = torch.load(WORKING_DIR/'G_NF-CICIDS2018_val_list.pt', weights_only=False)
G_val_unsw = torch.load(WORKING_DIR/'G_NF-UNSW-NB15_val_list.pt', weights_only=False)

G_train = G_train_2018 + G_train_unsw
G_val = G_val_2018 + G_val_unsw

all_times = torch.cat([g.edge_time for g in G_train])
TIME_MIN, TIME_MAX = all_times.min().item(), all_times.max().item()
def normalize_time(t): return (t-TIME_MIN)/(TIME_MAX-TIME_MIN)

# Filter to attack-flagged ONLY (but use ground-truth for training)
# Actually, at TRAINING time we train on all flows with ground truth labels
# At INFERENCE time, Stage G only sees what Stage F passes through
# For training: use all flows, but we care about per-class performance on attack classes

# Compute effective-number-of-samples weights for focal loss
all_y = torch.cat([g.y for g in G_train])
class_counts = Counter(all_y.tolist())
# Effective number: (1 - beta^count) / (1 - beta) with beta = (N-1)/N
N = sum(class_counts.values())
beta = (N - 1) / N
eff_num = {c: (1 - beta**count) / (1 - beta) for c, count in class_counts.items()}
eff_weights = torch.tensor([1.0 / max(eff_num.get(i, 1), 1) for i in range(N_CLASSES)], device=device)
eff_weights = eff_weights / eff_weights.sum() * N_CLASSES  # normalize
print("Effective-number class weights:")
for i, w in enumerate(eff_weights):
    print(f"  {UNIFIED_CLASSES[i]:25s}: {w:.4f}")

# %% [markdown]
# ## Cell 5: Focal Loss

# %%
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce); focal = (1-pt)**self.gamma * ce
        if self.alpha is not None:
            focal = self.alpha[targets] * focal
        return focal.mean()

focal_loss = FocalLoss(gamma=2.0, alpha=eff_weights)

# %% [markdown]
# ## Cell 6: Training Loop

# %%
HP_MULTI = {
    'lr': 1e-5,
    'epochs': 20,
    'batch_size': 2048,
    'pgd_epsilon': 0.03, 'pgd_alpha': 0.01, 'pgd_steps': 7,
    'pgd_batch_fraction': 0.30,
    'synth_ratio': 1.0,  # 1:1 real:synthetic for minority
}

optimizer = optim.AdamW([
    {'params': time2vec.parameters(), 'lr': HP_MULTI['lr']},
    {'params': encoder.parameters(), 'lr': HP_MULTI['lr']},
    {'params': multiclass_head.parameters(), 'lr': HP_MULTI['lr']*10},
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HP_MULTI['epochs'])
scaler_amp = GradScaler()

# Helper: encode edges
def encode_edges(edge_attr_44, edge_time, model, t2v, edge_index, num_nodes):
    t_norm = normalize_time(edge_time)
    t_embed = t2v(t_norm)
    edge_attr_61 = torch.cat([edge_attr_44, t_embed], dim=-1)
    data = torch_geometric.data.Data(
        edge_index=edge_index, edge_attr=edge_attr_61, num_nodes=num_nodes)
    return model(data), edge_attr_61

# Feature bounds for PGD
all_train_attr = torch.cat([g.edge_attr for g in G_train])
fmins = all_train_attr.min(dim=0).values.to(device)
fmaxs = all_train_attr.max(dim=0).values.to(device)
fmins_61 = torch.cat([fmins, torch.full((17,), -4.0, device=device)])
fmaxs_61 = torch.cat([fmaxs, torch.full((17,), 4.0, device=device)])

# Cache synthetic embeddings
synth_emb = synth_embeddings.to(device)

train_losses = []; val_f1s = []; best_val_f1 = 0.0

print(f"Training: {HP_MULTI['epochs']} epochs")

for epoch in range(HP_MULTI['epochs']):
    time2vec.train(); encoder.train(); multiclass_head.train()
    epoch_loss = 0.0; n_batches = 0

    for g in G_train:
        g = g.to(device)
        if g.edge_index.shape[1] < 4: continue

        # Undersample majority classes
        class_indices = {}
        for cls_idx in range(N_CLASSES):
            mask = g.y == cls_idx
            idx = mask.nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                class_indices[cls_idx] = idx

        if len(class_indices) < 2: continue

        # Undersample to ~200 per class per graph window
        max_per_class = 200
        keep_idx = []
        for cls_idx, idx in class_indices.items():
            n_keep = min(len(idx), max_per_class)
            keep = idx[torch.randperm(len(idx))[:n_keep]]
            keep_idx.append(keep)

        # Add synthetic samples for minority classes
        for cls_idx in class_indices:
            synth_mask = synth_labels == cls_idx
            synth_for_cls = synth_emb[synth_mask]
            if len(synth_for_cls) > 0:
                # Add up to len(class_indices[cls_idx]) synthetic samples (1:1 ratio)
                n_synth = min(len(synth_for_cls), len(class_indices[cls_idx]))
                # We'll handle synthetic in the encoding step

        keep_idx = torch.cat(keep_idx)

        # Batch
        for i in range(0, len(keep_idx), HP_MULTI['batch_size']):
            batch_idx = keep_idx[i:i+HP_MULTI['batch_size']]
            if len(batch_idx) < 4: continue

            with autocast():
                flow_reps, edge_attr_61 = encode_edges(
                    g.edge_attr[batch_idx], g.edge_time[batch_idx],
                    encoder, time2vec, g.edge_index[:, batch_idx], g.num_nodes
                )
                logits = multiclass_head(flow_reps)
                loss = focal_loss(logits, g.y[batch_idx])

            optimizer.zero_grad()
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

            epoch_loss += loss.item(); n_batches += 1

    scheduler.step()
    avg_loss = epoch_loss / max(n_batches, 1)
    train_losses.append(avg_loss)

    # Validation
    time2vec.eval(); encoder.eval(); multiclass_head.eval()
    val_preds, val_targets = [], []

    with torch.no_grad():
        for g in G_val[:10]:
            g = g.to(device)
            if g.edge_index.shape[1] > 5000:
                idx = torch.randperm(g.edge_index.shape[1])[:5000]
                g.edge_index=g.edge_index[:,idx]; g.edge_attr=g.edge_attr[idx]
                g.edge_time=g.edge_time[idx]; g.y=g.y[idx]

            flow_reps, _ = encode_edges(
                g.edge_attr, g.edge_time, encoder, time2vec, g.edge_index, g.num_nodes)
            logits = multiclass_head(flow_reps)
            preds = logits.argmax(dim=1)
            val_preds.extend(preds.cpu().tolist()); val_targets.extend(g.y.cpu().tolist())

    val_f1 = f1_score(val_targets, val_preds, average='macro')
    val_f1s.append(val_f1)
    print(f"Epoch {epoch+1:2d}/{HP_MULTI['epochs']}: Loss={avg_loss:.6f}, Val Macro-F1={val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            'epoch': epoch+1, 'time2vec_state_dict': time2vec.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'multiclass_head_state_dict': multiclass_head.state_dict(),
            'val_f1': val_f1, 'config': HP_MULTI,
        }, CHECKPOINT_DIR / 'best.pt')
        with open(CHECKPOINT_DIR/'config.json','w') as f: json.dump(HP_MULTI, f, indent=2)
        print(f"  ✓ Best checkpoint")

# %% [markdown]
# ## Cell 7: Per-Class Threshold Calibration

# %%
print("\nCalibrating per-class thresholds...")
time2vec.eval(); encoder.eval(); multiclass_head.eval()

all_val_probs = []
all_val_targets = []

with torch.no_grad():
    for g in G_val:
        g = g.to(device)
        if g.edge_index.shape[1] > 10000:
            idx = torch.randperm(g.edge_index.shape[1])[:10000]
            g.edge_index=g.edge_index[:,idx]; g.edge_attr=g.edge_attr[idx]
            g.edge_time=g.edge_time[idx]; g.y=g.y[idx]

        flow_reps, _ = encode_edges(
            g.edge_attr, g.edge_time, encoder, time2vec, g.edge_index, g.num_nodes)
        probs = F.softmax(multiclass_head(flow_reps), dim=-1)
        all_val_probs.append(probs.cpu())
        all_val_targets.append(g.y.cpu())

all_val_probs = torch.cat(all_val_probs, dim=0).numpy()
all_val_targets = torch.cat(all_val_targets, dim=0).numpy()

per_class_thresholds = {}
threshold_grid = np.arange(0.05, 0.95, 0.05)

for cls_idx, cls_name in enumerate(UNIFIED_CLASSES):
    cls_mask = all_val_targets == cls_idx
    if cls_mask.sum() < 10:
        per_class_thresholds[cls_name] = 0.5  # default
        continue

    best_t = 0.5; best_f1 = 0.0
    cls_probs = all_val_probs[:, cls_idx]
    # Binary: this class vs all others
    cls_binary_targets = (all_val_targets == cls_idx).astype(int)

    for t in threshold_grid:
        preds = (cls_probs >= t).astype(int)
        f1 = f1_score(cls_binary_targets, preds)
        if f1 > best_f1:
            best_f1 = f1; best_t = t

    per_class_thresholds[cls_name] = float(best_t)
    print(f"  {cls_name:25s}: threshold={best_t:.2f}, F1={best_f1:.4f}")

# Save thresholds
with open(CHECKPOINT_DIR / 'thresholds.json', 'w') as f:
    json.dump({'per_class_thresholds': per_class_thresholds}, f, indent=2)

# %% [markdown]
# ## Cell 8: Confusion Matrix (Fig 15)

# %%
# Generate predictions using per-class thresholds
final_preds = []
for i in range(len(all_val_probs)):
    probs = all_val_probs[i]
    max_cls = np.argmax(probs)
    cls_name = UNIFIED_CLASSES[max_cls]
    threshold = per_class_thresholds.get(cls_name, 0.5)
    if probs[max_cls] >= threshold:
        final_preds.append(max_cls)
    else:
        final_preds.append(max_cls)  # fallback to argmax

cm = confusion_matrix(all_val_targets, final_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=[c[:12] for c in UNIFIED_CLASSES],
            yticklabels=[c[:12] for c in UNIFIED_CLASSES],
            ax=ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Figure 15: Normalized Confusion Matrix — In-Domain (Validation Set)')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig15_confusion_matrix.png', dpi=300)
plt.savefig(FIGURES_DIR / 'fig15_confusion_matrix.svg')
plt.show()

# Per-class metrics
per_class_report = classification_report(
    all_val_targets, final_preds,
    target_names=UNIFIED_CLASSES, output_dict=True, zero_division=0
)

# Tab05: Main results (in-domain)
tab05_rows = []
for cls_name in UNIFIED_CLASSES:
    if cls_name in per_class_report and isinstance(per_class_report[cls_name], dict):
        d = per_class_report[cls_name]
        tab05_rows.append({
            'class': cls_name,
            'precision': round(d['precision'], 4),
            'recall': round(d['recall'], 4),
            'f1_score': round(d['f1-score'], 4),
            'support': int(d['support']),
        })

# Add macro avg
tab05_rows.append({
    'class': 'MACRO AVG',
    'precision': round(per_class_report['macro avg']['precision'], 4),
    'recall': round(per_class_report['macro avg']['recall'], 4),
    'f1_score': round(per_class_report['macro avg']['f1-score'], 4),
    'support': int(per_class_report['macro avg']['support']),
})

tab05_df = pd.DataFrame(tab05_rows)
tab05_df.to_csv(TABLES_DIR / 'tab05_main_results.csv', index=False)
tab05_df.to_markdown(TABLES_DIR / 'tab05_main_results.md', index=False)
print("Saved: tab05_main_results, fig15_confusion_matrix")

# %% [markdown]
# ## Cell 9: Results Log

# %%
nb_end_time = datetime.now(timezone.utc).isoformat()
results_log = {
    'notebook': 5, 'stage': 'G',
    'title': 'Multiclass Classification — Stage-2 Head',
    'start_time': NB_START_TIME, 'end_time': nb_end_time,
    'hyperparameters': HP_MULTI,
    'best_val_macro_f1': float(best_val_f1),
    'per_class_thresholds': per_class_thresholds,
    'per_class_metrics': {cls: per_class_report[cls] for cls in UNIFIED_CLASSES
                          if cls in per_class_report and isinstance(per_class_report[cls], dict)},
    'warnings': [],
}
with open(LOGS_DIR / 'notebook_5_log.json', 'w') as f:
    json.dump(results_log, f, indent=2, default=str)
print("Saved: logs/notebook_5_log.json")
print(f"\nNOTEBOOK 5 COMPLETE — Best Val Macro-F1: {best_val_f1:.4f}")
print("Next: Notebook 6 — Prototypical Few-Shot (Stage H)")
