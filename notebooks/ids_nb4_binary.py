"""
Notebook 4 — Binary Classification, Stage-1 Head (Stage F)
============================================================
Kaggle T4x2, fp16. Run cells sequentially.

Inputs:
  - Pretrained encoder checkpoint: checkpoints/D_mae_pretrain/best.pt (from NB2)
  - Windowed graphs with labels (from NB1)
  - feature_manifest.yaml, label_map.yaml

Outputs:
  - Stage F checkpoint: checkpoints/F_binary/best.pt + config.json
  - Calibrated decision threshold
  - logs/notebook_4_log.json
  - Training curves for fig16
"""

# %% [markdown]
# # Notebook 4: Binary Classification — Stage-1 Head (Stage F)
#
# **Pipeline Position:**
# ```
# Pretrained Encoder (NB2) → [NB4: Binary Head] → attack-flagged flows → [NB5: Multiclass]
# ```

# %% [markdown]
# ## Cell 1: Imports & Setup

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import yaml
import json
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)

# %% [markdown]
# ## Cell 2: Seed, Paths & Device

# %%
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

WORKING_DIR = Path('/kaggle/working')
CHECKPOINT_DIR = WORKING_DIR / 'checkpoints' / 'F_binary'
LOGS_DIR = WORKING_DIR / 'logs'
FIGURES_DIR = WORKING_DIR / 'outputs' / 'figures'
ARTIFACTS_DIR = WORKING_DIR / 'artifacts'

for d in [CHECKPOINT_DIR, LOGS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NB_START_TIME = datetime.now(timezone.utc).isoformat()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# %% [markdown]
# ## Cell 3: Load Artifacts & Model Definitions

# %%
# Load YAML files
with open(ARTIFACTS_DIR / 'feature_manifest.yaml', 'r') as f:
    feature_manifest = yaml.safe_load(f)
with open(ARTIFACTS_DIR / 'label_map.yaml', 'r') as f:
    label_map = yaml.safe_load(f)

UNIFIED_CLASSES = label_map['unified_classes']
EDGE_INPUT_DIM = feature_manifest['final_edge_input_dim']  # 61

# Re-define necessary classes (or import from shared module)
class Time2Vec(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        self.k = k
        self.w0 = nn.Parameter(torch.randn(1) * 0.1)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.omega = nn.Parameter(10.0 ** (torch.rand(k) * 6 - 3))
        self.bias = nn.Parameter(torch.zeros(k))
        self.output_dim = k + 1
    def forward(self, t):
        if t.dim() == 1: t = t.unsqueeze(-1)
        return torch.cat([self.w0 * t + self.b0, torch.sin(self.omega * t + self.bias)], dim=-1)

class EGATv2Encoder(nn.Module):
    def __init__(self, edge_dim=61, node_init_dim=128, hidden_dim=256,
                 num_heads=8, num_layers=3, dropout_attn=0.3, dropout_feat=0.2):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        self.hidden_dim = hidden_dim; self.num_heads = num_heads
        self.num_layers = num_layers; self.output_dim = hidden_dim * 3
        self.node_init_dim = node_init_dim
        self.node_embed = None
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_feat)
        for _ in range(num_layers):
            self.convs.append(GATv2Conv((-1, -1), hidden_dim // num_heads,
                heads=num_heads, edge_dim=hidden_dim, dropout=dropout_attn, concat=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.activation = nn.ELU()
    def _get_node_embed(self, num_nodes, device):
        if self.node_embed is None or self.node_embed.shape[0] < num_nodes:
            new = nn.Parameter(torch.randn(num_nodes, self.node_init_dim, device=device) * 0.1)
            if self.node_embed is not None:
                new.data[:self.node_embed.shape[0]] = self.node_embed.data
            self.node_embed = new
        return self.node_embed[:num_nodes]
    def forward(self, data):
        x = self._get_node_embed(data.num_nodes, data.edge_index.device)
        edge_attr = self.edge_proj(data.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            x_new, _ = conv(x, data.edge_index, edge_attr=edge_attr, return_attention_weights=True)
            x_new = self.activation(x_new); x_new = self.dropout(x_new); x_new = norm(x_new)
            x = x + x_new if x.shape == x_new.shape else x_new
        return torch.cat([x[data.edge_index[0]], x[data.edge_index[1]], edge_attr], dim=-1)

# Binary classifier head
class BinaryHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, bottleneck_dim=64, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, bottleneck_dim), nn.ELU(), nn.Dropout(0.2),
            nn.Linear(bottleneck_dim, num_classes),
        )
    def forward(self, x):
        return self.net(x)

# Load pretrained checkpoint
ckpt = torch.load(WORKING_DIR / 'checkpoints' / 'D_mae_pretrain' / 'best.pt',
                  map_location=device, weights_only=False)

time2vec = Time2Vec(k=16).to(device)
encoder = EGATv2Encoder(edge_dim=EDGE_INPUT_DIM).to(device)
time2vec.load_state_dict(ckpt['time2vec_state_dict'])
encoder.load_state_dict(ckpt['encoder_state_dict'])

print(f"Loaded pretrained encoder (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")

# %% [markdown]
# ## Cell 4: Load Training Data & Time Normalizer

# %%
G_train_2018 = torch.load(WORKING_DIR / 'G_NF-CICIDS2018_train_list.pt', weights_only=False)
G_train_unsw = torch.load(WORKING_DIR / 'G_NF-UNSW-NB15_train_list.pt', weights_only=False)
G_val_2018   = torch.load(WORKING_DIR / 'G_NF-CICIDS2018_val_list.pt', weights_only=False)
G_val_unsw   = torch.load(WORKING_DIR / 'G_NF-UNSW-NB15_val_list.pt', weights_only=False)

G_train = G_train_2018 + G_train_unsw
G_val   = G_val_2018 + G_val_unsw

# Time normalization
all_times = torch.cat([g.edge_time for g in G_train])
TIME_MIN = all_times.min().item()
TIME_MAX = all_times.max().item()
def normalize_time(t):
    return (t - TIME_MIN) / (TIME_MAX - TIME_MIN)

print(f"G_train: {len(G_train)} windows, G_val: {len(G_val)} windows")

# Compute class weights for focal loss
all_y_binary = torch.cat([g.y_binary for g in G_train])
n_benign = (all_y_binary == 0).sum().item()
n_attack = (all_y_binary == 1).sum().item()
alpha_benign = n_attack / (n_benign + n_attack)
alpha_attack = n_benign / (n_benign + n_attack)
alpha_weight = torch.tensor([alpha_benign, alpha_attack], device=device)
print(f"Benign: {n_benign:,}, Attack: {n_attack:,}")
print(f"Focal alpha: [{alpha_benign:.4f}, {alpha_attack:.4f}]")

# %% [markdown]
# ## Cell 5: Focal Loss & PGD Attack

# %%
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # (num_classes,) tensor
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            focal = self.alpha[targets] * focal
        return focal.mean()

def pgd_attack(model, data, time2vec, epsilon=0.03, alpha=0.01, steps=7, feat_mins=None, feat_maxs=None):
    """
    Generate PGD-adversarial examples on edge features.
    Returns: perturbed edge_attr (61-dim)
    """
    edge_attr = data.edge_attr.clone().detach()
    edge_attr_orig = edge_attr.clone()

    for _ in range(steps):
        edge_attr = edge_attr.clone().detach().requires_grad_(True)
        data_pert = torch_geometric.data.Data(
            edge_index=data.edge_index, edge_attr=edge_attr, num_nodes=data.num_nodes
        )
        flow_reps = model(data_pert, time2vec, data.edge_time)
        # Maximize loss (away from correct prediction)
        # For simplicity, use a random target as adversarial direction
        loss = -F.mse_loss(flow_reps, flow_reps.detach() + torch.randn_like(flow_reps) * 0.01)
        grad = torch.autograd.grad(loss, edge_attr)[0]
        edge_attr = edge_attr.detach() + alpha * grad.sign()
        # Project to epsilon-ball
        delta = torch.clamp(edge_attr - edge_attr_orig, -epsilon, epsilon)
        edge_attr = edge_attr_orig + delta
        # Clip to valid range
        if feat_mins is not None and feat_maxs is not None:
            edge_attr = torch.clamp(edge_attr, feat_mins, feat_maxs)

    return edge_attr.detach()

print("FocalLoss and PGD attack defined ✓")

# %% [markdown]
# ## Cell 6: Training — Phase A (Frozen Encoder)

# %%
HP_BINARY = {
    'phase_a_lr': 1e-3, 'phase_a_epochs': 5,
    'phase_b_lr_encoder': 1e-5, 'phase_b_lr_head': 1e-4, 'phase_b_epochs': 15,
    'focal_gamma': 2.0,
    'pgd_epsilon': 0.03, 'pgd_alpha': 0.01, 'pgd_steps': 7,
    'pgd_batch_fraction': 0.30,
    'batch_size': 4096,
    'undersample_ratio': 2.0,  # benign:attack = 2:1
}

binary_head = BinaryHead(input_dim=768).to(device)

# Compute feature bounds for PGD clipping
all_train_attr = torch.cat([g.edge_attr for g in G_train])
feat_mins = all_train_attr.min(dim=0).values.to(device)
feat_maxs = all_train_attr.max(dim=0).values.to(device)
# Pad for 61-dim (44 raw + 17 Time2Vec — Time2Vec values clipped to ±4)
feat_mins_61 = torch.cat([feat_mins, torch.full((17,), -4.0, device=device)])
feat_maxs_61 = torch.cat([feat_maxs, torch.full((17,), 4.0, device=device)])

# Helper: encoder forward with Time2Vec
def encode_edges(edge_attr_44, edge_time, model, t2v, edge_index, num_nodes):
    t_norm = normalize_time(edge_time)
    t_embed = t2v(t_norm)
    edge_attr_61 = torch.cat([edge_attr_44, t_embed], dim=-1)
    data = torch_geometric.data.Data(
        edge_index=edge_index, edge_attr=edge_attr_61, num_nodes=num_nodes
    )
    return model(data), edge_attr_61

# --- Phase A: Frozen encoder ---
print("="*60)
print("PHASE A: Train Binary Head (Frozen Encoder)")
print("="*60)

# Freeze encoder + Time2Vec
for p in time2vec.parameters(): p.requires_grad = False
for p in encoder.parameters(): p.requires_grad = False

optimizer_a = optim.Adam(binary_head.parameters(), lr=HP_BINARY['phase_a_lr'])
scheduler_a = optim.lr_scheduler.CosineAnnealingLR(optimizer_a, T_max=HP_BINARY['phase_a_epochs'])
focal_loss = FocalLoss(gamma=HP_BINARY['focal_gamma'], alpha=alpha_weight)
scaler_amp = GradScaler()

phase_a_losses = []

for epoch in range(HP_BINARY['phase_a_epochs']):
    binary_head.train()
    epoch_loss = 0.0; n_batches = 0

    for g in G_train:
        g = g.to(device)
        if g.edge_index.shape[1] < 4: continue

        # Per-epoch undersampling: 2:1 benign:attack
        benign_idx = (g.y_binary == 0).nonzero(as_tuple=True)[0]
        attack_idx = (g.y_binary == 1).nonzero(as_tuple=True)[0]

        n_attack_keep = attack_idx.shape[0]
        n_benign_keep = min(int(n_attack_keep * HP_BINARY['undersample_ratio']), benign_idx.shape[0])
        if n_benign_keep < 4: continue

        benign_keep = benign_idx[torch.randperm(benign_idx.shape[0])[:n_benign_keep]]
        keep_idx = torch.cat([benign_keep, attack_idx])

        # Batch in chunks of batch_size
        for i in range(0, len(keep_idx), HP_BINARY['batch_size']):
            batch_idx = keep_idx[i:i+HP_BINARY['batch_size']]
            if len(batch_idx) < 4: continue

            with autocast():
                flow_reps, _ = encode_edges(
                    g.edge_attr[batch_idx], g.edge_time[batch_idx],
                    encoder, time2vec, g.edge_index[:, batch_idx], g.num_nodes
                )
                logits = binary_head(flow_reps)
                loss = focal_loss(logits, g.y_binary[batch_idx])

            optimizer_a.zero_grad()
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer_a)
            scaler_amp.update()

            epoch_loss += loss.item(); n_batches += 1

    scheduler_a.step()
    avg_loss = epoch_loss / max(n_batches, 1)
    phase_a_losses.append(avg_loss)
    print(f"Phase A — Epoch {epoch+1}/{HP_BINARY['phase_a_epochs']}: Loss={avg_loss:.6f}")

# %% [markdown]
# ## Cell 7: Phase B — Joint Fine-Tuning with PGD

# %%
print("\n" + "="*60)
print("PHASE B: Joint Fine-Tune (Encoder + Head) with PGD")
print("="*60)

# Unfreeze encoder + Time2Vec
for p in time2vec.parameters(): p.requires_grad = True
for p in encoder.parameters(): p.requires_grad = True

optimizer_b = optim.Adam([
    {'params': time2vec.parameters(), 'lr': HP_BINARY['phase_b_lr_encoder']},
    {'params': encoder.parameters(), 'lr': HP_BINARY['phase_b_lr_encoder']},
    {'params': binary_head.parameters(), 'lr': HP_BINARY['phase_b_lr_head']},
])
scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=HP_BINARY['phase_b_epochs'])

phase_b_losses = []
phase_b_val_f1s = []
best_val_f1 = 0.0

for epoch in range(HP_BINARY['phase_b_epochs']):
    # Training
    time2vec.train(); encoder.train(); binary_head.train()
    epoch_loss = 0.0; n_batches = 0

    for g in G_train:
        g = g.to(device)
        if g.edge_index.shape[1] < 4: continue

        benign_idx = (g.y_binary == 0).nonzero(as_tuple=True)[0]
        attack_idx = (g.y_binary == 1).nonzero(as_tuple=True)[0]
        n_attack_keep = attack_idx.shape[0]
        n_benign_keep = min(int(n_attack_keep * HP_BINARY['undersample_ratio']), benign_idx.shape[0])
        if n_benign_keep < 4: continue
        benign_keep = benign_idx[torch.randperm(benign_idx.shape[0])[:n_benign_keep]]
        keep_idx = torch.cat([benign_keep, attack_idx])

        for i in range(0, len(keep_idx), HP_BINARY['batch_size']):
            batch_idx = keep_idx[i:i+HP_BINARY['batch_size']]
            if len(batch_idx) < 4: continue

            # PGD on 30% of batch
            n_pgd = int(len(batch_idx) * HP_BINARY['pgd_batch_fraction'])
            pgd_mask = torch.zeros(len(batch_idx), dtype=torch.bool)
            if n_pgd > 0:
                pgd_mask[:n_pgd] = True
                pgd_mask = pgd_mask[torch.randperm(len(pgd_mask))]

            edge_attr_44 = g.edge_attr[batch_idx].clone()
            edge_time = g.edge_time[batch_idx]
            edge_index = g.edge_index[:, batch_idx]

            with autocast():
                t_norm = normalize_time(edge_time)
                t_embed = time2vec(t_norm)
                edge_attr_61 = torch.cat([edge_attr_44, t_embed], dim=-1)

                # Apply PGD to subset
                if n_pgd > 0:
                    # Simplified PGD: perturb raw features
                    edge_attr_61_pert = edge_attr_61.clone()
                    for _ in range(HP_BINARY['pgd_steps']):
                        edge_attr_61_pert = edge_attr_61_pert.clone().detach().requires_grad_(True)
                        data_temp = torch_geometric.data.Data(
                            edge_index=edge_index, edge_attr=edge_attr_61_pert,
                            num_nodes=g.num_nodes
                        )
                        reps_temp = encoder(data_temp)
                        logits_temp = binary_head(reps_temp[pgd_mask])
                        loss_adv = -focal_loss(logits_temp, g.y_binary[batch_idx][pgd_mask].to(device))
                        grad = torch.autograd.grad(loss_adv, edge_attr_61_pert, retain_graph=True)[0]
                        with torch.no_grad():
                            edge_attr_61_pert[pgd_mask] += HP_BINARY['pgd_alpha'] * grad[pgd_mask].sign()
                            delta = torch.clamp(
                                edge_attr_61_pert[pgd_mask] - edge_attr_61[pgd_mask],
                                -HP_BINARY['pgd_epsilon'], HP_BINARY['pgd_epsilon']
                            )
                            edge_attr_61_pert[pgd_mask] = edge_attr_61[pgd_mask] + delta
                            edge_attr_61_pert = torch.clamp(edge_attr_61_pert, feat_mins_61, feat_maxs_61)
                    edge_attr_61 = edge_attr_61_pert

                data_batch = torch_geometric.data.Data(
                    edge_index=edge_index, edge_attr=edge_attr_61, num_nodes=g.num_nodes
                )
                flow_reps = encoder(data_batch)
                logits = binary_head(flow_reps)
                loss = focal_loss(logits, g.y_binary[batch_idx].to(device))

            optimizer_b.zero_grad()
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer_b)
            scaler_amp.update()

            epoch_loss += loss.item(); n_batches += 1

    scheduler_b.step()
    avg_loss = epoch_loss / max(n_batches, 1)
    phase_b_losses.append(avg_loss)

    # Validation
    time2vec.eval(); encoder.eval(); binary_head.eval()
    val_preds = []; val_targets = []

    with torch.no_grad():
        for g in G_val[:10]:  # Validate on subset for speed
            g = g.to(device)
            if g.edge_index.shape[1] > 5000:
                idx = torch.randperm(g.edge_index.shape[1])[:5000]
                g.edge_index = g.edge_index[:, idx]
                g.edge_attr = g.edge_attr[idx]
                g.edge_time = g.edge_time[idx]
                g.y_binary = g.y_binary[idx]

            flow_reps, _ = encode_edges(
                g.edge_attr, g.edge_time, encoder, time2vec,
                g.edge_index, g.num_nodes
            )
            logits = binary_head(flow_reps)
            preds = logits.argmax(dim=1)
            val_preds.extend(preds.cpu().tolist())
            val_targets.extend(g.y_binary.cpu().tolist())

    val_f1 = f1_score(val_targets, val_preds, average='macro')
    phase_b_val_f1s.append(val_f1)
    print(f"Phase B — Epoch {epoch+1}/{HP_BINARY['phase_b_epochs']}: "
          f"Loss={avg_loss:.6f}, Val Macro-F1={val_f1:.4f}")

    # Save best
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        checkpoint = {
            'epoch': epoch + 1,
            'time2vec_state_dict': time2vec.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'binary_head_state_dict': binary_head.state_dict(),
            'val_f1': val_f1,
            'config': HP_BINARY,
        }
        torch.save(checkpoint, CHECKPOINT_DIR / 'best.pt')
        with open(CHECKPOINT_DIR / 'config.json', 'w') as f:
            json.dump(HP_BINARY, f, indent=2)
        print(f"  ✓ Best checkpoint (val_f1={val_f1:.4f})")

# %% [markdown]
# ## Cell 8: Decision Threshold Calibration
#
# Tune threshold for attack-class recall ≥ 0.995 on validation set.

# %%
print("\nCalibrating decision threshold...")
time2vec.eval(); encoder.eval(); binary_head.eval()

all_val_probs = []
all_val_targets = []

with torch.no_grad():
    for g in G_val:
        g = g.to(device)
        if g.edge_index.shape[1] > 10000:
            idx = torch.randperm(g.edge_index.shape[1])[:10000]
            g.edge_index = g.edge_index[:, idx]
            g.edge_attr = g.edge_attr[idx]
            g.edge_time = g.edge_time[idx]
            g.y_binary = g.y_binary[idx]

        flow_reps, _ = encode_edges(
            g.edge_attr, g.edge_time, encoder, time2vec, g.edge_index, g.num_nodes
        )
        probs = F.softmax(binary_head(flow_reps), dim=-1)[:, 1]  # attack probability
        all_val_probs.extend(probs.cpu().tolist())
        all_val_targets.extend(g.y_binary.cpu().tolist())

all_val_probs = np.array(all_val_probs)
all_val_targets = np.array(all_val_targets)

# Grid search for threshold
best_threshold = 0.5
best_recall = 0.0
thresholds_test = np.arange(0.05, 0.95, 0.025)
results = []

for thresh in thresholds_test:
    preds = (all_val_probs >= thresh).astype(int)
    rec = recall_score(all_val_targets, preds)
    prec = precision_score(all_val_targets, preds, zero_division=0)
    f1 = f1_score(all_val_targets, preds)
    results.append({'threshold': thresh, 'recall': rec, 'precision': prec, 'f1': f1})

    if rec >= 0.995 and f1 > best_recall:
        best_recall = f1
        best_threshold = thresh

results_df = pd.DataFrame(results)

# If no threshold achieves recall >= 0.995, pick the closest
if best_recall == 0.0:
    best_idx = np.argmax([r['recall'] for r in results])
    best_threshold = results[best_idx]['threshold']
    print(f"WARNING: No threshold achieved recall ≥ 0.995. Best recall: {results[best_idx]['recall']:.4f} at threshold {best_threshold:.3f}")

print(f"\nCalibrated threshold: {best_threshold:.3f}")
print(f"At this threshold:")
best_result = [r for r in results if r['threshold'] == best_threshold][0]
print(f"  Recall:    {best_result['recall']:.4f}")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  F1:        {best_result['f1']:.4f}")

# Save threshold
with open(CHECKPOINT_DIR / 'threshold.json', 'w') as f:
    json.dump({'threshold': float(best_threshold), 'target_recall': 0.995}, f, indent=2)

# %% [markdown]
# ## Cell 9: Training Curves & Results Log

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
all_losses = phase_a_losses + phase_b_losses
ax1.plot(range(1, len(all_losses)+1), all_losses, 'b-', linewidth=1.5)
ax1.axvline(x=len(phase_a_losses)+0.5, color='r', linestyle='--', alpha=0.5, label='Phase A→B')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Binary Classifier Training Loss'); ax1.legend(); ax1.grid(alpha=0.3)

# Val F1
ax2.plot(range(HP_BINARY['phase_a_epochs']+1, HP_BINARY['phase_a_epochs']+len(phase_b_val_f1s)+1),
         phase_b_val_f1s, 'g-', linewidth=1.5)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Macro-F1')
ax2.set_title('Validation Macro-F1 (Phase B)'); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig16_binary_training_curve.png', dpi=300)
plt.savefig(FIGURES_DIR / 'fig16_binary_training_curve.svg')
plt.show()

# Results log
nb_end_time = datetime.now(timezone.utc).isoformat()
results_log = {
    'notebook': 4, 'stage': 'F',
    'title': 'Binary Classification — Stage-1 Head',
    'start_time': NB_START_TIME, 'end_time': nb_end_time,
    'hyperparameters': HP_BINARY,
    'best_val_f1': float(best_val_f1),
    'calibrated_threshold': float(best_threshold),
    'threshold_result': best_result,
    'epochs_phase_a': len(phase_a_losses),
    'epochs_phase_b': len(phase_b_losses),
    'warnings': [],
}
with open(LOGS_DIR / 'notebook_4_log.json', 'w') as f:
    json.dump(results_log, f, indent=2, default=str)

print("\n" + "="*60)
print("NOTEBOOK 4 COMPLETE")
print(f"Best Val F1: {best_val_f1:.4f}")
print(f"Threshold: {best_threshold:.3f}")
print("Next: Notebook 5 — Multiclass Classification (Stage G)")
