"""
Notebook 3 — CVAE Minority-Class Augmentation (Stage E)
=========================================================
Kaggle T4x2. Run cells sequentially.

Inputs:
  - Pretrained encoder checkpoint: checkpoints/D_mae_pretrain/best.pt (from NB2)
  - Windowed graphs with labels (from NB1)
  - feature_manifest.yaml, label_map.yaml

Outputs:
  - CVAE checkpoint: checkpoints/E_cvae/best.pt + config.json
  - Synthetic embedding pool: checkpoints/E_cvae/synthetic_embeddings.pt
  - fig06_cvae_diagram, fig08_class_distribution
  - logs/notebook_3_log.json
"""

# %% [markdown]
# # Notebook 3: CVAE Minority-Class Augmentation (Stage E)
#
# **Target:** Kaggle T4x2 GPU
# **Duration:** ~1-2 hours
#
# ## Pipeline Position
# ```
# Pretrained Encoder (NB2) → [NB3: CVAE] → Synthetic Embeddings → [NB5: Multiclass]
# ```

# %% [markdown]
# ## Cell 1: Imports & Configuration

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
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
import seaborn as sns
from sklearn.decomposition import PCA

# %% [markdown]
# ## Cell 2: Seed & Paths

# %%
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

WORKING_DIR = Path('/kaggle/working')
CHECKPOINT_DIR = WORKING_DIR / 'checkpoints' / 'E_cvae'
LOGS_DIR = WORKING_DIR / 'logs'
FIGURES_DIR = WORKING_DIR / 'outputs' / 'figures'
ARTIFACTS_DIR = WORKING_DIR / 'artifacts'

for d in [CHECKPOINT_DIR, LOGS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NB_START_TIME = datetime.now(timezone.utc).isoformat()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# %% [markdown]
# ## Cell 3: Load Artifacts

# %%
# Load feature manifest and label map
with open(ARTIFACTS_DIR / 'feature_manifest.yaml', 'r') as f:
    feature_manifest = yaml.safe_load(f)
with open(ARTIFACTS_DIR / 'label_map.yaml', 'r') as f:
    label_map = yaml.safe_load(f)

UNIFIED_CLASSES = label_map['unified_classes']
KEPT_FEATURES = feature_manifest['kept_features']
EDGE_INPUT_DIM = feature_manifest['final_edge_input_dim']  # 61
ENCODER_OUTPUT_DIM = 768
N_CLASSES = len(UNIFIED_CLASSES)  # 11

print(f"Unified classes ({N_CLASSES}): {UNIFIED_CLASSES}")
print(f"Encoder output dim: {ENCODER_OUTPUT_DIM}")

# %% [markdown]
# ## Cell 4: Load Encoder & Time2Vec (Frozen)

# %%
# Re-import encoder and Time2Vec from NB2 definitions
# In Kaggle, you'd copy the class definitions or import from a shared module
# For simplicity, we redefine them here (or import from a .py file)

# --- Copy class definitions from NB2 ---
class Time2Vec(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        self.k = k
        self.w0 = nn.Parameter(torch.randn(1) * 0.1)
        self.b0 = nn.Parameter(torch.zeros(1))
        log_omega = torch.rand(k) * 6 - 3
        omega_init = 10.0 ** log_omega
        self.omega = nn.Parameter(omega_init)
        self.bias = nn.Parameter(torch.zeros(k))
        self.output_dim = k + 1

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        linear = self.w0 * t + self.b0
        periodic = torch.sin(self.omega * t + self.bias)
        return torch.cat([linear, periodic], dim=-1)

class EGATv2Encoder(nn.Module):
    def __init__(self, edge_dim=61, node_init_dim=128, hidden_dim=256,
                 num_heads=8, num_layers=3, dropout_attn=0.3, dropout_feat=0.2):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = hidden_dim * 3
        self.node_init_dim = node_init_dim
        self.node_embed = None
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_feat)
        for i in range(num_layers):
            self.convs.append(
                GATv2Conv((-1, -1), hidden_dim // num_heads, heads=num_heads,
                          edge_dim=hidden_dim, dropout=dropout_attn, concat=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.activation = nn.ELU()

    def _get_node_embed(self, num_nodes, device):
        if self.node_embed is None or self.node_embed.shape[0] < num_nodes:
            new_embed = nn.Parameter(torch.randn(num_nodes, self.node_init_dim, device=device) * 0.1)
            if self.node_embed is not None:
                new_embed.data[:self.node_embed.shape[0]] = self.node_embed.data
            self.node_embed = new_embed
        return self.node_embed[:num_nodes]

    def forward(self, data):
        x = self._get_node_embed(data.num_nodes, data.edge_index.device)
        edge_attr = self.edge_proj(data.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            x_new, _ = conv(x, data.edge_index, edge_attr=edge_attr, return_attention_weights=True)
            x_new = self.activation(x_new)
            x_new = self.dropout(x_new)
            x_new = norm(x_new)
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
        src_embeds = x[data.edge_index[0]]
        dst_embeds = x[data.edge_index[1]]
        return torch.cat([src_embeds, dst_embeds, edge_attr], dim=-1)
# --- End class definitions ---

# Load checkpoint
checkpoint_path = WORKING_DIR / 'checkpoints' / 'D_mae_pretrain' / 'best.pt'
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

time2vec = Time2Vec(k=16).to(device)
encoder = EGATv2Encoder(
    edge_dim=EDGE_INPUT_DIM, node_init_dim=128, hidden_dim=256,
    num_heads=8, num_layers=3
).to(device)

time2vec.load_state_dict(ckpt['time2vec_state_dict'])
encoder.load_state_dict(ckpt['encoder_state_dict'])

# Freeze both
for p in time2vec.parameters():
    p.requires_grad = False
for p in encoder.parameters():
    p.requires_grad = False

time2vec.eval()
encoder.eval()

print(f"Loaded pretrained encoder from epoch {ckpt['epoch']}")
print(f"Encoder val loss: {ckpt['val_loss']:.6f}")

# %% [markdown]
# ## Cell 5: Extract Embeddings & Compute Class Distribution

# %%
# Load the time normalizer from NB2
# (In practice, you'd load TIME_MIN/TIME_MAX from NB2's checkpoint, but
# for simplicity we recompute from training graphs here)
G_train_2018 = torch.load(WORKING_DIR / 'G_NF-CICIDS2018_train_list.pt', weights_only=False)
G_train_unsw = torch.load(WORKING_DIR / 'G_NF-UNSW-NB15_train_list.pt', weights_only=False)
G_train = G_train_2018 + G_train_unsw

# Compute time range for normalization
all_times = torch.cat([g.edge_time for g in G_train])
TIME_MIN = all_times.min().item()
TIME_MAX = all_times.max().item()

def normalize_time(t):
    return (t - TIME_MIN) / (TIME_MAX - TIME_MIN)

# Extract 768-dim embeddings for all training flows
print("Extracting training embeddings...")
all_embeddings = []
all_labels = []
all_class_names = []
label_to_idx = {name: i for i, name in enumerate(UNIFIED_CLASSES)}

with torch.no_grad():
    for g in G_train:
        g = g.to(device)
        # Time2Vec
        t_norm = normalize_time(g.edge_time)
        t_embed = time2vec(t_norm)
        edge_features_61 = torch.cat([g.edge_attr, t_embed], dim=-1)

        # Forward pass
        g_data = torch_geometric.data.Data(
            edge_index=g.edge_index, edge_attr=edge_features_61, num_nodes=g.num_nodes
        )
        flow_reps = encoder(g_data)  # (E, 768)

        all_embeddings.append(flow_reps.cpu())
        all_labels.append(g.y.cpu())
        all_class_names.extend([UNIFIED_CLASSES[l.item()] for l in g.y])

embeddings = torch.cat(all_embeddings, dim=0)  # (total_train_edges, 768)
labels = torch.cat(all_labels, dim=0)            # (total_train_edges,)

print(f"Total training embeddings: {embeddings.shape[0]:,} × {embeddings.shape[1]}")
print(f"Memory: {embeddings.element_size() * embeddings.numel() / 1e9:.2f} GB")

# Compute class distribution
class_counts = Counter(all_class_names)
print("\nClass distribution (pre-augmentation):")
for cls in UNIFIED_CLASSES:
    count = class_counts.get(cls, 0)
    pct = count / len(all_class_names) * 100
    bar = '█' * int(pct * 2)
    print(f"  {cls:25s}: {count:8,} ({pct:5.2f}%) {bar}")

# Identify minority classes (below median count)
counts_sorted = sorted(class_counts.items(), key=lambda x: x[1])
median_count = np.median([c for _, c in counts_sorted])
minority_classes = [cls for cls, count in counts_sorted if count < median_count]
majority_count = max(class_counts.values())

print(f"\nMedian class count: {median_count:.0f}")
print(f"Majority class count: {majority_count:,}")
print(f"Minority classes (below median): {minority_classes}")
print(f"Target per minority class: ~{int(majority_count * 0.4):,} (40% of majority)")

# %% [markdown]
# ## Cell 6: CVAE Model Definition

# %%
class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for minority-class augmentation.
    Condition = 11-dim one-hot class label.
    """
    def __init__(self, input_dim=768, condition_dim=11, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        in_dim = input_dim + condition_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ELU(),
            nn.Dropout(0.1),
        )
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 128),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(128, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x, c):
        """x: (batch, 768), c: (batch, 11)"""
        h = self.encoder(torch.cat([x, c], dim=-1))
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        """z: (batch, 64), c: (batch, 11)"""
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

    def generate(self, c, device='cpu'):
        """Generate synthetic embeddings for given class conditions."""
        z = torch.randn(c.shape[0], self.latent_dim, device=device)
        return self.decode(z, c)

# Test
cvae = ConditionalVAE(input_dim=768, condition_dim=11, hidden_dim=256, latent_dim=64)
test_x = torch.randn(32, 768)
test_c = F.one_hot(torch.randint(0, 11, (32,)), num_classes=11).float()
recon, mu, logvar = cvae(test_x, test_c)
print(f"CVAE test: input {test_x.shape} → recon {recon.shape}, latent mu {mu.shape}")
assert recon.shape == (32, 768)
print("CVAE construction OK ✓")

# %% [markdown]
# ## Cell 7: Prepare Minority-Class Training Data

# %%
# Filter embeddings to minority classes only
minority_mask = torch.tensor([UNIFIED_CLASSES[l.item()] in minority_classes for l in labels])
minority_embeddings_np = embeddings[minority_mask].numpy()
minority_labels = labels[minority_mask]

print(f"Minority class embeddings: {minority_embeddings_np.shape[0]:,}")

# Normalize embeddings for CVAE training (helps convergence)
from sklearn.preprocessing import StandardScaler
embed_scaler = StandardScaler()
minority_embeddings_norm = embed_scaler.fit_transform(minority_embeddings_np)

# Convert to tensors
X_minority = torch.tensor(minority_embeddings_norm, dtype=torch.float32)
y_minority = torch.tensor([l.item() for l in minority_labels], dtype=torch.long)

print(f"X_minority: {X_minority.shape}, y_minority: {y_minority.shape}")

# %% [markdown]
# ## Cell 8: CVAE Training Loop

# %%
HP_CVAE = {
    'lr': 5e-4,
    'epochs': 50,
    'batch_size': 512,
    'beta': 0.5,  # β-KL weight
    'latent_dim': 64,
}

cvae = ConditionalVAE(input_dim=768, condition_dim=11, hidden_dim=256, latent_dim=64).to(device)
optimizer = optim.Adam(cvae.parameters(), lr=HP_CVAE['lr'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HP_CVAE['epochs'])

print(f"CVAE Training: {HP_CVAE['epochs']} epochs, β={HP_CVAE['beta']}")
print(f"Parameters: {sum(p.numel() for p in cvae.parameters()):,}")

train_losses = []
kl_losses = []
recon_losses = []

for epoch in range(HP_CVAE['epochs']):
    cvae.train()
    epoch_loss = 0.0
    epoch_kl = 0.0
    epoch_recon = 0.0
    n_batches = 0

    # Shuffle
    perm = torch.randperm(X_minority.shape[0])
    X_shuffled = X_minority[perm]
    y_shuffled = y_minority[perm]

    for i in range(0, X_minority.shape[0], HP_CVAE['batch_size']):
        x_batch = X_shuffled[i:i+HP_CVAE['batch_size']].to(device)
        y_batch = y_shuffled[i:i+HP_CVAE['batch_size']].to(device)

        # One-hot condition
        c_batch = F.one_hot(y_batch, num_classes=11).float()

        # Forward
        recon, mu, logvar = cvae(x_batch, c_batch)

        # Losses
        recon_loss = F.mse_loss(recon, x_batch)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + HP_CVAE['beta'] * kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_kl += kl_loss.item()
        epoch_recon += recon_loss.item()
        n_batches += 1

    scheduler.step()

    avg_loss = epoch_loss / n_batches
    avg_kl = epoch_kl / n_batches
    avg_recon = epoch_recon / n_batches

    train_losses.append(avg_loss)
    kl_losses.append(avg_kl)
    recon_losses.append(avg_recon)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{HP_CVAE['epochs']}: "
              f"Loss={avg_loss:.6f}, Recon={avg_recon:.6f}, KL={avg_kl:.6f}")

    # KL collapse check
    if avg_kl < 1e-4 and epoch > 20:
        print(f"  WARNING: KL loss near zero ({avg_kl:.2e}) — possible posterior collapse")

print(f"\nTraining complete.")
print(f"Final: Recon={recon_losses[-1]:.6f}, KL={kl_losses[-1]:.6f}")

# Save checkpoint
cvae_ckpt = {
    'model_state_dict': cvae.state_dict(),
    'embed_scaler_mean': embed_scaler.mean_.tolist(),
    'embed_scaler_scale': embed_scaler.scale_.tolist(),
    'config': HP_CVAE,
    'final_losses': {'recon': float(recon_losses[-1]), 'kl': float(kl_losses[-1])},
}
torch.save(cvae_ckpt, CHECKPOINT_DIR / 'best.pt')
with open(CHECKPOINT_DIR / 'config.json', 'w') as f:
    json.dump(HP_CVAE, f, indent=2)
print(f"Checkpoint saved: {CHECKPOINT_DIR / 'best.pt'}")

# %% [markdown]
# ## Cell 9: Generate Synthetic Embeddings

# %%
# Compute how many synthetic samples each minority class needs
synth_counts = {}
for cls in minority_classes:
    current = class_counts.get(cls, 0)
    target = int(majority_count * 0.4)
    needed = max(0, target - current)
    synth_counts[cls] = needed
    print(f"  {cls}: current={current:,}, target={target:,}, needed={needed:,}")

cvae.eval()
synthetic_embeddings = []
synthetic_labels = []
synthetic_class_names = []

with torch.no_grad():
    for cls_name, count in synth_counts.items():
        if count == 0:
            continue
        cls_idx = UNIFIED_CLASSES.index(cls_name)
        c = F.one_hot(torch.full((count,), cls_idx, dtype=torch.long), num_classes=11).float().to(device)

        # Generate in batches to avoid OOM
        gen_batch = 1024
        gen_embeds = []
        for j in range(0, count, gen_batch):
            c_batch = c[j:j+gen_batch]
            gen = cvae.generate(c_batch, device=device)
            gen_embeds.append(gen.cpu())
        gen_all = torch.cat(gen_embeds, dim=0)[:count]  # trim to exact count

        # Inverse-transform from normalized to original embedding space
        gen_all = torch.tensor(embed_scaler.inverse_transform(gen_all.numpy()), dtype=torch.float32)

        synthetic_embeddings.append(gen_all)
        synthetic_labels.extend([cls_idx] * count)
        synthetic_class_names.extend([cls_name] * count)
        print(f"  {cls_name}: generated {count:,} synthetic embeddings")

# Concatenate
synth_embeddings_all = torch.cat(synthetic_embeddings, dim=0)
synth_labels_all = torch.tensor(synthetic_labels, dtype=torch.long)

# Create metadata: is_synthetic=True for filtering
synth_metadata = {
    'embeddings': synth_embeddings_all,
    'labels': synth_labels_all,
    'class_names': synthetic_class_names,
    'is_synthetic': [True] * len(synthetic_class_names),
    'counts_per_class': synth_counts,
}

torch.save(synth_metadata, CHECKPOINT_DIR / 'synthetic_embeddings.pt')
print(f"\nTotal synthetic embeddings: {synth_embeddings_all.shape[0]:,}")
print(f"Saved: {CHECKPOINT_DIR / 'synthetic_embeddings.pt'}")

# %% [markdown]
# ## Cell 10: Class Distribution Visualization (Fig 08)

# %%
# Pre vs post augmentation
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

classes_ordered = [cls for cls in UNIFIED_CLASSES if class_counts.get(cls, 0) > 0]
pre_counts = [class_counts.get(cls, 0) for cls in classes_ordered]
post_counts = [class_counts.get(cls, 0) + synth_counts.get(cls, 0) for cls in classes_ordered]

# Pre-augmentation
axes[0].bar(range(len(classes_ordered)), pre_counts, color='#2196F3', edgecolor='black', linewidth=0.5)
axes[0].set_xticks(range(len(classes_ordered)))
axes[0].set_xticklabels(classes_ordered, rotation=45, ha='right', fontsize=8)
axes[0].set_ylabel('Flow Count')
axes[0].set_title('Pre-Augmentation Class Distribution')
axes[0].set_yscale('log')
axes[0].grid(axis='y', alpha=0.3)

# Post-augmentation
axes[1].bar(range(len(classes_ordered)), post_counts, color='#4CAF50', edgecolor='black', linewidth=0.5)
axes[1].set_xticks(range(len(classes_ordered)))
axes[1].set_xticklabels(classes_ordered, rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel('Flow Count')
axes[1].set_title('Post-Augmentation Class Distribution (with CVAE synthetic)')
axes[1].set_yscale('log')
axes[1].grid(axis='y', alpha=0.3)

fig.suptitle('Figure 8: Class Distribution — Pre vs Post CVAE Augmentation', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig08_class_distribution.png', dpi=300)
plt.savefig(FIGURES_DIR / 'fig08_class_distribution.svg')
plt.show()
print("Saved: fig08_class_distribution")

# %% [markdown]
# ## Cell 11: CVAE Diagram (Fig 06)

# %%
fig, ax = plt.subplots(figsize=(12, 5))

components = [
    ('Input\n(768-dim\nembedding)', 0.1, 0.5, '#e3f2fd'),
    ('Condition\n(11-dim\none-hot)', 0.1, 0.2, '#fff3e0'),
    ('Encoder\n256→128', 0.35, 0.5, '#c8e6c9'),
    ('μ, σ\n(64-dim\nlatent)', 0.55, 0.5, '#f3e5f5'),
    ('z ~ N(μ,σ)\n(reparam)', 0.75, 0.5, '#ede7f6'),
    ('Decoder\n128→256→768', 0.75, 0.2, '#ffccbc'),
    ('Reconstructed\nEmbedding\n(768-dim)', 0.92, 0.35, '#b3e5fc'),
]

for label, x, y, color in components:
    bbox = dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='black', linewidth=1.2)
    ax.text(x, y, label, ha='center', va='center', fontsize=7, bbox=bbox, transform=ax.transAxes)

# Arrows
arrows = [
    (0.2, 0.5, 0.3, 0.5),  # input → encoder
    (0.2, 0.25, 0.3, 0.42),  # condition → encoder
    (0.45, 0.5, 0.5, 0.5),  # encoder → latent
    (0.65, 0.5, 0.7, 0.5),  # latent → reparam
    (0.7, 0.45, 0.7, 0.3),  # reparam → decoder
    (0.2, 0.25, 0.7, 0.3),  # condition → decoder (skip)
    (0.85, 0.3, 0.88, 0.35),  # decoder → output
]

for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Figure 6: Conditional VAE Architecture for Minority-Class Augmentation', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig06_cvae_diagram.png', dpi=300)
plt.savefig(FIGURES_DIR / 'fig06_cvae_diagram.svg')
plt.show()
print("Saved: fig06_cvae_diagram")

# %% [markdown]
# ## Cell 12: Synthetic Quality Check (PCA Visualization)

# %%
# Quick PCA to verify synthetic embeddings don't look completely wrong
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PCA on real minority + synthetic for one minority class
sample_cls = minority_classes[0]
cls_idx = UNIFIED_CLASSES.index(sample_cls)

real_mask = torch.tensor([UNIFIED_CLASSES[l.item()] == sample_cls for l in labels])
real_for_cls = embeddings[real_mask].numpy()

synth_mask = torch.tensor([c == sample_cls for c in synthetic_class_names])
synth_for_cls = synth_embeddings_all[synth_mask].numpy()

combined = np.concatenate([real_for_cls, synth_for_cls], axis=0)
combined_pca = PCA(n_components=2).fit_transform(combined)

n_real = real_for_cls.shape[0]
axes[0].scatter(combined_pca[:n_real, 0], combined_pca[:n_real, 1],
                c='blue', alpha=0.3, s=1, label=f'Real {sample_cls}')
axes[0].scatter(combined_pca[n_real:, 0], combined_pca[n_real:, 1],
                c='red', alpha=0.3, s=1, label=f'Synthetic {sample_cls}')
axes[0].set_title(f'Real vs Synthetic: {sample_cls}')
axes[0].legend(markerscale=5)
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# Per-class PCA for all real classes (subsampled)
sample_per_class = 200
pca_data = []
pca_labels = []
for cls in UNIFIED_CLASSES:
    cls_mask = torch.tensor([UNIFIED_CLASSES[l.item()] == cls for l in labels])
    cls_embeds = embeddings[cls_mask].numpy()
    if len(cls_embeds) > sample_per_class:
        idx = np.random.choice(len(cls_embeds), sample_per_class, replace=False)
        cls_embeds = cls_embeds[idx]
    pca_data.append(cls_embeds)
    pca_labels.extend([cls] * len(cls_embeds))

pca_all = PCA(n_components=2).fit_transform(np.concatenate(pca_data, axis=0))
offset = 0
for cls in UNIFIED_CLASSES:
    cls_n = pca_labels.count(cls)
    if cls_n > 0:
        axes[1].scatter(pca_all[offset:offset+cls_n, 0], pca_all[offset:offset+cls_n, 1],
                        alpha=0.4, s=2, label=cls)
        offset += cls_n
axes[1].set_title('Real Embeddings by Class (PCA)')
axes[1].legend(fontsize=6, markerscale=5, loc='upper right')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')

fig.suptitle('CVAE Synthetic Quality Check — PCA Visualization', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_cvae_quality_pca.png', dpi=150)
plt.show()
print("Saved: PCA quality check")

# %% [markdown]
# ## Cell 13: Results Log & Summary

# %%
nb_end_time = datetime.now(timezone.utc).isoformat()

results_log = {
    'notebook': 3,
    'stage': 'E',
    'title': 'CVAE Minority-Class Augmentation',
    'start_time': NB_START_TIME,
    'end_time': nb_end_time,
    'seed': SEED,
    'hyperparameters': HP_CVAE,
    'class_distribution': {
        'pre_augmentation': dict(class_counts),
        'post_augmentation': {cls: class_counts.get(cls, 0) + synth_counts.get(cls, 0)
                              for cls in UNIFIED_CLASSES},
    },
    'minority_classes': minority_classes,
    'majority_class_count': majority_count,
    'synthetic_counts': synth_counts,
    'total_synthetic_generated': int(synth_embeddings_all.shape[0]),
    'final_losses': {
        'reconstruction': float(recon_losses[-1]),
        'kl_divergence': float(kl_losses[-1]),
    },
    'kl_collapse_warning': kl_losses[-1] < 1e-4,
    'warnings': [],
}

if kl_losses[-1] < 1e-4:
    results_log['warnings'].append('Possible posterior collapse — KL loss near zero')

with open(LOGS_DIR / 'notebook_3_log.json', 'w') as f:
    json.dump(results_log, f, indent=2, default=str)
print("Saved: logs/notebook_3_log.json")

print("\n" + "="*60)
print("NOTEBOOK 3 COMPLETE")
print("="*60)
print(f"Synthetic embeddings generated: {synth_embeddings_all.shape[0]:,}")
print(f"Minority classes augmented: {minority_classes}")
print("\nNext: Notebook 4 — Binary Classification (Stage F)")
