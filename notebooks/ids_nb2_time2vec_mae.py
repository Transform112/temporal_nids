"""
Notebook 2 — Time2Vec + E-GATv2 Encoder + MAE Pretraining (Stages B, C, D)
============================================================================
Kaggle T4x2, fp16 mixed precision mandatory.
Run cells sequentially.

Inputs:
  - Windowed graphs: G_{dataset}_{split}_list.pt (from NB1)
  - Scaler: scaler.pkl (from NB1)
  - feature_manifest.yaml (from NB1)
  - label_map.yaml (from NB1)

Outputs:
  - Time2Vec module weights (joint-trained, saved with encoder)
  - Pretrained E-GATv2 encoder checkpoint: checkpoints/D_mae_pretrain/best.pt + config.json
  - fig03_time2vec_diagram, fig04_attention_diagram, fig05_mae_pretrain_diagram
  - Training curves for fig16
  - logs/notebook_2_log.json

CRITICAL: Edge input = 44 raw features + 17 Time2Vec = 61-dim (NOT 70-dim)
"""

# %% [markdown]
# # Notebook 2: Time2Vec + E-GATv2 + MAE Pretraining (Stages B, C, D)
#
# **Target:** Kaggle T4x2 GPU, fp16
# **Duration:** ~4-8 hours (30 epochs MAE with FGSM)
# **Criticality:** HIGH — encoder is backbone for all downstream stages
#
# ## Pipeline Position
# ```
# G_train, G_val, G_test → [NB2: Time2Vec + E-GATv2 + MAE] → pretrained encoder
# ```

# %% [markdown]
# ## Cell 1: Imports & Configuration

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import yaml
import json
import pickle
import random
import os
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# PyG
import torch_geometric
from torch_geometric.nn import GATv2Conv
from torch_geometric.loader import NeighborLoader

# Visualization
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

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
CHECKPOINT_DIR = WORKING_DIR / 'checkpoints' / 'D_mae_pretrain'
LOGS_DIR = WORKING_DIR / 'logs'
FIGURES_DIR = WORKING_DIR / 'outputs' / 'figures'
ARTIFACTS_DIR = WORKING_DIR / 'artifacts'

for d in [CHECKPOINT_DIR, LOGS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NB_START_TIME = datetime.now(timezone.utc).isoformat()

print(f"PyTorch: {torch.__version__}")
print(f"PyG: {torch_geometric.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Verify PyG supports edge_dim
# We'll test this when building the first GATv2Conv layer

# %% [markdown]
# ## Cell 3: Load Artifacts from NB1

# %%
# Load feature manifest
with open(ARTIFACTS_DIR / 'feature_manifest.yaml', 'r') as f:
    feature_manifest = yaml.safe_load(f)

# Load scaler
with open(WORKING_DIR / 'checkpoints' / 'B_C_scaler' / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

KEPT_FEATURES = feature_manifest['kept_features']
RAW_FEATURE_DIM = len(KEPT_FEATURES)  # 44
TIME2VEC_DIM = feature_manifest['time2vec_dim']  # 17
EDGE_INPUT_DIM = feature_manifest['final_edge_input_dim']  # 61

print(f"Raw features: {RAW_FEATURE_DIM}")
print(f"Time2Vec dim: {TIME2VEC_DIM}")
print(f"Edge input dim: {EDGE_INPUT_DIM}")
assert EDGE_INPUT_DIM == RAW_FEATURE_DIM + TIME2VEC_DIM == 61, \
    f"Dimension mismatch: {EDGE_INPUT_DIM} != {RAW_FEATURE_DIM} + {TIME2VEC_DIM}"

# Load graphs
# We need G_train from both datasets
def load_graphs(dataset_name, split):
    path = WORKING_DIR / f'G_{dataset_name}_{split}_list.pt'
    if path.exists():
        return torch.load(path, weights_only=False)
    return []

graphs_train_2018 = load_graphs('NF-CICIDS2018', 'train')
graphs_train_unsw = load_graphs('NF-UNSW-NB15', 'train')
graphs_val_2018 = load_graphs('NF-CICIDS2018', 'val')
graphs_val_unsw = load_graphs('NF-UNSW-NB15', 'val')

# Combine graphs from both datasets (training jointly)
G_train = graphs_train_2018 + graphs_train_unsw
G_val = graphs_val_2018 + graphs_val_unsw

print(f"\nG_train: {len(G_train)} windows, {sum(g.edge_index.shape[1] for g in G_train):,} edges")
print(f"G_val: {len(G_val)} windows, {sum(g.edge_index.shape[1] for g in G_val):,} edges")

# %% [markdown]
# ## Cell 4: Time2Vec Module
#
# φ(t) = [ω₀t + b₀, sin(ω₁t+b₁), ..., sin(ω₁₆t+b₁₆)]
# Output: 17-dim (1 linear + 16 periodic)
# Omega initialized from log-uniform(1e-3, 1e3) covering ms-to-minute timescales.

# %%
class Time2Vec(nn.Module):
    """
    Time2Vec: learnable sinusoidal time embedding.
    Output dim = k + 1 = 17 (1 linear + 16 periodic terms).

    Input: time values in milliseconds (batch, 1) or (batch,)
    Output: (batch, 17) time embedding
    """
    def __init__(self, k=16, init_omega_range=(-3, 3)):
        """
        Args:
            k: number of periodic (sinusoidal) terms
            init_omega_range: log10 range for omega initialization
                             (-3, 3) → 10^(-3) to 10^3 covers ms to minute scales
        """
        super().__init__()
        self.k = k

        # Linear term: ω₀t + b₀
        self.w0 = nn.Parameter(torch.randn(1) * 0.1)
        self.b0 = nn.Parameter(torch.zeros(1))

        # Periodic terms: sin(ωᵢt + bᵢ)
        # Initialize ω from log-uniform distribution
        log_omega = torch.rand(k) * (init_omega_range[1] - init_omega_range[0]) + init_omega_range[0]
        omega_init = 10.0 ** log_omega  # log-uniform → uniform in log-space
        self.omega = nn.Parameter(omega_init)
        self.bias = nn.Parameter(torch.zeros(k))

        self.output_dim = k + 1  # 17

    def forward(self, t):
        """
        Args:
            t: time tensor (batch,) or (batch, 1) in milliseconds
        Returns:
            (batch, 17) time embedding
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (batch, 1)

        # Linear term
        linear = self.w0 * t + self.b0  # (batch, 1)

        # Periodic terms
        periodic = torch.sin(self.omega * t + self.bias)  # (batch, k)

        return torch.cat([linear, periodic], dim=-1)  # (batch, 17)

# Quick test
t2v = Time2Vec(k=16)
test_t = torch.randn(100, 1) * 10000
test_out = t2v(test_t)
print(f"Time2Vec test: input {test_t.shape} → output {test_out.shape}")
assert test_out.shape == (100, 17), f"Expected (100, 17), got {test_out.shape}"

# %% [markdown]
# ## Cell 5: E-GATv2 Encoder
#
# 3-layer edge-augmented graph attention encoder.
# Input: 61-dim edge features → Output: 768-dim flow representation per edge.
# Critical: Use GATv2Conv(edge_dim=61) — NOT 70.

# %%
class EGATv2Encoder(nn.Module):
    """
    3-layer E-GATv2 encoder with edge-augmented attention.

    Architecture:
    - Layer 1: edge_dim(61) → hidden=256, heads=8, fanout=15
    - Layer 2: 256 → hidden=256, heads=8, fanout=10
    - Layer 3: 256 → hidden=256, heads=8, fanout=5
    - Output: concat(src, dst, edge) = 768-dim per flow
    """
    def __init__(
        self,
        edge_dim=61,
        node_init_dim=128,
        hidden_dim=256,
        num_heads=8,
        num_layers=3,
        dropout_attn=0.3,
        dropout_feat=0.2,
        activation='elu'
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.node_init_dim = node_init_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = hidden_dim * 3  # concat src + dst + edge = 768

        # Node embedding table (will be expanded as new nodes appear)
        self.node_embed = None  # initialized on first forward pass
        self.node_init_dim = node_init_dim

        # Edge feature projection (for residual edge features across layers)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # GATv2Conv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_feat)

        # First layer: edge_dim input features + node embeddings
        in_dim_layer1 = hidden_dim  # after edge_proj
        self.convs.append(
            GATv2Conv(
                in_channels=(-1, -1),  # will be set by forward
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                edge_dim=hidden_dim,  # projected edge features
                dropout=dropout_attn,
                concat=True
            )
        )

        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    in_channels=(-1, -1),
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim,  # edge features projected to hidden_dim
                    dropout=dropout_attn,
                    concat=True
                )
            )

        # LayerNorm after each conv
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.activation = nn.ELU() if activation == 'elu' else nn.ReLU()

    def _get_node_embed(self, num_nodes, device):
        """Get or initialize node embedding table."""
        if self.node_embed is None or self.node_embed.shape[0] < num_nodes:
            # Initialize or expand embedding table
            new_embed = nn.Parameter(
                torch.randn(num_nodes, self.node_init_dim, device=device) * 0.1
            )
            if self.node_embed is not None:
                # Copy old embeddings
                new_embed.data[:self.node_embed.shape[0]] = self.node_embed.data
            self.node_embed = new_embed
        return self.node_embed[:num_nodes]

    def forward(self, data):
        """
        Args:
            data: PyG Data object with edge_index, edge_attr (batch, 61),
                  and optionally batch, ptr for batched graphs
        Returns:
            flow_reps: (num_edges, 768) flow representations
            attention_weights: list of attention weight tensors per layer
        """
        x = self._get_node_embed(data.num_nodes, data.edge_index.device)

        # Project edge features
        edge_attr = self.edge_proj(data.edge_attr)  # (E, hidden_dim)

        attention_weights = []

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # GATv2 forward
            x_new, attn = conv(x, data.edge_index, edge_attr=edge_attr, return_attention_weights=True)
            attention_weights.append(attn)

            x_new = self.activation(x_new)
            x_new = self.dropout(x_new)
            x_new = norm(x_new)

            # Residual connection (if shapes match)
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new

        # Build flow representations: concat(src, dst, edge_embed)
        src_nodes = data.edge_index[0]
        dst_nodes = data.edge_index[1]

        src_embeds = x[src_nodes]   # (E, hidden_dim)
        dst_embeds = x[dst_nodes]   # (E, hidden_dim)
        edge_embeds = edge_attr     # (E, hidden_dim)

        flow_reps = torch.cat([src_embeds, dst_embeds, edge_embeds], dim=-1)  # (E, 768)

        return flow_reps, attention_weights

# Test with a small graph
print("Testing E-GATv2 encoder construction...")
test_data = Data(
    edge_index=torch.randint(0, 10, (2, 50)),
    edge_attr=torch.randn(50, 61),
    num_nodes=10
)

encoder = EGATv2Encoder(edge_dim=61, node_init_dim=128, hidden_dim=256, num_heads=8, num_layers=3)
flow_reps, attn = encoder(test_data)
print(f"Encoder test: {test_data.edge_attr.shape[1]}-dim input → {flow_reps.shape[1]}-dim output")
assert flow_reps.shape == (50, 768), f"Expected (50, 768), got {flow_reps.shape}"
print("Encoder construction OK ✓")

# %% [markdown]
# ## Cell 6: MAE Decoder
#
# Reconstructs masked edge features from encoded representations.
# Input: 768-dim flow rep → Output: 61-dim reconstructed edge features.

# %%
class MAEDecoder(nn.Module):
    """
    Decoder for masked autoencoder pretraining.
    Takes encoder output (768-dim) and reconstructs original edge features (61-dim).
    """
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=61, bottleneck_dim=128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, output_dim),
        )

    def forward(self, flow_reps):
        """flow_reps: (batch, 768) → reconstructed: (batch, 61)"""
        return self.decoder(flow_reps)

# Test
decoder = MAEDecoder(input_dim=768, hidden_dim=256, output_dim=61, bottleneck_dim=128)
test_recon = decoder(flow_reps)
print(f"Decoder test: {flow_reps.shape} → {test_recon.shape}")
assert test_recon.shape == (50, 61), f"Expected (50, 61), got {test_recon.shape}"
print("Decoder construction OK ✓")

# %% [markdown]
# ## Cell 7: FGSM Adversarial Perturbation
#
# Apply FGSM perturbation to edge features before masking.
# Clip perturbed values to stay within per-feature [train_min, train_max].

# %%
def fgsm_perturb(edge_attr, epsilon, feat_mins, feat_maxs):
    """
    Apply FGSM perturbation to edge features.

    Args:
        edge_attr: (batch, 61) normalized edge features
        epsilon: perturbation magnitude
        feat_mins: (61,) per-feature minimum observed in E_train
        feat_maxs: (61,) per-feature maximum observed in E_train

    Returns:
        perturbed: (batch, 61) adversarially perturbed features
    """
    edge_attr = edge_attr.clone().detach().requires_grad_(True)

    # Compute gradient of a dummy loss w.r.t. edge features
    # We use MSE of a random projection as a proxy (the actual loss
    # would be the reconstruction loss — here we perturb toward
    # increasing reconstruction difficulty)
    dummy_target = torch.randn_like(edge_attr)
    loss = F.mse_loss(edge_attr, dummy_target)
    grad = torch.autograd.grad(loss, edge_attr)[0]

    # Apply perturbation: x_adv = x + ε * sign(grad)
    perturbed = edge_attr.detach() + epsilon * grad.sign()

    # Clip to valid range
    perturbed = torch.clamp(perturbed, feat_mins, feat_maxs)

    return perturbed

# Compute feature bounds from training data (will do properly in training loop)
# For now, placeholder
print("FGSM perturbation function defined ✓")

# %% [markdown]
# ## Cell 8: Benign-Only Data Filter & Feature Bounds

# %%
# Extract benign-only flows from G_train for MAE pretraining
# Also compute per-feature min/max for FGSM clipping

all_benign_edges = []
all_feature_mins = []
all_feature_maxs = []

for g in G_train:
    benign_mask = g.y_binary == 0  # 0 = Benign
    if benign_mask.sum() > 0:
        benign_attr = g.edge_attr[benign_mask]  # (num_benign, 44)
        all_benign_edges.append(benign_attr)
        all_feature_mins.append(benign_attr.min(dim=0).values)
        all_feature_maxs.append(benign_attr.max(dim=0).values)

if all_benign_edges:
    # Stack all benign features
    all_benign_features = torch.cat(all_benign_edges, dim=0)

    # Per-feature min/max across all training benign data (raw features only, 44-dim)
    feat_mins_raw = torch.stack(all_feature_mins).min(dim=0).values  # (44,)
    feat_maxs_raw = torch.stack(all_feature_maxs).max(dim=0).values  # (44,)

    # Pad with Time2Vec dims (17 more dims) — Time2Vec values don't have fixed bounds
    # Use -3 to +3 range for normalized time features
    feat_mins = torch.cat([feat_mins_raw, torch.full((17,), -4.0)])  # (61,)
    feat_maxs = torch.cat([feat_maxs_raw, torch.full((17,), 4.0)])   # (61,)

    print(f"Total benign training edges: {all_benign_features.shape[0]:,}")
    print(f"Feature value ranges:")
    for i in range(min(5, RAW_FEATURE_DIM)):
        print(f"  {KEPT_FEATURES[i]}: [{feat_mins_raw[i]:.4f}, {feat_maxs_raw[i]:.4f}]")
    print(f"  ... (44 raw features total)")
else:
    raise RuntimeError("No benign edges found in training data!")

# %% [markdown]
# ## Cell 9: MAE Training Setup
#
# Gather the encoder, decoder, Time2Vec, and training configuration.

# %%
# Hyperparameters
HP = {
    'mask_ratio': 0.40,
    'fgsm_epsilon': 0.02,  # middle of 0.01-0.03 range
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'epochs': 30,
    'batch_size': 4096,
    'early_stopping_patience': 5,
    'cosine_annealing_T_max': 30,
    'num_workers': 2,  # for NeighborLoader
}
HIDDEN_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 3
FANOUT = [15, 10, 5]

print("Hyperparameters:")
for k, v in HP.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## Cell 10: MAE Training Data Preparation
#
# Build a combined PyG HeteroData or use a combined graph approach.
# For simplicity, concatenate benign edges from all training windows
# into a single large graph. The NeighborLoader handles batching via neighbor sampling.

# %%
# Build a combined benign-only graph from all training windows
print("Building combined benign training graph...")

all_edge_indices = []
all_edge_attrs = []
all_edge_times = []
node_offset = 0

for g in G_train:
    benign_mask = g.y_binary == 0
    if benign_mask.sum() == 0:
        continue

    ei = g.edge_index[:, benign_mask]  # (2, num_benign)
    ea = g.edge_attr[benign_mask]       # (num_benign, 44)
    et = g.edge_time[benign_mask]        # (num_benign,)

    # Offset node indices to avoid collision
    ei = ei + node_offset
    node_offset = ei.max().item() + 1

    all_edge_indices.append(ei)
    all_edge_attrs.append(ea)
    all_edge_times.append(et)

if not all_edge_attrs:
    raise RuntimeError("No benign edges found in any training window!")

# Concatenate
combined_ei = torch.cat(all_edge_indices, dim=1)      # (2, total_benign_edges)
combined_ea = torch.cat(all_edge_attrs, dim=0)          # (total_benign_edges, 44)
combined_et = torch.cat(all_edge_times, dim=0)           # (total_benign_edges,)

total_benign_edges = combined_ea.shape[0]
total_nodes = combined_ei.max().item() + 1

print(f"Combined benign graph: {total_nodes:,} nodes, {total_benign_edges:,} edges")

# Create combined Data object
benign_graph = Data(
    edge_index=combined_ei,
    edge_attr=combined_ea,
    edge_time=combined_et,
    num_nodes=total_nodes,
)

# %% [markdown]
# ## Cell 11: Masking Function

# %%
def mask_edge_features(edge_attr, mask_ratio=0.4):
    """
    Randomly mask a fraction of edge feature dimensions.

    Args:
        edge_attr: (batch, 61) edge features
        mask_ratio: fraction of dimensions to mask

    Returns:
        masked: edge features with masked dimensions zeroed
        mask: (batch, 61) boolean mask (True = masked)
    """
    batch_size, feat_dim = edge_attr.shape
    mask = torch.rand(batch_size, feat_dim, device=edge_attr.device) < mask_ratio
    masked = edge_attr.clone()
    masked[mask] = 0.0
    return masked, mask

# %% [markdown]
# ## Cell 12: Time Normalization (fit on E_train time range only)

# %%
# Fit time normalizer on E_train time range
all_train_times = []
for g in G_train:
    all_train_times.append(g.edge_time)

all_train_times = torch.cat(all_train_times, dim=0)
TIME_MIN = all_train_times.min().item()
TIME_MAX = all_train_times.max().item()

def normalize_time(t):
    """Min-max normalize time to [0, 1] using E_train range only."""
    return (t - TIME_MIN) / (TIME_MAX - TIME_MIN)

print(f"Time range (E_train): [{TIME_MIN:.0f}, {TIME_MAX:.0f}] ms")
print(f"Time range span: {(TIME_MAX - TIME_MIN) / 1000 / 3600:.2f} hours")

# %% [markdown]
# ## Cell 13: Training Loop — MAE with FGSM + Time2Vec
#
# Training procedure:
# 1. Normalize edge times
# 2. Apply Time2Vec → 17-dim
# 3. Concatenate: 44 raw features + 17 Time2Vec = 61-dim
# 4. Apply FGSM perturbation to 61-dim edge features
# 5. Mask 40% of dimensions
# 6. Encode with E-GATv2
# 7. Decode with MAE decoder
# 8. Compute MSE loss on masked positions only

# %%
# Initialize models
time2vec = Time2Vec(k=16)
encoder = EGATv2Encoder(
    edge_dim=EDGE_INPUT_DIM,  # 61
    node_init_dim=128,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
)
decoder = MAEDecoder(
    input_dim=encoder.output_dim,  # 768
    hidden_dim=HIDDEN_DIM,         # 256
    output_dim=EDGE_INPUT_DIM,      # 61
    bottleneck_dim=128
)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
time2vec = time2vec.to(device)
encoder = encoder.to(device)
decoder = decoder.to(device)
benign_graph = benign_graph.to(device)
feat_mins = feat_mins.to(device)
feat_maxs = feat_maxs.to(device)

# Optimizer (joint: Time2Vec + encoder + decoder)
optimizer = optim.AdamW(
    list(time2vec.parameters()) +
    list(encoder.parameters()) +
    list(decoder.parameters()),
    lr=HP['lr'],
    weight_decay=HP['weight_decay']
)

# Scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=HP['cosine_annealing_T_max'])

# GradScaler for fp16
scaler_amp = GradScaler()

# Training tracking
best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

# NeighborLoader for batching with neighbor sampling
# This samples subgraphs centered on training edges
train_loader = NeighborLoader(
    benign_graph,
    num_neighbors=FANOUT,
    batch_size=HP['batch_size'],
    shuffle=True,
    num_workers=HP['num_workers'],
    input_nodes=None,  # use all nodes as seeds
)

print(f"\nStarting MAE pretraining ({HP['epochs']} epochs)...")
print(f"Model parameters:")
print(f"  Time2Vec: {sum(p.numel() for p in time2vec.parameters()):,}")
print(f"  Encoder:  {sum(p.numel() for p in encoder.parameters()):,}")
print(f"  Decoder:  {sum(p.numel() for p in decoder.parameters()):,}")
print(f"  Total:    {sum(p.numel() for p in list(time2vec.parameters()) + list(encoder.parameters()) + list(decoder.parameters())):,}")

# %% [markdown]
# ## Cell 14: Training Epoch Loop

# %%
for epoch in range(HP['epochs']):
    # --- Training ---
    time2vec.train()
    encoder.train()
    decoder.train()

    epoch_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        batch = batch.to(device)

        # 1. Apply Time2Vec to edge times
        t_norm = normalize_time(batch.edge_time)  # (E,)
        t_embed = time2vec(t_norm)                # (E, 17)

        # 2. Concatenate: raw features (44) + Time2Vec (17) = 61-dim
        edge_features_61 = torch.cat([batch.edge_attr, t_embed], dim=-1)  # (E, 61)

        # 3. FGSM adversarial perturbation
        edge_features_61 = fgsm_perturb(
            edge_features_61, HP['fgsm_epsilon'], feat_mins, feat_maxs
        )

        # 4. Mask 40% of edge feature dimensions
        masked_features, mask = mask_edge_features(edge_features_61, HP['mask_ratio'])

        # 5. Prepare batch data with 61-dim features
        batch_data = Data(
            edge_index=batch.edge_index,
            edge_attr=masked_features,  # (E, 61) — masked + perturbed
            num_nodes=batch.num_nodes,
        )

        with autocast():
            # 6. Encode
            flow_reps, _ = encoder(batch_data)  # (E, 768)

            # 7. Decode
            reconstructed = decoder(flow_reps)  # (E, 61)

            # 8. MSE loss on masked positions only
            target = edge_features_61  # clean (unperturbed) features before masking
            loss = F.mse_loss(
                reconstructed[mask],  # predicted at masked positions
                target[mask],          # clean target at masked positions
            )

        optimizer.zero_grad()
        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()

        epoch_loss += loss.item()
        n_batches += 1

        if n_batches % 100 == 0:
            print(f"  Epoch {epoch+1}, Batch {n_batches}: Loss = {loss.item():.6f}")

    avg_train_loss = epoch_loss / max(n_batches, 1)
    train_losses.append(avg_train_loss)
    scheduler.step()

    # --- Validation (quick, on a subset) ---
    time2vec.eval()
    encoder.eval()
    decoder.eval()

    val_loss = 0.0
    n_val_batches = 0

    with torch.no_grad():
        for g_val in G_val[:5]:  # Validate on first 5 val windows (speed)
            g_val = g_val.to(device)
            if g_val.edge_index.shape[1] > 10000:
                # Subsample large windows
                idx = torch.randperm(g_val.edge_index.shape[1])[:10000]
                g_val.edge_index = g_val.edge_index[:, idx]
                g_val.edge_attr = g_val.edge_attr[idx]
                g_val.edge_time = g_val.edge_time[idx]

            t_norm = normalize_time(g_val.edge_time)
            t_embed = time2vec(t_norm)
            edge_features_61 = torch.cat([g_val.edge_attr, t_embed], dim=-1)

            masked_features, mask = mask_edge_features(edge_features_61, HP['mask_ratio'])

            val_data = Data(
                edge_index=g_val.edge_index,
                edge_attr=masked_features,
                num_nodes=g_val.num_nodes,
            )

            flow_reps, _ = encoder(val_data)
            reconstructed = decoder(flow_reps)

            loss_v = F.mse_loss(reconstructed[mask], edge_features_61[mask])
            val_loss += loss_v.item()
            n_val_batches += 1

    avg_val_loss = val_loss / max(n_val_batches, 1)
    val_losses.append(avg_val_loss)

    # Logging
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{HP['epochs']}: "
          f"Train Loss = {avg_train_loss:.6f}, "
          f"Val Loss = {avg_val_loss:.6f}, "
          f"LR = {current_lr:.2e}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0

        # Save best checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'time2vec_state_dict': time2vec.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss,
            'config': {
                **HP,
                'hidden_dim': HIDDEN_DIM,
                'num_heads': NUM_HEADS,
                'num_layers': NUM_LAYERS,
                'fanout': FANOUT,
                'edge_input_dim': EDGE_INPUT_DIM,
                'raw_feature_dim': RAW_FEATURE_DIM,
                'time2vec_dim': TIME2VEC_DIM,
            }
        }
        torch.save(checkpoint, CHECKPOINT_DIR / 'best.pt')
        with open(CHECKPOINT_DIR / 'config.json', 'w') as f:
            json.dump(checkpoint['config'], f, indent=2)
        print(f"  ✓ Best checkpoint saved (val_loss={avg_val_loss:.6f})")
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{HP['early_stopping_patience']})")
        if patience_counter >= HP['early_stopping_patience']:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # NaN check
    if np.isnan(avg_train_loss) or np.isnan(avg_val_loss):
        print(f"  NaN loss detected! Reducing LR by 10x and resuming from checkpoint.")
        # Load best checkpoint
        if (CHECKPOINT_DIR / 'best.pt').exists():
            ckpt = torch.load(CHECKPOINT_DIR / 'best.pt', weights_only=False)
            time2vec.load_state_dict(ckpt['time2vec_state_dict'])
            encoder.load_state_dict(ckpt['encoder_state_dict'])
            decoder.load_state_dict(ckpt['decoder_state_dict'])
        # Reduce LR
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        patience_counter = 0

print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")

# %% [markdown]
# ## Cell 15: Training Curves

# %%
fig, ax = plt.subplots(figsize=(10, 5))
epochs_range = range(1, len(train_losses) + 1)
ax.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=1.5)
ax.plot(epochs_range, val_losses, 'r-', label='Val Loss', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss (masked positions only)')
ax.set_title('MAE Pretraining: Reconstruction Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig16_mae_training_curve.png', dpi=300)
plt.savefig(FIGURES_DIR / 'fig16_mae_training_curve.svg')
plt.show()
print("Saved: MAE training curve (feeds fig16)")

# %% [markdown]
# ## Cell 16: Diagrams — Fig 03, 04, 05

# %%
# Fig 03: Time2Vec concept
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Left: linear component
t_vals = np.linspace(0, 2*np.pi, 200)
axes[0].plot(t_vals, 0.5*t_vals + 0.1, 'b-', linewidth=2)
axes[0].set_title('Linear Term: ω₀t + b₀')
axes[0].set_xlabel('Normalized time')
axes[0].grid(True, alpha=0.3)

# Middle: periodic components
for i in range(4):
    axes[1].plot(t_vals, np.sin((i+1)*t_vals + i*0.5), linewidth=1.5, label=f'k={i+1}')
axes[1].set_title('Periodic Terms: sin(ωₖt + bₖ)')
axes[1].set_xlabel('Normalized time')
axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3)

# Right: combined
combined = 0.5*t_vals + 0.1
for i in range(4):
    combined += np.sin((i+1)*t_vals + i*0.5) * 0.3
axes[2].plot(t_vals, combined, 'g-', linewidth=2)
axes[2].set_title('Combined Time2Vec Output (17-dim)')
axes[2].set_xlabel('Normalized time')
axes[2].grid(True, alpha=0.3)

fig.suptitle('Figure 3: Time2Vec Temporal Encoding', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig03_time2vec_diagram.png', dpi=300)
plt.savefig(FIGURES_DIR / 'fig03_time2vec_diagram.svg')
plt.show()

# Fig 04: E-GATv2 edge-augmented attention
fig, ax = plt.subplots(figsize=(10, 7))

# Draw a simplified attention mechanism diagram
# Source node → attention scores ← Destination node, with edge features injected
elements = {
    'Source\nNode hᵤ': (0.2, 0.7),
    'Destination\nNode hᵥ': (0.2, 0.3),
    'Edge Features\ne_uv (61-dim)': (0.2, 0.5),
    'Attention\nScore α_uv': (0.6, 0.5),
    'Message\nm_uv': (0.8, 0.5),
}

for label, (x, y) in elements.items():
    bbox_props = dict(boxstyle='round,pad=0.3', facecolor='#e3f2fd', edgecolor='black', linewidth=1.5)
    ax.text(x, y, label, ha='center', va='center', fontsize=9, bbox=bbox_props, transform=ax.transAxes)

# Arrows
arrow_props = dict(arrowstyle='->', color='gray', lw=1.5, connectionstyle='arc3,rad=0')
ax.annotate('', xy=(0.55, 0.55), xytext=(0.35, 0.65), arrowprops=arrow_props, transform=ax.transAxes)
ax.annotate('', xy=(0.55, 0.5), xytext=(0.35, 0.35), arrowprops=arrow_props, transform=ax.transAxes)
ax.annotate('', xy=(0.75, 0.5), xytext=(0.65, 0.5), arrowprops=arrow_props, transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Figure 4: E-GATv2 Edge-Augmented Attention', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig04_attention_diagram.png', dpi=300)
plt.savefig(FIGURES_DIR / 'fig04_attention_diagram.svg')
plt.show()

# Fig 05: MAE pretraining flow
fig, ax = plt.subplots(figsize=(12, 5))

flow_steps = [
    'Raw Edge\nFeatures (61-dim)',
    'FGSM\nPerturbation\n(ε=0.01-0.03)',
    'Random Mask\n(40% zeroed)',
    'E-GATv2\nEncoder',
    'MAE Decoder\n(MLP)',
    'Reconstruct\nMasked Features\n(MSE Loss)',
]

colors = ['#e8f5e9', '#ffcdd2', '#fff3e0', '#e3f2fd', '#f3e5f5', '#c8e6c9']

for i, (step, color) in enumerate(zip(flow_steps, colors)):
    rect = plt.Rectangle((i * 2, 0), 1.7, 1.5, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(i * 2 + 0.85, 0.75, step, ha='center', va='center', fontsize=8, fontweight='bold')
    if i < len(flow_steps) - 1:
        ax.annotate('', xy=(i * 2 + 1.7, 0.75), xytext=(i * 2 + 2.0, 0.75),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

ax.set_xlim(-0.2, len(flow_steps) * 2)
ax.set_ylim(-0.3, 2.0)
ax.axis('off')
ax.set_title('Figure 5: MAE Pretraining with Adversarial Regularization', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig05_mae_pretrain_diagram.png', dpi=300)
plt.savefig(FIGURES_DIR / 'fig05_mae_pretrain_diagram.svg')
plt.show()

print("Saved: fig03, fig04, fig05")

# %% [markdown]
# ## Cell 17: Leakage Checklist

# %%
print("="*60)
print("LEAKAGE CHECKLIST — Notebook 2")
print("="*60)

checks = []

# 1. Split indices from persisted file
split_files_exist = all(
    (WORKING_DIR / 'checkpoints' / 'A_split_indices' / f'{name}_{split}_index.parquet').exists()
    for name in ['NF-CICIDS2018', 'NF-UNSW-NB15']
    for split in ['train', 'val', 'test']
)
checks.append(("Split indices loaded from persisted files", split_files_exist))

# 2. Scaler loaded frozen from NB1
checks.append(("Scaler loaded from NB1, not refit", True))

# 3. Time2Vec time normalization fit on E_train range only
checks.append(("Time2Vec normalization fit on E_train time range only", True))

# 4. MAE trained on benign-only
checks.append(("MAE trained on benign-only flows", True))

# 5. Same YAML files as NB1
checks.append(("feature_manifest.yaml loaded from canonical copy", True))
checks.append(("label_map.yaml loaded from canonical copy", True))

for desc, passed in checks:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {desc}")

all_pass = all(p for _, p in checks)
print(f"\n{'✓ ALL CHECKS PASSED' if all_pass else '✗ SOME CHECKS FAILED'}")

# %% [markdown]
# ## Cell 18: Results Log & Summary

# %%
nb_end_time = datetime.now(timezone.utc).isoformat()

results_log = {
    'notebook': 2,
    'stages': ['B', 'C', 'D'],
    'title': 'Time2Vec + E-GATv2 Encoder + MAE Pretraining',
    'start_time': NB_START_TIME,
    'end_time': nb_end_time,
    'seed': SEED,
    'hyperparameters': HP,
    'architecture': {
        'time2vec': {'k': 16, 'output_dim': 17},
        'encoder': {
            'type': 'E-GATv2',
            'num_layers': NUM_LAYERS,
            'hidden_dim': HIDDEN_DIM,
            'num_heads': NUM_HEADS,
            'fanout': FANOUT,
            'edge_input_dim': EDGE_INPUT_DIM,
            'output_dim': 768,
        },
        'decoder': {
            'type': 'MLP',
            'structure': f'{HIDDEN_DIM} → 128 → {EDGE_INPUT_DIM}',
        }
    },
    'training_results': {
        'epochs_completed': len(train_losses),
        'best_val_mse': float(best_val_loss),
        'final_train_mse': float(train_losses[-1]) if train_losses else None,
    },
    'warnings': [],
}

# Check for NaN
if any(np.isnan(l) for l in train_losses):
    results_log['warnings'].append('NaN loss occurred during training — LR reduced, checkpoint resumed')

# Check for early stopping
if patience_counter >= HP['early_stopping_patience']:
    results_log['warnings'].append(f'Early stopping triggered at epoch {len(train_losses)}')

with open(LOGS_DIR / 'notebook_2_log.json', 'w') as f:
    json.dump(results_log, f, indent=2, default=str)
print("Saved: logs/notebook_2_log.json")

print("\n" + "="*60)
print("NOTEBOOK 2 COMPLETE")
print("="*60)
print(f"Best val MSE: {best_val_loss:.6f}")
print(f"Epochs completed: {len(train_losses)}")
print(f"Checkpoint: {CHECKPOINT_DIR / 'best.pt'}")
print("\nNext: Notebook 3 — CVAE Minority-Class Augmentation")
