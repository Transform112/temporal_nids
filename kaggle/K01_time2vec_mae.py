"""
K01 — Time2Vec + E-GATv2 Encoder + MAE Pretraining (Stages B/C/D)
===================================================================
KAGGLE T4x2 GPU. Copy-paste into a Kaggle notebook.
Loads preprocessed graphs from laptop pipeline.

Edge input: 41 raw features + 17 Time2Vec = 58-dim
"""

# %% [cell 1] Install & Imports
# !pip install -q torch-geometric pyyaml

import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np, yaml, json, pickle, random
from datetime import datetime, timezone
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch_geometric
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# %% [cell 2] Seed & Paths
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.deterministic = True

WORKING = Path('/kaggle/working')
INPUT   = Path('/kaggle/input/datasets/mysteriousavailable/ids-nf3-processed/ids-nf3-processed')
CKPT_DIR = WORKING / 'checkpoints' / 'D_mae_pretrain'
LOGS_DIR = WORKING / 'logs'
FIGS_DIR = WORKING / 'outputs' / 'figures'
for d in [CKPT_DIR, LOGS_DIR, FIGS_DIR]: d.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NB_START = datetime.now(timezone.utc).isoformat()
print(f"Device: {device} | PyTorch: {torch.__version__} | PyG: {torch_geometric.__version__}")

# %% [cell 3] Load Manifests & Scaler
with open(INPUT/'feature_manifest.yaml') as f: fm = yaml.safe_load(f)
with open(INPUT/'label_map.yaml') as f: lm = yaml.safe_load(f)
with open(INPUT/'scaler.pkl','rb') as f: scaler = pickle.load(f)

KEPT = fm['kept_features']          # 41
EDGE_DIM = fm['final_edge_input_dim']  # 58 (41 + 17)
UNIFIED = lm['unified_classes']
print(f"Edge input dim: {EDGE_DIM} ({len(KEPT)} kept + 17 Time2Vec)")
print(f"Unified classes: {len(UNIFIED)}")

# %% [cell 4] Time2Vec
class Time2Vec(nn.Module):
    def __init__(self, k=16):
        super().__init__(); self.k = k
        self.w0 = nn.Parameter(torch.randn(1)*0.1)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.omega = nn.Parameter(10.0**(torch.rand(k)*6 - 3))  # log-uniform [1e-3, 1e3]
        self.bias = nn.Parameter(torch.zeros(k))
        self.output_dim = k + 1  # 17
    def forward(self, t):
        if t.dim() == 1: t = t.unsqueeze(-1)
        return torch.cat([self.w0*t + self.b0, torch.sin(self.omega*t + self.bias)], dim=-1)

# %% [cell 5] E-GATv2 Encoder
class EGATv2Encoder(nn.Module):
    def __init__(self, edge_dim=58, node_init=128, hidden=256, heads=8, layers=3,
                 d_attn=0.3, d_feat=0.2):
        super().__init__()
        self.hidden=hidden; self.heads=heads; self.layers=layers
        self.output_dim = hidden*3  # 768
        self.node_init = node_init; self.node_embed = None
        self.edge_proj = nn.Linear(edge_dim, hidden)
        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(d_feat)
        for _ in range(layers):
            self.convs.append(GATv2Conv((-1,-1), hidden//heads, heads=heads,
                edge_dim=hidden, dropout=d_attn, concat=True))
            self.norms.append(nn.LayerNorm(hidden))
        self.activation = nn.ELU()

    def _get_node_embed(self, n, dev):
        if self.node_embed is None or self.node_embed.shape[0] < n:
            new = nn.Parameter(torch.randn(n, self.node_init, device=dev)*0.1)
            if self.node_embed is not None:
                new.data[:self.node_embed.shape[0]] = self.node_embed.data
            self.node_embed = new
        return self.node_embed[:n]

    def forward(self, data):
        x = self._get_node_embed(data.num_nodes, data.edge_index.device)
        ea = self.edge_proj(data.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            x_new, _ = conv(x, data.edge_index, edge_attr=ea, return_attention_weights=True)
            x_new = self.activation(x_new); x_new = self.dropout(x_new); x_new = norm(x_new)
            x = x + x_new if x.shape == x_new.shape else x_new
        return torch.cat([x[data.edge_index[0]], x[data.edge_index[1]], ea], dim=-1)

# %% [cell 6] MAE Decoder
class MAEDecoder(nn.Module):
    def __init__(self, in_dim=768, hidden=256, out_dim=58, bottleneck=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(hidden, bottleneck), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(bottleneck, out_dim))
    def forward(self, x): return self.net(x)

# %% [cell 7] Load Preprocessed Graphs (memory-efficient)
def load_graphs(name, split):
    p = INPUT / f'{name}_{split}_list.pt'
    return torch.load(p, weights_only=False) if p.exists() else []

import gc

G_train = load_graphs('NF-CICIDS2018','train') + load_graphs('NF-UNSW-NB15','train')
G_val   = load_graphs('NF-CICIDS2018','val')   + load_graphs('NF-UNSW-NB15','val')

print(f"G_train: {len(G_train)} windows, {sum(g.edge_index.shape[1] for g in G_train):,} edges")
print(f"G_val:   {len(G_val)} windows, {sum(g.edge_index.shape[1] for g in G_val):,} edges")

# %% [cell 8] Build Benign-Only Graph (no giant concatenations)
# Compute time range and feature bounds ITERATIVELY — avoid creating all_ea / all_times tensors
TIME_MIN = float('inf'); TIME_MAX = float('-inf')
fmins_41 = torch.full((41,),  float('inf'))
fmaxs_41 = torch.full((41,), -float('inf'))

ei_all, ea_all, et_all = [], [], []
node_off = 0
total_benign_edges = 0

for g in G_train:
    # Track time bounds (scalar min/max — no tensor cat)
    tmin_g = g.edge_time.min().item(); tmax_g = g.edge_time.max().item()
    TIME_MIN = min(TIME_MIN, tmin_g); TIME_MAX = max(TIME_MAX, tmax_g)
    # Track feature bounds iteratively
    fmins_41 = torch.min(fmins_41, g.edge_attr.min(dim=0).values)
    fmaxs_41 = torch.max(fmaxs_41, g.edge_attr.max(dim=0).values)
    # Extract benign edges and accumulate to combined graph
    mask = (g.y == 0)
    if mask.sum() > 0:
        ei = g.edge_index[:, mask] + node_off
        ea = g.edge_attr[mask]
        et = g.edge_time[mask]
        ei_all.append(ei); ea_all.append(ea); et_all.append(et)
        node_off = ei.max().item() + 1 if ei.numel() > 0 else node_off
        total_benign_edges += mask.sum().item()

print(f"Time range: {((TIME_MAX-TIME_MIN)/3.6e6):.1f} hours")

def norm_time(t):
    return (t - TIME_MIN) / (TIME_MAX - TIME_MIN)

# Build combined benign graph tensors
benign_graph = Data(
    edge_index=torch.cat(ei_all, dim=1),
    edge_attr=torch.cat(ea_all, dim=0),
    edge_time=torch.cat(et_all, dim=0),
    num_nodes=node_off
)  # Keep on CPU — NeighborLoader requires CPU tensors for sampling

# Free intermediate lists (they duplicate all benign edge data)
del ei_all, ea_all, et_all; gc.collect(); torch.cuda.empty_cache()

print(f"Benign-only: {benign_graph.edge_index.shape[1]:,} edges, {node_off:,} nodes")

# Feature bounds for FGSM/PGD clamping (41 raw + 17 Time2Vec)
fmins_58 = torch.cat([fmins_41, torch.full((17,), -4.0)]).to(device)
fmaxs_58 = torch.cat([fmaxs_41, torch.full((17,),  4.0)]).to(device)

# Free G_train — no longer needed after benign graph is built + bounds computed
# (NeighborLoader uses benign_graph; G_val kept for validation)
del G_train; gc.collect(); torch.cuda.empty_cache()
print(f"Freed G_train. RAM available for training.")

# %% [cell 9] FGSM Perturbation & Masking
def fgsm_perturb_correct(ea58_clean, encoder, decoder, eps, fmins, fmaxs, batch_edge_index, batch_num_nodes):
    """FGSM: perturb input in direction that maximizes reconstruction error.
    Returns perturbed features (ea58_perturbed).
    Requires one extra forward pass through encoder+decoder to compute the attack gradient."""
    ea_grad = ea58_clean.clone().detach().requires_grad_(True)
    d = Data(edge_index=batch_edge_index, edge_attr=ea_grad, num_nodes=batch_num_nodes)
    rep = encoder(d)
    recon = decoder(rep)
    # Gradient of reconstruction MSE w.r.t. input → direction that INCREASES error
    loss = F.mse_loss(recon, ea_grad)
    grad = torch.autograd.grad(loss, ea_grad)[0]
    perturbed = ea58_clean + eps * grad.sign()
    return torch.clamp(perturbed, fmins, fmaxs).detach()

def mask_features(ea, ratio=0.4):
    mask = torch.rand_like(ea) < ratio
    masked = ea.clone(); masked[mask] = 0.0
    return masked, mask

# %% [cell 10] Initialize & Train (reduced batch/fanout for VRAM)
HP = {'mask_ratio':0.40, 'fgsm_eps':0.02, 'lr':1e-3, 'wd':1e-5,
      'epochs':30, 'batch':2048, 'patience':5, 'fanout':[10,5,3]}

t2v = Time2Vec(k=16).to(device)
encoder = EGATv2Encoder(edge_dim=EDGE_DIM).to(device)
decoder = MAEDecoder(out_dim=EDGE_DIM).to(device)

# Dummy forward to materialize lazy GATv2Conv parameters (avoids UninitializedParameter error)
with torch.no_grad():
    dummy_ea = torch.randn(2, EDGE_DIM, device=device)
    dummy_ei = torch.tensor([[0, 1],[1, 0]], device=device)
    _ = encoder(Data(edge_index=dummy_ei, edge_attr=dummy_ea, num_nodes=2))

opt = optim.AdamW(list(t2v.parameters())+list(encoder.parameters())+list(decoder.parameters()),
                  lr=HP['lr'], weight_decay=HP['wd'])
sched = CosineAnnealingLR(opt, T_max=HP['epochs'])
amp_scaler = GradScaler()

loader = NeighborLoader(benign_graph, num_neighbors=HP['fanout'],
                        batch_size=HP['batch'], shuffle=True, num_workers=0)

train_losses, val_losses = [], []
best_val, patience = float('inf'), 0

print(f"\nMAE Pretraining: {HP['epochs']} epochs, {sum(p.numel() for p in list(t2v.parameters())+list(encoder.parameters())+list(decoder.parameters())):,} params")

for epoch in range(HP['epochs']):
    t2v.train(); encoder.train(); decoder.train()
    epoch_loss, nb = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        tn = norm_time(batch.edge_time); te = t2v(tn)
        ea58_clean = torch.cat([batch.edge_attr, te], dim=-1)
        ea58 = fgsm_perturb_correct(ea58_clean, encoder, decoder, HP['fgsm_eps'],
                                     fmins_58, fmaxs_58, batch.edge_index, batch.num_nodes)
        masked, mask = mask_features(ea58, HP['mask_ratio'])

        batch_data = Data(edge_index=batch.edge_index, edge_attr=masked, num_nodes=batch.num_nodes)
        with autocast():
            reps = encoder(batch_data)
            recon = decoder(reps)
            loss = F.mse_loss(recon[mask], ea58_clean[mask])

        opt.zero_grad(); amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(opt)  # unscale for correct gradient clipping
        torch.nn.utils.clip_grad_norm_(list(t2v.parameters())+list(encoder.parameters())+list(decoder.parameters()), 1.0)
        amp_scaler.step(opt)
        if not torch.isnan(loss): amp_scaler.update()
        # Zero node_embed gradient (not in optimizer param_groups — would leak across batches)
        if encoder.node_embed is not None and encoder.node_embed.grad is not None:
            encoder.node_embed.grad = None
        epoch_loss += loss.item(); nb += 1

    if not np.isnan(epoch_loss / max(nb, 1)):
        sched.step()  # skip scheduler step on NaN epoch
    avg_train = epoch_loss / max(nb, 1); train_losses.append(avg_train)

    # Validation
    t2v.eval(); encoder.eval(); decoder.eval()
    vloss, nv = 0.0, 0
    with torch.no_grad():
        val_sample = random.sample(G_val, min(5, len(G_val)))
        for g in val_sample:
            g = g.to(device)
            if g.edge_index.shape[1] > 5000:
                idx = torch.randperm(g.edge_index.shape[1])[:5000]
                g.edge_index = g.edge_index[:, idx]  # select edge columns
                g.edge_attr = g.edge_attr[idx]       # select edge rows
                g.edge_time = g.edge_time[idx]       # select edge rows
            tn = norm_time(g.edge_time); te = t2v(tn)
            ea58 = torch.cat([g.edge_attr, te], dim=-1)
            masked, mask = mask_features(ea58, HP['mask_ratio'])
            d = Data(edge_index=g.edge_index, edge_attr=masked, num_nodes=g.num_nodes)
            recon = decoder(encoder(d))
            vloss += F.mse_loss(recon[mask], ea58[mask]).item(); nv += 1

    avg_val = vloss / max(nv, 1); val_losses.append(avg_val)
    print(f"Epoch {epoch+1:2d}/{HP['epochs']}: train={avg_train:.6f}, val={avg_val:.6f}, lr={opt.param_groups[0]['lr']:.2e}")

    if avg_val < best_val:
        best_val = avg_val; patience = 0
        torch.save({'epoch':epoch+1, 't2v':t2v.state_dict(), 'encoder':encoder.state_dict(),
                    'decoder':decoder.state_dict(), 'val_loss':avg_val,
                    'time_min':TIME_MIN, 'time_max':TIME_MAX, 'config':HP,
                    'edge_dim':EDGE_DIM}, CKPT_DIR/'best.pt')
        with open(CKPT_DIR/'config.json','w') as f: json.dump(HP, f, indent=2)
        print(f"  [OK] checkpoint saved (val_loss={avg_val:.6f})")
    else:
        patience += 1
        if patience >= HP['patience']:
            print(f"  Early stopping at epoch {epoch+1}"); break

    if np.isnan(avg_train):
        print(f"  NaN detected! Reduce LR 10x, re-init optimizer, resume best...")
        if (CKPT_DIR/'best.pt').exists():
            ckpt = torch.load(CKPT_DIR/'best.pt', weights_only=False)
            t2v.load_state_dict(ckpt['t2v']); encoder.load_state_dict(ckpt['encoder'],strict=False)
            decoder.load_state_dict(ckpt['decoder'])
        # Re-create optimizer (fresh momentum) and scheduler with reduced LR
        new_lr = HP['lr'] * (0.1 ** (patience//HP['patience'] + 1))
        opt = optim.AdamW(list(t2v.parameters())+list(encoder.parameters())+list(decoder.parameters()),
                          lr=new_lr, weight_decay=HP['wd'])
        sched = CosineAnnealingLR(opt, T_max=HP['epochs']-epoch)  # schedule for remaining epochs
        amp_scaler = GradScaler()  # reset scale factor to avoid repeated fp16 overflow
        patience = 0

# %% [cell 11] Training Curve & Log
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(train_losses, 'b-', label='Train'); ax.plot(val_losses, 'r-', label='Val')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE (masked)'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('MAE Pretraining — Reconstruction Loss')
plt.tight_layout(); plt.savefig(FIGS_DIR/'mae_training_curve.png', dpi=150); plt.show()

log = {'notebook':'K01','stages':['B','C','D'],'best_val_mse':float(best_val),
       'epochs':len(train_losses),'hp':HP,'edge_dim':EDGE_DIM,
       'time_range_h':round((TIME_MAX-TIME_MIN)/3.6e6,1)}
with open(LOGS_DIR/'k01_log.json','w') as f: json.dump(log, f, indent=2)

print(f"\nK01 COMPLETE. Best val MSE: {best_val:.6f}")
print(f"Checkpoint: {CKPT_DIR/'best.pt'}")
print(f"Next: K02 (CVAE)")
