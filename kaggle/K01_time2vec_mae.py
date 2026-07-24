"""
K01 - Self-Supervised MAE Pretraining for E-GATv2 Encoder
=============================================================
KAGGLE T4x2. Trains encoder+decoder to reconstruct masked edge features
on the BENIGN-ONLY graph.

FIXES vs previous version:
1. Constant ones-vector node init (matches E-GraphSAGE/Anomal-E literature).
   No node_embed table, no max_nodes sizing -> K03 loads this encoder with
   ZERO architecture mismatch (was silently broken via strict=False before).
2. Edge-wise masking (whole 58-dim row masked) instead of per-feature masking.
   Per-feature masking let the decoder cheat off correlated unmasked features
   in the same row instead of learning from graph structure.
3. Learnable mask token instead of hard zero-fill (GraphMAE).
4. Scaled Cosine Error loss instead of MSE (GraphMAE finding: more robust).
5. patience 5->8, VAL_POOL_CAP 900->1500 (cheap given RAM headroom, less
   premature stopping from noisy small-val-pool variance).
"""

# %% [cell 1] Imports
import os, gc, time, pickle, random, warnings
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
warnings.filterwarnings('ignore')

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np, yaml, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import sys
from pathlib import Path
MODELS_PATH = Path('/kaggle/input/datasets/harshitpachahara/models-py')
if MODELS_PATH.exists() and str(MODELS_PATH) not in sys.path:
    sys.path.append(str(MODELS_PATH))
from models import Time2Vec, EGATv2Encoder, ClassifierHead, FocalLoss


# %% [cell 2] Seed, paths, RAM check
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.deterministic = True

WORKING = Path('../working')
INPUT = Path('/kaggle/input/datasets/mysteriousavailable/ids-nf3-processed/ids-nf3-processed')
CKPT_DIR = WORKING / 'checkpoints' / 'D_mae_pretrain'
LOGS_DIR = WORKING / 'logs'
FIGS_DIR = WORKING / 'outputs' / 'figures'
for d in [CKPT_DIR, LOGS_DIR, FIGS_DIR]: d.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
try:
    import psutil
    print(f"RAM available: {psutil.virtual_memory().available/1e9:.1f} GB")
except ImportError:
    pass

# %% [cell 3] Manifests & scaler
with open(INPUT/'feature_manifest.yaml') as f: fm = yaml.safe_load(f)
with open(INPUT/'label_map.yaml') as f: lm = yaml.safe_load(f)
with open(INPUT/'scaler.pkl', 'rb') as f: scaler = pickle.load(f)

KEPT = fm['kept_features']              # 41
EDGE_DIM = fm['final_edge_input_dim']   # 58 = 41 raw + 17 Time2Vec
UNIFIED = lm['unified_classes']
print(f"Edge dim: {EDGE_DIM} | Classes ({len(UNIFIED)}): {UNIFIED}")

# %% [cell 4] Time2Vec

assert 16 + 1 + len(KEPT) == EDGE_DIM, "Time2Vec dim + KEPT features must equal EDGE_DIM"

# %% [cell 5] E-GATv2 Encoder
# FIX: constant ones-vector node init. No node_embed table, no max_nodes,
# no wraparound logic. Same class must be reused verbatim in K03 -> guaranteed
# encoder compatibility, no strict=False silent-skip risk.

# %% [cell 6] MAE Decoder + Scaled Cosine Error loss (GraphMAE)
class MAEDecoder(nn.Module):
    def __init__(self, in_dim=768, hidden=256, out_dim=58, bottleneck=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(hidden, bottleneck), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(bottleneck, out_dim))
    def forward(self, x): return self.net(x)

def sce_loss(pred, target, gamma=2):
    pred, target = F.normalize(pred, dim=-1), F.normalize(target, dim=-1)
    return (1 - (pred * target).sum(-1)).pow(gamma).mean()

def mask_features(ea, mask_token, ratio=0.4):
    # FIX: mask whole edges (entire 58-dim row), not individual features.
    # Old per-feature masking let the decoder reconstruct from correlated
    # unmasked features in the same row -> pretext task was too easy.
    n = ea.shape[0]
    idx = torch.randperm(n, device=ea.device)[:int(n * ratio)]
    mask = torch.zeros(n, dtype=torch.bool, device=ea.device); mask[idx] = True
    masked = ea.clone(); masked[idx] = mask_token
    return masked, mask

# %% [cell 7] Load validation graphs
def load_graphs(name, split):
    p = INPUT / f'{name}_{split}_list.pt'
    if not p.exists(): return []
    raw = torch.load(p, weights_only=False)
    out = [Data(edge_index=g.edge_index.long(), edge_attr=g.edge_attr.float(),
                edge_time=g.edge_time.float(), num_nodes=int(g.num_nodes), y=g.y.long()) for g in raw]
    del raw; gc.collect()
    return out

G_val = load_graphs('NF-CICIDS2018', 'val') + load_graphs('NF-UNSW-NB15', 'val')
print(f"G_val: {len(G_val)} windows, {sum(g.edge_index.shape[1] for g in G_val):,} edges")

VAL_POOL_CAP = 1500  # ~1GB resident, negligible vs Kaggle's ~30GB RAM (see analysis above)
if len(G_val) > VAL_POOL_CAP:
    random.shuffle(G_val); G_val = G_val[:VAL_POOL_CAP]; gc.collect()
print(f"G_val capped to {len(G_val)} windows")

# %% [cell 8] Build benign-only training graph (2-pass: stats sweep, then single
# preallocated write. Avoids holding raw + filtered copies of 14M+ edge datasets at once.)
TIME_MIN, TIME_MAX = float('inf'), float('-inf')
fmins_41, fmaxs_41 = torch.full((41,), float('inf')), torch.full((41,), -float('inf'))

def extract_benign(name, node_off):
    global TIME_MIN, TIME_MAX, fmins_41, fmaxs_41
    g_list = load_graphs(name, 'train')
    for g in g_list:
        TIME_MIN = min(TIME_MIN, g.edge_time.min().item()); TIME_MAX = max(TIME_MAX, g.edge_time.max().item())
        fmins_41 = torch.min(fmins_41, g.edge_attr.min(dim=0).values)
        fmaxs_41 = torch.max(fmaxs_41, g.edge_attr.max(dim=0).values)
    n_benign = sum(int((g.y == 0).sum()) for g in g_list)
    ei = torch.empty((2, n_benign), dtype=torch.long)
    ea = torch.empty((n_benign, 41), dtype=torch.float32)
    et = torch.empty((n_benign,), dtype=torch.float32)
    ptr = 0
    for g in g_list:
        m = g.y == 0; n = int(m.sum())
        if n == 0: continue
        ei[:, ptr:ptr+n] = g.edge_index[:, m] + node_off
        ea[ptr:ptr+n] = g.edge_attr[m]; et[ptr:ptr+n] = g.edge_time[m]
        node_off = int(ei[:, ptr:ptr+n].max()) + 1
        ptr += n
    print(f"{name}: {ptr:,} benign edges, node_off={node_off:,}")
    del g_list; gc.collect()
    return ei, ea, et, node_off

node_off = 0
ei1, ea1, et1, node_off = extract_benign('NF-CICIDS2018', node_off)
ei2, ea2, et2, node_off = extract_benign('NF-UNSW-NB15', node_off)

benign_graph = Data(edge_index=torch.cat([ei1, ei2], 1), edge_attr=torch.cat([ea1, ea2]),
                     edge_time=torch.cat([et1, et2]), num_nodes=node_off)
del ei1, ea1, et1, ei2, ea2, et2; gc.collect()
print(f"Benign graph: {benign_graph.edge_index.shape[1]:,} edges, {node_off:,} nodes")

def norm_time(t): return (t - TIME_MIN) / (TIME_MAX - TIME_MIN)

fmins_58 = torch.cat([fmins_41, torch.full((17,), -4.0)]).to(device)
fmaxs_58 = torch.cat([fmaxs_41, torch.full((17,),  4.0)]).to(device)

# %% [cell 9] Init model, optimizer
HP = {'mask_ratio': 0.4, 'fgsm_eps': 0.02, 'lr': 1e-3, 'wd': 1e-5,
      'epochs': 30, 'batch': 4096, 'patience': 8, 'fanout': [15, 10, 5]}
print(f"HP: {HP}")

loader = NeighborLoader(benign_graph, num_neighbors=HP['fanout'], batch_size=HP['batch'],
                         shuffle=True, num_workers=0)
print(f"{len(loader)} batches/epoch")

t2v = Time2Vec(16).to(device)
encoder = EGATv2Encoder(edge_dim=EDGE_DIM).to(device)
decoder = MAEDecoder(out_dim=EDGE_DIM).to(device)
mask_token = nn.Parameter(torch.zeros(EDGE_DIM, device=device))

with torch.no_grad():  # materialize lazy GATv2Conv params
    dummy = Data(edge_index=torch.tensor([[0, 1], [1, 0]], device=device),
                 edge_attr=torch.randn(2, EDGE_DIM, device=device), num_nodes=2)
    _ = encoder(dummy)

params = list(t2v.parameters()) + list(encoder.parameters()) + list(decoder.parameters()) + [mask_token]
opt = optim.AdamW(params, lr=HP['lr'], weight_decay=HP['wd'])
sched = CosineAnnealingLR(opt, T_max=HP['epochs'])
scaler = GradScaler()
print(f"Total trainable params: {sum(p.numel() for p in params):,}")

# %% [cell 10] Train
train_losses, val_losses = [], []
best_val_score, patience = float('inf'), 0


# Resume from checkpoint if exists
if (CKPT_DIR / 'best.pt').exists():
    print(f'Resuming training from {CKPT_DIR / "best.pt"}')
    ckpt = torch.load(CKPT_DIR / 'best.pt', map_location=device, weights_only=False)
    if 't2v' in ckpt: t2v.load_state_dict(ckpt['t2v'])
    if 'encoder' in ckpt: encoder.load_state_dict(ckpt['encoder'])
    if 'decoder' in ckpt: decoder.load_state_dict(ckpt['decoder'])
    if 'opt' in ckpt: opt.load_state_dict(ckpt['opt'])
    if 'sched' in ckpt: sched.load_state_dict(ckpt['sched'])
    if 'epoch' in ckpt: start_epoch = ckpt['epoch']
    else: start_epoch = 0
    if 'val_score' in ckpt: best_val_score = ckpt['val_score']
else:
    start_epoch = 0

for epoch in range(start_epoch, HP['epochs']):
    t0 = time.time()
    t2v.train(); encoder.train(); decoder.train()
    epoch_loss, nb = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        ea_clean = torch.cat([batch.edge_attr, t2v(norm_time(batch.edge_time))], dim=-1)

        # FGSM adversarial perturbation (single step, small eps)
        ea_grad = ea_clean.clone().requires_grad_(True)
        rep = encoder(Data(edge_index=batch.edge_index, edge_attr=ea_grad, num_nodes=batch.num_nodes))
        grad = torch.autograd.grad(F.mse_loss(decoder(rep), ea_grad), ea_grad)[0]
        ea_adv = torch.clamp(ea_clean + HP['fgsm_eps'] * grad.sign(), fmins_58, fmaxs_58).detach()

        masked, mask = mask_features(ea_adv, mask_token, HP['mask_ratio'])
        with autocast():
            recon = decoder(encoder(Data(edge_index=batch.edge_index, edge_attr=masked, num_nodes=batch.num_nodes)))
            loss = sce_loss(recon[mask], ea_clean[mask])

        opt.zero_grad(); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()
        epoch_loss += loss.item(); nb += 1

    sched.step()
    avg_train = epoch_loss / nb; train_losses.append(avg_train)

    # Validation: reconstruction error on benign vs attack edges separately
    t2v.eval(); encoder.eval(); decoder.eval()
    vloss_b, vloss_a, nb_, na_ = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for g in G_val:
            ei, ea, et, y = g.edge_index, g.edge_attr, g.edge_time, g.y
            if ei.shape[1] > 20000:
                idx = torch.randperm(ei.shape[1])[:20000]
                ei, ea, et, y = ei[:, idx], ea[idx], et[idx], y[idx]
            ei, ea, et, y = ei.to(device), ea.to(device), et.to(device), y.to(device)
            ea58 = torch.cat([ea, t2v(norm_time(et))], dim=-1)
            masked, mask = mask_features(ea58, mask_token, HP['mask_ratio'])
            recon = decoder(encoder(Data(edge_index=ei, edge_attr=masked, num_nodes=g.num_nodes)))
            b_mask, a_mask = mask & (y == 0), mask & (y != 0)
            if b_mask.any(): vloss_b += sce_loss(recon[b_mask], ea58[b_mask]).item(); nb_ += 1
            if a_mask.any(): vloss_a += sce_loss(recon[a_mask], ea58[a_mask]).item(); na_ += 1

    avg_val_b, avg_val_a = vloss_b / max(nb_, 1), vloss_a / max(na_, 1)
    val_losses.append(avg_val_b)
    val_score = avg_val_b / (avg_val_a + 1e-6)  # want LOW: benign reconstructs well, attack doesn't

    print(f"Epoch {epoch+1:2d}/{HP['epochs']}: train={avg_train:.4f} val_score={val_score:.4f} "
          f"val_benign={avg_val_b:.4f} val_attack={avg_val_a:.4f} time={time.time()-t0:.0f}s")

    if val_score < best_val_score:
        best_val_score = val_score; patience = 0
        torch.save({'epoch': epoch+1, 't2v': t2v.state_dict(), 'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(), 'mask_token': mask_token.data,
                    'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'val_score': val_score, 'time_min': TIME_MIN, 'time_max': TIME_MAX,
                    'config': HP, 'edge_dim': EDGE_DIM}, CKPT_DIR/'best.pt')
        print(f"  [OK] saved best (val_score={val_score:.4f})")
    else:
        patience += 1
        print(f"  no improvement, patience={patience}/{HP['patience']}")
        if patience >= HP['patience']:
            print(f"  Early stopping at epoch {epoch+1}"); break

    if (epoch + 1) % 5 == 0:
        gc.collect(); torch.cuda.empty_cache()

# %% [cell 11] Save curve & log
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_losses, 'b-', label='Train'); ax.plot(val_losses, 'r-', label='Val (benign SCE)')
ax.set_xlabel('Epoch'); ax.set_ylabel('SCE loss (masked)'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('MAE Pretraining — Reconstruction Loss')
plt.tight_layout(); plt.savefig(FIGS_DIR/'mae_training_curve.png', dpi=150); plt.show()

with open(LOGS_DIR/'k01_log.json', 'w') as f:
    json.dump({'notebook': 'K01', 'best_val_score': float(best_val_score), 'epochs': len(train_losses),
               'hp': HP, 'edge_dim': EDGE_DIM, 'val_losses': val_losses, 'train_losses': train_losses}, f, indent=2)

print(f"\nK01 COMPLETE. Best val score: {best_val_score:.4f}")
print(f"Checkpoint: {CKPT_DIR/'best.pt'}")