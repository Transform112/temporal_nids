import os, gc, time
# Mitigates allocator fragmentation on repeaupdateted variable-size allocations (per the
# suggestion in the CUDA OOM error message). Must be set before CUDA initializes.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
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

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False

# %% [cell 1b] Logging helpers
def log(msg):
    ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

def cpu_mem():
    if _PSUTIL_OK:
        rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
        avail = psutil.virtual_memory().available / 1e9
        return f"cpu_rss={rss:.2f}GB cpu_avail={avail:.2f}GB"
    return "cpu_rss=NA (psutil not installed)"

def gpu_mem():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"gpu_alloc={alloc:.2f}GB gpu_reserved={reserved:.2f}GB"
    return "cpu-mode"

def mem():
    return f"{cpu_mem()} | {gpu_mem()}"

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# %% [cell 2] Seed & Paths
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
NB_START = datetime.now(timezone.utc).isoformat()
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
log(f"Device: {device} ({gpu_name}) | PyTorch: {torch.__version__} | PyG: {torch_geometric.__version__}")
log(f"Seed: {SEED} | Checkpoint dir: {CKPT_DIR} | {mem()}")

# RAM: cap how many validation windows stay resident across the whole run (they're
# re-sampled every epoch so they can't be streamed from disk cheaply). Set to None to
# keep the entire validation set instead.
VAL_POOL_CAP = 300

# %% [cell 3] Load Manifests & Scaler
with open(INPUT/'feature_manifest.yaml') as f: fm = yaml.safe_load(f)
with open(INPUT/'label_map.yaml') as f: lm = yaml.safe_load(f)
with open(INPUT/'scaler.pkl','rb') as f: scaler = pickle.load(f)

KEPT = fm['kept_features']          # 41
EDGE_DIM = fm['final_edge_input_dim']  # 58 (41 + 17)
UNIFIED = lm['unified_classes']
log(f"Edge input dim: {EDGE_DIM} ({len(KEPT)} kept + 17 Time2Vec)")
log(f"Unified classes ({len(UNIFIED)}): {UNIFIED}")
if hasattr(scaler, 'n_features_in_'):
    log(f"Scaler loaded: {type(scaler).__name__}, n_features_in_={scaler.n_features_in_} (expect {len(KEPT)})")
    assert scaler.n_features_in_ == len(KEPT), "scaler feature count mismatch vs KEPT features"
else:
    log(f"Scaler loaded: {type(scaler).__name__} (no n_features_in_ attribute to cross-check)")

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

# BUG FIX 5: catch a feature-dim mismatch here, not three cells downstream as a shape error.
_t2v_dim_check = Time2Vec(k=16).output_dim
assert _t2v_dim_check + len(KEPT) == EDGE_DIM, (
    f"Time2Vec output_dim ({_t2v_dim_check}) + len(KEPT) ({len(KEPT)}) != EDGE_DIM ({EDGE_DIM})"
)
log(f"Feature-dim check OK: {len(KEPT)} raw + {_t2v_dim_check} Time2Vec = {EDGE_DIM}")

# %% [cell 5] E-GATv2 Encoder
class EGATv2Encoder(nn.Module):
    """
    BUG FIX 1: node_embed is preallocated ONCE at construction time to `max_nodes`
    rows and registered as a normal nn.Parameter, so it exists (at full size) in
    encoder.parameters() before the optimizer is ever built. It is never reassigned
    or resized after __init__, so it can't become detached from the optimizer.

    BUG FIX 6: max_nodes should be sized from the largest LOCAL subgraph a single
    forward() call will see (NeighborLoader batches are always locally reindexed
    0..batch.num_nodes-1, so this is bounded by fanout/batch_size — typically a few
    thousand) and the largest single validation window — NOT from any global/dataset-
    wide node-id counter, which can be orders of magnitude larger and will blow up
    GPU memory for no benefit. As a defensive fallback (in case a batch is still
    larger than expected), indices wrap via modulo instead of raising, since these
    embeddings are already position-based slots rather than persistent per-host
    identities (a pre-existing property of this design, not something introduced
    here) — wrapping just reuses slots gracefully instead of crashing.
    """
    def __init__(self, max_nodes, edge_dim=58, node_init=128, hidden=256, heads=8, layers=3,
                 d_attn=0.3, d_feat=0.2, return_attention=False):
        super().__init__()
        assert max_nodes is not None and max_nodes > 0, "max_nodes must be a positive int"
        self.hidden=hidden; self.heads=heads; self.layers=layers
        self.output_dim = hidden*3  # 768
        self.node_init = node_init
        self.max_nodes = max_nodes
        self.return_attention = return_attention  # BUG FIX 4: off by default during pretraining

        self.node_embed = nn.Parameter(torch.randn(max_nodes, node_init) * 0.1)
        self.edge_proj = nn.Linear(edge_dim, hidden)
        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(d_feat)
        for _ in range(layers):
            self.convs.append(GATv2Conv((-1,-1), hidden//heads, heads=heads,
                edge_dim=hidden, dropout=d_attn, concat=True))
            self.norms.append(nn.LayerNorm(hidden))
        self.activation = nn.ELU()

    def forward(self, data):
        n = data.num_nodes
        table_size = self.node_embed.shape[0]
        if n <= table_size:
            x = self.node_embed[:n]
        else:
            # Defensive fallback (should be rare if max_nodes was sized correctly) —
            # wrap indices instead of crashing or growing the table mid-training.
            idx = torch.arange(n, device=self.node_embed.device) % table_size
            x = self.node_embed[idx]
        ea = self.edge_proj(data.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            if self.return_attention:
                x_new, _ = conv(x, data.edge_index, edge_attr=ea, return_attention_weights=True)
            else:
                x_new = conv(x, data.edge_index, edge_attr=ea)
            x_new = self.activation(x_new); x_new = self.dropout(x_new)
            x = norm(x + x_new) if x.shape == x_new.shape else norm(x_new)
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

# %% [cell 7] Load Validation Graphs (kept resident, RAM-capped)
def load_graphs_stripped(name, split, keep_labels):
    """
    RAM FIX R3: load the raw pickled list, then immediately rebuild each graph
    keeping ONLY the fields this pipeline uses (edge_index, edge_attr, edge_time,
    num_nodes, optionally y), cast to consistent float32/long dtypes. Whatever else
    the source file carries (node features, metadata, extra columns) is dropped
    right away instead of sitting in RAM for the rest of the run. The raw list is
    deleted as soon as the stripped copies exist.
    """
    p = INPUT / f'{name}_{split}_list.pt'
    if not p.exists():
        return []
    raw = torch.load(p, weights_only=False)
    out = []
    for g in raw:
        d = Data(edge_index=g.edge_index.long(),
                 edge_attr=g.edge_attr.float(),
                 edge_time=g.edge_time.float(),
                 num_nodes=int(g.num_nodes))
        if keep_labels:
            d.y = g.y.long()
        out.append(d)
    del raw; gc.collect()
    return out

t0 = time.time()
log(f"Loading validation graphs... {mem()}")
G_val = load_graphs_stripped('NF-CICIDS2018', 'val', keep_labels=False) \
      + load_graphs_stripped('NF-UNSW-NB15', 'val', keep_labels=False)
log(f"G_val loaded in {time.time()-t0:.1f}s: {len(G_val)} windows, "
    f"{sum(g.edge_index.shape[1] for g in G_val):,} edges {mem()}")

# RAM FIX R4: cap resident validation windows (they must stay in memory for the whole
# run since random.sample() draws from them every epoch).
if VAL_POOL_CAP is not None and len(G_val) > VAL_POOL_CAP:
    random.shuffle(G_val)
    dropped = len(G_val) - VAL_POOL_CAP
    G_val = G_val[:VAL_POOL_CAP]
    gc.collect()
    log(f"Capped G_val to {VAL_POOL_CAP} windows (dropped {dropped}) -> {mem()}")

# %% [cell 8] Build Benign-Only Training Graph — sequential per-dataset, RAM-efficient
TIME_MIN = float('inf'); TIME_MAX = float('-inf')
fmins_41 = torch.full((41,),  float('inf'))
fmaxs_41 = torch.full((41,), -float('inf'))

def process_train_dataset(name, node_off):
    """
    RAM FIX R1 + R2: loads ONE dataset's train windows, does a cheap stats-only pass
    to get the exact benign edge count (+ time/feature bounds), then writes directly
    into a single preallocated buffer of that exact size. This avoids ever holding
    two full training datasets in memory simultaneously (R1), and avoids the classic
    "python list of small chunks -> torch.cat" pattern that briefly needs ~2x the
    final tensor size in RAM (R2). The loaded window list is freed before returning.
    """
    global TIME_MIN, TIME_MAX, fmins_41, fmaxs_41
    t0 = time.time()
    log(f"Loading {name} train... {mem()}")
    g_list = load_graphs_stripped(name, 'train', keep_labels=True)
    n_total_edges = sum(g.edge_index.shape[1] for g in g_list)
    log(f"  {name}: {len(g_list)} windows, {n_total_edges:,} edges, loaded in {time.time()-t0:.1f}s {mem()}")

    # Pass 1: stats-only sweep (no large tensors retained)
    n_benign = 0
    for g in g_list:
        tmin_g = g.edge_time.min().item(); tmax_g = g.edge_time.max().item()
        TIME_MIN = min(TIME_MIN, tmin_g); TIME_MAX = max(TIME_MAX, tmax_g)
        fmins_41 = torch.min(fmins_41, g.edge_attr.min(dim=0).values)
        fmaxs_41 = torch.max(fmaxs_41, g.edge_attr.max(dim=0).values)
        n_benign += int((g.y == 0).sum().item())
    log(f"  {name}: {n_benign:,} benign edges of {n_total_edges:,} "
        f"({100*n_benign/max(1,n_total_edges):.1f}%) [pass 1 done] {mem()}")

    # Pass 2: fill preallocated buffers directly — single allocation, no intermediate list.
    feat_dim = g_list[0].edge_attr.shape[1] if g_list else 41
    ei_buf = torch.empty((2, n_benign), dtype=torch.long)
    ea_buf = torch.empty((n_benign, feat_dim), dtype=torch.float32)
    et_buf = torch.empty((n_benign,), dtype=torch.float32)
    ptr = 0
    for g in g_list:
        mask = (g.y == 0)
        n = int(mask.sum().item())
        if n == 0:
            continue
        ei_slice = g.edge_index[:, mask] + node_off
        ei_buf[:, ptr:ptr+n] = ei_slice
        ea_buf[ptr:ptr+n] = g.edge_attr[mask]
        et_buf[ptr:ptr+n] = g.edge_time[mask]
        node_off = int(ei_slice.max().item()) + 1
        ptr += n

    del g_list; gc.collect()
    log(f"  {name}: extraction complete ({ptr:,} edges written), node_off={node_off:,} {mem()}")
    return ei_buf, ea_buf, et_buf, node_off, n_total_edges

node_off = 0
ei_cic, ea_cic, et_cic, node_off, n_edges_cic = process_train_dataset('NF-CICIDS2018', node_off)
ei_unsw, ea_unsw, et_unsw, node_off, n_edges_unsw = process_train_dataset('NF-UNSW-NB15', node_off)
total_train_edges = n_edges_cic + n_edges_unsw

log(f"Time range: {((TIME_MAX-TIME_MIN)/3.6e6):.1f} hours ({TIME_MIN:.0f} .. {TIME_MAX:.0f})")

def norm_time(t):
    return (t - TIME_MIN) / (TIME_MAX - TIME_MIN)

# Combine the two (small — only 2 chunks) per-dataset buffers into the final graph.
benign_graph = Data(
    edge_index=torch.cat([ei_cic, ei_unsw], dim=1),
    edge_attr=torch.cat([ea_cic, ea_unsw], dim=0),
    edge_time=torch.cat([et_cic, et_unsw], dim=0),
    num_nodes=node_off
)  # Keep on CPU — NeighborLoader requires CPU tensors for sampling
del ei_cic, ea_cic, et_cic, ei_unsw, ea_unsw, et_unsw
gc.collect()

total_benign_edges = benign_graph.edge_index.shape[1]
log(f"Benign-only graph: {total_benign_edges:,} edges, {node_off:,} nodes "
    f"({100*total_benign_edges/max(1,total_train_edges):.1f}% of all train edges) {mem()}")

# Feature bounds for FGSM/PGD clamping (41 raw + 17 Time2Vec)
fmins_58 = torch.cat([fmins_41, torch.full((17,), -4.0)]).to(device)
fmaxs_58 = torch.cat([fmaxs_41, torch.full((17,),  4.0)]).to(device)
log(f"Feature bounds (raw 41): min range=[{fmins_41.min():.3f},{fmins_41.max():.3f}] "
    f"max range=[{fmaxs_41.min():.3f},{fmaxs_41.max():.3f}]")

# BUG FIX 6: max_val_nodes (largest single validation window's LOCAL node count) is a
# legitimate, small bound. node_off (benign_graph's global cumulative id count) is NOT
# a bound on what any single forward() call needs — see BUG FIX 6 note on the encoder.
# The training-side bound is measured empirically from real NeighborLoader batches in
# cell 10 (needs `loader` + HP, defined there), and combined with max_val_nodes then.
max_val_nodes = max((g.num_nodes for g in G_val), default=0)
log(f"Largest validation window: {max_val_nodes:,} nodes (benign_graph total nodes, "
    f"for reference only, NOT used to size node_embed: {node_off:,})")

torch.cuda.empty_cache()
log(f"Cell 8 complete. {mem()}")

# %% [cell 9] FGSM Perturbation & Masking
def fgsm_perturb_correct(ea58_clean, encoder, decoder, eps, fmins, fmaxs, batch_edge_index, batch_num_nodes):
    """FGSM: perturb input in direction that maximizes reconstruction error.
    Returns perturbed features (ea58_perturbed).
    Requires one extra forward pass through encoder+decoder to compute the attack gradient."""
    ea_grad = ea58_clean.clone().detach().requires_grad_(True)
    d = Data(edge_index=batch_edge_index, edge_attr=ea_grad, num_nodes=batch_num_nodes)
    with autocast():
        rep = encoder(d)
        recon = decoder(rep)
        # Gradient of reconstruction MSE w.r.t. input -> direction that INCREASES error
        loss = F.mse_loss(recon, ea_grad)
    grad = torch.autograd.grad(loss, ea_grad)[0]
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)  # safety: never let a
    # bad attack-gradient batch inject inf/nan into the training input itself
    perturbed = ea58_clean + eps * grad.sign()
    return torch.clamp(perturbed, fmins, fmaxs).detach()

def mask_features(ea, ratio=0.4):
    mask = torch.rand_like(ea) < ratio
    masked = ea.clone(); masked[mask] = 0.0
    return masked, mask

# %% [cell 10] Initialize & Train (reduced batch/fanout for VRAM)
HP = {'mask_ratio':0.40, 'fgsm_eps':0.02, 'lr':1e-3, 'wd':1e-5,
      'epochs':30, 'batch':4096, 'patience':5, 'fanout':[15,10,5]}
log(f"Hyperparameters: {HP}")

loader = NeighborLoader(benign_graph, num_neighbors=HP['fanout'],
                        batch_size=HP['batch'], shuffle=True, num_workers=0)
log(f"NeighborLoader: {len(loader)} batches/epoch (batch_size={HP['batch']}, fanout={HP['fanout']})")

# BUG FIX 6: size node_embed from REAL sampled batch sizes, not from node_off (see
# notes above). Probe a handful of batches up front — NeighborLoader always returns
# locally reindexed subgraphs, so batch.num_nodes here is the true per-forward-call
# node count the model will need, typically a few thousand for this fanout/batch_size.
PROBE_BATCHES = min(20, len(loader))
probe_max = 0
t0 = time.time()
for i, batch in enumerate(loader):
    probe_max = max(probe_max, batch.num_nodes)
    if i + 1 >= PROBE_BATCHES:
        break
log(f"Probed {PROBE_BATCHES} batches in {time.time()-t0:.1f}s: max batch.num_nodes={probe_max:,}")

SAFETY_FACTOR = 2.0   # headroom for batch-to-batch variance beyond what we probed
NODE_TABLE_FLOOR = 2000
NODE_TABLE_CEIL = 500_000  # generous hard ceiling; at this size the table is still <1GB
MAX_NODES = int(max(probe_max * SAFETY_FACTOR, max_val_nodes, NODE_TABLE_FLOOR))
MAX_NODES = min(MAX_NODES, NODE_TABLE_CEIL)
log(f"MAX_NODES for node_embed table: {MAX_NODES:,} "
    f"(probe_max={probe_max:,} x{SAFETY_FACTOR}, max_val_window={max_val_nodes:,}, "
    f"floor={NODE_TABLE_FLOOR:,}, ceiling={NODE_TABLE_CEIL:,}) "
    f"-> table size ~{MAX_NODES*128*4/1e6:.1f} MB")

t2v = Time2Vec(k=16).to(device)
encoder = EGATv2Encoder(max_nodes=MAX_NODES, edge_dim=EDGE_DIM).to(device)
decoder = MAEDecoder(out_dim=EDGE_DIM).to(device)

# Dummy forward to materialize lazy GATv2Conv parameters (avoids UninitializedParameter error)
with torch.no_grad():
    dummy_ea = torch.randn(2, EDGE_DIM, device=device)
    dummy_ei = torch.tensor([[0, 1],[1, 0]], device=device)
    _ = encoder(Data(edge_index=dummy_ei, edge_attr=dummy_ea, num_nodes=2))

all_params = list(t2v.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
opt = optim.AdamW(all_params, lr=HP['lr'], weight_decay=HP['wd'])
sched = CosineAnnealingLR(opt, T_max=HP['epochs'])
amp_scaler = GradScaler()

log(f"Param counts | t2v={count_params(t2v):,} | encoder={count_params(encoder):,} "
    f"(node_embed={encoder.node_embed.numel():,}) | decoder={count_params(decoder):,} | "
    f"total={sum(p.numel() for p in all_params):,}")

train_losses, val_losses = [], []
best_val, patience = float('inf'), 0
intra_epoch_log_every = max(1, len(loader) // 5)
train_start = time.time()

log(f"\nMAE Pretraining: {HP['epochs']} epochs, {sum(p.numel() for p in all_params):,} params {mem()}")

for epoch in range(HP['epochs']):
    epoch_start = time.time()
    t2v.train(); encoder.train(); decoder.train()
    epoch_loss, nb = 0.0, 0
    grad_norm_sum, pert_mag_sum, nan_batches, skipped_batches, n_finite_gn = 0.0, 0.0, 0, 0, 0
    rel_err_sum = 0.0
    nb_finite = 0

    for bi, batch in enumerate(loader):
        batch = batch.to(device)
        tn = norm_time(batch.edge_time); te = t2v(tn)
        ea58_clean = torch.cat([batch.edge_attr, te], dim=-1)
        ea58 = fgsm_perturb_correct(ea58_clean, encoder, decoder, HP['fgsm_eps'],
                                     fmins_58, fmaxs_58, batch.edge_index, batch.num_nodes)
        pert_mag_sum += (ea58 - ea58_clean).abs().mean().item()
        masked, mask = mask_features(ea58, HP['mask_ratio'])

        batch_data = Data(edge_index=batch.edge_index, edge_attr=masked, num_nodes=batch.num_nodes)
        with autocast():
            reps = encoder(batch_data)
            recon = decoder(reps)
            loss = F.mse_loss(recon[mask], ea58_clean[mask])

        if torch.isnan(loss):
            nan_batches += 1
            log(f"    [warn] NaN loss at epoch {epoch+1} batch {bi+1}/{len(loader)}")

        opt.zero_grad()
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(opt)  # unscale for correct gradient clipping
        gn = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        gn_f = float(gn)
        if np.isfinite(gn_f):
            grad_norm_sum += gn_f; n_finite_gn += 1

        # BUG FIX 2: always call update() after step() — GradScaler itself detects
        # inf/nan grads (post-unscale) and skips the optimizer step + shrinks the
        # scale factor when needed. Gating update() on `not isnan(loss)` skipped the
        # scale-factor correction on exactly the batches that needed it.
        scale_before = amp_scaler.get_scale()
        amp_scaler.step(opt)
        amp_scaler.update()
        if amp_scaler.get_scale() < scale_before:
            skipped_batches += 1

        with torch.no_grad():
            re = (F.mse_loss(recon[mask], ea58_clean[mask]).sqrt()
                  / (ea58_clean[mask].abs().mean() + 1e-8)).item()
            if np.isfinite(re):
                rel_err_sum += re
        epoch_loss += (loss.item() if torch.isfinite(loss) else 0.0)
        nb_finite += (1 if torch.isfinite(loss) else 0)
        nb += 1

        if (bi + 1) % intra_epoch_log_every == 0:
            log(f"    epoch {epoch+1} batch {bi+1}/{len(loader)} running_loss={epoch_loss/nb:.6f}")

    if nb_finite > 0:
        sched.step()  # only skip if the whole epoch was unusable
    avg_train = epoch_loss / max(nb_finite, 1); train_losses.append(avg_train)
    avg_grad_norm = grad_norm_sum / max(n_finite_gn, 1)
    avg_pert_mag = pert_mag_sum / max(nb, 1)
    avg_rel_err = rel_err_sum / max(nb_finite, 1)  # RMSE / mean|target|, masked positions only
    nan_frac = nan_batches / max(nb, 1)

    # Validation
    t2v.eval(); encoder.eval(); decoder.eval()
    vloss, nv = 0.0, 0
    with torch.no_grad():
        val_sample = random.sample(G_val, min(5, len(G_val)))
        for g in val_sample:
            # BUG FIX 3: work on local copies — never mutate the objects stored in
            # G_val. The original code reassigned g.edge_index/edge_attr/edge_time
            # (freezing each >5000-edge window to whatever subset was first drawn)
            # and called g.to(device) on the shared object (permanently pinning
            # validation graphs to GPU memory).
            ei, ea, et = g.edge_index, g.edge_attr, g.edge_time
            if ei.shape[1] > 5000:
                idx = torch.randperm(ei.shape[1])[:5000]
                ei, ea, et = ei[:, idx], ea[idx], et[idx]
            ei = ei.to(device); ea = ea.to(device); et = et.to(device)

            tn = norm_time(et); te = t2v(tn)
            ea58 = torch.cat([ea, te], dim=-1)
            masked, mask = mask_features(ea58, HP['mask_ratio'])
            d = Data(edge_index=ei, edge_attr=masked, num_nodes=g.num_nodes)
            recon = decoder(encoder(d))
            vloss += F.mse_loss(recon[mask], ea58[mask]).item(); nv += 1

    avg_val = vloss / max(nv, 1); val_losses.append(avg_val)
    epoch_time = time.time() - epoch_start
    log(f"Epoch {epoch+1:2d}/{HP['epochs']}: train={avg_train:.6f} val={avg_val:.6f} "
        f"rel_err={avg_rel_err:.3f} lr={opt.param_groups[0]['lr']:.2e} "
        f"grad_norm={avg_grad_norm:.3f} nan_batches={nan_batches}/{nb} "
        f"scaler_skips={skipped_batches} time={epoch_time:.1f}s {mem()}")

    # RAM FIX R5: periodic cleanup — NeighborLoader + autograd graphs can leave
    # fragmented allocations behind over many epochs.
    if (epoch + 1) % 5 == 0:
        gc.collect(); torch.cuda.empty_cache()
        log(f"  periodic gc/cache cleanup -> {mem()}")

    if avg_val < best_val:
        best_val = avg_val; patience = 0
        torch.save({'epoch':epoch+1, 't2v':t2v.state_dict(), 'encoder':encoder.state_dict(),
                    'decoder':decoder.state_dict(), 'val_loss':avg_val,
                    'time_min':TIME_MIN, 'time_max':TIME_MAX, 'config':HP,
                    'edge_dim':EDGE_DIM, 'max_nodes':MAX_NODES}, CKPT_DIR/'best.pt')
        with open(CKPT_DIR/'config.json','w') as f: json.dump(HP, f, indent=2)
        log(f"  [OK] checkpoint saved (val_loss={avg_val:.6f}) -> {CKPT_DIR/'best.pt'}")
    else:
        patience += 1
        log(f"  no improvement (best={best_val:.6f}), patience={patience}/{HP['patience']}")
        if patience >= HP['patience']:
            log(f"  Early stopping at epoch {epoch+1}"); break

    NAN_FRAC_TRIGGER = 0.05  # emergency reset only if >5% of an epoch's batches went NaN
    if nb_finite == 0 or nan_frac > NAN_FRAC_TRIGGER:
        log(f"  [warn] {nan_batches}/{nb} batches NaN this epoch ({nan_frac:.1%}) — genuine "
            f"instability, not a stray batch. Reducing LR 10x, resuming from best checkpoint...")
        if (CKPT_DIR/'best.pt').exists():
            ckpt = torch.load(CKPT_DIR/'best.pt', weights_only=False)
            t2v.load_state_dict(ckpt['t2v']); encoder.load_state_dict(ckpt['encoder'],strict=False)
            decoder.load_state_dict(ckpt['decoder'])
            log(f"  resumed weights from best checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")
            del ckpt
        # Re-create optimizer (fresh momentum) and scheduler with reduced LR
        new_lr = HP['lr'] * (0.1 ** (patience//HP['patience'] + 1))
        all_params = list(t2v.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
        opt = optim.AdamW(all_params, lr=new_lr, weight_decay=HP['wd'])
        sched = CosineAnnealingLR(opt, T_max=HP['epochs']-epoch)  # schedule for remaining epochs
        amp_scaler = GradScaler()  # reset scale factor to avoid repeated fp16 overflow
        patience = 0
        gc.collect(); torch.cuda.empty_cache()
        log(f"  new_lr={new_lr:.2e}, optimizer/scheduler/scaler reset {mem()}")

total_train_time = time.time() - train_start
log(f"Training loop finished in {total_train_time/60:.1f} min "
    f"({len(train_losses)} epochs run) {mem()}")

# %% [cell 11] Training Curve & Log
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(train_losses, 'b-', label='Train'); ax.plot(val_losses, 'r-', label='Val')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE (masked)'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('MAE Pretraining — Reconstruction Loss')
plt.tight_layout(); plt.savefig(FIGS_DIR/'mae_training_curve.png', dpi=150); plt.show()
log(f"Saved training curve -> {FIGS_DIR/'mae_training_curve.png'}")

run_summary = {'notebook':'K01','stages':['B','C','D'],'best_val_mse':float(best_val),
       'epochs':len(train_losses),'hp':HP,'edge_dim':EDGE_DIM,'max_nodes':MAX_NODES,
       'time_range_h':round((TIME_MAX-TIME_MIN)/3.6e6,1),
       'total_train_time_min':round(total_train_time/60,1)}
with open(LOGS_DIR/'k01_log.json','w') as f: json.dump(run_summary, f, indent=2)

print(f"\nK01 COMPLETE. Best val MSE: {best_val:.6f}")
print(f"Checkpoint: {CKPT_DIR/'best.pt'}")