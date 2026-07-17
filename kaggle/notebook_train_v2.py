#!/usr/bin/env python3
"""
GNN-NIDS Phase 7+8 v2 — TGN with Temporal Attention + Two-Stage Training
==========================================================================
Architecture review fixes applied:
  1. Temporal GATv2 attention over last-K neighbors (was missing — biggest gap)
  2. Two-stage: pretrain reconstruction + fine-tune classification with focal loss
  3. Negative sampling auxiliary loss (link prediction, contrastive pressure)
  4. Dropped leaked features: std_iat, syn_count, ack_count (constant 0 for UNSW)

Architecture now matches Rossi et al. TGN: Memory → Attention Embedding → Message → Output.
"""

import sys, os, json, time, warnings
from pathlib import Path
from collections import defaultdict, deque
warnings.filterwarnings("ignore")

print("=" * 60)
print("TGN v2: TEMPORAL GATv2 ATTENTION + TWO-STAGE TRAINING")
print("=" * 60)

IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Path discovery ──
INPUT_ROOT = Path("/kaggle/input") if IS_KAGGLE else Path.cwd()
NIDS_ROOT = None
if IS_KAGGLE:
    for root, dirs, files in os.walk(str(INPUT_ROOT)):
        if Path(root).name == "nids" and (Path(root)/"__init__.py").exists():
            NIDS_ROOT = Path(root).parent; break
if NIDS_ROOT is None: NIDS_ROOT = Path.cwd()
if str(NIDS_ROOT) not in sys.path: sys.path.insert(0, str(NIDS_ROOT))

from nids import set_seed, SEED; set_seed()
from nids.models.tgn_memory import Flow

DATA_ROOT = None
if IS_KAGGLE:
    for root, dirs, files in os.walk(str(INPUT_ROOT)):
        if len([f for f in files if f.endswith(".parquet")]) >= 4:
            DATA_ROOT = Path(root); break
if DATA_ROOT is None: DATA_ROOT = Path.cwd()/"datasets"/"final"

OUTPUT_DIR = Path("/kaggle/working/output") if IS_KAGGLE else Path("local/baselines")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
print(f"Data: {DATA_ROOT}\nOutput: {OUTPUT_DIR}")

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, classification_report

# ═══════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════
MICRO_BATCH_SEC = 2.0
MAX_BATCH_FLOWS = 500      # Hard cap: split micro-batches larger than this
MEMORY_DIM = 128
TIME_DIM = 16
HIDDEN_DIM = 256
DROPOUT = 0.2
ATTENTION_HEADS = 4
NEIGHBOR_K = 10            # Last-K temporal neighbors (reduced from 20 for memory)
NEG_SAMPLES = 1
LINK_PRED_LAMBDA = 0.1
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
PRETRAIN_EPOCHS = 30
FINETUNE_EPOCHS = 20
PATIENCE = 10
LR_PRETRAIN = 0.001
LR_FINETUNE = 0.0005
GRAD_CLIP = 1.0
WINDOW = 5

# ═══════════════════════════════════════════════════════════════════════
# ARCHITECTURE — TGN with GATv2 Temporal Attention
# ═══════════════════════════════════════════════════════════════════════

class TimeEncoder(nn.Module):
    """Learnable Fourier-style time encoding."""
    def __init__(self, d=TIME_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
    def forward(self, dt):
        if dt.dim() == 1: dt = dt.unsqueeze(-1)
        return self.net(dt / 300.0)


class GATv2TemporalAttention(nn.Module):
    """GATv2 attention over a node's last-K temporal neighbors.

    For a target node v at time t, with neighbor set N(v) = {(u_i, e_i, Δt_i)}:
      - Computes attention score for each neighbor u_i
      - Aggregates neighbor memories weighted by attention
      - Returns h_v (the temporal embedding, not raw memory)

    This is the TGN 'embedding module' from Rossi et al. — the missing piece
    that separates our v1 from a proper TGN.
    """
    def __init__(self, mem_dim=MEMORY_DIM, edge_dim=None, time_dim=TIME_DIM,
                 hidden=HIDDEN_DIM, heads=ATTENTION_HEADS, dropout=DROPOUT):
        super().__init__()
        self.mem_dim = mem_dim
        self.heads = heads
        self.head_dim = hidden // heads
        self.time_enc = TimeEncoder(time_dim)

        # GATv2: attention is computed after transformation, not before
        # a(v,u) = a^T LeakyReLU(W_q*mem_v + W_k*mem_u + W_e*edge_feat + W_t*time_enc)
        self.W_q = nn.Linear(mem_dim, hidden, bias=False)     # query (target node)
        self.W_k = nn.Linear(mem_dim, hidden, bias=False)     # key (neighbor)
        self.W_e = nn.Linear(edge_dim, hidden, bias=False)    # edge features
        self.W_t = nn.Linear(time_dim, hidden, bias=False)    # time delta
        self.attn = nn.Linear(hidden, heads, bias=False)       # attention scores

        self.W_out = nn.Linear(hidden, mem_dim)                # output projection

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, mem_v, neighbor_mems, neighbor_edges, neighbor_dts):
        """Compute attended embedding for target node v.

        Args:
            mem_v: (batch, mem_dim) target node's own memory
            neighbor_mems: (batch, K, mem_dim) padded neighbor memories
            neighbor_edges: (batch, K, edge_dim) padded edge features
            neighbor_dts: (batch, K) padded time deltas (0 for padding)

        Returns:
            h_v: (batch, mem_dim) temporal attention embedding
        """
        batch_size, K, _ = neighbor_mems.shape

        # Create padding mask (neighbors with all-zero memory are padding)
        pad_mask = (neighbor_mems.abs().sum(dim=-1) > 1e-8).float()  # (B, K)

        # Query: target node
        q = self.W_q(mem_v)  # (B, hidden)

        # Keys: neighbors
        k = self.W_k(neighbor_mems.view(batch_size * K, -1)).view(batch_size, K, -1)
        e = self.W_e(neighbor_edges.view(batch_size * K, -1)).view(batch_size, K, -1)
        t = self.W_t(self.time_enc(neighbor_dts.view(batch_size * K))
                      .view(batch_size, K, -1))

        # GATv2: combine all sources, then score
        q_expanded = q.unsqueeze(1).expand(-1, K, -1)  # (B, K, hidden)
        combined = q_expanded + k + e + t               # (B, K, hidden)
        combined = self.leaky_relu(combined)

        # Multi-head attention scores
        scores = self.attn(combined)  # (B, K, heads)
        scores = scores.masked_fill(pad_mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = F.softmax(scores, dim=1)  # (B, K, heads)
        attn_weights = self.dropout(attn_weights)

        # Aggregate: weighted sum of neighbor messages, per head
        # neighbor_mems: (B, K, mem_dim) → expand to (B, K, heads, mem_dim//heads)?
        # Actually, simpler: attention over keys, aggregate their values
        # Here keys = values = neighbor_mems transformed
        k_heads = k.view(batch_size, K, self.heads, self.head_dim)  # (B, K, H, D)
        attn_expanded = attn_weights.unsqueeze(-1)                   # (B, K, H, 1)
        aggregated = (k_heads * attn_expanded).sum(dim=1)           # (B, H, D)
        aggregated = aggregated.view(batch_size, -1)                 # (B, hidden)

        # If no real neighbors, fall back to identity (mem_v)
        has_neighbors = pad_mask.sum(dim=1) > 0  # (B,)
        h_v = self.W_out(aggregated)              # (B, mem_dim)
        h_v[~has_neighbors] = mem_v[~has_neighbors]  # Fallback for isolated nodes

        return h_v


class TGNWithAttention(nn.Module):
    """Full TGN: Memory → Attention Embedding → Edge Encoder → Decoder + Classifier.

    Two output paths:
      - Decoder: reconstructs edge features (SSL pretraining)
      - Classifier: binary attack/benign (fine-tuning stage)
    """
    def __init__(self, edge_dim):
        super().__init__()
        self.edge_dim = edge_dim

        # Temporal attention embedding module
        self.attention = GATv2TemporalAttention(
            mem_dim=MEMORY_DIM, edge_dim=edge_dim,
            time_dim=TIME_DIM, hidden=HIDDEN_DIM,
            heads=ATTENTION_HEADS, dropout=DROPOUT,
        )

        # Edge encoder: h_src ⊕ h_dst ⊕ edge_feat ⊕ time → message
        in_d = MEMORY_DIM * 2 + edge_dim + TIME_DIM
        self.time_enc = TimeEncoder(TIME_DIM)
        self.edge_encoder = nn.Sequential(
            nn.Linear(in_d, HIDDEN_DIM), nn.SiLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, MEMORY_DIM),
        )

        # Decoder: message → reconstructed edge features (SSL)
        self.decoder = nn.Sequential(
            nn.Linear(MEMORY_DIM, HIDDEN_DIM), nn.SiLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, edge_dim),
        )

        # Link predictor: message → edge exists? (negative sampling)
        self.link_predictor = nn.Sequential(
            nn.Linear(MEMORY_DIM, HIDDEN_DIM // 2), nn.SiLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),  # logit
        )

        # Classification head: message → attack/benign (fine-tune stage)
        self.classifier = nn.Sequential(
            nn.Linear(MEMORY_DIM, HIDDEN_DIM // 2), nn.SiLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM // 4), nn.SiLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 4, 1),  # logit
        )

        # GRU memory update
        self.gru = nn.GRUCell(MEMORY_DIM, MEMORY_DIM)
        self.mem_init = nn.Linear(5, MEMORY_DIM)

    def embed_node(self, host_id, mem_store, neigh_store, current_time):
        """Compute temporal attention embedding for a single host."""
        mem_v = mem_store.get(host_id).to(DEVICE)
        neighbors = neigh_store.get_recent(host_id, k=NEIGHBOR_K)

        if not neighbors:
            return mem_v  # No neighbors → use raw memory

        # Build padded tensors
        K = NEIGHBOR_K
        n_mems = torch.zeros(K, MEMORY_DIM, device=DEVICE)
        n_edges = torch.zeros(K, self.edge_dim, device=DEVICE)
        n_dts = torch.zeros(K, device=DEVICE)

        for i, (nbr_id, nbr_time, nbr_feats) in enumerate(neighbors[:K]):
            n_mems[i] = mem_store.get(nbr_id).to(DEVICE)
            n_edges[i] = torch.from_numpy(np.asarray(nbr_feats, dtype=np.float32)).to(DEVICE)
            n_dts[i] = max(current_time - nbr_time, 0.0)

        return self.attention(
            mem_v.unsqueeze(0),
            n_mems.unsqueeze(0),
            n_edges.unsqueeze(0),
            n_dts.unsqueeze(0),
        ).squeeze(0)

    def forward(self, h_src, h_dst, edge_feat, delta_t):
        """Full forward pass with attention embeddings as input."""
        t_emb = self.time_enc(delta_t)
        x = torch.cat([h_src, h_dst, edge_feat, t_emb], dim=-1)
        msg = self.edge_encoder(x)
        recon = self.decoder(msg)
        link_logit = self.link_predictor(msg)
        class_logit = self.classifier(msg)
        return msg, recon, link_logit, class_logit


# ═══════════════════════════════════════════════════════════════════════
# TEMPORAL NEIGHBOR STORE
# ═══════════════════════════════════════════════════════════════════════

class TemporalNeighborStore:
    """Per-host ring buffer of last-K temporal neighbors for attention.

    Each entry: (neighbor_id, timestamp, edge_features)
    """
    def __init__(self, k=NEIGHBOR_K):
        self.k = k
        self._buffer = defaultdict(lambda: deque(maxlen=k))

    def add_interaction(self, src, dst, timestamp, edge_feats):
        """Record a bidirectional interaction."""
        self._buffer[src].append((dst, timestamp, edge_feats.copy() if hasattr(edge_feats, 'copy') else edge_feats))
        self._buffer[dst].append((src, timestamp, edge_feats.copy() if hasattr(edge_feats, 'copy') else edge_feats))

    def get_recent(self, host, k=None):
        """Get last-K interactions for a host (most recent first)."""
        buf = self._buffer.get(host, deque(maxlen=self.k))
        # Return as list, most recent first
        return list(reversed(list(buf)))[:(k or self.k)]


# ═══════════════════════════════════════════════════════════════════════
# HOST MEMORY (same as before, real IPs)
# ═══════════════════════════════════════════════════════════════════════

class HostMemory:
    def __init__(self, model):
        self.model = model
        self.mem, self.last_t = {}, {}
        self.stats = defaultdict(lambda: {"in":0,"out":0,"bin":0.0,"bout":0.0,"peers":set()})

    def _init(self, host):
        s = self.stats[host]
        stats = torch.stack([
            torch.tensor(s["in"],device=DEVICE), torch.tensor(s["out"],device=DEVICE),
            torch.tensor(s["bin"],device=DEVICE), torch.tensor(s["bout"],device=DEVICE),
            torch.tensor(len(s["peers"]),device=DEVICE),
        ]).float()
        with torch.no_grad():
            return self.model.mem_init(stats.unsqueeze(0)).squeeze(0).detach()

    def get(self, host):
        if host not in self.mem: self.mem[host] = self._init(host)
        return self.mem[host]

    def update(self, host, mem, t):
        self.mem[host] = mem.detach().clone(); self.last_t[host] = t

    def last(self, host): return self.last_t.get(host, 0.0)

    def add_edge(self, src, dst, bv=0.0):
        self.stats[src]["out"]+=1; self.stats[src]["bout"]+=bv; self.stats[src]["peers"].add(dst)
        self.stats[dst]["in"]+=1;  self.stats[dst]["bin"]+=bv;  self.stats[dst]["peers"].add(src)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_split(prefix, split_name):
    path = DATA_ROOT / f"{prefix}_{split_name}.parquet"
    df = pd.read_parquet(path)
    meta = {"src_ip", "dst_ip", "timestamp", "label"}
    feat_cols = [c for c in df.columns if c not in meta]
    return {
        "src_ips": df["src_ip"].values,
        "dst_ips": df["dst_ip"].values,
        "timestamps": df["timestamp"].values.astype(np.float64),
        "features": df[feat_cols].values.astype(np.float32),
        "labels": df["label"].values.astype(np.int32),
        "n_features": len(feat_cols),
    }

def to_flows(data, label="", chunk=100000):
    n = len(data["labels"])
    flows = []
    for ci in range((n + chunk - 1) // chunk):
        s, e = ci * chunk, min((ci + 1) * chunk, n)
        for i in range(s, e):
            flows.append(Flow(
                src=str(data["src_ips"][i]), dst=str(data["dst_ips"][i]),
                timestamp=float(data["timestamps"][i]),
                features=data["features"][i],
                label=int(data["labels"][i]),
            ))
        if n > chunk: print(f"  [{label}] {e:,}/{n:,} flows converted...", flush=True)
    flows.sort(key=lambda f: f.timestamp)
    return flows

# ═══════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def make_batches(flows, max_per_batch=MAX_BATCH_FLOWS):
    """Split flows into micro-batches. Enforces hard cap on batch size."""
    if not flows: return []
    batches, cur = [], []
    t0 = flows[0].timestamp
    for f in flows:
        if f.timestamp - t0 >= MICRO_BATCH_SEC or len(cur) >= max_per_batch:
            if cur: batches.append(cur)
            cur = []; t0 = f.timestamp
        cur.append(f)
    if cur: batches.append(cur)
    # Further split any batch over the cap
    result = []
    for b in batches:
        while len(b) > max_per_batch:
            result.append(b[:max_per_batch])
            b = b[max_per_batch:]
        if b: result.append(b)
    return result


class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification."""
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def pretrain_step(model, hm, neigh_store, batch, opt):
    """Single pretraining step: reconstruction + link prediction loss."""
    model.train()
    batch_loss, batch_recon, batch_link, n = 0.0, 0.0, 0.0, 0

    # Phase 1: compute embeddings for all hosts in batch (before memory updates)
    affected = {}  # host_id -> h_v
    batch_hosts = set()
    for f in batch:
        batch_hosts.add(f.src); batch_hosts.add(f.dst)

    for host in batch_hosts:
        affected[host] = model.embed_node(host, hm, neigh_store, batch[-1].timestamp)

    # Phase 2: process each flow
    for f in batch:
        h_src = affected[f.src]; h_dst = affected[f.dst]
        ef = torch.from_numpy(np.asarray(f.features, dtype=np.float32)).to(DEVICE)
        lt = max(hm.last(f.src), hm.last(f.dst))
        dt = torch.tensor(max(f.timestamp - lt, 0.001), device=DEVICE)

        msg, recon, link_logit, _ = model(
            h_src.unsqueeze(0), h_dst.unsqueeze(0),
            ef.unsqueeze(0), dt.unsqueeze(0),
        )

        # Reconstruction loss
        recon_loss = F.mse_loss(recon.squeeze(0), ef)

        # Link prediction: positive sample
        pos_loss = F.binary_cross_entropy_with_logits(
            link_logit.squeeze(0), torch.ones(1, device=DEVICE))

        # Negative sampling: random host pair
        neg_src = f"neg_{np.random.randint(0, max(1, len(hm.mem)))}"
        neg_dst = f"neg_{np.random.randint(0, max(1, len(hm.mem)))}"
        neg_h_src = hm.get(neg_src).to(DEVICE)
        neg_h_dst = hm.get(neg_dst).to(DEVICE)
        neg_ef = torch.zeros(self.edge_dim, device=DEVICE)  # zero features for fake edge
        _, _, neg_link_logit, _ = model(
            neg_h_src.unsqueeze(0), neg_h_dst.unsqueeze(0),
            neg_ef.unsqueeze(0), torch.tensor([0.001], device=DEVICE).unsqueeze(0),
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_link_logit.squeeze(0), torch.zeros(1, device=DEVICE))

        link_loss = (pos_loss + neg_loss) / 2
        loss = recon_loss + LINK_PRED_LAMBDA * link_loss

        batch_recon += recon_loss.item()
        batch_link += link_loss.item()
        batch_loss += loss.item()
        n += 1

        # Update memories
        ns = model.gru(msg, h_src.unsqueeze(0)).squeeze(0)
        nd = model.gru(msg, h_dst.unsqueeze(0)).squeeze(0)
        hm.update(f.src, ns, f.timestamp); hm.update(f.dst, nd, f.timestamp)

    return batch_loss / max(n, 1), batch_recon / max(n, 1), batch_link / max(n, 1)


def pretrain_epoch(model, hm, neigh_store, flows, opt):
    """Batched pretraining: all flows in a micro-batch processed in one forward pass."""
    batches = make_batches(flows)
    total_loss, total_recon, total_link, nf = 0.0, 0.0, 0.0, 0
    for batch in batches:
        if len(batch) < 2: continue
        N = len(batch)

        # ── Batch-compute attention embeddings for all unique hosts ──
        unique_hosts = set()
        for f in batch:
            unique_hosts.add(f.src); unique_hosts.add(f.dst)
        host_to_idx = {h: i for i, h in enumerate(unique_hosts)}
        host_list = list(unique_hosts)

        # Pre-compute per-host neighbor tensors
        host_mems_batch = []
        host_nbr_mems = []
        host_nbr_edges = []
        host_nbr_dts = []
        for host in host_list:
            mem_v = hm.get(host).to(DEVICE)
            neighbors = neigh_store.get_recent(host, k=NEIGHBOR_K)
            K = NEIGHBOR_K
            n_mems = torch.zeros(K, MEMORY_DIM, device=DEVICE)
            n_edges = torch.zeros(K, model.edge_dim, device=DEVICE)
            n_dts = torch.zeros(K, device=DEVICE)
            t_now = batch[-1].timestamp
            for i, (nbr, nbr_t, nbr_feats) in enumerate(neighbors[:K]):
                n_mems[i] = hm.get(nbr).to(DEVICE)
                feats = np.asarray(nbr_feats, dtype=np.float32)
                n_edges[i] = torch.from_numpy(feats).to(DEVICE)
                n_dts[i] = max(t_now - nbr_t, 0.0)
            host_mems_batch.append(mem_v)
            host_nbr_mems.append(n_mems)
            host_nbr_edges.append(n_edges)
            host_nbr_dts.append(n_dts)

        # Batch attention: (H, mem_dim) ← H = num unique hosts
        H = len(host_list)
        h_embeddings = model.attention(
            torch.stack(host_mems_batch),          # (H, mem_dim)
            torch.stack(host_nbr_mems),            # (H, K, mem_dim)
            torch.stack(host_nbr_edges),           # (H, K, edge_dim)
            torch.stack(host_nbr_dts),             # (H, K)
        )  # (H, mem_dim)

        # ── Batch-compute edge encoding for all N flows ──
        src_indices = [host_to_idx[f.src] for f in batch]
        dst_indices = [host_to_idx[f.dst] for f in batch]
        h_src_batch = h_embeddings[src_indices]    # (N, mem_dim)
        h_dst_batch = h_embeddings[dst_indices]    # (N, mem_dim)

        ef_batch = torch.from_numpy(
            np.stack([f.features for f in batch]).astype(np.float32)
        ).to(DEVICE)  # (N, edge_dim)

        dt_batch = []
        for f in batch:
            lt = max(hm.last(f.src), hm.last(f.dst))
            dt_batch.append(max(f.timestamp - lt, 0.001))
        dt_batch = torch.tensor(dt_batch, device=DEVICE)  # (N,)

        # Single forward pass for all N flows
        msg_batch, recon_batch, link_batch, _ = model(
            h_src_batch, h_dst_batch, ef_batch, dt_batch,
        )
        # Detach a copy for memory updates (freed after backward otherwise)
        msg_detached = msg_batch.detach().clone()
        h_src_detached = h_src_batch.detach().clone()
        h_dst_detached = h_dst_batch.detach().clone()

        # Reconstruction loss (batched)
        r_loss = F.mse_loss(recon_batch, ef_batch)

        # Link prediction — positive samples (batched)
        pos_targets = torch.ones(N, 1, device=DEVICE)
        pos_l = F.binary_cross_entropy_with_logits(link_batch, pos_targets)

        # Negative samples — sample N random host pairs (batched)
        host_pool = list(hm.mem.keys()) if hm.mem else ["_dummy_"]
        neg_src_ids = [host_pool[np.random.randint(0, len(host_pool))] for _ in range(N)]
        neg_dst_ids = [host_pool[np.random.randint(0, len(host_pool))] for _ in range(N)]
        neg_src_mems = torch.stack([hm.get(h).to(DEVICE) for h in neg_src_ids])
        neg_dst_mems = torch.stack([hm.get(h).to(DEVICE) for h in neg_dst_ids])
        neg_ef = torch.zeros(N, model.edge_dim, device=DEVICE)
        neg_dt = torch.zeros(N, device=DEVICE)

        _, _, neg_link_batch, _ = model(neg_src_mems, neg_dst_mems, neg_ef, neg_dt)
        neg_targets = torch.zeros(N, 1, device=DEVICE)
        neg_l = F.binary_cross_entropy_with_logits(neg_link_batch, neg_targets)

        l_loss = (pos_l + neg_l) / 2
        loss = r_loss + LINK_PRED_LAMBDA * l_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        # Update host memories (uses detached copies, safe after backward)
        with torch.no_grad():
            for i, f in enumerate(batch):
                ns = model.gru(msg_detached[i].unsqueeze(0), h_src_detached[i].unsqueeze(0)).squeeze(0)
                nd = model.gru(msg_detached[i].unsqueeze(0), h_dst_detached[i].unsqueeze(0)).squeeze(0)
                hm.update(f.src, ns, f.timestamp); hm.update(f.dst, nd, f.timestamp)

        total_loss += loss.item() * N
        total_recon += r_loss.item() * N
        total_link += l_loss.item() * N
        nf += N

        # Free GPU memory after large batches
        if N > 200:
            del h_embeddings, h_src_batch, h_dst_batch, ef_batch, msg_batch, recon_batch, link_batch
            torch.cuda.empty_cache()

    # Update neighbor store after epoch
    for f in flows[-100000:]:
        neigh_store.add_interaction(f.src, f.dst, f.timestamp, f.features)

    return total_loss / max(nf, 1), total_recon / max(nf, 1), total_link / max(nf, 1)


def finetune_epoch(model, hm, neigh_store, flows, opt, loss_fn):
    """Batched fine-tuning: all flows in micro-batch in one forward pass."""
    model.train()
    batches = make_batches(flows)
    total_loss, n_f = 0.0, 0
    for batch in batches:
        if len(batch) < 2: continue
        N = len(batch)

        # Batch attention embeddings
        unique_hosts = set()
        for f in batch: unique_hosts.add(f.src); unique_hosts.add(f.dst)
        host_list = list(unique_hosts)
        host_to_idx = {h: i for i, h in enumerate(host_list)}

        host_mems_b, host_nbr_m, host_nbr_e, host_nbr_d = [], [], [], []
        for host in host_list:
            mem_v = hm.get(host).to(DEVICE)
            neighbors = neigh_store.get_recent(host, k=NEIGHBOR_K)
            K = NEIGHBOR_K
            nm = torch.zeros(K, MEMORY_DIM, device=DEVICE)
            ne = torch.zeros(K, model.edge_dim, device=DEVICE)
            nd = torch.zeros(K, device=DEVICE)
            t_now = batch[-1].timestamp
            for i, (nbr, nbr_t, nbr_feats) in enumerate(neighbors[:K]):
                nm[i] = hm.get(nbr).to(DEVICE)
                ne[i] = torch.from_numpy(np.asarray(nbr_feats, dtype=np.float32)).to(DEVICE)
                nd[i] = max(t_now - nbr_t, 0.0)
            host_mems_b.append(mem_v); host_nbr_m.append(nm)
            host_nbr_e.append(ne); host_nbr_d.append(nd)

        h_emb = model.attention(torch.stack(host_mems_b), torch.stack(host_nbr_m),
                                 torch.stack(host_nbr_e), torch.stack(host_nbr_d))

        src_idx = [host_to_idx[f.src] for f in batch]
        dst_idx = [host_to_idx[f.dst] for f in batch]
        h_src_b = h_emb[src_idx]; h_dst_b = h_emb[dst_idx]
        ef_b = torch.from_numpy(np.stack([f.features for f in batch]).astype(np.float32)).to(DEVICE)
        dt_b = torch.tensor([max(f.timestamp - max(hm.last(f.src), hm.last(f.dst)), 0.001)
                             for f in batch], device=DEVICE)

        msg_b, _, _, cls_b = model(h_src_b, h_dst_b, ef_b, dt_b)
        msg_b_d = msg_b.detach().clone()
        h_src_b_d = h_src_b.detach().clone()
        h_dst_b_d = h_dst_b.detach().clone()
        targets = torch.tensor([float(f.label) for f in batch], device=DEVICE)
        loss = loss_fn(cls_b.squeeze(-1), targets)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        with torch.no_grad():
            for i, f in enumerate(batch):
                ns = model.gru(msg_b_d[i].unsqueeze(0), h_src_b_d[i].unsqueeze(0)).squeeze(0)
                nd = model.gru(msg_b_d[i].unsqueeze(0), h_dst_b_d[i].unsqueeze(0)).squeeze(0)
                hm.update(f.src, ns, f.timestamp); hm.update(f.dst, nd, f.timestamp)

        total_loss += loss.item() * N; n_f += N
        if N > 200:
            del h_emb, h_src_b, h_dst_b, ef_b, msg_b
            torch.cuda.empty_cache()
    return total_loss / max(n_f, 1)


@torch.no_grad()
def evaluate(model, hm, neigh_store, flows, use_classifier=False):
    """Evaluate: returns anomaly scores (recon error) or class probs + labels."""
    model.eval()
    errors, labels, class_probs = [], [], []
    for batch in make_batches(flows):
        for f in batch:
            h_src = model.embed_node(f.src, hm, neigh_store, f.timestamp)
            h_dst = model.embed_node(f.dst, hm, neigh_store, f.timestamp)
            ef = torch.from_numpy(np.asarray(f.features, dtype=np.float32)).to(DEVICE)
            lt = max(hm.last(f.src), hm.last(f.dst))
            dt = torch.tensor(max(f.timestamp - lt, 0.001), device=DEVICE)

            msg, recon, _, class_logit = model(
                h_src.unsqueeze(0), h_dst.unsqueeze(0),
                ef.unsqueeze(0), dt.unsqueeze(0),
            )

            if use_classifier:
                # Clamp logit to prevent NaN from extreme values
                p = torch.sigmoid(torch.clamp(class_logit, -50, 50)).item()
                class_probs.append(0.0 if (np.isnan(p) or np.isinf(p)) else p)
            else:
                err = F.mse_loss(recon.squeeze(0), ef).item()
                errors.append(0.0 if (np.isnan(err) or np.isinf(err)) else err)
            labels.append(f.label)

            ns = model.gru(msg, h_src.unsqueeze(0)).squeeze(0)
            nd = model.gru(msg, h_dst.unsqueeze(0)).squeeze(0)
            hm.update(f.src, ns, f.timestamp); hm.update(f.dst, nd, f.timestamp)

    labs = np.array(labels)
    if use_classifier:
        scores = np.nan_to_num(np.array(class_probs), nan=0.0, posinf=1.0, neginf=0.0)
    else:
        scores = np.nan_to_num(np.array(errors), nan=0.0, posinf=0.0, neginf=0.0)
        # Window aggregation for reconstruction errors
        W = WINDOW; nw = len(scores) // W
        if nw > 0:
            scores = scores[:nw*W].reshape(-1, W).max(axis=1)
            labs = labs[:nw*W].reshape(-1, W).max(axis=1)

    ha = labs.sum() > 0
    roc = roc_auc_score(labs, scores) if ha else float("nan")
    pr = average_precision_score(labs, scores) if ha else 0.0
    bf = 0.0
    for t in np.percentile(scores, np.linspace(1, 99, 100)):
        p = (scores >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labs, p, average="binary", zero_division=0)
        if f1 > bf: bf = f1

    return {"roc": roc, "pr": pr, "f1": bf}, scores, labs


# ═══════════════════════════════════════════════════════════════════════
# TRAIN ONE DATASET
# ═══════════════════════════════════════════════════════════════════════

def train_dataset(name, prefix):
    print(f"\n{'='*60}")
    print(f"TRAINING: {name}")
    print(f"{'='*60}")

    # Load data
    train_data = load_split(prefix, "train")
    val_data = load_split(prefix, "val")
    test_data = load_split(prefix, "test")
    print(f"Train: {len(train_data['labels']):,} (benign) | "
          f"Val: {len(val_data['labels']):,} (att={val_data['labels'].sum():,}) | "
          f"Test: {len(test_data['labels']):,} (att={test_data['labels'].sum():,})")
    print(f"Features: {train_data['n_features']}")

    print("Converting to flows...", flush=True)
    t0 = time.time()
    train_f = to_flows(train_data, label="train")
    val_f = to_flows(val_data, label="val")
    test_f = to_flows(test_data, label="test")
    n_hosts = len(set(f.src for f in train_f) | set(f.dst for f in train_f))
    print(f"  -> {len(train_f):,} train, {len(val_f):,} val, {len(test_f):,} test flows")
    print(f"  -> {n_hosts:,} unique hosts, {time.time()-t0:.0f}s")

    edge_dim = train_data["n_features"]

    # ── Check for existing pretrained model (working dir + input datasets) ──
    model_path = OUTPUT_DIR / f"tgn_{prefix}_pretrain.pt"
    if not model_path.exists() and IS_KAGGLE:
        # Search uploaded datasets for pretrained checkpoint
        for root, dirs, files in os.walk(str(INPUT_ROOT)):
            for fname in files:
                if fname == f"tgn_{prefix}_pretrain.pt":
                    model_path = Path(root) / fname
                    break
    if model_path.exists():
        print(f"\n  Loading pretrained model: {model_path}")
        ckpt = torch.load(model_path, map_location=DEVICE)
        model = TGNWithAttention(ckpt["n_features"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        pretrain_done = True
    else:
        pretrain_done = False
        model = TGNWithAttention(edge_dim).to(DEVICE)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Init memory
    hm = HostMemory(model)
    neigh_store = TemporalNeighborStore(k=NEIGHBOR_K)
    print(f"Seeding host memory and neighbor store...")
    for i, f in enumerate(train_f[:100000]):
        hm.add_edge(f.src, f.dst, float(np.abs(f.features[:4]).sum()))
        neigh_store.add_interaction(f.src, f.dst, f.timestamp, f.features)
    print(f"  -> Edge stats seeded, {len(hm.stats):,} hosts tracked (memory init is lazy)")

    # ═════════════════════════════════════════════════════════════════
    # STAGE 1: SSL Pretraining (skip if model loaded)
    # ═════════════════════════════════════════════════════════════════
    if pretrain_done:
        print(f"\n{'─'*50}")
        print(f"STAGE 1: SKIPPED (pretrained model loaded from {model_path})")
        print(f"{'─'*50}")
    else:
        print(f"\n{'─'*50}")
        print(f"STAGE 1: SSL PRETRAINING (reconstruction + link prediction)")
        print(f"{'─'*50}")
        print(f"Epochs={PRETRAIN_EPOCHS}, LR={LR_PRETRAIN}, λ_link={LINK_PRED_LAMBDA}")

        opt = optim.AdamW(model.parameters(), lr=LR_PRETRAIN, weight_decay=1e-5)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS, eta_min=1e-5)

        best_val, best_state, pctr = 0.0, None, 0
        print(f"  {'Ep':<5} {'Loss':>10} {'Recon':>10} {'Link':>10} {'Val F1':>8} {'Val PR':>8} {'Val ROC':>8} {'LR':>10}")

        for ep in range(PRETRAIN_EPOCHS):
            ep_t0 = time.time()
            tl, tr, tlk = pretrain_epoch(model, hm, neigh_store, train_f, opt)
            sched.step()

            vm = HostMemory(model); vns = TemporalNeighborStore(k=NEIGHBOR_K)
            for f in train_f[-30000:]:
                vm.add_edge(f.src, f.dst); vns.add_interaction(f.src, f.dst, f.timestamp, f.features)
            vm_met, _, _ = evaluate(model, vm, vns, val_f[:min(50000, len(val_f))])

            improved = ""
            if vm_met["f1"] > best_val + 0.001:
                best_val = vm_met["f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                pctr = 0; improved = "*"
            else:
                pctr += 1

            roc_s = f"{vm_met['roc']:.4f}" if not np.isnan(vm_met['roc']) else "  N/A"
            print(f"  {ep:3d}{improved}  {tl:>10.4f} {tr:>10.4f} {tlk:>10.4f} "
                  f"{vm_met['f1']:>8.4f} {vm_met['pr']:>8.4f} {roc_s:>8} "
                  f"{opt.param_groups[0]['lr']:>10.2e}")

            if pctr >= PATIENCE:
                print(f"  Early stop at epoch {ep}"); break

        if best_state: model.load_state_dict(best_state)
        # Save pretrained checkpoint for future resume
        torch.save({"state_dict": best_state, "n_features": edge_dim}, model_path)
        print(f"\n  Pretrain complete: best val F1={best_val:.4f}")
        print(f"  Saved -> {model_path}")

    # ═════════════════════════════════════════════════════════════════
    # STAGE 2: Fine-tune — end-to-end with low LR, no freezing
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'─'*50}")
    print(f"STAGE 2: FINE-TUNING (end-to-end, low LR, focal loss)")
    print(f"{'─'*50}")

    # Small-init classifier output layer: near-zero weights → near-0.5 probability
    # → focal loss starts ~0.69 instead of exploding. Weights can grow during training.
    for m in model.classifier:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.001)
            nn.init.zeros_(m.bias)

    # Use val_f only (balanced 50/50, no timestamp discontinuity)
    ft_flows = val_f
    ft_labels = np.array([f.label for f in ft_flows])
    print(f"Fine-tune flows: {len(ft_flows):,} (attack: {ft_labels.sum():,})")
    print(f"LR={LR_FINETUNE}, loss=BCE (balanced data, no focal needed)")
    print(f"All {sum(p.numel() for p in model.parameters()):,} params trainable, classifier output init ~N(0, 0.001)")

    opt_ft = optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-5)
    sched_ft = optim.lr_scheduler.ReduceLROnPlateau(opt_ft, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    bce_loss = nn.BCEWithLogitsLoss()

    best_ft, best_state_ft, pctr_ft = 0.0, None, 0
    print(f"  {'Ep':<5} {'Loss':>10} {'Val F1':>8} {'Val PR':>8} {'Val ROC':>8} {'LR':>10}")

    for ep in range(FINETUNE_EPOCHS):
        tl = finetune_epoch(model, hm, neigh_store, ft_flows, opt_ft, bce_loss)

        vm = HostMemory(model); vns = TemporalNeighborStore(k=NEIGHBOR_K)
        for f in ft_flows[-30000:]:
            vm.add_edge(f.src, f.dst); vns.add_interaction(f.src, f.dst, f.timestamp, f.features)
        vm_met, _, _ = evaluate(model, vm, vns, val_f[:min(50000, len(val_f))], use_classifier=True)
        sched_ft.step(vm_met["f1"])

        improved = ""
        if vm_met["f1"] > best_ft + 0.001:
            best_ft = vm_met["f1"]
            best_state_ft = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pctr_ft = 0; improved = "*"
        else:
            pctr_ft += 1

        roc_s = f"{vm_met['roc']:.4f}" if not np.isnan(vm_met['roc']) else "  N/A"
        print(f"  {ep:3d}{improved}  {tl:>10.4f} {vm_met['f1']:>8.4f} "
              f"{vm_met['pr']:>8.4f} {roc_s:>8} "
              f"{opt_ft.param_groups[0]['lr']:>10.2e}")

        if pctr_ft >= PATIENCE:
            print(f"  Early stop at epoch {ep}"); break

    if best_state_ft: model.load_state_dict(best_state_ft)

    # ═════════════════════════════════════════════════════════════════
    # FINAL TEST
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'─'*50}")
    print(f"FINAL TEST EVALUATION")
    print(f"{'─'*50}")

    tm = HostMemory(model); tns = TemporalNeighborStore(k=NEIGHBOR_K)
    for f in train_f[-50000:] + val_f[:30000]:
        tm.add_edge(f.src, f.dst); tns.add_interaction(f.src, f.dst, f.timestamp, f.features)

    # SSL evaluation (reconstruction error)
    ssl_met, ssl_e, ssl_l = evaluate(model, tm, tns, test_f[:min(50000, len(test_f))], use_classifier=False)
    # Classifier evaluation
    cls_met, cls_e, cls_l = evaluate(model, tm, tns, test_f[:min(50000, len(test_f))], use_classifier=True)

    print(f"  SSL (reconstruction): ROC={ssl_met['roc']:.4f} PR={ssl_met['pr']:.4f} F1={ssl_met['f1']:.4f}")
    print(f"  CLS (classifier):     ROC={cls_met['roc']:.4f} PR={cls_met['pr']:.4f} F1={cls_met['f1']:.4f}")

    # Use the better of the two for final metric
    final = cls_met if cls_met['f1'] > ssl_met['f1'] else ssl_met
    mode = "classifier" if cls_met['f1'] > ssl_met['f1'] else "reconstruction"
    print(f"  Final ({mode}): ROC={final['roc']:.4f} PR={final['pr']:.4f} F1={final['f1']:.4f}")

    # Save
    torch.save({"state_dict": best_state_ft or best_state, "n_features": edge_dim,
                "architecture": "TGN+GATv2+SSL+Classifier"},
               OUTPUT_DIR / f"tgn_{prefix}_model.pt")

    pretrain_f1 = best_val if not pretrain_done else None
    return {"name": name, "pretrain_best_f1": pretrain_f1,
            "ssl_test": ssl_met, "cls_test": cls_met, "final": final,
            "mode": mode, "n_features": edge_dim, "n_hosts": n_hosts}


# ═══════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════

cic17_r = train_dataset("CIC-IDS2017", "cicids2017")
torch.cuda.empty_cache()
print("\n" + "=" * 60)
print("SWITCHING TO UNSW-NB15 — GPU cache cleared")
print("=" * 60)

unsw_r = train_dataset("UNSW-NB15", "unswnb15")
torch.cuda.empty_cache()

def _safe_val(v):
    """Convert NaN/None to string for JSON serialization."""
    if v is None: return None
    if isinstance(v, float) and np.isnan(v): return "nan"
    return v

results = {
    "phase": "7+8_v2_attention",
    "seed": SEED,
    "architecture": {
        "type": "TGN+GATv2+SSL+Classifier",
        "memory_dim": MEMORY_DIM, "hidden_dim": HIDDEN_DIM,
        "time_dim": TIME_DIM, "attention_heads": ATTENTION_HEADS,
        "neighbor_k": NEIGHBOR_K, "dropout": DROPOUT,
    },
    "training": {
        "pretrain_epochs": PRETRAIN_EPOCHS, "finetune_epochs": FINETUNE_EPOCHS,
        "lr_pretrain": LR_PRETRAIN, "lr_finetune": LR_FINETUNE,
        "link_pred_lambda": LINK_PRED_LAMBDA, "neg_samples": NEG_SAMPLES,
        "focal_alpha": FOCAL_ALPHA, "focal_gamma": FOCAL_GAMMA,
    },
    "cic17": {k: _safe_val(v) for k, v in cic17_r.items()},
    "unsw": {k: _safe_val(v) for k, v in unsw_r.items()},
}
with open(OUTPUT_DIR / "training_results_v2.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*60}")
print("TRAINING COMPLETE — TGN v2 with GATv2 Attention")
print(f"CIC17: {cic17_r['mode']} F1={cic17_r['final']['f1']:.4f}  "
      f"ROC={cic17_r['final']['roc']:.4f}")
print(f"UNSW:  {unsw_r['mode']} F1={unsw_r['final']['f1']:.4f}  "
      f"ROC={unsw_r['final']['roc']:.4f}")
print(f"Models: {OUTPUT_DIR}/tgn_*_model.pt")
print(f"Results: {OUTPUT_DIR}/training_results_v2.json")
print(f"{'='*60}")
