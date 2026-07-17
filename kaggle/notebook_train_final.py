#!/usr/bin/env python3
"""
GNN-NIDS Phase 7+8 — FINAL Training Notebook
==============================================
Uses pre-built final datasets (IPs + scaled features + splits in one file).
No merging, scaling, or splitting at runtime. Just load, convert, train.

Upload datasets/final/ to Kaggle as a dataset named 'nids-final'.
Each file: src_ip, dst_ip, timestamp, 26|35 feature columns, label
"""

import sys, os, json, time, warnings
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings("ignore")

print("=" * 60)
print("TGN TRAINING — REAL IPs, SCALED, PRE-SPLIT")
print("=" * 60)

IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Find package ──
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

# ── Find data ──
DATA_ROOT = None
if IS_KAGGLE:
    for root, dirs, files in os.walk(str(INPUT_ROOT)):
        parqs = [f for f in files if f.endswith(".parquet")]
        if len(parqs) >= 4:
            DATA_ROOT = Path(root); break
if DATA_ROOT is None: DATA_ROOT = Path.cwd()/"datasets"/"final"

OUTPUT_DIR = Path("/kaggle/working/output") if IS_KAGGLE else Path("local/baselines")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
print(f"Data: {DATA_ROOT}")
print(f"Output: {OUTPUT_DIR}")

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

# ── Hyperparams ──
MICRO_BATCH_SEC = 2.0
MEMORY_DIM = 128
TIME_DIM = 16
HIDDEN_DIM = 256
DROPOUT = 0.15
EPOCHS = 40
PATIENCE = 10
LR = 0.001
GRAD_CLIP = 1.0
WINDOW = 5  # anomaly aggregation window

# ═══════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════

class TimeEncoder(nn.Module):
    def __init__(self, d=TIME_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1,d), nn.SiLU(), nn.Linear(d,d))
    def forward(self, dt):
        if dt.dim()==1: dt=dt.unsqueeze(-1)
        return self.net(dt/300.0)

class TGNModel(nn.Module):
    def __init__(self, edge_d):
        super().__init__()
        self.time_enc = TimeEncoder()
        in_d = MEMORY_DIM*2 + edge_d + TIME_DIM
        self.encoder = nn.Sequential(
            nn.Linear(in_d, HIDDEN_DIM), nn.SiLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, MEMORY_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(MEMORY_DIM, HIDDEN_DIM), nn.SiLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, edge_d),
        )
        self.gru = nn.GRUCell(MEMORY_DIM, MEMORY_DIM)
        self.mem_init = nn.Linear(5, MEMORY_DIM)

    def forward(self, sm, dm, ef, dt):
        te = self.time_enc(dt)
        msg = self.encoder(torch.cat([sm,dm,ef,te], dim=-1))
        recon = self.decoder(msg)
        return msg, recon

# ═══════════════════════════════════════════════════════════════════════
# HOST MEMORY (real IPs now!)
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
        return self.model.mem_init(stats.unsqueeze(0)).squeeze(0)

    def get(self, host):
        if host not in self.mem: self.mem[host] = self._init(host)
        return self.mem[host]

    def update(self, host, mem, t):
        self.mem[host]=mem.detach().clone(); self.last_t[host]=t

    def last(self, host): return self.last_t.get(host, 0.0)

    def add_edge(self, src, dst, bv=0.0):
        self.stats[src]["out"]+=1; self.stats[src]["bout"]+=bv; self.stats[src]["peers"].add(dst)
        self.stats[dst]["in"]+=1;  self.stats[dst]["bin"]+=bv;  self.stats[dst]["peers"].add(src)

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING — one-liner, everything is pre-built
# ═══════════════════════════════════════════════════════════════════════

def load_split(prefix, split_name):
    """Load a pre-built split file. Returns numpy arrays for fast iteration."""
    path = DATA_ROOT / f"{prefix}_{split_name}.parquet"
    df = pd.read_parquet(path)
    # Identify feature columns (everything except src_ip, dst_ip, timestamp, label)
    meta = {"src_ip", "dst_ip", "timestamp", "label"}
    feat_cols = [c for c in df.columns if c not in meta]
    # Pre-extract as numpy for fast iteration
    return {
        "src_ips": df["src_ip"].values,
        "dst_ips": df["dst_ip"].values,
        "timestamps": df["timestamp"].values.astype(np.float64),
        "features": df[feat_cols].values.astype(np.float32),
        "labels": df["label"].values.astype(np.int32),
        "n_features": len(feat_cols),
    }

def to_flows(data, label="", max_n=None, chunk_size=100000):
    """Convert pre-extracted numpy arrays to Flow objects. Prints progress."""
    n = len(data["labels"])
    if max_n: n = min(n, max_n)
    flows = []
    n_chunks = (n + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, n)
        for i in range(start, end):
            flows.append(Flow(
                src=str(data["src_ips"][i]),
                dst=str(data["dst_ips"][i]),
                timestamp=float(data["timestamps"][i]),
                features=data["features"][i],
                label=int(data["labels"][i]),
            ))
        if n_chunks > 1:
            print(f"  [{label}] {end:,}/{n:,} flows converted...", flush=True)
    print(f"  [{label}] Sorting {len(flows):,} flows by timestamp...", flush=True)
    flows.sort(key=lambda f: f.timestamp)
    return flows

# ═══════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def make_batches(flows):
    if not flows: return []
    batches, cur = [], []
    t0 = flows[0].timestamp
    for f in flows:
        if f.timestamp - t0 >= MICRO_BATCH_SEC:
            if cur: batches.append(cur)
            cur=[]; t0=f.timestamp
        cur.append(f)
    if cur: batches.append(cur)
    return batches

def train_epoch(model, hm, flows):
    model.train()
    batches = make_batches(flows)
    total_loss, n_f = 0.0, 0
    for batch in batches:
        if len(batch)<2: continue
        optimizer.zero_grad()
        bl, n = 0.0, 0
        for f in batch:
            sm=hm.get(f.src); dm=hm.get(f.dst)
            ef=torch.from_numpy(np.asarray(f.features,dtype=np.float32)).to(DEVICE)
            lt=max(hm.last(f.src), hm.last(f.dst))
            dt=torch.tensor(max(f.timestamp-lt,0.001),device=DEVICE)
            msg, recon = model(sm.unsqueeze(0),dm.unsqueeze(0),ef.unsqueeze(0),dt.unsqueeze(0))
            loss=F.mse_loss(recon.squeeze(0),ef)
            bl+=loss; n+=1
            ns=model.gru(msg,sm.unsqueeze(0)).squeeze(0)
            nd=model.gru(msg,dm.unsqueeze(0)).squeeze(0)
            hm.update(f.src,ns,f.timestamp); hm.update(f.dst,nd,f.timestamp)
        (bl/n).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
        optimizer.step()
        total_loss+=bl.item(); n_f+=n
    return total_loss/max(n_f,1)

@torch.no_grad()
def evaluate(model, hm, flows):
    model.eval()
    errors, labels = [], []
    for batch in make_batches(flows):
        for f in batch:
            sm=hm.get(f.src); dm=hm.get(f.dst)
            ef=torch.from_numpy(np.asarray(f.features,dtype=np.float32)).to(DEVICE)
            lt=max(hm.last(f.src), hm.last(f.dst))
            dt=torch.tensor(max(f.timestamp-lt,0.001),device=DEVICE)
            msg, recon = model(sm.unsqueeze(0),dm.unsqueeze(0),ef.unsqueeze(0),dt.unsqueeze(0))
            errors.append(F.mse_loss(recon.squeeze(0),ef).item())
            labels.append(f.label)
            ns=model.gru(msg,sm.unsqueeze(0)).squeeze(0)
            nd=model.gru(msg,dm.unsqueeze(0)).squeeze(0)
            hm.update(f.src,ns,f.timestamp); hm.update(f.dst,nd,f.timestamp)
    errs=np.array(errors); labs=np.array(labels)
    # Window aggregation
    W=WINDOW; nw=len(errs)//W
    if nw>0:
        we=errs[:nw*W].reshape(-1,W).max(axis=1)
        wl=labs[:nw*W].reshape(-1,W).max(axis=1)
    else: we,wl=errs,labs
    ha=wl.sum()>0
    roc=roc_auc_score(wl,we) if ha else float("nan")
    pr=average_precision_score(wl,we) if ha else 0.0
    bf=0.0
    for t in np.percentile(we,np.linspace(1,99,100)):
        p=(we>=t).astype(int)
        _,_,f1,_=precision_recall_fscore_support(wl,p,average="binary",zero_division=0)
        if f1>bf: bf=f1
    return {"roc":roc,"pr":pr,"f1":bf}, we, wl

# ═══════════════════════════════════════════════════════════════════════
# TRAIN ONE DATASET
# ═══════════════════════════════════════════════════════════════════════

def train_dataset(name, prefix):
    print(f"\n{'='*60}")
    print(f"TRAINING: {name}")
    print(f"{'='*60}")

    # Load — one line per split
    train_data = load_split(prefix, "train")
    val_data = load_split(prefix, "val")
    test_data = load_split(prefix, "test")

    print(f"Train: {len(train_data['labels']):,} (benign only)")
    print(f"Val:   {len(val_data['labels']):,} (attack: {val_data['labels'].sum():,})")
    print(f"Test:  {len(test_data['labels']):,} (attack: {test_data['labels'].sum():,})")
    print(f"Feats: {train_data['n_features']}")

    # ── Convert to flows ──
    print("\n[1/5] Converting data to flow objects...")
    t0 = time.time()
    train_f = to_flows(train_data, label="train")
    val_f = to_flows(val_data, label="val")
    test_f = to_flows(test_data, label="test")
    n_hosts = len(set(f.src for f in train_f) | set(f.dst for f in train_f))
    n_batches = len(make_batches(train_f))
    print(f"  -> {len(train_f):,} flows, {n_hosts:,} unique hosts, "
          f"~{n_batches} micro-batches, {time.time()-t0:.0f}s")

    # ── Build model ──
    print("\n[2/5] Building TGN model...")
    model = TGNModel(train_data["n_features"]).to(DEVICE)
    global optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  -> {n_params:,} parameters on {DEVICE}")

    # ── Init host memory ──
    print("\n[3/5] Initializing host memory from edge statistics...")
    hm = HostMemory(model)
    n_seeded = min(100000, len(train_f))
    for i, f in enumerate(train_f[:n_seeded]):
        hm.add_edge(f.src, f.dst, float(np.abs(f.features[:4]).sum()))
        if (i+1) % 50000 == 0:
            print(f"  -> {i+1:,}/{n_seeded:,} edges registered...")
    n_initialized = len(hm.mem)
    print(f"  -> {n_initialized:,} hosts initialized with identity-free stats")

    best_val, best_state, pctr = 0.0, None, 0
    t_start = time.time()

    print(f"\n[4/5] Training ({EPOCHS} epochs max, patience={PATIENCE})...")
    print(f"  Micro-batch={MICRO_BATCH_SEC}s, LR={LR}, grad_clip={GRAD_CLIP}")
    print(f"  {'Epoch':<6} {'Loss':>10} {'Val F1':>8} {'Val PR':>8} {'Val ROC':>8} {'LR':>10} {'Time'}")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")

    for ep in range(EPOCHS):
        ep_t0 = time.time()
        tl = train_epoch(model, hm, train_f)
        scheduler.step()

        vm = HostMemory(model)
        for f in train_f[-30000:]:
            vm.add_edge(f.src, f.dst)
        vm_met, _, _ = evaluate(model, vm, val_f[:min(50000,len(val_f))])
        ep_time = time.time() - ep_t0

        if vm_met["f1"] > best_val + 0.001:
            best_val = vm_met["f1"]
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            pctr = 0
            improved = "*"
        else:
            pctr += 1
            improved = " "

        roc_s = f"{vm_met['roc']:.4f}" if not np.isnan(vm_met['roc']) else "  N/A"
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  {ep:3d}{improved}   {tl:>10.4f} {vm_met['f1']:>8.4f} "
              f"{vm_met['pr']:>8.4f} {roc_s:>8} {lr_now:>10.2e} {ep_time:>5.0f}s")

        if pctr >= PATIENCE:
            print(f"\n  Early stopping: no improvement for {PATIENCE} epochs"); break

    train_time = time.time()-t_start
    n_epochs = ep + 1
    if best_state: model.load_state_dict(best_state)
    print(f"\n  Best val F1: {best_val:.4f}  |  {n_epochs} epochs  |  "
          f"{train_time:.0f}s ({train_time/60:.1f}m)")

    # ── Test evaluation ──
    print(f"\n[5/5] Final test evaluation...")
    tm = HostMemory(model)
    n_warmup = min(50000, len(train_f))
    for i, f in enumerate(train_f[-n_warmup:]+val_f[:30000]):
        tm.add_edge(f.src, f.dst)
    n_test_eval = min(50000, len(test_f))
    tmet, test_errs, test_labs = evaluate(model, tm, test_f[:n_test_eval])

    roc_s = f"{tmet['roc']:.4f}" if not np.isnan(tmet['roc']) else "N/A"
    print(f"\n  {'='*50}")
    print(f"  {name} — FINAL TEST RESULTS")
    print(f"  {'='*50}")
    print(f"  ROC-AUC:     {roc_s}")
    print(f"  PR-AUC:      {tmet['pr']:.4f}")
    print(f"  Best F1:     {tmet['f1']:.4f}")
    print(f"  Test flows:  {n_test_eval:,}")
    print(f"  Test attack: {int(test_labs.sum()):,}")
    print(f"  Train time:  {train_time:.0f}s ({train_time/60:.1f}m)")
    print(f"  {'='*50}")

    # Save model
    model_path = OUTPUT_DIR / f"tgn_{prefix}_model.pt"
    torch.save({"state_dict": best_state, "n_features": train_data["n_features"]},
               model_path)
    print(f"\n  Model saved -> {model_path}")

    return {"name": name, "test": tmet, "train_time": train_time,
            "n_features": train_data["n_features"], "n_hosts": n_hosts,
            "n_epochs": n_epochs}

# ═══════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════

cic17_r = train_dataset("CIC-IDS2017", "cicids2017")
unsw_r = train_dataset("UNSW-NB15", "unswnb15")

results = {
    "phase":"7+8_final","seed":SEED,"micro_batch_sec":MICRO_BATCH_SEC,
    "architecture":{"memory_dim":MEMORY_DIM,"hidden_dim":HIDDEN_DIM,
                    "time_dim":TIME_DIM,"dropout":DROPOUT},
    "cic17":{k:str(v) if isinstance(v,(np.floating,float)) and np.isnan(v) else v
             for k,v in cic17_r.items()},
    "unsw":{k:str(v) if isinstance(v,(np.floating,float)) and np.isnan(v) else v
            for k,v in unsw_r.items()},
}
with open(OUTPUT_DIR/"training_results.json","w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"Models: {OUTPUT_DIR}/tgn_*_model.pt")
print(f"Results: {OUTPUT_DIR}/training_results.json")
print(f"{'='*60}")
