#!/usr/bin/env python3
"""
GNN-NIDS Phase 6 — Kaggle Notebook
====================================
Copy this entire file into a SINGLE Kaggle notebook cell and run it.
It auto-discovers your uploaded datasets regardless of their path names.

PREREQUISITES (already done per your file listing):
  - nids-package dataset added         (contains nids/ folder)
  - nids-data-cic17-unsw15 dataset     (contains all parquet + joblib files)
"""

import sys, os, json, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# =============================================================================
# STEP 0: Environment
# =============================================================================
print("=" * 60)
print("STEP 0: ENVIRONMENT")
print("=" * 60)

IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
print(f"Kaggle: {IS_KAGGLE}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")

# Install PyG if missing
try:
    import torch_geometric
except ImportError:
    print("Installing torch_geometric...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch_geometric", "-q"])
    import torch_geometric
print(f"torch_geometric: {torch_geometric.__version__}")

# =============================================================================
# STEP 1: Auto-discover nids package and data files
# =============================================================================
print("\n" + "=" * 60)
print("STEP 1: DISCOVERING PACKAGE AND DATA")
print("=" * 60)

INPUT_ROOT = Path("/kaggle/input") if IS_KAGGLE else Path.cwd()

# --- Find nids package ---
NIDS_ROOT = None

if IS_KAGGLE:
    # Walk all input subdirectories to find nids/__init__.py
    for root, dirs, files in os.walk(str(INPUT_ROOT)):
        root_path = Path(root)
        if root_path.name == "nids" and (root_path / "__init__.py").exists():
            # Found the nids/ directory. Its PARENT goes on sys.path
            NIDS_ROOT = root_path.parent
            print(f"Found nids/ at: {root_path}")
            break

    # Also search for nids directly inside a dataset folder
    if NIDS_ROOT is None:
        for d in sorted(INPUT_ROOT.rglob("*")):
            if d.is_dir():
                for sub in d.iterdir():
                    if sub.is_dir() and sub.name == "nids" and (sub / "__init__.py").exists():
                        NIDS_ROOT = d
                        print(f"Found nids/ at: {sub}")
                        break
                if NIDS_ROOT:
                    break

if NIDS_ROOT is None:
    # Local fallback
    NIDS_ROOT = Path.cwd()
    print("Local mode: using current directory")

print(f"NIDS_ROOT: {NIDS_ROOT}")

if str(NIDS_ROOT) not in sys.path:
    sys.path.insert(0, str(NIDS_ROOT))

# Verify import works
try:
    from nids import set_seed, SEED
    set_seed()
    print(f"nids imported: SEED={SEED}")
except ImportError as e:
    print(f"ERROR importing nids: {e}")
    print(f"sys.path: {sys.path[:5]}")
    print("Contents of NIDS_ROOT:", list(NIDS_ROOT.iterdir())[:10] if NIDS_ROOT.exists() else "NOT FOUND")
    sys.exit(1)

# --- Find data files ---
DATA_ROOT = None

if IS_KAGGLE:
    # Search all input for parquet files
    for root, dirs, files in os.walk(str(INPUT_ROOT)):
        parqs = [f for f in files if f.endswith(".parquet")]
        if len(parqs) >= 4:  # Need at least 4 parquet files
            DATA_ROOT = Path(root)
            print(f"Found data at: {DATA_ROOT} ({len(parqs)} parquet files)")
            break

if DATA_ROOT is None:
    DATA_ROOT = Path.cwd() / "datasets"
    print("Local mode data:", DATA_ROOT)

# Verify key files
required = ["cic17_common.parquet", "unsw_common.parquet"]
for fname in required:
    exists = (DATA_ROOT / fname).exists()
    print(f"  [{('OK' if exists else 'MISSING')}] {fname}")

# Import TGN components
from nids.models.tgn_memory import (
    TGNMemoryModule, HostMemoryStore,
    MicroBatchProcessor, Flow,
)
print("TGN memory module: OK")

import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support,
)
print("All libraries: OK")

# =============================================================================
# STEP 2: Phase 6 — Micro-Batch Ablation
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: PHASE 6 — MICRO-BATCH ABLATION")
print("=" * 60)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FLOWS = 200_000
OUTPUT_DIR = Path("/kaggle/working/output") if IS_KAGGLE else Path("local/baselines")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print(f"Device: {DEVICE}")
print(f"Max flows: {MAX_FLOWS:,}")

# --- Load data ---
print("\n--- Loading data ---")

# Try scaled splits first, then common track (all flat in DATA_ROOT)
train_path = DATA_ROOT / "cicids2017_chrono_train_scaled.parquet"
val_path = DATA_ROOT / "cicids2017_chrono_val_scaled.parquet"
common_path = DATA_ROOT / "cic17_common.parquet"

if train_path.exists() and val_path.exists():
    print("Using scaled chronological splits...")
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)
    df = pd.concat([train, val], ignore_index=True)
else:
    print("Falling back to common track...")
    df = pd.read_parquet(common_path)

if len(df) > MAX_FLOWS:
    df = df.iloc[:MAX_FLOWS]
print(f"Loaded: {len(df):,} rows x {len(df.columns)} cols")

# --- Split warmup/eval ---
label_col = "label" if "label" in df.columns else "Label"
benign = df[df[label_col] == 0]
attack = df[df[label_col] == 1]

n_benign_train = int(len(benign) * 0.70)
n_val = min(len(df) - n_benign_train, int(len(df) * 0.30))

train_df = benign.iloc[:n_benign_train]
val_df = pd.concat([benign.iloc[n_benign_train:], attack]) \
           .sample(frac=1, random_state=SEED).iloc[:n_val]

print(f"Warmup (benign only): {len(train_df):,}")
print(f"Eval:                  {len(val_df):,}  "
      f"(attack: {(val_df[label_col]==1).sum():,})")

# --- Convert to Flow objects ---
print("\n--- Converting to flows ---")
exclude = {"label","label_str","attack_cat","srcip","dstip",
           "sport","dsport","timestamp","stime","ltime",
           "Source IP","Destination IP","Source Port","Destination Port",
           "Flow ID","Timestamp"}
feature_cols = [c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def df_to_flows(dataframe):
    flows = []
    for i, (_, row) in enumerate(dataframe.iterrows()):
        src = f"host_{hash(str(i)) % 5000}"
        dst = f"host_{hash(str(i + 1000)) % 5000}"
        feats = np.array([float(row.get(c, 0) or 0) for c in feature_cols],
                         dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        flows.append(Flow(
            src=src, dst=dst,
            timestamp=i * 0.001,
            features=feats,
            label=int(row.get(label_col, 0) or 0),
            src_port=0, dst_port=0, protocol=6,
        ))
    return flows

warmup_flows = df_to_flows(train_df)
val_flows = df_to_flows(val_df)
val_labels = val_df[label_col].values.astype(int)
edge_feat_dim = len(feature_cols)
print(f"Edge feature dim: {edge_feat_dim}")
print(f"Warmup flows: {len(warmup_flows):,}  Eval flows: {len(val_flows):,}")

# --- Ablation sweep ---
print("\n--- Sweeping 4 micro-batch configs ---")

configs = [
    {"name": "0.5s",     "type": "time",  "value": 0.5},
    {"name": "1.0s",     "type": "time",  "value": 1.0},
    {"name": "2.0s",     "type": "time",  "value": 2.0},
    {"name": "count-50", "type": "count", "value": 50},
]

results = []
for cfg in configs:
    print(f"  [{cfg['name']}] ", end="", flush=True)

    # Fresh memory init per config
    mem_module = TGNMemoryModule(
        memory_dim=128, edge_feat_dim=edge_feat_dim,
        time_dim=16, msg_hidden_dim=256, dropout=0.1,
    ).to(DEVICE)
    store = HostMemoryStore(memory_dim=128)

    # Warmup on benign flows
    t0 = time.time()
    wp = MicroBatchProcessor(mem_module, store, None,
        micro_batch_sec=cfg["value"] if cfg["type"]=="time" else 999.0)
    if cfg["type"] == "count":
        batches = [warmup_flows[i:i+cfg["value"]]
                   for i in range(0, len(warmup_flows), cfg["value"])]
        for batch in batches:
            wp._process_micro_batch(batch, DEVICE)
    else:
        wp.process_flows(warmup_flows, DEVICE)
    warmup_t = time.time() - t0

    # Evaluate on val
    t0 = time.time()
    errors = []
    eval_wp = MicroBatchProcessor(mem_module, store, None,
        micro_batch_sec=cfg["value"])
    batches = eval_wp._make_micro_batches(val_flows)

    mem_module.eval()
    with torch.no_grad():
        for batch in batches:
            for flow in batch:
                src_m = store.get_memory(flow.src).to(DEVICE)
                dst_m = store.get_memory(flow.dst).to(DEVICE)
                ef = torch.from_numpy(
                    np.asarray(flow.features, dtype=np.float32)).to(DEVICE)
                lt = max(store.get_last_update(flow.src),
                         store.get_last_update(flow.dst))
                dt = max(flow.timestamp - lt, 0.0)

                msg_s, _ = mem_module(
                    src_m.unsqueeze(0), dst_m.unsqueeze(0),
                    ef.unsqueeze(0),
                    torch.tensor([dt/300.0], device=DEVICE).unsqueeze(0))

                errors.append(msg_s.squeeze(0).norm(p=2).item())

                ns = mem_module.update_memory(src_m.unsqueeze(0), msg_s).squeeze(0)
                nd = mem_module.update_memory(dst_m.unsqueeze(0), msg_s).squeeze(0)
                store.update_memory(flow.src, ns.cpu(), flow.timestamp)
                store.update_memory(flow.dst, nd.cpu(), flow.timestamp)

    eval_t = time.time() - t0
    errs = np.array(errors)

    # Metrics
    has_attack = val_labels.sum() > 0
    roc = roc_auc_score(val_labels, errs) if has_attack else float("nan")
    pr = average_precision_score(val_labels, errs) if has_attack else 0.0
    best_f1, best_t = 0.0, 0.0
    for t in np.percentile(errs, np.linspace(1, 99, 100)):
        preds = (errs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            val_labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    m = {
        "micro_batch": cfg["name"], "batch_type": cfg["type"],
        "batch_value": cfg["value"],
        "roc_auc": round(float(roc), 4) if not np.isnan(roc) else None,
        "pr_auc": round(float(pr), 4),
        "best_f1": round(float(best_f1), 4),
        "warmup_sec": round(warmup_t, 1),
        "eval_sec": round(eval_t, 1),
    }
    results.append(m)
    print(f"F1={best_f1:.4f}  PR={pr:.4f}  ROC={roc:.4f}  "
          f"warmup={warmup_t:.1f}s  eval={eval_t:.1f}s")

# --- Decision ---
print("\n--- Decision ---")
best = max(results, key=lambda r: r["best_f1"])
candidates = [r for r in results if best["best_f1"] - r["best_f1"] < 0.01]
chosen = max(candidates, key=lambda r: r["batch_value"]) if len(candidates) > 1 else best

print(f"Best F1: {best['best_f1']:.4f} ({best['micro_batch']})")
print(f"Chosen:  {chosen['micro_batch']}  (tiebreak -> larger batch)")
print(f">>> LOCKED for all subsequent phases <<<")

# --- Save ---
output = {
    "phase": 6,
    "seed": SEED,
    "locked_micro_batch": chosen["micro_batch"],
    "locked_value": chosen["batch_value"],
    "locked_type": chosen["batch_type"],
    "decision_rule": "Best F1; within 1 point -> larger batch",
    "device": str(DEVICE),
    "results": results,
}
out_path = OUTPUT_DIR / "microbatch_ablation.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

# --- Summary ---
print(f"\n{'='*60}")
print(f"PHASE 6 COMPLETE")
print(f"{'='*60}")
print(f"Output: {out_path}")
print(f"Locked: {chosen['micro_batch']}")
print(f"\n{'Config':<12} {'F1':>8} {'PR-AUC':>8} {'ROC-AUC':>8} {'Warmup':>8} {'Eval':>8}")
print("-" * 56)
for r in results:
    roc_str = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "N/A"
    print(f"{r['micro_batch']:<12} {r['best_f1']:>8.4f} {r['pr_auc']:>8.4f} "
          f"{roc_str:>8} {r['warmup_sec']:>7.1f}s {r['eval_sec']:>7.1f}s")
