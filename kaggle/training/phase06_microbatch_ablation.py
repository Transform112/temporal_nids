#!/usr/bin/env python3
"""
Phase 6 — Micro-Batch Size Ablation
Target: KAGGLE (T4x2)
Input:  datasets/processed/cic17_common.parquet or datasets/splits/cicids2017_chrono_train_scaled.parquet
Output: microbatch_ablation.json, locked micro-batch size

Sweep: 0.5s, 1s, 2s time-based + count-based (50 flows/batch)
Metric: F1 on validation slice
Decision rule: best F1; if within 1 point, choose larger batch size
"""

import sys, os
from pathlib import Path
import json
import time
import argparse
from typing import List

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support,
)

# ── Kaggle path setup ──────────────────────────────────────────────────────
IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))

if IS_KAGGLE:
    KAGGLE_WORKING = Path("/kaggle/working")
    KAGGLE_INPUT = Path("/kaggle/input")
else:
    KAGGLE_WORKING = Path.cwd()
    KAGGLE_INPUT = Path.cwd()

# Try multiple locations for the package
for p in [KAGGLE_WORKING, KAGGLE_INPUT / "nids-package",
          Path.cwd(), Path(__file__).resolve().parent.parent]:
    if (p / "nids" / "__init__.py").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

from nids import set_seed, SEED
from nids.models.tgn_memory import (
    TGNMemoryModule, HostMemoryStore,
    MicroBatchProcessor, Flow,
)

set_seed()

# ── Paths ───────────────────────────────────────────────────────────────────
if IS_KAGGLE:
    DATA_DIR = KAGGLE_INPUT / "nids-data"
else:
    DATA_DIR = Path.cwd() / "datasets"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Output dir
if IS_KAGGLE:
    OUTPUT_DIR = KAGGLE_WORKING / "output"
else:
    OUTPUT_DIR = Path.cwd() / "local" / "baselines"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ──────────────────────────────────────────────────────────────────
MICRO_BATCH_SIZES = [0.5, 1.0, 2.0]  # seconds
COUNT_BASED = 50                      # flows per batch
MAX_FLOWS = 200_000                   # Limit flows for fast ablation


def load_data(max_flows: int = MAX_FLOWS) -> pd.DataFrame:
    """Load CIC17 common track (or scaled train split if available)."""
    # Prefer scaled train split
    train_path = SPLITS_DIR / "cicids2017_chrono_train_scaled.parquet"
    val_path = SPLITS_DIR / "cicids2017_chrono_val_scaled.parquet"

    if train_path.exists() and val_path.exists():
        print(f"Loading scaled splits...")
        train = pd.read_parquet(train_path)
        val = pd.read_parquet(val_path)
        df = pd.concat([train, val], ignore_index=True)
    else:
        # Fall back to common track
        common_path = PROCESSED_DIR / "cic17_common.parquet"
        if not common_path.exists():
            raise FileNotFoundError(
                f"No data found. Tried:\n  {train_path}\n  {common_path}\n"
                "Upload processed parquet files as a Kaggle dataset."
            )
        print(f"Loading common track (unscaled)...")
        df = pd.read_parquet(common_path)

    if len(df) > max_flows:
        df = df.iloc[:max_flows]  # Take first N (chronological order)
        print(f"  Trimmed to {max_flows:,} flows")

    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} cols")
    return df


def df_to_flows(df: pd.DataFrame) -> List[Flow]:
    """Convert DataFrame rows to Flow objects for TGN processing."""
    # Find feature columns (exclude labels/metadata)
    exclude = {"label", "label_str", "attack_cat", "srcip", "dstip",
               "sport", "dsport", "timestamp", "stime", "ltime",
               "Source IP", "Destination IP", "Source Port", "Destination Port",
               "Flow ID", "Timestamp"}
    feature_cols = [c for c in df.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    label_col = "label" if "label" in df.columns else "Label"

    flows = []
    for i, (_, row) in enumerate(df.iterrows()):
        # Synthetic host IDs (no real IPs in common track)
        src = f"host_{hash(str(i)) % 5000}"
        dst = f"host_{hash(str(i + 1000)) % 5000}"

        feats = np.array([float(row.get(c, 0) or 0) for c in feature_cols], dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        flows.append(Flow(
            src=src, dst=dst,
            timestamp=i * 0.001,  # ~1000 flows/sec
            features=feats,
            label=int(row.get(label_col, 0) or 0),
            src_port=0, dst_port=0, protocol=6,
        ))

    return flows, feature_cols


def setup_tgn(edge_feat_dim: int) -> tuple:
    """Initialize TGN memory module, store, and forensic log."""
    memory_dim = 128
    mem_module = TGNMemoryModule(
        memory_dim=memory_dim,
        edge_feat_dim=edge_feat_dim,
        time_dim=16,
        msg_hidden_dim=256,
        dropout=0.1,
    ).to(DEVICE)

    store = HostMemoryStore(memory_dim=memory_dim)
    processor = MicroBatchProcessor(
        memory_module=mem_module,
        memory_store=store,
        forensic_log=None,
        micro_batch_sec=1.0,  # Default, will be overridden per sweep
    )

    return mem_module, store, processor


def compute_reconstruction_errors(
    flows: List[Flow],
    mem_module: TGNMemoryModule,
    store: HostMemoryStore,
    micro_batch_sec: float,
) -> np.ndarray:
    """Process flows and return per-flow reconstruction errors."""
    processor = MicroBatchProcessor(
        memory_module=mem_module,
        memory_store=store,
        forensic_log=None,
        micro_batch_sec=micro_batch_sec,
    )

    errors = []
    batches = processor._make_micro_batches(flows)

    mem_module.eval()
    with torch.no_grad():
        for batch in batches:
            batch_errors = []
            for flow in batch:
                src_mem = store.get_memory(flow.src).to(DEVICE)
                dst_mem = store.get_memory(flow.dst).to(DEVICE)
                edge_feats = torch.from_numpy(
                    np.asarray(flow.features, dtype=np.float32)
                ).to(DEVICE)

                last_t = max(
                    store.get_last_update(flow.src),
                    store.get_last_update(flow.dst),
                )
                delta_t = max(flow.timestamp - last_t, 0.0)
                delta_t_tensor = torch.tensor([delta_t / 300.0], device=DEVICE)

                msg_src, msg_dst = mem_module(
                    src_mem.unsqueeze(0),
                    dst_mem.unsqueeze(0),
                    edge_feats.unsqueeze(0),
                    delta_t_tensor.unsqueeze(0),
                )

                # Anomaly score: L2 norm of the message vector
                # (message norm quantifies how "surprising" the flow is —
                #  attacks produce larger deviations from expected patterns)
                error = msg_src.squeeze(0).norm(p=2).item()
                batch_errors.append(error)

                # Update memories
                new_src = mem_module.update_memory(
                    src_mem.unsqueeze(0), msg_src
                ).squeeze(0)
                new_dst = mem_module.update_memory(
                    dst_mem.unsqueeze(0), msg_dst
                ).squeeze(0)
                store.update_memory(flow.src, new_src.cpu(), flow.timestamp)
                store.update_memory(flow.dst, new_dst.cpu(), flow.timestamp)

            errors.extend(batch_errors)

    return np.array(errors)


def evaluate_errors(errors: np.ndarray, labels: np.ndarray) -> dict:
    """Compute detection metrics from reconstruction errors."""
    roc_auc = roc_auc_score(labels, errors)
    pr_auc = average_precision_score(labels, errors)

    # Sweep threshold for best F1
    best_f1, best_thresh, best_prec, best_rec = 0, 0, 0, 0
    thresholds = np.percentile(errors, np.linspace(1, 99, 200))
    for t in thresholds:
        preds = (errors >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_thresh, best_prec, best_rec = f1, t, prec, rec

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "best_f1": float(best_f1),
        "best_threshold": float(best_thresh),
        "precision": float(best_prec),
        "recall": float(best_rec),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Micro-batch ablation")
    parser.add_argument("--target", default="kaggle", choices=["laptop", "kaggle"])
    parser.add_argument("--max-flows", type=int, default=MAX_FLOWS)
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 6: MICRO-BATCH SIZE ABLATION")
    print(f"Device: {DEVICE}")
    print(f"Max flows: {args.max_flows:,}")
    print(f"Seed: {SEED}")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    df = load_data(max_flows=args.max_flows)

    # Split: ensure val has attack flows for meaningful metrics
    label_col = "label" if "label" in df.columns else "Label"
    benign = df[df[label_col] == 0]
    attack = df[df[label_col] == 1]

    n_benign_train = int(len(benign) * 0.70)
    n_val = min(len(df) - n_benign_train, int(len(df) * 0.30))

    train_df = benign.iloc[:n_benign_train]  # benign only
    val_df = pd.concat([
        benign.iloc[n_benign_train:],
        attack
    ]).sample(frac=1, random_state=SEED).iloc[:n_val]

    print(f"  Warmup (benign): {len(train_df):,}")
    print(f"  Eval:             {len(val_df):,} "
          f"(attack: {(val_df[label_col] == 1).sum():,})")

    # Convert to flows
    print("\n[2/4] Converting to flows...")
    warmup_flows, feature_cols = df_to_flows(pd.concat([train_df, val_df.iloc[:0]]))
    all_flows, feature_cols = df_to_flows(df)
    val_flows, _ = df_to_flows(val_df)
    edge_feat_dim = len(feature_cols)
    print(f"  Edge feature dim: {edge_feat_dim}")

    # Get val labels
    val_labels = val_df[label_col].values.astype(int)
    n_attack = int(val_labels.sum())
    if n_attack == 0:
        print("  WARNING: No attack flows in validation set — metrics will be degenerate")
        print("  Increase --max-flows or use a dataset slice with attack traffic")

    # Sweep micro-batch sizes
    print("\n[3/4] Sweeping micro-batch sizes...")
    results = []

    configs = [
        {"name": "0.5s", "type": "time", "value": 0.5},
        {"name": "1.0s", "type": "time", "value": 1.0},
        {"name": "2.0s", "type": "time", "value": 2.0},
        {"name": "count-50", "type": "count", "value": 50},
    ]

    for cfg in configs:
        print(f"\n  --- {cfg['name']} ---")

        # Fresh init for each config
        mem_module, store, _ = setup_tgn(edge_feat_dim)

        # Warmup on benign
        t0 = time.time()
        warmup_processor = MicroBatchProcessor(
            mem_module, store, None,
            micro_batch_sec=cfg["value"] if cfg["type"] == "time" else 999.0,
        )
        if cfg["type"] == "count":
            # Count-based: use large time window, but batch in groups of N
            warmup_processor.micro_batch_sec = 999.0
            # Override batch creation for count-based
            batches = [warmup_flows[i:i+cfg["value"]]
                       for i in range(0, len(warmup_flows), cfg["value"])]
            for batch in batches:
                warmup_processor._process_micro_batch(batch, DEVICE)
        else:
            warmup_processor.process_flows(warmup_flows, DEVICE)

        warmup_time = time.time() - t0

        # Evaluate
        t0 = time.time()
        errors = compute_reconstruction_errors(val_flows, mem_module, store, cfg["value"])
        eval_time = time.time() - t0

        metrics = evaluate_errors(errors, val_labels)
        metrics["micro_batch"] = cfg["name"]
        metrics["batch_type"] = cfg["type"]
        metrics["batch_value"] = cfg["value"]
        metrics["warmup_time_sec"] = round(warmup_time, 2)
        metrics["eval_time_sec"] = round(eval_time, 2)
        metrics["n_warmup_flows"] = len(warmup_flows)
        metrics["n_eval_flows"] = len(val_flows)

        results.append(metrics)

        print(f"    Warmup: {warmup_time:.1f}s, Eval: {eval_time:.1f}s")
        print(f"    F1={metrics['best_f1']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, "
              f"ROC-AUC={metrics['roc_auc']:.4f}")

    # Decision rule
    print("\n[4/4] Selecting micro-batch size...")
    best = max(results, key=lambda r: r["best_f1"])
    # Within 1 F1 point → prefer larger batch
    candidates = [r for r in results
                  if best["best_f1"] - r["best_f1"] < 0.01]
    if len(candidates) > 1:
        # Choose largest batch among tied candidates
        chosen = max(candidates, key=lambda r: r["batch_value"])
    else:
        chosen = best

    print(f"\n  Best F1: {best['best_f1']:.4f} ({best['micro_batch']})")
    print(f"  Chosen:   {chosen['micro_batch']} (tiebreak -> larger batch)")
    print(f"  This value is now LOCKED for all subsequent phases.")

    # Save
    output = {
        "phase": 6,
        "seed": SEED,
        "locked_micro_batch": chosen["micro_batch"],
        "locked_value": chosen.get("batch_value", None),
        "locked_type": chosen.get("batch_type", None),
        "decision_rule": "Best F1; within 1 point → larger batch",
        "results": results,
    }

    output_path = OUTPUT_DIR / "microbatch_ablation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  -> Saved: {output_path}")

    print("\n" + "=" * 70)
    print("PHASE 6 COMPLETE")
    print(f"Locked micro-batch: {chosen['micro_batch']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
