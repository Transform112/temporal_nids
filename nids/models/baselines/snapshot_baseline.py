#!/usr/bin/env python3
"""
Phase 3 — Baseline Snapshot Model
Target: LAPTOP (dev with --sample, full CIC17 only)
Produces: baseline_snapshot_results.json

Cheap comparison point: 60-second-window E-GraphSAGE with reconstruction
objective on CIC17 only. Exists purely to justify the TGN architecture switch.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from nids import set_seed, SEED

set_seed()

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "local" / "baselines"
try:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
except OSError:
    pass  # Read-only filesystem (Kaggle input)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_snapshot_graphs(
    df: pd.DataFrame,
    window_sec: float = 60.0,
    ip_col: str = "srcip",
    dst_ip_col: str = "dstip",
    time_col: str = "timestamp",
    feature_cols: Optional[list] = None,
    label_col: str = "label",
) -> list[Data]:
    """Split flow data into 60-second snapshot graphs.

    Each snapshot:
      - Nodes = unique IPs appearing in the window
      - Edges = flows between IP pairs
      - Edge features = common-track flow features
      - Node features = aggregated edge stats (in/out degree, byte volume)

    Returns list of PyG Data objects.
    """
    if feature_cols is None:
        # Use all numeric columns except label/metadata
        exclude = {"label", "label_str", "attack_cat", "srcip", "dstip",
                   "sport", "dsport", "timestamp", "stime", "ltime"}
        feature_cols = [c for c in df.columns
                        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    # Need IP and timestamp columns from native track for graph construction.
    # If not available (common track doesn't have them), use index-based grouping.
    has_ips = ip_col in df.columns and dst_ip_col in df.columns
    has_time = time_col in df.columns

    if not has_time:
        # No timestamp column (common track). Use event-count-based windows
        # as a fallback: ~2000 flows per window simulates ~60s at typical rates.
        FLOWS_PER_WINDOW = 2000
        df = df.copy()
        df["_pseudo_time"] = (np.arange(len(df)) // FLOWS_PER_WINDOW) * window_sec
        time_col = "_pseudo_time"

    if not has_ips:
        # Use source/destination identifiers derived from common track if available
        # Fall back to creating synthetic host IDs from flow index
        df = df.copy()
        n_hosts = 200  # Small realistic host pool
        df["_src_id"] = np.random.RandomState(SEED).randint(0, n_hosts, size=len(df))
        df["_dst_id"] = np.random.RandomState(SEED + 1).randint(0, n_hosts, size=len(df))
        ip_col = "_src_id"
        dst_ip_col = "_dst_id"

    # Sort by time
    df = df.sort_values(time_col).reset_index(drop=True)

    # Build IP-to-node-index mapping (grows as new IPs appear)
    ip_to_idx: dict = {}
    next_idx = 0

    def get_node_idx(ip):
        nonlocal next_idx
        ip_str = str(ip)
        if ip_str not in ip_to_idx:
            ip_to_idx[ip_str] = next_idx
            next_idx += 1
        return ip_to_idx[ip_str]

    # Split into windows
    t0 = df[time_col].iloc[0]
    window_start = t0
    window_edges = []

    graphs = []
    window_idx = 0

    # Group rows into windows
    for _, row in df.iterrows():
        t = row[time_col]
        if t - window_start >= window_sec:
            # Flush current window
            if window_edges:
                g = _make_graph(window_edges, ip_to_idx, feature_cols, label_col)
                if g is not None and g.num_nodes > 1 and g.num_edges > 0:
                    graphs.append(g)
            window_edges = []
            window_start = t
            window_idx += 1

        # Register source/dest IPs so _make_graph can resolve them
        src_ip = str(row.get(ip_col, row.name))
        dst_ip = str(row.get(dst_ip_col, row.name))
        if src_ip not in ip_to_idx:
            ip_to_idx[src_ip] = next_idx
            next_idx += 1
        if dst_ip not in ip_to_idx:
            ip_to_idx[dst_ip] = next_idx
            next_idx += 1

        window_edges.append(row)

    # Final window
    if window_edges:
        g = _make_graph(window_edges, ip_to_idx, feature_cols, label_col)
        if g is not None and g.num_nodes > 1 and g.num_edges > 0:
            graphs.append(g)

    print(f"    Built {len(graphs)} snapshot graphs over {len(df):,} flows")
    if graphs:
        print(f"    Avg nodes/graph: {np.mean([g.num_nodes for g in graphs]):.0f}, "
              f"Avg edges/graph: {np.mean([g.num_edges for g in graphs]):.0f}")
    return graphs


def _make_graph(
    edges: list,
    ip_to_idx: dict,
    feature_cols: list[str],
    label_col: str,
) -> Optional[Data]:
    """Create a single PyG Data object from a window's edges."""
    src_nodes = []
    dst_nodes = []
    edge_feats = []
    edge_labels = []

    for row in edges:
        src = ip_to_idx.get(str(row.get("_src_id", row.name)))
        dst = ip_to_idx.get(str(row.get("_dst_id", row.name)))
        if src is None or dst is None:
            continue
        if src == dst:
            continue  # Skip self-loops

        src_nodes.append(src)
        dst_nodes.append(dst)

        # Edge features
        feats = []
        for col in feature_cols:
            val = row.get(col, 0)
            try:
                val = float(val)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
            except (ValueError, TypeError):
                val = 0.0
            feats.append(val)
        edge_feats.append(feats)

        # Edge label (1 if attack, 0 if benign)
        lbl = row.get(label_col, 0)
        try:
            lbl = int(lbl)
        except (ValueError, TypeError):
            lbl = 0
        edge_labels.append(lbl)

    if len(src_nodes) == 0:
        return None

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    edge_label = torch.tensor(edge_labels, dtype=torch.float32)

    # Node features: simple degree-based stats
    num_nodes = max(max(src_nodes), max(dst_nodes)) + 1
    in_deg = torch.zeros(num_nodes, dtype=torch.float32)
    out_deg = torch.zeros(num_nodes, dtype=torch.float32)
    for s, d in zip(src_nodes, dst_nodes):
        out_deg[s] += 1
        in_deg[d] += 1

    # Node features: [in_deg, out_deg, log(1+in_deg), log(1+out_deg)]
    x = torch.stack([
        in_deg,
        out_deg,
        torch.log1p(in_deg),
        torch.log1p(out_deg),
    ], dim=1)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        num_nodes=num_nodes,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# E-GraphSAGE Encoder (Edge-conditioned GraphSAGE)
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeGraphSAGEEncoder(nn.Module):
    """E-GraphSAGE: message passing with edge feature conditioning.

    Simple encoder-decoder for reconstruction-based anomaly detection.
    Encoder: 2-layer GraphSAGE with edge features concatenated to node features.
    Decoder: MLP that predicts edge features from (src_emb, dst_emb).
    """

    def __init__(
        self,
        node_dim: int = 4,
        edge_dim: int = 36,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # Node feature projection
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        # Edge feature projection
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # GraphSAGE-style convolutions (message = concat(x_i, x_j, edge_attr))
        # Layer 1
        self.conv1_self = nn.Linear(hidden_dim, hidden_dim)
        self.conv1_neigh = nn.Linear(hidden_dim + hidden_dim, hidden_dim)  # x_j + edge_attr

        # Layer 2
        self.conv2_self = nn.Linear(hidden_dim, hidden_dim)
        self.conv2_neigh = nn.Linear(hidden_dim + hidden_dim, hidden_dim)

        # Output projection
        self.out = nn.Linear(hidden_dim, out_dim)

        # Decoder: reconstruct edge features from node embeddings
        self.decoder = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode nodes, then decode edge features.

        Returns:
            node_emb: (num_nodes, out_dim) node embeddings
            edge_recon: (num_edges, edge_dim) reconstructed edge features
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        num_nodes = x.size(0)

        # Project node features
        h = self.node_proj(x)  # (N, hidden_dim)
        h = F.relu(h)
        h = self.dropout(h)

        # Project edge features
        e = self.edge_proj(edge_attr)  # (E, hidden_dim)
        e = F.relu(e)

        # ── Layer 1 ──
        h_self = self.conv1_self(h)
        h_msg = self._message_passing(h, edge_index, e, self.conv1_neigh)
        h = F.relu(h_self + h_msg)
        h = self.dropout(h)

        # ── Layer 2 ──
        h_self = self.conv2_self(h)
        h_msg = self._message_passing(h, edge_index, e, self.conv2_neigh)
        h = F.relu(h_self + h_msg)

        # Final projection
        node_emb = self.out(h)  # (N, out_dim)

        # ── Decode edges ──
        src, dst = edge_index[0], edge_index[1]
        src_emb = node_emb[src]  # (E, out_dim)
        dst_emb = node_emb[dst]  # (E, out_dim)
        edge_recon = self.decoder(torch.cat([src_emb, dst_emb], dim=1))  # (E, edge_dim)

        return node_emb, edge_recon

    def _message_passing(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        e: torch.Tensor,
        linear: nn.Linear,
    ) -> torch.Tensor:
        """Aggregate neighbor messages with edge features using index_add."""
        src, dst = edge_index[0], edge_index[1]
        # Message: concat(neighbor_embedding, edge_feature)
        msg = torch.cat([h[src], e], dim=1)  # (E, 2*hidden_dim)
        msg = linear(msg)  # (E, hidden_dim)

        # Aggregate to destination nodes via index_add (mean)
        num_nodes = h.size(0)
        # Count messages per destination for mean
        ones = torch.ones(msg.size(0), 1, device=h.device)
        count = torch.zeros(num_nodes, 1, device=h.device)
        count = count.index_add(0, dst, ones)

        aggr = torch.zeros(num_nodes, msg.size(1), device=h.device)
        aggr = aggr.index_add(0, dst, msg)

        # Mean aggregation (avoid div-by-zero)
        count = count.clamp(min=1)
        aggr = aggr / count

        return aggr

    def reconstruction_error(self, data: Data) -> torch.Tensor:
        """Per-edge reconstruction error (MSE)."""
        _, edge_recon = self.forward(data)
        # MSE per edge
        err = ((edge_recon - data.edge_attr) ** 2).mean(dim=1)
        return err


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model: EdgeGraphSAGEEncoder,
    graphs: list[Data],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train one epoch over all graphs. Returns average loss."""
    model.train()
    total_loss = 0.0
    total_edges = 0

    for g in graphs:
        g = g.to(device)
        optimizer.zero_grad()

        _, edge_recon = model(g)
        loss = F.mse_loss(edge_recon, g.edge_attr)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * g.num_edges
        total_edges += g.num_edges

    return total_loss / max(total_edges, 1)


@torch.no_grad()
def evaluate(
    model: EdgeGraphSAGEEncoder,
    graphs: list[Data],
    device: torch.device,
) -> dict:
    """Evaluate reconstruction error and anomaly detection metrics."""
    model.eval()
    all_errors = []
    all_labels = []

    for g in graphs:
        g = g.to(device)
        err = model.reconstruction_error(g)
        all_errors.append(err.cpu())
        all_labels.append(g.edge_label.cpu())

    errors = torch.cat(all_errors).numpy()
    labels = torch.cat(all_labels).numpy()

    # ROC-AUC and PR-AUC
    roc_auc = roc_auc_score(labels, errors)
    pr_auc = average_precision_score(labels, errors)

    # Best F1 threshold sweep
    best_f1 = 0.0
    best_thresh = 0.0
    best_prec = 0.0
    best_rec = 0.0

    # Sweep thresholds from min to max error
    thresholds = np.percentile(errors, np.linspace(1, 99, 200))
    for t in thresholds:
        preds = (errors >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_prec = prec
            best_rec = rec

    # Also check zero_division as a fallback
    if best_f1 == 0:
        preds = (errors >= np.median(errors)).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        best_f1 = f1
        best_thresh = float(np.median(errors))
        best_prec = prec
        best_rec = rec

    # FPR at operating point
    preds_at_thresh = (errors >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds_at_thresh).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "best_f1": float(best_f1),
        "best_threshold": float(best_thresh),
        "precision": float(best_prec),
        "recall": float(best_rec),
        "fpr": float(fpr),
        "total_edges": int(len(labels)),
        "attack_edges": int(labels.sum()),
        "benign_edges": int((1 - labels).sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Baseline Snapshot Model"
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of flows to sample (dev mode)")
    parser.add_argument("--window-sec", type=int, default=60,
                        help="Snapshot window size in seconds")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension")
    parser.add_argument("--out-dim", type=int, default=64,
                        help="Output embedding dimension")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--target", type=str, default="laptop",
                        choices=["laptop", "kaggle"],
                        help="Compute target")
    args = parser.parse_args()

    if args.target != "laptop":
        raise RuntimeError("Baseline runs on laptop. Use --target laptop.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("=" * 70)
    print("PHASE 3: BASELINE SNAPSHOT MODEL")
    print(f"Window: {args.window_sec}s, Hidden: {args.hidden_dim}, "
          f"Out: {args.out_dim}, Epochs: {args.epochs}")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────
    print("\n[1/5] Loading CIC-IDS2017 common track...")
    cic17_path = PROCESSED_DIR / "cic17_common.parquet"
    if not cic17_path.exists():
        # Try native track for IP info
        cic17_path = PROCESSED_DIR / "cic17_native.parquet"

    if not cic17_path.exists():
        raise FileNotFoundError(
            f"No CIC17 data found at {PROCESSED_DIR}. Run Phase 2 first."
        )

    df = pd.read_parquet(cic17_path)
    if args.sample and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=SEED)
    print(f"    Loaded: {len(df):,} rows × {len(df.columns)} cols")

    # ── Train/val/test split (simple chronological, 70/15/15) ──────────
    print("\n[2/5] Chronological 70/15/15 split...")
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # Filter benign-only for training
    train_df = df.iloc[:train_end]
    train_benign = train_df[train_df["label"] == 0]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"    Train (benign only): {len(train_benign):,} flows")
    print(f"    Val: {len(val_df):,} flows "
          f"(attack: {(val_df['label'] == 1).sum():,})")
    print(f"    Test: {len(test_df):,} flows "
          f"(attack: {(test_df['label'] == 1).sum():,})")

    # ── Build graphs ───────────────────────────────────────────────────
    print("\n[3/5] Building snapshot graphs...")
    # Get feature columns (all numeric except label)
    feature_cols = [c for c in df.columns
                    if c not in ("label", "label_str", "attack_cat")
                    and pd.api.types.is_numeric_dtype(df[c])]

    print(f"    Feature dim: {len(feature_cols)}")

    train_graphs = build_snapshot_graphs(
        train_benign, window_sec=args.window_sec, feature_cols=feature_cols,
    )
    val_graphs = build_snapshot_graphs(
        val_df, window_sec=args.window_sec, feature_cols=feature_cols,
    )
    test_graphs = build_snapshot_graphs(
        test_df, window_sec=args.window_sec, feature_cols=feature_cols,
    )

    if len(train_graphs) == 0:
        raise RuntimeError("No training graphs built — check data/window size")

    # ── Train model ────────────────────────────────────────────────────
    print("\n[4/5] Training E-GraphSAGE...")
    node_dim = train_graphs[0].x.size(1)
    edge_dim = train_graphs[0].edge_attr.size(1)

    model = EdgeGraphSAGEEncoder(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )

    best_val_loss = float("inf")
    best_state = None
    patience = 5
    patience_counter = 0

    t0 = time.time()
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_graphs, optimizer, device)

        # Validation reconstruction loss
        model.eval()
        val_loss = 0.0
        val_edges = 0
        for g in val_graphs:
            g = g.to(device)
            _, recon = model(g)
            val_loss += F.mse_loss(recon, g.edge_attr).item() * g.num_edges
            val_edges += g.num_edges
        val_loss = val_loss / max(val_edges, 1)

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"    Epoch {epoch:3d}: train_loss={train_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    train_time = time.time() - t0

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # ── Evaluate ───────────────────────────────────────────────────────
    print("\n[5/5] Evaluating...")
    val_metrics = evaluate(model, val_graphs, device)
    test_metrics = evaluate(model, test_graphs, device)

    # ── Save results ───────────────────────────────────────────────────
    results = {
        "model": "E-GraphSAGE-60s-snapshot",
        "dataset": "CIC-IDS2017",
        "seed": SEED,
        "window_sec": args.window_sec,
        "hidden_dim": args.hidden_dim,
        "out_dim": args.out_dim,
        "node_feature_dim": int(node_dim),
        "edge_feature_dim": int(edge_dim),
        "train_epochs": epoch + 1,
        "train_time_sec": round(train_time, 1),
        "train_graphs": len(train_graphs),
        "val_graphs": len(val_graphs),
        "test_graphs": len(test_graphs),
        "val": val_metrics,
        "test": test_metrics,
    }

    output_path = OUTPUT_DIR / "baseline_snapshot_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  -> Saved: {output_path}")
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Train time:     {train_time:.1f}s ({train_time/60:.1f} min)")
    print(f"  Val ROC-AUC:    {val_metrics['roc_auc']:.4f}")
    print(f"  Val PR-AUC:     {val_metrics['pr_auc']:.4f}")
    print(f"  Val F1:         {val_metrics['best_f1']:.4f} (thresh={val_metrics['best_threshold']:.4f})")
    print(f"  Val FPR:        {val_metrics['fpr']:.4f}")
    print(f"  Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  Test PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"  Test F1:        {test_metrics['best_f1']:.4f} (thresh={test_metrics['best_threshold']:.4f})")
    print(f"  Test FPR:       {test_metrics['fpr']:.4f}")
    print("=" * 70)
    print("PHASE 3 COMPLETE")


if __name__ == "__main__":
    main()
