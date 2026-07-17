# GNN-NIDS — Technical Progress Report
**Date:** 2026-07-15 | **Seed:** 42 | **Phases complete:** 0–5 of 16

---

## 1. Dataset Summary

### CIC-IDS2017 (Official UNB `GeneratedLabelledFlows.zip`)
| Property | Value |
|---|---|
| Source | 8 daily CSVs (Mon–Fri, 3 Friday splits) |
| Raw rows | 3,119,345 |
| After dedup | 2,830,541 (**288,804 dupes, 9.26%**) |
| Benign / Attack | 2,272,896 / 557,645 |
| Raw columns | 85 (CICFlowMeter) |
| Key actions | Duration µs→sec, Inf→0 for rate columns, header-duplicate rows stripped |

### UNSW-NB15 (Official ACCS/UNSW — 4 raw CSV parts)
| Property | Value |
|---|---|
| Source | `UNSW-NB15_1.csv` through `UNSW-NB15_4.csv` |
| Raw rows | 2,540,047 |
| After dedup | 2,059,415 (**480,632 dupes, 18.92%**) |
| Benign / Attack | 1,959,772 / 99,643 |
| Raw columns | 49 (from `NUSW-NB15_features.csv`) |
| Key actions | Case-normalized PascalCase columns, mixed-type sport→string, official train/test split discarded |
| Attack categories | 9: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms |

---

## 2. Phase 2: Feature Harmonization

### Common Track Schema (37 columns, identical between datasets)

**16 continuous features:**

| # | Feature | CIC17 Source | UNSW Source | Transform |
|---|---|---|---|---|
| 1 | `duration` | Flow Duration | dur | CIC17 µs→sec |
| 2 | `protocol` | Protocol (IANA #) | proto (text) | Shared categorical index |
| 3 | `fwd_packets` | Total Fwd Packets | Spkts | Passthrough |
| 4 | `bwd_packets` | Total Backward Packets | Dpkts | Passthrough |
| 5 | `fwd_bytes` | Total Length of Fwd Packets | sbytes | Passthrough |
| 6 | `bwd_bytes` | Total Length of Bwd Packets | dbytes | Passthrough |
| 7 | `byte_rate` | Flow Bytes/s | DERIVED: (Sload+Dload)/2 | CIC17 Inf→0 |
| 8 | `packet_rate` | Flow Packets/s | DERIVED: Spkts/dur | Div-by-zero guarded |
| 9 | `mean_iat` | Flow IAT Mean | DERIVED: (Sintpkt+Dintpkt)/2 | Approximation |
| 10 | `std_iat` | Flow IAT Std | DERIVED: (Sjit+Djit)/2 | **Flagged:** jitter ≠ std(IAT) |
| 11 | `mean_pkt_len` | Average Packet Size | DERIVED: (Smeansz+Dmeansz)/2 | Approximation |
| 12 | `syn_count` | SYN Flag Count | DERIVED: 0 | **Imputed** + `_imputed` flag |
| 13 | `ack_count` | ACK Flag Count | DERIVED: 0 | **Imputed** + `_imputed` flag |
| 14 | `init_win_fwd` | Init_Win_bytes_forward | swin | Passthrough |
| 15 | `init_win_bwd` | Init_Win_bytes_backward | dwin | Passthrough |
| 16 | `down_up_ratio` | Down/Up Ratio | DERIVED: dbytes/sbytes | Div-by-zero→0 |
| 17 | `state_summary` | DERIVED from 8 TCP flags | `state` field | **Shared one-hot** (17 buckets) |

**17 state one-hot columns:** `state_acc`, `state_clo`, `state_con`, `state_eco`, `state_ecr`, `state_fin`, `state_int`, `state_mas`, `state_no`, `state_par`, `state_req`, `state_rst`, `state_tst`, `state_txd`, `state_unknown`, `state_urh`, `state_urn`

**CIC17 flag→state mapping:** RST→"RST", FIN→"FIN", SYN-only→"REQ", SYN+ACK→"CON", ACK-only→"INT", URG-only→"URH", none→"no", else→"unknown"

**2 imputed flags:** `syn_count_imputed`, `ack_count_imputed` (CIC17=0, UNSW=1)

**Label columns:** `label` (binary 0/1), plus `label_str` (CIC17 only), `attack_cat` (UNSW only)

### Output Files
| File | Rows | Cols | Size |
|---|---|---|---|
| `datasets/processed/cic17_native.parquet` | 2,830,541 | 85 | 444 MB |
| `datasets/processed/cic17_common.parquet` | 2,830,541 | 28 | 86 MB |
| `datasets/processed/unsw_native.parquet` | 2,059,415 | 57 | 240 MB |
| `datasets/processed/unsw_common.parquet` | 2,059,415 | 37 | 63 MB |

### Validation
- `feature_provenance.json`: 17 features, all `verified: true`
- `protocol_encoding_map.json`: 39-protocol shared encoding
- Zero nulls, zero infs in all output files

---

## 3. Phase 3: Baseline Snapshot Model

### Architecture
```
EdgeGraphSAGEEncoder:
  Input: node_stats(4) + edge_features(26|36)
  Encoder: 2-layer edge-conditioned GraphSAGE
    message = Linear(concat(neighbor_emb, edge_attr)) → mean-aggregate + self-loop
  Output: node embeddings (dim=64)
  Decoder: MLP(concat(src_emb, dst_emb)) → reconstructed edge features
  Objective: MSE reconstruction loss (benign-only training)
```

### Graph Construction
- 60-second non-overlapping windows
- Nodes = unique hosts, edges = flows
- Node features: [in_deg, out_deg, log1p(in_deg), log1p(out_deg)]
- Edge features: common-track feature vector

### Dev-Bench Results (50K sample, no scaling, synthetic IPs)

| Metric | Val | Test |
|---|---|---|
| ROC-AUC | 0.6184 | 0.6326 |
| PR-AUC | 0.3215 | 0.3467 |
| F1 (best thresh) | 0.4020 | 0.4351 |
| FPR | 0.0544 | 0.0518 |

Note: Dev only — synthetic IPs + unscaled. Full native-track run with scaling will produce meaningful numbers.

### File
- `local/baselines/baseline_snapshot_results.json`

---

## 4. Phase 4: TGN Memory Module (Core Architecture)

### Components

**`TGNMemoryModule` (nn.Module):**
- Time encoder: MLP(1→16→16), SiLU — captures inter-flow temporal dynamics
- Edge message function: MLP(2×mem + edge + time → 256 → 256 → mem), SiLU + Dropout
- GRU cell: h_new = GRU(aggregated_messages, h_old)
- Memory init: Linear(5 stats → memory_dim) — identity-free

**`HostMemoryStore`:**
- Per-host memory vectors (dim=128) with last-update timestamps
- 1-hop neighbor adjacency for local recomputation
- Running stats: in_deg, out_deg, byte_vol_in, byte_vol_out, unique_peers
- Lazy memory init on first observation

**`MicroBatchProcessor`:**
- Groups flows into time-slice micro-batches
- Per-batch: msg compute → per-host mean-aggregate → GRU update
- Local recomputation: touched hosts + 1-hop neighbors only
- Neighbor awareness decay for stale neighbors

**`RollingForensicLog`:**
- Stores 5-tuple (src, dst, sport, dport, protocol) + wall-clock timestamp
- 5-min rolling window, 100K entry max, queryable by host+time

### Synthetic Test
```
5 hosts, 7 flows → All memories non-zero and distinct
Forensic log: 7 entries, correct 5-tuples
Local recomputation: Neighbor updates verified
PASSED
```

### Design Decisions (Methodology-defensible)
1. Identity-free stats init (not all-ones) — non-trivial starting state
2. Learnable Fourier-style time encoding — captures periodicity
3. SiLU activations — smoother gradients for temporal sequences
4. Symmetric message passing — both endpoints learn from each flow
5. Neighbor awareness decay — prevents stale-neighbor divergence

### File
- `nids/models/tgn_memory.py` (24 KB, ~400 lines, zero PyTorch Geometric dependency)

---

## 5. Phase 5: Data Splitting

### Split Strategy

| Condition | Train | Val | Test | Role |
|---|---|---|---|---|
| Chronological | First 70% time, benign only | Next 15%, natural mix | Last 15%, natural mix | **Primary headline** |
| Stratified | First 70% benign | Benign chrono + 40% per attack cat | Benign chrono + 60% per attack cat + rare (<5) → test | Ablation |

### CIC-IDS2017 (sampled at 50K)
| Split | Train | Val | Test |
|---|---|---|---|
| Chrono | 30,295 (att=0) | 7,500 (att=818) | 7,500 (att=4,210) |
| Strat | 28,186 (att=0) | 9,933 (att=3,893) | 11,881 (att=5,840) |

### UNSW-NB15 (sampled at 50K)
| Split | Train | Val | Test |
|---|---|---|---|
| Chrono | 33,844 (att=0) | 7,500 (att=594) | 7,500 (att=624) |
| Strat | 33,338 (att=0) | 8,086 (att=942) | 8,576 (att=1,432) |
| Rare→test | — | — | Backdoors(4), Shellcode(4), Worms(2) |

### Scaling
- StandardScaler per-dataset, fit on chronological train ONLY
- CIC17: 26 features, 30,295 train samples
- UNSW: 36 features, 33,844 train samples
- Applied to ALL splits (chrono + strat, train + val + test, scaled + unscaled)
- **No-leakage assertion: PASSED**

### Output Files (30 total)
```
datasets/splits/
  cicids2017_{chrono,strat}_{train,val,test}.parquet         # Unscaled ×6
  cicids2017_{chrono,strat}_{train,val,test}_scaled.parquet   # Scaled ×6
  unswnb15_{chrono,strat}_{train,val,test}.parquet            # Unscaled ×6
  unswnb15_{chrono,strat}_{train,val,test}_scaled.parquet     # Scaled ×6
  cicids2017_scaler.joblib
  unswnb15_scaler.joblib
  split_summary.json
```

---

## 6. Package Structure

```
nids/
├── __init__.py                 # SEED=42, set_seed(), KAGGLE detection
├── data/
│   ├── __init__.py             # Exports preprocessing + splitting
│   ├── preprocessing.py        # Phase 2 — CSV→parquet (680 lines)
│   └── splitting.py            # Phase 5 — splits + scalers (450 lines)
├── models/
│   ├── __init__.py             # Exports TGN + baseline
│   ├── tgn_memory.py           # Phase 4 — TGN core (400 lines)
│   └── baselines/
│       ├── __init__.py
│       └── snapshot_baseline.py # Phase 3 — E-GraphSAGE (650 lines)
├── attacks/__init__.py         # Phase 12 (placeholder)
├── eval/__init__.py            # Phase 9 (placeholder)
├── training/__init__.py        # Phase 8 (placeholder)
└── xai/__init__.py             # Phase 13 (placeholder)
```

All scripts accept `--sample N` for dev mode and `--target laptop|kaggle` for compute routing.

---

## 7. Phases Remaining

| Phase | Status | Compute | Description |
|---|---|---|---|
| 6 | Ready | Laptop(sample) → **Kaggle(full)** | Micro-batch size ablation |
| 7 | Ready | Laptop(dev) → **Kaggle** | TGN encoder assembly |
| 8 | After 7 | **Kaggle** | SSL training (benign-only, both datasets) |
| 9 | After 8 | **Kaggle** | Threshold calibration + core eval |
| 10 | After 9 | **Kaggle** | Cross-dataset generalization |
| 11 | After 10 | **Kaggle** | Dormant-host reactivation |
| 12 | After 11 | **Kaggle** | Adversarial robustness suite |
| 13 | After 12 | **Kaggle** | XAI (PGExplainer, GNNExplainer, SHAP) |
| 14 | After 13 | **Kaggle** | Results consolidation |
| 15 | Gated | TBD | CIC-IDS2018 extension |
| 16 | Continuous | Laptop | Paper writing |

---

## 8. Kaggle Dependencies
```
torch>=2.0
torch_geometric>=2.5
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
pyarrow>=14.0
joblib>=1.3
matplotlib>=3.7
seaborn>=0.12
```
