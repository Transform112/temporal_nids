# NIDS Research Pipeline — Master Execution Sequence

**Generated**: 2026-07-14
**Hardware**: Laptop (i5-9th, 16GB RAM, GTX 1050 3GB) + Kaggle (T4×2)
**Principle**: Local does EDA, preprocessing, baseline sanity, and code dev on subsamples. Kaggle does full training, adversarial, XAI.

---

## 0. Verified Dataset Facts (ground truth, not assumptions)

| Fact | Value | Plan doc was |
|---|---|---|
| CIC-IDS2017 columns | **85** (not 91) | §3.2 said 91 — **wrong** |
| CIC-IDS2017 rows | ~3.12M across 8 daily CSVs | — |
| CIC-IDS2017 header | Present, but "Label" duplicates as data row in every file | — |
| CIC-IDS2017 labels | BENIGN, DDoS, PortScan, Bot, Infiltration, Web Attack (Sql Injection/Brute Force/XSS), FTP-Patator, SSH-Patator, DoS Hulk/GoldenEye/slowloris/Slowhttptest, Heartbleed | §3.2 assumed "- Attempted" variants — **NOT present** (quality upgrade) |
| CIC-IDS2017 line endings | `\r` (CR, not LF) — need `lineterminator` or strip | — |
| UNSW-NB15 raw columns | **49**, NO header row | §3.3 correct on count |
| UNSW-NB15 raw rows | ~2.54M (700K+700K+700K+440K) | — |
| UNSW-NB15 train/test columns | **45** (different structure: has `id`, reordered/renamed) | §3.3 correct |
| UNSW-NB15 column names source | `NUSW-NB15_features.csv` (49 entries) | — |
| GTX 1050 VRAM | **3072 MiB** (3GB) | CLAUDE.md was conservative — confirmed |
| Free disk | 127 GB | — |

### Column name mismatches between raw features file and train/test partition (UNSW-NB15)

| Raw (features.csv) | Train/test CSV | Action |
|---|---|---|
| `Sload` | `sload` | Use lowercase |
| `Dload` | `dload` | Use lowercase |
| `Spkts` | `spkts` | Use lowercase |
| `Dpkts` | `dpkts` | Use lowercase |
| `Sjit` | `sjit` | Use lowercase |
| `Djit` | `djit` | Use lowercase |
| `Sintpkt` | `sinpkt` | Use lowercase |
| `Dintpkt` | `dinpkt` | Use lowercase |
| `Stime` | *(absent)* | Drop — not in 45-col version |
| `Ltime` | *(absent)* | Drop — not in 45-col version |
| *(absent)* | `id` | Only in train/test — drop for harmonization |
| `smeansz` | `smean` | Map `smeansz` → `smean` |
| `dmeansz` | `dmean` | Map `dmeansz` → `dmean` |

---

## 1. Directory Structure

```
temporal_nids/
├── PIPELINE.md                    # THIS FILE — master execution sequence
├── GNN_NIDS_PLAN.md               # What & why (authoritative)
├── GNN_NIDS_EXECUTION.md          # Phase order (authoritative)
├── GNN_NIDS_BASELINE_COMPARISON.md # Baseline models
├── nids/                          # Python package (shared code)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # Chunked CSV → Parquet, header validation
│   │   ├── preprocess.py          # Feature harmonization, scaling
│   │   ├── graph.py               # Flow→graph construction (hosts=nodes, flows=edges)
│   │   └── split.py               # Chronological + stratified split logic
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tgn_memory.py          # TGN GRU memory module
│   │   ├── encoder.py             # Edge-feature-aware GraphSAGE encoder
│   │   ├── decoder.py             # Reconstruction decoder
│   │   └── baselines/
│   │       ├── __init__.py
│   │       ├── e_graphsage.py     # E-GraphSAGE baseline
│   │       ├── anomal_e.py        # Anomal-E (GraphSAGE + DGI)
│   │       └── graphids.py        # GraphIDS (masked autoencoder, shuffled)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── ssl_trainer.py         # Self-supervised reconstruction training
│   │   ├── metrics.py             # All evaluation metrics
│   │   └── checkpoint.py          # Save/resume logic
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── detector.py            # Threshold selection, anomaly scoring
│   │   └── reporter.py            # Results JSON generation
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── feature_perturb.py     # Feature-level adversarial
│   │   ├── structural.py          # Edge injection/removal
│   │   └── memory_poison.py       # TGN slow memory poisoning
│   └── xai/
│       ├── __init__.py
│       ├── pg_explainer.py        # PGExplainer wrapper
│       ├── gnnexplainer.py        # GNNExplainer wrapper
│       └── feature_attr.py        # SHAP/Captum attribution
├── local/                         # LAPTOP EXECUTION
│   ├── eda/
│   │   ├── 01_cicids2017_eda.py   # CIC-IDS2017 exploratory analysis
│   │   ├── 02_unsw_nb15_eda.py    # UNSW-NB15 exploratory analysis
│   │   └── 03_feature_comparison.py # Cross-dataset feature analysis
│   ├── preprocessing/
│   │   ├── 01_validate_and_clean.py  # Data validation + cleaning
│   │   ├── 02_feature_harmonization.py # Common track + native track
│   │   └── 03_split_and_scale.py   # Splitting + scaling
│   ├── baselines/
│   │   ├── 01_rf_baseline.py      # Classical ML sanity check
│   │   └── 02_graph_construction.py # Graph construction validation
│   └── logs/                      # Local execution logs
├── kaggle/                        # KAGGLE EXECUTION
│   ├── training/
│   │   ├── 01_tgn_training.py     # Full TGN model training
│   │   ├── 02_baseline_egraphsage.py # E-GraphSAGE training
│   │   ├── 03_baseline_anomal_e.py   # Anomal-E training
│   │   └── 04_baseline_graphids.py   # GraphIDS training
│   ├── evaluation/
│   │   ├── 01_core_metrics.py     # Phase 9: threshold + metrics
│   │   ├── 02_cross_dataset.py    # Phase 10: cross-dataset generalization
│   │   └── 03_dormancy.py         # Phase 11: dormancy reactivation
│   ├── adversarial/
│   │   ├── 01_feature_attack.py   # Phase 12.1
│   │   ├── 02_structural_attack.py # Phase 12.2
│   │   └── 03_memory_poison.py    # Phase 12.3
│   ├── xai/
│   │   ├── 01_pgexplainer.py      # Phase 13.1
│   │   ├── 02_gnnexplainer.py     # Phase 13.2
│   │   └── 03_feature_attribution.py # Phase 13.3
│   └── logs/                      # Kaggle execution logs (paste here)
└── feature_provenance.json        # Populated by Phase 2
```

---

## 2. EDA Plan (NEW — not in original docs)

Rationale: The plan docs defer data validation. We must understand feature distributions, label balance, temporal patterns, and cross-dataset drift BEFORE designing the model — not after.

### 2.1 EDA Phase A — CIC-IDS2017 (local/eda/01_cicids2017_eda.py)

**Script target**: Laptop, `--sample N` mode defaulting to full (85 cols × 3.1M rows fits RAM in chunks).

| # | Analysis | Chart | Output |
|---|---|---|---|
| A1 | **Column inventory**: 85 column names, dtypes, missing % per column | Table (printed) | Verify against Plan §3.5 map; log mismatches |
| A2 | **Label distribution**: count + % per label, per day | Stacked bar (days × labels) | `eda/cic17_label_dist_day.png` |
| A3 | **Benign vs. Attack ratio**: overall + per day | Pie or donut chart | `eda/cic17_benign_attack_ratio.png` |
| A4 | **Temporal flow density**: flows/minute over the week timeline | Line plot (x=time, y=flows/min) | `eda/cic17_temporal_density.png` |
| A5 | **Feature distributions (top 12 numerical)**: KDE/histogram, colored by BENIGN/Attack | 4×3 grid of KDE plots | `eda/cic17_feature_kde.png` |
| A6 | **Feature-feature correlation (top 20 numerical)**: Spearman correlation heatmap | Heatmap (20×20) | `eda/cic17_corr_heatmap.png` |
| A7 | **Protocol distribution**: pie chart of Protocol values | Pie chart | `eda/cic17_protocol_pie.png` |
| A8 | **Inf/Nan audit**: count of Inf in `Flow Bytes/s`, `Flow Packets/s`, any NaN | Printed table | Confirms Plan §3.2 handling needed |
| A9 | **Duplicate row check**: exact-match duplicates (%) | Printed | Data quality metric |
| A10 | **Duration distribution**: log-scale histogram (µs → seconds conversion noted) | Histogram (log x-axis) | `eda/cic17_duration_hist.png` |

### 2.2 EDA Phase B — UNSW-NB15 (local/eda/02_unsw_nb15_eda.py)

**Script target**: Laptop. Load 4 raw CSVs, assign column names from `NUSW-NB15_features.csv`.

| # | Analysis | Chart | Output |
|---|---|---|---|
| B1 | **Column inventory**: 49 columns with dtypes from features file | Table (printed) | Verify against Plan §3.5 map; log mismatches |
| B2 | **Label distribution**: `attack_cat` (9 categories) + binary `Label` | Stacked bar | `eda/unsw_label_dist.png` |
| B3 | **Category imbalance**: count per attack_cat (check Generic/Exploit dominance) | Horizontal bar chart | `eda/unsw_category_imbalance.png` |
| B4 | **Benign/Attack ratio**: overall | Pie/donut | `eda/unsw_benign_attack.png` |
| B5 | **Feature distributions (top 12 numerical)**: KDE colored by label | 4×3 KDE grid | `eda/unsw_feature_kde.png` |
| B6 | **Feature-feature correlation (top 20 numerical)**: Spearman heatmap | Heatmap (20×20) | `eda/unsw_corr_heatmap.png` |
| B7 | **Protocol (`proto`) distribution**: text labels (tcp/udp/...) | Pie chart | `eda/unsw_protocol_pie.png` |
| B8 | **Service distribution**: values of `service` field | Bar chart (top 15) | `eda/unsw_service_bar.png` |
| B9 | **State distribution**: values of `state` field | Bar chart (top 15) | `eda/unsw_state_bar.png` |
| B10 | **Timeline check**: `Stime`/`Ltime` continuity — gap detection | Line plot (x=time, y=cumulative flows) | `eda/unsw_timeline.png` |
| B11 | **Duplicate check**: exact-match duplicates across 4 files (%) | Printed | Critical — Plan §3.3 flags known duplicates |
| B12 | **Train/test partition comparison**: distribution drift between raw concatenated vs. official train/test split files | Overlaid KDE for top 5 features | `eda/unsw_train_test_drift.png` |

### 2.3 EDA Phase C — Cross-Dataset Feature Comparison (local/eda/03_feature_comparison.py)

| # | Analysis | Chart | Output |
|---|---|---|---|
| C1 | **Feature overlap matrix**: which common-track features exist in both, one-only, neither | Table | Populates `feature_provenance.json` v1 |
| C2 | **Distribution shift**: for each harmonizable feature, overlaid KDE (CIC17 vs. UNSW) | 4×4 grid of overlaid KDEs | `eda/cross_dataset_drift.png` |
| C3 | **Protocol encoding map**: all unique protocol values across both datasets → shared encoding | Table | `eda/protocol_encoding_map.json` |
| C4 | **Duration scale comparison**: CIC17 (µs) vs UNSW (sec) — box plot side by side | Box plot | `eda/duration_scale_compare.png` |
| C5 | **Feature completeness report**: which Plan §3.5 common features can be mapped, which need imputation, which must be dropped | Table → `feature_provenance.json` | **This is the definitive harmonization map** |

---

## 3. Predefined Metrics (for all experiments)

### 3.1 Core detection metrics (every experiment computes these)

```python
METRICS = {
    "roc_auc": "Area under ROC curve",
    "pr_auc": "Area under Precision-Recall curve — LEAD WITH THIS (class imbalance)",
    "f1": "F1-score at validation-selected threshold",
    "precision": "Precision at threshold",
    "recall": "Recall at threshold",
    "fpr": "False Positive Rate at threshold",
    "threshold": "Reconstruction error threshold (val-selected, max F1)",
    "per_category_recall": "dict[attack_category → recall] — categories <5 samples excluded",
}
```

### 3.2 Generalization metrics (cross-dataset only)

```python
GENERALIZATION_METRICS = {
    "pr_auc_delta": "in_distribution_pr_auc - cross_dataset_pr_auc",
    "f1_delta": "in_distribution_f1 - cross_dataset_f1",
    "fpr_delta": "cross_dataset_fpr - in_distribution_fpr",
}
```

### 3.3 Robustness metrics (adversarial only)

```python
ROBUSTNESS_METRICS = {
    "pr_auc_degradation": "clean_pr_auc - attacked_pr_auc at each epsilon/ratio",
    "detection_rate_at_fpr": "recall at fixed FPR (the clean model's operating point)",
}
```

### 3.4 XAI metrics

```python
XAI_METRICS = {
    "fidelity_plus": "Accuracy DROP when explained-important edges are REMOVED (necessity)",
    "fidelity_minus": "Accuracy RETAINED when ONLY explained-important edges are KEPT (sufficiency)",
    "sparsity": "Fraction of graph the explanation uses (lower = more interpretable)",
    "edge_overlap": "Jaccard overlap of top-K edges between PGExplainer and GNNExplainer",
}
```

### 3.5 Reporting hierarchy (what goes in the paper)

| Priority | Metric | Where it appears |
|---|---|---|
| 1 (headline) | PR-AUC + cross-dataset delta | Abstract, Results headline |
| 2 | Per-attack-category recall | Results table (not aggregate accuracy!) |
| 3 | F1 at threshold | Results table |
| 4 | FPR at operating point | Results table |
| 5 | ROC-AUC | Supplementary (for comparability with prior work) |
| Never lead with | Aggregate accuracy | Known field-wide artifact on CIC17/UNSW-NB15 |

---

## 4. Predefined Architecture (locked before implementation)

### 4.1 TGN Memory Module

```
Input: flow events in chronological order, micro-batched by time
State: memory_store: Dict[node_id, Tensor[mem_dim=128]]  # per-host persistent
       memory_ts: Dict[node_id, float]                     # last update timestamp
       time_encoder: Time2Vec(dim=16)                       # temporal encoding
       gru: nn.GRUCell(input_dim=256+16, hidden_dim=128)    # memory update

Node initialization (NOT all-ones — Plan §3.7):
  init_features[node] = [in_degree, out_degree, byte_volume, unique_peers,
                         avg_flow_duration, avg_pkt_size, tcp_ratio, udp_ratio]
  → Linear(8 → 128) → initial memory (learnable projection)

Per-event update:
  1. msg = edge_encoder(event.features)  # [common_dim] → [256]
  2. dt = time_encoder(t - memory_ts[host])
  3. memory[host] = gru(concat(msg, dt), memory[host])
  4. memory_ts[host] = t

Local recomputation scope: touched_hosts ∪ 1-hop-neighbors(touched_hosts)
```

### 4.2 Edge Encoder (shared across CIC17 and UNSW-NB15)

```
Input: common-track features (dim ≈ 17 after harmonization)
Architecture: MLP(common_dim → 128 → 256), BatchNorm, ReLU, Dropout(0.2)
Output: edge_embedding[dim=256]
```

### 4.3 Reconstruction Decoder

```
Input: concat(memory[src], memory[dst], edge_embedding)  # 128+128+256 = 512
Architecture: MLP(512 → 256 → 128 → common_dim)
Output: reconstructed edge features
Loss: MSE(reconstructed, original_edge_features)
```

### 4.4 Baseline Architectures (for comparison)

| Model | Encoder | Paradigm | Temporal | Edge features |
|---|---|---|---|---|
| **Random Forest** | — (flat features) | Supervised | None | Flat tabular |
| **E-GraphSAGE** | 3-layer edge-aware GraphSAGE (64→128→256) | Supervised | None (static snapshot) | Yes |
| **Anomal-E** | E-GraphSAGE encoder + DGI (256→128→256) | Self-supervised (DGI) | None (static snapshot) | Yes |
| **GraphIDS** | GNN + Transformer masked-autoencoder (256→128→256) | Self-supervised (masked) | **Shuffled** (this is the contrast) | Yes |
| **Ours (TGN-NIDS)** | TGN memory + edge-aware encoder + reconstruction | Self-supervised (recon) | Continuous-time TGN | Yes |

### 4.5 Hyperparameters (locked, not swept except where noted)

| Parameter | Value | Lock/Sweep |
|---|---|---|
| Memory dimension | 128 | Locked |
| Edge embedding dimension | 256 | Locked |
| Time encoding dimension | 16 | Locked |
| GRU hidden dim | 128 | Locked |
| Encoder layers | 3 (128→256→256) | Locked |
| Decoder layers | 3 (512→256→128→common_dim) | Locked |
| Dropout | 0.2 | Locked |
| BatchNorm | After every linear | Locked |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-5) | Locked |
| LR schedule | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) | Locked |
| Early stopping patience | 5 epochs | Locked (Plan §8) |
| Early stopping min_delta | 1e-4 | Locked (Plan §8) |
| **Micro-batch size** | **0.5s, 1s, 2s, count=50** | **SWEPT in Phase 6** |
| **Dormancy threshold** | **2min, 5min, 10min** | **SWEPT in Phase 11** |
| **Reactivation N flows** | **3, 5, 10** | **SWEPT in Phase 11** |
| Random seed | 42 | Locked (Phase 0) |
| GPU batch size (laptop) | ≤16 (3GB VRAM) | Adaptive |
| GPU batch size (Kaggle T4) | Auto-tune based on OOM | Adaptive |

---

## 5. Execution Sequence — Phase by Phase

### LEGEND
- 🖥️ = Local (laptop)
- ☁️ = Kaggle
- 📦 = Produces artifact used by later phases

---

### PHASE 0: Environment Setup 🖥️

| Step | Script/Task | Target | Output |
|---|---|---|---|
| 0.1 | Create `nids/` package with `__init__.py` files | Laptop | Package importable |
| 0.2 | `pip install torch torch_geometric pandas numpy scikit-learn matplotlib seaborn shap captum` | Laptop | Requirements |
| 0.3 | Set random seed = 42 in `nids/__init__.py` (single source of truth) | Laptop | `nids/__init__.py` |
| 0.4 | Verify GPU: `nvidia-smi` and `torch.cuda.is_available()` | Laptop | Confirmation log |
| 0.5 | Kaggle environment: install same packages, verify T4×2 | Kaggle | Confirmation log |

---

### PHASE 1: EDA 🖥️ (3 scripts)

| Step | Script | What it produces |
|---|---|---|
| 1.1 | `local/eda/01_cicids2017_eda.py` | 10 analyses (A1–A10), 6 charts, column inventory |
| 1.2 | `local/eda/02_unsw_nb15_eda.py` | 12 analyses (B1–B12), 8 charts, column inventory |
| 1.3 | `local/eda/03_feature_comparison.py` | 5 analyses (C1–C5), cross-dataset drift, definitive harmonization map → `feature_provenance.json` v1 |

**Decision gate after Phase 1**: Review EDA charts. Confirm:
- [ ] 85 CIC17 columns mapped to Plan §3.5 features
- [ ] UNSW-NB15 raw files are the right source (not official train/test)
- [ ] Timeline continuity confirmed (or gaps documented)
- [ ] No showstopper data-quality issues
- [ ] Harmonization map is complete → lock `feature_provenance.json`

---

### PHASE 2: Data Preprocessing & Feature Harmonization 🖥️ 📦

| Step | Script | What it produces |
|---|---|---|
| 2.1 | `local/preprocessing/01_validate_and_clean.py` | Cleaned CSVs → `cic17_cleaned.parquet`, `unsw_cleaned.parquet`. Drops duplicate header rows. Handles Inf→0. Dedup. |
| 2.2 | `local/preprocessing/02_feature_harmonization.py` | Reads `feature_provenance.json`. Produces `cic17_common.parquet`, `cic17_native.parquet`, `unsw_common.parquet`, `unsw_native.parquet` |
| 2.3 | `local/preprocessing/03_split_and_scale.py` | Chronological 70/15/15 split + stratified split. Fits StandardScaler per-dataset train-only. Produces `{dataset}_{split}_{train,val,test}.parquet` |

**Decision gate after Phase 2**:
- [ ] `feature_provenance.json` is complete (every feature traced to source column)
- [ ] No scaler leakage: assertion passes (scaler fit on train indices only)
- [ ] Common track dimensionality locked (= number of successfully mapped features)
- [ ] All `.parquet` files verified loadable and correctly shaped

---

### PHASE 3: Baseline — Random Forest (sanity check) 🖥️

| Step | Script | What it produces |
|---|---|---|
| 3.1 | `local/baselines/01_rf_baseline.py` | RF on flat common-track features. ROC-AUC, PR-AUC, F1, per-category recall. `local/logs/rf_baseline.json` |

**Purpose**: Establish sanity floor. If RF gets 99.5%+ PR-AUC, we confirm the "near-saturated" field-wide caveat — our model won't beat this on aggregate, and shouldn't try.

---

### PHASE 4: Graph Construction Validation 🖥️

| Step | Script | What it produces |
|---|---|---|
| 4.1 | `local/baselines/02_graph_construction.py` | Validates: flow→graph conversion, node initialization (not all-ones), adjacency structure, TGN memory module correctness on a 1-day CIC17 subsample. Unit tests: local recomputation scope, gradient flow. |

---

### PHASE 5: TGN Model Implementation 🖥️ (code, no training)

| Step | What | Where |
|---|---|---|
| 5.1 | Implement `nids/models/tgn_memory.py` | Memory store, GRU update, time encoder, local recomputation |
| 5.2 | Implement `nids/models/encoder.py` | Edge-feature-aware GraphSAGE encoder |
| 5.3 | Implement `nids/models/decoder.py` | Reconstruction decoder |
| 5.4 | Implement `nids/training/ssl_trainer.py` | Self-supervised training loop with checkpointing |
| 5.5 | Implement `nids/training/metrics.py` | All metrics from §3 |
| 5.6 | Unit-test on synthetic 10-node graph | Correctness verification |
| 5.7 | Smoke-test on CIC17 Monday-only subsample (N=10K flows) | `--sample 10000 --target laptop` |

**Decision gate**: All unit tests pass. Memory update touches only correct nodes. Gradient flows through full encoder→decoder path.

---

### PHASE 6: Micro-Batch Ablation ☁️ 📦

| Step | Script | What it produces |
|---|---|---|
| 6.1 | `kaggle/training/01_tgn_training.py --ablation microbatch` | Train on CIC17 1-day slice with 0.5s, 1s, 2s, count=50. Measure F1 + wall-clock time per epoch. |
| 6.2 | Choose best F1 (tiebreak → larger batch) | `kaggle/logs/microbatch_ablation.json` → locked micro-batch constant |

---

### PHASE 7: Baseline Model Training ☁️

| Step | Script | What it produces |
|---|---|---|
| 7.1 | `kaggle/training/02_baseline_egraphsage.py` | E-GraphSAGE on CIC17 + UNSW-NB15 (chronological split, supervised). `kaggle/logs/egraphsage_results.json` |
| 7.2 | `kaggle/training/03_baseline_anomal_e.py` | Anomal-E on CIC17 + UNSW-NB15. `kaggle/logs/anomale_results.json` |
| 7.3 | `kaggle/training/04_baseline_graphids.py` | GraphIDS (shuffled-order reconstruction). `kaggle/logs/graphids_results.json` |

---

### PHASE 8: TGN Self-Supervised Training ☁️ 📦

| Step | Script | What it produces |
|---|---|---|
| 8.1 | `kaggle/training/01_tgn_training.py` | Full TGN training: CIC17-train (benign-only) + UNSW-NB15-train (benign-only). Checkpoint every N steps. Resumable. |
| 8.2 | Output: `cic17_tgn_model.pt`, `unsw_tgn_model.pt`, training logs | `kaggle/logs/tgn_training_log.json` |

---

### PHASE 9: Core Evaluation ☁️ 📦

| Step | Script | What it produces |
|---|---|---|
| 9.1 | `kaggle/evaluation/01_core_metrics.py` | Threshold selection (val F1-max), test metrics, per-category recall, rare-attack table. Both split conditions. All 4 models. |
| 9.2 | Output: `kaggle/logs/core_metrics.json` + rare-attack table | Paper-ready metrics |

---

### PHASE 10: Cross-Dataset Generalization ☁️ 📦

| Step | Script | What it produces |
|---|---|---|
| 10.1 | `kaggle/evaluation/02_cross_dataset.py` | Train-CIC17 → test-UNSW, Train-UNSW → test-CIC17. Source scaler applied. Compute generalization deltas. |
| 10.2 | Output: `kaggle/logs/cross_dataset_results.json` | Generalization-delta table |

**Decision gate**: If cross-dataset PR-AUC drops >50% relative → report as finding, don't silently patch.

---

### PHASE 11: Dormancy Reactivation Rule ☁️

| Step | Script | What it produces |
|---|---|---|
| 11.1 | `kaggle/evaluation/03_dormancy.py` | Sweep dormancy threshold × reactivation window. Evaluate on clean val + synthetic dormancy-exploit cases. |
| 11.2 | Lock dormancy parameters | `kaggle/logs/dormancy_results.json` |

---

### PHASE 12: Adversarial Robustness ☁️ 📦

| Step | Script | What it produces |
|---|---|---|
| 12.1 | `kaggle/adversarial/01_feature_attack.py` | Feature perturbation at ε = 0.01, 0.05, 0.1, 0.2 × feature_std |
| 12.2 | `kaggle/adversarial/02_structural_attack.py` | Edge injection/removal at 5%, 10%, 20% |
| 12.3 | `kaggle/adversarial/03_memory_poison.py` | Slow memory poisoning at 5min, 15min, 30min duration |
| 12.4 | Output: `kaggle/logs/robustness_results.json` | 3 sub-tables → combined robustness figure |

---

### PHASE 13: Explainability ☁️ 📦

| Step | Script | What it produces |
|---|---|---|
| 13.1 | `kaggle/xai/01_pgexplainer.py` | PGExplainer on 200 sampled TPs. Fidelity+/Fidelity-/Sparsity. |
| 13.2 | `kaggle/xai/02_gnnexplainer.py` | GNNExplainer on same 200 TPs. Compare edge overlap. |
| 13.3 | `kaggle/xai/03_feature_attribution.py` | SHAP/Captum on edge features. |
| 13.4 | Case studies: 1 TP + 1 FP visualization | `kaggle/logs/xai_results.json` + 2 case-study figures |

**Decision gate**: If PGExplainer vs. GNNExplainer edge overlap < 0.3 → flag as explanation-stability finding.

---

### PHASE 14: Results Consolidation 📊

| Step | Task |
|---|---|
| 14.1 | Assemble all JSON results into paper-ready tables |
| 14.2 | Confirm chronological split remains primary → only override if stratified shows qualitatively different result |
| 14.3 | TGN vs. snapshot delta: report plainly even if TGN doesn't beat snapshot on raw accuracy |
| 14.4 | Update Related Work comparison table (add UNSW-NB15 row, XAI column, TE-G-SAGE differentiation) |

---

## 6. Kaggle Execution Protocol

### How you (the user) will run Kaggle code

1. I write the script in `kaggle/<subfolder>/<script>.py`
2. You copy it to a Kaggle notebook, run it, paste the full output into `kaggle/logs/<script>_run.log`
3. I read the log, diagnose errors, fix the code, and the cycle continues

### Kaggle notebook structure (thin wrapper pattern)

```python
# Kaggle notebook — thin wrapper
import sys
sys.path.append('/kaggle/working/temporal_nids')
from nids.training.ssl_trainer import train_tgn
from nids.data.loader import load_parquet

# Load preprocessed data (uploaded from local Phase 2)
train_data = load_parquet('/kaggle/input/temporal-nids-data/cic17_common_train.parquet')
model, logs = train_tgn(train_data, config=TGNConfig(target='kaggle'))
```

### Data transfer

After Phase 2 completes locally, you upload these to Kaggle as a dataset:
- `cic17_common_{train,val,test}.parquet`
- `unsw_common_{train,val,test}.parquet`
- `cic17_native_{train,val,test}.parquet`
- `unsw_native_{train,val,test}.parquet`
- `feature_provenance.json`

---

## 7. Immediate Next Steps (what I do now)

1. ✅ Pipeline document created
2. [ ] Create `nids/__init__.py` with seed constant + package setup
3. [ ] Write `local/eda/01_cicids2017_eda.py` — execute locally
4. [ ] Write `local/eda/02_unsw_nb15_eda.py` — execute locally
5. [ ] Write `local/eda/03_feature_comparison.py` — execute locally
6. [ ] Review EDA output → lock harmonization map → proceed to Phase 2

---

## 8. Quick-Reference: Locked Values

| Parameter | Value | Locked at |
|---|---|---|
| Random seed | 42 | Phase 0 |
| Memory dim | 128 | Architecture §4.1 |
| Edge embedding dim | 256 | Architecture §4.2 |
| Time encoding dim | 16 | Architecture §4.1 |
| Optimizer | AdamW(lr=1e-3, wd=1e-5) | Architecture §4.5 |
| Early stopping | 5 epochs, Δ=1e-4 | Execution §8 (fixed) |
| Micro-batch size | *(from Phase 6 ablation)* | Phase 6 |
| Dormancy threshold | *(from Phase 11 sweep)* | Phase 11 |
| Reactivation N | *(from Phase 11 sweep)* | Phase 11 |
| Primary split | Chronological 70/15/15 | Phase 14 (confirmed) |
