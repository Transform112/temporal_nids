# GNN-NIDS Execution Sequence
**Step-by-step implementation order, with parameters to test and decision rules at each stage**

Companion to `GNN_NIDS_PLAN.md` (read that first for rationale). This document is written so each phase can be implemented without needing to make an open judgment call — where a decision is required, a rule or a search grid is given.

**Do not skip phases or reorder them.** Each phase's output is a required input to the next. Do not start CIC-IDS2018 work (Phase 15) until explicitly gated open.

---

## Phase 0 — Environment & Reference Setup

**Objective**: have a working, reproducible environment and verified access to every source this project depends on.

**Actions**:
1. Set up the Kaggle notebook/environment with: PyTorch, PyTorch Geometric (PyG), PyTorch Geometric Temporal or a hand-rolled TGN memory module, scikit-learn, pandas, numpy.
2. Fix random seeds (numpy, torch, and any sampling step) to a single project-wide seed constant defined once, reused everywhere. Record the seed value in every experiment's output metadata.
3. Verify access to the two paywalled references before citing them in the paper:
   - [1] NID-TGN (Sai et al., 2025) — use the ResearchGate mirror already identified: `researchgate.net/publication/387017532`.
   - [7] "GNNs for anomaly detection" survey (Springer AI Review, 2026) — access via university library proxy or Google Scholar. If neither works, do not cite specifics from it beyond what is available in its public abstract.
4. Create the `feature_provenance.json` schema (empty at this point) that Phase 2 will populate — fields: `dataset`, `common_feature_name`, `source_column`, `transform_applied`, `is_imputed` (bool).

**Output**: working environment; a checked-in note confirming both paywalled sources are actually accessible (or a fallback note if not).

---

## Phase 1 — Dataset Acquisition & Validation

**Objective**: confirm both raw datasets are usable and trustworthy before any preprocessing time is spent on them.

### 1.1 CICIDS2017
- Load `bousalihhamza/cicids2017` from Kaggle.
- **Label spot-check (required, not optional)**: pull a random sample of ~2,000 rows per day, cross-reference their `Label` values against the canonical UNB CICIDS2017 label distribution for that day/attack window (published attack timing tables from the original CICIDS2017 paper). If overall label agreement on the sample is below 95%, flag this explicitly as a limitation in the paper's Dataset section rather than silently proceeding.
- Confirm no NaNs (previously verified); confirm `Flow Bytes/s` / `Flow Packets/s` `Inf` values are still present and still handled by post-aggregation zero-fill.
- Discard `balanced_sample.csv` (unusable, 3 unnamed columns).

### 1.2 UNSW-NB15
- Load the 4 raw part-files (not the official train/test partition — that partition is not trusted per Plan §3.3).
- Print and record the exact column headers from the raw files — do not assume the names listed in the Plan §3.5 table are exact; update the harmonization map with the verified names before Phase 2.
- Check for duplicate rows across the 4 part-files (exact-match dedup on the full feature row) and drop duplicates.
- Encode categorical fields (`proto`, `service`, `state`) — record the encoding scheme used (ordinal vs. one-hot) in `feature_provenance.json`.
- Verify `stime`/`ltime` fields give a genuinely continuous, gap-free-enough timeline suitable for a chronological split; if there are large capture gaps (analogous to CIC17's inter-day gaps), apply the same offset-stitching approach as CIC17 (Plan §3.4).

**Decision rule**: if either dataset fails its validation checks badly enough that label trust is in real doubt (>10% spot-check disagreement, or unresolvable timeline gaps), stop and flag to the paper's limitations section — do not silently proceed with a compromised dataset.

**Output**: two cleaned, validated raw datasets; a validation report (spot-check %, dedup count, timeline continuity confirmation) saved alongside the data.

---

## Phase 2 — Preprocessing & Feature Harmonization

**Objective**: produce, for each dataset, both the dataset-native feature track and the common harmonized track.

1. Implement the harmonization map from Plan §3.5 using the **verified** column names from Phase 1.2.
2. For every mapped feature, log an entry in `feature_provenance.json` (source column, transform applied, whether imputed).
3. For features marked "drop" (no reasonable cross-dataset analogue), exclude from the common track entirely — do not force a mapping.
4. Unit conversion: CIC17 durations are in microseconds, UNSW-NB15's `dur` is in seconds — convert CIC17 to seconds before scaling.
5. Protocol encoding: build one shared categorical encoding for `protocol` covering the union of values seen in both datasets, with an explicit "unknown" bucket for any value seen in only one dataset at test time.
6. Fit `StandardScaler` **separately per dataset**, on that dataset's train split only (train split defined in Phase 5, so this step is revisited after Phase 5 — do not fit scalers before splits exist).

**Output**: `cic17_common.parquet`, `cic17_native.parquet`, `unsw_common.parquet`, `unsw_native.parquet`, plus the populated `feature_provenance.json`.

---

## Phase 3 — Baseline Snapshot Model (cheap comparison point)

**Objective**: produce one directly comparable number against the new TGN architecture, at low cost, using the pipeline that already exists from the earlier project phase.

1. Reuse the existing 60-second-window, hosts-as-nodes/flows-as-edges snapshot construction on CIC17 only (this baseline does not need to be run cross-dataset — it exists purely to justify the architecture switch, not as a headline result).
2. Reuse the existing E-GraphSAGE encoder + reconstruction objective from the prior implementation.
3. Train on the benign-only train split, evaluate on val/test using the metrics in Plan §10.
4. Record ROC-AUC, PR-AUC, F1 at threshold, and wall-clock training time.

**Output**: `baseline_snapshot_results.json` — one row per metric, to be placed in the paper's results table next to the TGN numbers for direct comparison.

---

## Phase 4 — TGN-Style Graph/Memory Construction

**Objective**: build the continuous-time host-memory representation that replaces the 60-second snapshot for the main pipeline.

1. Initialize node memory using the identity-free statistics from Plan §3.7 (in-degree count, out-degree count, byte volume, unique-peer count), computed from the first observation window per host — not all-ones.
2. Implement the **lightweight rolling forensic log** (Plan §6.6): flow 5-tuple (src, dst, port, protocol, timestamp) only — not full feature vectors — retained on disk for a rolling 5-minute window, overwritten continuously. Confirm this does not reintroduce the large-buffer problem the TGN design was meant to avoid (check disk write throughput at expected traffic rate).
3. Implement the GRU-style memory update function per host, triggered on each incoming flow.
4. Implement **local recomputation**: on a new flow, recompute memory only for the two endpoint hosts plus their immediate 1-hop neighbors — not the full graph.

**Output**: memory module implementation, forensic log module, unit tests confirming memory updates only touch the intended local neighborhood (not the full node set) on a synthetic small graph.

---

## Phase 5 — Splitting Implementation (both datasets, both conditions)

**Objective**: implement the two splitting conditions from Plan §5, for both datasets independently.

1. **Chronological split (primary)**: 70/15/15 time-ordered, no stratification, applied separately to CIC17 and UNSW-NB15.
2. **Pooled/stratified split (secondary/ablation)**: benign chronological 70/15/15; attack traffic stratified per category at ~40% val / 60% test; categories with <5 samples go entirely to test.
3. Confirm training set is 100% benign in both conditions, both datasets.
4. **Now** fit the per-dataset `StandardScaler` from Phase 2, step 6, using each dataset's own chronological-split train set.
5. Record final split sizes (row counts and graph/snapshot counts if applicable) for both datasets, both conditions, in a single summary table.

**Output**: `{dataset}_{split_condition}_{train,val,test}.parquet` × 2 datasets × 2 conditions; split-size summary table.

---

## Phase 6 — Micro-Batch Size Ablation

**Objective**: replace the previously unjustified 1–2s micro-batch guess with an empirically chosen value.

1. On a **small data slice** (e.g., one day of CIC17 train+val, to keep this cheap), run the TGN pipeline with micro-batch sizes: **0.5s, 1s, 2s**, and one **count-based** alternative (e.g., batches of 50 flows regardless of elapsed time).
2. For each setting, measure: detection precision/recall/F1 on the val slice, and training wall-clock time per epoch (parallelism proxy).
3. **Decision rule**: choose the setting with the best F1 on the val slice; if two settings are within 1 F1 point of each other, choose the larger batch size (better parallelism, same effective detection quality). Report all four settings in an ablation table — do not report only the winner.
4. Lock the chosen micro-batch size for every subsequent phase. Do not revisit this choice per-dataset — use the same value for both CIC17 and UNSW-NB15 to keep the comparison fair, unless Phase 10's cross-dataset results reveal a clear failure mode traceable to batch size (if so, document as a limitation rather than re-tuning per dataset).

**Output**: `microbatch_ablation.json`, locked micro-batch size constant used from here on.

---

## Phase 7 — Architecture Implementation

**Objective**: assemble the full TGN-style encoder using the components built in Phase 4 and the locked batch size from Phase 6.

1. Edge encoder: process each micro-batch's flows in parallel (GraphSAGE-style edge-feature-aware aggregation) using the harmonized common-track features from Phase 2.
2. Memory update: at each micro-batch boundary, update memory for touched hosts + neighbors only (Phase 4, step 4).
3. Output: a per-host embedding at each micro-batch boundary, and a per-flow (edge) embedding for the reconstruction objective.
4. Keep the encoder architecture identical across CIC17 and UNSW-NB15 runs — only the input feature dimensionality (common track) and scaler differ. This is required for the cross-dataset comparison in Phase 10 to be meaningful.

**Output**: trained-model-ready architecture code, unit-tested on a small synthetic graph for correctness (memory update shape, gradient flow).

---

## Phase 8 — Self-Supervised Training

**Objective**: train the reconstruction-based detector on benign-only data, per dataset.

1. Train two separate models: one on CIC17-train (chronological split), one on UNSW-NB15-train (chronological split). Do not mix training data across datasets at this stage — cross-dataset evaluation happens at test time only (Phase 10), not by pooling training data.
2. Reconstruction objective: reconstruct the memory-based host embedding / flow feature representation (Plan §4.2, §9) — not a static snapshot.
3. Also train two more models on the stratified-split train sets (same benign-only train data, since training set is identical in both conditions per Phase 5 step 3) — actually only the val/test differ between conditions, so this may not require separate training runs; confirm and avoid redundant training if train sets are identical across split conditions.
4. Track training loss curves and wall-clock time per epoch; stop training using early stopping on validation reconstruction loss (patience: 5 epochs, min-delta: 1e-4 — fixed values, not to be tuned per run).

**Output**: `cic17_model.pt`, `unsw_model.pt`, training logs.

---

## Phase 9 — Threshold Calibration & Core Evaluation

**Objective**: pick a detection threshold and compute the headline in-distribution metrics.

1. On each model's own validation set, sweep the reconstruction-error threshold and select the value maximizing F1 (this is the "validation-selected threshold" referenced throughout).
2. Apply that fixed threshold to the corresponding test set (never re-tune on test).
3. Compute, per model, per split condition (chronological primary + stratified secondary): ROC-AUC, PR-AUC, Precision/Recall/F1 at threshold, per-attack-category recall, FPR at the fixed operating point.
4. Build the rare-attack-category table separately (categories with <5 samples, stratified condition only) with the caveat noted in Plan §5.

**Output**: `results_core_metrics.json`, rare-attack-category table, chronological-vs-stratified comparison table.

---

## Phase 10 — Cross-Dataset Generalization Test

**Objective**: the core new experiment enabled by adding UNSW-NB15 — measure real generalization, not just held-out performance from the same capture.

1. Evaluate the CIC17-trained model (Phase 8) on UNSW-NB15's test set (common track, CIC17-fitted scaler applied — per Plan §3.6, do not re-fit).
2. Evaluate the UNSW-NB15-trained model on CIC17's test set (same rule, reversed).
3. Compute the same metric set as Phase 9 for both directions.
4. Compute and report explicitly: **(in-distribution metric) − (cross-dataset metric)** for each core metric (PR-AUC, F1, FPR) — this delta is a headline result, not a footnote.
5. If cross-dataset PR-AUC drops by more than 50% relative to in-distribution, treat this as a genuine finding (report and discuss root causes — e.g., harmonization gaps, synthetic-vs-real attack traffic differences per Plan §3.3) rather than as a bug to silently patch by loosening the harmonization rules.

**Output**: `cross_dataset_results.json`, generalization-delta table for the paper.

---

## Phase 11 — Dormant-Host Reactivation Rule

**Objective**: close the memory-staleness blind spot (Plan §6.7).

1. Define "dormancy": a host with no flows for longer than a fixed threshold — start with **10× the locked micro-batch size worth of elapsed time**, or a flat 5 minutes, whichever is larger. Treat this as a starting value to sweep, not a final answer.
2. On reactivation, apply a lowered anomaly threshold (more suspicious) to that host's first **5 flows** post-reactivation, then return to the normal threshold.
3. Sweep the dormancy threshold and the "first N flows" window over a small grid (dormancy: 2min/5min/10min; N: 3/5/10) on the validation set only; pick the combination that most improves recall on injected-dormancy-exploit synthetic test cases (see Phase 12.3) without increasing FPR on the clean validation set by more than 1 percentage point.
4. Lock the chosen values and apply identically to both CIC17 and UNSW-NB15 models.

**Output**: dormancy-rule parameters, before/after comparison on the synthetic dormancy-exploit test case.

---

## Phase 12 — Adversarial Robustness Testing

**Objective**: run all three attack families from Plan §7 against both trained models on held-out test data.

### 12.1 Feature-level perturbation
- Follow the Pujol-Perich protocol: apply bounded perturbations to edge features at increasing magnitude (e.g., ε = 0.01, 0.05, 0.1, 0.2 of feature std) and measure detection accuracy degradation curve.

### 12.2 Structural perturbation
- Adapt Nettack/Metattack-style edge injection/removal to the edge-level anomaly-detection setting (custom implementation required — no existing library targets this directly).
- Test at a small number of injected/removed edges per attack window (e.g., 5%, 10%, 20% of window's edges) and measure detection accuracy degradation.

### 12.3 TGN-specific slow memory poisoning (new attack, Plan §6.9/§7.3)
- Construct a synthetic test case: inject a sequence of fake low-rate, benign-looking flows targeting a specific host over an extended period (parametrize poisoning duration: 5min/15min/30min), then launch a real attack from/to that host.
- Compare detection accuracy on the poisoned case vs. the same real attack launched without the poisoning warm-up.
- This synthetic case is also the test bed used to validate the dormancy rule in Phase 11.3.

**Decision rule for all three**: report full degradation curves/tables, not single numbers. Do not cherry-pick the mildest perturbation level for the paper's headline claim.

**Output**: `robustness_results.json` with three sub-tables (feature, structural, poisoning), combined into one robustness figure for the paper.

---

## Phase 13 — Explainability (XAI) Integration

**Objective**: implement and evaluate the explainability layer from Plan §8.

1. Implement **PGExplainer** as the primary edge-focused explainer over the trained TGN model.
2. Implement **GNNExplainer** as the comparison baseline (same model, same test instances).
3. Implement a feature-attribution pass using **SHAP or Captum's Integrated Gradients** on the edge feature vectors of flagged anomalies.
4. Compute **Fidelity+**, **Fidelity-**, and **Sparsity** for both PGExplainer and GNNExplainer on a sample of test-set true positives (e.g., 200 flagged anomalies, fixed seed for the sample).
5. Select two illustrative cases for the paper: one true positive, one false positive, each with its explanation subgraph and top-attributed features visualized.

**Decision rule**: if PGExplainer and GNNExplainer disagree substantially on the same instance (e.g., <0.3 edge-overlap on top-5 important edges), flag this in the paper as an explanation-stability finding rather than silently picking whichever looks better.

**Output**: `xai_fidelity_sparsity.json`, two case-study figures, explanation-stability note if triggered.

---

## Phase 14 — Results Consolidation & Reporting Decisions

**Objective**: turn Phases 9–13 outputs into the paper's actual tables/figures, with explicit decision rules so no judgment calls are left open at write-up time.

1. **Primary reported split**: chronological (per Plan §5) — confirm this decision holds given the actual numbers from Phase 9; only override if the stratified condition shows a qualitatively different conclusion (e.g., a rare attack type completely undetected only in one condition) — if so, report both prominently rather than picking one.
2. **Primary architecture claim**: TGN vs. snapshot baseline comparison from Phase 3 vs. Phase 9 — report the delta plainly; if the TGN model does *not* outperform the snapshot baseline on the same dataset, do not suppress this — report it and discuss (this is a legitimate, publishable negative/mixed result given the paper's contribution is broader than raw accuracy).
3. Assemble the Related Work comparison table update (add UNSW-NB15 row, add an "Explainability" column) per Plan §11.

**Output**: finalized results tables/figures ready to paste into the LaTeX draft.

---

## Phase 15 — CIC-IDS2018 Integration (gated, do not start early)

**Gate condition**: only begin once Phase 10's cross-dataset generalization results are stable and the harmonization approach (Phase 2) has proven workable end-to-end (i.e., no unresolved feature-mapping or scaler-leakage issues remain open).

1. Repeat Phases 1–2 (acquisition, validation, harmonization) for CIC-IDS2018, focusing on `02-20-2018.csv` (the only file with retained IP columns, per prior project notes).
2. Do not retrain — evaluate the CIC17-trained and UNSW-NB15-trained models on CIC18 zero-shot/few-shot, using the same cross-dataset methodology as Phase 10.
3. Extend the generalization-delta table (Phase 10 output) with the CIC18 row.

**Output**: extended cross-dataset generalization table (3-dataset version), for a later paper revision or follow-up section.

---

## Phase 16 — Paper Writing Sync

**Objective**: keep the LaTeX draft in lockstep with completed phases rather than writing it all at the end.

- After Phase 2: update Dataset section with UNSW-NB15 subsection + harmonization methodology.
- After Phase 7: fill in the Model Architecture subsection (currently a placeholder).
- After Phase 14: fill in Experiments/Results section, write the new Explainability section, write Conclusion and Abstract last.
- Confirm bibliography entries (GraphIDS, LGSMOTE-IDS, EL-GNN, feature-importance paper, plus the new TGN/XAI references from the temporal-modeling discussion) have exact author names/venues/DOIs before final submission.
- Add co-author details/affiliations to the LaTeX author block once finalized.

---

## Quick-Reference: Locked Parameters (fill in as each phase completes)

| Parameter | Value | Locked in Phase |
|---|---|---|
| Random seed | *(set in Phase 0)* | 0 |
| Micro-batch size | *(chosen in Phase 6)* | 6 |
| Dormancy threshold | *(chosen in Phase 11)* | 11 |
| Reactivation window (N flows) | *(chosen in Phase 11)* | 11 |
| Early-stopping patience / min-delta | 5 epochs / 1e-4 | 8 (fixed, not swept) |
| Primary split condition | Chronological | 5 / confirmed 14 |
