# GNN-NIDS Project Plan
**Temporally-Aware, Self-Supervised, Adversarially-Robust, Explainable GNN for Network Intrusion Detection**

Status date: reflects all decisions made through the temporal-modeling discussion (TGN pivot) plus the dataset-expansion and vulnerability-mitigation decisions made after that.
This document is the "what and why." See `GNN_NIDS_EXECUTION.md` for the "how, in order."

---

## 1. Project Goal

Build a GNN-based NIDS that is:
- Primarily **self-supervised** (reconstruction-based, trained on benign traffic only), with supervised/semi-supervised classification as a later stage.
- **Temporally aware** using a continuous-time memory architecture (TGN-style), not fixed snapshot windows.
- **Robust to adversarial perturbation** — feature-level, structural, and a novel TGN-specific memory-poisoning attack.
- **Explainable** — every flagged anomaly must come with an edge/feature-level explanation, not just a score.
- **Cross-dataset generalizable** — validated across two independently captured datasets (not just held-out data from one capture).
- Deployable under Kaggle compute constraints (limited GPU memory/session time).
- Scoped as **binary anomaly detection first**; multiclass classification is a deferred Stage 2.

## 2. Core Novelty / Research Gap

No existing published work combines all four of: (a) continuous-time temporal graph modeling, (b) self-supervised training, (c) structural + temporal adversarial robustness evaluation, (d) GNN-specific explainability, in a single NIDS pipeline evaluated across multiple independent datasets. This combination is the paper's contribution.

---

## 3. Datasets

### 3.1 Dataset roster

| Dataset | Role now | Role later |
|---|---|---|
| **CICIDS2017** | Primary train + test | — |
| **UNSW-NB15** | Secondary train + test (cross-dataset generalization pair with CIC17) | — |
| **CIC-IDS2018** | Not used yet | Added only once a concrete cross-dataset strategy is validated on the CIC17/UNSW-NB15 pair (see §12) |

Both CIC17 and UNSW-NB15 are used **symmetrically**: train on one, test on both (in-distribution + cross-dataset), and vice versa. This directly answers the "unproven generalization" gap identified earlier.

### 3.2 CICIDS2017 — known facts and issues
- Source in use: `bousalihhamza/cicids2017` (Kaggle) — a community-reprocessed version with extra columns and finer-grained "- Attempted" labels not present in the canonical UNB release.
- **Known issue**: this dataset has documented label-noise and flow-construction artifacts in the literature (skewed benign/attack separability in some feature combinations). Must be spot-checked against canonical labels (see Execution Phase 1).
- 5 days of capture (Monday–Friday), captured independently per day with large inter-day gaps — requires timeline reconstruction (see §3.4 below, unchanged from earlier decision).
- 91 raw CICFlowMeter columns; `Flow Bytes/s` / `Flow Packets/s` contain `Inf` from near-zero-duration divisions (replace with 0 post-aggregation).
- `balanced_sample.csv` file in this Kaggle source is unusable (3 unnamed columns) — ignore it.

### 3.3 UNSW-NB15 — known facts and issues
- Created by ACCS/UNSW using IXIA PerfectStorm traffic generation; a hybrid of real normal traffic and synthetic attack traffic (less "organically real" than CIC17's attack traffic — note this as a limitation in the paper).
- 49 raw features (Argus + Bro-IDS derived) + `attack_cat` (9 categories: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms) + binary `label`.
- Provided as 4 raw CSV parts, plus a separate official `training-set.csv` / `testing-set.csv` partition (~175K / ~82K rows) with a slightly different column set (includes an `id` column, some renamed fields).
- **Known issues to check for, not assume away:**
  - Reported duplicate records between the official train/test partition files in prior literature — do not trust the official split; re-split yourself using the same chronological methodology as CIC17 (§3.5).
  - Heavy class imbalance within attacks (Generic and Exploits dominate; Worms/Shellcode/Analysis are tiny).
  - Categorical fields (`proto`, `service`, `state`) need explicit encoding — not present in CIC17, so this encoding step is dataset-specific and must not leak into the shared feature pipeline.
  - Feature availability differs structurally from CIC17 (see harmonization map below) — some CIC17 features (e.g., TCP window sizes, several IAT statistics) do not exist in UNSW-NB15 in the same form.
- Exact column names must be verified against the actual CSV header at load time before implementing the harmonization map — names vary slightly between the 4 raw part-files and the official train/test partition file.

### 3.4 Timestamp reconstruction (CIC17 — unchanged decision)
- Each day captured independently; stitched into a single continuous timeline by offsetting each subsequent day's timestamps to start immediately after the previous day ends (Mon→Tue→Wed→Thu→Fri).
- UNSW-NB15 also has real timestamps (`stime`/`ltime` fields) — verify continuity/gaps the same way before assuming a clean chronological order exists; do not assume it mirrors CIC17's structure.

### 3.5 Feature harmonization strategy (new — required for the CIC17 + UNSW-NB15 pairing)

Because the GNN edge encoder needs a fixed-dimension edge feature vector regardless of which dataset produced the flow, define **one common feature space** both datasets are mapped into. Two feature tracks are maintained:

1. **Dataset-native track**: full feature set per dataset, used only for single-dataset experiments (sanity baselines).
2. **Common track**: the harmonized feature set below, used for every cross-dataset experiment (this is the track that matters for the paper's headline results).

**Common feature map (verify exact source column names against each dataset's header before coding):**

| Common feature name | CICIDS2017 source | UNSW-NB15 source | Notes |
|---|---|---|---|
| duration | Flow Duration | dur | scale-align: CIC17 is µs, UNSW-NB15 is seconds — convert to seconds before scaling |
| protocol | Protocol | proto | UNSW-NB15 has text labels (tcp/udp/...), CIC17 has numeric IANA codes — map both to a shared categorical encoding |
| fwd_packets | Total Fwd Packets | spkts | source→dest direction |
| bwd_packets | Total Backward Packets | dpkts | dest→source direction |
| fwd_bytes | Total Length of Fwd Packets | sbytes | |
| bwd_bytes | Total Length of Bwd Packets | dbytes | |
| byte_rate | Flow Bytes/s | (sload+dload)/2 or rate | verify UNSW rate field definition before using |
| packet_rate | Flow Packets/s | rate | |
| mean_iat | Flow IAT Mean | sinpkt / dinpkt (avg) | UNSW splits by direction, CIC17 gives combined — decide: average the two UNSW directions to match |
| std_iat | Flow IAT Std | sjit / djit (as proxy) | jitter is not identical to std-of-IAT; flag as an approximation in the paper |
| mean_pkt_len | Average Packet Size | smeansz / dmeansz (avg) | |
| syn_count | SYN Flag Count | — | not available in UNSW-NB15 in this granularity — **impute 0 + add an `is_imputed` companion flag bit**, do not silently zero-fill |
| ack_count | ACK Flag Count | — | same imputation rule as above |
| init_win_fwd | Init_Win_bytes_forward | swin | |
| init_win_bwd | Init_Win_bytes_backward | dwin | |
| down_up_ratio | Down/Up Ratio | dbytes/sbytes (derived) | guard divide-by-zero |
| state/flags_summary | derived from flag counts | state | UNSW's `state` field is closest analogue to a flow-state summary; use one-hot encoding shared across both datasets with an explicit "unknown" bucket for values only seen in one dataset |

Rules for any feature that cannot be mapped cleanly:
- **Drop from the common track** if it cannot be reasonably approximated (better to lose a weak feature than inject fabricated signal).
- **Never let per-dataset scalers leak** — see §3.6.
- Log every imputed/approximated field explicitly in a `feature_provenance.json` so it is auditable and citable in the paper's limitations section.

### 3.6 Scaling and leakage rules (unchanged principle, now applied per-dataset)
- Fit `StandardScaler` (or equivalent) separately per dataset, on that dataset's **train split only**. Apply to that dataset's val/test.
- When testing cross-dataset (train CIC17 → test UNSW-NB15), apply the **CIC17-fitted scaler** to UNSW-NB15's common-track features. This is intentional: it simulates real deployment (model sees a differently-distributed feed) and any resulting performance drop is a genuine, reportable generalization metric — not a bug to "fix" by re-fitting on the target set.

### 3.7 Node features (revised from the original all-ones placeholder)
- **Old design (deprecated)**: all-ones constant vector, dim=8.
- **New design**: node memory is initialized with a small set of cheap, identity-free statistics computed from the first observation window per host: in-degree flow count, out-degree flow count, total byte volume, unique-peer count in the last N minutes. This is not identity leakage (no IP embedding) and gives the memory module something non-trivial to update from step one, addressing the earlier flagged weakness that an all-ones vector wastes the model's first several updates learning nothing.

---

## 4. Architecture: TGN-Style Continuous Memory (supersedes the 60-second snapshot design)

### 4.1 Why the snapshot design was abandoned
Fixed 60-second non-overlapping windows (the original design) split attacks across window boundaries: an attack that starts near the end of a window is seen as two weak, incomplete signals rather than one strong one, which directly weakens the reconstruction-error detection objective. It also unintentionally helps low-and-slow attacks blend in further. This is a real, structural limitation, not a tuning issue — hence the pivot.

### 4.2 Chosen direction
- **Nodes = hosts**, each with a persistent memory vector updated via a GRU-style update the instant a new flow touches it (TGN-style). No window boundary exists for an attack to be split across.
- **Edges = flows**, processed in **micro-batches** (grouped by short time slices, e.g. 1–2 seconds, exact size to be determined empirically — see Execution Phase 6) rather than per-event or per-60s-window. This keeps most of the training parallelism of a static GNN while shrinking the attack-splitting risk from a 60-second boundary down to a 1–2 second one.
- **Local recomputation**: only the memory of hosts directly touched by a new flow (plus immediate neighbors) is recomputed — not the whole graph. This is what makes the design tractable at real traffic rates (a naive full-graph recomputation per event can collapse throughput by three orders of magnitude at realistic account/host counts, per the TGN literature).
- **No raw flow buffering required** — only the current memory state per host is kept, continuously refreshed. (Partially revised — see §6.6 for the forensics exception.)
- Self-supervised reconstruction objective is adapted to reconstruct the memory-based embedding / flow features rather than a static 60-second snapshot.

### 4.3 Acknowledged, named limitation (state this plainly in the paper — do not hide it)
Sequential memory updates mean a host's memory at time *t* depends on its memory at *t-1*: training cannot be fully parallelized the way a static GNN can. This is a real, literature-acknowledged tradeoff, not a solved problem, and should be reported as such alongside whatever throughput numbers are measured.

### 4.4 Baseline comparison model (kept deliberately, not discarded)
Because the TGN direction is unproven against the original snapshot design on this specific data, a cheap 60-second-snapshot E-GraphSAGE baseline is still built and run side-by-side (see Execution Phase 3). This gives one directly comparable number to defend the architecture switch in the paper, at low implementation cost since the snapshot pipeline already exists from the earlier project phase.

---

## 5. Data Splitting Strategy

Two splitting conditions are implemented and compared for **each dataset independently**, then the same conditions are used again for the cross-dataset (train-on-A/test-on-B) experiments.

1. **Chronological split (primary, headline result)**: strictly time-ordered 70/15/15 split with no stratification. This is the more defensible "realistic" condition and avoids leaking future attack-pattern statistics into validation — report this as the paper's main number.
2. **Pooled/stratified split (secondary, ablation only)**: benign traffic split chronologically as above; attack traffic split stratified per attack category (~40% val / 60% test) so every attack type appears in both val and test. Categories with fewer than 5 samples are kept entirely in test and reported in a **separate small "rare attack" table** with an explicit caveat — do not fold them into headline metrics.
- Training set is 100% benign in both conditions (required for the self-supervised reconstruction objective).
- Report both conditions' metrics side by side; use the chronological split as the number that appears in the abstract/results headline.

---

## 6. Known Vulnerabilities and Built-In Mitigations

This table is the authoritative list of weaknesses identified in review and the specific, already-decided fix for each. Do not re-litigate these during implementation — implement the mitigation as written.

| # | Vulnerability | Mitigation (decided) | Addressed in |
|---|---|---|---|
| 6.1 | CICIDS2017 label noise / flow-construction artifacts | Spot-check a sample of the Kaggle source's labels against canonical UNB CICIDS2017 labels before trusting the full set | Execution Phase 1 |
| 6.2 | Single-dataset generalization claim | UNSW-NB15 added as a second, independently-captured dataset; every core experiment run cross-dataset in both directions | §3.1, Execution Phase 10 |
| 6.3 | All-ones node features carry no signal | Replaced with cheap identity-free statistics (degree, byte volume, unique-peer count) | §3.7 |
| 6.4 | 60-second fixed windows split attacks / help low-and-slow attacks hide | Replaced with TGN-style continuous memory + micro-batching | §4 |
| 6.5 | Split-strategy internal contradiction (chronological benign vs. stratified attack) | Chronological split promoted to primary/headline; stratified kept only as a labeled ablation | §5 |
| 6.6 | Dropping raw buffering removes forensic/audit capability | Lightweight rolling metadata-only log (flow 5-tuple + timestamp, not full features) kept for the last ~5 minutes on disk; cheap, restores auditability without reintroducing the large-buffer problem | Execution Phase 4 |
| 6.7 | Dormant-host memory staleness — a host inactive for a long time reactivates with stale memory, creating a blind spot an attacker could exploit | On reactivation after a defined dormancy threshold, apply a lower (more suspicious) anomaly threshold to that host's first few post-reactivation flows instead of trusting stale memory at face value | Execution Phase 11 |
| 6.8 | Micro-batch size (1–2s) was an unjustified guess | Empirical ablation over a small grid (0.5s / 1s / 2s, plus a count-based alternative), decided by measured precision/recall tradeoff, reported as an ablation table | Execution Phase 6 |
| 6.9 | No adversarial-robustness plan specific to the new TGN architecture (Nettack/Metattack are static-graph attacks) | New TGN-specific "slow memory poisoning" attack designed and tested: inject a sequence of fake low-rate, benign-looking flows over time to shift a host's memory before launching the real attack | §7, Execution Phase 12 |
| 6.10 | Feature-schema mismatch between CIC17 and UNSW-NB15 | Common harmonized feature track with explicit imputation logging (§3.5); dataset-native track kept only for single-dataset sanity baselines | §3.5 |
| 6.11 | TGN vs. snapshot architecture switch is currently justified only theoretically | Cheap 60s-snapshot baseline retained and run side-by-side for a direct comparison number | §4.4 |
| 6.12 | Two cited temporal-modeling references are paywalled ([1] NID-TGN, [7] AI Review survey) | Access [1] via the ResearchGate mirror already identified; access [7] via university proxy/Google Scholar before citing methodology details from it — do not cite specifics you have not actually read | Execution Phase 0 |

---

## 7. Adversarial Robustness Plan

Three attack families are evaluated against the trained model, all on **held-out test data only** (never used to tune the model):

1. **Feature-level perturbation** — following the Pujol-Perich protocol: bounded perturbations to edge (flow) features, measuring detection-accuracy degradation as perturbation magnitude increases.
2. **Structural perturbation** — Nettack/Metattack-style edge injection/removal, adapted from their original static-graph setting to this edge-level anomaly-detection setting. This adaptation is itself a novel contribution and needs a custom implementation (no off-the-shelf library targets this exact setting).
3. **TGN-specific slow memory poisoning (new, see §6.9)** — a sequence of fake low-rate, benign-looking flows injected over an extended period to gradually shift a target host's memory vector before the real attack is launched, then measuring whether detection accuracy drops relative to launching the same real attack without the poisoning warm-up.

Report all three as a single robustness table: clean accuracy vs. accuracy under each attack, at a fixed operating point (the same threshold used for the clean headline number).

---

## 8. Explainability (XAI) Plan

NIDS-appropriate GNN explainability targets **edges** (flows) as the primary explanation unit, since flows — not nodes or raw features in isolation — are what a security analyst needs to see justified ("why was *this flow* flagged").

- **Primary method: PGExplainer** — parametric, edge-focused, generalizes across instances (cheaper at inference time than re-running an optimization per alert, which matters for a NIDS that needs to explain many alerts).
- **Secondary/comparison method: GNNExplainer** — the standard baseline every GNN-NIDS explainability paper compares against; include it for direct comparability with prior work (Anomal-E follow-ups, XGA-E, X-CBA all use this pairing).
- **Feature-attribution layer: SHAP or Captum's Integrated Gradients** applied to the edge feature vector itself, to answer "which *features* of this flow drove the score" as a complement to "which *edges* in the graph drove the score."
- **Evaluation metrics for the explanations themselves** (not just the detector): **Fidelity+** (accuracy drop when the explained-important edges are removed — measures necessity), **Fidelity-** (accuracy retained when only the explained-important edges are kept — measures sufficiency), and **Sparsity** (fraction of the graph the explanation actually uses — smaller is more interpretable at equal fidelity).
- **Deliverable**: a small case-study figure set for the paper — one true-positive detection and one false-positive, each with its explanation subgraph visualized, plus the fidelity/sparsity numbers for the explainer on the test set as a whole.

---

## 9. Self-Supervised Training Objective
- Trained exclusively on the benign-only training split (both datasets, kept separate at training time).
- Reconstruction objective operates on the memory-based host embedding / flow feature representation (revised from reconstructing a static 60s snapshot, per §4.2).
- Anomaly score = reconstruction error; threshold selected on the validation split, never on test.

## 10. Evaluation Metrics
- ROC-AUC, PR-AUC (PR-AUC is the more informative number given class imbalance — lead with it).
- Precision / Recall / F1 at the validation-selected threshold.
- Per-attack-category recall (with the <5-sample-category caveat from §5 applied).
- False Positive Rate at a fixed, stated operating point.
- Cross-dataset generalization drop: (in-distribution metric) − (cross-dataset metric), reported explicitly rather than only reporting the two numbers separately.
- Robustness table per §7.
- Explanation fidelity/sparsity per §8.

## 11. Paper Section Alignment

| Paper section | Status | Source |
|---|---|---|
| Introduction | Drafted | existing LaTeX |
| Related Work | Drafted | existing LaTeX; update comparison table to include the two datasets and XAI column |
| Problem Statement & Threat Model | Drafted | existing LaTeX; extend threat model with the slow-poisoning attack (§7.3) |
| Dataset section | Needs extension | add UNSW-NB15 subsection + harmonization methodology (§3.5) |
| Methodology — feature selection/preprocessing/graph construction | Drafted for CIC17/snapshot version | needs rewrite for TGN architecture + harmonization |
| Model Architecture | Placeholder → now fillable | §4 of this document |
| Experiments/Results | Not started | populate after Execution Phases 9–13 |
| Explainability section (new) | Not started | populate after Execution Phase 13 |
| Conclusion / Abstract | Placeholder | last, after all results exist |

## 12. Deferred / Future Work (explicitly out of scope for now)
- **CIC-IDS2018 integration**: triggered only once the CIC17↔UNSW-NB15 cross-dataset strategy (harmonization map, scaler-leakage handling, chronological split) is validated end-to-end and produces stable, explainable results. Do not begin CIC18 work before that gate is passed.
- **Multiclass classification (Stage 2)**: GraphSMOTE/focal-loss balanced classifier on flagged anomalies.
- **Continual/incremental adaptation**: concept-drift detection + selective incremental update (EL-GNN-inspired).
- **Node-transductive vs. inductive comparison**: currently inductive-only (separate graph universe per split); a transductive comparison is a stretch goal, not required for the first paper submission.
