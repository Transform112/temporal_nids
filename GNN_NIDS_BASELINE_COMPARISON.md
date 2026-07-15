# GNN-NIDS Baseline Model Comparison & Selection

Companion to `GNN_NIDS_PLAN.md` and `GNN_NIDS_EXECUTION.md`. This document answers two specific questions:
1. Which existing model is the **most suitable baseline** to implement and compare against in the paper's Experiments section?
2. Which existing model is one **our model is positioned to certainly outperform**, and exactly why — so this comparison can be stated confidently and defensibly in the paper, not just hopefully.

All figures below are the numbers reported in each model's own paper, under that paper's own evaluation protocol — they are **not** directly comparable to each other as-is (different splits, different feature sets, different contamination levels). They are included to establish what each model claims, and to ground the weakness analysis. Every baseline actually used in our Experiments section must be re-run under our own harmonized pipeline and split (Execution Phases 1–5) before any head-to-head number is reported.

---

## 1. Candidate Baseline Models

| Model | Architecture type | Training paradigm | Temporal modeling | Adversarial robustness tested | Explainability | Datasets evaluated on (in its own paper) | Reported headline numbers | Code availability |
|---|---|---|---|---|---|---|---|---|
| **E-GraphSAGE** (Lo et al.) | Edge-feature-aware GraphSAGE, hosts-as-nodes/flows-as-edges | Fully supervised | None (static graph) | None | None | CIC-IDS2017, UNSW-NB15, ToN-IoT, BoT-IoT | CIC-IDS2017: 99.96% detection accuracy, 0.04% FPR. UNSW-NB15: 99.76% F1 | Public reference implementations exist |
| **Anomal-E** (Caville et al.) | E-GraphSAGE encoder + Deep Graph Infomax (DGI) | Self-supervised (no labels), edge-feature-leveraging | None (static graph) | Not adversarial — tested against label-noise ("contamination") robustness, a different property | None | NF-UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2 | Macro-F1 ≈ 85.45% on NF-UNSW-NB15-v2 (0% contamination); clearly outperforms node-feature-only GraphSAGE/DGI baselines used for comparison in its own paper | Public repo (paper authors) |
| **GraphIDS** ("Self-Supervised Learning of Graph Representations for NIDS") | GNN + Transformer masked-autoencoder | Self-supervised (masked reconstruction) | Attempted, but flow order is shuffled — not real chronological temporal modeling | None | None | NF-UNSW-NB15-v3, NF-CSE-CIC-IDS2018 (multiple NetFlow versions) | Reported PR-AUC comparisons vs. Anomal-E; own literature review already flagged Infiltration-class recall as low as 18–25% | Not confirmed public |
| **Pujol-Perich et al.** ("Unveiling the Potential of GNNs for Robust Intrusion Detection") | GNN with feature-level adversarial training | Supervised | None (static graph) | Feature-level only — explicitly no structural adversarial testing | None | Not the focus of re-implementation — donates the feature-level attack protocol only | N/A (methodology paper, not a full system to benchmark against) | N/A |
| **TE-G-SAGE** (2025) | Temporal, edge-aware GraphSAGE + SHAP | Supervised (implied) | Yes — temporal communication graphs, chronological evaluation | None reported | Yes — SHAP-based, edge/feature attribution | NF-UNSW-NB15-v3 | Outperforms a GCN baseline and a tuned XGBoost model on recall under chronological evaluation | Not confirmed public |
| **Classical ML ensemble** (Random Forest / Extra Trees / XGBoost on flat tabular flow features) | Non-graph | Supervised | None | None | Feature-importance only (not GNN-specific) | CIC-IDS2017, UNSW-NB15 (many papers) | Frequently reported at 99.5–99.99% accuracy on both datasets | Trivial (standard sklearn/XGBoost) |
| **StrGNN / TGN** (Zheng et al. / Rossi et al.) | Structural-temporal GNN / continuous-time memory GNN | Self-supervised or supervised depending on application | Yes — this is the architecture family our design is built on | Not evaluated in the NIDS setting in the original papers | None | Not NIDS-specific in their original papers (general graph anomaly / dynamic graph benchmarks) | N/A — architecture donors, not directly benchmarkable NIDS systems | Public reference implementations exist |

---

## 2. Important Cross-Cutting Caution: Near-Saturated Accuracy on CIC17/UNSW-NB15

Both E-GraphSAGE and multiple classical ML baselines report accuracy above 99.5% on CIC-IDS2017 and UNSW-NB15. This is a **known, field-wide caveat**, not evidence that intrusion detection on these datasets is a solved problem:
- These datasets have documented separability artifacts (some attack classes are trivially distinguishable on a single feature, e.g., flow duration or a specific port), which inflates aggregate accuracy without reflecting real-world difficulty.
- Aggregate accuracy/F1 hides per-class weakness — this is exactly how GraphIDS's near-perfect aggregate numbers coexist with an 18–25% recall on Infiltration, a stealthy attack class.
- **Implication for our paper**: never lead with aggregate accuracy against these baselines. Lead with per-attack-category recall (already planned in Plan §10 / Execution Phase 9) and the cross-dataset generalization delta (Execution Phase 10) — this is where a real difference between our model and these baselines will actually show up, and it is a more defensible, reviewer-proof comparison than "we got 99.97% and they got 99.96%."

---

## 3. Selected Primary Baseline: **E-GraphSAGE**

**Recommendation**: implement E-GraphSAGE as the primary head-to-head baseline in the paper's Experiments section.

**Why this one, specifically:**
- It is already the base encoder our own architecture is built on (per the original literature review) — comparing against it isolates exactly the contribution of the temporal memory module, self-supervised objective, adversarial robustness layer, and XAI layer, rather than confounding the comparison with an unrelated architecture change. This is the cleanest possible ablation-style comparison.
- It has publicly reported numbers on **both** of our chosen datasets (CIC-IDS2017 and UNSW-NB15), so a like-for-like re-implementation is feasible without needing to guess at an unfamiliar dataset's behavior.
- It is fully supervised, which gives a clean paradigm contrast against our self-supervised design — useful for arguing the practical advantage of not needing labeled attack data for training.
- Reference implementations are available, keeping re-implementation cost low relative to the rest of the pipeline.

**How to use it fairly**: re-train E-GraphSAGE from scratch inside our own harmonized pipeline (Execution Phase 2 common feature track) and our own chronological split (Execution Phase 5) — do not simply quote the numbers from its original paper, since those were produced under a different split and feature set. The 99.96%/0.04% FPR figures above are a ceiling reference to sanity-check your re-implementation is working correctly, not a number to report as-is in a comparison table.

---

## 4. Selected "Certain-Advantage" Baseline: **GraphIDS**

**Recommendation**: use GraphIDS as the baseline the paper positions itself most directly against, because its own documented weaknesses map one-to-one onto specific, already-built features of our design. This is the safest comparison to feature prominently, because the advantage is not speculative — it is a direct fix for a named, published limitation.

| GraphIDS documented weakness | Our design's corresponding fix | Where it's implemented |
|---|---|---|
| No real temporal modeling — flow order is shuffled during training | Continuous-time TGN-style memory, updated in true chronological order, no shuffling possible by construction | Plan §4, Execution Phase 4/7 |
| Weak on stealthy/mimicry attacks — Infiltration recall only 18–25% | Micro-batching keeps attack signal from being smeared or split (vs. GraphIDS's shuffled-order problem, and vs. our own earlier 60s-snapshot design's boundary-splitting problem); dormant-host reactivation rule specifically targets the low-and-slow evasion pattern that produces this exact failure mode | Plan §4.2, §6.7, Execution Phase 6/11 |
| No adversarial evaluation | Full three-attack robustness suite: feature-level, structural, and TGN-specific slow memory poisoning | Plan §7, Execution Phase 12 |
| 1-hop neighborhood only | Local recomputation still aggregates from immediate neighbors on every update, but the persistent memory vector carries forward accumulated multi-hop-equivalent history over time, rather than resetting at 1 hop every forward pass | Plan §4.2 |
| Still needs labeled data for threshold tuning | Threshold is selected on validation reconstruction error (unlabeled-benign-only assumption preserved) — this is a genuinely lighter labeling requirement, not a full fix, so state this comparison carefully (see caveat below) | Plan §9, Execution Phase 9 |

**Caveat — state this honestly in the paper, do not overclaim:** GraphIDS is also self-supervised and also needs *some* labeled validation data for threshold selection, same as our design (row 5 above is a partial, not total, advantage — phrase it as "comparable labeling burden" in the paper, not "eliminates labeling"). The genuine, unambiguous advantages are rows 1–4: temporal integrity, stealthy-attack handling, adversarial evaluation, and effective receptive field over time. Lead with those.

**Implementation note**: GraphIDS's own code availability is unconfirmed. If a public implementation cannot be found, re-implement a faithful minimal version (GNN encoder + Transformer masked-autoencoder reconstruction, trained with shuffled flow order as originally described) rather than skipping the comparison — the shuffled-order training procedure is the specific mechanism being contrasted against, so it must be reproduced accurately, not approximated.

---

## 5. Secondary Baseline: **Anomal-E**

**Recommendation**: include as a second comparison point specifically to isolate the value of the *temporal* contribution, separate from the self-supervised contribution.

Anomal-E is self-supervised and edge-feature-leveraging, same as our design, but has no temporal modeling at all (static graph, same as the original E-GraphSAGE encoder it's built on). Comparing our model against Anomal-E (both self-supervised, both edge-feature-based) while comparing against E-GraphSAGE (supervised, non-temporal) creates a clean 2x2 story for the paper:

| | Non-temporal | Temporal (ours) |
|---|---|---|
| **Supervised** | E-GraphSAGE | — |
| **Self-supervised** | Anomal-E | Our model |

This framing lets the Results section attribute performance gains specifically to the temporal contribution (ours vs. Anomal-E) and separately to the self-supervised contribution (Anomal-E vs. E-GraphSAGE), rather than presenting one undifferentiated "we're better" number.

Anomal-E's own paper also tests robustness to label contamination — worth citing as related but distinct from our adversarial robustness evaluation (Plan §7); do not conflate the two in the paper's Related Work table.

---

## 6. Related Work to Explicitly Differentiate From: **TE-G-SAGE**

TE-G-SAGE (2025) is the closest existing published work to our own design: it is temporal, edge-aware, built on GraphSAGE, evaluated with a chronological split on NF-UNSW-NB15-v3 (one of our two chosen datasets' NetFlow-formatted variants), and already includes SHAP-based explainability — three of the four pillars of our own contribution (temporal + explainable + one of our datasets).

**This is not a baseline to beat empirically so much as a related-work entry that must be carefully differentiated in the Related Work section**, to avoid a reviewer flagging insufficient novelty. Based on available information, the differentiation points are:
- TE-G-SAGE appears to be **supervised**; our design is primarily self-supervised.
- TE-G-SAGE's temporal handling is not confirmed to be continuous-time/memory-based (TGN-style) — likely a windowed or sequence-based temporal graph, which would still carry some form of the boundary-splitting risk our TGN pivot was specifically designed to avoid. Verify this directly from the paper before finalizing the Related Work paragraph — do not assume.
- No adversarial robustness evaluation is reported for TE-G-SAGE.
- TE-G-SAGE uses SHAP only; our XAI plan adds PGExplainer/GNNExplainer edge-level explanations specifically, which is the more standard comparison point in the GNN-explainability literature for edge-based NIDS.

**Action item**: read the full TE-G-SAGE paper (not just the abstract-level summary here) before finalizing the Related Work comparison table in Plan §11 — this is the one related work entry with enough surface similarity to our contribution that an imprecise differentiation would weaken the paper's novelty claim.

---

## 7. Baselines Deliberately Not Re-Implemented

- **Pujol-Perich et al.**: not a full system, used only as the source of the feature-level adversarial perturbation protocol (already planned in Plan §7.1) — no separate benchmarking needed.
- **StrGNN / TGN (original papers)**: these are the architecture donors for our own temporal module, not competing NIDS systems — nothing to benchmark against directly; their contribution is already folded into our own architecture (Plan §4).
- **Classical ML ensemble (RF/XGBoost/Extra Trees)**: include as a **sanity-check row only**, not a serious comparison target — near-saturated accuracy on these datasets (§2 above) makes it a weak signal of real capability. Report it in the results table for completeness/reviewer expectation, but do not build any part of the paper's argument around beating it.

---

## 8. Recommended Baseline Suite for the Paper's Experiments Section

In this order:

1. **Classical ML ensemble** (Random Forest, one line) — sanity-check floor, minimal implementation effort.
2. **E-GraphSAGE** — primary architecture-isolation baseline (§3).
3. **Anomal-E** — self-supervised, non-temporal comparison point, completes the 2x2 framing (§5).
4. **GraphIDS** — certain-advantage comparison, headline weakness-vs-fix story (§4).
5. **Our model** (chronological split, primary result) vs. **our model** (stratified split, ablation) — per Plan §5/§14.

Every row above must be evaluated on: ROC-AUC, PR-AUC, per-attack-category recall, and the cross-dataset generalization delta (CIC17→UNSW-NB15 and reverse) — the same metric set as Execution Phase 9/10, applied uniformly so the comparison table is apples-to-apples across every model, not just against our own model's two split conditions.

---

## 9. Sources Referenced in This Comparison

- Lo, W. W. et al. — *E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT*
- Caville, E. et al. — *Anomal-E: A Self-Supervised Network Intrusion Detection System based on Graph Neural Networks* (arXiv:2207.06819)
- *Self-Supervised Learning of Graph Representations for Network Intrusion Detection* (GraphIDS; arXiv:2509.16625)
- Pujol-Perich et al. — *Unveiling the Potential of Graph Neural Networks for Robust Intrusion Detection*
- *TE-G-SAGE: Explainable Edge-Aware Graph Neural Networks for Network Intrusion Detection* (Modelling, 2025)
- Various classical-ML NIDS benchmarking papers reporting CIC-IDS2017/UNSW-NB15 accuracy figures cited in §2 (aggregated field observation, not a single-paper claim)

Verify exact author lists, venues, and DOIs for each before adding to the paper's bibliography, per the existing bibliography-confirmation task in Plan §11.
