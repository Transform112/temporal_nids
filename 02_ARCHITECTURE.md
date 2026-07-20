# NIDS — Final Architecture, Data Flow, and Parameters

**Datasets:** NF3-CSE-CIC-IDS2018 + NF3-UNSW-NB15 (train/val/test). NF3-ToN-IoT + NF3-BoT-IoT (in-schema cross-dataset blind test). CIC-DDoS2019 + CIC-Darknet2020 raw (out-of-schema blind test).
**Compute:** Kaggle T4x2, mixed precision (fp16) throughout.
**Target:** ~0.99 macro-F1 in-domain, robust under PGD ε≤0.05, functional zero-day flagging.
**Paper limitation to state explicitly:** NF3 timestamps are flow-level (start/end), not packet-level — CTDNE-style or Time2Vec-style temporal signal is coarser than what raw CICIDS2017 pcaps would give; sub-flow packet timing dynamics are lost. Cite this as a threats-to-validity item.

---

## 0. Data flow overview (top to bottom)

```
Raw NF3 flows (53 fields, 2 datasets)
        ↓
[Stage A] Time-respecting split + graph construction  →  G_train, G_val, G_test (separate physical graphs)
        ↓
[Stage B] Time2Vec temporal encoding  →  appended to edge features
        ↓
[Stage C] E-GATv2 encoder  →  768-dim flow representation per edge
        ↓
[Stage D] MAE self-supervised pretraining (benign only, adversarially regularized)  →  pretrained encoder weights
        ↓
[Stage E] CVAE minority-class augmentation  →  synthetic embeddings, mixed into training pool
        ↓
[Stage F] Binary classifier (Stage-1 head)  →  benign / attack flag
        ↓ (attack-flagged flows only)
[Stage G] Multiclass classifier (Stage-2 head)  →  11-class label + confidence
        ↓ (low-confidence / novel flows)
[Stage H] Prototypical few-shot network  →  known-class match or "novel/zero-day" flag
        ↓
[Stage I] Evaluation — in-domain, in-schema cross-dataset, out-of-schema blind, adversarial robustness curve
```

Every arrow above is a hard boundary: no stage sees data/labels from a later stage during its own training. This is what stops leakage across the whole pipeline, not just at the train/val/test split.

---

## Stage A — Time-respecting split & graph construction

**Why this exists:** random/stratified splits cause look-ahead leakage in temporal graphs (per TE-G-SAGE). Fixing this first prevents every downstream stage from inheriting the bug.

1. Compute cutoffs τ_train, τ_val once, from flow `start_time`, per dataset.
   - E_train = {flow | start_time ≤ τ_train} (70%)
   - E_val = {flow | τ_train < start_time ≤ τ_val} (15%)
   - E_test = {flow | start_time > τ_val} (15%)
2. Persist index sets to disk (parquet) in notebook 1. Every later notebook loads these — never recomputed independently. Prevents split drift across your 7-notebook pipeline.
3. Build **three separate physical graphs**: G_train, G_val, G_test. Nodes = host IP:port endpoints. Edges = flows. G_val's neighbor sampling can never reach an E_train-only edge's future context or vice versa — structural isolation, not a post-hoc mask.
4. Node features: constant/learned embedding only (no running/global aggregate stats computed across the full unsplit dataset — that's a silent leak vector even with correct edge splitting).
5. Feature scaler (z-score) fit on E_train edge features only, applied frozen to val/test.
6. Sliding window for graph batching: 120s window (starting point, tune on val macro-F1), edges within window form the local subgraph passed to the encoder.

**Parameters:** window = 120s (tunable 60–300s), split = 70/15/15 chronological, scaler = StandardScaler (train-fit only).

---

## Stage B — Time2Vec temporal encoding

Replaces the earlier CTDNE plan. No separate walk-training stage; fully inductive.

- φ(t) = [ω₀t + b₀, sin(ω₁t+b₁), ..., sin(ω_kt+b_k)], k=16 (17-dim total, 1 linear + 16 periodic terms).
- ω, b are learned parameters, initialized random, trained jointly with the encoder (not pretrained separately).
- φ(t) computed from flow `start_time`, **normalized using only E_train's time range** (min-max fit on train, applied frozen elsewhere — same discipline as Stage A's feature scaler).
- Output concatenated onto the 53 raw edge features → 70-dim edge input to the encoder.

**Parameters:** k=16 periodic terms, output dim=17, no separate optimizer (trained end-to-end with Stage C).

---

## Stage C — E-GATv2 encoder

- Input: 70-dim edge features (53 raw + 17 time), node init = learned embedding table (128-dim, random init, trainable).
- Layers: 3, hidden dim: 256/layer, attention heads: 8.
- Edge-augmented attention: edge features (70-dim → linear projection to 256) injected into the attention score computation at every layer, not just at aggregation — this is the GATv2 "edge-augmented" variant.
- Dropout: 0.3 attention, 0.2 feature. Activation: ELU. Residual + LayerNorm between layers.
- Neighbor sampling fan-out: [15, 10, 5] across the 3 layers (keeps memory bounded on T4x2).
- Output per flow: concat(src node embed, dst node embed, edge embed) = 256+256+256 = 768-dim flow representation.
- This 768-dim vector is the shared representation every downstream stage (D–H) consumes.

**Parameters:** 3 layers, hidden=256, heads=8, dropout=0.3/0.2, fan-out=[15,10,5], output=768-dim.

---

## Stage D — MAE pretraining (self-supervised, benign traffic only, adversarially regularized)

- Mask ratio: 40% of edge features zeroed per batch (post-Time2Vec-concat, so time signal can also be masked).
- Encoder: Stage C (E-GATv2). Decoder: MLP 256(edge-embed)→128→70, reconstructs masked positions.
- **Adversarial step (from ARGANIDS):** before masking, apply FGSM perturbation to unmasked edge features, ε=0.01–0.03 in normalized feature space. Encoder must reconstruct the clean target despite perturbed + masked input — this is the robustness hook for the whole pipeline.
- Loss: MSE on masked positions only.
- Optimizer: AdamW, lr=1e-3, weight decay=1e-5, cosine annealing schedule, 30 epochs, batch=4096.
- Output: pretrained E-GATv2 weights, carried forward into Stage F (decoder discarded).

**Parameters:** mask=40%, FGSM ε=0.01–0.03, lr=1e-3, epochs=30, batch=4096.

---

## Stage E — CVAE rare-class augmentation

- Input: post-MAE 768-dim flow embeddings + 11-dim one-hot class label (condition).
- Encoder: 768+11 → 256 → 128 → (μ,σ) 64-dim latent.
- Decoder: 64+11 → 128 → 256 → 768.
- Loss: MSE reconstruction + β-KL (β=0.5, avoids posterior collapse).
- Trained only on classes below median class count.
- Optimizer: Adam, lr=5e-4, 50 epochs, batch=512.
- Generation: synthetic embeddings created until each minority class reaches ~40% of majority class count (not full balance — avoids synthetic-dominated training).
- Output: synthetic 768-dim embeddings, injected into Stage G's training pool at a 1:1 ratio with real minority samples.

**Parameters:** latent=64, β=0.5, lr=5e-4, epochs=50, target ratio=40% of majority.

---

## Stage F — Binary classification (Stage-1 head)

- Head: MLP 768→256→64→2, on top of Stage D's pretrained encoder.
- Phase A: freeze encoder, train head only. lr=1e-3, 5 epochs.
- Phase B: unfreeze encoder, joint fine-tune. lr=1e-5 (encoder) / 1e-4 (head), 15 epochs.
- Loss: focal loss (γ=2, α inverse-class-frequency).
- Balancing: per-epoch undersampling of benign to 2:1 benign:attack (resampled every epoch).
- Adversarial training: PGD, ε=0.03, α=0.01, steps=7, applied to 30% of each batch.
- Decision threshold: tuned on val set for attack-class recall ≥ 0.995 (favor recall — Stage G cleans up false positives).
- Output: binary flag per flow, routes attack-flagged flows to Stage G.

**Parameters:** lr=1e-3→1e-5/1e-4, epochs=5+15, focal γ=2, PGD ε=0.03/α=0.01/steps=7, threshold tuned for recall≥0.995.

---

## Stage G — Multiclass classification (Stage-2 head, 11-class)

- Runs only on Stage F's attack-flagged flows.
- Head: MLP 768→256→11. Encoder continues fine-tuning from Stage F state (not reset), lr=1e-5.
- Loss: focal loss (γ=2), per-class α from effective-number-of-samples reweighting (not plain inverse frequency — handles extreme imbalance better).
- Training pool: real minority samples + Stage E synthetic samples (1:1) + per-epoch undersampled majority classes.
- Adversarial training: same PGD config as Stage F.
- Epochs: 20, batch=2048.
- **Per-class threshold calibration** (from TE-G-SAGE finding — pure focal+undersample isn't enough when classes share overlapping features): calibrate decision threshold per class on val set, not one global argmax cutoff.
- Output: 11-class label + confidence score per flow.

**Parameters:** lr=1e-5, epochs=20, batch=2048, focal γ=2 w/ effective-number α, per-class threshold calibration.

---

## Stage H — Prototypical few-shot network (zero-day detection)

- Input: 768-dim flow reps from Stage G's fine-tuned encoder (frozen at this stage).
- Episodic training: 5-way, 5-shot, 15 query/class, 200 episodes/epoch, 30 epochs.
- Prototype = attention-weighted mean of support embeddings (small MLP 768→1 scores each support sample, softmax-weighted sum), not plain mean.
- Distance metric: cosine similarity.
- Inference-time voting: 5 episodic prototype sets (different support draws) ensembled, majority vote — reduces support-sampling variance.
- Novelty flag: if max similarity to any known prototype < τ (τ tuned via leave-one-class-out on a held-out class during dev), flow is flagged "novel/zero-day" instead of forced into a known class.
- Routes here: flows where Stage G confidence < τ2, or where novelty flag trips independently.

**Parameters:** 5-way/5-shot, 15 query, 200 episodes/epoch, 30 epochs, cosine distance, 5-way inference voting ensemble.

---

## Stage I — Evaluation protocol

1. **In-domain:** G_test (chronological holdout), report macro-F1, per-class recall, FAR, AUC-ROC.
2. **In-schema cross-dataset (zero fine-tuning):** train on NF3-CIC2018+NB15, blind test on NF3-ToN-IoT, NF3-BoT-IoT.
3. **Out-of-schema blind (raw, separate feature-mapping pass):** CIC-DDoS2019, CIC-Darknet2020.
4. **Adversarial robustness curve:** macro-F1 at PGD ε = {0, 0.01, 0.03, 0.05} — report the curve, not a single number.
5. **Zero-day eval:** leave-one-attack-class-out during Stage F–G training, test whether Stage H correctly flags it as novel rather than misclassifying into a known class.
6. **Inference latency:** ms/flow, single-flow path vs batched throughput path, target <30ms/flow.

---

## Inference pipeline (deployment-time data flow)

```
Incoming flow
  → NF3 feature extraction (53 fields) + Time2Vec(start_time) → 70-dim
  → insert into sliding-window graph (evict edges outside window)
  → E-GATv2 forward (encoder only) → 768-dim flow rep
  → Stage F binary head → benign (stop) / attack (continue)
  → Stage G multiclass head → label + confidence
      → if confidence ≥ τ2 and not novel: return label
      → else: Stage H prototypical match → known-class-via-prototype or "novel/zero-day" alert
```

- Export: TorchScript/ONNX for encoder + F/G/H heads only (Stage D decoder, Stage E CVAE, Time2Vec's Stage A split logic are training-only, not shipped).
- Target latency: <30ms/flow single-path; batch mode for throughput scenarios.

---

## Unified 11-class attack taxonomy (cross-dataset label mapping)

**Raw label inventory:**
- NF3-UNSW-NB15 attack labels (9): Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms.
- NF3-CSE-CIC-IDS2018 attack labels (6): Bot, DoS, DDoS, Infiltration, Brute Force, Web Attack.
- Raw union = 14 distinct labels (DoS shared). Collapsed to 11 behavioral categories below — merges are grouped by attack *behavior*, not by source dataset, since that's what lets both datasets share one label space for joint training.

| Unified class | Source labels merged in | Behavioral rationale |
|---|---|---|
| Benign | Benign (both datasets) | — |
| DoS/DDoS | UNSW DoS + CIC DoS + CIC DDoS | Same disruption mechanism (volumetric/resource exhaustion), only difference is single-source vs distributed — not worth a separate class |
| Reconnaissance | UNSW Reconnaissance + UNSW Analysis | Analysis (port scans, spam, html penetration probing) is behaviorally pre-attack probing, same family as Reconnaissance |
| Exploits | UNSW Exploits + UNSW Fuzzers | Fuzzing is malformed-input exploitation; both target known/unknown vulnerabilities via crafted payloads |
| Backdoor | UNSW Backdoor | Kept standalone — C2/persistent-access behavior distinct from Bot's automated botnet traffic |
| Bot | CIC Bot | Automated botnet C2 traffic, distinct enough from Backdoor to keep separate (different traffic signature — periodic beaconing vs manual access) |
| Brute Force | CIC Brute Force | — |
| Web Attack | CIC Web Attack (incl. XSS, SQLi as originally bundled in CIC's own labeling) | — |
| Infiltration | CIC Infiltration | — |
| Generic | UNSW Generic | Kept standalone — cipher/block-algorithm attacks are structurally unlike anything else in either dataset, merging would blur signal |
| Shellcode/Worms | UNSW Shellcode + UNSW Worms | Both are payload-delivery/self-propagation mechanisms once a foothold exists — closest behavioral pairing available given both are near-singleton classes |

**Implementation:** build a `label_map.yaml` (raw_label → unified_class) per dataset, applied at Stage A right after loading raw NF3 CSVs, before any split/graph construction. One file, versioned, loaded identically across all 7 notebooks — same discipline as the split-index persistence in Stage A, prevents label-mapping drift.

**Known imbalance after unification (report in paper, justifies Stage E CVAE augmentation targeting):** Generic, Shellcode/Worms, Backdoor, and Infiltration will remain the smallest classes even post-merge — flag these as the primary CVAE augmentation targets and the primary classes to watch in the ablation.

---

## Stage J — Explainable AI (attack-wise feature importance)

Runs after Stage G/H are finalized, on the frozen trained model — doesn't feed back into training, purely for the paper's interpretability section.

**Two complementary methods, cross-validated against each other:**

1. **Native E-GATv2 attention weights (structural importance):** extract per-edge attention scores from the final encoder layer at inference time. Aggregate per unified attack class → shows *which neighboring flows/hosts* the model relies on most when classifying a given class (e.g. does DDoS classification lean on high-fan-in neighbor structure, does Brute Force lean on a single repeated src-dst edge). Cheap — no extra forward passes, weights already computed during normal inference.

2. **SHAP (per-feature importance):** apply KernelSHAP or GradientSHAP on the Stage G multiclass head, using the 70-dim input (53 raw NF3 features + 17 Time2Vec dims) as the attribution space rather than the opaque 768-dim embedding — this keeps SHAP values interpretable to a human reader (e.g. "FLOW_DURATION_MILLISECONDS" or "TCP_FLAGS" rather than embedding dimension 214). Compute per-class mean |SHAP value| across a stratified sample of val-set flows (SHAP is expensive; don't run on full test set — sample ~2000 flows/class, cap total at compute budget).

**Outputs for the paper:**
- One global summary plot: mean |SHAP| per feature, faceted by unified attack class (11 small multiples or one grouped bar chart, top-8 features per class).
- One table: top-5 discriminative features per class, cross-checked against attention-weight findings — where SHAP and attention agree, that's your strongest interpretability claim; where they disagree, discuss it explicitly rather than hiding it (reviewers respect honest disagreement analysis more than a clean-looking story).
- Tie back to Stage G's per-class threshold calibration finding: if SHAP shows two classes (e.g. Backdoor vs Bot) share top features, that explains why those classes needed per-class threshold tuning rather than a shared global threshold — connects your XAI section back to your architecture decisions instead of leaving it as a bolted-on afterthought.

**Parameters:** SHAP sample = ~2000 flows/class (stratified from val set), KernelSHAP background = 100 benign flows, attention extraction = final E-GATv2 layer only (layer 3).

---

## Full parameter summary table

| Stage | Component | Key params | Optimizer/LR | Epochs | Batch |
|---|---|---|---|---|---|
| A | Graph construction | window=120s, split=70/15/15 | — | — | — |
| B | Time2Vec | k=16, dim=17 | joint w/ C | joint | joint |
| C | E-GATv2 | 3 layers, hid=256, heads=8, fanout=[15,10,5] | joint w/ D–G | joint | joint |
| D | MAE pretrain | mask=40%, FGSM ε=0.01–0.03 | AdamW 1e-3 | 30 | 4096 |
| E | CVAE augment | latent=64, β=0.5 | Adam 5e-4 | 50 | 512 |
| F | Binary head | focal γ=2, PGD ε=0.03/steps=7 | 1e-3→1e-5/1e-4 | 5+15 | 4096 |
| G | Multiclass head | focal γ=2, per-class threshold | AdamW 1e-5 | 20 | 2048 |
| H | Prototypical net | 5-way/5-shot, cosine | Adam 1e-4 | 30 | episodic |

Mixed precision (fp16) all stages. Early stopping patience=5 epochs on val macro-F1, all stages. Checkpoint best-val-F1, not final epoch.

---

## Novelty / paper framing recap

- Combination not found elsewhere: Time2Vec + E-GATv2 + adversarially-regularized MAE + CVAE + focal + prototypical, in one time-respecting pipeline.
- Adversarial training folded into a graph-NIDS few-shot pipeline — underexplored intersection per REAL-IoT/ARGANIDS gap.
- Cross-dataset blind eval (in-schema NF3 + out-of-schema raw) — most cited papers skip this; your strongest empirical claim.
- Ablation required for defensibility: report macro-F1 with each of {Time2Vec, CVAE, adversarial training, prototypical stage} removed individually.
- Stated limitation: flow-level (not packet-level) temporal granularity in NF3 — acknowledge explicitly, don't let a reviewer catch it first.
- Unified 11-class behavioral taxonomy across two structurally different datasets (UNSW-NB15's 9 attack types + CIC-IDS2018's 6) is itself a contribution — most cross-dataset papers either stick to binary classification or don't attempt label unification at all.
- XAI section (Stage J) grounds the interpretability claim in two independent signals (attention + SHAP) rather than one, and explicitly ties findings back to the per-class threshold calibration design choice in Stage G — reviewers respond well to interpretability that explains architecture decisions rather than existing as a disconnected add-on figure.
