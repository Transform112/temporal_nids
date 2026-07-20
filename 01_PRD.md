# PRD — Graph-based NIDS for IEEE Access Publication

## 1. Objective

Build a staged (binary → multiclass → few-shot zero-day) graph neural network NIDS, trained on NF3-CSE-CIC-IDS2018 and NF3-UNSW-NB15, that:
- Reaches ~0.99 macro-F1 in-domain.
- Generalizes cross-dataset (blind test on NF3-ToN-IoT, NF3-BoT-IoT, raw CIC-DDoS2019, raw CIC-Darknet2020).
- Stays robust under PGD adversarial perturbation (ε up to 0.05).
- Detects novel/zero-day attack behavior via prototypical few-shot matching.
- Produces a full result set (figures, tables, ablations) sufficient for an IEEE Access submission.

## 2. Success criteria (must all be true at completion)

- [ ] In-domain macro-F1 ≥ 0.97 (target 0.99) on chronological test split, both datasets.
- [ ] Cross-dataset blind macro-F1 reported (no target threshold — must be honestly reported even if lower; this is a generalization study, not a leaderboard chase).
- [ ] Adversarial robustness curve reported at ε = {0, 0.01, 0.03, 0.05}.
- [ ] Zero-day leave-one-class-out detection reported for all 11 unified classes.
- [ ] Ablation table complete: {full model, no-Time2Vec, no-CVAE, no-adversarial-training, no-prototypical-stage}.
- [ ] All 16 figures and 12 tables from the paper figure/table list produced and saved to the correct output paths (see `06_OUTPUTS_REQUIRED.md`).
- [ ] No chronological leakage anywhere in the pipeline (verified via the leakage checklist in `04_IMPLEMENTATION_GUIDE.md`).
- [ ] Every stage's checkpoint, config, and log saved and reproducible from a fresh kernel restart.

## 3. Constraints

- Compute: Kaggle T4x2, fp16 mixed precision mandatory.
- No internet-restricted operations beyond dataset download (Kaggle datasets or UQ eSpace direct links).
- 7-notebook execution structure (see `05_PLAN.md` for notebook-to-stage mapping) — must not collapse into fewer notebooks or exceed 7, since checkpoints are handed off between notebooks via saved artifacts, not shared memory.
- Target venue: IEEE Access. Contribution framing is combination-of-existing-techniques + rigorous cross-domain/adversarial evaluation, not a novel single-technique claim (see `02_ARCHITECTURE.md` novelty framing section).

## 4. Datasets (exact, fixed — no substitutions)

| Dataset | Role | Source |
|---|---|---|
| NF3-CSE-CIC-IDS2018 | Train/val/test | UQ eSpace / Kaggle mirror |
| NF3-UNSW-NB15 | Train/val/test | UQ eSpace / Kaggle mirror |
| NF3-ToN-IoT | Cross-dataset blind test (in-schema) | UQ eSpace |
| NF3-BoT-IoT | Cross-dataset blind test (in-schema) | UQ eSpace |
| CIC-DDoS2019 | Out-of-schema blind test | CIC official |
| CIC-Darknet2020 | Out-of-schema blind test | CIC official |

## 5. Non-goals

- Not building a live/production deployment system — inference pipeline is specified for latency benchmarking only, not shipped as a service.
- Not attempting packet-level (pcap) temporal granularity — explicitly out of scope, documented as a stated limitation.
- Not tuning beyond what's needed to hit success criteria — no open-ended hyperparameter search once target metrics are met, to keep compute budget bounded on T4x2.

## 6. Deliverables

1. Trained model checkpoints for every stage (A–H per `02_ARCHITECTURE.md`).
2. All figures/tables per `06_OUTPUTS_REQUIRED.md`.
3. `label_map.yaml` — the unified 11-class taxonomy mapping.
4. `feature_manifest.yaml` — the finalized feature list per `03_FEATURE_SELECTION.md`.
5. Written results log per notebook (metrics, seeds, runtime) — feeds directly into the paper's results section.
6. Final consolidated results summary (`RESULTS_SUMMARY.md`) generated at the end of notebook 7.

## 7. Definitions (avoid ambiguity for the implementing AI)

- **"Chronological leakage"** — any case where a later-in-time flow's information (label, feature, aggregate stat, or graph neighbor) influences a model's training on an earlier-in-time flow. Zero tolerance.
- **"In-domain"** — test split drawn from the same dataset/time-range distribution as training (G_test from Stage A).
- **"Cross-dataset"** — evaluation on a dataset never seen during any training stage, zero fine-tuning.
- **"Blind test"** — synonym for cross-dataset, used interchangeably in this project.
- **"Stage" vs "notebook"** — a stage (A–J) is an architectural component from `02_ARCHITECTURE.md`; a notebook is an execution unit from `05_PLAN.md`. Multiple stages can share one notebook; no stage may span multiple notebooks without an explicit checkpoint handoff.
