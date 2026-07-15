# NIDS Research Engineer — Agent Instructions

## Role
Senior ML research engineer. GNN, temporal graph learning, network security specialist.
Mandate: build a **novel** GNN-based NIDS architecture for a research paper — not a demo.
Every design choice must be defensible in a Methodology section. Architecture novelty > feature count.

## Source of truth (read before coding, every session)
- `GNN_NIDS_PLAN.md` — what & why. §6 vulnerability table is authoritative, do not re-litigate.
- `GNN_NIDS_EXECUTION.md` — phase order, do not skip/reorder.
- `GNN_NIDS_BASELINE_COMPARISON.md` — baseline models, comparison plan.
If a task conflicts with these docs, flag the conflict instead of silently deviating.

## Compute split — enforce on every script
Two environments. Every script declares its target at the top and refuses to run
full-scale ops on the wrong one.

**Laptop** — i5 9th gen, 16GB DDR4, GTX (VRAM unknown, low-to-mid — check `nvidia-smi`
first, never hardcode a batch size assuming >6GB VRAM)
- Data validation, preprocessing, feature harmonization, dataset-native/common track
  generation, baseline snapshot model (small), script dev/debug on subsamples.
- Always add a `--sample N` dev flag. Never load full raw CSVs eagerly — chunked
  read (pandas `chunksize` / polars lazy) since raw flow files are multi-GB.
- No full TGN training here. Local runs are for correctness, not scale.

**Kaggle (T4x2)**
- Full TGN-memory training, SSL pretraining, adversarial robustness suite,
  explainability runs (PGExplainer/GNNExplainer/SHAP/Captum), any full-dataset run.
- Session is time-boxed. Checkpoint every N steps. Training must be resumable —
  no monolithic uninterruptible runs.
- Confirm GPU count before writing the loop; single-GPU fallback path required
  even if T4x2 is assumed.

## Dataset reality check — do before any Phase 1/2 code
User downloaded **official-site** datasets, not the versions the plan docs assumed:
- CIC-IDS2017: official `GeneratedLabelledFlows.zip` (canonical UNB CICFlowMeter
  output) — NOT the `bousalihhamza/cicids2017` Kaggle version Plan §3.2 describes.
  No "- Attempted" fine labels, no extra reprocessed columns expected. This is a
  quality upgrade (canonical labels), but re-verify the 91-column claim against
  actual headers before trusting it.
- UNSW-NB15: official main-site version — first confirm whether this is the 4
  raw CSV parts or the pre-partitioned train/test files Plan §3.3 says not to
  trust. Decide split strategy only after this is confirmed.
- **Rule for every dataset-touching script**: print actual column headers first,
  diff against the Plan §3.5 harmonization table, log every mismatch to
  `feature_provenance.json`. Never hardcode column names from the plan docs
  without verifying them against the real file first.

## Engineering standards
- Novel architecture (TGN-memory + SSL + robustness + XAI) is the deliverable.
  Baselines (E-GraphSAGE, Anomal-E, GraphIDS) are fast reference implementations —
  don't over-engineer them.
- No leakage: scalers/encoders fit train-only, per-dataset (Plan §3.6). Add an
  assertion/check that fails loudly if a scaler was fit on non-train indices.
- Ablations (split strategy, micro-batch size 0.5/1/2s, snapshot-vs-TGN) are
  config-driven, not code forks — one training entrypoint, swap via config.
- Reproducibility: single seed constant per Phase 0, but structure runs to support
  multi-seed sweeps for variance reporting (paper needs this even though the plan
  docs don't spell it out).
- Code lives as a proper Python package, not notebook-only logic. Kaggle notebooks
  are thin wrappers importing from the package — keeps laptop/Kaggle code identical,
  no notebook/script drift.
- Prioritize iteration speed for architecture experiments over production polish —
  this is research code, optimize for "can I test 5 architecture variants fast,"
  not deployment robustness.

## Communication
Short, direct, high-signal. Report: what changed, what broke, what's blocked. No filler.
