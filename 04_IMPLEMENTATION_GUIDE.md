# Implementation Guide — for the implementing AI

Read `01_PRD.md`, `02_ARCHITECTURE.md`, `03_FEATURE_SELECTION.md` before starting. This file resolves ambiguity so no clarification round-trip is needed mid-implementation. If a genuine gap is found that isn't resolved here, log it in `WORK_COMPLETION.md` under "Open questions" and proceed with the most conservative reasonable choice — don't block.

## Environment

- Kaggle notebook, T4x2 GPU, PyTorch (latest stable at time of running, pinned via `!pip freeze > environment_snapshot.txt` in notebook 1 for reproducibility).
- Graph library: PyTorch Geometric (PyG) — use `GATv2Conv` as the base, wrap with edge-augmentation per Stage C spec (PyG's `GATv2Conv` supports `edge_dim` natively — use that parameter rather than hand-rolling edge injection).
- SHAP library: `shap` package, `KernelExplainer` or `GradientExplainer` depending on what's compatible with the PyG model output — test both on a tiny batch first, use whichever runs without custom wrapper hacks.
- Set global seed = 42 at the top of every notebook (`torch.manual_seed`, `np.random.seed`, `random.seed`, plus `torch.backends.cudnn.deterministic = True`). Log the seed in every results file.

## Global rules (apply to every notebook, every stage)

1. **Leakage checklist — run before training any stage, not just once:**
   - [ ] Split indices loaded from the persisted file, never recomputed inline.
   - [ ] Scaler/normalizer fit only on E_train, loaded frozen elsewhere.
   - [ ] `label_map.yaml` and `feature_manifest.yaml` loaded identically, not redefined per-notebook.
   - [ ] No global aggregate statistic (mean, count, running total) computed across the full unsplit dataset before the split is applied.
   - [ ] Graph neighbor sampling for G_val/G_test cannot reach E_train-only future edges or vice versa.
   - [ ] Time2Vec's min-max time normalization is fit on E_train's time range only.
   Print a one-line confirmation of each checked item at the top of every notebook's output — makes leakage auditing possible from the executed notebook alone, without re-reading code.

2. **Checkpointing:** every stage saves its model weights, its exact hyperparameter dict (as JSON, not just code comments), and the epoch/metric at which it was saved. Path convention: `/kaggle/working/checkpoints/{stage_letter}_{stage_name}/best.pt` + `config.json`. Never overwrite a previous stage's checkpoint directory — if rerunning, version it (`best_v2.pt`) and note which version is canonical in `WORK_COMPLETION.md`.

3. **Logging:** every notebook writes a `results_log.json` at `/kaggle/working/logs/notebook_{n}_log.json` containing: start time, end time, all metrics computed, seed, any warnings/anomalies encountered (e.g. NaN loss, OOM recovery, unexpected class counts). This feeds the paper's results section directly — write it as if a human will read it without re-running anything.

4. **Figures and tables — mandatory output discipline:**
   - All figures saved to `/kaggle/working/outputs/figures/` as both `.png` (300dpi, for paper) and `.svg` (editable).
   - Naming: `fig{NN}_{short_name}.png` matching the numbering in `06_OUTPUTS_REQUIRED.md` exactly (e.g. `fig08_class_distribution.png`).
   - All tables saved to `/kaggle/working/outputs/tables/` as both `.csv` (raw data) and `.md` (paper-ready markdown table).
   - Naming: `tab{NN}_{short_name}.csv` / `.md` matching `06_OUTPUTS_REQUIRED.md`.
   - Do not regenerate a figure/table in a later notebook without versioning the old one (`_v1`, `_v2`) — never silently overwrite.

5. **NaN/Inf handling:** if loss goes NaN, don't silently continue. Log it, reduce LR by 10x, resume from last checkpoint, note the incident in `results_log.json`. If it recurs 3 times at the same stage, stop and flag in `WORK_COMPLETION.md` rather than looping indefinitely.

6. **Compute budget awareness:** T4x2, fp16 mandatory. If a batch size in `02_ARCHITECTURE.md` causes OOM, halve the batch size and double gradient accumulation steps to preserve the effective batch size — don't silently shrink effective batch size without logging the change.

## Stage-by-stage implementation notes (fills gaps not in the architecture doc)

**Stage A (graph construction):** Use a bipartite-friendly graph representation — nodes keyed by `(IP, port)` tuple hashed to an integer ID, not raw IP strings (memory efficiency at 20M+ flow scale). Edge attributes = the 44 kept features (Stage `03_FEATURE_SELECTION.md`) + Time2Vec output, computed at graph-build time, not recomputed per epoch.

**Stage B (Time2Vec):** Implement as a small `nn.Module` with learnable `omega` (16,) and `bias` (16,) parameters plus one linear term. Initialize `omega` from a log-uniform distribution spanning expected flow-duration timescales (milliseconds to minutes) rather than pure random — speeds convergence. Document the exact init range used in `results_log.json`.

**Stage C (E-GATv2):** Use PyG's built-in `GATv2Conv(edge_dim=61)`. Verify `edge_dim` support in the installed PyG version at notebook start; if unsupported, fall back to manual edge-feature concatenation into node features before attention (document which path was taken).

**Stage D (MAE + adversarial):** FGSM implementation: single-step gradient sign perturbation on the *normalized* edge feature tensor, clipped to stay within valid normalized range (do not let a perturbed feature go outside the range the scaler could produce from real data — clip to observed train-set min/max per feature after perturbation).

**Stage E (CVAE):** Train per-class or one conditional model for all minority classes together (conditional on class one-hot) — use the conditional approach per spec, don't train 4 separate CVAEs, that fragments the compute budget unnecessarily.

**Stage F/G (classifiers):** Implement per-class threshold calibration (Stage G) via a simple grid search (0.1 to 0.9, step 0.05) per class on the validation set, optimizing per-class F1. Save the resulting threshold vector as part of the stage's `config.json`.

**Stage H (prototypical):** Episode sampling must respect the same G_train/G_val/G_test boundary — support and query sets for a training episode are drawn only from G_train, never crossing into val/test flows even within the same class.

**Stage J (XAI):** Run only after Stage H is finalized and all checkpoints are frozen. SHAP background set = 100 benign flows sampled from E_train (not test) to avoid the background distribution leaking test-time information into attributions.

## Decision authority (what the AI may decide on its own vs must flag)

**May decide autonomously:** exact library versions, minor batch-size adjustments for OOM avoidance (with logging), correlation-pruning specific feature drops within the 3-4 limit, plotting library choice (matplotlib/seaborn/plotly — pick one, use consistently), random seed value if 42 is somehow unavailable.

**Must flag in `WORK_COMPLETION.md` and use conservative default, not block:** any metric that comes in more than 5 points below the PRD's success criteria after a full training run (don't silently retry with different hyperparameters indefinitely — flag it, try one reasonable adjustment, then report honestly if still short), any schema mismatch between expected and actual downloaded dataset fields, any case where a stage's output shape doesn't match the next stage's expected input (fix the connective code, but log that a mismatch was found and resolved).
