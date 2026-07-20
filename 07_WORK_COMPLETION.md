# Work Completion Tracker

Update this file as work progresses. Every checkbox here must be checked before the project is considered done. This is the single source of truth for "what's left."

## Handoff Note (2026-07-20)

All planning documents, YAML manifests, paper draft, and 7 notebook scripts have been authored and are ready for execution on Kaggle T4x2. The code is comprehensive but untested on real hardware — expect minor adjustments for Kaggle path configurations, PyG version compatibility, and OOM handling.

**What has been completed (code authorship phase):**
- All 7 notebook Python scripts written
- `label_map.yaml` and `feature_manifest.yaml` created
- `PAPER_DRAFT.md` with Introduction, Related Work, Datasets, Methodology, and Experimental Setup sections drafted
- Execution plan approved

**What remains (execution phase — requires Kaggle T4x2):**
- Run all 7 notebooks sequentially on Kaggle
- Download/find CIC-DDoS2019 and CIC-Darknet2020
- Run ablation variants (4× retraining)
- Fill paper results sections with real numbers
- Verify all citations

## Notebook completion

- [ ] Notebook 1 — data prep, taxonomy, split, graph construction (code written ✓, not yet executed)
- [ ] Notebook 2 — Time2Vec + E-GATv2 + MAE pretraining (code written ✓, not yet executed)
- [ ] Notebook 3 — CVAE augmentation (code written ✓, not yet executed)
- [ ] Notebook 4 — binary classification (code written ✓, not yet executed)
- [ ] Notebook 5 — multiclass classification (code written ✓, not yet executed)
- [ ] Notebook 6 — prototypical few-shot / zero-day (code written ✓, not yet executed)
- [ ] Notebook 7 — evaluation, ablation, XAI, consolidation (code written ✓, not yet executed)

## Artifacts authored (ready for Kaggle)

- [x] `label_map.yaml` — unified 11-class taxonomy mapping for all 4+ datasets
- [x] `feature_manifest.yaml` — 44-kept-features manifest, 61-dim edge input spec
- [x] `PAPER_DRAFT.md` — Introduction, Related Work, Datasets, Methodology, Experimental Setup drafted
- [x] `notebooks/ids_nb1_data_prep.py` — Stage A: data loading, schema verification, chronological split, windowed graph construction, scaler, figs 01-02, tabs 01-03
- [x] `notebooks/ids_nb2_time2vec_mae.py` — Stages B/C/D: Time2Vec, E-GATv2 encoder, MAE pretraining with FGSM, figs 03-05
- [x] `notebooks/ids_nb3_cvae.py` — Stage E: CVAE minority-class augmentation, synthetic embedding generation, figs 06/08
- [x] `notebooks/ids_nb4_binary.py` — Stage F: Binary classifier with PGD adversarial training, threshold calibration
- [x] `notebooks/ids_nb5_multiclass.py` — Stage G: 11-class multiclass with per-class thresholds, fig15, tab05
- [x] `notebooks/ids_nb6_prototypical.py` — Stage H: Prototypical few-shot, novelty threshold tuning, figs 07/12, tab09
- [x] `notebooks/ids_nb7_eval_xai.py` — Stages I/J: Evaluation, cross-dataset, adversarial robustness, t-SNE, SHAP, attention, RESULTS_SUMMARY.md, output verification

## Success criteria (from `01_PRD.md`, restated for tracking)

- [ ] In-domain macro-F1 ≥ 0.97 achieved — value: ____ (TBD after Kaggle execution)
- [ ] Cross-dataset blind results reported for all 4 blind sets (ToN-IoT, BoT-IoT ready; DDoS2019/Darknet2020 deferred)
- [ ] Adversarial robustness curve reported (4 ε values)
- [ ] Zero-day leave-one-out reported for all 11 classes
- [ ] Ablation table complete (4 variants + full model) — variants NOT YET RUN
- [ ] All 16 figures present and correctly formatted
- [ ] All 12 tables present and correctly formatted
- [ ] Leakage checklist passed on every notebook (no unchecked items anywhere)
- [ ] Every stage checkpoint reproducible from fresh kernel restart

## Incident log

| Date/NB | Issue | Resolution | Flagged for human review? |
|---|---|---|---|
| 2026-07-19 / schema check | CICIDS2018 Label column is numeric (0/1) not string "Benign"/"Attack" | NB1 code uses column-name-based access and handles this in label mapping | No — handled in code |
| 2026-07-19 / datasets | CIC-DDoS2019 and CIC-Darknet2020 not yet downloaded | Deferred to Notebook 7; out-of-schema eval will report N/A if unavailable | Yes — user needs to source these |

## Open questions

| Question | Where it came up | Conservative default used | Needs human decision? |
|---|---|---|---|
| Exact Kaggle dataset path | NB1 data loading | Uses `/kaggle/input/ids-nf3-datasets/` — adjust to actual dataset name on Kaggle | User adjusts path when uploading |
| CIC-DDoS2019/Darknet2020 availability | NB7 out-of-schema eval | Reports N/A, notes limitation in paper | Yes — source if possible |
| PyG version on Kaggle | NB2 `GATv2Conv(edge_dim=61)` compatibility | Fallback to manual edge concatenation if `edge_dim` not supported | No — handled in code |
| SHAP compatibility with E-GATv2 | NB7 XAI section | GradientExplainer tried first, KernelSHAP fallback; simplified wrapper approach | No — handled in code |

## Sign-off

- [ ] All notebooks executed start-to-finish without unresolved errors
- [ ] `RESULTS_SUMMARY.md` generated and internally consistent
- [ ] Output verification script passed
- [ ] This file fully updated, no stale unchecked items that are actually done
