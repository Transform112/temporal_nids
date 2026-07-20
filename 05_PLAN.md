# Execution Plan — 7 Notebooks

Each notebook lists: inputs (what it must load), outputs (what it must produce/checkpoint), and its checklist. Do not proceed to the next notebook until every checklist item in the current one is checked.

---

## Notebook 1 — Data prep, taxonomy, split, graph construction (Stage A + label/feature manifests)

**Inputs:** raw NF3-CSE-CIC-IDS2018, raw NF3-UNSW-NB15 CSVs.

**Outputs:**
- `label_map.yaml`, `feature_manifest.yaml`
- Persisted split indices (parquet) for E_train/E_val/E_test, both datasets
- G_train, G_val, G_test graph objects (serialized, e.g. `.pt` via PyG `Data`/`HeteroData`)
- Fitted scaler object (`.pkl`)
- `fig01_architecture_diagram` (reuse the already-rendered version, export to file)
- `tab01_dataset_statistics`, `tab02_taxonomy_mapping`, `tab03_feature_schema`

**Checklist:**
- [ ] Both datasets downloaded, schema verified against `feature_manifest.yaml` expectations
- [ ] `label_map.yaml` applied, unified 11-class distribution computed and logged
- [ ] Chronological split computed once, persisted, leakage checklist run and printed
- [ ] Three separate physical graphs built (not one graph with a mask)
- [ ] Scaler fit on E_train only, applied to all splits
- [ ] `fig02_graph_construction_diagram` produced (illustrates chronological split / separate graphs concept)
- [ ] Class distribution logged pre-augmentation (feeds `fig08` later, computed here, saved for reuse)

---

## Notebook 2 — Time2Vec + E-GATv2 encoder + MAE pretraining (Stage B, C, D)

**Inputs:** Notebook 1's graphs, scaler, feature manifest.

**Outputs:**
- Time2Vec module weights (joint-trained, checkpointed with encoder)
- Pretrained E-GATv2 encoder checkpoint (`checkpoints/D_mae_pretrain/best.pt`)
- `fig03_time2vec_diagram`, `fig04_attention_diagram`, `fig05_mae_pretrain_diagram`
- `results_log.json` for notebook 2 (reconstruction loss curve, adversarial ε used)

**Checklist:**
- [ ] Time2Vec time normalization fit on E_train time range only
- [ ] Benign-only filtering confirmed correct before MAE training starts
- [ ] FGSM perturbation clipped to observed train feature range
- [ ] Reconstruction loss curve plotted → feeds `fig16_training_curves`
- [ ] Encoder checkpoint saved with config (mask ratio, ε, layer dims)

---

## Notebook 3 — CVAE minority-class augmentation (Stage E)

**Inputs:** Notebook 2's pretrained encoder (frozen, used to generate embeddings), Notebook 1's class distribution.

**Outputs:**
- CVAE checkpoint
- Synthetic embedding pool (saved separately from real data, tagged `is_synthetic=True`)
- `fig06_cvae_diagram`
- `fig08_class_distribution` (pre vs post augmentation, both datasets)
- `tab04_hyperparameters` (can be assembled incrementally across notebooks, finalized in notebook 7)

**Checklist:**
- [ ] CVAE trained only on below-median classes
- [ ] Synthetic pool capped at ~40% of majority class count per minority class
- [ ] Synthetic samples clearly tagged, never mixed into val/test — training pool only
- [ ] Class distribution before/after plotted on log scale

---

## Notebook 4 — Binary classification, Stage-1 head (Stage F)

**Inputs:** Notebook 2's encoder checkpoint (continues fine-tuning from here, not from scratch).

**Outputs:**
- Stage F checkpoint (encoder + binary head, both phases A and B)
- Calibrated decision threshold
- `results_log.json` — binary precision/recall/F1, threshold value, PGD ε used
- Contributes to `fig16_training_curves`, `fig10_adversarial_robustness_curve` (binary-stage numbers)

**Checklist:**
- [ ] Phase A (frozen encoder) then Phase B (joint fine-tune) run in order, both checkpointed
- [ ] Per-epoch undersampling confirmed re-sampling each epoch, not static
- [ ] PGD adversarial training applied to the specified 30% batch fraction
- [ ] Threshold tuned for attack-class recall ≥ 0.995, value logged

---

## Notebook 5 — Multiclass classification, Stage-2 head (Stage G)

**Inputs:** Notebook 4's Stage F checkpoint, Notebook 3's synthetic embeddings.

**Outputs:**
- Stage G checkpoint
- Per-class threshold vector
- `fig15_confusion_matrix` (in-domain)
- `tab05_main_results` (in-domain rows populated here)

**Checklist:**
- [ ] Only attack-flagged flows (from Stage F) used for training/eval here
- [ ] Real minority + synthetic minority mixed 1:1 as specified
- [ ] Per-class threshold grid search run and logged
- [ ] Confusion matrix generated on chronological test split

---

## Notebook 6 — Prototypical few-shot, zero-day detection (Stage H)

**Inputs:** Notebook 5's Stage G checkpoint (frozen encoder from here on).

**Outputs:**
- Prototypical network checkpoint
- Novelty threshold τ (from leave-one-class-out dev tuning)
- `fig07_prototypical_diagram`
- `fig12_zero_day_roc_pr`
- `tab09_zero_day_results`

**Checklist:**
- [ ] Episodic sampling confirmed to respect G_train/G_val/G_test boundaries
- [ ] Attention-weighted prototype computation implemented (not plain mean)
- [ ] Leave-one-class-out run for all 11 classes, results tabulated
- [ ] 5-way inference voting ensemble implemented and used for reported numbers

---

## Notebook 7 — Evaluation, ablation, XAI, final consolidation (Stage I, J + wrap-up)

**Inputs:** all prior checkpoints, all prior figures/tables, both blind-test dataset families.

**Outputs — evaluation:**
- `fig09_tsne_embeddings` (post-MAE, post-multiclass, post-fewshot — 3 panels)
- `fig10_adversarial_robustness_curve`, `tab08_adversarial_robustness`
- `fig11_cross_dataset_bar_chart`, `tab06_cross_dataset_results`
- `tab07_ablation` — rerun with each component removed: no-Time2Vec, no-CVAE, no-adversarial-training, no-prototypical-stage
- `tab10_inference_latency`

**Outputs — XAI (Stage J):**
- `fig13_shap_summary`, `fig14_attention_visualization`, `tab11_shap_top_features`

**Outputs — wrap-up:**
- `tab12_related_work_comparison` (populated from literature, not computed)
- `RESULTS_SUMMARY.md` — one consolidated document pulling every metric from every notebook's `results_log.json` into paper-ready form
- Final `WORK_COMPLETION.md` update — every checklist item across all 7 notebooks marked done or flagged

**Checklist:**
- [ ] Blind-test evaluation run with zero fine-tuning on ToN-IoT/BoT-IoT (in-schema)
- [ ] Out-of-schema raw CIC-DDoS2019/Darknet2020 evaluated via separate feature-mapping pass, clearly labeled as out-of-schema in results
- [ ] Ablation reruns use identical seeds/splits to the main run for fair comparison
- [ ] SHAP computed on 70... **44+17=61-dim** input space (per corrected `03_FEATURE_SELECTION.md`), not the 768-dim embedding
- [ ] All 16 figures and 12 tables present in `/kaggle/working/outputs/` and match `06_OUTPUTS_REQUIRED.md` exactly
- [ ] `RESULTS_SUMMARY.md` complete and internally consistent (no contradicting numbers across sections)
