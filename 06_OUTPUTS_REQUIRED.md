# Outputs Required — Exact Manifest

All paths relative to `/kaggle/working/outputs/`. Every figure saved as `.png` (300dpi) + `.svg`. Every table saved as `.csv` + `.md`.

## Figures (`figures/`)

| # | Filename stem | Produced in | Content |
|---|---|---|---|
| 01 | fig01_architecture_diagram | NB1 | 9-stage pipeline overview |
| 02 | fig02_graph_construction_diagram | NB1 | Chronological split / separate G_train,val,test |
| 03 | fig03_time2vec_diagram | NB2 | Sinusoidal time embedding concept |
| 04 | fig04_attention_diagram | NB2 | E-GATv2 edge-augmented attention |
| 05 | fig05_mae_pretrain_diagram | NB2 | Mask + FGSM + reconstruction flow |
| 06 | fig06_cvae_diagram | NB3 | CVAE encoder/decoder + latent sampling |
| 07 | fig07_prototypical_diagram | NB6 | Support/query embedding space, prototypes, τ |
| 08 | fig08_class_distribution | NB3 | Pre/post augmentation, both datasets, log scale |
| 09 | fig09_tsne_embeddings | NB7 | 3-panel: post-MAE, post-multiclass, post-fewshot |
| 10 | fig10_adversarial_robustness_curve | NB7 | Macro-F1 vs PGD ε, model vs baselines |
| 11 | fig11_cross_dataset_bar_chart | NB7 | Macro-F1 across in-domain + 4 blind sets |
| 12 | fig12_zero_day_roc_pr | NB6 | Leave-one-class-out novelty detection |
| 13 | fig13_shap_summary | NB7 | Per-class feature importance, top-8/class |
| 14 | fig14_attention_visualization | NB7 | Example subgraph with attention highlighted |
| 15 | fig15_confusion_matrix | NB5 | Normalized, 11-class, in-domain (+ optional cross-dataset variant) |
| 16 | fig16_training_curves | NB2,4,5,6 (assembled NB7) | Loss/macro-F1 vs epoch per stage |

## Tables (`tables/`)

| # | Filename stem | Produced in | Content |
|---|---|---|---|
| 01 | tab01_dataset_statistics | NB1 | Per dataset: total flows, benign/attack, per-class counts |
| 02 | tab02_taxonomy_mapping | NB1 | Raw label → unified class (reuse from `02_ARCHITECTURE.md`) |
| 03 | tab03_feature_schema | NB1 | 44 kept + 17 Time2Vec, grouped |
| 04 | tab04_hyperparameters | Assembled NB7 | Stage/component/params/optimizer/epochs/batch |
| 05 | tab05_main_results | NB5 (+ baselines added NB7) | Macro-F1, per-class recall, FAR, AUC-ROC vs baselines, in-domain |
| 06 | tab06_cross_dataset_results | NB7 | Same metrics, all blind sets, vs baselines |
| 07 | tab07_ablation | NB7 | Full model vs each component removed |
| 08 | tab08_adversarial_robustness | NB7 | Macro-F1 per ε, numeric companion to fig10 |
| 09 | tab09_zero_day_results | NB6 | Per leave-one-out class: precision/recall/F1 |
| 10 | tab10_inference_latency | NB7 | ms/flow, model size, vs baselines if available |
| 11 | tab11_shap_top_features | NB7 | Top-5 SHAP features per class |
| 12 | tab12_related_work_comparison | NB7 (manual/literature) | Method vs {temporal, adversarial, few-shot, cross-dataset} checkmarks |

## Non-figure/table artifacts

| File | Produced in | Purpose |
|---|---|---|
| `label_map.yaml` | NB1 | Unified taxonomy mapping, loaded by all notebooks |
| `feature_manifest.yaml` | NB1 | Feature selection manifest, loaded by all notebooks |
| `environment_snapshot.txt` | NB1 | Pinned package versions |
| `checkpoints/{stage}/best.pt` + `config.json` | NB1–6 | Per-stage model weights + hyperparameters |
| `logs/notebook_{n}_log.json` | NB1–7 | Per-notebook run log |
| `RESULTS_SUMMARY.md` | NB7 | Final consolidated paper-ready results document |

## Verification step (run at end of NB7)

Script/cell that walks `figures/` and `tables/` and asserts all 16 figure stems and 12 table stems exist in both required formats. Print a pass/fail table. Do not consider the project complete until this check passes fully — record the pass in `07_WORK_COMPLETION.md`.
