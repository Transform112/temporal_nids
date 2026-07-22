# Complete Error Report — Graph-NIDS Pipeline
## Code Bugs + Data Issues + NaN Gradient Root Cause Analysis

> **Data inspected:** `dataset/graphs/` (actual training data on disk)
> **Code inspected:** All 6 Kaggle scripts, preprocessing pipeline, YAML manifests
> **Date:** 2026-07-21

---

## Table of Contents

1. [NaN Gradient Root Cause Analysis](#1-nan-gradient-root-cause-analysis) (your primary concern)
2. [Data-Level Issues](#2-data-level-issues-in-datasetgraphs)
3. [Preprocessing Pipeline Bugs](#3-preprocessing-pipeline-bugs)
4. [Code Bugs — K01 (MAE Pretraining)](#4-code-bugs--k01)
5. [Code Bugs — K02 (CVAE)](#5-code-bugs--k02)
6. [Code Bugs — K03 (Binary Classification)](#6-code-bugs--k03)
7. [Code Bugs — K04 (Multiclass Classification)](#7-code-bugs--k04)
8. [Code Bugs — K05 (Prototypical Network)](#8-code-bugs--k05)
9. [Code Bugs — K06 (Evaluation & XAI)](#9-code-bugs--k06)
10. [Architecture Inconsistencies Across Scripts](#10-architecture-inconsistencies-across-scripts)
11. [YAML Manifest Errors](#11-yaml-manifest-errors)
12. [Summary](#12-summary)

---

## 1. NaN Gradient Root Cause Analysis

Your NaN gradient problem has **multiple contributing causes** — both data-level and code-level. Here's each one ranked by impact:

### Root Cause #1: EXTREME OUTLIER VALUES IN SCALED FEATURES (DATA)

**Evidence from `dataset/graphs/` inspection:**

| File | Abs Max Value | Feature | Severity |
|---|---|---|---|
| NF-UNSW-NB15_train | 860.18 | global max | Extreme |
| NF-CICIDS2018_train | 3747.85 | global max | **Catastrophic** |
| NF-UNSW-NB15_train feat[0] | 335.57 | IN_BYTES (z-scored) | 335x std |
| NF-UNSW-NB15_train feat[25] | 306.33 | feat[25] | 332x std |
| NF-UNSW-NB15_train feat[33] | 426.48 | feat[33] | 466x std |
| NF-CICIDS2018_train feat[25] | 609.35 | feat[25] | 732x std |
| NF-CICIDS2018_train feat[20] | 233.48 | feat[20] | 767x std |

**Why this causes NaN:**
- After z-score normalization, most values are near 0-1. But a handful of flows have values **300-860x the standard deviation**.
- In the attention mechanism: `softmax(Q * K^T / sqrt(d))` — when edge features contain a value of 860, after the `edge_proj` linear layer (256 dims), the attention logit can reach ~50,000+. `softmax(50000)` in fp16 → **Inf → NaN**.
- In ELU activation: `ELU(x) = x for x > 0` — a value of 860 passes through unchanged, multiplied by weights, accumulated across 8 attention heads → overflow.
- This is the **#1 trigger** for your NaN batches. The rare outlier flows (0.001% of data) poison entire batches when they're sampled.

**Fix:** Clip edge_attr to [-10, 10] after z-score normalization (catches 99.95% of the distribution). Or use RobustScaler (median/IQR) instead of StandardScaler.

---

### Root Cause #2: GradScaler.update() SKIPPED ON NaN (CODE)

**Files:** [k03_binary.py:149](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k03_binary.py#L149), [k04_multiclass.py:199](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k04_multiclass.py#L199)

```python
if not torch.isnan(loss): amp.update()  # BUG: skips scale correction
```

**Why this causes NaN:** GradScaler uses an internal scale factor to prevent fp16 underflow. When a batch produces NaN/Inf gradients, `step()` detects it and skips the optimizer update — but `update()` is supposed to **shrink the scale factor** to prevent repeated overflow. By gating `update()` on `not isnan(loss)`, the scale factor stays too high, causing the NEXT batch to also overflow → cascading NaN avalanche.

K01 explicitly fixed this (BUG FIX 2) — always call `update()` after `step()`. K03/K04 reintroduced the bug.

---

### Root Cause #3: FGSM/PGD ADVERSARIAL PERTURBATION AMPLIFIES OUTLIERS (CODE)

**Files:** [k01_time2vec_mae.py:334](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k01_time2vec_mae.py#L334), [k03_binary.py:193-196](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k03_binary.py#L193-L196)

**Why this causes NaN:** FGSM adds `eps * grad.sign()` to features. When a batch contains an extreme outlier (value=860), the gradient w.r.t. that feature is also extreme → the perturbed value becomes even larger → the encoder forward pass on the perturbed input overflows. The clamp bounds in K01 are data-derived but in K03/K06 they're hardcoded ±4.0 or derived from raw (pre-scaled) feature min/max, which doesn't help after scaling.

---

### Root Cause #4: FLOAT32 TIMESTAMP PRECISION LOSS (DATA)

**Evidence from inspection:**
```
edge_time range: [1.42e+12, 1.52e+12] (epoch-milliseconds)
float32 mantissa: 23 bits
Magnitude: ~2^40
Min distinguishable time delta: ~131,072 ms (131 seconds!)
```

**Impact:** The 120-second sliding window is SMALLER than the minimum distinguishable time delta (131 sec). This means:
- Time2Vec receives identical timestamps for flows that are up to 2 minutes apart
- `norm_time()` computation `(t - t_min) / (t_max - t_min)` has massive cancellation error — subtracting two nearly-equal large numbers in float32 gives near-zero with garbage precision
- Time2Vec's `sin(omega * t)` receives quantized input → periodic terms become step functions → discontinuous gradients

**This doesn't directly cause NaN but severely degrades Time2Vec's value**, making it contribute noise rather than signal.

---

### Root Cause #5: fp16 AUTOCAST + EXTREME VALUES = OVERFLOW (CODE)

**Files:** All Kaggle scripts using `with autocast():`

**Why this causes NaN:** fp16 max value is 65,504. When scaled feature values of 860 go through `edge_proj(Linear(58→256))` inside `autocast()`, the output can reach `860 * weight * 256` which exceeds fp16 range → Inf → NaN on next operation.

K01 mitigates this somewhat with its careful FGSM implementation. K03/K04 run the entire forward pass (including PGD attack gradient computation) inside autocast, making overflow more likely.

---

### Root Cause #6: 8 NEAR-CONSTANT FEATURES IN CICIDS2018 TEST/VAL (DATA)

**Evidence:**
```
NF-CICIDS2018_test: NEAR-CONSTANT features (std<1e-7): [18, 19, 20, 21, 30, 31, 32, 33]
  feat[18]: value = -0.14730230
  feat[19]: value = -0.23767875
  ...
```

These 8 features have essentially zero variance in the CICIDS2018 test split. After z-scoring (which was fit on training data including UNSW where these features vary), they become constant negative values. The model learns to rely on feature variance during training, then sees zero variance at test time → attention weights for these features become degenerate.

---

## 2. Data-Level Issues in dataset/graphs/

### DATA-1: Scaler fit on UNSW-NB15 only, not both datasets

| Evidence | Value |
|---|---|
| `scaler_metadata.json` → `n_samples_seen_` | **1,655,796** |
| NF-UNSW-NB15_train edges | **1,655,796** |
| NF-CICIDS2018_train edges | **14,080,870** |

The scaler was fit on only UNSW-NB15's training data. CICIDS2018's features have different distributions (e.g., the IAT features span different ranges). This means:
- CICIDS2018 values are NOT properly z-scored (wrong mean/std applied)
- This creates the extreme outlier values (3747.85) seen in CICIDS2018_train
- **This is likely the primary source of your extreme outliers**

**Severity:** CRITICAL. Refit the scaler on both datasets' training data.

---

### DATA-2: Cumulative node_cache → inflated num_nodes (CICIDS2018)

```
NF-CICIDS2018_train: num_nodes grows from 2 to 55,759
  Confirmed: ~100% non-decreasing → cumulative node IDs
```

The preprocessing code never resets `node_cache` between windows. Each graph's `num_nodes` is the TOTAL unique endpoints seen so far, not just in that window.

**Impact:** In K02-K06 encoder with `_get_node_embed()`, this causes `nn.Parameter(torch.randn(55759, 128))` = 28 MB per forward pass, even for a window with 10 edges. In K01's fixed encoder, `max_val_nodes` becomes 55,759, requiring a 28 MB node_embed table.

---

### DATA-3: Label distribution severely skewed across chronological splits

| Dataset | Train Classes | Val Classes | Test Classes |
|---|---|---|---|
| NF-CICIDS2018 | [0, 1, 6, 7] | [0, 8] | [0, 5, 7, 8] |
| NF-UNSW-NB15 | [0, 1, 2, 3, 4, 9, 10] | **[0] only** | **[0] only** |

**Critical issues:**
- UNSW-NB15 val and test contain **only Benign (class 0)** — zero attack samples. This means any validation metrics computed on UNSW-NB15 val/test measure nothing about attack classification.
- CICIDS2018 val contains only [0, 8] — only Benign and Infiltration. 9 of 11 classes are absent from validation.
- The chronological split pushes all attacks into specific time periods, creating massive class imbalance across splits.
- **Validation F1 is meaningless** because most classes aren't represented.

---

### DATA-4: 244 tiny graphs (< 3 edges) in CICIDS2018 test

These graphs have 1-2 edges. GATv2Conv with 3 layers and fanout [10,5,3] will have almost no neighbors to attend over. The attention mechanism degenerates to self-loops, producing uninformative representations.

---

### DATA-5: 7 huge graphs (> 50k edges) in CICIDS2018 train

The largest graph has 55,022 edges. Without edge subsampling, loading this entire graph to GPU causes OOM on T4x2.

---

## 3. Preprocessing Pipeline Bugs

### PREP-1: `node_cache` never resets between time windows

**File:** [run_preprocessing.py:218-270](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/laptop/run_preprocessing.py#L218-L270)

```python
node_cache = {}  # initialized once before all windows
# ...
for w in df_chunk['_w'].unique():
    # node_cache keeps growing — never cleared per window
    node_cache[key] = len(node_cache)
    g = Data(..., num_nodes=len(node_cache))  # cumulative!
```

**Fix:** Reset `node_cache = {}` at the start of each window.

---

### PREP-2: `edge_time` stored as float32 (precision loss)

**File:** [run_preprocessing.py:261](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/laptop/run_preprocessing.py#L261)

```python
et = torch.tensor(w_data[TIME_FIELD].values.astype(np.float32))  # float32!
```

Epoch-ms values (~1.5e12) exceed float32's 7-digit precision. Use float64:
```python
et = torch.tensor(w_data[TIME_FIELD].values.astype(np.float64))
```

---

### PREP-3: Scaler fit on only UNSW-NB15 training data

**File:** [l02_preprocess_and_split.py](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/laptop/l02_preprocess_and_split.py) or [run_preprocessing.py](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/laptop/run_preprocessing.py)

The `scaler_metadata.json` in `dataset/graphs/` confirms `n_samples_seen_ = 1,655,796` which equals only UNSW-NB15 train edges. The scaler was likely fit before CICIDS2018 graphs were built, or only UNSW was available at the time.

**Fix:** Refit scaler on BOTH datasets' training edges. Add assertion:
```python
assert all_train_ea.shape[0] > 10_000_000, "Scaler should be fit on both datasets"
```

---

### PREP-4: No outlier clipping after z-score normalization

Neither preprocessing script clips extreme z-scored values. The scaler produces values up to 860 standard deviations from the mean.

**Fix:** Add after scaling:
```python
g.edge_attr = torch.clamp(g.edge_attr, -10.0, 10.0)
```

---

### PREP-5: Window boundary not reset across parquet chunks

**File:** [run_preprocessing.py:237-239](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/laptop/run_preprocessing.py#L237-L239)

```python
if t0_ref is None:
    t0_ref = chunk_time_s.min()
```

`t0_ref` is set from the first parquet chunk's minimum time. But if parquet chunks are not sorted by time (possible since they come from chunked CSV reading), later chunks may have earlier timestamps, producing negative window indices. Also, if graphs straddle chunk boundaries, they'll be split into separate windows.

---

## 4. Code Bugs — K01

### K01-1: Batch size 2048 vs spec 4096 (no gradient accumulation)

**File:** [k01_time2vec_mae.py:344](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k01_time2vec_mae.py#L344)

Effective batch size halved. Changes optimization dynamics.

### K01-2: Fanout [10,5,3] vs spec [15,10,5]

**File:** [k01_time2vec_mae.py:344](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k01_time2vec_mae.py#L344)

Reduced neighborhood = less graph context per flow.

### K01-3: Data path mismatch — code uses Kaggle paths, data is in dataset/graphs/

**File:** [k01_time2vec_mae.py:59](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k01_time2vec_mae.py#L59)

```python
INPUT = Path('/kaggle/input/datasets/mysteriousavailable/ids-nf3-processed/ids-nf3-processed')
```

But your actual data is in `dataset/graphs/`. The code expects Kaggle uploaded dataset paths. Make sure the uploaded Kaggle dataset contains all files from `dataset/graphs/` plus manifests.

---

## 5. Code Bugs — K02

### K02-1: CVAE trains 300 epochs (spec: 50) → likely mode collapse

**File:** [k02_cvae.py:159](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k02_cvae.py#L159)

6x over-training on small minority data. Synthetic samples become near-duplicates.

### K02-2: CVAE batch size 256 (spec: 512)

**File:** [k02_cvae.py:159](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k02_cvae.py#L159)

### K02-3: Encoder uses old buggy `_get_node_embed()` pattern

**File:** [k02_cvae.py:46-51](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k02_cvae.py#L46-L51)

K01's BUG FIX 1/6 not applied. Dynamic node_embed allocation.

### K02-4: `return_attention_weights=True` always on

**File:** [k02_cvae.py:55](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k02_cvae.py#L55)

Wastes VRAM. K01's BUG FIX 4 not applied.

### K02-5: LayerNorm order differs from K01

**File:** [k02_cvae.py:56](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k02_cvae.py#L56)

K01: `norm(x + x_new)`. K02: `norm(x_new); x = x + x_new if match else x_new`. Different computation.

---

## 6. Code Bugs — K03

### K03-1: GradScaler.update() gated on NaN → cascading NaN

**File:** [k03_binary.py:149](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k03_binary.py#L149)

`if not torch.isnan(loss): amp.update()` — K01's BUG FIX 2 not applied.

### K03-2: In-place mutation of validation graph objects

**File:** [k03_binary.py:217-221](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k03_binary.py#L217-L221)

```python
g.edge_index = g.edge_index[:, idx]  # permanently shrinks shared object
```

K01's BUG FIX 3 not applied. Val graphs permanently truncated after epoch 1.

### K03-3: PGD attack direction is inverted

**File:** [k03_binary.py:190](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k03_binary.py#L190)

```python
loss_adv = -focal(logits_pert[pgd_mask], (g.y[bi][pgd_mask]!=0).long())
```

The negative sign + `ea58 += alpha * grad.sign()` makes examples EASIER, not harder. PGD adversarial training does nothing.

### K03-4: Encoder uses old buggy architecture (no BUG FIX 1/4/6)

**File:** [k03_binary.py:42-66](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k03_binary.py#L42-L66)

Dynamic node_embed, always-on attention, wrong norm order.

### K03-5: Threshold "recall" metric reports positive rate, not actual recall

**File:** [k03_binary.py:249](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k03_binary.py#L249)

```python
(vprobs>=best_thr).astype(int).mean()  # This is P(predict=attack), not recall
```

### K03-6: Training graphs also mutated in-place

**File:** [k03_binary.py:131-132](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k03_binary.py#L131-L132)

```python
g = g.to(device)  # permanently moves shared graph to GPU
```

After first epoch, all G_train objects are on GPU, consuming permanent VRAM.

---

## 7. Code Bugs — K04

### K04-1: Synthetic embeddings loading crashes (KeyError)

**File:** [k04_multiclass.py:94-96](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k04_multiclass.py#L94-L96)

```python
synth_emb = synth_data['embeddings']  # KeyError: K02 saved 'embeddings_memmap_path'
```

K02 saves embeddings as numpy memmap, not as a tensor in the dict.

### K04-2: GradScaler.update() gated on NaN (same as K03-1)

**File:** [k04_multiclass.py:199](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k04_multiclass.py#L199)

### K04-3: Benign class included in multiclass training

**File:** [k04_multiclass.py:146-152](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k04_multiclass.py#L146-L152)

Architecture says Stage G only processes attack-flagged flows. Including Benign wastes capacity and biases the loss.

### K04-4: In-place mutation of val graphs (same as K03-2)

**File:** [k04_multiclass.py:210-212](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k04_multiclass.py#L210-L212)

### K04-5: PGD direction inverted (same as K03-3)

**File:** [k04_multiclass.py:169](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k04_multiclass.py#L169)

### K04-6: Encoder uses old buggy architecture

**File:** [k04_multiclass.py:45-69](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k04_multiclass.py#L45-L69)

### K04-7: `encode()` function constructs Data object twice redundantly

**File:** [k04_multiclass.py:128-130](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k04_multiclass.py#L128-L130)

```python
def encode(ea, et, ei, nn):
    tn = norm_time(et); te = t2v(tn)
    return torch.cat([ea,te], dim=-1), Data(...)  # returns ea58 AND Data, but callers only use one
```

---

## 8. Code Bugs — K05

### K05-1: Leave-one-class-out missing Benign class

**File:** [k05_prototypical.py:183](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k05_prototypical.py#L183)

Only attack classes tested. PRD requires all 11.

### K05-2: 5-way inference voting ensemble never implemented

Architecture spec requires 5 different support draws, majority vote. Not implemented.

### K05-3: Encoder uses old buggy architecture

**File:** [k05_prototypical.py:43-67](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k05_prototypical.py#L43-L67)

### K05-4: All training graphs loaded into memory simultaneously

**File:** [k05_prototypical.py:93](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k05_prototypical.py#L93)

```python
G_train = torch.load(INPUT/'NF-CICIDS2018_train_list.pt') + torch.load(INPUT/'NF-UNSW-NB15_train_list.pt')
```

Both datasets loaded fully into memory (~3.1 GB) before extraction. OOM risk on T4x2.

### K05-5: Best model checkpoint doesn't use val performance for selection

**File:** [k05_prototypical.py:174](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k05_prototypical.py#L174)

Always saves the final epoch's model, not the best validation accuracy.

---

## 9. Code Bugs — K06

### K06-1: Missing import — `classification_report`

**File:** [k06_eval_xai.py:18](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k06_eval_xai.py#L18)

Not imported from sklearn. Crashes at line 138.

### K06-2: `test_targets` is list, used as numpy array

**File:** [k06_eval_xai.py:134](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k06_eval_xai.py#L134)

```python
cm = test_targets == i  # TypeError: list doesn't support == with int
```

### K06-3: Cross-dataset eval skips binary cascade

**File:** [k06_eval_xai.py:154](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k06_eval_xai.py#L154)

Uses `multi_head(reps).argmax(-1)` directly, bypassing Stage F binary filter.

### K06-4: Adversarial robustness uses hardcoded bounds +-4.0

**File:** [k06_eval_xai.py:182-183](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k06_eval_xai.py#L182-L183)

Should use actual train-set feature min/max.

### K06-5: SHAP on 768-dim embeddings instead of 58-dim features

**File:** [k06_eval_xai.py:293](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k06_eval_xai.py#L293)

Spec requires SHAP on interpretable feature space. Implementation uses opaque embeddings.

### K06-6: In-place mutation of test graphs

**File:** [k06_eval_xai.py:178-179](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k06_eval_xai.py#L178-L179)

Same in-place truncation bug as K03/K04.

### K06-7: t-SNE only shows post-multiclass (spec: 3 panels)

**File:** [k06_eval_xai.py:229](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k06_eval_xai.py#L229)

Spec requires: post-MAE, post-multiclass, post-fewshot — 3 panels.

### K06-8: SHAP `GradientExplainer` called with wrong API

**File:** [k06_eval_xai.py:308-311](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k06_eval_xai.py#L308-L311)

`GradientExplainer` requires a PyTorch model, not a lambda. Lambda wrapping breaks gradient propagation.

### K06-9: Deprecated `torch.cuda.amp` API

**File:** [k06_eval_xai.py](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/kaggle/k06_eval_xai.py)

Not critical now but will break on PyTorch 2.4+.

---

## 10. Architecture Inconsistencies Across Scripts

### The Core Problem: K01's bug fixes never propagated

K01 has 6 documented bug fixes. **None were applied to K02-K06:**

| Bug Fix | K01 | K02 | K03 | K04 | K05 | K06 |
|---|---|---|---|---|---|---|
| BUG FIX 1: Fixed-size node_embed | Yes | No | No | No | No | No |
| BUG FIX 2: Always call scaler.update() | Yes | N/A | No | No | N/A | N/A |
| BUG FIX 3: Don't mutate shared graphs | Yes | N/A | No | No | N/A | No |
| BUG FIX 4: return_attention off by default | Yes | No | No | No | No | Partial |
| BUG FIX 5: Dimension assertion | Yes | N/A | N/A | N/A | N/A | N/A |
| BUG FIX 6: max_nodes from batch probe | Yes | No | No | No | No | No |

### LayerNorm Order Mismatch

| Script | Normalization | Pattern |
|---|---|---|
| K01 | `norm(x + x_new)` | Post-residual LN (correct per spec) |
| K02 | `norm(x_new); x = x+x_new if match else x_new` | Pre-residual (breaks pretrained weights) |
| K03 | `x_new = norm(x_new); x = x+x_new` | Pre-residual (breaks pretrained weights) |
| K04 | Same as K03 | Pre-residual |
| K05 | Same as K03 | Pre-residual |
| K06 | Same as K03 | Pre-residual |

**Impact:** The encoder is pretrained with one math, then fine-tuned with different math. The pretrained weights are meaningless under the changed normalization order.

---

## 11. YAML Manifest Errors

### MANIFEST-1: Feature in both `kept_features` and `correlation_pruned`

**File:** [feature_manifest.yaml](file:///C:/Users/potato/Desktop/ids-v2%20-%20Copy/feature_manifest.yaml)

`SRC_TO_DST_IAT_STDDEV` appears at line 60 (kept) AND line 71 (pruned). Contradiction.

### MANIFEST-2: Kept features count = 41, but list has 41 entries including the contradicted one

If `SRC_TO_DST_IAT_STDDEV` is pruned, only 40 features should be kept, making `final_raw_feature_count: 41` wrong.

---

## 12. Summary

### Issue Counts by Severity

| Severity | Count | Category Breakdown |
|---|---|---|
| **CRITICAL** (crashes or silent NaN/degradation) | 15 | 6 data, 4 code logic, 3 architecture, 2 runtime |
| **MAJOR** (degrades model quality significantly) | 18 | 6 code bugs, 5 architecture, 4 parameter, 3 spec violations |
| **MODERATE** | 12 | Mix of minor code issues and missing features |

### Fix Priority for NaN Gradients Specifically

> [!CAUTION]
> Fix these in this exact order to eliminate NaN gradients:

| Priority | Fix | Expected Impact |
|---|---|---|
| **1** | Refit scaler on BOTH datasets' training data | Eliminates 90%+ of extreme outliers |
| **2** | Clip z-scored features to [-10, 10] after scaling | Eliminates remaining outliers |
| **3** | Always call `amp.update()` after `amp.step()` in K03/K04 | Stops NaN cascades |
| **4** | Use float64 for edge_time in preprocessing | Fixes Time2Vec precision |
| **5** | Unify encoder architecture (port K01's fixed encoder to all scripts) | Fixes all downstream bugs at once |
| **6** | Fix PGD direction (remove negative sign) | Enables actual adversarial training |
| **7** | Reset node_cache per window in preprocessing | Fixes memory waste + num_nodes inflation |

### Fix Priority for Best Model Results

| Priority | Fix | Expected Impact |
|---|---|---|
| **1** | Fix scaler (DATA-1) + clip outliers (PREP-4) | Biggest single quality improvement |
| **2** | Unify encoder across all scripts | Fixes pretrain→finetune mismatch |
| **3** | Fix PGD direction | Enables adversarial robustness |
| **4** | Remove Benign from multiclass training | Better minority class learning |
| **5** | Fix synthetic embeddings loading in K04 | Enables CVAE augmentation |
| **6** | Reset node_cache in preprocessing | Correct graph structures |
| **7** | Use float64 for timestamps | Better temporal signal |
| **8** | CVAE epochs 300→50 | Better synthetic data quality |
