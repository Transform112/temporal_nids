# Kaggle Execution Guide

## How to run on Kaggle

### 1. Upload dataset

Upload the `datasets/processed/` directory as a Kaggle dataset named `nids-data`:

```
datasets/processed/
├── cic17_common.parquet
├── cic17_native.parquet
├── unsw_common.parquet
├── unsw_native.parquet
```

Or upload the full `datasets/` directory including `splits/` for Phase 5+ work.

### 2. Upload package

Upload the `nids/` directory as a Kaggle dataset named `nids-package`, or include it directly in the notebook's working directory.

### 3. Notebook setup

```python
# First cell — install deps (PyG needs special handling)
!pip install torch_geometric

# Second cell — verify everything imports
import sys
sys.path.insert(0, '/kaggle/working')
from nids import SEED, set_seed
set_seed()
print(f"Seed: {SEED}, CUDA: {torch.cuda.is_available()}")
```

### 4. Run phases

Each phase is a standalone script:
- `kaggle/training/phase06_microbatch_ablation.py` — Phase 6
- Future: `phase07_architecture.py`, `phase08_ssl_training.py`, etc.

Run from notebook:
```python
%run kaggle/training/phase06_microbatch_ablation.py
```

Or import and call main:
```python
from kaggle.training.phase06_microbatch_ablation import main
main()
```

## Compute split

| Phase | Where | Why |
|---|---|---|
| 1-5 | Laptop **DONE** | Data prep, EDA, dev tests — no GPU needed |
| 6-14 | **Kaggle T4x2** | TGN training, SSL, adversarial, XAI — need GPU |
| 15-16 | Laptop | Data integration + paper writing |

## Current state

Phases 0-5 complete. Phase 6 is ready to run on Kaggle.
All code accepts `--sample N` for fast dev testing.

## Quick test

```bash
# Local laptop dev test (Phase 6)
python kaggle/training/phase06_microbatch_ablation.py --target laptop --max-flows 5000
```
