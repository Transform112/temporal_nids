"""Data loading, preprocessing, graph construction, splitting.

Exports:
    Preprocessing (Phase 2):
        load_cic17, load_unsw — load & clean raw CSVs
        build_common_track_cic17, build_common_track_unsw — feature harmonization
        derive_unsw_features — compute UNSW features missing from native columns

    Splitting (Phase 5):
        chronological_split, stratified_split — 70/15/15 with train=benign only
        fit_scaler, apply_scaler — per-dataset StandardScaler

    Final Dataset (Phase 7):
        build_dataset — merge native+common, scale, split, save as ready-to-use parquet
"""
from nids.data.preprocessing import (
    load_cic17,
    load_unsw,
    deduplicate,
    apply_cic17_conversions,
    apply_unsw_conversions,
    derive_unsw_features,
    build_common_track_cic17,
    build_common_track_unsw,
    load_provenance,
)

from nids.data.splitting import (
    chronological_split,
    stratified_split,
    fit_scaler,
    apply_scaler,
)

from nids.data.final_dataset import build_dataset
