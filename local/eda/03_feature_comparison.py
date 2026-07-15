#!/usr/bin/env python3
"""
Cross-Dataset Feature Comparison & Harmonization Map
Target: LAPTOP
Produces: definitive feature_provenance.json, drift charts, protocol encoding map
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

from nids import set_seed, SEED

set_seed()

# ── Config ────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(os.path.join(os.path.dirname(__file__), "..", "eda_output"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
PROJECT_ROOT = Path(os.path.join(os.path.dirname(__file__), "..", ".."))

plt.rcParams.update({"figure.dpi": 150, "figure.figsize": (14, 8), "font.size": 9})


def load_cic17_sample(sample: int | None = None) -> pd.DataFrame:
    """Load CIC-IDS2017 (with headers, 85 cols)."""
    cic_dir = PROJECT_ROOT / "datasets" / "CICIDS2017"
    dfs = []
    for fpath in sorted(cic_dir.glob("*.csv")):
        df = pd.read_csv(fpath, engine="python", encoding="latin-1", on_bad_lines="skip")
        df.columns = [c.strip() for c in df.columns]  # Fix leading spaces in column names
        label_col = "Label" if "Label" in df.columns else None
        if label_col:
            df = df[df[label_col] != "Label"].copy()
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.strip()
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    if sample and len(combined) > sample:
        combined = combined.sample(n=sample, random_state=SEED)
    return combined


def load_unsw_sample(sample: int | None = None) -> pd.DataFrame:
    """Load UNSW-NB15 raw CSVs (no header)."""
    feat_file = PROJECT_ROOT / "datasets" / "UNSWNB15" / "NUSW-NB15_features.csv"
    feat_df = pd.read_csv(feat_file, encoding="latin-1")
    col_names = feat_df["Name"].str.strip().tolist()

    unsw_dir = PROJECT_ROOT / "datasets" / "UNSWNB15"
    dfs = []
    for i in range(1, 5):
        fpath = unsw_dir / f"UNSW-NB15_{i}.csv"
        df = pd.read_csv(fpath, header=None, names=col_names, engine="python",
                         encoding="latin-1", on_bad_lines="skip")
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    for col in combined.columns:
        if combined[col].dtype == object:
            combined[col] = combined[col].str.strip()
    if sample and len(combined) > sample:
        combined = combined.sample(n=sample, random_state=SEED)
    return combined


# ── C1: Feature Overlap Matrix ────────────────────────────────────────────
def analysis_c1(cic17: pd.DataFrame, unsw: pd.DataFrame) -> dict:
    """Map Plan §3.5 common features to actual columns in each dataset."""
    print("\n" + "=" * 70)
    print("C1: FEATURE OVERLAP MATRIX")
    print("=" * 70)

    cic17_cols = set(cic17.columns.str.strip().str.lower())
    unsw_cols = set(unsw.columns.str.strip().str.lower())

    # The harmonization map from Plan §3.5 -- adapted for verified column names
    # Format: {common_name: {cic17: actual_col, unsw: actual_col, transform: str, notes: str}}
    harmonization = {
        "duration": {
            "cic17": "Flow Duration",
            "unsw": "dur",
            "transform": "cic17_us_to_sec",  # CIC17 µs -> seconds
            "notes": "CIC17 Flow Duration is µs, UNSW dur is seconds"
        },
        "protocol": {
            "cic17": "Protocol",
            "unsw": "proto",
            "transform": "shared_categorical",
            "notes": "CIC17 uses IANA numbers, UNSW uses text labels (tcp/udp/...) -- map to shared encoding"
        },
        "fwd_packets": {
            "cic17": "Total Fwd Packets",
            "unsw": "spkts",
            "transform": "passthrough",
            "notes": "Source->dest direction packets"
        },
        "bwd_packets": {
            "cic17": "Total Backward Packets",
            "unsw": "dpkts",
            "transform": "passthrough",
            "notes": "Dest->source direction packets"
        },
        "fwd_bytes": {
            "cic17": "Total Length of Fwd Packets",
            "unsw": "sbytes",
            "transform": "passthrough",
            "notes": "Source->dest bytes"
        },
        "bwd_bytes": {
            "cic17": "Total Length of Bwd Packets",
            "unsw": "dbytes",
            "transform": "passthrough",
            "notes": "Dest->source bytes"
        },
        "byte_rate": {
            "cic17": "Flow Bytes/s",
            "unsw": None,  # Need to derive or use (Sload+Dload)/2
            "transform": "cic17_inf_to_zero; unsw_derive_sload_dload_avg",
            "notes": "CIC17 Flow Bytes/s has Inf (->0). UNSW: use (Sload+Dload)/2 as byte rate proxy"
        },
        "packet_rate": {
            "cic17": "Flow Packets/s",
            "unsw": None,  # 'rate' only in train/test partition, not in raw 49-col
            "transform": "cic17_inf_to_zero; unsw_derive_spkts_div_dur",
            "notes": "UNSW raw has no 'rate' col (train/test only). Derive as Spkts/dur for raw."
        },
        "mean_iat": {
            "cic17": "Flow IAT Mean",
            "unsw": None,  # Approximate: (Sintpkt + Dintpkt) / 2
            "transform": "unsw_avg_sintpkt_dintpkt",
            "notes": "UNSW splits IAT by direction; average the two to match CIC17's combined mean"
        },
        "std_iat": {
            "cic17": "Flow IAT Std",
            "unsw": None,  # Approximate: (Sjit + Djit) / 2
            "transform": "unsw_avg_sjit_djit",
            "notes": "Jitter ≠ std(IAT); flagged as approximation in paper"
        },
        "mean_pkt_len": {
            "cic17": "Average Packet Size",
            "unsw": None,  # Approximate: (smeansz + dmeansz) / 2
            "transform": "unsw_avg_smeansz_dmeansz",
            "notes": "Average of source and destination mean packet sizes"
        },
        "syn_count": {
            "cic17": "SYN Flag Count",
            "unsw": None,
            "transform": "impute_zero_and_flag",
            "notes": "SYN flag count not available in UNSW-NB15 -- impute 0 + is_imputed flag"
        },
        "ack_count": {
            "cic17": "ACK Flag Count",
            "unsw": None,
            "transform": "impute_zero_and_flag",
            "notes": "ACK flag count not available in UNSW-NB15 -- impute 0 + is_imputed flag"
        },
        "init_win_fwd": {
            "cic17": "Init_Win_bytes_forward",
            "unsw": "swin",
            "transform": "passthrough",
            "notes": "TCP window advertisement -- source direction"
        },
        "init_win_bwd": {
            "cic17": "Init_Win_bytes_backward",
            "unsw": "dwin",
            "transform": "passthrough",
            "notes": "TCP window advertisement -- destination direction"
        },
        "down_up_ratio": {
            "cic17": "Down/Up Ratio",
            "unsw": None,  # Derived: dbytes / sbytes
            "transform": "derive_dbytes_div_sbytes; guard_div_zero",
            "notes": "Guard divide-by-zero: if sbytes=0 -> 0.0"
        },
        "state_summary": {
            "cic17": None,  # Derived from flag counts
            "unsw": "state",
            "transform": "onehot_shared_buckets",
            "notes": "UNSW state field is closest analogue. One-hot with 'unknown' bucket for unseen values."
        },
    }

    # Verify each column exists in the actual data
    cic17_lower_map = {c.strip().lower(): c for c in cic17.columns}
    unsw_lower_map = {c.strip().lower(): c for c in unsw.columns}

    provenance = []
    for common_name, mapping in harmonization.items():
        entry = {
            "common_name": common_name,
            "cic17_source": None,
            "unsw_source": None,
            "transform": mapping["transform"],
            "notes": mapping["notes"],
            "is_imputed": "impute" in mapping["transform"],
            "verified": True,
            "mismatch": None,
        }

        # Verify CIC17 column
        cic17_target = mapping["cic17"]
        if cic17_target:
            cic17_lower = cic17_target.strip().lower()
            if cic17_lower in cic17_lower_map:
                entry["cic17_source"] = cic17_lower_map[cic17_lower]
            else:
                entry["verified"] = False
                entry["mismatch"] = f"CIC17 column '{cic17_target}' NOT FOUND in data"
                print(f"  MISMATCH: {common_name} -> CIC17 '{cic17_target}' not found!")
        else:
            entry["cic17_source"] = "DERIVED"

        # Verify UNSW column
        unsw_target = mapping["unsw"]
        if unsw_target:
            unsw_lower = unsw_target.strip().lower()
            if unsw_lower in unsw_lower_map:
                entry["unsw_source"] = unsw_lower_map[unsw_lower]
            else:
                entry["verified"] = False
                entry["mismatch"] = f"UNSW column '{unsw_target}' NOT FOUND in data"
                print(f"  MISMATCH: {common_name} -> UNSW '{unsw_target}' not found!")
        else:
            entry["unsw_source"] = "DERIVED"

        if entry["verified"]:
            print(f"  [OK] {common_name:25s}  CIC17: {str(entry['cic17_source']):30s}  UNSW: {str(entry['unsw_source']):25s}  [{entry['transform']}]")
        provenance.append(entry)

    # Save v1
    with open(PROJECT_ROOT / "feature_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"\n  -> Saved: feature_provenance.json ({len(provenance)} features)")

    return provenance


# ── C2: Distribution Shift ────────────────────────────────────────────────
def analysis_c2(cic17: pd.DataFrame, unsw: pd.DataFrame, provenance: list):
    print("\n" + "=" * 70)
    print("C2: DISTRIBUTION SHIFT -- CIC17 vs UNSW (Common Features)")
    print("=" * 70)

    # Select features that have direct (non-derived) columns in both datasets
    pairable = [p for p in provenance
                if p["cic17_source"] not in [None, "DERIVED"]
                and p["unsw_source"] not in [None, "DERIVED"]]

    if len(pairable) < 4:
        print("  Not enough directly-mapped features for drift analysis -- need derived features first")
        return

    n_rows = min(4, (len(pairable) + 3) // 4)
    fig, axes = plt.subplots(n_rows, 4, figsize=(24, n_rows * 5))
    axes = axes.flatten()

    for i, p in enumerate(pairable):
        if i >= len(axes):
            break
        ax = axes[i]

        cic_vals = pd.to_numeric(cic17[p["cic17_source"]], errors="coerce").dropna()
        unsw_vals = pd.to_numeric(unsw[p["unsw_source"]], errors="coerce").dropna()

        if len(cic_vals) == 0 or len(unsw_vals) == 0:
            ax.text(0.5, 0.5, f"{p['common_name']}\n(no numeric data)", transform=ax.transAxes, ha="center")
            continue

        # Clip
        combined = pd.concat([cic_vals, unsw_vals])
        lo, hi = combined.quantile(0.01), combined.quantile(0.99)

        sns.kdeplot(cic_vals.clip(lo, hi).sample(min(5000, len(cic_vals)), random_state=SEED),
                    ax=ax, label="CIC-IDS2017", color="#1a237e", fill=True, alpha=0.3)
        sns.kdeplot(unsw_vals.clip(lo, hi).sample(min(5000, len(unsw_vals)), random_state=SEED),
                    ax=ax, label="UNSW-NB15", color="#c62828", fill=True, alpha=0.3)
        ax.set_title(p["common_name"][:50], fontsize=9)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Cross-Dataset Feature Distribution Shift (CIC-IDS2017 vs UNSW-NB15)", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "cross_dataset_drift.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: cross_dataset_drift.png ({min(i+1, len(pairable))} features plotted)")


# ── C3: Protocol Encoding Map ─────────────────────────────────────────────
def analysis_c3(cic17: pd.DataFrame, unsw: pd.DataFrame):
    print("\n" + "=" * 70)
    print("C3: PROTOCOL ENCODING MAP")
    print("=" * 70)

    cic_protos = cic17["Protocol"].dropna().astype(str).str.strip().unique()
    unsw_protos = unsw["proto"].dropna().astype(str).str.strip().unique()

    print(f"  CIC17 unique protocol values: {len(cic_protos)}")
    print(f"    Sample: {sorted(cic_protos)[:15]}")
    print(f"  UNSW unique protocol values: {len(unsw_protos)}")
    print(f"    Values: {sorted(unsw_protos)}")

    # CIC17 has IANA protocol numbers as strings (e.g., "6" = TCP, "17" = UDP)
    # UNSW has text labels (e.g., "tcp", "udp")
    # Build shared mapping
    IANA_TO_NAME = {
        "0": "hopopt", "1": "icmp", "2": "igmp", "6": "tcp", "17": "udp",
        "41": "ipv6", "47": "gre", "50": "esp", "51": "ah", "58": "ipv6-icmp",
        "89": "ospf", "132": "sctp",
    }

    all_protos = set()
    for v in cic_protos:
        name = IANA_TO_NAME.get(v, f"iana_{v}")
        all_protos.add(name.lower())
    for v in unsw_protos:
        all_protos.add(v.lower())

    all_protos.add("unknown")
    proto_list = sorted(all_protos)
    proto_map = {p: i for i, p in enumerate(proto_list)}

    print(f"\n  Shared protocol encoding space: {len(proto_list)} values")
    for p, idx in proto_map.items():
        print(f"    {idx:3d}: {p}")

    # Save
    with open(OUTPUT_DIR / "protocol_encoding_map.json", "w") as f:
        json.dump({"proto_to_idx": proto_map, "idx_to_proto": proto_list, "iana_mapping": IANA_TO_NAME}, f, indent=2)
    print(f"  -> Saved: protocol_encoding_map.json")

    return proto_map


# ── C4: Duration Scale Comparison ──────────────────────────────────────────
def analysis_c4(cic17: pd.DataFrame, unsw: pd.DataFrame):
    print("\n" + "=" * 70)
    print("C4: DURATION SCALE COMPARISON")
    print("=" * 70)

    cic_dur = pd.to_numeric(cic17["Flow Duration"], errors="coerce").dropna()
    # CIC17 is microseconds -> convert to seconds
    cic_dur_sec = cic_dur.clip(lower=0) / 1_000_000

    unsw_dur = pd.to_numeric(unsw["dur"], errors="coerce").dropna()
    # UNSW dur is already in seconds (per features file: "Record total duration" as Float)

    print(f"  CIC17 Flow Duration (converted to seconds):")
    print(f"    Min={cic_dur_sec.min():.6f}, Median={cic_dur_sec.median():.4f}, "
          f"Mean={cic_dur_sec.mean():.4f}, Max={cic_dur_sec.max():.2f}")
    print(f"    Zero-duration flows: {(cic_dur_sec == 0).sum():,} ({(cic_dur_sec == 0).mean()*100:.1f}%)")

    print(f"  UNSW dur (seconds):")
    print(f"    Min={unsw_dur.min():.6f}, Median={unsw_dur.median():.4f}, "
          f"Mean={unsw_dur.mean():.4f}, Max={unsw_dur.max():.2f}")
    print(f"    Zero-duration flows: {(unsw_dur == 0).sum():,} ({(unsw_dur == 0).mean()*100:.1f}%)")

    # Box plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Log scale for visibility
    cic_log = np.log10(cic_dur_sec.clip(lower=1e-9))
    unsw_log = np.log10(unsw_dur.clip(lower=1e-9))

    bp = axes[0].boxplot([cic_log, unsw_log])
    axes[0].set_xticklabels(["CIC-IDS2017 (us->sec)", "UNSW-NB15 (sec)"])
    axes[0].set_title("Duration Distribution -- log₁₀(seconds)")
    axes[0].set_ylabel("log₁₀(Duration in seconds)")

    axes[1].hist(cic_log, bins=100, alpha=0.5, label="CIC-IDS2017", color="#1a237e")
    axes[1].hist(unsw_log, bins=100, alpha=0.5, label="UNSW-NB15", color="#c62828")
    axes[1].set_title("Duration Histogram -- log₁₀(seconds)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "duration_scale_compare.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: duration_scale_compare.png")


# ── C5: Feature Completeness Report ───────────────────────────────────────
def analysis_c5(provenance: list):
    print("\n" + "=" * 70)
    print("C5: FEATURE COMPLETENESS REPORT")
    print("=" * 70)

    direct = [p for p in provenance if p["cic17_source"] not in [None, "DERIVED"]
              and p["unsw_source"] not in [None, "DERIVED"]]
    derived_cic = [p for p in provenance if p["cic17_source"] == "DERIVED"]
    derived_unsw = [p for p in provenance if p["unsw_source"] == "DERIVED"]
    imputed = [p for p in provenance if p["is_imputed"]]
    unverified = [p for p in provenance if not p["verified"]]

    print(f"  Total common features:        {len(provenance)}")
    print(f"  Directly mapped (both sides): {len(direct)}")
    print(f"  CIC17-derived only:           {len(derived_cic)}")
    print(f"  UNSW-derived only:            {len(derived_unsw)}")
    print(f"  Imputed features:             {len(imputed)}")
    print(f"  UNVERIFIED (column mismatch): {len(unverified)}")

    if imputed:
        print(f"\n  Imputed features (flagged in provenance):")
        for p in imputed:
            print(f"    {p['common_name']}: {p['notes']}")

    if unverified:
        print(f"\n  ⚠ UNVERIFIED FEATURES -- fix before Phase 2:")
        for p in unverified:
            print(f"    {p['common_name']}: {p['mismatch']}")

    # Print mapping completeness table
    print(f"\n  Full harmonization map:")
    print(f"  {'Common Feature':<25s} {'CIC17 Source':<35s} {'UNSW Source':<30s} {'Transform':<30s}")
    print(f"  {'-'*25} {'-'*35} {'-'*30} {'-'*30}")
    for p in provenance:
        cic_src = str(p['cic17_source'] or '--')[:34]
        unsw_src = str(p['unsw_source'] or '--')[:29]
        xform = p['transform'][:29]
        print(f"  {p['common_name']:<25s} {cic_src:<35s} {unsw_src:<30s} {xform:<30s}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Feature Comparison")
    parser.add_argument("--sample", type=int, default=100000, help="Subsample N rows per dataset (default 100K)")
    parser.add_argument("--target", type=str, default="laptop", choices=["laptop", "kaggle"])
    args = parser.parse_args()

    if args.target != "laptop":
        raise RuntimeError("EDA scripts must run on laptop.")

    print("=" * 70)
    print("CROSS-DATASET FEATURE COMPARISON")
    print(f"Sample per dataset: {args.sample:,}")
    print("=" * 70)

    print("\nLoading CIC-IDS2017...")
    cic17 = load_cic17_sample(sample=args.sample)
    print(f"  -> {len(cic17):,} rows × {len(cic17.columns)} cols")

    print("\nLoading UNSW-NB15...")
    unsw = load_unsw_sample(sample=args.sample)
    print(f"  -> {len(unsw):,} rows × {len(unsw.columns)} cols")

    # Run analyses
    provenance = analysis_c1(cic17, unsw)
    analysis_c2(cic17, unsw, provenance)
    proto_map = analysis_c3(cic17, unsw)
    analysis_c4(cic17, unsw)
    analysis_c5(provenance)

    print("\n" + "=" * 70)
    print("CROSS-DATASET COMPARISON COMPLETE")
    print(f"feature_provenance.json: {PROJECT_ROOT / 'feature_provenance.json'}")
    print(f"All charts: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
