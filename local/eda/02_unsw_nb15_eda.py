#!/usr/bin/env python3
"""
UNSW-NB15 Exploratory Data Analysis
Target: LAPTOP
Key fact: Raw CSVs have NO HEADER ROW -- column names come from NUSW-NB15_features.csv
Produces: 12 analyses, 8 charts, column inventory
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
from pathlib import Path

from nids import set_seed, SEED

set_seed()

# ── Config ────────────────────────────────────────────────────────────────
DATASET_DIR = Path(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "UNSWNB15"))
OUTPUT_DIR = Path(os.path.join(os.path.dirname(__file__), "..", "eda_output"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

RAW_FILES = [DATASET_DIR / f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
FEATURES_FILE = DATASET_DIR / "NUSW-NB15_features.csv"
TRAIN_FILE = DATASET_DIR / "Training and Testing Sets" / "UNSW_NB15_training-set.csv"
TEST_FILE = DATASET_DIR / "Training and Testing Sets" / "UNSW_NB15_testing-set.csv"

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (14, 8),
    "font.size": 9,
})


def load_features() -> list[str]:
    """Parse NUSW-NB15_features.csv to get column names for raw CSVs."""
    feat_df = pd.read_csv(FEATURES_FILE, encoding="latin-1")
    # Columns in features file: No., Name, Type, Description
    col_names = feat_df["Name"].str.strip().tolist()
    print(f"  Features file columns: {len(col_names)}")
    return col_names


def load_raw_data(sample: int | None = None) -> pd.DataFrame:
    """Load all 4 raw CSVs (no header), assign column names."""
    col_names = load_features()

    dfs = []
    for fpath in RAW_FILES:
        print(f"  Loading {fpath.name} ...")
        df = pd.read_csv(fpath, header=None, names=col_names, engine="python",
                         encoding="latin-1", on_bad_lines="skip")
        dfs.append(df)
        print(f"    -> {len(df):,} rows")

    combined = pd.concat(dfs, ignore_index=True)
    # Strip whitespace from string columns (attack_cat has leading spaces)
    for col in combined.columns:
        if combined[col].dtype == object:
            combined[col] = combined[col].str.strip()
    print(f"  Combined: {len(combined):,} rows, {len(combined.columns)} columns")

    if sample and len(combined) > sample:
        combined = combined.sample(n=sample, random_state=SEED)

    return combined


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load official train/test partition (for comparison only -- we don't trust it)."""
    train = pd.read_csv(TRAIN_FILE, encoding="latin-1")
    test = pd.read_csv(TEST_FILE, encoding="latin-1")
    # Strip BOM from first column name if present
    train.columns = train.columns.str.strip().str.lstrip("﻿")
    test.columns = test.columns.str.strip().str.lstrip("﻿")
    return train, test


# ── B1: Column Inventory ──────────────────────────────────────────────────
def analysis_b1(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("B1: COLUMN INVENTORY (Raw CSV, 49 cols)")
    print("=" * 70)

    inventory = pd.DataFrame({
        "column": df.columns.tolist(),
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_pct": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
        "n_unique": [df[c].nunique() for c in df.columns],
    })
    print(f"  Total columns: {len(inventory)}")
    numeric = inventory["dtype"].str.contains("float|int")
    print(f"  Numeric: {numeric.sum()}")
    print(f"  Object/categorical: {(~numeric).sum()}")
    print(f"  Columns with >0% missing: {(inventory['missing_pct'] > 0).sum()}")
    print(f"\n  Full inventory:\n{inventory.to_string(index=False)}")

    inventory.to_csv(OUTPUT_DIR / "unsw_column_inventory.csv", index=False)
    return inventory


# ── B2 + B3 + B4: Label Distribution ──────────────────────────────────────
def analysis_b2_b3_b4(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("B2 + B3 + B4: LABEL DISTRIBUTION")
    print("=" * 70)

    # Binary label
    label_col = "Label" if "Label" in df.columns else "label"
    if label_col in df.columns:
        bin_counts = df[label_col].value_counts()
        print(f"\n  Binary label distribution:")
        for val, cnt in bin_counts.items():
            print(f"    {val}: {cnt:,} ({cnt/len(df)*100:.2f}%)")

    # Attack categories
    cat_col = "attack_cat" if "attack_cat" in df.columns else None
    if cat_col:
        cat_counts = df[cat_col].value_counts()
        print(f"\n  Attack category distribution ({len(cat_counts)} categories):")
        for cat, cnt in cat_counts.items():
            print(f"    {str(cat):30s} {cnt:>10,}  ({cnt/len(df)*100:.2f}%)")

        # Horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#4CAF50" if str(c).strip().lower() == "normal" else "#F44336" for c in cat_counts.index]
        ax.barh(range(len(cat_counts)), cat_counts.values, color=colors)
        ax.set_yticks(range(len(cat_counts)))
        ax.set_yticklabels(cat_counts.index)
        ax.set_xlabel("Count")
        ax.set_title("UNSW-NB15 -- Attack Category Distribution")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "unsw_category_imbalance.png", dpi=150)
        plt.close(fig)
        print(f"  -> Saved: unsw_category_imbalance.png")

    # Pie chart of binary
    if label_col in df.columns:
        benign = bin_counts.get(0, 0)
        attack = bin_counts.get(1, 0)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie([benign, attack], labels=["Normal (0)", "Attack (1)"],
               autopct="%1.1f%%", colors=["#4CAF50", "#F44336"], startangle=90)
        ax.set_title(f"UNSW-NB15 -- Normal vs Attack Ratio\n(Normal: {benign:,}, Attack: {attack:,})")
        fig.savefig(OUTPUT_DIR / "unsw_benign_attack.png", dpi=150)
        plt.close(fig)
        print(f"  -> Saved: unsw_benign_attack.png")


# ── B5: Feature Distributions ─────────────────────────────────────────────
def analysis_b5(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("B5: FEATURE DISTRIBUTIONS (Top 12 Numerical)")
    print("=" * 70)

    label_col = "Label" if "Label" in df.columns else "label"
    df = df.copy()
    df["is_attack"] = df[label_col] == 1

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [label_col, "id"]]
    variances = df[numeric_cols].var().sort_values(ascending=False)
    top12 = variances.head(12).index.tolist()

    print(f"  Top 12 by variance: {top12}")

    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    axes = axes.flatten()
    for i, col in enumerate(top12):
        ax = axes[i]
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        benign = df.loc[~df["is_attack"], col].clip(lo, hi).dropna()
        attack = df.loc[df["is_attack"], col].clip(lo, hi).dropna()

        if len(benign) > 0:
            sns.kdeplot(benign, ax=ax, color="#4CAF50", label="Normal", fill=True, alpha=0.3)
        if len(attack) > 0:
            sns.kdeplot(attack, ax=ax, color="#F44336", label="Attack", fill=True, alpha=0.3)
        ax.set_title(col[:60])
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("UNSW-NB15 -- Top 12 Numerical Feature Distributions (Normal vs Attack)", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "unsw_feature_kde.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: unsw_feature_kde.png")


# ── B6: Correlation ──────────────────────────────────────────────────────
def analysis_b6(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("B6: FEATURE CORRELATION (Top 20 Numerical)")
    print("=" * 70)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["Label", "label", "id"]]
    variances = df[numeric_cols].var().sort_values(ascending=False)
    top20 = variances.head(20).index.tolist()

    corr = df[top20].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"fontsize": 6})
    ax.set_title("UNSW-NB15 -- Spearman Correlation (Top 20 Numerical Features)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "unsw_corr_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: unsw_corr_heatmap.png")

    high_corr = [(top20[i], top20[j], corr.iloc[i, j])
                 for i in range(len(top20)) for j in range(i+1, len(top20))
                 if abs(corr.iloc[i, j]) > 0.9]
    if high_corr:
        print("  Highly correlated pairs (|r| > 0.9):")
        for a, b, v in high_corr:
            print(f"    {a} <-> {b}: r = {v:.3f}")


# ── B7: Protocol Distribution ─────────────────────────────────────────────
def analysis_b7(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("B7: PROTOCOL DISTRIBUTION")
    print("=" * 70)

    proto_col = "proto" if "proto" in df.columns else None
    if proto_col is None:
        print("  No 'proto' column -- skipping")
        return

    counts = df[proto_col].value_counts()
    print(f"  Protocol values ({len(counts)}):")
    for val, cnt in counts.items():
        print(f"    {str(val):20s} {cnt:>10,}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=sns.color_palette("Set3", len(counts)))
    ax.set_title("UNSW-NB15 -- Protocol Distribution")
    fig.savefig(OUTPUT_DIR / "unsw_protocol_pie.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: unsw_protocol_pie.png")


# ── B8: Service Distribution ──────────────────────────────────────────────
def analysis_b8(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("B8: SERVICE DISTRIBUTION")
    print("=" * 70)

    svc_col = "service" if "service" in df.columns else None
    if svc_col is None:
        print("  No 'service' column -- skipping")
        return

    counts = df[svc_col].value_counts().head(15)
    print(f"  Top 15 service values:")
    for val, cnt in counts.items():
        print(f"    {str(val):20s} {cnt:>10,}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(counts)), counts.values, color="#1a237e")
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index)
    ax.set_xlabel("Count")
    ax.set_title("UNSW-NB15 -- Top 15 Service Values")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "unsw_service_bar.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: unsw_service_bar.png")


# ── B9: State Distribution ────────────────────────────────────────────────
def analysis_b9(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("B9: STATE DISTRIBUTION")
    print("=" * 70)

    state_col = "state" if "state" in df.columns else None
    if state_col is None:
        print("  No 'state' column -- skipping")
        return

    counts = df[state_col].value_counts().head(15)
    print(f"  Top 15 state values:")
    for val, cnt in counts.items():
        print(f"    {str(val):20s} {cnt:>10,}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(counts)), counts.values, color="#1a237e")
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index)
    ax.set_xlabel("Count")
    ax.set_title("UNSW-NB15 -- Top 15 State Values")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "unsw_state_bar.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: unsw_state_bar.png")


# ── B10: Timeline Check ───────────────────────────────────────────────────
def analysis_b10(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("B10: TIMELINE CHECK")
    print("=" * 70)

    stime_col = "Stime" if "Stime" in df.columns else None
    ltime_col = "Ltime" if "Ltime" in df.columns else None

    if stime_col is None or ltime_col is None:
        print("  Stime/Ltime columns not found in raw data -- skipping timeline")
        print("  (These columns exist only in raw 49-col format, not in train/test 45-col)")
        return

    try:
        stime = pd.to_datetime(df[stime_col], unit="s", errors="coerce")
    except Exception:
        try:
            stime = pd.to_datetime(df[stime_col], errors="coerce")
        except Exception:
            print("  Could not parse Stime -- skipping")
            return

    valid = stime.dropna()
    if len(valid) < 2:
        print("  Not enough valid timestamps -- skipping")
        return

    stime_sorted = valid.sort_values()
    gaps = stime_sorted.diff().dropna()

    print(f"  Timestamp range: {stime_sorted.min()} -> {stime_sorted.max()}")
    print(f"  Valid timestamps: {len(valid):,} / {len(df):,}")
    print(f"  Median inter-flow gap: {gaps.median()}")
    print(f"  Max inter-flow gap: {gaps.max()}")
    print(f"  Gaps > 1 hour: {(gaps > pd.Timedelta(hours=1)).sum()}")
    print(f"  Gaps > 1 day: {(gaps > pd.Timedelta(days=1)).sum()}")

    # Cumulative flows over time
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(stime_sorted, range(len(stime_sorted)), linewidth=0.5, color="#1a237e")
    ax.set_title("UNSW-NB15 -- Cumulative Flows Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Cumulative Flow Count")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "unsw_timeline.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: unsw_timeline.png")


# ── B11: Duplicate Check ──────────────────────────────────────────────────
def analysis_b11(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("B11: DUPLICATE ROW CHECK (Across 4 Raw Files)")
    print("=" * 70)

    # Check per-file
    for i, fpath in enumerate(RAW_FILES, 1):
        col_names = load_features()
        raw = pd.read_csv(fpath, header=None, names=col_names, engine="python",
                          on_bad_lines="skip")
        n_rows = len(raw)
        n_unique = len(raw.drop_duplicates())
        dup_pct = (n_rows - n_unique) / n_rows * 100 if n_rows > 0 else 0
        print(f"  File {i}: {n_rows:>8,} rows -> {n_unique:>8,} unique -> {dup_pct:.2f}% duplicates")

    n_total = len(df)
    n_unique = len(df.drop_duplicates())
    print(f"  Combined: {n_total:>8,} rows -> {n_unique:>8,} unique -> {(n_total-n_unique)/n_total*100:.2f}% duplicates")


# ── B12: Train/Test Partition Drift ───────────────────────────────────────
def analysis_b12(df_raw: pd.DataFrame):
    print("\n" + "=" * 70)
    print("B12: TRAIN/TEST PARTITION vs RAW -- DISTRIBUTION DRIFT")
    print("=" * 70)

    try:
        train, test = load_train_test()
    except Exception as e:
        print(f"  Could not load train/test CSVs: {e}")
        return

    print(f"  Train partition: {len(train):,} rows, {len(train.columns)} cols")
    print(f"  Test partition:  {len(test):,} rows, {len(test.columns)} cols")
    print(f"  Raw combined:    {len(df_raw):,} rows, {len(df_raw.columns)} cols")

    # Map raw column names to train/test column names
    raw_names = set(df_raw.columns.str.lower().str.strip())
    tt_names = set(train.columns.str.lower().str.strip())
    in_both = raw_names & tt_names
    raw_only = raw_names - tt_names
    tt_only = tt_names - raw_names
    print(f"  Columns in both: {len(in_both)}")
    print(f"  Raw only: {raw_only}")
    print(f"  Train/test only: {tt_only}")

    # Compare top 5 overlapping numerical features
    overlap_numeric = []
    for col in in_both:
        raw_col = [c for c in df_raw.columns if c.lower().strip() == col]
        tt_col = [c for c in train.columns if c.lower().strip() == col]
        if raw_col and tt_col:
            if pd.api.types.is_numeric_dtype(train[tt_col[0]]):
                overlap_numeric.append((raw_col[0], tt_col[0]))

    if len(overlap_numeric) >= 5:
        top5 = overlap_numeric[:5]
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        for i, (raw_c, tt_c) in enumerate(top5):
            ax = axes[i]
            raw_vals = df_raw[raw_c].dropna().sample(min(5000, len(df_raw)), random_state=SEED)
            train_vals = train[tt_c].dropna().sample(min(5000, len(train)), random_state=SEED)
            test_vals = test[tt_c].dropna().sample(min(5000, len(test)), random_state=SEED)
            lo = np.quantile(pd.concat([raw_vals, train_vals, test_vals]), 0.01)
            hi = np.quantile(pd.concat([raw_vals, train_vals, test_vals]), 0.99)
            sns.kdeplot(raw_vals.clip(lo, hi), ax=ax, label="Raw", color="blue")
            sns.kdeplot(train_vals.clip(lo, hi), ax=ax, label="Train", color="green")
            sns.kdeplot(test_vals.clip(lo, hi), ax=ax, label="Test", color="red")
            ax.set_title(raw_c[:30])
            ax.legend(fontsize=7)
        fig.suptitle("UNSW-NB15 -- Raw vs. Official Train/Test Distribution (Top 5 Features)")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "unsw_train_test_drift.png", dpi=150)
        plt.close(fig)
        print(f"  -> Saved: unsw_train_test_drift.png")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="UNSW-NB15 EDA")
    parser.add_argument("--sample", type=int, default=None, help="Subsample N rows total")
    parser.add_argument("--target", type=str, default="laptop", choices=["laptop", "kaggle"])
    args = parser.parse_args()

    if args.target != "laptop":
        raise RuntimeError("EDA scripts must run on laptop.")

    print("=" * 70)
    print("UNSW-NB15 EXPLORATORY DATA ANALYSIS")
    print(f"Sample: {args.sample or 'FULL'}")
    print("=" * 70)

    print("\nLoading raw data (no header -> names from features file)...")
    df = load_raw_data(sample=args.sample)
    print(f"Loaded: {len(df):,} rows × {len(df.columns)} cols")

    # Run analyses
    analysis_b1(df)
    analysis_b2_b3_b4(df)
    analysis_b5(df)
    analysis_b6(df)
    analysis_b7(df)
    analysis_b8(df)
    analysis_b9(df)
    analysis_b10(df)
    analysis_b11(df)
    analysis_b12(df)

    print("\n" + "=" * 70)
    print("UNSW-NB15 EDA COMPLETE")
    print(f"All outputs in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
