#!/usr/bin/env python3
"""
CIC-IDS2017 Exploratory Data Analysis
Target: LAPTOP
Produces: 10 analyses, 6 charts, column inventory
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
from collections import Counter
import json
from pathlib import Path

from nids import set_seed, SEED

set_seed()

# ── Config ────────────────────────────────────────────────────────────────
DATASET_DIR = Path(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "CICIDS2017"))
OUTPUT_DIR = Path(os.path.join(os.path.dirname(__file__), "..", "eda_output"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CSV_FILES = sorted(DATASET_DIR.glob("*.csv"))
DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday-WebAttacks",
              "Thursday-Infiltration", "Friday-Morning", "Friday-DDoS", "Friday-PortScan"]

# Chart style
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (14, 8),
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


def _sanitize(s: str) -> str:
    """Replace non-printable chars with space, collapse whitespace."""
    import re
    return re.sub(r"\s+", " ", "".join(c if c.isprintable() or c in "\n\r\t" else " " for c in s)).strip()


def load_all_days(sample: int | None = None) -> dict[str, pd.DataFrame]:
    """Load all CICIDS2017 daily CSVs. Each file has a header row plus one
    spurious 'Label' data row that is actually the header repeated."""
    dfs = {}
    for fpath in CSV_FILES:
        day = fpath.stem.split("-")[0]
        # Disambiguate days with multiple files (Thursday × 2, Friday × 3)
        stem = fpath.stem.lower()
        if "afternoon" in stem:
            if "ddos" in stem:
                day = "Friday-DDoS"
            elif "portscan" in stem:
                day = "Friday-PortScan"
            elif "infilteration" in stem:
                day = "Thursday-Infiltration"
            else:
                day = f"{day}-Afternoon"
        elif "morning" in stem:
            if "webattacks" in stem:
                day = "Thursday-WebAttacks"
            else:
                day = f"{day}-Morning"
        print(f"  Loading {fpath.name} -> {day} ...")

        # Use Python engine for CR line endings
        df = pd.read_csv(
            fpath,
            engine="python",
            encoding="latin-1",
            on_bad_lines="skip",
        )
        # Strip whitespace from column names (CICFlowMeter has " Label", etc.)
        df.columns = [c.strip() for c in df.columns]
        # Remove the spurious header-duplicate row (where "Label" appears as data)
        label_col = "Label" if "Label" in df.columns else None
        if label_col:
            df = df[df[label_col] != "Label"].copy()
        # Strip whitespace and sanitize non-printable chars from string columns
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.strip()
        # Explicitly sanitize Label column (has cp1252 artifacts like en-dash)
        if label_col:
            df[label_col] = df[label_col].fillna("BENIGN").apply(_sanitize)

        if sample and len(df) > sample:
            df = df.sample(n=sample, random_state=SEED)

        dfs[day] = df
        print(f"    -> {len(df):,} rows, {len(df.columns)} columns")

    return dfs


# ── A1: Column Inventory ──────────────────────────────────────────────────
def analysis_a1(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("A1: COLUMN INVENTORY")
    print("=" * 70)

    # Use Monday as reference (full column set)
    sample_df = list(dfs.values())[0]
    inventory = pd.DataFrame({
        "column": sample_df.columns.tolist(),
        "dtype": [str(sample_df[c].dtype) for c in sample_df.columns],
        "missing_pct": [round(sample_df[c].isna().mean() * 100, 2) for c in sample_df.columns],
        "n_unique": [sample_df[c].nunique() for c in sample_df.columns],
    })
    print(f"  Total columns: {len(inventory)}")
    print(f"  Numeric: {(inventory['dtype'].str.contains('float|int')).sum()}")
    print(f"  Object: {(inventory['dtype'] == 'object').sum()}")
    print(f"  Columns with >0% missing: {(inventory['missing_pct'] > 0).sum()}")
    print(f"\n  First 10 columns:\n{inventory.head(10).to_string(index=False)}")
    print(f"\n  Last 10 columns:\n{inventory.tail(10).to_string(index=False)}")

    inventory.to_csv(OUTPUT_DIR / "cic17_column_inventory.csv", index=False)
    return inventory


# ── A2 + A3: Label Distribution ───────────────────────────────────────────
def analysis_a2_a3(dfs: dict[str, pd.DataFrame]):
    print("\n" + "=" * 70)
    print("A2 + A3: LABEL DISTRIBUTION")
    print("=" * 70)

    # Per-day label counts
    rows = []
    for day, df in dfs.items():
        counts = df["Label"].value_counts()
        for label, cnt in counts.items():
            rows.append({"day": day, "label": label, "count": cnt, "pct": cnt / len(df) * 100})
    label_df = pd.DataFrame(rows)

    # Print
    for day in DAYS_ORDER:
        if day in dfs:
            subset = label_df[label_df["day"] == day].sort_values("count", ascending=False)
            print(f"\n  {day}: {len(subset)} label types")
            for _, r in subset.iterrows():
                print(f"    {r['label']:30s} {r['count']:>10,}  ({r['pct']:.2f}%)")

    # ── Stacked bar: days × labels ──
    pivot = label_df.pivot_table(index="day", columns="label", values="count", aggfunc="sum", fill_value=0)
    # Keep only top 10 labels, rest -> "Other"
    top_labels = label_df.groupby("label")["count"].sum().nlargest(10).index.tolist()
    pivot["Other"] = pivot[[c for c in pivot.columns if c not in top_labels]].sum(axis=1)
    pivot = pivot[[l for l in top_labels if l in pivot.columns] + ["Other"]]

    fig, ax = plt.subplots(figsize=(16, 7))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("CIC-IDS2017 -- Label Distribution per Day")
    ax.set_xlabel("Day")
    ax.set_ylabel("Flow Count")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "cic17_label_dist_day.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: cic17_label_dist_day.png")

    # ── Benign/Attack ratio pie ──
    all_labels = label_df.groupby("label")["count"].sum()
    benign_count = all_labels.get("BENIGN", 0)
    attack_count = all_labels.sum() - benign_count
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie([benign_count, attack_count], labels=["BENIGN", "Attack"],
           autopct="%1.1f%%", colors=["#4CAF50", "#F44336"], startangle=90)
    ax.set_title(f"CIC-IDS2017 -- Benign vs Attack Ratio\n(Benign: {benign_count:,}, Attack: {attack_count:,})")
    fig.savefig(OUTPUT_DIR / "cic17_benign_attack_ratio.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: cic17_benign_attack_ratio.png")

    return label_df


# ── A4: Temporal Flow Density ─────────────────────────────────────────────
def analysis_a4(dfs: dict[str, pd.DataFrame]):
    print("\n" + "=" * 70)
    print("A4: TEMPORAL FLOW DENSITY")
    print("=" * 70)

    time_col = "Timestamp" if "Timestamp" in list(dfs.values())[0].columns else None
    if time_col is None:
        print("  WARNING: No 'Timestamp' column found -- skipping temporal analysis")
        return

    # CIC17 days were captured independently; stitch them
    all_times = []
    cumulative_offset = pd.Timedelta(0)
    base_date = pd.Timestamp("2017-07-03")  # Monday

    for day in DAYS_ORDER:
        if day not in dfs:
            continue
        df = dfs[day]
        try:
            times = pd.to_datetime(df[time_col], format="mixed", dayfirst=False)
        except Exception:
            print(f"  WARNING: Could not parse timestamps for {day} -- skipping")
            continue

        # Offset Monday to a reference date, then apply cumulative offset
        day_of_week = DAYS_ORDER.index(day)
        day_start = base_date + pd.Timedelta(days=day_of_week)
        # Shift times so they start at day_start
        min_t = times.min()
        shifted = times - min_t + day_start + cumulative_offset
        all_times.extend(shifted.tolist())

    if not all_times:
        print("  No timestamps parsed -- skipping")
        return

    all_times = pd.Series(all_times).sort_values()

    # Flows per minute using resample on the sorted times
    ts_index = pd.DatetimeIndex(all_times)
    flows_per_min = pd.Series(1, index=ts_index).resample("1min").count()

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(flows_per_min.index, flows_per_min.values, linewidth=0.5, color="#1a237e")
    ax.set_title("CIC-IDS2017 -- Flow Density (flows/minute) -- Stitched Week Timeline")
    ax.set_xlabel("Time (stitched Mon->Fri)")
    ax.set_ylabel("Flows / minute")
    median_val = flows_per_min.median()
    ax.axhline(y=median_val, color="red", linestyle="--", alpha=0.5, label=f"Median: {median_val:.0f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "cic17_temporal_density.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: cic17_temporal_density.png")
    print(f"  Total time range: {all_times.min()} -> {all_times.max()}")
    print(f"  Median flows/min: {flows_per_min.median():.0f}")


# ── A5: Feature KDE Distributions ─────────────────────────────────────────
def analysis_a5(dfs: dict[str, pd.DataFrame]):
    print("\n" + "=" * 70)
    print("A5: FEATURE DISTRIBUTIONS (Top 12 Numerical)")
    print("=" * 70)

    # Combine all days, label as BENIGN/Attack
    combined = pd.concat(dfs.values(), ignore_index=True)
    combined["is_attack"] = combined["Label"] != "BENIGN"

    # Select top numerical columns (exclude IPs, ports, flags, identifiers)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    # Remove known non-feature columns (like unnamed)
    numeric_cols = [c for c in numeric_cols if not c.startswith("Unnamed")]
    # Take top 12 by variance
    variances = combined[numeric_cols].var().sort_values(ascending=False)
    top12 = variances.head(12).index.tolist()

    print(f"  Top 12 numerical features by variance: {top12}")

    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    axes = axes.flatten()
    for i, col in enumerate(top12):
        ax = axes[i]
        # Clip to 1st-99th percentile for readability
        lo, hi = combined[col].quantile(0.01), combined[col].quantile(0.99)
        benign_vals = combined.loc[~combined["is_attack"], col].clip(lo, hi)
        attack_vals = combined.loc[combined["is_attack"], col].clip(lo, hi)

        if len(benign_vals) > 0:
            sns.kdeplot(benign_vals, ax=ax, color="#4CAF50", label="BENIGN", fill=True, alpha=0.3)
        if len(attack_vals) > 0:
            sns.kdeplot(attack_vals, ax=ax, color="#F44336", label="Attack", fill=True, alpha=0.3)
        ax.set_title(col[:60])
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("CIC-IDS2017 -- Top 12 Numerical Feature Distributions (BENIGN vs Attack)", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "cic17_feature_kde.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: cic17_feature_kde.png")


# ── A6: Correlation Heatmap ───────────────────────────────────────────────
def analysis_a6(dfs: dict[str, pd.DataFrame]):
    print("\n" + "=" * 70)
    print("A6: FEATURE CORRELATION (Top 20 Numerical)")
    print("=" * 70)

    combined = pd.concat(dfs.values(), ignore_index=True)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.startswith("Unnamed")]

    variances = combined[numeric_cols].var().sort_values(ascending=False)
    top20 = variances.head(20).index.tolist()

    corr = combined[top20].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                annot_kws={"fontsize": 6})
    ax.set_title("CIC-IDS2017 -- Spearman Correlation (Top 20 Numerical Features)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "cic17_corr_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: cic17_corr_heatmap.png")

    # Print pairs with |corr| > 0.9
    high_corr = []
    for i in range(len(top20)):
        for j in range(i + 1, len(top20)):
            if abs(corr.iloc[i, j]) > 0.9:
                high_corr.append((top20[i], top20[j], corr.iloc[i, j]))
    if high_corr:
        print("  Highly correlated pairs (|r| > 0.9):")
        for a, b, v in high_corr:
            print(f"    {a} <-> {b}: r = {v:.3f}")


# ── A7: Protocol Distribution ─────────────────────────────────────────────
def analysis_a7(dfs: dict[str, pd.DataFrame]):
    print("\n" + "=" * 70)
    print("A7: PROTOCOL DISTRIBUTION")
    print("=" * 70)

    combined = pd.concat(dfs.values(), ignore_index=True)
    proto_col = "Protocol" if "Protocol" in combined.columns else None
    if proto_col is None:
        print("  WARNING: No 'Protocol' column -- skipping")
        return

    proto_counts = combined[proto_col].value_counts()
    print(f"  Protocol values: {proto_counts.to_dict()}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(proto_counts.values, labels=proto_counts.index, autopct="%1.1f%%",
           colors=sns.color_palette("Set3", len(proto_counts)))
    ax.set_title(f"CIC-IDS2017 -- Protocol Distribution ({len(proto_counts)} values)")
    fig.savefig(OUTPUT_DIR / "cic17_protocol_pie.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: cic17_protocol_pie.png")


# ── A8: Inf/NaN Audit ─────────────────────────────────────────────────────
def analysis_a8(dfs: dict[str, pd.DataFrame]):
    print("\n" + "=" * 70)
    print("A8: INF / NAN AUDIT")
    print("=" * 70)

    combined = pd.concat(dfs.values(), ignore_index=True)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()

    issues = {}
    for col in numeric_cols:
        inf_count = np.isinf(combined[col]).sum()
        nan_count = combined[col].isna().sum()
        if inf_count > 0 or nan_count > 0:
            issues[col] = {"inf": int(inf_count), "nan": int(nan_count), "total": len(combined)}

    if issues:
        print(f"  Columns with Inf/NaN issues: {len(issues)}")
        for col, counts in sorted(issues.items(), key=lambda x: x[1]["inf"] + x[1]["nan"], reverse=True):
            print(f"    {col:40s}  Inf={counts['inf']:>10,}  NaN={counts['nan']:>10,}  ({counts['inf'] + counts['nan']:,} / {counts['total']:,})")
    else:
        print("  No Inf/NaN issues found in any column")


# ── A9: Duplicate Check ───────────────────────────────────────────────────
def analysis_a9(dfs: dict[str, pd.DataFrame]):
    print("\n" + "=" * 70)
    print("A9: DUPLICATE ROW CHECK")
    print("=" * 70)

    for day, df in dfs.items():
        n_rows = len(df)
        n_unique = len(df.drop_duplicates())
        dup_pct = (n_rows - n_unique) / n_rows * 100 if n_rows > 0 else 0
        print(f"  {day:15s}: {n_rows:>8,} rows -> {n_unique:>8,} unique -> {dup_pct:.2f}% duplicates")

    combined = pd.concat(dfs.values(), ignore_index=True)
    n_total = len(combined)
    n_unique_total = len(combined.drop_duplicates())
    print(f"  {'ALL DAYS':15s}: {n_total:>8,} rows -> {n_unique_total:>8,} unique -> {(n_total - n_unique_total) / n_total * 100:.2f}% duplicates")


# ── A10: Duration Distribution ────────────────────────────────────────────
def analysis_a10(dfs: dict[str, pd.DataFrame]):
    print("\n" + "=" * 70)
    print("A10: DURATION DISTRIBUTION")
    print("=" * 70)

    combined = pd.concat(dfs.values(), ignore_index=True)
    dur_col = "Flow Duration" if "Flow Duration" in combined.columns else None
    if dur_col is None:
        print("  WARNING: No 'Flow Duration' column -- skipping")
        return

    durations = combined[dur_col].clip(lower=0).replace([np.inf, -np.inf], np.nan).dropna()
    # CIC17 durations are in microseconds -- show in seconds
    durations_sec = durations / 1_000_000
    print(f"  Duration range (seconds): {durations_sec.min():.6f} -> {durations_sec.max():.2f}")
    print(f"  Duration median (seconds): {durations_sec.median():.6f}")
    print(f"  Duration mean (seconds): {durations_sec.mean():.2f}")
    print(f"  Flows with duration = 0: {(durations_sec == 0).sum():,}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(np.log10(durations_sec.clip(lower=1e-9)), bins=100, color="#1a237e", alpha=0.8)
    ax.set_title("CIC-IDS2017 -- Flow Duration Distribution (log₁₀ seconds)")
    ax.set_xlabel("log₁₀(Duration in seconds)")
    ax.set_ylabel("Count")
    ax.axvline(x=np.log10(durations_sec.median()), color="red", linestyle="--",
               label=f"Median: {durations_sec.median():.4f}s")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "cic17_duration_hist.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: cic17_duration_hist.png")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CIC-IDS2017 EDA")
    parser.add_argument("--sample", type=int, default=None, help="Subsample N rows per day for dev")
    parser.add_argument("--target", type=str, default="laptop", choices=["laptop", "kaggle"],
                        help="Compute target (laptop only for EDA)")
    args = parser.parse_args()

    if args.target != "laptop":
        raise RuntimeError("EDA scripts must run on laptop. Use --target laptop.")

    print("=" * 70)
    print("CIC-IDS2017 EXPLORATORY DATA ANALYSIS")
    print(f"Sample per day: {args.sample or 'FULL'}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    # Load
    print("\nLoading data...")
    dfs = load_all_days(sample=args.sample)
    print(f"Loaded {len(dfs)} days: {list(dfs.keys())}")

    # Run analyses
    inventory = analysis_a1(dfs)
    analysis_a2_a3(dfs)
    analysis_a4(dfs)
    analysis_a5(dfs)
    analysis_a6(dfs)
    analysis_a7(dfs)
    analysis_a8(dfs)
    analysis_a9(dfs)
    analysis_a10(dfs)

    print("\n" + "=" * 70)
    print("CIC-IDS2017 EDA COMPLETE")
    print(f"All outputs in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
