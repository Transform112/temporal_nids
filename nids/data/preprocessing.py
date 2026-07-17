#!/usr/bin/env python3
"""
Phase 2 — Preprocessing & Feature Harmonization
Target: LAPTOP (chunked reads, --sample flag for dev)
Produces: cic17_native.parquet, cic17_common.parquet, unsw_native.parquet,
          unsw_common.parquet

DO NOT run full-scale on laptop without --sample.
StandardScaler fitting is deferred to Phase 5 (after data splits exist).
"""

import sys
import os
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from nids import set_seed, SEED

set_seed()

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
CIC17_DIR = DATASET_DIR / "CICIDS2017"
UNSW_DIR = DATASET_DIR / "UNSWNB15"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed"
try:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
except OSError:
    pass  # Read-only filesystem (Kaggle input)

PROVENANCE_PATH = PROJECT_ROOT / "feature_provenance.json"
PROTOCOL_MAP_PATH = PROJECT_ROOT / "local" / "eda_output" / "protocol_encoding_map.json"

# ── Constants ───────────────────────────────────────────────────────────────
CHUNKSIZE = 100_000  # rows per chunk for laptop-safe loading

# IANA protocol number → name mapping (used by CIC17)
IANA_TO_NAME: Dict[str, str] = {
    "0": "hopopt", "1": "icmp", "2": "igmp", "6": "tcp", "17": "udp",
    "41": "ipv6", "47": "gre", "50": "esp", "51": "ah", "58": "ipv6-icmp",
    "89": "ospf", "132": "sctp",
}


def _sanitize(s: str) -> str:
    """Replace non-printable chars with space, collapse whitespace."""
    if not isinstance(s, str):
        return s
    return re.sub(r"\s+", " ", "".join(
        c if c.isprintable() or c in "\n\r\t" else " " for c in s
    )).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# CIC-IDS2017 Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_cic17(sample: Optional[int] = None, chunksize: int = CHUNKSIZE) -> pd.DataFrame:
    """Load all CIC-IDS2017 daily CSVs, clean, and combine.

    Cleans:
      - Strips leading/trailing whitespace from column names
      - Removes spurious header-duplicate rows (where 'Label' == 'Label')
      - Strips whitespace from string columns
      - Sanitizes non-printable chars in Label column
      - Converts numeric columns from object to float where possible
    """
    csv_files = sorted(CIC17_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {CIC17_DIR}")

    print(f"  Found {len(csv_files)} CIC-IDS2017 daily files")

    chunks = []
    total_rows = 0

    for fpath in csv_files:
        fname = fpath.name
        print(f"    Loading {fname} ...")

        try:
            df = pd.read_csv(
                fpath, engine="python", encoding="latin-1",
                on_bad_lines="skip",
            )
        except Exception as e:
            print(f"      WARNING: Failed to load {fname}: {e}")
            continue

        # Strip whitespace from column names (" Label" -> "Label", etc.)
        df.columns = [c.strip() for c in df.columns]

        # Remove spurious header-duplicate rows
        label_col = "Label" if "Label" in df.columns else None
        if label_col:
            before = len(df)
            df = df[df[label_col] != "Label"].copy()
            removed = before - len(df)
            if removed > 0:
                print(f"      Removed {removed} header-duplicate rows")

        # Strip whitespace from all string columns
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.strip()

        # Sanitize Label column (handles cp1252 artifacts like en-dash)
        if label_col:
            df[label_col] = df[label_col].fillna("BENIGN").apply(_sanitize)

        # Convert numeric-looking object columns to numeric
        for col in df.columns:
            if col == label_col:
                continue
            if df[col].dtype == object:
                try:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    # Only convert if most values parse as numeric
                    if numeric.notna().mean() > 0.95:
                        df[col] = numeric
                except Exception:
                    pass

        if sample and len(df) > sample:
            # Sample proportionally from each file if a total target is given
            n_take = max(1, int(sample * len(df) / sum(
                1 for _ in csv_files
            )))
            df = df.sample(n=min(n_take, len(df)), random_state=SEED)

        total_rows += len(df)
        chunks.append(df)
        print(f"      -> {len(df):,} rows")

    combined = pd.concat(chunks, ignore_index=True)
    print(f"    Total CIC-IDS2017: {total_rows:,} rows × {len(combined.columns)} columns")

    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# UNSW-NB15 Loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load_unsw_column_names() -> list:
    """Parse NUSW-NB15_features.csv to get column names for raw CSVs."""
    feat_file = UNSW_DIR / "NUSW-NB15_features.csv"
    if not feat_file.exists():
        raise FileNotFoundError(f"UNSW features file not found: {feat_file}")
    feat_df = pd.read_csv(feat_file, encoding="latin-1")
    col_names = feat_df["Name"].str.strip().tolist()
    print(f"    Features file: {len(col_names)} column names")
    return col_names


def load_unsw(sample: Optional[int] = None, chunksize: int = CHUNKSIZE) -> pd.DataFrame:
    """Load all 4 UNSW-NB15 raw CSVs (no header row), assign column names, combine.

    Cleans:
      - Strips whitespace from string columns (attack_cat has leading spaces)
      - Converts numeric-looking columns from object to numeric
    """
    col_names = _load_unsw_column_names()
    raw_files = sorted(UNSW_DIR.glob("UNSW-NB15_[1-4].csv"))

    if len(raw_files) < 4:
        raise FileNotFoundError(
            f"Expected 4 UNSW-NB15 raw files, found {len(raw_files)}: {raw_files}"
        )

    print(f"  Found {len(raw_files)} UNSW-NB15 raw part files")

    dfs = []
    total_rows = 0

    for fpath in raw_files:
        fname = fpath.name
        print(f"    Loading {fname} ...")

        try:
            df = pd.read_csv(
                fpath, header=None, names=col_names,
                engine="python", encoding="latin-1",
                on_bad_lines="skip",
            )
        except Exception as e:
            print(f"      WARNING: Failed to load {fname}: {e}")
            continue

        # Strip whitespace from string columns
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.strip()

        # Convert numeric-looking columns
        for col in df.columns:
            if col in ("attack_cat", "Label", "label"):
                continue
            if df[col].dtype == object:
                try:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    if numeric.notna().mean() > 0.95:
                        df[col] = numeric
                except Exception:
                    pass

        if sample and len(df) > sample:
            n_take = max(1, sample // len(raw_files))
            df = df.sample(n=min(n_take, len(df)), random_state=SEED)

        total_rows += len(df)
        dfs.append(df)
        print(f"      -> {len(df):,} rows")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"    Total UNSW-NB15: {total_rows:,} rows × {len(combined.columns)} columns")

    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# Deduplication
# ═══════════════════════════════════════════════════════════════════════════════

def deduplicate(df: pd.DataFrame, label: str = "dataset") -> pd.DataFrame:
    """Drop exact-duplicate rows. Reports dedup count."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"    [{label}] Dedup: {before:,} -> {after:,} "
          f"({(before - after):,} removed, {(before - after) / before * 100:.2f}%)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Conversions & Inf handling
# ═══════════════════════════════════════════════════════════════════════════════

def apply_cic17_conversions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply CIC17-specific conversions:
    - Flow Duration: µs -> seconds
    - Flow Bytes/s, Flow Packets/s: Inf -> 0, NaN -> 0
    """
    df = df.copy()

    # Duration: µs to seconds
    if "Flow Duration" in df.columns:
        dur = pd.to_numeric(df["Flow Duration"], errors="coerce")
        # Clip negative values to 0, convert µs to seconds
        df["Flow Duration"] = dur.clip(lower=0).fillna(0) / 1_000_000
        print(f"    CIC17: Flow Duration converted µs -> seconds")

    # Rate columns: Inf -> 0
    for rate_col in ["Flow Bytes/s", "Flow Packets/s"]:
        if rate_col in df.columns:
            col = pd.to_numeric(df[rate_col], errors="coerce")
            inf_mask = np.isinf(col)
            nan_mask = col.isna()
            n_bad = inf_mask.sum() + nan_mask.sum()
            if n_bad > 0:
                col[inf_mask | nan_mask] = 0.0
                df[rate_col] = col
                print(f"    CIC17: {rate_col} -> {n_bad:,} Inf/NaN values replaced with 0")

    return df


def apply_unsw_conversions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply UNSW-specific conversions:
    - dur is already in seconds (no conversion needed, just clip negatives)
    """
    df = df.copy()

    if "dur" in df.columns:
        dur = pd.to_numeric(df["dur"], errors="coerce")
        neg_count = (dur < 0).sum()
        if neg_count > 0:
            dur = dur.clip(lower=0)
            df["dur"] = dur
            print(f"    UNSW: dur -> {neg_count:,} negative values clipped to 0")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Derived Features for UNSW-NB15
# ═══════════════════════════════════════════════════════════════════════════════

def _unsw_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first column name from candidates that exists in df.
    Case-insensitive fallback.
    """
    for c in candidates:
        if c in df.columns:
            return c
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def derive_unsw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive UNSW-NB15 features that lack direct column mappings.

    Per feature_provenance.json:
      - byte_rate: (Sload + Dload) / 2  (proxy for Flow Bytes/s)
      - packet_rate: Spkts / dur  (pkts per second)
      - mean_iat: (Sintpkt + Dintpkt) / 2  (avg of src/dst inter-arrival times)
      - std_iat: (Sjit + Djit) / 2  (jitter proxy for std(IAT))
      - mean_pkt_len: (Smeansz + Dmeansz) / 2
      - syn_count: 0 (imputed)
      - ack_count: 0 (imputed)
      - down_up_ratio: dbytes / sbytes (guard div-by-zero)
      - state_summary: from 'state' column (one-hot handled later)

    Returns df with new derived columns added.
    """
    df = df.copy()

    # Resolve actual column names (UNSW uses PascalCase: Sload, Dload, etc.)
    sload_c = _unsw_col(df, "Sload", "sload")
    dload_c = _unsw_col(df, "Dload", "dload")
    spkts_c = _unsw_col(df, "Spkts", "spkts")
    dur_c = _unsw_col(df, "dur")
    sintpkt_c = _unsw_col(df, "Sintpkt", "sintpkt")
    dintpkt_c = _unsw_col(df, "Dintpkt", "dintpkt")
    sjit_c = _unsw_col(df, "Sjit", "sjit")
    djit_c = _unsw_col(df, "Djit", "djit")
    smeansz_c = _unsw_col(df, "Smeansz", "smeansz")
    dmeansz_c = _unsw_col(df, "Dmeansz", "dmeansz")
    sbytes_c = _unsw_col(df, "sbytes")
    dbytes_c = _unsw_col(df, "dbytes")

    # Ensure numeric types for source columns
    for c in [sload_c, dload_c, spkts_c, dur_c, sintpkt_c, dintpkt_c,
              sjit_c, djit_c, smeansz_c, dmeansz_c, sbytes_c, dbytes_c]:
        if c:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # byte_rate: average of source/destination load
    if sload_c and dload_c:
        df["_derived_byte_rate"] = (df[sload_c] + df[dload_c]) / 2
        print("    UNSW derived: byte_rate = (Sload + Dload) / 2")

    # packet_rate: Spkts / dur
    if spkts_c and dur_c:
        dur_safe = df[dur_c].replace(0, np.nan)
        df["_derived_packet_rate"] = df[spkts_c] / dur_safe
        df["_derived_packet_rate"] = df["_derived_packet_rate"].fillna(0).replace([np.inf, -np.inf], 0)
        print("    UNSW derived: packet_rate = Spkts / dur (Inf/NaN -> 0)")

    # mean_iat: average of source and destination inter-packet times
    if sintpkt_c and dintpkt_c:
        df["_derived_mean_iat"] = (df[sintpkt_c] + df[dintpkt_c]) / 2
        print("    UNSW derived: mean_iat = (Sintpkt + Dintpkt) / 2")

    # std_iat: jitter proxy (Sjit + Djit) / 2
    if sjit_c and djit_c:
        df["_derived_std_iat"] = (df[sjit_c] + df[djit_c]) / 2
        print("    UNSW derived: std_iat = (Sjit + Djit) / 2  [jitter proxy, flagged]")

    # mean_pkt_len: average of source/dest mean packet sizes
    if smeansz_c and dmeansz_c:
        df["_derived_mean_pkt_len"] = (df[smeansz_c] + df[dmeansz_c]) / 2
        print("    UNSW derived: mean_pkt_len = (Smeansz + Dmeansz) / 2")

    # syn_count / ack_count: not available, impute 0
    df["_derived_syn_count"] = 0
    df["_derived_ack_count"] = 0
    print("    UNSW derived: syn_count = 0 (imputed), ack_count = 0 (imputed)")

    # down_up_ratio: dbytes / sbytes
    if sbytes_c and dbytes_c:
        ratio = df[dbytes_c] / df[sbytes_c].replace(0, np.nan)
        df["_derived_down_up_ratio"] = ratio.fillna(0).replace([np.inf, -np.inf], 0)
        print("    UNSW derived: down_up_ratio = dbytes / sbytes (0-guarded)")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Common Track Assembly
# ═══════════════════════════════════════════════════════════════════════════════

def load_provenance() -> list:
    """Load feature_provenance.json."""
    with open(PROVENANCE_PATH) as f:
        return json.load(f)


def load_protocol_map() -> dict:
    """Load protocol encoding map (proto name -> idx)."""
    if PROTOCOL_MAP_PATH.exists():
        with open(PROTOCOL_MAP_PATH) as f:
            data = json.load(f)
        return data.get("proto_to_idx", {})
    # Fallback: build on the fly from IANA mapping
    print("    WARNING: protocol_encoding_map.json not found, building fallback")
    return {name: i for i, name in enumerate(sorted(set(IANA_TO_NAME.values()) | {"unknown"}))}


def map_protocol_to_shared(proto_val, unsw: bool = False,
                           proto_map: Optional[dict] = None) -> int:
    """Map a raw protocol value to the shared encoding index.

    CIC17: IANA number string (e.g. "6") -> name -> idx
    UNSW: text label (e.g. "tcp") -> name -> idx
    """
    if proto_map is None:
        proto_map = {}

    unknown_idx = proto_map.get("unknown", len(proto_map))

    if pd.isna(proto_val):
        return unknown_idx

    if unsw:
        # UNSW already has text labels
        name = str(proto_val).lower().strip()
    else:
        # CIC17: IANA number -> name
        try:
            iana_str = str(int(float(proto_val)))
        except (ValueError, TypeError):
            return unknown_idx
        name = IANA_TO_NAME.get(iana_str, f"iana_{iana_str}")

    return proto_map.get(name, unknown_idx)


# ── Shared state vocabulary (UNSW states + "unknown") ───────────────────
# Canonical state values from UNSW-NB15. CIC17 flag combos are mapped to the
# closest match; anything unrecognized goes to "unknown".
_SHARED_STATES = [
    "ACC", "CLO", "CON", "ECO", "ECR", "FIN", "INT",
    "MAS", "PAR", "REQ", "RST", "TST", "TXD", "URH", "URN",
    "no", "unknown",
]


def _cic17_flags_to_state(df: pd.DataFrame) -> pd.Series:
    """Derive approximate UNSW-style state strings from CIC17 flag counts.

    Heuristic (per Plan §3.5 — best-effort mapping, differences are flagged):
      - RST flag  > 0  → "RST"
      - FIN flag  > 0  → "FIN"
      - SYN > 0, ACK == 0, no FIN, no RST → "REQ"
      - SYN > 0, ACK > 0, no FIN, no RST  → "CON"
      - ACK > 0, no SYN, no FIN, no RST   → "INT"
      - URG > 0 (no TCP control flags)    → "URH"
      - No flags set                      → "no"
      - Anything else                     → "unknown"
    """
    def _flag_val(col_name: str) -> pd.Series:
        col = f"{col_name} Flag Count"
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0)
        return pd.Series(0, index=df.index)

    fin = _flag_val("FIN")
    syn = _flag_val("SYN")
    rst = _flag_val("RST")
    ack = _flag_val("ACK")
    urg = _flag_val("URG")

    result = pd.Series("unknown", index=df.index)
    result[rst > 0] = "RST"
    result[(fin > 0) & (rst == 0)] = "FIN"
    result[(syn > 0) & (ack == 0) & (fin == 0) & (rst == 0)] = "REQ"
    result[(syn > 0) & (ack > 0) & (fin == 0) & (rst == 0)] = "CON"
    result[(ack > 0) & (syn == 0) & (fin == 0) & (rst == 0) & (urg == 0)] = "INT"
    result[(urg > 0) & (syn == 0) & (fin == 0) & (rst == 0) & (ack == 0)] = "URH"
    result[(fin == 0) & (syn == 0) & (rst == 0) & (ack == 0) & (urg == 0)] = "no"

    return result


def _build_state_onehot(state_series: pd.Series) -> pd.DataFrame:
    """One-hot encode a state series using the shared vocabulary.
    Returns DataFrame with columns 'state_<name>' for each vocab entry.
    """
    state_df = pd.DataFrame(index=state_series.index)
    for st in _SHARED_STATES:
        col_name = f"state_{st.lower()}"
        state_df[col_name] = (state_series.str.upper() == st.upper()).astype(np.float32)
    return state_df


def build_common_track_cic17(df: pd.DataFrame, provenance: list) -> pd.DataFrame:
    """Extract common-track features from cleaned CIC17 DataFrame."""
    proto_map = load_protocol_map()

    common = pd.DataFrame(index=df.index)

    for entry in provenance:
        common_name = entry["common_name"]
        cic17_src = entry["cic17_source"]

        if cic17_src is None or cic17_src == "DERIVED":
            if common_name == "state_summary":
                # Derive state from CIC17 flags, map to shared state vocabulary
                cic17_state = _cic17_flags_to_state(df)
                state_onehot = _build_state_onehot(cic17_state)
                for col in state_onehot.columns:
                    common[col] = state_onehot[col]
                continue
            else:
                # Other features missing from CIC17 — skip
                continue

        if cic17_src not in df.columns:
            print(f"    WARNING: CIC17 column '{cic17_src}' not found for '{common_name}'")
            continue

        col = df[cic17_src]

        # Protocol needs encoding
        if common_name == "protocol":
            common[common_name] = col.apply(
                lambda x: map_protocol_to_shared(x, unsw=False, proto_map=proto_map)
            ).astype(np.int32)
        else:
            # Numeric feature
            common[common_name] = pd.to_numeric(col, errors="coerce").fillna(0).astype(np.float32)

    # Add imputed feature flags
    for entry in provenance:
        if entry.get("is_imputed"):
            common[f"{entry['common_name']}_imputed"] = np.float32(0)  # CIC17 has these natively

    # Add label
    if "Label" in df.columns:
        common["label"] = (df["Label"] != "BENIGN").astype(np.int32)
        common["label_str"] = df["Label"]

    print(f"    CIC17 common track: {len(common.columns)} columns")
    return common


def build_common_track_unsw(df: pd.DataFrame, provenance: list) -> pd.DataFrame:
    """Extract common-track features from cleaned UNSW DataFrame (with derived features)."""
    proto_map = load_protocol_map()

    common = pd.DataFrame(index=df.index)

    # Map: common_name -> actual column in UNSW DataFrame (may be derived)
    for entry in provenance:
        common_name = entry["common_name"]
        unsw_src = entry["unsw_source"]

        # Determine actual column to use
        actual_col = None

        if unsw_src == "DERIVED":
            # Use the derived column (prefixed _derived_)
            derived_col = f"_derived_{common_name}"
            if derived_col in df.columns:
                actual_col = derived_col
            elif common_name in df.columns:
                actual_col = common_name
            else:
                # state_summary: use 'state' field
                if common_name == "state_summary" and "state" in df.columns:
                    actual_col = "state"  # Categorical, handled separately
                else:
                    print(f"    WARNING: No derived column found for UNSW '{common_name}'")
                    continue
        elif unsw_src is None:
            print(f"    WARNING: UNSW source is None for '{common_name}'")
            continue
        else:
            # Direct column mapping
            if unsw_src in df.columns:
                actual_col = unsw_src
            else:
                # Case-insensitive fallback
                cols_lower = {c.lower(): c for c in df.columns}
                if unsw_src.lower() in cols_lower:
                    actual_col = cols_lower[unsw_src.lower()]
                else:
                    print(f"    WARNING: UNSW column '{unsw_src}' not found for '{common_name}'")
                    continue

        # Handle special encodings
        if common_name == "protocol":
            if actual_col and actual_col in df.columns:
                common[common_name] = df[actual_col].apply(
                    lambda x: map_protocol_to_shared(x, unsw=True, proto_map=proto_map)
                ).astype(np.int32)
            else:
                common[common_name] = np.int32(0)

        elif common_name == "state_summary":
            if actual_col and actual_col in df.columns:
                # Use shared state vocabulary for one-hot encoding
                state_series = df[actual_col].fillna("unknown").astype(str)
                state_onehot = _build_state_onehot(state_series)
                for col in state_onehot.columns:
                    common[col] = state_onehot[col]
            else:
                # All zeros if state not available
                for st in _SHARED_STATES:
                    common[f"state_{st.lower()}"] = np.float32(0)

        elif common_name == "duration":
            # UNSW dur is already in seconds
            common[common_name] = pd.to_numeric(df[actual_col], errors="coerce").fillna(0).astype(np.float32)

        else:
            # Generic numeric feature
            common[common_name] = pd.to_numeric(
                df[actual_col], errors="coerce"
            ).fillna(0).astype(np.float32)

    # Add imputed feature flags
    for entry in provenance:
        if entry.get("is_imputed"):
            flag_col = f"{entry['common_name']}_imputed"
            if entry["unsw_source"] == "DERIVED":
                common[flag_col] = np.float32(1)  # UNSW has these imputed
            else:
                common[flag_col] = np.float32(0)

    # Add label
    label_col = "Label" if "Label" in df.columns else "label"
    if label_col in df.columns:
        common["label"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(np.int32)

    # Add attack category string if available
    if "attack_cat" in df.columns:
        common["attack_cat"] = df["attack_cat"].fillna("normal").astype(str)

    print(f"    UNSW common track: {len(common.columns)} columns")
    return common


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Preprocessing & Feature Harmonization"
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N rows per dataset for dev (laptop-safe)")
    parser.add_argument("--target", type=str, default="laptop",
                        choices=["laptop", "kaggle"],
                        help="Compute target")
    args = parser.parse_args()

    if args.target != "laptop":
        raise RuntimeError(
            "Preprocessing runs on laptop. Use --target laptop."
        )

    print("=" * 70)
    print("PHASE 2: PREPROCESSING & FEATURE HARMONIZATION")
    print(f"Sample: {args.sample or 'FULL'}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Seed: {SEED}")
    print("=" * 70)

    # Load provenance
    provenance = load_provenance()
    print(f"\nLoaded feature_provenance.json: {len(provenance)} features")

    # ── CIC-IDS2017 ─────────────────────────────────────────────────────
    print("\n" + "-" * 50)
    print("CIC-IDS2017")
    print("-" * 50)

    print("\n[1/5] Loading raw CSVs...")
    cic17 = load_cic17(sample=args.sample)

    print("\n[2/5] Deduplicating...")
    cic17 = deduplicate(cic17, label="CIC17")

    print("\n[3/5] Unit conversions...")
    cic17 = apply_cic17_conversions(cic17)

    print("\n[4/5] Building native track...")
    # Native track = all cleaned columns (no feature removal)
    cic17_native = cic17.copy()
    print(f"    CIC17 native: {cic17_native.shape[1]} columns, {cic17_native.shape[0]:,} rows")

    print("\n[5/5] Building common track...")
    cic17_common = build_common_track_cic17(cic17, provenance)

    # ── Save CIC17 ──────────────────────────────────────────────────────
    native_path = OUTPUT_DIR / "cic17_native.parquet"
    common_path = OUTPUT_DIR / "cic17_common.parquet"
    cic17_native.to_parquet(native_path, index=False)
    cic17_common.to_parquet(common_path, index=False)
    print(f"\n  -> Saved: {native_path} ({native_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  -> Saved: {common_path} ({common_path.stat().st_size / 1e6:.1f} MB)")

    # ── UNSW-NB15 ───────────────────────────────────────────────────────
    print("\n" + "-" * 50)
    print("UNSW-NB15")
    print("-" * 50)

    print("\n[1/5] Loading raw CSVs...")
    unsw = load_unsw(sample=args.sample)

    print("\n[2/5] Deduplicating...")
    unsw = deduplicate(unsw, label="UNSW")

    print("\n[3/5] Unit conversions + derived features...")
    unsw = apply_unsw_conversions(unsw)
    unsw = derive_unsw_features(unsw)

    print("\n[4/5] Building native track...")
    unsw_native = unsw.copy()
    # Convert remaining object columns to string for parquet compatibility
    # (sport, dsport, ct_ftp_cmd can be mixed-type)
    for col in unsw_native.columns:
        if unsw_native[col].dtype == object:
            unsw_native[col] = unsw_native[col].astype(str)
    print(f"    UNSW native: {unsw_native.shape[1]} columns, {unsw_native.shape[0]:,} rows")

    print("\n[5/5] Building common track...")
    unsw_common = build_common_track_unsw(unsw, provenance)

    # ── Save UNSW ───────────────────────────────────────────────────────
    native_path = OUTPUT_DIR / "unsw_native.parquet"
    common_path = OUTPUT_DIR / "unsw_common.parquet"
    unsw_native.to_parquet(native_path, index=False)
    unsw_common.to_parquet(common_path, index=False)
    print(f"\n  -> Saved: {native_path} ({native_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  -> Saved: {common_path} ({common_path.stat().st_size / 1e6:.1f} MB)")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print(f"  cic17_native.parquet: {cic17_native.shape[0]:,} rows × {cic17_native.shape[1]} cols")
    print(f"  cic17_common.parquet: {cic17_common.shape[0]:,} rows × {cic17_common.shape[1]} cols")
    print(f"  unsw_native.parquet:  {unsw_native.shape[0]:,} rows × {unsw_native.shape[1]} cols")
    print(f"  unsw_common.parquet:  {unsw_common.shape[0]:,} rows × {unsw_common.shape[1]} cols")
    print(f"\n  Next: Phase 3 — Baseline Snapshot Model")
    print("=" * 70)


if __name__ == "__main__":
    main()
