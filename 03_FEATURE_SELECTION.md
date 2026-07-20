# Feature Selection — NF3 Schema (53 raw fields)

## Rule for the implementing AI

Do not use all 53 NF3 fields as classifier input. Apply the manifest below exactly. If a field name in the actual downloaded CSV doesn't match this list (schema drift between UQ eSpace versions), flag it and stop rather than guessing — do not silently include/exclude an unrecognized field.

## Dropped fields (9) — never enter the classifier, some are used elsewhere

| Field | Reason for exclusion |
|---|---|
| IPV4_SRC_ADDR | Direct identity leakage — model would memorize attacker IPs instead of learning behavior. Still used as a **node identity** for graph construction (Stage A), never as a classifier feature. |
| IPV4_DST_ADDR | Same as above — node identity only. |
| L4_SRC_PORT | High-cardinality, near-identity field for some attack types (e.g. specific C2 ports); documented exclusion practice in prior NF-based NIDS studies. Used for edge typing only, not raw feature. |
| L4_DST_PORT | Same as above. |
| L7_PROTO | Documented in NF-dataset literature as excluded — L7 protocol tagging is sometimes near-unique to a class, so keeping it lets the model shortcut on protocol identity instead of behavior. |
| FTP_COMMAND_RET_CODE | Mostly NaN outside FTP flows; near-constant/sparse, contributes noise more than signal at this scale. |
| DNS_QUERY_ID | Effectively a random identifier per query, no behavioral signal, risk of spurious correlation with specific capture sessions. |
| FLOW_START_MILLISECONDS | Not fed as a raw classifier feature — consumed exclusively by Stage B (Time2Vec) and Stage A (chronological split/graph edge ordering). Feeding the raw absolute timestamp into the classifier as well would double-count temporal signal and risk the model learning "which capture session" instead of "what behavior." |
| FLOW_END_MILLISECONDS | Same reasoning — duration is already captured via FLOW_DURATION_MILLISECONDS; the absolute end timestamp adds no behavioral information beyond what duration + start already provide. |

## Kept fields (44) — the actual classifier input, grouped

**Volume (4):** IN_BYTES, IN_PKTS, OUT_BYTES, OUT_PKTS

**Protocol/flags (4):** PROTOCOL, TCP_FLAGS, CLIENT_TCP_FLAGS, SERVER_TCP_FLAGS

**Duration (3):** FLOW_DURATION_MILLISECONDS, DURATION_IN, DURATION_OUT

**TTL (2):** MIN_TTL, MAX_TTL

**Packet size (4):** LONGEST_FLOW_PKT, SHORTEST_FLOW_PKT, MIN_IP_PKT_LEN, MAX_IP_PKT_LEN

**Per-second throughput (2):** SRC_TO_DST_SECOND_BYTES, DST_TO_SRC_SECOND_BYTES

**Retransmission (4):** RETRANSMITTED_IN_BYTES, RETRANSMITTED_IN_PKTS, RETRANSMITTED_OUT_BYTES, RETRANSMITTED_OUT_PKTS

**Average throughput (2):** SRC_TO_DST_AVG_THROUGHPUT, DST_TO_SRC_AVG_THROUGHPUT

**Packet size histogram (5):** NUM_PKTS_UP_TO_128_BYTES, NUM_PKTS_128_TO_256_BYTES, NUM_PKTS_256_TO_512_BYTES, NUM_PKTS_512_TO_1024_BYTES, NUM_PKTS_1024_TO_1514_BYTES

**TCP window (2):** TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT

**ICMP (2):** ICMP_TYPE, ICMP_IPV4_TYPE

**DNS (2):** DNS_QUERY_TYPE, DNS_TTL_ANSWER

**Inter-arrival time, NF3-exclusive (8):** SRC_TO_DST_IAT_MIN, SRC_TO_DST_IAT_MAX, SRC_TO_DST_IAT_AVG, SRC_TO_DST_IAT_STDDEV, DST_TO_SRC_IAT_MIN, DST_TO_SRC_IAT_MAX, DST_TO_SRC_IAT_AVG, DST_TO_SRC_IAT_STDDEV — these are the reason NF3 was chosen over NF-v2; keep all 8, they're the primary burst/timing signal short of full Time2Vec.

**Total: 44 features → z-score normalized (Stage A, train-fit scaler) → concatenated with 17-dim Time2Vec output (Stage B) = 61-dim edge input... correction: architecture doc specifies 53 raw + 17 Time2Vec = 70-dim. Reconcile: use 44 kept raw fields, NOT 53, as the "raw" component. Edge input dimension is therefore 44 + 17 = 61-dim, not 70-dim. This supersedes the 70-dim figure in `02_ARCHITECTURE.md` Stage B/C — update E-GATv2's edge feature projection layer to `Linear(61 → 256)` instead of `Linear(70 → 256)` when implementing.**

## Handling missing/protocol-conditional fields

ICMP_TYPE, ICMP_IPV4_TYPE, DNS_QUERY_TYPE, DNS_TTL_ANSWER are protocol-conditional (only meaningful for ICMP/DNS flows). For non-applicable flows: fill with 0, not NaN, not -1. Add no separate "is_missing" indicator column — the earlier `flag_missing` feature caused a dataset-identity leakage bug in prior work on this project (per project history); do not reintroduce that pattern.

## Correlation pruning (optional, run once, document result)

Before finalizing, compute pairwise Pearson correlation across the 44 kept features on E_train only. If any pair exceeds |r| > 0.95, drop the less-interpretable of the two (prefer keeping the one with a clearer SHAP-readable name for the Stage J XAI section) and document the drop in `feature_manifest.yaml`. Do not drop more than 3-4 features this way — over-pruning removes the redundancy that guards against a single feature dominating attention weights.

## Output artifact

Produce `feature_manifest.yaml`:
```yaml
dropped_fields: [IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_SRC_PORT, L4_DST_PORT, L7_PROTO, FTP_COMMAND_RET_CODE, DNS_QUERY_ID, FLOW_START_MILLISECONDS, FLOW_END_MILLISECONDS]
node_identity_fields: [IPV4_SRC_ADDR, IPV4_DST_ADDR]
edge_typing_fields: [L4_SRC_PORT, L4_DST_PORT, PROTOCOL]
time_signal_field: FLOW_START_MILLISECONDS  # consumed by Stage B only
kept_features: [ <44 fields listed above> ]
correlation_pruned: []  # filled in after running the correlation check, max 3-4 entries
final_raw_feature_count: 44
time2vec_dim: 17
final_edge_input_dim: 61
```
This file is loaded identically by every notebook — same discipline as `label_map.yaml`.
