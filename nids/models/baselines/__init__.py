"""Baseline models for comparison against the TGN architecture.

E-GraphSAGE snapshot baseline (Phase 3): fast reference for justifying
the temporal architecture switch.
"""
from nids.models.baselines.snapshot_baseline import (
    EdgeGraphSAGEEncoder,
    build_snapshot_graphs,
)

from nids.models.tgn_memory import (
    TGNMemoryModule,
    HostMemoryStore,
    MicroBatchProcessor,
    RollingForensicLog,
    Flow,
    ForensicEntry,
)
