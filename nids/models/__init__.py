"""GNN models: TGN memory (Phase 4), baseline snapshot (Phase 3), encoder/decoder."""
from nids.models.tgn_memory import (
    TGNMemoryModule,
    HostMemoryStore,
    MicroBatchProcessor,
    RollingForensicLog,
    Flow,
    ForensicEntry,
)

from nids.models.baselines.snapshot_baseline import (
    EdgeGraphSAGEEncoder,
    build_snapshot_graphs,
)
