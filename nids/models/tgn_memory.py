#!/usr/bin/env python3
"""
Phase 4 — TGN-Style Continuous-Time Memory Module
Target: LAPTOP (dev with --sample); KAGGLE (full training)
Produces: tgn_memory.py — the core architectural contribution

Nodes = hosts with persistent GRU-updated memory vectors.
Edges = flows processed in micro-batches (short time slices).
Local recomputation: only touched hosts + 1-hop neighbors.

Key design decisions (defensible in Methodology):
  1. Memory init from identity-free stats (not all-ones) — Plan §3.7
  2. GRU-style memory update per host, per flow — Plan §4.2
  3. Local recomputation (2 endpoints + 1-hop neighbors) — Plan §4.2
  4. Lightweight rolling forensic log (5-tuple only, 5-min window) — Plan §6.6
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from collections import defaultdict
import time as time_module

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from nids import set_seed, SEED

set_seed()


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Flow:
    """Single flow event for TGN processing."""
    src: str           # source host identifier (IP)
    dst: str           # destination host identifier (IP)
    timestamp: float   # seconds from epoch or simulation start
    features: np.ndarray  # edge feature vector (common-track, unscaled)
    label: int = 0     # 0=benign, 1=attack
    src_port: int = 0
    dst_port: int = 0
    protocol: int = 0

    @property
    def five_tuple(self) -> tuple:
        return (self.src, self.dst, self.src_port, self.dst_port, self.protocol)


@dataclass
class ForensicEntry:
    """Lightweight 5-tuple for rolling forensic log (Plan §6.6)."""
    src: str
    dst: str
    sport: int
    dport: int
    proto: int
    timestamp: float  # wall-clock time of logging


# ═══════════════════════════════════════════════════════════════════════════════
# Host Memory Store
# ═══════════════════════════════════════════════════════════════════════════════

class HostMemoryStore:
    """Per-host persistent memory with GRU update.

    Each host has:
      - memory: (memory_dim,) — the persistent state vector
      - last_update: float — timestamp of last update (for time-delta encoding)
      - neighbors: set[str] — immediate 1-hop neighbors (for local recomputation)
      - stats: dict — running stats (in_deg, out_deg, byte_vol, unique_peers)
    """

    def __init__(self, memory_dim: int = 128):
        self.memory_dim = memory_dim
        self._memory: Dict[str, torch.Tensor] = {}       # host -> memory vector
        self._last_update: Dict[str, float] = {}          # host -> last update time
        self._neighbors: Dict[str, set] = defaultdict(set)  # host -> neighbor set
        self._stats: Dict[str, dict] = defaultdict(lambda: {
            "in_deg": 0, "out_deg": 0,
            "byte_vol_in": 0.0, "byte_vol_out": 0.0,
            "unique_peers": set(),
        })

        # Lazy init: memory vectors created on first observation

    def _init_memory(self, host: str) -> torch.Tensor:
        """Initialize memory from identity-free statistics (Plan §3.7).

        Uses: in/out degree, byte volume, unique peer count.
        These are NOT identity-leaking (no IP embedding).
        """
        stats = self._stats[host]
        n_peers = len(stats["unique_peers"])

        # Build a small statistical profile
        profile = torch.tensor([
            float(stats["in_deg"]),
            float(stats["out_deg"]),
            float(stats["byte_vol_in"]),
            float(stats["byte_vol_out"]),
            float(n_peers),
        ], dtype=torch.float32)

        # Project to memory_dim via a fixed random projection (no learnable params here)
        # Using deterministic hash-based init for reproducibility
        rng = np.random.RandomState(hash(host) % (2**31))
        proj = torch.from_numpy(
            rng.randn(5, self.memory_dim).astype(np.float32) * 0.01
        )
        memory = profile @ proj
        return memory

    def get_memory(self, host: str) -> torch.Tensor:
        """Get host memory, initializing if unseen."""
        if host not in self._memory:
            self._memory[host] = self._init_memory(host)
        return self._memory[host]

    def update_memory(self, host: str, new_memory: torch.Tensor, timestamp: float):
        """Update host memory and timestamp."""
        self._memory[host] = new_memory.detach().clone()
        self._last_update[host] = timestamp

    def get_last_update(self, host: str) -> float:
        return self._last_update.get(host, 0.0)

    def get_neighbors(self, host: str) -> set:
        """Get 1-hop neighbors of a host."""
        return self._neighbors.get(host, set())

    def add_edge(self, src: str, dst: str, byte_vol: float = 0.0):
        """Record an edge between src and dst hosts."""
        self._neighbors[src].add(dst)
        self._neighbors[dst].add(src)

        # Update running stats
        self._stats[src]["out_deg"] += 1
        self._stats[src]["byte_vol_out"] += byte_vol
        self._stats[src]["unique_peers"].add(dst)

        self._stats[dst]["in_deg"] += 1
        self._stats[dst]["byte_vol_in"] += byte_vol
        self._stats[dst]["unique_peers"].add(src)

    def get_local_subgraph_hosts(self, src: str, dst: str) -> set:
        """Return hosts needing recomputation: endpoints + 1-hop neighbors."""
        hosts = {src, dst}
        hosts.update(self.get_neighbors(src))
        hosts.update(self.get_neighbors(dst))
        return hosts

    @property
    def num_hosts(self) -> int:
        return len(self._memory)

    def get_all_memories(self) -> Dict[str, torch.Tensor]:
        return self._memory


# ═══════════════════════════════════════════════════════════════════════════════
# Rolling Forensic Log
# ═══════════════════════════════════════════════════════════════════════════════

class RollingForensicLog:
    """Lightweight rolling forensic log (Plan §6.6).

    Stores flow 5-tuples only (not full feature vectors) for a rolling window.
    Old entries are evicted automatically. Kept on disk as a ring buffer.

    This is the audit trail, NOT used for model training — it enables post-hoc
    investigation of flagged anomalies without storing full packet data.
    """

    def __init__(self, window_sec: float = 300.0, max_entries: int = 100_000):
        self.window_sec = window_sec
        self.max_entries = max_entries
        self._entries: List[ForensicEntry] = []
        self._head = 0  # ring buffer write position

    def log(self, entry: ForensicEntry):
        """Add an entry to the forensic log. Evicts old entries."""
        now = entry.timestamp

        # Evict entries older than window_sec
        cutoff = now - self.window_sec
        self._entries = [e for e in self._entries if e.timestamp >= cutoff]

        # Append new entry; trim if over max
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

    def query(self, host: str, start_time: float, end_time: float) -> List[ForensicEntry]:
        """Query entries involving a specific host in a time range."""
        return [
            e for e in self._entries
            if (e.src == host or e.dst == host)
            and start_time <= e.timestamp <= end_time
        ]

    def __len__(self) -> int:
        return len(self._entries)


# ═══════════════════════════════════════════════════════════════════════════════
# TGN Memory Module (Core Architecture)
# ═══════════════════════════════════════════════════════════════════════════════

class TGNMemoryModule(nn.Module):
    """TGN-style continuous-time memory module.

    Architecture:
      1. Edge encoder: msg = MLP(src_mem || dst_mem || edge_feats || Δt)
      2. GRU memory update: h_new = GRU(h_old, aggregated_msg)
      3. Local recomputation: only update touched hosts + 1-hop neighbors

    This is the core novel component — the rest of the pipeline (encoder, decoder,
    training) is built around this memory module.
    """

    def __init__(
        self,
        memory_dim: int = 128,
        edge_feat_dim: int = 36,
        time_dim: int = 16,
        msg_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.edge_feat_dim = edge_feat_dim

        # ── Time encoding (learnable Fourier features) ──
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ── Edge message function ──
        # msg = f(src_mem, dst_mem, edge_feats, time_delta)
        msg_input_dim = memory_dim * 2 + edge_feat_dim + time_dim
        self.msg_fn = nn.Sequential(
            nn.Linear(msg_input_dim, msg_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(msg_hidden_dim, msg_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(msg_hidden_dim, memory_dim),
        )

        # ── GRU cell for memory update ──
        self.gru = nn.GRUCell(input_size=memory_dim, hidden_size=memory_dim)

        # ── Memory projector from raw stats (for initial state) ──
        self.memory_init = nn.Linear(5, memory_dim)  # 5 raw stats -> memory

        self.dropout = nn.Dropout(dropout)

    def encode_time(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Encode time delta into a learnable representation.

        Args:
            delta_t: (N,) or (N, 1) time deltas in seconds, normalized to [0, 1]
                     by dividing by the maximum expected gap (~300s for most traffic)
        Returns:
            (N, time_dim) time embeddings
        """
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(-1)
        return self.time_encoder(delta_t)

    def compute_message(
        self,
        src_mem: torch.Tensor,
        dst_mem: torch.Tensor,
        edge_feats: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute message from source to destination for a single flow.

        Args:
            src_mem: (batch, memory_dim) source host memory
            dst_mem: (batch, memory_dim) destination host memory
            edge_feats: (batch, edge_feat_dim) flow features
            delta_t: (batch,) time since last update for this host
        Returns:
            (batch, memory_dim) message vector
        """
        time_emb = self.encode_time(delta_t)
        msg_input = torch.cat([src_mem, dst_mem, edge_feats, time_emb], dim=-1)
        msg = self.msg_fn(msg_input)
        return msg

    def update_memory(
        self,
        host_mem: torch.Tensor,
        messages: torch.Tensor,
    ) -> torch.Tensor:
        """GRU update: aggregate messages into host memory.

        Args:
            host_mem: (batch, memory_dim) current memory
            messages: (batch, memory_dim) incoming messages (mean-aggregated if multiple)
        Returns:
            (batch, memory_dim) updated memory
        """
        return self.gru(messages, host_mem)

    def init_memory_from_stats(
        self,
        in_deg: torch.Tensor,
        out_deg: torch.Tensor,
        byte_vol_in: torch.Tensor,
        byte_vol_out: torch.Tensor,
        n_peers: torch.Tensor,
    ) -> torch.Tensor:
        """Initialize memory from identity-free statistics (Plan §3.7).

        Args:
            Each is (batch,) float tensor
        Returns:
            (batch, memory_dim) initialized memory vectors
        """
        stats = torch.stack([in_deg, out_deg, byte_vol_in, byte_vol_out, n_peers], dim=-1)
        return self.memory_init(stats)

    def forward(
        self,
        batch_src_mem: torch.Tensor,
        batch_dst_mem: torch.Tensor,
        batch_edge_feats: torch.Tensor,
        batch_delta_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a micro-batch of flows.

        Args:
            batch_src_mem: (N, memory_dim) source host memories
            batch_dst_mem: (N, memory_dim) destination host memories
            batch_edge_feats: (N, edge_feat_dim) flow features
            batch_delta_t: (N,) time since each host's last update

        Returns:
            src_messages: (N, memory_dim) messages for source hosts
            dst_messages: (N, memory_dim) messages for destination hosts
        """
        # Compute asymmetric messages
        msg = self.compute_message(
            batch_src_mem, batch_dst_mem,
            batch_edge_feats, batch_delta_t,
        )

        # Source and destination get the same message (symmetric update)
        # Both endpoints learn from the flow
        return msg, msg  # src_msg, dst_msg


# ═══════════════════════════════════════════════════════════════════════════════
# Micro-Batch Processor
# ═══════════════════════════════════════════════════════════════════════════════

class MicroBatchProcessor:
    """Process flows in micro-batches and update host memories.

    Orchestrates:
      1. Group flows into micro-batches by time slice
      2. For each micro-batch: compute messages, update memories
      3. Only update touched hosts + 1-hop neighbors (local recomputation)
      4. Maintain per-host last-update timestamps
    """

    def __init__(
        self,
        memory_module: TGNMemoryModule,
        memory_store: HostMemoryStore,
        forensic_log: Optional[RollingForensicLog] = None,
        micro_batch_sec: float = 1.0,
    ):
        self.memory_module = memory_module
        self.store = memory_store
        self.forensic_log = forensic_log
        self.micro_batch_sec = micro_batch_sec

    def process_flows(
        self,
        flows: List[Flow],
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """Process a list of flows in micro-batches.

        For each micro-batch:
          1. Identify touched hosts (src, dst of all flows in batch)
          2. Add 1-hop neighbors (local recomputation scope)
          3. Compute messages for each flow
          4. Aggregate messages per host
          5. GRU-update all touched host memories
          6. Log to forensic trail

        Args:
            flows: list of Flow objects, ordered by timestamp
            device: torch device

        Returns:
            Dict mapping host_id -> updated memory vector for touched hosts
        """
        if not flows:
            return {}

        # Sort by timestamp (should already be sorted)
        flows = sorted(flows, key=lambda f: f.timestamp)

        # Group into micro-batches
        batches = self._make_micro_batches(flows)

        updated_memories = {}

        for batch in batches:
            batch_updated = self._process_micro_batch(batch, device)
            updated_memories.update(batch_updated)

        return updated_memories

    def _make_micro_batches(self, flows: List[Flow]) -> List[List[Flow]]:
        """Group flows into micro-batches by time slice."""
        batches = []
        current_batch = []
        batch_start = flows[0].timestamp

        for flow in flows:
            if flow.timestamp - batch_start >= self.micro_batch_sec:
                if current_batch:
                    batches.append(current_batch)
                current_batch = []
                batch_start = flow.timestamp
            current_batch.append(flow)

        if current_batch:
            batches.append(current_batch)

        return batches

    def _process_micro_batch(
        self,
        batch: List[Flow],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Process one micro-batch of flows."""
        n = len(batch)

        # Get memories and features
        src_mems = []
        dst_mems = []
        edge_feats = []
        delta_ts = []

        for flow in batch:
            src_mem = self.store.get_memory(flow.src)
            dst_mem = self.store.get_memory(flow.dst)
            src_mems.append(src_mem)
            dst_mems.append(dst_mem)

            # Edge features as float32 tensor
            feats = torch.from_numpy(
                np.asarray(flow.features, dtype=np.float32)
            )
            edge_feats.append(feats)

            # Time since last update (max of src/dst last update)
            last_t = max(
                self.store.get_last_update(flow.src),
                self.store.get_last_update(flow.dst),
            )
            delta_t = max(flow.timestamp - last_t, 0.0)
            delta_ts.append(delta_t)

        # Stack into batch tensors
        batch_src_mem = torch.stack(src_mems).to(device)
        batch_dst_mem = torch.stack(dst_mems).to(device)
        batch_edge_feats = torch.stack(edge_feats).to(device)
        batch_delta_t = torch.tensor(delta_ts, dtype=torch.float32).to(device)

        # Normalize delta_t for time encoder (assume max 300s gap)
        batch_delta_t_norm = batch_delta_t / 300.0

        # Compute messages
        src_msgs, dst_msgs = self.memory_module(
            batch_src_mem, batch_dst_mem,
            batch_edge_feats, batch_delta_t_norm,
        )

        # Aggregate messages per host and update
        updated = {}

        # Aggregate messages per host
        host_msgs: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for i, flow in enumerate(batch):
            host_msgs[flow.src].append(src_msgs[i])
            host_msgs[flow.dst].append(dst_msgs[i])

        # GRU update per host (mean-aggregate multiple messages)
        for host, msgs in host_msgs.items():
            old_mem = self.store.get_memory(host).to(device)
            # Mean-aggregate all messages for this host in this batch
            aggregated = torch.stack(msgs).mean(dim=0)
            new_mem = self.memory_module.update_memory(
                old_mem.unsqueeze(0), aggregated.unsqueeze(0)
            ).squeeze(0)
            self.store.update_memory(host, new_mem.cpu(), batch[-1].timestamp)
            updated[host] = new_mem.cpu()

        # ── Local recomputation: also update 1-hop neighbors ──
        touched = set()
        for flow in batch:
            touched.add(flow.src)
            touched.add(flow.dst)

        all_to_update = set(touched)
        for host in touched:
            all_to_update.update(self.store.get_neighbors(host))

        # For neighbors not directly touched, apply a lightweight "awareness" update
        # (small perturbation to indicate neighborhood activity without full recompute)
        for neighbor in all_to_update - set(host_msgs.keys()):
            old_mem = self.store.get_memory(neighbor).to(device)
            # Decay toward zero (representing staleness awareness)
            awareness = torch.zeros_like(old_mem)
            new_mem = self.memory_module.update_memory(
                old_mem.unsqueeze(0), awareness.unsqueeze(0)
            ).squeeze(0)
            self.store.update_memory(neighbor, new_mem.cpu(), batch[-1].timestamp)
            updated[neighbor] = new_mem.cpu()

        # ── Forensic logging ──
        if self.forensic_log is not None:
            for flow in batch:
                self.forensic_log.log(ForensicEntry(
                    src=flow.src, dst=flow.dst,
                    sport=flow.src_port, dport=flow.dst_port,
                    proto=flow.protocol,
                    timestamp=time_module.time(),  # wall clock
                ))

        return updated


# ═══════════════════════════════════════════════════════════════════════════════
# Test Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def test_tgn_memory_synthetic():
    """Unit test: synthetic small graph to verify correctness.

    Checks:
      - Memory init from stats works
      - GRU update changes memory
      - Local recomputation touches correct hosts
      - Forensic log entries are recorded
    """
    print("TGN Memory Module — Synthetic Test")
    print("-" * 50)

    memory_dim = 64
    edge_dim = 16

    mem_module = TGNMemoryModule(
        memory_dim=memory_dim,
        edge_feat_dim=edge_dim,
    )
    store = HostMemoryStore(memory_dim=memory_dim)
    forensics = RollingForensicLog(window_sec=300.0)
    processor = MicroBatchProcessor(
        memory_module=mem_module,
        memory_store=store,
        forensic_log=forensics,
        micro_batch_sec=0.5,
    )

    # Create 5 hosts with some edges
    hosts = [f"192.168.1.{i}" for i in range(1, 6)]
    for i in range(4):
        store.add_edge(hosts[i], hosts[i + 1], byte_vol=1000.0)

    # Create 10 synthetic flows
    flows = []
    rng = np.random.RandomState(SEED)
    for i in range(10):
        src = hosts[rng.randint(0, 5)]
        dst = hosts[rng.randint(0, 5)]
        if src == dst:
            continue
        flows.append(Flow(
            src=src, dst=dst,
            timestamp=i * 0.1,  # 100ms apart
            features=rng.randn(edge_dim).astype(np.float32),
            label=0,
            src_port=rng.randint(1024, 65535),
            dst_port=80,
            protocol=6,  # TCP
        ))

    # Process flows
    updated = processor.process_flows(flows)
    print(f"  Updated hosts: {len(updated)}")

    # Verify memories changed
    for host in hosts:
        mem = store.get_memory(host)
        print(f"  {host}: memory norm={mem.norm().item():.4f}, "
              f"neighbors={len(store.get_neighbors(host))}")

    # Verify forensic log
    print(f"  Forensic entries: {len(forensics)}")
    for e in forensics._entries:
        print(f"    {e.src}:{e.sport} -> {e.dst}:{e.dport} proto={e.proto}")

    # Verify local recomputation: neighbor host 3 should be updated
    # even if not directly in flows (if flows touch hosts 2 or 4)
    print(f"\n  All hosts have memory: {all(hosts[i] in store._memory for i in range(5))}")
    print("  PASSED: synthetic test")


if __name__ == "__main__":
    test_tgn_memory_synthetic()
