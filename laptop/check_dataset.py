import torch
import math
from pathlib import Path

processed_dir = Path(r"C:\Users\potato\Desktop\ids-v2 - Copy\laptop\processed\graphs")

def check_graph_list(filepath):
    print(f"Checking {filepath.name}...")
    if not filepath.exists():
        print("  [ERROR] File does not exist")
        return
    
    # Load first 10 windows to check
    graphs = torch.load(filepath, weights_only=False)
    print(f"  Total windows: {len(graphs)}")
    
    issues = []
    
    for i, g in enumerate(graphs[:10]):
        # 1. Check data types
        if g.edge_time.dtype != torch.float64:
            issues.append(f"Window {i}: edge_time is {g.edge_time.dtype}, expected float64")
        if g.edge_attr.dtype != torch.float32:
            issues.append(f"Window {i}: edge_attr is {g.edge_attr.dtype}, expected float32")
            
        # 2. Check bounds (clamping)
        vmin, vmax = g.edge_attr.min().item(), g.edge_attr.max().item()
        if vmin < -10.001 or vmax > 10.001:
            issues.append(f"Window {i}: edge_attr out of bounds [{vmin:.2f}, {vmax:.2f}]")
            
        # 3. Check node IDs local bounds
        max_node_id = g.edge_index.max().item() if g.edge_index.numel() > 0 else 0
        if max_node_id >= g.num_nodes:
            issues.append(f"Window {i}: max_node_id {max_node_id} >= num_nodes {g.num_nodes}")
            
        # 4. Check edge attr shape
        if g.edge_attr.shape[1] != 44:
            issues.append(f"Window {i}: edge_attr has {g.edge_attr.shape[1]} features, expected 44")
            
        # 5. Check NaN/Inf
        if not torch.isfinite(g.edge_attr).all():
            issues.append(f"Window {i}: NaN or Inf found in edge_attr")
            
    if issues:
        print("  [FAIL] Corruption detected:")
        for iss in issues[:10]:
            print(f"    - {iss}")
        if len(issues) > 10:
            print(f"    ... and {len(issues)-10} more.")
    else:
        print("  [PASS] First 10 windows pass all corruption checks.")

if __name__ == "__main__":
    check_graph_list(processed_dir / "NF-CICIDS2018_train_list.pt")
    check_graph_list(processed_dir / "NF-UNSW-NB15_train_list.pt")
    
    # Check scaler
    import pickle
    scaler_path = processed_dir.parent / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            print(f"Scaler: n_samples_seen_ = {scaler.n_samples_seen_}")
            print(f"Scaler: n_features_in_ = {scaler.n_features_in_}")
    else:
        print("Scaler not found.")
