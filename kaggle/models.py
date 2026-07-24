import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import degree

class Time2Vec(nn.Module):
    """
    Time2Vec captures periodic and linear temporal patterns.
    Reference: Kazemi et al. (2019) 'Time2Vec: Learning a Vector Representation of Time'
    """
    def __init__(self, k=16):
        super().__init__()
        self.k = k
        self.w0 = nn.Parameter(torch.randn(1) * 0.1)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.omega = nn.Parameter(10.0 ** (torch.rand(k) * 6 - 3))
        self.bias = nn.Parameter(torch.zeros(k))
        self.output_dim = k + 1

    def forward(self, t):
        if t.dim() == 1: t = t.unsqueeze(-1)
        return torch.cat([self.w0 * t + self.b0, torch.sin(self.omega * t + self.bias)], dim=-1)

class EGATv2Encoder(nn.Module):
    """
    Edge-Featured GATv2 Encoder with Centrality (Degree) Encoding.
    State-of-the-art graph transformers (e.g. Graphormer by Ying et al. 2021) 
    utilize degree encoding to provide strong structural priors to anonymous nodes.
    """
    def __init__(self, edge_dim=58, node_init=128, hidden=256, heads=8, layers=3, d_attn=0.3, d_feat=0.2, return_attention=False):
        super().__init__()
        self.output_dim = hidden * 3
        self.return_attention = return_attention
        
        # Learnable Centrality/Degree Encoding (handles up to degree 500)
        self.deg_emb = nn.Embedding(500, node_init)
        nn.init.constant_(self.deg_emb.weight, 1.0)  # Prevents random collapse of pretrained weights
        self.edge_proj = nn.Linear(edge_dim, hidden)
        
        self.convs = nn.ModuleList([
            GATv2Conv((-1, -1), hidden // heads, heads=heads, edge_dim=hidden, dropout=d_attn, concat=True) 
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(d_feat)
        self.activation = nn.ELU()

    def forward(self, data, return_attn=False):
        # Compute node degrees dynamically on the fly
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes).long()
        deg = torch.clamp(deg, 0, 499)
        x = self.deg_emb(deg)
        
        ea = self.edge_proj(data.edge_attr)
        all_attn = []
        for conv, norm in zip(self.convs, self.norms):
            if return_attn or self.return_attention:
                x_new, attn = conv(x, data.edge_index, edge_attr=ea, return_attention_weights=True)
                all_attn.append(attn)
            else:
                x_new = conv(x, data.edge_index, edge_attr=ea)
            x_new = self.dropout(self.activation(x_new))
            x = norm(x + x_new) if x.shape == x_new.shape else norm(x_new)
            
        out = torch.cat([x[data.edge_index[0]], x[data.edge_index[1]], ea], dim=-1)
        return (out, all_attn) if (return_attn or self.return_attention) else out

class ResNetBlock(nn.Module):
    """
    Standard ResNet block for tabular/network feature vectors.
    Reference: Gorishniy et al. (2021) 'Revisiting Deep Learning Models for Tabular Data'
    """
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.block(x)

class ClassifierHead(nn.Module):
    """
    Deep Residual MLP Head for classification (Binary or Multiclass).
    Consolidates previously duplicate BinaryHead and MulticlassHead into one configurable SOTA module.
    """
    def __init__(self, in_dim=768, hidden=512, out_dim=2, num_blocks=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResNetBlock(hidden, dropout) for _ in range(num_blocks)])
        self.out = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        return self.out(x)

class FocalLoss(nn.Module):
    """
    Focal Loss for handling severe class imbalance.
    Reference: Lin et al. (2017) 'Focal Loss for Dense Object Detection'
    """
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal = self.alpha[targets] * focal
        return focal.mean()
