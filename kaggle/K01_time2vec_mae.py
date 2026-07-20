"""
K01 — Time2Vec + E-GATv2 Encoder + MAE Pretraining (Stages B/C/D)
===================================================================
KAGGLE T4x2 GPU. Copy-paste this entire script into a Kaggle notebook cell,
or upload as a dataset and run as: !python K01_time2vec_mae.py

Inputs (upload to Kaggle dataset):
  - dataset/graphs/{dataset}_{split}_list.pt (from L04)
  - dataset/graphs/scaler.pkl (from L04)
  - label_map.yaml, feature_manifest.yaml

Outputs:
  - /kaggle/working/checkpoints/D_mae_pretrain/best.pt + config.json
  - /kaggle/working/outputs/figures/fig03-05
  - /kaggle/working/logs/k01_log.json
"""

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np, yaml, json, pickle, random, os, gc
from datetime import datetime, timezone; from pathlib import Path
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== CONFIG =====
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# Paths — ADJUST to your Kaggle dataset name
INPUT_DIR  = Path('/kaggle/input/ids-processed')  # your uploaded dataset
WORKING    = Path('/kaggle/working')
CKPT_DIR   = WORKING / 'checkpoints' / 'D_mae_pretrain'
FIGS_DIR   = WORKING / 'outputs' / 'figures'
LOGS_DIR   = WORKING / 'logs'
for d in [CKPT_DIR, FIGS_DIR, LOGS_DIR]: d.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}"); print(f"PyG: {torch_geometric.__version__}")

# Load manifests
with open(INPUT_DIR/'feature_manifest.yaml') as f: fm=yaml.safe_load(f)
with open(INPUT_DIR/'label_map.yaml') as f: lm=yaml.safe_load(f)
EDGE_DIM = fm['final_edge_input_dim']  # 61
KEPT = fm['kept_features']; RAW_DIM = len(KEPT)
TV_DIM = fm['time2vec_dim']  # 17
assert EDGE_DIM == RAW_DIM + TV_DIM, f"Dim mismatch: {EDGE_DIM} != {RAW_DIM}+{TV_DIM}"

# ===== MODEL DEFINITIONS =====
class Time2Vec(nn.Module):
    def __init__(self,k=16):
        super().__init__(); self.k=k
        self.w0=nn.Parameter(torch.randn(1)*0.1); self.b0=nn.Parameter(torch.zeros(1))
        self.omega=nn.Parameter(10.0**(torch.rand(k)*6-3)); self.bias=nn.Parameter(torch.zeros(k))
        self.output_dim=k+1
    def forward(self,t):
        if t.dim()==1:t=t.unsqueeze(-1)
        return torch.cat([self.w0*t+self.b0,torch.sin(self.omega*t+self.bias)],dim=-1)

class EGATv2Encoder(nn.Module):
    def __init__(self,edge_dim=61,node_init_dim=128,hidden_dim=256,num_heads=8,num_layers=3,
                 dropout_attn=0.3,dropout_feat=0.2):
        super().__init__()
        self.hidden_dim=hidden_dim; self.num_heads=num_heads; self.num_layers=num_layers
        self.output_dim=hidden_dim*3; self.node_init_dim=node_init_dim; self.node_embed=None
        self.edge_proj=nn.Linear(edge_dim,hidden_dim)
        self.convs=nn.ModuleList(); self.norms=nn.ModuleList()
        self.dropout=nn.Dropout(dropout_feat)
        for _ in range(num_layers):
            self.convs.append(GATv2Conv((-1,-1),hidden_dim//num_heads,heads=num_heads,
                edge_dim=hidden_dim,dropout=dropout_attn,concat=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.activation=nn.ELU()
    def _get_node_embed(self,num_nodes,device):
        if self.node_embed is None or self.node_embed.shape[0]<num_nodes:
            new=nn.Parameter(torch.randn(num_nodes,self.node_init_dim,device=device)*0.1)
            if self.node_embed is not None: new.data[:self.node_embed.shape[0]]=self.node_embed.data
            self.node_embed=new
        return self.node_embed[:num_nodes]
    def forward(self,data):
        x=self._get_node_embed(data.num_nodes,data.edge_index.device)
        e=self.edge_proj(data.edge_attr)
        for conv,norm in zip(self.convs,self.norms):
            xn,_=conv(x,data.edge_index,edge_attr=e,return_attention_weights=True)
            xn=self.activation(xn); xn=self.dropout(xn); xn=norm(xn)
            x=x+xn if x.shape==xn.shape else xn
        return torch.cat([x[data.edge_index[0]],x[data.edge_index[1]],e],dim=-1)

class MAEDecoder(nn.Module):
    def __init__(self,input_dim=768,hidden_dim=256,output_dim=61,bottleneck_dim=128):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),nn.ELU(),nn.Dropout(0.1),
            nn.Linear(hidden_dim,bottleneck_dim),nn.ELU(),nn.Dropout(0.1),
            nn.Linear(bottleneck_dim,output_dim))
    def forward(self,x): return self.net(x)

# ===== DATA LOADING =====
print("\nLoading graphs...")
G_train=[]; G_val=[]
for ds in ['NF-CICIDS2018','NF-UNSW-NB15']:
    for sp, lst in [('train',G_train),('val',G_val)]:
        p=INPUT_DIR/f'graphs'/{ds}_{sp}_list.pt' if (INPUT_DIR/'graphs').exists() else INPUT_DIR/f'{ds}_{sp}_list.pt'
        if p.exists():
            gs=torch.load(p,weights_only=False); lst.extend(gs)
            print(f"  {ds}_{sp}: {len(gs)} windows")

all_times=torch.cat([g.edge_time for g in G_train])
T_MIN,T_MAX=all_times.min().item(),all_times.max().item()

# Build benign-only combined graph
print("Building benign training graph...")
all_ei=[]; all_ea=[]; all_et=[]; off=0
for g in G_train:
    bm=g.y_binary==0
    if bm.sum()==0: continue
    ei=g.edge_index[:,bm]+off; off=ei.max().item()+1
    all_ei.append(ei); all_ea.append(g.edge_attr[bm]); all_et.append(g.edge_time[bm])
bg=Data(edge_index=torch.cat(all_ei,dim=1),edge_attr=torch.cat(all_ea,dim=0),
        edge_time=torch.cat(all_et,dim=0),num_nodes=off).to(device)
print(f"Benign graph: {bg.num_nodes:,} nodes, {bg.edge_index.shape[1]:,} edges")

# Feature bounds for FGSM
all_attr=torch.cat([g.edge_attr for g in G_train])
fmins=all_attr.min(dim=0).values.to(device); fmaxs=all_attr.max(dim=0).values.to(device)
fmins61=torch.cat([fmins,torch.full((17,),-4.0,device=device)])
fmaxs61=torch.cat([fmaxs,torch.full((17,),4.0,device=device)])

# ===== TRAINING SETUP =====
HP={'mask_ratio':0.40,'fgsm_eps':0.02,'lr':1e-3,'wd':1e-5,'epochs':30,'bs':4096,'patience':5}
H=256; HEADS=8; LAYERS=3; FANOUT=[15,10,5]

t2v=Time2Vec(k=16).to(device); enc=EGATv2Encoder(edge_dim=EDGE_DIM,hidden_dim=H,num_heads=HEADS,num_layers=LAYERS).to(device)
dec=MAEDecoder(input_dim=enc.output_dim,hidden_dim=H,output_dim=EDGE_DIM).to(device)
opt=optim.AdamW(list(t2v.parameters())+list(enc.parameters())+list(dec.parameters()),lr=HP['lr'],weight_decay=HP['wd'])
sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=HP['epochs'])
amp=GradScaler()
print(f"Params — T2V:{sum(p.numel() for p in t2v.parameters()):,} "
      f"Enc:{sum(p.numel() for p in enc.parameters()):,} "
      f"Dec:{sum(p.numel() for p in dec.parameters()):,}")

# ===== HELPERS =====
def norm_t(t): return (t-T_MIN)/(T_MAX-T_MIN)
def fgsm(ea,eps):
    ea=ea.clone().detach().requires_grad_(True)
    g=torch.autograd.grad(F.mse_loss(ea,torch.randn_like(ea)),ea)[0]
    p=ea.detach()+eps*g.sign()
    return torch.clamp(p,fmins61,fmaxs61)
def mask_edges(ea,r=0.4):
    m=torch.rand(*ea.shape,device=ea.device)<r
    ea_m=ea.clone(); ea_m[m]=0.0
    return ea_m,m

from torch_geometric.loader import NeighborLoader
loader=NeighborLoader(bg,num_neighbors=FANOUT,batch_size=HP['bs'],shuffle=True,num_workers=2)

# ===== TRAINING LOOP =====
print(f"\nTraining {HP['epochs']} epochs...")
best_vl=float('inf'); pc=0; tl=[]; vl=[]
for ep in range(HP['epochs']):
    t2v.train(); enc.train(); dec.train()
    el=0.0; nb=0
    for b in loader:
        b=b.to(device)
        tn=norm_t(b.edge_time); te=t2v(tn)
        ef=torch.cat([b.edge_attr,te],dim=-1)
        ef=fgsm(ef,HP['fgsm_eps'])
        ef_m,m=mask_edges(ef,HP['mask_ratio'])
        bd=Data(edge_index=b.edge_index,edge_attr=ef_m,num_nodes=b.num_nodes)
        with autocast():
            fr=enc(bd); rc=dec(fr)
            loss=F.mse_loss(rc[m],ef[m])
        opt.zero_grad(); amp.scale(loss).backward(); amp.step(opt); amp.update()
        el+=loss.item(); nb+=1
    sch.step(); tl.append(el/max(nb,1))

    # Val
    t2v.eval(); enc.eval(); dec.eval(); vl_=0.0; nv=0
    with torch.no_grad():
        for gv in G_val[:5]:
            gv=gv.to(device)
            if gv.edge_index.shape[1]>5000:
                ix=torch.randperm(gv.edge_index.shape[1])[:5000]
                gv.edge_index=gv.edge_index[:,ix]; gv.edge_attr=gv.edge_attr[ix]; gv.edge_time=gv.edge_time[ix]
            tn=norm_t(gv.edge_time); te=t2v(tn)
            ef=torch.cat([gv.edge_attr,te],dim=-1)
            ef_m,m=mask_edges(ef,HP['mask_ratio'])
            fr=enc(Data(edge_index=gv.edge_index,edge_attr=ef_m,num_nodes=gv.num_nodes))
            rc=dec(fr); vl_+=F.mse_loss(rc[m],ef[m]).item(); nv+=1
    va=vl_/max(nv,1); vl.append(va)

    print(f"Ep {ep+1:2d}/{HP['epochs']}: Tr={tl[-1]:.6f} Val={va:.6f} LR={opt.param_groups[0]['lr']:.2e}")
    if va<best_vl:
        best_vl=va; pc=0
        torch.save({'epoch':ep+1,'t2v':t2v.state_dict(),'enc':enc.state_dict(),'dec':dec.state_dict(),
                    'val_loss':va,'config':HP},CKPT_DIR/'best.pt')
        json.dump(HP,open(CKPT_DIR/'config.json','w'),indent=2)
        print(f"  ✓ Saved")
    else:
        pc+=1
        if pc>=HP['patience']: print(f"Early stop ep {ep+1}"); break
    if np.isnan(tl[-1]): print(f"NaN! Reducing LR"); opt.param_groups[0]['lr']*=0.1; pc=0

# ===== SAVE FIGURES =====
fig,ax=plt.subplots(figsize=(10,4))
ax.plot(range(1,len(tl)+1),tl,'b-',label='Train'); ax.plot(range(1,len(vl)+1),vl,'r-',label='Val')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('MAE Pretraining Loss')
plt.tight_layout(); plt.savefig(FIGS_DIR/'fig16_k01_mae_loss.png',dpi=150)
plt.close(); print(f"\n✓ K01 complete. Best val loss: {best_vl:.6f}")

# Save log
json.dump({'notebook':'K01','best_val_loss':float(best_vl),'epochs':len(tl),'hp':HP,
           'start':datetime.now(timezone.utc).isoformat()},open(LOGS_DIR/'k01_log.json','w'),indent=2,default=str)
print("Log saved. Download /kaggle/working/checkpoints/D_mae_pretrain/best.pt for K02.")
