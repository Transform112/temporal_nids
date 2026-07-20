"""
K03 — Binary Classification Stage-1 Head (Stage F)
=====================================================
KAGGLE T4x2. Loads K01 encoder, trains binary head with PGD adversarial training.
"""

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np, yaml, json, random, gc
from datetime import datetime, timezone; from pathlib import Path
from torch_geometric.nn import GATv2Conv; from torch_geometric.data import Data
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.deterministic=True

INPUT_DIR=Path('/kaggle/input/ids-processed'); WORKING=Path('/kaggle/working')
CKPT_DIR=WORKING/'checkpoints'/'F_binary'; LOGS_DIR=WORKING/'logs'; FIGS_DIR=WORKING/'outputs'/'figures'
for d in [CKPT_DIR,LOGS_DIR,FIGS_DIR]: d.mkdir(parents=True,exist_ok=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(INPUT_DIR/'feature_manifest.yaml') as f: fm=yaml.safe_load(f)
with open(INPUT_DIR/'label_map.yaml') as f: lm=yaml.safe_load(f)
EDGE_DIM=fm['final_edge_input_dim']; UNIFIED=lm['unified_classes']; NC=len(UNIFIED)

# ---- Models ----
class Time2Vec(nn.Module):
    def __init__(self,k=16):
        super().__init__(); self.k=k; self.w0=nn.Parameter(torch.randn(1)*0.1); self.b0=nn.Parameter(torch.zeros(1))
        self.omega=nn.Parameter(10.0**(torch.rand(k)*6-3)); self.bias=nn.Parameter(torch.zeros(k)); self.output_dim=k+1
    def forward(self,t):
        if t.dim()==1:t=t.unsqueeze(-1)
        return torch.cat([self.w0*t+self.b0,torch.sin(self.omega*t+self.bias)],dim=-1)

class EGATv2Encoder(nn.Module):
    def __init__(self,edge_dim=61,node_init_dim=128,hidden_dim=256,num_heads=8,num_layers=3,
                 dropout_attn=0.3,dropout_feat=0.2):
        super().__init__(); self.hidden_dim=hidden_dim; self.num_heads=num_heads; self.num_layers=num_layers
        self.output_dim=hidden_dim*3; self.node_init_dim=node_init_dim; self.node_embed=None
        self.edge_proj=nn.Linear(edge_dim,hidden_dim); self.convs=nn.ModuleList(); self.norms=nn.ModuleList()
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
        x=self._get_node_embed(data.num_nodes,data.edge_index.device); e=self.edge_proj(data.edge_attr)
        for conv,norm in zip(self.convs,self.norms):
            xn,_=conv(x,data.edge_index,edge_attr=e,return_attention_weights=True)
            xn=self.activation(xn); xn=self.dropout(xn); xn=norm(xn); x=x+xn if x.shape==xn.shape else xn
        return torch.cat([x[data.edge_index[0]],x[data.edge_index[1]],e],dim=-1)

class BinaryHead(nn.Module):
    def __init__(self,input_dim=768,hidden_dim=256,btl=64,nc=2):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ELU(),nn.Dropout(0.3),
                               nn.Linear(hidden_dim,btl),nn.ELU(),nn.Dropout(0.2),nn.Linear(btl,nc))
    def forward(self,x): return self.net(x)

# ---- Load K01 checkpoint ----
ckpt=torch.load(WORKING/'checkpoints'/'D_mae_pretrain'/'best.pt',map_location=device,weights_only=False)
t2v=Time2Vec(k=16).to(device); enc=EGATv2Encoder(edge_dim=EDGE_DIM).to(device)
t2v.load_state_dict(ckpt['t2v']); enc.load_state_dict(ckpt['enc'])
bhead=BinaryHead().to(device)

# ---- Data ----
G_train=[]; G_val=[]
for ds in ['NF-CICIDS2018','NF-UNSW-NB15']:
    for sp,lst in [('train',G_train),('val',G_val)]:
        p=INPUT_DIR/f'{ds}_{sp}_list.pt'
        if p.exists(): lst.extend(torch.load(p,weights_only=False))
print(f"Train: {len(G_train)} windows, Val: {len(G_val)} windows")

all_times=torch.cat([g.edge_time for g in G_train]); T_MIN,T_MAX=all_times.min().item(),all_times.max().item()
def nt(t): return (t-T_MIN)/(T_MAX-T_MIN)

# Focal loss
all_yb=torch.cat([g.y_binary for g in G_train]); n_benign=(all_yb==0).sum(); n_attack=(all_yb==1).sum()
alpha_w=torch.tensor([n_attack/(n_benign+n_attack),n_benign/(n_benign+n_attack)],device=device)

class FocalLoss(nn.Module):
    def __init__(self,g=2.0,a=None): super().__init__(); self.g=g; self.a=a
    def forward(self,logits,targets):
        ce=F.cross_entropy(logits,targets,reduction='none'); pt=torch.exp(-ce)
        f=(1-pt)**self.g*ce
        if self.a is not None: f=self.a[targets]*f
        return f.mean()
fl=FocalLoss(g=2.0,a=alpha_w)

# Feature bounds
all_attr=torch.cat([g.edge_attr for g in G_train])
fmins=all_attr.min(dim=0).values.to(device); fmaxs=all_attr.max(dim=0).values.to(device)
fm61=torch.cat([fmins,torch.full((17,),-4.0,device=device)])
fx61=torch.cat([fmaxs,torch.full((17,),4.0,device=device)])

HP={'pa_lr':1e-3,'pa_ep':5,'pb_lr_enc':1e-5,'pb_lr_head':1e-4,'pb_ep':15,
    'pgd_eps':0.03,'pgd_alpha':0.01,'pgd_steps':7,'pgd_frac':0.30,'bs':4096,'ur':2.0}
amp=GradScaler()

# ---- Phase A: frozen encoder ----
print("Phase A: Frozen encoder...")
for p in list(t2v.parameters())+list(enc.parameters()): p.requires_grad=False
opt_a=optim.Adam(bhead.parameters(),lr=HP['pa_lr'])
for ep in range(HP['pa_ep']):
    bhead.train(); el=0.0; nb=0
    for g in G_train:
        g=g.to(device); bi=(g.y_binary==0).nonzero(as_tuple=True)[0]; ai=(g.y_binary==1).nonzero(as_tuple=True)[0]
        if len(ai)<4: continue
        nbk=min(int(len(ai)*HP['ur']),len(bi))
        if nbk<4: continue
        bk=bi[torch.randperm(len(bi))[:nbk]]; ki=torch.cat([bk,ai])
        for i in range(0,len(ki),HP['bs']):
            bix=ki[i:i+HP['bs']]; if len(bix)<4: continue
            tn=nt(g.edge_time[bix]); te=t2v(tn); ef=torch.cat([g.edge_attr[bix],te],dim=-1)
            bd=Data(edge_index=g.edge_index[:,bix],edge_attr=ef,num_nodes=g.num_nodes)
            with autocast():
                fr=enc(bd); lo=bhead(fr); loss=fl(lo,g.y_binary[bix])
            opt_a.zero_grad(); amp.scale(loss).backward(); amp.step(opt_a); amp.update()
            el+=loss.item(); nb+=1
    print(f"  Phase A Ep {ep+1}: Loss={el/max(nb,1):.6f}")

# ---- Phase B: joint fine-tune ----
print("Phase B: Joint fine-tune...")
for p in list(t2v.parameters())+list(enc.parameters()): p.requires_grad=True
opt_b=optim.Adam([{'params':t2v.parameters(),'lr':HP['pb_lr_enc']},
                  {'params':enc.parameters(),'lr':HP['pb_lr_enc']},
                  {'params':bhead.parameters(),'lr':HP['pb_lr_head']}])
best_vf1=0.0
for ep in range(HP['pb_ep']):
    t2v.train(); enc.train(); bhead.train(); el=0.0; nb=0
    for g in G_train:
        g=g.to(device); bi=(g.y_binary==0).nonzero(as_tuple=True)[0]; ai=(g.y_binary==1).nonzero(as_tuple=True)[0]
        if len(ai)<4: continue
        nbk=min(int(len(ai)*HP['ur']),len(bi))
        if nbk<4: continue
        bk=bi[torch.randperm(len(bi))[:nbk]]; ki=torch.cat([bk,ai])
        for i in range(0,len(ki),HP['bs']):
            bix=ki[i:i+HP['bs']]; if len(bix)<4: continue
            tn=nt(g.edge_time[bix]); te=t2v(tn); ef=torch.cat([g.edge_attr[bix],te],dim=-1)
            # PGD on 30% subset
            npgd=int(len(bix)*HP['pgd_frac'])
            if npgd>0:
                ef_pgd=ef.clone()
                for _ in range(HP['pgd_steps']):
                    ef_pgd=ef_pgd.clone().detach().requires_grad_(True)
                    fr_pgd=enc(Data(edge_index=g.edge_index[:,bix],edge_attr=ef_pgd,num_nodes=g.num_nodes))
                    lo_pgd=bhead(fr_pgd[:npgd]); la=-fl(lo_pgd,g.y_binary[bix][:npgd])
                    gr=torch.autograd.grad(la,ef_pgd,retain_graph=True)[0]
                    ef_pgd=ef_pgd.detach()+HP['pgd_alpha']*gr.sign()
                    d=torch.clamp(ef_pgd-ef,-HP['pgd_eps'],HP['pgd_eps'])
                    ef_pgd=torch.clamp(ef+d,fm61,fx61)
                ef[:npgd]=ef_pgd[:npgd]
            bd=Data(edge_index=g.edge_index[:,bix],edge_attr=ef,num_nodes=g.num_nodes)
            with autocast():
                fr=enc(bd); lo=bhead(fr); loss=fl(lo,g.y_binary[bix])
            opt_b.zero_grad(); amp.scale(loss).backward(); amp.step(opt_b); amp.update()
            el+=loss.item(); nb+=1

    # Val F1
    t2v.eval(); enc.eval(); bhead.eval(); vp=[]; vt=[]
    with torch.no_grad():
        for g in G_val[:10]:
            g=g.to(device)
            if g.edge_index.shape[1]>5000:
                ix=torch.randperm(g.edge_index.shape[1])[:5000]
                g.edge_index=g.edge_index[:,ix]; g.edge_attr=g.edge_attr[ix]; g.edge_time=g.edge_time[ix]; g.y_binary=g.y_binary[ix]
            tn=nt(g.edge_time); te=t2v(tn); ef=torch.cat([g.edge_attr,te],dim=-1)
            fr=enc(Data(edge_index=g.edge_index,edge_attr=ef,num_nodes=g.num_nodes))
            pr=bhead(fr).argmax(dim=1); vp.extend(pr.cpu().tolist()); vt.extend(g.y_binary.cpu().tolist())
    vf1=f1_score(vt,vp,average='macro')
    print(f"  Phase B Ep {ep+1}: Loss={el/max(nb,1):.6f} Val F1={vf1:.4f}")
    if vf1>best_vf1:
        best_vf1=vf1
        torch.save({'t2v':t2v.state_dict(),'enc':enc.state_dict(),'bhead':bhead.state_dict(),'val_f1':vf1,'config':HP},CKPT_DIR/'best.pt')
        json.dump(HP,open(CKPT_DIR/'config.json','w'),indent=2)

# ---- Threshold calibration ----
t2v.eval(); enc.eval(); bhead.eval(); vp=[]; vt=[]
with torch.no_grad():
    for g in G_val:
        g=g.to(device)
        if g.edge_index.shape[1]>10000:
            ix=torch.randperm(g.edge_index.shape[1])[:10000]
            g.edge_index=g.edge_index[:,ix]; g.edge_attr=g.edge_attr[ix]; g.edge_time=g.edge_time[ix]; g.y_binary=g.y_binary[ix]
        tn=nt(g.edge_time); te=t2v(tn); ef=torch.cat([g.edge_attr,te],dim=-1)
        fr=enc(Data(edge_index=g.edge_index,edge_attr=ef,num_nodes=g.num_nodes))
        pr=F.softmax(bhead(fr),dim=-1)[:,1]; vp.extend(pr.cpu().tolist()); vt.extend(g.y_binary.cpu().tolist())
vp=np.array(vp); vt=np.array(vt)
best_t=0.5; best_r=0.0
for t in np.arange(0.05,0.95,0.025):
    r=recall_score(vt,(vp>=t).astype(int))
    if r>=0.995: best_t=t; break
    if r>best_r: best_r=r; best_t=t
print(f"Threshold: {best_t:.3f} (recall={recall_score(vt,(vp>=best_t).astype(int)):.4f})")
json.dump({'threshold':float(best_t),'target_recall':0.995},open(CKPT_DIR/'threshold.json','w'),indent=2)

json.dump({'notebook':'K03','best_val_f1':float(best_vf1),'threshold':float(best_t)},
          open(LOGS_DIR/'k03_log.json','w'),indent=2,default=str)
print(f"✓ K03 complete. Best F1: {best_vf1:.4f}, Threshold: {best_t:.3f}")
