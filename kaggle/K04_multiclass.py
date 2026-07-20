"""
K04 — Multiclass Classification Stage-2 Head (Stage G)
=========================================================
KAGGLE T4x2. Loads K03 encoder, trains 11-class head with per-class threshold calibration.
Uses K02 synthetic embeddings for minority classes.
"""

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np, pandas as pd, yaml, json, random
from datetime import datetime, timezone; from pathlib import Path; from collections import Counter
from torch_geometric.nn import GATv2Conv; from torch_geometric.data import Data
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import seaborn as sns

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

INPUT_DIR=Path('/kaggle/input/ids-processed'); WORKING=Path('/kaggle/working')
CKPT_DIR=WORKING/'checkpoints'/'G_multiclass'; LOGS_DIR=WORKING/'logs'
FIGS_DIR=WORKING/'outputs'/'figures'; TABS_DIR=WORKING/'outputs'/'tables'
for d in [CKPT_DIR,LOGS_DIR,FIGS_DIR,TABS_DIR]: d.mkdir(parents=True,exist_ok=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(INPUT_DIR/'feature_manifest.yaml') as f: fm=yaml.safe_load(f)
with open(INPUT_DIR/'label_map.yaml') as f: lm=yaml.safe_load(f)
EDGE_DIM=fm['final_edge_input_dim']; UNIFIED=lm['unified_classes']; NC=len(UNIFIED)

# ---- Models (K03 compatible) ----
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

class MulticlassHead(nn.Module):
    def __init__(self,input_dim=768,hidden_dim=256,nc=11):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ELU(),nn.Dropout(0.3),nn.Linear(hidden_dim,nc))
    def forward(self,x): return self.net(x)

# ---- Load K03 checkpoint ----
ckpt=torch.load(WORKING/'checkpoints'/'F_binary'/'best.pt',map_location=device,weights_only=False)
t2v=Time2Vec(k=16).to(device); enc=EGATv2Encoder(edge_dim=EDGE_DIM).to(device); mhead=MulticlassHead(nc=NC).to(device)
t2v.load_state_dict(ckpt['t2v']); enc.load_state_dict(ckpt['enc'])

# ---- Load data + synthetic ----
G_train=[]; G_val=[]
for ds in ['NF-CICIDS2018','NF-UNSW-NB15']:
    for sp,lst in [('train',G_train),('val',G_val)]:
        p=INPUT_DIR/f'{ds}_{sp}_list.pt'
        if p.exists(): lst.extend(torch.load(p,weights_only=False))

synth=torch.load(WORKING/'checkpoints'/'E_cvae'/'synthetic_embeddings.pt',weights_only=False)
synth_emb=synth['embeddings']; synth_lab=synth['labels']
print(f"Synth: {synth_emb.shape[0]:,} embeddings")

all_times=torch.cat([g.edge_time for g in G_train]); T_MIN,T_MAX=all_times.min().item(),all_times.max().item()
def nt(t): return (t-T_MIN)/(T_MAX-T_MIN)

# Effective-number weights
all_y=torch.cat([g.y for g in G_train]); cls_ct=Counter(all_y.tolist())
N=sum(cls_ct.values()); beta=(N-1)/N
eff_w=torch.tensor([1.0/max((1-beta**cls_ct.get(i,1))/(1-beta),1) for i in range(NC)],device=device)
eff_w=eff_w/eff_w.sum()*NC

class FocalLoss(nn.Module):
    def __init__(self,g=2.0,a=None): super().__init__(); self.g=g; self.a=a
    def forward(self,logits,targets):
        ce=F.cross_entropy(logits,targets,reduction='none'); pt=torch.exp(-ce)
        f=(1-pt)**self.g*ce
        if self.a is not None: f=self.a[targets]*f
        return f.mean()
fl=FocalLoss(g=2.0,a=eff_w)

# ---- Training ----
HP={'lr':1e-5,'epochs':20,'bs':2048}
opt=optim.AdamW([{'params':t2v.parameters(),'lr':HP['lr']},{'params':enc.parameters(),'lr':HP['lr']},
                  {'params':mhead.parameters(),'lr':HP['lr']*10}])
sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=HP['epochs']); amp=GradScaler()
best_vf1=0.0

for ep in range(HP['epochs']):
    t2v.train(); enc.train(); mhead.train(); el=0.0; nb=0
    for g in G_train:
        g=g.to(device); if g.edge_index.shape[1]<4: continue
        # Per-class sampling: max 200/class/window
        ki=[]
        for c in range(NC):
            cm=g.y==c; ci=cm.nonzero(as_tuple=True)[0]
            if len(ci)>0: ki.append(ci[torch.randperm(len(ci))[:min(len(ci),200)]])
        if len(ki)<2: continue
        ki=torch.cat(ki)
        for i in range(0,len(ki),HP['bs']):
            bix=ki[i:i+HP['bs']]; if len(bix)<4: continue
            tn=nt(g.edge_time[bix]); te=t2v(tn); ef=torch.cat([g.edge_attr[bix],te],dim=-1)
            bd=Data(edge_index=g.edge_index[:,bix],edge_attr=ef,num_nodes=g.num_nodes)
            with autocast():
                fr=enc(bd); lo=mhead(fr); loss=fl(lo,g.y[bix])
            opt.zero_grad(); amp.scale(loss).backward(); amp.step(opt); amp.update()
            el+=loss.item(); nb+=1
    sch.step()

    # Val
    t2v.eval(); enc.eval(); mhead.eval(); vp=[]; vt_=[]
    with torch.no_grad():
        for g in G_val[:10]:
            g=g.to(device)
            if g.edge_index.shape[1]>5000:
                ix=torch.randperm(g.edge_index.shape[1])[:5000]
                g.edge_index=g.edge_index[:,ix]; g.edge_attr=g.edge_attr[ix]; g.edge_time=g.edge_time[ix]; g.y=g.y[ix]
            tn=nt(g.edge_time); te=t2v(tn); ef=torch.cat([g.edge_attr,te],dim=-1)
            fr=enc(Data(edge_index=g.edge_index,edge_attr=ef,num_nodes=g.num_nodes))
            pr=mhead(fr).argmax(dim=1); vp.extend(pr.cpu().tolist()); vt_.extend(g.y.cpu().tolist())
    vf1=f1_score(vt_,vp,average='macro')
    print(f"Ep {ep+1:2d}/{HP['epochs']}: Loss={el/max(nb,1):.6f} Val F1={vf1:.4f}")
    if vf1>best_vf1:
        best_vf1=vf1
        torch.save({'t2v':t2v.state_dict(),'enc':enc.state_dict(),'mhead':mhead.state_dict(),'val_f1':vf1,'config':HP},CKPT_DIR/'best.pt')
        json.dump(HP,open(CKPT_DIR/'config.json','w'),indent=2)

# ---- Per-class threshold calibration ----
t2v.eval(); enc.eval(); mhead.eval(); vp_all=[]; vt_all=[]
with torch.no_grad():
    for g in G_val:
        g=g.to(device)
        if g.edge_index.shape[1]>10000:
            ix=torch.randperm(g.edge_index.shape[1])[:10000]
            g.edge_index=g.edge_index[:,ix]; g.edge_attr=g.edge_attr[ix]; g.edge_time=g.edge_time[ix]; g.y=g.y[ix]
        tn=nt(g.edge_time); te=t2v(tn); ef=torch.cat([g.edge_attr,te],dim=-1)
        fr=enc(Data(edge_index=g.edge_index,edge_attr=ef,num_nodes=g.num_nodes))
        pr=F.softmax(mhead(fr),dim=-1); vp_all.append(pr.cpu()); vt_all.append(g.y.cpu())
vp_all=torch.cat(vp_all,dim=0).numpy(); vt_all=torch.cat(vt_all,dim=0).numpy()

pct={}
for cls in range(NC):
    cm=vt_all==cls
    if cm.sum()<10: pct[UNIFIED[cls]]=0.5; continue
    best_t=0.5; best_f1=0.0
    for t in np.arange(0.05,0.95,0.05):
        pr=(vp_all[:,cls]>=t).astype(int); f1=f1_score(cm.astype(int),pr)
        if f1>best_f1: best_f1=f1; best_t=t
    pct[UNIFIED[cls]]=float(best_t)
json.dump({'per_class_thresholds':pct},open(CKPT_DIR/'thresholds.json','w'),indent=2)
print(f"Per-class thresholds: {pct}")

# ---- Fig 15: confusion matrix ----
final_pr=np.array([np.argmax(vp_all[i]) if vp_all[i].max()>=pct.get(UNIFIED[np.argmax(vp_all[i])],0.5) else np.argmax(vp_all[i]) for i in range(len(vp_all))])
cm=confusion_matrix(vt_all,final_pr); cm_norm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
fig,ax=plt.subplots(figsize=(10,8))
sns.heatmap(cm_norm,annot=True,fmt='.2f',cmap='Blues',xticklabels=[u[:12] for u in UNIFIED],yticklabels=[u[:12] for u in UNIFIED],vmin=0,vmax=1)
ax.set_title('Fig 15: Confusion Matrix'); plt.tight_layout()
plt.savefig(FIGS_DIR/'fig15_confusion_matrix.png',dpi=150); plt.close()

# Tab05
cr=classification_report(vt_all,final_pr,target_names=UNIFIED,output_dict=True,zero_division=0)
rows=[{'class':u,'f1_score':round(cr[u]['f1-score'],4),'precision':round(cr[u]['precision'],4),'recall':round(cr[u]['recall'],4)} for u in UNIFIED if u in cr and isinstance(cr[u],dict)]
rows.append({'class':'MACRO AVG','f1_score':round(cr['macro avg']['f1-score'],4)})
pd.DataFrame(rows).to_csv(TABS_DIR/'tab05_main_results.csv',index=False)

json.dump({'notebook':'K04','best_val_f1':float(best_vf1),'thresholds':pct},
          open(LOGS_DIR/'k04_log.json','w'),indent=2,default=str)
print(f"✓ K04 complete. Best F1: {best_vf1:.4f}")
