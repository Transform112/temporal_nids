"""
K02 — CVAE Minority-Class Augmentation (Stage E)
===================================================
KAGGLE T4x2. Loads K01 encoder, generates synthetic minority-class embeddings.
Copy-paste into Kaggle notebook cell.
"""

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np, yaml, json, pickle, random, gc
from datetime import datetime, timezone; from pathlib import Path
from collections import Counter
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

INPUT_DIR=Path('/kaggle/input/ids-processed'); WORKING=Path('/kaggle/working')
CKPT_DIR=WORKING/'checkpoints'/'E_cvae'; FIGS_DIR=WORKING/'outputs'/'figures'; LOGS_DIR=WORKING/'logs'
for d in [CKPT_DIR,FIGS_DIR,LOGS_DIR]: d.mkdir(parents=True,exist_ok=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load manifests
with open(INPUT_DIR/'feature_manifest.yaml') as f: fm=yaml.safe_load(f)
with open(INPUT_DIR/'label_map.yaml') as f: lm=yaml.safe_load(f)
EDGE_DIM=fm['final_edge_input_dim']; UNIFIED=lm['unified_classes']; NC=len(UNIFIED)

# ---- Model definitions (same as K01) ----
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

class ConditionalVAE(nn.Module):
    def __init__(self,input_dim=768,condition_dim=11,hidden_dim=256,latent_dim=64):
        super().__init__(); self.latent_dim=latent_dim
        id_=input_dim+condition_dim
        self.enc=nn.Sequential(nn.Linear(id_,hidden_dim),nn.ELU(),nn.Dropout(0.1),
                               nn.Linear(hidden_dim,128),nn.ELU(),nn.Dropout(0.1))
        self.mu=nn.Linear(128,latent_dim); self.lv=nn.Linear(128,latent_dim)
        self.dec=nn.Sequential(nn.Linear(latent_dim+condition_dim,128),nn.ELU(),nn.Dropout(0.1),
                               nn.Linear(128,hidden_dim),nn.ELU(),nn.Dropout(0.1),nn.Linear(hidden_dim,input_dim))
    def encode(self,x,c): h=self.enc(torch.cat([x,c],dim=-1)); return self.mu(h),self.lv(h)
    def reparam(self,mu,lv): return mu+torch.randn_like(lv)*torch.exp(0.5*lv)
    def forward(self,x,c):
        mu,lv=self.encode(x,c); z=self.reparam(mu,lv); rc=self.dec(torch.cat([z,c],dim=-1))
        return rc,mu,lv
    def generate(self,c,device='cpu'):
        return self.dec(torch.cat([torch.randn(c.shape[0],self.latent_dim,device=device),c],dim=-1))

# ---- Load frozen encoder from K01 ----
ckpt=torch.load(WORKING/'checkpoints'/'D_mae_pretrain'/'best.pt',map_location=device,weights_only=False)
t2v=Time2Vec(k=16).to(device); enc=EGATv2Encoder(edge_dim=EDGE_DIM).to(device)
t2v.load_state_dict(ckpt['t2v']); enc.load_state_dict(ckpt['enc'])
for p in list(t2v.parameters())+list(enc.parameters()): p.requires_grad=False
t2v.eval(); enc.eval()

# ---- Load training graphs & extract embeddings ----
G_train=[]
for ds in ['NF-CICIDS2018','NF-UNSW-NB15']:
    p=INPUT_DIR/f'{ds}_train_list.pt'
    if p.exists(): G_train.extend(torch.load(p,weights_only=False))

all_times=torch.cat([g.edge_time for g in G_train])
T_MIN,T_MAX=all_times.min().item(),all_times.max().item()

print("Extracting embeddings...")
embs,labels_list=[],[]
with torch.no_grad():
    for g in G_train:
        g=g.to(device); tn=(g.edge_time-T_MIN)/(T_MAX-T_MIN); te=t2v(tn)
        ef=torch.cat([g.edge_attr,te],dim=-1)
        reps=enc(Data(edge_index=g.edge_index,edge_attr=ef,num_nodes=g.num_nodes))
        embs.append(reps.cpu()); labels_list.append(g.y.cpu())
embeddings=torch.cat(embs,dim=0); labels=torch.cat(labels_list,dim=0)
print(f"Embeddings: {embeddings.shape}")

# ---- Class distribution & minority detection ----
cls_counts=Counter(labels.tolist())
median_ct=np.median(list(cls_counts.values()))
majority_ct=max(cls_counts.values())
minority_cls=[c for c,ct in cls_counts.items() if ct<median_ct]
print(f"Minority classes: {[UNIFIED[c] for c in minority_cls]}, target={int(majority_ct*0.4):,}")

# ---- Train CVAE ----
minority_mask=torch.tensor([l.item() in minority_cls for l in labels])
X_min=embeddings[minority_mask]; y_min=labels[minority_mask]
es=StandardScaler(); X_min_norm=torch.tensor(es.fit_transform(X_min.numpy()),dtype=torch.float32)

cvae=ConditionalVAE().to(device); opt=optim.Adam(cvae.parameters(),lr=5e-4)
sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50)
HP={'epochs':50,'bs':512,'beta':0.5}

print(f"Training CVAE ({HP['epochs']} epochs)...")
for ep in range(HP['epochs']):
    cvae.train(); perm=torch.randperm(X_min_norm.shape[0]); el=0.0; nb=0
    for i in range(0,X_min_norm.shape[0],HP['bs']):
        xb=X_min_norm[perm[i:i+HP['bs']]].to(device)
        yb=y_min[perm[i:i+HP['bs']]].to(device)
        cb=F.one_hot(yb,num_classes=NC).float()
        rc,mu,lv=cvae(xb,cb)
        rl=F.mse_loss(rc,xb); kl=-0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
        loss=rl+HP['beta']*kl
        opt.zero_grad(); loss.backward(); opt.step()
        el+=loss.item(); nb+=1
    sch.step()
    if (ep+1)%10==0: print(f"  Ep {ep+1}: Loss={el/max(nb,1):.6f}")

# ---- Generate synthetic embeddings ----
cvae.eval(); synth_embs=[]; synth_labels=[]; synth_counts={}
with torch.no_grad():
    for cls in minority_cls:
        needed=int(majority_ct*0.4-cls_counts.get(cls,0))
        if needed<=0: continue
        c=F.one_hot(torch.full((needed,),cls,dtype=torch.long),num_classes=NC).float().to(device)
        gen=torch.cat([cvae.generate(c[i:i+1024],device) for i in range(0,needed,1024)],dim=0)[:needed]
        gen=torch.tensor(es.inverse_transform(gen.cpu().numpy()),dtype=torch.float32)
        synth_embs.append(gen); synth_labels.extend([cls]*needed); synth_counts[UNIFIED[cls]]=needed
        print(f"  {UNIFIED[cls]}: +{needed:,} synthetic")

synth_all=torch.cat(synth_embs,dim=0) if synth_embs else torch.empty(0,768)
print(f"Total synthetic: {synth_all.shape[0]:,}")

torch.save({'embeddings':synth_all,'labels':torch.tensor(synth_labels,dtype=torch.long),
            'is_synthetic':True,'counts':synth_counts,'scaler_mean':es.mean_.tolist(),
            'scaler_scale':es.scale_.tolist()},CKPT_DIR/'synthetic_embeddings.pt')
torch.save({'model':cvae.state_dict(),'config':HP},CKPT_DIR/'best.pt')
json.dump(HP,open(CKPT_DIR/'config.json','w'),indent=2)

# ---- Fig 08: class distribution ----
fig,axes=plt.subplots(1,2,figsize=(14,5))
pre_cts=[cls_counts.get(i,0) for i in range(NC)]
post_cts=[cls_counts.get(i,0)+synth_counts.get(UNIFIED[i],0) for i in range(NC)]
axes[0].bar(range(NC),pre_cts,color='#2196F3'); axes[1].bar(range(NC),post_cts,color='#4CAF50')
for ax in axes: ax.set_xticks(range(NC)); ax.set_xticklabels([u[:8] for u in UNIFIED],rotation=45,ha='right',fontsize=7)
axes[0].set_title('Pre-Augmentation'); axes[1].set_title('Post-Augmentation')
axes[0].set_yscale('log'); axes[1].set_yscale('log')
fig.suptitle('Fig 08: Class Distribution'); plt.tight_layout()
plt.savefig(FIGS_DIR/'fig08_class_distribution.png',dpi=150); plt.close()

json.dump({'notebook':'K02','synth_count':int(synth_all.shape[0]),'minority':[UNIFIED[c] for c in minority_cls]},
          open(LOGS_DIR/'k02_log.json','w'),indent=2,default=str)
print(f"✓ K02 complete. {synth_all.shape[0]:,} synth embeddings saved.")
