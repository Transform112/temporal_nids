"""
K02 — CVAE Minority-Class Augmentation (Stage E)
==================================================
KAGGLE T4x2 GPU. Loads frozen encoder from K01, extracts 768-dim embeddings,
trains conditional VAE on minority classes, generates synthetic embeddings.

Prerequisite: K01 checkpoint at /kaggle/working/checkpoints/D_mae_pretrain/best.pt
Edge input: 58-dim (41 raw + 17 Time2Vec)
"""

# %% [cell 1]
# !pip install -q torch-geometric pyyaml

import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import numpy as np, yaml, json, random
from datetime import datetime, timezone; from pathlib import Path
from collections import Counter
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch_geometric
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

# %% [cell 2]
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

WORKING = Path('/kaggle/working'); INPUT = Path('/kaggle/input/datasets/mysteriousavailable/ids-nf3-processed/ids-nf3-processed')
CKPT_DIR = WORKING/'checkpoints'/'E_cvae'; LOGS_DIR=WORKING/'logs'; FIGS_DIR=WORKING/'outputs'/'figures'
for d in [CKPT_DIR,LOGS_DIR,FIGS_DIR]: d.mkdir(parents=True,exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [cell 3] Model definitions (matching K01 exactly)
with open(INPUT/'feature_manifest.yaml') as f: fm=yaml.safe_load(f)
with open(INPUT/'label_map.yaml') as f: lm=yaml.safe_load(f)
UNIFIED=lm['unified_classes']; N_CLASSES=len(UNIFIED); EDGE_DIM=fm['final_edge_input_dim']

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
    def __init__(self,edge_dim=58,node_init=128,hidden=256,heads=8,layers=3,d_attn=0.3,d_feat=0.2):
        super().__init__()
        self.hidden=hidden;self.heads=heads;self.layers=layers;self.output_dim=hidden*3
        self.node_init=node_init;self.node_embed=None
        self.edge_proj=nn.Linear(edge_dim,hidden)
        self.convs=nn.ModuleList();self.norms=nn.ModuleList();self.dropout=nn.Dropout(d_feat)
        for _ in range(layers):
            self.convs.append(GATv2Conv((-1,-1),hidden//heads,heads=heads,edge_dim=hidden,dropout=d_attn,concat=True))
            self.norms.append(nn.LayerNorm(hidden))
        self.activation=nn.ELU()
    def _get_node_embed(self,n,dev):
        if self.node_embed is None or self.node_embed.shape[0]<n:
            new=nn.Parameter(torch.randn(n,self.node_init,device=dev)*0.1)
            if self.node_embed is not None:new.data[:self.node_embed.shape[0]]=self.node_embed.data
            self.node_embed=new
        return self.node_embed[:n]
    def forward(self,data):
        x=self._get_node_embed(data.num_nodes,data.edge_index.device)
        ea=self.edge_proj(data.edge_attr)
        for conv,norm in zip(self.convs,self.norms):
            x_new,_=conv(x,data.edge_index,edge_attr=ea,return_attention_weights=True)
            x_new=self.activation(x_new);x_new=self.dropout(x_new);x_new=norm(x_new)
            x=x+x_new if x.shape==x_new.shape else x_new
        return torch.cat([x[data.edge_index[0]],x[data.edge_index[1]],ea],dim=-1)

# %% [cell 4] Load frozen encoder
ckpt=torch.load(WORKING/'checkpoints'/'D_mae_pretrain'/'best.pt',map_location=device,weights_only=False)
t2v=Time2Vec(k=16).to(device);encoder=EGATv2Encoder(edge_dim=EDGE_DIM).to(device)
t2v.load_state_dict(ckpt['t2v']);encoder.load_state_dict(ckpt['encoder'],strict=False)
for p in list(t2v.parameters())+list(encoder.parameters()):p.requires_grad=False
t2v.eval();encoder.eval()
TIME_MIN=ckpt['time_min'];TIME_MAX=ckpt['time_max']
def norm_time(t):return(t-TIME_MIN)/(TIME_MAX-TIME_MIN)
print(f"Encoder loaded. Time range: {(TIME_MAX-TIME_MIN)/3.6e6:.1f}h")

# %% [cell 5] Extract embeddings
def load_graphs(name,split):
    p=INPUT/f'{name}_{split}_list.pt'
    return torch.load(p,weights_only=False) if p.exists() else []

G_train=load_graphs('NF-CICIDS2018','train')+load_graphs('NF-UNSW-NB15','train')
embs,lbls=[],[]
with torch.no_grad():
    for g in G_train:
        g=g.to(device);tn=norm_time(g.edge_time);te=t2v(tn)
        ea58=torch.cat([g.edge_attr,te],dim=-1)
        d=Data(edge_index=g.edge_index,edge_attr=ea58,num_nodes=g.num_nodes)
        reps=encoder(d);embs.append(reps.cpu());lbls.append(g.y.cpu())
embeddings=torch.cat(embs,dim=0);labels=torch.cat(lbls,dim=0)
print(f"Embeddings: {embeddings.shape}")

class_counts=Counter(labels.tolist())
majority=max(class_counts.values());median=np.median(list(class_counts.values()))
minority_classes=[c for c,cnt in class_counts.items() if cnt<median]
print(f"Majority: {majority:,} | Median: {median:.0f} | Minority: {len(minority_classes)} classes")

# %% [cell 6] CVAE model
class ConditionalVAE(nn.Module):
    def __init__(self,in_dim=768,cond_dim=11,hidden=256,latent=64):
        super().__init__();self.latent=latent
        self.encoder=nn.Sequential(nn.Linear(in_dim+cond_dim,hidden),nn.ELU(),nn.Dropout(0.1),
                                    nn.Linear(hidden,128),nn.ELU(),nn.Dropout(0.1))
        self.mu=nn.Linear(128,latent);self.logvar=nn.Linear(128,latent)
        self.decoder=nn.Sequential(nn.Linear(latent+cond_dim,128),nn.ELU(),nn.Dropout(0.1),
                                    nn.Linear(128,hidden),nn.ELU(),nn.Dropout(0.1),nn.Linear(hidden,in_dim))
    def encode(self,x,c):h=self.encoder(torch.cat([x,c],dim=-1));return self.mu(h),self.logvar(h)
    def reparameterize(self,mu,logvar):return mu+torch.randn_like(mu)*torch.exp(0.5*logvar)
    def forward(self,x,c):
        mu,logvar=self.encode(x,c);z=self.reparameterize(mu,logvar)
        return self.decoder(torch.cat([z,c],dim=-1)),mu,logvar
    def generate(self,c):
        z=torch.randn(c.shape[0],self.latent,device=c.device)
        return self.decoder(torch.cat([z,c],dim=-1))

# %% [cell 7] Prepare minority data
minority_mask=torch.tensor([l.item() in minority_classes for l in labels])
X_min=embeddings[minority_mask].numpy();y_min=labels[minority_mask].numpy()
# Free full embedding tensors (can be 48GB+ CPU RAM for 15.7M flows)
# Keep only minority data needed for CVAE training
n_full=embeddings.shape[0]; del embeddings, labels, minority_mask
import gc; gc.collect(); torch.cuda.empty_cache()
print(f"Freed full embeddings ({n_full:,} flows). Minority subset: {X_min.shape[0]:,}")
emb_scaler=StandardScaler();X_min_norm=emb_scaler.fit_transform(X_min)
X_t=torch.tensor(X_min_norm,dtype=torch.float32);y_t=torch.tensor(y_min,dtype=torch.long)

# %% [cell 8] Train CVAE
HP={'lr':5e-4,'epochs':50,'batch':512,'beta':0.5,'latent':64}
cvae=ConditionalVAE(latent=HP['latent']).to(device)
opt=optim.Adam(cvae.parameters(),lr=HP['lr'])
sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=HP['epochs'])
rec_losses,kl_losses=[],[]

for epoch in range(HP['epochs']):
    cvae.train();perm=torch.randperm(X_t.shape[0]);er,ek,nb=0.0,0.0,0
    for i in range(0,X_t.shape[0],HP['batch']):
        idx=perm[i:i+HP['batch']];xb=X_t[idx].to(device);yb=y_t[idx].to(device)
        cb=F.one_hot(yb,num_classes=N_CLASSES).float()
        recon,mu,logvar=cvae(xb,cb)
        r_loss=F.mse_loss(recon,xb);k_loss=-0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
        loss=r_loss+HP['beta']*k_loss
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(cvae.parameters(),1.0);opt.step()
        er+=r_loss.item();ek+=k_loss.item();nb+=1
    sched.step();rec_losses.append(er/nb);kl_losses.append(ek/nb)
    if (epoch+1)%10==0:print(f"Epoch {epoch+1:2d}: Recon={rec_losses[-1]:.6f} KL={kl_losses[-1]:.6f}")
    if kl_losses[-1]<1e-5 and epoch>20:print(f"  WARNING: KL collapse possible")

torch.save({'model':cvae.state_dict(),'emb_scaler_mean':emb_scaler.mean_.tolist(),
            'emb_scaler_scale':emb_scaler.scale_.tolist(),'config':HP},CKPT_DIR/'best.pt')
with open(CKPT_DIR/'config.json','w') as f:json.dump(HP,f,indent=2)

# %% [cell 9] Generate synthetic embeddings
target=int(majority*0.4);synth_embs,synth_lbls=[],[]
cvae.eval()
with torch.no_grad():
    for cls in minority_classes:
        current=class_counts.get(cls,0);needed=max(0,target-current)
        if needed==0:continue
        c=F.one_hot(torch.full((needed,),cls,dtype=torch.long),N_CLASSES).float().to(device)
        gen=[]
        for j in range(0,needed,1024):gen.append(cvae.generate(c[j:j+1024]).cpu())
        gen_all=torch.cat(gen)[:needed]
        gen_all=torch.tensor(emb_scaler.inverse_transform(gen_all.numpy()),dtype=torch.float32)
        synth_embs.append(gen_all);synth_lbls.extend([cls]*needed)
        print(f"  {UNIFIED[cls]:25s}: +{needed:,} synth -> {current+needed:,} total")

synth_all=torch.cat(synth_embs);synth_lbls=torch.tensor(synth_lbls,dtype=torch.long)
torch.save({'embeddings':synth_all,'labels':synth_lbls},CKPT_DIR/'synthetic_embeddings.pt')
print(f"Total synthetic: {synth_all.shape[0]:,}")

# %% [cell 10] PCA quality check (uses minority data only — full embeddings freed above)
fig,axes=plt.subplots(1,2,figsize=(14,5))
# Left: Real vs Synthetic for first minority class
cls0=minority_classes[0]
rm=(y_min==cls0);sm=(synth_lbls.numpy()==cls0)
real_c=X_min[rm];synth_c=synth_all[sm].numpy()
n_plot=min(1000,real_c.shape[0],synth_c.shape[0])
combined=np.concatenate([real_c[:n_plot],synth_c[:n_plot]])
pca=PCA(n_components=2).fit_transform(combined)
axes[0].scatter(pca[:n_plot,0],pca[:n_plot,1],c='blue',alpha=0.3,s=2,label=f'Real {UNIFIED[cls0]}')
axes[0].scatter(pca[n_plot:,0],pca[n_plot:,1],c='red',alpha=0.3,s=2,label=f'Synth {UNIFIED[cls0]}')
axes[0].legend(markerscale=5);axes[0].set_title('Real vs Synthetic (PCA)')
# Right: Minority class embeddings by class (sampled from kept X_min)
sn=min(300,X_min.shape[0]//len(minority_classes));pd_data,pd_lbls=[],[]
for c in minority_classes:
    cm=y_min==c
    if cm.sum()>sn:
        idx=np.random.choice(cm.sum(),sn,replace=False);pd_data.append(X_min[cm][idx]);pd_lbls.extend([UNIFIED[c]]*sn)
if pd_data:
    pca_all=PCA(n_components=2).fit_transform(np.concatenate(pd_data))
    off=0
    for c in minority_classes:
        n=sum(1 for l in pd_lbls if l==UNIFIED[c])
        if n>0:axes[1].scatter(pca_all[off:off+n,0],pca_all[off:off+n,1],alpha=0.4,s=2,label=UNIFIED[c][:15]);off+=n
    axes[1].legend(fontsize=6,markerscale=5)
axes[1].set_title('Minority Embeddings by Class (PCA)')
plt.tight_layout();plt.savefig(FIGS_DIR/'cvae_quality_pca.png',dpi=150);plt.show()

log={'notebook':'K02','stage':'E','minority_classes':[UNIFIED[c] for c in minority_classes],
     'synthetic_count':int(synth_all.shape[0]),'final_rec_loss':rec_losses[-1],'final_kl':kl_losses[-1]}
with open(LOGS_DIR/'k02_log.json','w') as f:json.dump(log,f,indent=2)
print(f"\nK02 DONE. Next: K03 (Binary)")
