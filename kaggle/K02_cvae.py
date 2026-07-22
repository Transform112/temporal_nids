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

INPUT_DIR=Path('/kaggle/input/datasets/mysteriousavailable/ids-nf3-processed/ids-nf3-processed')
WORKING=Path('../working')
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
    def __init__(self, max_nodes, edge_dim=58, node_init=128, hidden=256, heads=8, layers=3,
                 d_attn=0.3, d_feat=0.2, return_attention=False):
        super().__init__()
        assert max_nodes is not None and max_nodes > 0, "max_nodes must be a positive int"
        self.hidden=hidden; self.heads=heads; self.layers=layers
        self.output_dim = hidden*3  # 768
        self.node_init = node_init
        self.max_nodes = max_nodes
        self.return_attention = return_attention

        self.node_embed = nn.Parameter(torch.randn(max_nodes, node_init) * 0.1)
        self.edge_proj = nn.Linear(edge_dim, hidden)
        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(d_feat)
        for _ in range(layers):
            self.convs.append(GATv2Conv((-1,-1), hidden//heads, heads=heads,
                edge_dim=hidden, dropout=d_attn, concat=True))
            self.norms.append(nn.LayerNorm(hidden))
        self.activation = nn.ELU()

    def forward(self, data):
        n = data.num_nodes
        table_size = self.node_embed.shape[0]
        if n <= table_size:
            x = self.node_embed[:n]
        else:
            idx = torch.arange(n, device=self.node_embed.device) % table_size
            x = self.node_embed[idx]
        ea = self.edge_proj(data.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            if self.return_attention:
                x_new, _ = conv(x, data.edge_index, edge_attr=ea, return_attention_weights=True)
            else:
                x_new = conv(x, data.edge_index, edge_attr=ea)
            x_new = self.activation(x_new); x_new = self.dropout(x_new)
            x = norm(x + x_new) if x.shape == x_new.shape else norm(x_new)
        return torch.cat([x[data.edge_index[0]], x[data.edge_index[1]], ea], dim=-1)

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
    def reparam(self,mu,lv): lv=torch.clamp(lv,-10,10); return mu+torch.randn_like(lv)*torch.exp(0.5*lv)
    def forward(self,x,c):
        mu,lv=self.encode(x,c); z=self.reparam(mu,lv); rc=self.dec(torch.cat([z,c],dim=-1))
        return rc,mu,lv
    def generate(self,c,device='cpu'):
        return self.dec(torch.cat([torch.randn(c.shape[0],self.latent_dim,device=device),c],dim=-1))

# ---- Load frozen encoder from K01 ----
K01_CKPT_DIR=Path('/kaggle/input/datasets/mysteriousavailable/checkpoint-k01/checkpoints/D_mae_pretrain')
ckpt=torch.load(K01_CKPT_DIR/'best.pt',map_location=device,weights_only=False)
t2v=Time2Vec(k=16).to(device)
enc=EGATv2Encoder(max_nodes=ckpt['max_nodes'], edge_dim=EDGE_DIM).to(device)
t2v.load_state_dict(ckpt['t2v']); enc.load_state_dict(ckpt['encoder'])
for p in list(t2v.parameters())+list(enc.parameters()): p.requires_grad=False
t2v.eval(); enc.eval()
if torch.cuda.is_available():
    t2v.half(); enc.half()  # frozen + eval-only: half precision here is safe and roughly halves
    # VRAM for the dominant cost in this script (Pass 2's per-graph forward passes)
T_MIN,T_MAX=ckpt['time_min'],ckpt['time_max']  # use K01's exact training-time bounds, not a recompute
del ckpt; gc.collect()  # ckpt also carries K01's decoder weights we never use — free that RAM now

DATASET_FILES=[INPUT_DIR/f'{ds}_train_list.pt' for ds in ['NF-CICIDS2018','NF-UNSW-NB15']]
DATASET_FILES=[p for p in DATASET_FILES if p.exists()]

# ---- PASS 1: label counts only, no encoder — cheap, tells us which classes are minority ----
# Streams one file at a time and frees it immediately: never holds both datasets in RAM together.
print("Pass 1/2: counting labels...")
cls_counts=Counter()
for p in DATASET_FILES:
    graphs=torch.load(p,weights_only=False)
    for g in graphs: cls_counts.update(g.y.tolist())
    del graphs; gc.collect()
median_ct=np.median(list(cls_counts.values()))
majority_ct=max(cls_counts.values())
minority_cls=set(c for c,ct in cls_counts.items() if ct<median_ct)
minority_cls_t=torch.tensor(list(minority_cls))
print(f"Minority classes: {[UNIFIED[c] for c in minority_cls]}, median_ct={int(median_ct):,} "
      f"(per-class targets computed after CVAE training, below)")

# ---- PASS 2: run the frozen encoder per graph, but keep ONLY minority-class rows. ----
# CRITICAL RAM FIX #1: the encoder must see the whole graph to compute correct attention-based
# edge reps, so the forward pass itself can't be skipped for majority edges — but we never need
# to KEEP majority-class embeddings (only X_min feeds the CVAE below). Discarding them immediately
# instead of accumulating full-dataset embeddings avoids tens of GB of unused embeddings in RAM.
# CRITICAL RAM FIX #2: files are streamed one at a time (not pre-loaded together) so peak RAM is
# bounded by ONE dataset's raw graphs, not both.
print("Pass 2/2: extracting minority-class embeddings...")
min_embs,min_labels=[],[]
n_processed=0; n_oom_skipped=0
with torch.no_grad():
    for p in DATASET_FILES:
        graphs=torch.load(p,weights_only=False)
        for g in graphs:
            y=g.y
            keep=torch.isin(y,minority_cls_t)
            if keep.any():
                try:
                    gd=g.to(device)
                    tn=(gd.edge_time-T_MIN)/(T_MAX-T_MIN)
                    if torch.cuda.is_available(): tn=tn.half()
                    te=t2v(tn)
                    ef=torch.cat([gd.edge_attr.half() if torch.cuda.is_available() else gd.edge_attr,te],dim=-1)
                    reps=enc(Data(edge_index=gd.edge_index,edge_attr=ef,num_nodes=gd.num_nodes))
                    reps_min=reps[keep.to(device)]
                    min_embs.append(reps_min.half().cpu()); min_labels.append(y[keep])
                    del gd,reps,reps_min
                except RuntimeError as e:
                    if 'out of memory' not in str(e).lower(): raise
                    # one oversized graph shouldn't kill the whole run — skip it, clear cache, continue
                    n_oom_skipped+=1
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    print(f"  [warn] CUDA OOM on a graph with {g.edge_index.shape[1]:,} edges — skipped")
            n_processed+=1
            if n_processed%50==0:
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                print(f"  {n_processed} graphs processed")
        del graphs; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
if n_oom_skipped: print(f"  [warn] {n_oom_skipped} graph(s) skipped due to CUDA OOM")
X_min=torch.cat(min_embs,dim=0).float(); y_min=torch.cat(min_labels,dim=0)
del min_embs,min_labels; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print(f"Minority embeddings: {X_min.shape}")

# ---- Train CVAE ----
es=StandardScaler(); X_min_norm=torch.tensor(es.fit_transform(X_min.numpy()),dtype=torch.float32)

cvae=ConditionalVAE(hidden_dim=512, latent_dim=128).to(device); opt=optim.Adam(cvae.parameters(),lr=5e-4)
HP={'epochs':100,'bs':128,'beta':0.2}
sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=HP['epochs'])

print(f"Training CVAE ({HP['epochs']} epochs)...")
train_hist=[]
for ep in range(HP['epochs']):
    cvae.train(); perm=torch.randperm(X_min_norm.shape[0]); el=0.0; rl_sum=0.0; kl_sum=0.0; nb=0; nan_b=0
    for i in range(0,X_min_norm.shape[0],HP['bs']):
        xb=X_min_norm[perm[i:i+HP['bs']]].to(device)
        yb=y_min[perm[i:i+HP['bs']]].to(device)
        cb=F.one_hot(yb,num_classes=NC).float()
        rc,mu,lv=cvae(xb,cb)
        lv_c=torch.clamp(lv,-10,10)
        rl=F.mse_loss(rc,xb); kl=-0.5*torch.mean(1+lv_c-mu.pow(2)-lv_c.exp())
        loss=rl+HP['beta']*kl
        if not torch.isfinite(loss):
            nan_b+=1; opt.zero_grad(); continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(cvae.parameters(),5.0)
        opt.step()
        el+=loss.item(); rl_sum+=rl.item(); kl_sum+=kl.item(); nb+=1
    sch.step()
    avg_loss=el/max(nb,1); avg_rl=rl_sum/max(nb,1); avg_kl=kl_sum/max(nb,1)
    train_hist.append({'epoch':ep+1,'loss':avg_loss,'recon_mse':avg_rl,'kl':avg_kl,'nan_batches':nan_b})
    if (ep+1)%10==0 or ep==0:
        print(f"  Ep {ep+1:2d}: loss={avg_loss:.6f} recon_mse={avg_rl:.6f} kl={avg_kl:.6f} nan_batches={nan_b}")

# ---- Generate synthetic embeddings ----
# Target formula redesigned. Two problems with majority_ct*0.4:
#  1. RAM: majority_ct can be huge (13M+), producing multi-million-row targets per class.
#  2. QUALITY (the more important one): a CVAE trained on very few real samples produces
#     degraded synthetic data at extreme oversampling ratios. Proven empirically in your last
#     run — Web Attack (2,078 real) was asked for 200,000 synthetic (96x) and scored a 0.3988
#     centroid distance, 4-10x worse than classes oversampled more modestly. That's the
#     generator overfitting/mode-collapsing, not noise — training a classifier on that much
#     near-duplicate synthetic data would teach it the CVAE's artifacts, not real attack
#     patterns, hurting generalization on your blind test sets.
# New approach: bring each minority class up toward the dataset MEDIAN class size (standard
# balancing target, not an arbitrary fraction of the single largest class), capped at a max
# oversampling ratio relative to the class's OWN real count (SMOTE-family literature typically
# stays under ~10-15x; we use 10x as a safe default backed by the evidence above).
MAX_OVERSAMPLE_RATIO=10  # don't generate more than 10x a class's real sample count
MAX_SYNTH_PER_CLASS=200_000  # hard backstop regardless of the above, purely for RAM/disk safety
needed_per_cls={}
for cls in minority_cls:
    real_n=cls_counts.get(cls,0)
    target=min(median_ct, MAX_OVERSAMPLE_RATIO*real_n)
    needed=int(target-real_n)
    if needed<=0: continue
    if needed>MAX_SYNTH_PER_CLASS:
        needed=MAX_SYNTH_PER_CLASS
    needed_per_cls[cls]=needed
    print(f"  {UNIFIED[cls]}: real={real_n:,} -> target={real_n+needed:,} "
          f"({(real_n+needed)/max(real_n,1):.1f}x oversample)")
total_synth=sum(needed_per_cls.values())
print(f"Generating {total_synth:,} synthetic embeddings total, streamed to disk...")

EMB_DIM=X_min.shape[1]
synth_path=CKPT_DIR/'synthetic_embeddings.dat'
synth_mm=np.memmap(synth_path,dtype='float32',mode='w+',shape=(total_synth,EMB_DIM))
synth_labels=np.empty(total_synth,dtype=np.int64)
synth_counts={}; quality={}
offset=0
cvae.eval()
with torch.no_grad():
    for cls,needed in needed_per_cls.items():
        cls_start=offset
        real_cls_emb=X_min[y_min==cls]  # small (real minority data only) — safe to keep in RAM
        gen_sum=torch.zeros(EMB_DIM)  # running sum for centroid, avoids keeping all synth rows
        for i in range(0,needed,1024):
            n=min(1024,needed-i)
            c=F.one_hot(torch.full((n,),cls,dtype=torch.long),num_classes=NC).float().to(device)
            chunk=cvae.generate(c,device).cpu()
            chunk=torch.tensor(es.inverse_transform(chunk.numpy()),dtype=torch.float32)
            synth_mm[offset:offset+n]=chunk.numpy()
            synth_labels[offset:offset+n]=cls
            gen_sum+=chunk.sum(0)
            offset+=n
        synth_counts[UNIFIED[cls]]=needed
        centroid_dist=torch.norm(gen_sum/needed-real_cls_emb.mean(0)).item()
        real_norm=torch.norm(real_cls_emb.mean(0)).item()+1e-8
        quality[UNIFIED[cls]]=round(centroid_dist/real_norm,4)
        print(f"  {UNIFIED[cls]}: +{needed:,} synthetic (real n={int(cls_counts.get(cls,0)):,}, "
              f"centroid_rel_dist={quality[UNIFIED[cls]]:.4f})")
synth_mm.flush()
print(f"Total synthetic: {total_synth:,} (written to {synth_path})")

torch.save({'embeddings_memmap_path':str(synth_path),'embeddings_shape':(total_synth,EMB_DIM),
            'embeddings_dtype':'float32','labels':torch.from_numpy(synth_labels),
            'is_synthetic':True,'counts':synth_counts,'scaler_mean':es.mean_.tolist(),
            'scaler_scale':es.scale_.tolist()},CKPT_DIR/'synthetic_embeddings.pt')
# NOTE for K03: load embeddings back with
#   np.memmap(d['embeddings_memmap_path'], dtype=d['embeddings_dtype'], mode='r', shape=d['embeddings_shape'])
# instead of a plain tensor — avoids re-loading the whole synthetic set into RAM there too.
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

json.dump({'notebook':'K02','synth_count':int(total_synth),
           'minority':[UNIFIED[c] for c in minority_cls],
           'train_history':train_hist,'synth_quality_rel_centroid_dist':quality,
           'synth_counts':synth_counts},
          open(LOGS_DIR/'k02_log.json','w'),indent=2,default=str)
print(f"✓ K02 complete. {total_synth:,} synth embeddings saved.")
print(f"  Final CVAE: loss={train_hist[-1]['loss']:.6f} recon_mse={train_hist[-1]['recon_mse']:.6f} "
      f"kl={train_hist[-1]['kl']:.6f}")
print(f"  Mean synth quality (rel. centroid dist, lower=better): {np.mean(list(quality.values())):.4f}")