"""
K05 — Prototypical Few-Shot / Zero-Day Detection (Stage H)
=============================================================
KAGGLE T4x2 GPU. Episodic training with attention-weighted prototypes.
Leave-one-class-out novelty threshold tuning.

Prerequisite: K04 checkpoint + preprocessed graphs
"""

# %% [cell 1]
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np, pandas as pd, yaml, json, random; from pathlib import Path
from datetime import datetime, timezone; from collections import defaultdict, Counter
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve, average_precision_score
import torch_geometric; from torch_geometric.nn import GATv2Conv; from torch_geometric.data import Data

# %% [cell 2]
SEED=42;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)

WORKING=Path('/kaggle/working');INPUT=Path('/kaggle/input/datasets/mysteriousavailable/ids-nf3-processed/ids-nf3-processed')
CKPT_DIR=WORKING/'checkpoints'/'H_prototypical'
LOGS_DIR=WORKING/'logs';FIGS_DIR=WORKING/'outputs'/'figures';TABS_DIR=WORKING/'outputs'/'tables'
for d in [CKPT_DIR,LOGS_DIR,FIGS_DIR,TABS_DIR]:d.mkdir(parents=True,exist_ok=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [cell 3] Model defs (matching K01-K04)
with open(INPUT/'feature_manifest.yaml') as f:fm=yaml.safe_load(f)
with open(INPUT/'label_map.yaml') as f:lm=yaml.safe_load(f)
UNIFIED=lm['unified_classes'];N_CLASSES=len(UNIFIED);EDGE_DIM=fm['final_edge_input_dim']

class Time2Vec(nn.Module):
    def __init__(self,k=16):
        super().__init__();self.k=k;self.w0=nn.Parameter(torch.randn(1)*0.1)
        self.b0=nn.Parameter(torch.zeros(1));self.omega=nn.Parameter(10.0**(torch.rand(k)*6-3))
        self.bias=nn.Parameter(torch.zeros(k));self.output_dim=k+1
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

# %% [cell 4] Load frozen encoder + extract embeddings
ckpt_g=torch.load(WORKING/'checkpoints'/'G_multiclass'/'best.pt',map_location=device,weights_only=False)
t2v=Time2Vec(k=16).to(device);encoder=EGATv2Encoder(edge_dim=EDGE_DIM).to(device)
t2v.load_state_dict(ckpt_g['t2v']);encoder.load_state_dict(ckpt_g['encoder'],strict=False)
for p in list(t2v.parameters())+list(encoder.parameters()):p.requires_grad=False
t2v.eval();encoder.eval()

# Time normalizer — MUST use saved values from training, not recompute
TMIN=ckpt_g['time_min'];TMAX=ckpt_g['time_max']

def extract_embeddings(graph_list):
    embs,lbls=[],[]
    with torch.no_grad():
        for g in graph_list:
            g=g.to(device);tn=(g.edge_time-TMIN)/(TMAX-TMIN);te=t2v(tn)
            ea58=torch.cat([g.edge_attr,te],dim=-1)
            d=Data(edge_index=g.edge_index,edge_attr=ea58,num_nodes=g.num_nodes)
            embs.append(encoder(d).cpu());lbls.append(g.y.cpu())
    return torch.cat(embs,dim=0),torch.cat(lbls,dim=0)

import gc

# Load, extract, delete sequentially to keep RAM low
print("Extracting train embeddings...")
G_train=torch.load(INPUT/'NF-CICIDS2018_train_list.pt',weights_only=False)+torch.load(INPUT/'NF-UNSW-NB15_train_list.pt',weights_only=False)
train_emb,train_lbl=extract_embeddings(G_train)
del G_train; gc.collect(); torch.cuda.empty_cache()

print("Extracting val embeddings...")
G_val=torch.load(INPUT/'NF-CICIDS2018_val_list.pt',weights_only=False)+torch.load(INPUT/'NF-UNSW-NB15_val_list.pt',weights_only=False)
val_emb,val_lbl=extract_embeddings(G_val)
del G_val; gc.collect(); torch.cuda.empty_cache()
print(f"Train: {train_emb.shape}, Val: {val_emb.shape}")

# Organize by class
train_by_cls={c:train_emb[train_lbl==c] for c in range(N_CLASSES) if (train_lbl==c).sum()>0}
val_by_cls={c:val_emb[val_lbl==c] for c in range(N_CLASSES) if (val_lbl==c).sum()>0}
attack_classes=[c for c in train_by_cls if UNIFIED[c]!='Benign']
print(f"Attack classes: {len(attack_classes)}")

# %% [cell 5] Prototypical Network with Attention
class AttentionPrototype(nn.Module):
    def __init__(self,ed=768):
        super().__init__();self.attn=nn.Sequential(nn.Linear(ed,128),nn.Tanh(),nn.Linear(128,1))
    def forward(self,support):
        w=F.softmax(self.attn(support).squeeze(-1),dim=0)
        return (support*w.unsqueeze(-1)).sum(dim=0)

class ProtoNet(nn.Module):
    def __init__(self,ed=768):
        super().__init__();self.ap=AttentionPrototype(ed)
    def forward(self,sup,query,sup_lbls,n_way):
        protos=torch.stack([self.ap(sup[sup_lbls==c]) for c in range(n_way)])
        return torch.mm(F.normalize(query,dim=-1),F.normalize(protos,dim=-1).t()),protos

# %% [cell 6] Episodic Training
HP={'n_way':5,'n_shot':5,'n_query':15,'ep_per_epoch':200,'epochs':30,'lr':1e-4}
pn=ProtoNet().to(device);opt=optim.Adam(pn.parameters(),lr=HP['lr'])
sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=HP['epochs'])

train_accs,val_accs=[],[]
print(f"Prototypical training: {HP['epochs']} epochs, {HP['ep_per_epoch']} eps/epoch")

for epoch in range(HP['epochs']):
    pn.train();ea=0.0
    for _ in range(HP['ep_per_epoch']):
        cls_sample=random.sample(attack_classes,HP['n_way'])
        sup,que=[],[];sl,ql=[],[]
        for li,cls in enumerate(cls_sample):
            cd=train_by_cls[cls];n=cd.shape[0]
            if n<HP['n_shot']+HP['n_query']:
                idx=torch.randint(0,n,(HP['n_shot']+HP['n_query'],))
            else:
                idx=torch.randperm(n)[:HP['n_shot']+HP['n_query']]
            sup.append(cd[idx[:HP['n_shot']]]);que.append(cd[idx[HP['n_shot']:]])
            sl.extend([li]*HP['n_shot']);ql.extend([li]*HP['n_query'])
        s_all=torch.cat(sup).to(device);q_all=torch.cat(que).to(device)
        s_l=torch.tensor(sl).to(device);q_l=torch.tensor(ql).to(device)
        logits,_=pn(s_all,q_all,s_l,HP['n_way'])
        loss=F.cross_entropy(logits,q_l)
        opt.zero_grad();loss.backward();opt.step()
        ea+=(logits.argmax(1)==q_l).float().mean().item()
    sched.step();train_accs.append(ea/HP['ep_per_epoch'])

    # Validation
    pn.eval();va,nv=0.0,0
    with torch.no_grad():
        for _ in range(20):
            cs=random.sample(attack_classes,HP['n_way'])
            sv,qv=[],[];slv,qlv=[],[]
            valid=True
            for li,cls in enumerate(cs):
                if cls not in val_by_cls:valid=False;break
                cd=val_by_cls[cls]
                if cd.shape[0]<HP['n_shot']+HP['n_query']:valid=False;break
                idx=torch.randperm(cd.shape[0])[:HP['n_shot']+HP['n_query']]
                sv.append(cd[idx[:HP['n_shot']]]);qv.append(cd[idx[HP['n_shot']:]])
                slv.extend([li]*HP['n_shot']);qlv.extend([li]*HP['n_query'])
            if not valid:continue
            s_all=torch.cat(sv).to(device);q_all=torch.cat(qv).to(device)
            logits,_=pn(s_all,q_all,torch.tensor(slv).to(device),HP['n_way'])
            va+=(logits.argmax(1)==torch.tensor(qlv).to(device)).float().mean().item();nv+=1
    val_accs.append(va/max(nv,1))
    print(f"Epoch {epoch+1:2d}: Train={train_accs[-1]:.4f} Val={val_accs[-1]:.4f}")

torch.save({'model':pn.state_dict(),'train_accs':train_accs,'val_accs':val_accs,'config':HP},CKPT_DIR/'best.pt')

# %% [cell 7] Leave-One-Class-Out Novelty Threshold Tuning
pn.eval()

# Per-class evaluation: for each held-out class, compute prototypes from OTHER classes only
novel_sims,known_sims=[],[]
locoo_results=[]

for held_out in attack_classes:
    if held_out not in val_by_cls:continue

    # Compute prototypes from ALL classes EXCEPT the held-out one
    held_protos={}
    with torch.no_grad():
        for cls in attack_classes:
            if cls==held_out:continue
            idx=torch.randperm(train_by_cls[cls].shape[0])[:5]
            proto=pn.ap(train_by_cls[cls][idx].to(device))
            held_protos[cls]=F.normalize(proto,dim=-1)

    proto_list=list(held_protos.values())
    if not proto_list:continue

    # Compute max similarity for held-out class (should be low → novel)
    nd=val_by_cls[held_out]
    held_sims=[]
    with torch.no_grad():
        for i in range(0,nd.shape[0],100):
            batch=F.normalize(nd[i:i+100].to(device),dim=-1)
            mx=torch.stack([torch.mm(batch,p.unsqueeze(-1)).squeeze() for p in proto_list]).max(dim=0).values
            held_sims.extend(mx.cpu().tolist())
    novel_sims.extend(held_sims)

    # Compute max similarity for KNOWN classes (should be high)
    held_known_sims=[]
    for cls in attack_classes:
        if cls==held_out or cls not in val_by_cls:continue
        kd=val_by_cls[cls][:100]
        with torch.no_grad():
            for i in range(0,kd.shape[0],100):
                batch=F.normalize(kd[i:i+100].to(device),dim=-1)
                mx=torch.stack([torch.mm(batch,p.unsqueeze(-1)).squeeze() for p in proto_list]).max(dim=0).values
                held_known_sims.extend(mx.cpu().tolist())
    known_sims.extend(held_known_sims)

    # Per-class tau from PR curve
    if held_sims and held_known_sims:
        ns=-np.array(held_known_sims+held_sims)
        yt=np.array([0]*len(held_known_sims)+[1]*len(held_sims))
        prec2,rec2,_=precision_recall_curve(yt,ns)
        f1v=2*prec2*rec2/(prec2+rec2+1e-10)
        best_idx=np.argmax(f1v)
        locoo_results.append({
            'held_out_class':UNIFIED[held_out],
            'f1':float(f1v[best_idx]),
            'precision':float(prec2[best_idx]),
            'recall':float(rec2[best_idx]),
        })

# ROC/PR from aggregated scores
novelty_scores=-np.array(known_sims+novel_sims)
y_true=np.array([0]*len(known_sims)+[1]*len(novel_sims))
fpr,tpr,_=roc_curve(y_true,novelty_scores);roc_auc=auc(fpr,tpr)
prec,rec,thresh=precision_recall_curve(y_true,novelty_scores);ap=average_precision_score(y_true,novelty_scores)

# Find best tau from PR curve: align F1 to threshold positions (thresh has N-1 elements)
f1s=2*prec[:len(thresh)]*rec[:len(thresh)]/(prec[:len(thresh)]+rec[:len(thresh)]+1e-10)
best_f1_idx=np.argmax(f1s)
best_tau=float(-thresh[best_f1_idx])  # actual PR-curve threshold, not recall value
# Fallback: use median similarity if PR-derived tau is unreasonable
if np.isnan(best_tau) or best_tau<=0 or best_tau>=1:
    best_tau=float(np.median(-np.array(novel_sims+known_sims)))
print(f"Global tau: {best_tau:.4f}, AUC: {roc_auc:.4f}, AP: {ap:.4f}")

# Figs + Tab09
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
ax1.plot(fpr,tpr,'b-',lw=2,label=f'AUC={roc_auc:.3f}');ax1.plot([0,1],[0,1],'k--',alpha=0.3)
ax1.set_xlabel('FPR');ax1.set_ylabel('TPR');ax1.set_title('Zero-Day ROC');ax1.legend();ax1.grid(alpha=0.3)
ax2.plot(rec,prec,'r-',lw=2,label=f'AP={ap:.3f}');ax2.set_xlabel('Recall');ax2.set_ylabel('Precision')
ax2.set_title('Zero-Day PR');ax2.legend();ax2.grid(alpha=0.3)
plt.tight_layout();plt.savefig(FIGS_DIR/'fig12_zero_day_roc_pr.png',dpi=300);plt.show()

# locoo_results already computed in Cell 7 above — save to tab09
pd.DataFrame(locoo_results).to_csv(TABS_DIR/'tab09_zero_day_results.csv',index=False)
pd.DataFrame(locoo_results).to_markdown(TABS_DIR/'tab09_zero_day_results.md',index=False)

with open(CKPT_DIR/'tau.json','w') as f:json.dump({'global_tau':float(best_tau),'roc_auc':float(roc_auc),'ap':float(ap)},f)
print(f"\nK05 DONE. AUC: {roc_auc:.4f}, AP: {ap:.4f}. Next: K06 (Eval+XAI)")
