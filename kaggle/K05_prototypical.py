"""
K05 — Prototypical Few-Shot / Zero-Day Detection (Stage H)
=============================================================
KAGGLE T4x2. Loads K04 encoder (frozen). Episodic training + novelty threshold tuning.
"""

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np, pandas as pd, yaml, json, random
from datetime import datetime, timezone; from pathlib import Path; from collections import defaultdict
from torch_geometric.nn import GATv2Conv; from torch_geometric.data import Data
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve, auc, average_precision_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

INPUT_DIR=Path('/kaggle/input/ids-processed'); WORKING=Path('/kaggle/working')
CKPT_DIR=WORKING/'checkpoints'/'H_prototypical'; LOGS_DIR=WORKING/'logs'
FIGS_DIR=WORKING/'outputs'/'figures'; TABS_DIR=WORKING/'outputs'/'tables'
for d in [CKPT_DIR,LOGS_DIR,FIGS_DIR,TABS_DIR]: d.mkdir(parents=True,exist_ok=True)
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

class PrototypicalNetwork(nn.Module):
    def __init__(self,embed_dim=768):
        super().__init__()
        self.attn=nn.Sequential(nn.Linear(embed_dim,128),nn.Tanh(),nn.Linear(128,1))
    def compute_proto(self,support_embeddings):
        s=self.attn(support_embeddings).squeeze(-1); w=F.softmax(s,dim=0)
        return (support_embeddings*w.unsqueeze(-1)).sum(dim=0)
    def forward(self,support,query,s_labels,n_way):
        protos=torch.stack([self.compute_proto(support[s_labels==c]) for c in range(n_way)],dim=0)
        return torch.mm(F.normalize(query,p=2,dim=-1),F.normalize(protos,p=2,dim=-1).t()),protos

# ---- Load K04 encoder (frozen) ----
ckpt=torch.load(WORKING/'checkpoints'/'G_multiclass'/'best.pt',map_location=device,weights_only=False)
t2v=Time2Vec(k=16).to(device); enc=EGATv2Encoder(edge_dim=EDGE_DIM).to(device)
t2v.load_state_dict(ckpt['t2v']); enc.load_state_dict(ckpt['enc'])
for p in list(t2v.parameters())+list(enc.parameters()): p.requires_grad=False
t2v.eval(); enc.eval()

# ---- Load data & extract embeddings ----
G_train=[]; G_val=[]
for ds in ['NF-CICIDS2018','NF-UNSW-NB15']:
    for sp,lst in [('train',G_train),('val',G_val)]:
        p=INPUT_DIR/f'{ds}_{sp}_list.pt'
        if p.exists(): lst.extend(torch.load(p,weights_only=False))

all_times=torch.cat([g.edge_time for g in G_train]); T_MIN,T_MAX=all_times.min().item(),all_times.max().item()
def nt(t): return (t-T_MIN)/(T_MAX-T_MIN)

def extract_embeddings(graphs):
    embs, lbs = [], []
    with torch.no_grad():
        for g in graphs:
            g=g.to(device); tn=nt(g.edge_time); te=t2v(tn)
            ef=torch.cat([g.edge_attr,te],dim=-1)
            reps=enc(Data(edge_index=g.edge_index,edge_attr=ef,num_nodes=g.num_nodes))
            embs.append(reps.cpu()); lbs.append(g.y.cpu())
    return torch.cat(embs,dim=0), torch.cat(lbs,dim=0)

print("Extracting embeddings...")
tr_emb,tr_lab=extract_embeddings(G_train); val_emb,val_lab=extract_embeddings(G_val)
cls_emb={c:tr_emb[tr_lab==c] for c in range(NC) if (tr_lab==c).sum()>0}
val_cls={c:val_emb[val_lab==c] for c in range(NC) if (val_lab==c).sum()>0}
attack_cls=[c for c in range(NC) if UNIFIED[c]!='Benign' and c in cls_emb]
print(f"Attack classes: {len(attack_cls)}")

# ---- Episodic training ----
proto_net=PrototypicalNetwork().to(device); opt=optim.Adam(proto_net.parameters(),lr=1e-4)
HP={'n_way':5,'n_shot':5,'n_query':15,'ep_per_ep':200,'epochs':30}
sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=HP['epochs'])

def sample_episode():
    sampled=random.sample(attack_cls,HP['n_way'])
    sup,que=[],[]
    for li,c in enumerate(sampled):
        cd=cls_emb[c]; n=cd.shape[0]; need=HP['n_shot']+HP['n_query']
        ix=torch.randint(0,n,(need,)) if n<need else torch.randperm(n)[:need]
        sup.append(cd[ix[:HP['n_shot']]]); que.append(cd[ix[HP['n_shot']:]])
    return torch.cat(sup,dim=0),torch.cat(que,dim=0),torch.arange(HP['n_way']).repeat_interleave(HP['n_shot']),torch.arange(HP['n_way']).repeat_interleave(HP['n_query'])

print(f"Training {HP['epochs']} epochs...")
for ep in range(HP['epochs']):
    proto_net.train(); acc=0.0
    for _ in range(HP['ep_per_ep']):
        s,q,sl,ql=sample_episode(); s,q,sl,ql=s.to(device),q.to(device),sl.to(device),ql.to(device)
        lo,_=proto_net(s,q,sl,HP['n_way']); loss=F.cross_entropy(lo,ql)
        opt.zero_grad(); loss.backward(); opt.step()
        acc+=(lo.argmax(dim=1)==ql).float().mean().item()
    sch.step()
    if (ep+1)%10==0: print(f"  Ep {ep+1}: Acc={acc/HP['ep_per_ep']:.4f}")

# ---- Leave-one-class-out novelty tuning ----
proto_net.eval(); all_protos={}
with torch.no_grad():
    for c in attack_cls:
        cd=cls_emb[c]; ix=torch.randperm(cd.shape[0])[:5]
        all_protos[c]=F.normalize(proto_net.compute_proto(cd[ix].to(device)),p=2,dim=-1)

looco=[]; novel_sims=[]; known_sims=[]
for ho in attack_cls:
    if ho not in val_cls: continue
    nd=val_cls[ho]; kd=torch.cat([val_cls[c][:100] for c in attack_cls if c!=ho and c in val_cls],dim=0)
    with torch.no_grad():
        nn_=-(F.normalize(nd.to(device),p=2,dim=-1)@torch.stack(list(all_protos.values())).T).max(dim=1).values.cpu()
        kn_=-(F.normalize(kd.to(device),p=2,dim=-1)@torch.stack(list(all_protos.values())).T).max(dim=1).values.cpu()
    novel_sims.extend(nn_.tolist()); known_sims.extend(kn_.tolist())
    yt=np.array([0]*len(kn_)+[1]*len(nn_)); sc=np.concatenate([kn_,nn_])
    pr,rc,th=precision_recall_curve(yt,sc); fs=2*pr*rc/(pr+rc+1e-10); bi=np.argmax(fs)
    looco.append({'held_out':UNIFIED[ho],'tau':float(th[bi] if bi<len(th) else 0.5),
                  'precision':float(pr[bi]),'recall':float(rc[bi]),'f1':float(fs[bi])})
    print(f"  {UNIFIED[ho]:25s}: τ={looco[-1]['tau']:.3f} F1={looco[-1]['f1']:.4f}")

global_tau=float(np.median([r['tau'] for r in looco]))
torch.save({'model':proto_net.state_dict(),'config':HP},CKPT_DIR/'best.pt')
json.dump({'global_tau':global_tau,'per_class':looco},open(CKPT_DIR/'tau.json','w'),indent=2)

# ---- Fig 12: ROC/PR ----
all_ns=-np.array(known_sims+novel_sims); all_yt=np.array([0]*len(known_sims)+[1]*len(novel_sims))
fpr,tpr,_=roc_curve(all_yt,all_ns); roc_auc=auc(fpr,tpr)
prec,rec,_=precision_recall_curve(all_yt,all_ns); ap_=average_precision_score(all_yt,all_ns)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
ax1.plot(fpr,tpr,'b-',lw=2,label=f'AUC={roc_auc:.3f}'); ax1.plot([0,1],[0,1],'k--',alpha=0.3)
ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR'); ax1.set_title('ROC'); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(rec,prec,'r-',lw=2,label=f'AP={ap_:.3f}'); ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.set_title('PR Curve'); ax2.legend(); ax2.grid(alpha=0.3)
fig.suptitle('Fig 12: Zero-Day Detection'); plt.tight_layout()
plt.savefig(FIGS_DIR/'fig12_zero_day_roc_pr.png',dpi=150); plt.close()

pd.DataFrame(looco).to_csv(TABS_DIR/'tab09_zero_day_results.csv',index=False)
json.dump({'notebook':'K05','global_tau':float(global_tau),'auc':float(roc_auc),'ap':float(ap_)},
          open(LOGS_DIR/'k05_log.json','w'),indent=2,default=str)
print(f"✓ K05 complete. τ={global_tau:.4f} AUC={roc_auc:.4f} AP={ap_:.4f}")
