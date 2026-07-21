"""
K03 — Binary Classification Stage-1 Head (Stage F)
=====================================================
KAGGLE T4x2 GPU. Two-phase training with PGD adversarial regularization.
Loads pretrained encoder from K01, trains binary head.

Prerequisite: K01 checkpoint + preprocessed graphs
Edge input: 58-dim (41 raw + 17 Time2Vec)
"""

# %% [cell 1]
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np, yaml, json, random, gc; from pathlib import Path
import warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score, recall_score
from torch_geometric.nn import GATv2Conv; from torch_geometric.data import Data

# %% [cell 2]
SEED=42;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED);torch.backends.cudnn.deterministic=True

WORKING=Path('/kaggle/working');INPUT=Path('/kaggle/input/datasets/mysteriousavailable/ids-nf3-processed/ids-nf3-processed')
CKPT_DIR=WORKING/'checkpoints'/'F_binary';LOGS_DIR=WORKING/'logs';FIGS_DIR=WORKING/'outputs'/'figures'
for d in [CKPT_DIR,LOGS_DIR,FIGS_DIR]:d.mkdir(parents=True,exist_ok=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [cell 3] Model definitions (matching K01/K02)
with open(INPUT/'feature_manifest.yaml') as f:fm=yaml.safe_load(f)
with open(INPUT/'label_map.yaml') as f:lm=yaml.safe_load(f)
UNIFIED=lm['unified_classes'];N_CLASSES=len(UNIFIED);EDGE_DIM=fm['final_edge_input_dim']

class Time2Vec(nn.Module):
    def __init__(self,k=16):
        super().__init__();self.k=k
        self.w0=nn.Parameter(torch.randn(1)*0.1);self.b0=nn.Parameter(torch.zeros(1))
        self.omega=nn.Parameter(10.0**(torch.rand(k)*6-3));self.bias=nn.Parameter(torch.zeros(k));self.output_dim=k+1
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

class BinaryHead(nn.Module):
    def __init__(self,in_dim=768,hidden=256,bottleneck=64,nc=2):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,hidden),nn.ELU(),nn.Dropout(0.3),
                               nn.Linear(hidden,bottleneck),nn.ELU(),nn.Dropout(0.2),nn.Linear(bottleneck,nc))
    def forward(self,x):return self.net(x)

class FocalLoss(nn.Module):
    def __init__(self,gamma=2.0,alpha=None):
        super().__init__();self.gamma=gamma;self.alpha=alpha
    def forward(self,logits,targets):
        ce=F.cross_entropy(logits,targets,reduction='none');pt=torch.exp(-ce)
        focal=(1-pt)**self.gamma*ce
        if self.alpha is not None:focal=self.alpha[targets]*focal
        return focal.mean()

# %% [cell 4] Load encoder & training data
ckpt=torch.load(WORKING/'checkpoints'/'D_mae_pretrain'/'best.pt',map_location=device,weights_only=False)
t2v=Time2Vec(k=16).to(device);encoder=EGATv2Encoder(edge_dim=EDGE_DIM).to(device)
t2v.load_state_dict(ckpt['t2v']);encoder.load_state_dict(ckpt['encoder'],strict=False)
TIME_MIN=ckpt['time_min'];TIME_MAX=ckpt['time_max']
def norm_time(t):return(t-TIME_MIN)/(TIME_MAX-TIME_MIN)

def load_graphs(n,s):
    p=INPUT/f'{n}_{s}_list.pt';return torch.load(p,weights_only=False) if p.exists() else []
G_train=load_graphs('NF-CICIDS2018','train')+load_graphs('NF-UNSW-NB15','train')
G_val=load_graphs('NF-CICIDS2018','val')+load_graphs('NF-UNSW-NB15','val')
print(f"G_train: {len(G_train)} windows | G_val: {len(G_val)} windows")
gc.collect(); torch.cuda.empty_cache()  # clean up after torch.load

# Class weights + feature bounds — compute iteratively (no giant cat tensors)
n_benign=0; n_attack=0
fmins_41=torch.full((41,), float('inf'))
fmaxs_41=torch.full((41,),-float('inf'))
for g in G_train:
    n_benign+=(g.y==0).sum().item(); n_attack+=(g.y!=0).sum().item()
    fmins_41=torch.min(fmins_41,g.edge_attr.min(dim=0).values)
    fmaxs_41=torch.max(fmaxs_41,g.edge_attr.max(dim=0).values)
alpha_w=torch.tensor([n_attack/(n_benign+n_attack),n_benign/(n_benign+n_attack)],device=device)
fmins_58=torch.cat([fmins_41,torch.full((17,),-4.0)]).to(device)
fmaxs_58=torch.cat([fmaxs_41,torch.full((17,),4.0)]).to(device)
print(f"Benign: {n_benign:,} | Attack: {n_attack:,} | Focal alpha: {alpha_w.tolist()}")

# %% [cell 5] Helper: encode edges with Time2Vec
def encode_edges(ea44,et,model,t2v,ei,nn_nodes):
    tn=norm_time(et);te=t2v(tn);ea58=torch.cat([ea44,te],dim=-1)
    d=Data(edge_index=ei,edge_attr=ea58,num_nodes=nn_nodes);return model(d),ea58

# %% [cell 6] Phase A: Frozen Encoder
HP={'phaseA_lr':1e-3,'phaseA_epochs':5,'phaseB_lr_enc':1e-5,'phaseB_lr_head':1e-4,
    'phaseB_epochs':15,'focal_gamma':2.0,'pgd_eps':0.03,'pgd_alpha':0.01,'pgd_steps':7,
    'pgd_frac':0.30,'batch':4096,'us_ratio':2.0}

binary_head=BinaryHead().to(device);focal=FocalLoss(gamma=HP['focal_gamma'],alpha=alpha_w)
for p in list(t2v.parameters())+list(encoder.parameters()):p.requires_grad=False
optA=optim.Adam(binary_head.parameters(),lr=HP['phaseA_lr'])
schedA=optim.lr_scheduler.CosineAnnealingLR(optA,T_max=HP['phaseA_epochs'])
amp=GradScaler();phaseA_losses=[]

print("Phase A: Train head (frozen encoder)")
for epoch in range(HP['phaseA_epochs']):
    binary_head.train();el,nb=0.0,0
    random.shuffle(G_train)  # prevent dataset-order bias (CICIDS→UNSW fixed order)
    for g in G_train:
        g=g.to(device)
        if g.edge_index.shape[1]<4:continue
        benign_idx=(g.y==0).nonzero(as_tuple=True)[0];attack_idx=(g.y!=0).nonzero(as_tuple=True)[0]
        n_ak=attack_idx.shape[0];n_bk=min(int(n_ak*HP['us_ratio']),benign_idx.shape[0])
        if n_bk<4:continue
        bk=benign_idx[torch.randperm(benign_idx.shape[0])[:n_bk]]
        keep=torch.cat([bk,attack_idx])
        for i in range(0,len(keep),HP['batch']):
            bi=keep[i:i+HP['batch']];n_edges=len(bi)
            if n_edges<4:continue
            with autocast():
                reps,_=encode_edges(g.edge_attr[bi],g.edge_time[bi],encoder,t2v,g.edge_index[:,bi],g.num_nodes)
                loss=focal(binary_head(reps),(g.y[bi]!=0).long())
            optA.zero_grad();amp.scale(loss).backward()
            amp.unscale_(optA)  # unscale BEFORE clipping
            torch.nn.utils.clip_grad_norm_(binary_head.parameters(),1.0)
            amp.step(optA)
            if not torch.isnan(loss):amp.update();el+=loss.item();nb+=1
    schedA.step();avg=el/max(nb,1);phaseA_losses.append(avg)
    gc.collect(); torch.cuda.empty_cache()
    print(f"  Epoch {epoch+1}: Loss={avg:.6f}")

# %% [cell 7] Phase B: Joint Fine-Tune with PGD
for p in list(t2v.parameters())+list(encoder.parameters()):p.requires_grad=True
optB=optim.Adam([{'params':t2v.parameters(),'lr':HP['phaseB_lr_enc']},
                  {'params':encoder.parameters(),'lr':HP['phaseB_lr_enc']},
                  {'params':binary_head.parameters(),'lr':HP['phaseB_lr_head']}])
schedB=optim.lr_scheduler.CosineAnnealingLR(optB,T_max=HP['phaseB_epochs'])
phaseB_losses,val_f1s=[],[];best_f1=0.0

print("\nPhase B: Joint fine-tune with PGD")
for epoch in range(HP['phaseB_epochs']):
    t2v.train();encoder.train();binary_head.train();el,nb=0.0,0
    random.shuffle(G_train)  # prevent dataset-order bias
    for g in G_train:
        g=g.to(device)
        if g.edge_index.shape[1]<4:continue
        benign_idx=(g.y==0).nonzero(as_tuple=True)[0];attack_idx=(g.y!=0).nonzero(as_tuple=True)[0]
        n_ak=attack_idx.shape[0];n_bk=min(int(n_ak*HP['us_ratio']),benign_idx.shape[0])
        if n_bk<4:continue
        bk=benign_idx[torch.randperm(benign_idx.shape[0])[:n_bk]];keep=torch.cat([bk,attack_idx])
        for i in range(0,len(keep),HP['batch']):
            bi=keep[i:i+HP['batch']];n_edges=len(bi)
            if n_edges<4:continue
            # Build features and run PGD OUTSIDE autocast (fp32 for accurate attack gradients)
            tn=norm_time(g.edge_time[bi]);te=t2v(tn)
            ea58=torch.cat([g.edge_attr[bi],te],dim=-1)
            # PGD adversarial attack on 30% of batch (fp32 precision)
            n_pgd=int(n_edges*HP['pgd_frac'])
            if n_pgd>0:
                pgd_mask=torch.zeros(n_edges,dtype=torch.bool,device=device)
                pgd_mask[:n_pgd]=True;pgd_mask=pgd_mask[torch.randperm(n_edges)]
                ea58_orig=ea58.clone()
                for _ in range(HP['pgd_steps']):
                    ea58_pert=ea58.clone().detach().requires_grad_(True)
                    d_pert=Data(edge_index=g.edge_index[:,bi],edge_attr=ea58_pert,num_nodes=g.num_nodes)
                    reps_pert=encoder(d_pert)
                    logits_pert=binary_head(reps_pert)
                    loss_adv=-focal(logits_pert[pgd_mask],(g.y[bi][pgd_mask]!=0).long())
                    grad=torch.autograd.grad(loss_adv,ea58_pert)[0]
                    with torch.no_grad():
                        ea58[pgd_mask]+=HP['pgd_alpha']*grad[pgd_mask].sign()
                        delta=torch.clamp(ea58[pgd_mask]-ea58_orig[pgd_mask],-HP['pgd_eps'],HP['pgd_eps'])
                        ea58[pgd_mask]=ea58_orig[pgd_mask]+delta
                        ea58=torch.clamp(ea58,fmins_58,fmaxs_58)
            # Main forward pass in autocast (fp16 for throughput)
            with autocast():
                d_final=Data(edge_index=g.edge_index[:,bi],edge_attr=ea58,num_nodes=g.num_nodes)
                reps=encoder(d_final)
                loss=focal(binary_head(reps),(g.y[bi]!=0).long())
            optB.zero_grad();amp.scale(loss).backward()
            amp.unscale_(optB)  # unscale BEFORE clipping
            torch.nn.utils.clip_grad_norm_(list(t2v.parameters())+list(encoder.parameters())+list(binary_head.parameters()),1.0)
            amp.step(optB)
            if not torch.isnan(loss):amp.update()  # skip scale update on NaN to avoid underflow
            el+=loss.item();nb+=1
    schedB.step();avg=el/max(nb,1);phaseB_losses.append(avg)
    gc.collect(); torch.cuda.empty_cache()

    # Validation
    t2v.eval();encoder.eval();binary_head.eval();vp,vt=[],[]
    with torch.no_grad():
        val_sample=random.sample(G_val,min(10,len(G_val)))
        for g in val_sample:
            g=g.to(device)
            if g.edge_index.shape[1]>5000:
                idx=torch.randperm(g.edge_index.shape[1])[:5000]
                g.edge_index=g.edge_index[:,idx];g.edge_attr=g.edge_attr[idx];g.edge_time=g.edge_time[idx]
            reps,_=encode_edges(g.edge_attr,g.edge_time,encoder,t2v,g.edge_index,g.num_nodes)
            preds=binary_head(reps).argmax(dim=1);vp.extend(preds.cpu().tolist());vt.extend((g.y!=0).long().cpu().tolist())
    vf1=f1_score(vt,vp,average='macro');val_f1s.append(vf1)
    print(f"  Epoch {epoch+1:2d}: Loss={avg:.6f} Val-F1={vf1:.4f}")
    if vf1>best_f1:
        best_f1=vf1
        torch.save({'t2v':t2v.state_dict(),'encoder':encoder.state_dict(),'head':binary_head.state_dict(),
                    'val_f1':vf1,'config':HP,'time_min':TIME_MIN,'time_max':TIME_MAX},CKPT_DIR/'best.pt')

# %% [cell 8] Threshold Calibration
t2v.eval();encoder.eval();binary_head.eval();vprobs,vtargets=[],[]
with torch.no_grad():
    for g in G_val:
        g=g.to(device)
        if g.edge_index.shape[1]>10000:
            idx=torch.randperm(g.edge_index.shape[1])[:10000]
            g.edge_index=g.edge_index[:,idx];g.edge_attr=g.edge_attr[idx];g.edge_time=g.edge_time[idx];g.y=g.y[idx]
        reps,_=encode_edges(g.edge_attr,g.edge_time,encoder,t2v,g.edge_index,g.num_nodes)
        probs=F.softmax(binary_head(reps),dim=-1)[:,1];vprobs.extend(probs.cpu().tolist());vtargets.extend((g.y!=0).long().cpu().tolist())

vprobs=np.array(vprobs);vtargets=np.array(vtargets)
best_thr,best_f1_thr=0.5,0.0
for thr in np.arange(0.05,0.95,0.025):
    preds=(vprobs>=thr).astype(int);rec=recall_score(vtargets,preds)
    f1v=f1_score(vtargets,preds)
    if rec>=0.995 and f1v>best_f1_thr:best_f1_thr=f1v;best_thr=thr
if best_f1_thr==0.0:
    best_idx=np.argmax([recall_score(vtargets,(vprobs>=t).astype(int)) for t in np.arange(0.05,0.95,0.025)])
    best_thr=np.arange(0.05,0.95,0.025)[best_idx];print(f"WARNING: No threshold reached recall>=0.995")
print(f"Threshold: {best_thr:.3f} (recall={(vprobs>=best_thr).astype(int).mean():.4f})")
with open(CKPT_DIR/'threshold.json','w') as f:json.dump({'threshold':float(best_thr),'val_f1':float(best_f1)},f)

log={'notebook':'K03','stage':'F','best_val_f1':float(best_f1),'threshold':float(best_thr),
     'phaseA_epochs':len(phaseA_losses),'phaseB_epochs':len(phaseB_losses)}
with open(LOGS_DIR/'k03_log.json','w') as f:json.dump(log,f,indent=2)
print(f"K03 DONE. Best F1: {best_f1:.4f}, Threshold: {best_thr:.3f}")
print(f"Next: K04 (Multiclass)")
