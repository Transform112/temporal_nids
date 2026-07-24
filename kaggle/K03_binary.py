"""
K03 — Binary Classification Stage-1 Head (Stage F)
=====================================================
KAGGLE T4x2 GPU. Two-phase training with PGD adversarial regularization.
Loads pretrained encoder from K01, trains binary head.

FIXES vs previous version:
1. rankdom -> random typo (import would crash before ever reaching training).
2. Encoder class now byte-identical to K01/K02 (constant ones node init, no
   max_nodes/node_embed/deg_proj) -> checkpoint loads with strict=True, no more
   silent architecture mismatch.
3. phaseB_lr_enc 5e-6 -> 5e-5 (old value was too small to move the encoder at all,
   confirmed by flat Phase B F1 in the earlier run).
4. BatchNorm recalibration pass added before threshold search: Phase A/B train on
   an undersampled (~1.5:1) benign:attack ratio, but val is the natural (~6:1)
   ratio. BN running stats absorb the training-time ratio, skewing eval-time
   probabilities. A short train()-mode, no-grad pass over G_val resets BN stats
   to the true inference distribution before calibrating the threshold.

Prerequisite: K01 checkpoint + preprocessed graphs
Edge input: 58-dim (41 raw + 17 Time2Vec)
"""

# %% [cell 1]
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np, yaml, json, random, gc; from pathlib import Path
import warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score, recall_score, precision_score
from torch_geometric.nn import GATv2Conv; from torch_geometric.data import Data
import sys
from pathlib import Path
MODELS_PATH = Path('/kaggle/input/datasets/harshitpachahara/models-py')
if MODELS_PATH.exists() and str(MODELS_PATH) not in sys.path:
    sys.path.append(str(MODELS_PATH))
from models import Time2Vec, EGATv2Encoder, ClassifierHead, FocalLoss


# %% [cell 2]
SEED=42;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED);torch.backends.cudnn.deterministic=True

WORKING=Path('../working');INPUT = Path('/kaggle/input/datasets/mysteriousavailable/ids-nf3-processed/ids-nf3-processed')
CKPT_DIR=WORKING/'checkpoints'/'F_binary';LOGS_DIR=WORKING/'logs';FIGS_DIR=WORKING/'outputs'/'figures'
for d in [CKPT_DIR,LOGS_DIR,FIGS_DIR]:d.mkdir(parents=True,exist_ok=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [cell 3] Model definitions — MUST match K01/K02 exactly (copy-paste, don't redefine)
with open(INPUT/'feature_manifest.yaml') as f:fm=yaml.safe_load(f)
with open(INPUT/'label_map.yaml') as f:lm=yaml.safe_load(f)
UNIFIED=lm['unified_classes'];N_CLASSES=len(UNIFIED);EDGE_DIM=fm['final_edge_input_dim']





# %% [cell 4] Load encoder & training data
K01_CKPT_DIR=Path('/kaggle/input/datasets/harshitpachahara/k01-output/checkpoints/D_mae_pretrain')
ckpt=torch.load(K01_CKPT_DIR/'best.pt',map_location=device,weights_only=False)
if 't2v' not in globals(): t2v = Time2Vec(k=16).to(device);encoder=EGATv2Encoder(edge_dim=EDGE_DIM).to(device)
t2v.load_state_dict(ckpt['t2v']);encoder.load_state_dict(ckpt['encoder'])  # strict=True — same class now
TIME_MIN=ckpt['time_min'];TIME_MAX=ckpt['time_max']
del ckpt; gc.collect()
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
alpha_w=torch.tensor([0.5, 0.5],device=device)  # uniform: undersampling handles the balance
fmins_58=torch.cat([fmins_41,torch.full((17,),-4.0)]).to(device)
fmaxs_58=torch.cat([fmaxs_41,torch.full((17,),4.0)]).to(device)
print(f"Benign: {n_benign:,} | Attack: {n_attack:,} | Focal alpha: {alpha_w.tolist()} (uniform, undersampling handles balance)")

# %% [cell 5] Helper: encode edges with Time2Vec
def encode_edges(ea44,et,model,t2v,ei,nn_nodes):
    tn=norm_time(et.float());te=t2v(tn);ea58=torch.cat([ea44.float(),te],dim=-1)
    d=Data(edge_index=ei,edge_attr=ea58,num_nodes=nn_nodes);return model(d),ea58

# %% [cell 6] Phase A: Frozen Encoder
HP={'phaseA_lr':1e-3,'phaseA_epochs':5,'phaseB_lr_enc':5e-5,'phaseB_lr_head':2e-4,
    'phaseB_epochs':20,'focal_gamma':2.0,'pgd_eps':0.03,'pgd_alpha':0.01,'pgd_steps':3,
    'pgd_frac':0.10,'batch':4096,'us_ratio':1.5,'patience':5}

if 'binary_head' not in globals(): binary_head = ClassifierHead(out_dim=2).to(device);focal=FocalLoss(gamma=HP['focal_gamma'],alpha=alpha_w)
for p in list(t2v.parameters())+list(encoder.parameters()):p.requires_grad=False
optA=optim.Adam(binary_head.parameters(),lr=HP['phaseA_lr'])
schedA=optim.lr_scheduler.CosineAnnealingLR(optA,T_max=HP['phaseA_epochs'])
amp=GradScaler();phaseA_losses=[]

print("Phase A: Train head (frozen encoder)")
for epoch in range(HP['phaseA_epochs']):
    binary_head.train();el,nb=0.0,0
    random.shuffle(G_train)  # prevent dataset-order bias (CICIDS->UNSW fixed order)
    for g in G_train:
        if g.edge_index.shape[1]<4:continue
        benign_idx=(g.y==0).nonzero(as_tuple=True)[0];attack_idx=(g.y!=0).nonzero(as_tuple=True)[0]
        n_ak=attack_idx.shape[0]
        # Balanced sampling: keep us_ratio benign per attack in mixed windows,
        # and a small sample (500) from pure-benign windows for baseline learning
        if n_ak > 0:
            n_bk=min(int(n_ak*HP['us_ratio']), benign_idx.shape[0])
        else:
            n_bk=min(250, benign_idx.shape[0])
        if n_bk < 4 and n_ak < 4: continue
        bk=benign_idx[torch.randperm(benign_idx.shape[0])[:n_bk]]
        keep=torch.cat([bk,attack_idx])
        for i in range(0,len(keep),HP['batch']):
            bi=keep[i:i+HP['batch']];n_edges=len(bi)
            if n_edges<4:continue
            with autocast():
                reps,_=encode_edges(g.edge_attr[bi].to(device),g.edge_time[bi].to(device),encoder,t2v,g.edge_index[:,bi].to(device),g.num_nodes)
                loss=focal(binary_head(reps),(g.y[bi].to(device)!=0).long())
            optA.zero_grad();amp.scale(loss).backward()
            amp.unscale_(optA)  # unscale BEFORE clipping
            torch.nn.utils.clip_grad_norm_(binary_head.parameters(),1.0)
            amp.step(optA)
            amp.update();el+=loss.item();nb+=1
            if nb % 500 == 0: print(f"    [Phase A] Epoch {epoch+1} - Batch {nb} - Loss: {el/nb:.6f}")
    schedA.step();avg=el/max(nb,1);phaseA_losses.append(avg)
    gc.collect(); torch.cuda.empty_cache()
    print(f"  Epoch {epoch+1}: Loss={avg:.6f}")

# %% [cell 7] Phase B: Joint Fine-Tune with PGD
for p in list(t2v.parameters())+list(encoder.parameters()):p.requires_grad=True
optB=optim.Adam([{'params':t2v.parameters(),'lr':HP['phaseB_lr_enc']},
                  {'params':encoder.parameters(),'lr':HP['phaseB_lr_enc']},
                  {'params':binary_head.parameters(),'lr':HP['phaseB_lr_head']}])
schedB=optim.lr_scheduler.CosineAnnealingLR(optB,T_max=HP['phaseB_epochs'])
phaseB_losses,val_f1s=[],[];best_f1=0.0;no_improve=0

print("\nPhase B: Joint fine-tune with PGD")
for epoch in range(HP['phaseB_epochs']):
    t2v.train();encoder.train();binary_head.train();el,nb=0.0,0
    random.shuffle(G_train)  # prevent dataset-order bias
    for g in G_train:
        if g.edge_index.shape[1]<4:continue
        benign_idx=(g.y==0).nonzero(as_tuple=True)[0];attack_idx=(g.y!=0).nonzero(as_tuple=True)[0]
        n_ak=attack_idx.shape[0]
        if n_ak > 0:
            n_bk=min(int(n_ak*HP['us_ratio']), benign_idx.shape[0])
        else:
            n_bk=min(250, benign_idx.shape[0])
        if n_bk < 4 and n_ak < 4: continue
        bk=benign_idx[torch.randperm(benign_idx.shape[0])[:n_bk]];keep=torch.cat([bk,attack_idx])
        for i in range(0,len(keep),HP['batch']):
            bi=keep[i:i+HP['batch']];n_edges=len(bi)
            if n_edges<4:continue
            # Build features and run PGD OUTSIDE autocast (fp32 for accurate attack gradients)
            tn=norm_time(g.edge_time[bi].float().to(device));te=t2v(tn)
            ea58=torch.cat([g.edge_attr[bi].float().to(device),te],dim=-1)
            # PGD adversarial attack on a fraction of the batch (fp32 precision)
            n_pgd=int(n_edges*HP['pgd_frac'])
            if n_pgd>0:
                pgd_mask=torch.zeros(n_edges,dtype=torch.bool,device=device)
                pgd_mask[:n_pgd]=True;pgd_mask=pgd_mask[torch.randperm(n_edges)]
                ea58_orig=ea58.clone()
                for _ in range(HP['pgd_steps']):
                    ea58_pert=ea58.clone().detach().requires_grad_(True)
                    d_pert=Data(edge_index=g.edge_index[:,bi].to(device),edge_attr=ea58_pert,num_nodes=g.num_nodes)
                    reps_pert=encoder(d_pert)
                    logits_pert=binary_head(reps_pert)
                    loss_adv=focal(logits_pert[pgd_mask],(g.y[bi].to(device)[pgd_mask]!=0).long())
                    grad=torch.autograd.grad(loss_adv,ea58_pert)[0]
                    with torch.no_grad():
                        ea58[pgd_mask]+=HP['pgd_alpha']*grad[pgd_mask].sign()
                        delta=torch.clamp(ea58[pgd_mask]-ea58_orig[pgd_mask],-HP['pgd_eps'],HP['pgd_eps'])
                        ea58[pgd_mask]=ea58_orig[pgd_mask]+delta
                        ea58=torch.clamp(ea58,fmins_58,fmaxs_58)
            # Main forward pass in autocast (fp16 for throughput)
            with autocast():
                d_final=Data(edge_index=g.edge_index[:,bi].to(device),edge_attr=ea58,num_nodes=g.num_nodes)
                reps=encoder(d_final)
                loss=focal(binary_head(reps),(g.y[bi].to(device)!=0).long())
            optB.zero_grad();amp.scale(loss).backward()
            amp.unscale_(optB)  # unscale BEFORE clipping
            torch.nn.utils.clip_grad_norm_(list(t2v.parameters())+list(encoder.parameters())+list(binary_head.parameters()),1.0)
            amp.step(optB)
            amp.update()  # skip scale update on NaN to avoid underflow
            el+=loss.item();nb+=1
            if nb % 500 == 0: print(f"    [Phase B] Epoch {epoch+1} - Batch {nb} - Loss: {el/nb:.6f}")
    schedB.step();avg=el/max(nb,1);phaseB_losses.append(avg)
    gc.collect(); torch.cuda.empty_cache()

    # Validation — use optimal threshold search instead of hardcoded argmax at 0.5
    t2v.eval();encoder.eval();binary_head.eval()
    val_probs_ep,val_targets_ep=[],[]
    with torch.no_grad():
        for g in G_val:
            ei,ea,et,y = g.edge_index,g.edge_attr,g.edge_time,g.y
            reps,_=encode_edges(ea.to(device),et.to(device),encoder,t2v,ei.to(device),g.num_nodes)
            probs=F.softmax(binary_head(reps),dim=-1)[:,1]
            val_probs_ep.extend(probs.cpu().tolist());val_targets_ep.extend((y!=0).long().cpu().tolist())
    val_probs_np=np.array(val_probs_ep);val_targets_np=np.array(val_targets_ep)
    best_epoch_f1=0.0;best_epoch_thr=0.5
    for t in np.arange(0.05,0.95,0.05):
        ep_preds=(val_probs_np>=t).astype(int)
        ef1=f1_score(val_targets_np,ep_preds,average='macro')
        if ef1>best_epoch_f1:best_epoch_f1=ef1;best_epoch_thr=t
    vf1=best_epoch_f1;val_f1s.append(vf1)
    print(f"  Epoch {epoch+1:2d}: Loss={avg:.6f} Val-F1={vf1:.4f} (thr={best_epoch_thr:.2f})")
    if vf1>best_f1:
        best_f1=vf1;no_improve=0
        torch.save({'t2v':t2v.state_dict(),'encoder':encoder.state_dict(),'head':binary_head.state_dict(),
                    'val_f1':vf1,'config':HP,'time_min':TIME_MIN,'time_max':TIME_MAX},CKPT_DIR/'best.pt')
    else:
        no_improve+=1
        if no_improve>=HP['patience']:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {HP['patience']} epochs)")
            break

# %% [cell 8] Threshold Calibration
print("\nReloading best checkpoint for threshold calibration...")
best_ckpt=torch.load(CKPT_DIR/'best.pt',map_location=device,weights_only=False)
t2v.load_state_dict(best_ckpt['t2v']);encoder.load_state_dict(best_ckpt['encoder'])
binary_head.load_state_dict(best_ckpt['head'])

# BN recalibration: Phase A/B trained on an undersampled (~1.5:1) benign:attack
# ratio, but real inference sees the natural (~6:1) ratio. BatchNorm's running
# mean/var reflect whatever marginal input distribution they were updated on, so
# they're currently skewed toward the undersampled mix. Run a short train()-mode,
# no-grad pass over G_val (natural ratio) to reset them before calibrating.
t2v.eval();encoder.eval();binary_head.train()
with torch.no_grad():
    for g in G_val:
        ei,ea,et,y = g.edge_index,g.edge_attr,g.edge_time,g.y
        reps,_=encode_edges(ea.to(device),et.to(device),encoder,t2v,ei.to(device),g.num_nodes)
        _ = binary_head(reps)
binary_head.eval()

vprobs,vtargets=[],[]
with torch.no_grad():
    for g in G_val:
        ei,ea,et,y = g.edge_index,g.edge_attr,g.edge_time,g.y
        reps,_=encode_edges(ea.to(device),et.to(device),encoder,t2v,ei.to(device),g.num_nodes)
        probs=F.softmax(binary_head(reps),dim=-1)[:,1];vprobs.extend(probs.cpu().tolist());vtargets.extend((y!=0).long().cpu().tolist())

vprobs=np.array(vprobs);vtargets=np.array(vtargets)
best_thr,best_f1_thr=0.5,0.0
for thr in np.arange(0.01,0.99,0.01):
    preds=(vprobs>=thr).astype(int)
    f1v=f1_score(vtargets,preds,average='macro')
    if f1v>best_f1_thr:best_f1_thr=f1v;best_thr=thr
final_recall=recall_score(vtargets,(vprobs>=best_thr).astype(int))
print(f"Threshold: {best_thr:.3f} (F1={best_f1_thr:.4f}, recall={final_recall:.4f})")
with open(CKPT_DIR/'threshold.json','w') as f:json.dump({'threshold':float(best_thr),'val_f1':float(best_f1)},f)

log={'notebook':'K03','stage':'F','best_val_f1':float(best_f1),'threshold':float(best_thr),
     'phaseA_epochs':len(phaseA_losses),'phaseB_epochs':len(phaseB_losses)}
with open(LOGS_DIR/'k03_log.json','w') as f:json.dump(log,f,indent=2)
print(f"K03 DONE. Best F1: {best_f1:.4f}, Threshold: {best_thr:.3f}")
print(f"Next: K04 (Multiclass)")