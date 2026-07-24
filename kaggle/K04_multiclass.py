"""
K04 — Multiclass Classification Stage-2 Head (Stage G)
=========================================================
KAGGLE T4x2 GPU. 11-class multiclass with focal loss, per-class thresholds,
PGD adversarial training, and synthetic minority samples.

Prerequisite: K03 checkpoint + K02 synthetic embeddings + preprocessed graphs
Edge input: 58-dim
"""

# %% [cell 1]
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np, pandas as pd, yaml, json, random, gc; from pathlib import Path
from datetime import datetime, timezone; from collections import Counter
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import torch_geometric; from torch_geometric.nn import GATv2Conv; from torch_geometric.data import Data
from torch_geometric.utils import degree
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
CKPT_DIR=WORKING/'checkpoints'/'G_multiclass'
LOGS_DIR=WORKING/'logs';FIGS_DIR=WORKING/'outputs'/'figures';TABS_DIR=WORKING/'outputs'/'tables'
for d in [CKPT_DIR,LOGS_DIR,FIGS_DIR,TABS_DIR]:d.mkdir(parents=True,exist_ok=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [cell 3] Load manifests & model defs (same as K01-K03)
with open(INPUT/'feature_manifest.yaml') as f:fm=yaml.safe_load(f)
with open(INPUT/'label_map.yaml') as f:lm=yaml.safe_load(f)
UNIFIED=lm['unified_classes'];N_CLASSES=len(UNIFIED);EDGE_DIM=fm['final_edge_input_dim']





# %% [cell 4] Load from K03 checkpoint (Uploaded Dataset)
K03_CKPT = Path('/kaggle/input/datasets/harshitpachahara/k03-output/checkpoints/F_binary/best.pt')
ckpt_f=torch.load(K03_CKPT,map_location=device,weights_only=False)
if 't2v' not in globals(): t2v = Time2Vec(k=16).to(device)
if 'encoder' not in globals(): encoder = EGATv2Encoder(edge_dim=EDGE_DIM).to(device)
t2v.load_state_dict(ckpt_f['t2v']);encoder.load_state_dict(ckpt_f['encoder'],strict=False)
TIME_MIN=ckpt_f['time_min'];TIME_MAX=ckpt_f['time_max']
def norm_time(t):return(t-TIME_MIN)/(TIME_MAX-TIME_MIN)

# Load synthetic embeddings from K02 (Uploaded Dataset)
K02_DIR = Path('/kaggle/input/datasets/harshitpachahara/k02-output/checkpoints/E_cvae')
synth_data=torch.load(K02_DIR/'synthetic_embeddings.pt',weights_only=False)
# Override original absolute path saved in dict to point to current Kaggle input directory
memmap_path = K02_DIR/'synthetic_embeddings.dat'
synth_emb = np.memmap(memmap_path, dtype=synth_data['embeddings_dtype'], mode='r', shape=synth_data['embeddings_shape'])
synth_emb = torch.from_numpy(synth_emb);synth_lbl=synth_data['labels']
print(f"Synthetic: {synth_emb.shape[0]:,} embeddings")

# Load graphs
def load_graphs(n,s):
    p=INPUT/f'{n}_{s}_list.pt';return torch.load(p,weights_only=False) if p.exists() else []
G_train=load_graphs('NF-CICIDS2018','train')+load_graphs('NF-UNSW-NB15','train')
G_val=load_graphs('NF-CICIDS2018','val')+load_graphs('NF-UNSW-NB15','val')

# Effective-number weights + feature bounds — compute iteratively (no giant cat tensors)
class_counts=Counter()
fmins_41=torch.full((41,), float('inf'))
fmaxs_41=torch.full((41,),-float('inf'))
for g in G_train:
    class_counts.update(g.y.tolist())
    fmins_41=torch.min(fmins_41,g.edge_attr.min(dim=0).values)
    fmaxs_41=torch.max(fmaxs_41,g.edge_attr.max(dim=0).values)
N=sum(class_counts.values());beta=(N-1)/N
eff_num={c:(1-beta**cnt)/(1-beta) for c,cnt in class_counts.items()}
eff_weights=torch.tensor([1.0/max(eff_num.get(i,1),1) for i in range(N_CLASSES)],device=device)
eff_weights=eff_weights/eff_weights.sum()*N_CLASSES
print(f"Effective-number weights range: [{eff_weights.min():.3f}, {eff_weights.max():.3f}]")
gc.collect(); torch.cuda.empty_cache()  # clean up after graph loading

# %% [cell 5] Training
if 'multiclass_head' not in globals(): multiclass_head = ClassifierHead(out_dim=N_CLASSES).to(device);focal=FocalLoss(gamma=2.0,alpha=eff_weights)
HP={'lr':1e-5,'epochs':20,'batch':2048,'pgd_eps':0.03,'pgd_alpha':0.01,'pgd_steps':3,'pgd_frac':0.10}

opt=optim.AdamW([{'params':t2v.parameters(),'lr':HP['lr']},
                  {'params':encoder.parameters(),'lr':HP['lr']},
                  {'params':multiclass_head.parameters(),'lr':HP['lr']*10}])
sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=HP['epochs']);amp=GradScaler()

def encode(ea,et,ei,nn):
    tn=norm_time(et.float());te=t2v(tn)
    ea58 = torch.cat([ea.float(),te],dim=-1)
    return ea58,Data(edge_index=ei,edge_attr=ea58,num_nodes=nn)

# Feature bounds from iterative computation above (no giant cat)
fmins=torch.cat([fmins_41,torch.full((17,),-4.0)]).to(device)
fmaxs=torch.cat([fmaxs_41,torch.full((17,),4.0)]).to(device)

train_losses,val_f1s=[],[];best_f1=0.0
synth_emb_dev=synth_emb.to(device);synth_lbl_dev=synth_lbl.to(device)

print(f"\nMulticlass training: {HP['epochs']} epochs")

# Resume from checkpoint if exists
if (CKPT_DIR / 'best.pt').exists():
    print(f'Resuming training from {CKPT_DIR / "best.pt"}')
    ckpt = torch.load(CKPT_DIR / 'best.pt', map_location=device, weights_only=False)
    if 't2v' in ckpt: t2v.load_state_dict(ckpt['t2v'])
    if 'encoder' in ckpt: encoder.load_state_dict(ckpt['encoder'])
    if 'head' in ckpt: multiclass_head.load_state_dict(ckpt['head'])
    if 'opt' in ckpt: opt.load_state_dict(ckpt['opt'])
    if 'sched' in ckpt: sched.load_state_dict(ckpt['sched'])
    if 'epoch' in ckpt: start_epoch = ckpt['epoch'] + 1
    else: start_epoch = 0
    if 'val_f1' in ckpt: best_f1 = ckpt['val_f1']
else:
    start_epoch = 0

for epoch in range(start_epoch, HP['epochs']):
    t2v.train();encoder.train();multiclass_head.train();el,nb=0.0,0
    random.shuffle(G_train)  # prevent dataset-order bias (CICIDS→UNSW fixed order)
    for g in G_train:
        n_edges=g.edge_index.shape[1]
        if n_edges<4:continue
        # Undersample to ~200 per class
        keep_idx=[]
        for cls in range(1, N_CLASSES):
            cm=(g.y==cls).nonzero(as_tuple=True)[0]
            if len(cm)>0:keep_idx.append(cm)
        if not keep_idx:continue
        keep=torch.cat(keep_idx)
        for i in range(0,len(keep),HP['batch']):
            bi=keep[i:i+HP['batch']];n_bi=len(bi)
            if n_bi<4:continue
            # Build features and run PGD OUTSIDE autocast (fp32 for accurate attack gradients)
            tn=norm_time(g.edge_time[bi].float().to(device));te=t2v(tn)
            ea58=torch.cat([g.edge_attr[bi].float().to(device),te],dim=-1)
            # PGD adversarial attack on 30% of batch (fp32 precision)
            n_pgd=int(n_bi*HP['pgd_frac'])
            if n_pgd>0:
                pgd_mask=torch.zeros(n_bi,dtype=torch.bool,device=device)
                pgd_mask[:n_pgd]=True;pgd_mask=pgd_mask[torch.randperm(n_bi)]
                ea58_orig=ea58.clone()
                for _ in range(HP['pgd_steps']):
                    ea58_pert=ea58.clone().detach().requires_grad_(True)
                    d_pert=Data(edge_index=g.edge_index[:,bi].to(device),edge_attr=ea58_pert,num_nodes=g.num_nodes)
                    logits_pert=multiclass_head(encoder(d_pert))
                    loss_adv=focal(logits_pert[pgd_mask],g.y[bi].to(device)[pgd_mask])
                    grad=torch.autograd.grad(loss_adv,ea58_pert)[0]
                    with torch.no_grad():
                        ea58[pgd_mask]+=HP['pgd_alpha']*grad[pgd_mask].sign()
                        delta=torch.clamp(ea58[pgd_mask]-ea58_orig[pgd_mask],-HP['pgd_eps'],HP['pgd_eps'])
                        ea58[pgd_mask]=ea58_orig[pgd_mask]+delta
                        ea58=torch.clamp(ea58,fmins,fmaxs)
            # Main forward pass in autocast (fp16 for throughput)
            with autocast():
                d_final=Data(edge_index=g.edge_index[:,bi].to(device),edge_attr=ea58,num_nodes=g.num_nodes)
                reps=encoder(d_final)
                # Mix synthetic embeddings for minority classes (1:1 ratio as per architecture)
                yb=g.y[bi].to(device)
                real_reps_list=[reps];real_labels_list=[yb]
                for cls in yb.unique():
                    cls_mask=(synth_lbl_dev==cls)
                    if cls_mask.sum()>0:
                        n_synth_needed=(yb==cls).sum().item()
                        synth_pool=synth_emb_dev[cls_mask]
                        if synth_pool.shape[0] > 0:
                            idx_s=torch.randint(0, synth_pool.shape[0], (n_synth_needed,))
                            real_reps_list.append(synth_pool[idx_s])
                            real_labels_list.append(torch.full((n_synth_needed,),cls,dtype=torch.long,device=device))
                all_reps=torch.cat(real_reps_list,dim=0)
                all_labels=torch.cat(real_labels_list,dim=0)
                logits=multiclass_head(all_reps);loss=focal(logits,all_labels)
            opt.zero_grad();amp.scale(loss).backward()
            amp.unscale_(opt)  # unscale BEFORE clipping
            torch.nn.utils.clip_grad_norm_(list(t2v.parameters())+list(encoder.parameters())+list(multiclass_head.parameters()),1.0)
            amp.step(opt)
            amp.update();el+=loss.item();nb+=1
            if nb % 500 == 0: print(f"    [Phase A] Epoch {epoch+1} - Batch {nb} - Loss: {el/nb:.6f}")
    sched.step();avg=el/max(nb,1);train_losses.append(avg)
    gc.collect(); torch.cuda.empty_cache()

    # Validation
    t2v.eval();encoder.eval();multiclass_head.eval();vp,vt=[],[]
    with torch.no_grad():
        val_sample=G_val
        for g in val_sample:
            ei,ea,et,y = g.edge_index,g.edge_attr,g.edge_time,g.y
            ea58,data_b=encode(ea.to(device),et.to(device),ei.to(device),g.num_nodes)
            preds=multiclass_head(encoder(data_b)).argmax(dim=-1)
            vp.extend(preds.cpu().tolist());vt.extend(y.cpu().tolist())
    vf1=f1_score(vt,vp,average='macro',labels=list(range(1,N_CLASSES)));val_f1s.append(vf1)
    per_class_f1 = f1_score(vt, vp, average=None, labels=list(range(1, N_CLASSES)))
    f1_str = " ".join([f"C{i+1}:{f:.4f}" for i, f in enumerate(per_class_f1)])
    print(f"Epoch {epoch+1:2d}: Loss={avg:.6f} Val-F1={vf1:.4f} | {f1_str}")
    if vf1>best_f1:
        best_f1=vf1
        torch.save({'t2v':t2v.state_dict(),'encoder':encoder.state_dict(),'head':multiclass_head.state_dict(),
                    'opt':opt.state_dict(),'sched':sched.state_dict(),'epoch':epoch,
                    'val_f1':vf1,'config':HP,'time_min':TIME_MIN,'time_max':TIME_MAX},CKPT_DIR/'best.pt')

# %% [cell 6] Per-Class Threshold Calibration
# CRITICAL: Reload best checkpoint so thresholds are calibrated on the best model, not the last epoch
print("\nReloading best checkpoint for threshold calibration...")
best_ckpt=torch.load(CKPT_DIR/'best.pt',map_location=device,weights_only=False)
t2v.load_state_dict(best_ckpt['t2v']);encoder.load_state_dict(best_ckpt['encoder'],strict=False)
multiclass_head.load_state_dict(best_ckpt['head'])

t2v.eval();encoder.eval();multiclass_head.eval()
all_probs,all_targets=[],[]
with torch.no_grad():
    for g in G_val:
        ei,ea,et,y = g.edge_index,g.edge_attr,g.edge_time,g.y
        ea58,data_b=encode(ea.to(device),et.to(device),ei.to(device),g.num_nodes)
        probs=F.softmax(multiclass_head(encoder(data_b)),dim=-1)
        all_probs.append(probs.cpu());all_targets.append(y.cpu())
all_probs=torch.cat(all_probs).numpy();all_targets=torch.cat(all_targets).numpy()

per_class_thr={}
for cls_i,cls_n in enumerate(UNIFIED):
    cm=all_targets==cls_i
    if cm.sum()<10:per_class_thr[cls_n]=0.5;continue
    bt,bf=0.5,0.0
    for t in np.arange(0.05,0.95,0.05):
        preds=(all_probs[:,cls_i]>=t).astype(int);f1v=f1_score(cm.astype(int),preds)
        if f1v>bf:bf=f1v;bt=t
    per_class_thr[cls_n]=float(bt);print(f"  {cls_n:25s}: thr={bt:.2f} F1={bf:.4f}")

with open(CKPT_DIR/'thresholds.json','w') as f:json.dump({'per_class_thresholds':per_class_thr},f,indent=2)

# %% [cell 7] Confusion Matrix (Fig 15)
final_preds=[]
for i in range(len(all_probs)):
    # Per-class threshold gating: only trust prediction if confidence >= class threshold
    sorted_cls=np.argsort(all_probs[i])[::-1]  # descending confidence
    pred=sorted_cls[0]  # default: argmax
    for cls in sorted_cls:
        thr=per_class_thr.get(UNIFIED[cls],0.5)
        if all_probs[i][cls]>=thr:
            pred=cls; break  # first class (highest prob) that exceeds its threshold
    final_preds.append(pred)

cm=confusion_matrix(all_targets,final_preds);cm_norm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
fig,ax=plt.subplots(figsize=(10,8))
sns.heatmap(cm_norm,annot=True,fmt='.2f',cmap='Blues',
            xticklabels=[c[:12] for c in UNIFIED],yticklabels=[c[:12] for c in UNIFIED],ax=ax,vmin=0,vmax=1)
ax.set_xlabel('Predicted');ax.set_ylabel('True')
ax.set_title('Confusion Matrix — In-Domain (Validation)')
plt.tight_layout();plt.savefig(FIGS_DIR/'fig15_confusion_matrix.png',dpi=300);plt.show()

# Tab05
report=classification_report(all_targets,final_preds,target_names=UNIFIED,output_dict=True,zero_division=0)
tab05=[{'class':c,'f1_score':round(report[c]['f1-score'],4),'precision':round(report[c]['precision'],4),
        'recall':round(report[c]['recall'],4),'support':int(report[c]['support'])}
       for c in UNIFIED if c in report and isinstance(report[c],dict)]
tab05.append({'class':'MACRO AVG','f1_score':round(report['macro avg']['f1-score'],4)})
pd.DataFrame(tab05).to_csv(TABS_DIR/'tab05_main_results.csv',index=False)
pd.DataFrame(tab05).to_markdown(TABS_DIR/'tab05_main_results.md',index=False)

log={'notebook':'K04','stage':'G','best_val_f1':float(best_f1),'per_class_thresholds':per_class_thr}
with open(LOGS_DIR/'k04_log.json','w') as f:json.dump(log,f,indent=2)
print(f"\nK04 DONE. Best F1: {best_f1:.4f}. Next: K05 (Prototypical)")