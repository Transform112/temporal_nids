"""
K06 — Evaluation, XAI & Consolidation (Stages I, J)
======================================================
KAGGLE T4x2 GPU. Runs cross-dataset eval, adversarial robustness,
t-SNE, inference latency, SHAP, attention viz. Generates RESULTS_SUMMARY.md.

Prerequisite: All K01-K05 checkpoints + preprocessed blind-test graphs
"""

# %% [cell 1] Imports
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, yaml, json, random, time, gc
from datetime import datetime, timezone; from pathlib import Path
from collections import defaultdict, Counter
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.manifold import TSNE

import torch_geometric; from torch_geometric.nn import GATv2Conv; from torch_geometric.data import Data

# %% [cell 2] Seed & Paths
SEED=42;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)

WORKING=Path('../working');INPUT = Path('/kaggle/input/datasets/mysteriousavailable/ids-nf3-processed/ids-nf3-processed')
FIGS_DIR=WORKING/'outputs'/'figures';TABS_DIR=WORKING/'outputs'/'tables';LOGS_DIR=WORKING/'logs'
for d in [FIGS_DIR,TABS_DIR,LOGS_DIR]:d.mkdir(parents=True,exist_ok=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NB_START=datetime.now(timezone.utc).isoformat()

# %% [cell 3] Model defs + load checkpoints
with open(INPUT/'feature_manifest.yaml') as f:fm=yaml.safe_load(f)
with open(INPUT/'label_map.yaml') as f:lm=yaml.safe_load(f)
UNIFIED=lm['unified_classes'];N_CLASSES=len(UNIFIED);EDGE_DIM=fm['final_edge_input_dim']

class Time2Vec(nn.Module):
    def __init__(self,k=16):
        super().__init__();self.k=k;self.w0=nn.Parameter(torch.randn(1)*0.1);self.b0=nn.Parameter(torch.zeros(1))
        self.omega=nn.Parameter(10.0**(torch.rand(k)*6-3));self.bias=nn.Parameter(torch.zeros(k));self.output_dim=k+1
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

    def forward(self, data, return_attn=False):
        n = data.num_nodes
        table_size = self.node_embed.shape[0]
        if n <= table_size:
            x = self.node_embed[:n]
        else:
            idx = torch.arange(n, device=self.node_embed.device) % table_size
            x = self.node_embed[idx]
        ea = self.edge_proj(data.edge_attr)
        all_attn = []
        for conv, norm in zip(self.convs, self.norms):
            if return_attn or self.return_attention:
                x_new, attn = conv(x, data.edge_index, edge_attr=ea, return_attention_weights=True)
                all_attn.append(attn)
            else:
                x_new = conv(x, data.edge_index, edge_attr=ea)
            x_new = self.activation(x_new); x_new = self.dropout(x_new)
            x = norm(x + x_new) if x.shape == x_new.shape else norm(x_new)
        out = torch.cat([x[data.edge_index[0]], x[data.edge_index[1]], ea], dim=-1)
        return (out, all_attn) if (return_attn or self.return_attention) else out

class BinaryHead(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(768,256),nn.ELU(),nn.Dropout(0.3),nn.Linear(256,64),nn.ELU(),nn.Dropout(0.2),nn.Linear(64,2))
    def forward(self,x):return self.net(x)

class MulticlassHead(nn.Module):
    def __init__(self,nc=11):super().__init__();self.net=nn.Sequential(nn.Linear(768,256),nn.ELU(),nn.Dropout(0.3),nn.Linear(256,nc))
    def forward(self,x):return self.net(x)

# Load checkpoints
ckpt_f=torch.load(WORKING/'checkpoints'/'F_binary'/'best.pt',map_location=device,weights_only=False)
ckpt_g=torch.load(WORKING/'checkpoints'/'G_multiclass'/'best.pt',map_location=device,weights_only=False)

t2v=Time2Vec(k=16).to(device)
enc_max_nodes = ckpt_g['encoder']['node_embed'].shape[0]
encoder=EGATv2Encoder(max_nodes=enc_max_nodes, edge_dim=EDGE_DIM).to(device)
bin_head=BinaryHead().to(device);multi_head=MulticlassHead(nc=N_CLASSES).to(device)
t2v.load_state_dict(ckpt_g['t2v']);encoder.load_state_dict(ckpt_g['encoder'],strict=False)
bin_head.load_state_dict(ckpt_f['head']);multi_head.load_state_dict(ckpt_g['head'])
for m in [t2v,encoder,bin_head,multi_head]:
    for p in m.parameters():p.requires_grad=False;m.eval()

with open(WORKING/'checkpoints'/'F_binary'/'threshold.json') as f:bin_thr=json.load(f)['threshold']
with open(WORKING/'checkpoints'/'G_multiclass'/'thresholds.json') as f:per_cls_thr=json.load(f)['per_class_thresholds']

# Time normalizer — load saved values from training checkpoint
TMIN=ckpt_g['time_min'];TMAX=ckpt_g['time_max']
def nt(t):return(t-TMIN)/(TMAX-TMIN)

def encode(ea,et,ei,nn):
    tn=nt(et);te=t2v(tn);ea58=torch.cat([ea,te],dim=-1)
    return Data(edge_index=ei,edge_attr=ea58,num_nodes=nn)

def load_graphs(n,s):
    p=INPUT/f'{n}_{s}_list.pt';return torch.load(p,weights_only=False) if p.exists() else []

# %% [cell 4] In-Domain Eval
print("="*50+"\nIN-DOMAIN EVALUATION\n"+"="*50)
G_test=load_graphs('NF-CICIDS2018','test')+load_graphs('NF-UNSW-NB15','test')
gc.collect(); torch.cuda.empty_cache()  # clean up after torch.load
test_preds,test_targets=[],[]
with torch.no_grad():
    for g in G_test:
        g=g.to(device);reps=encoder(encode(g.edge_attr,g.edge_time,g.edge_index,g.num_nodes))
        bin_probs=F.softmax(bin_head(reps),-1);attack_mask=bin_probs[:,1]>=bin_thr
        preds=torch.zeros(reps.shape[0],dtype=torch.long,device=device)
        if attack_mask.any():
            multi_probs=F.softmax(multi_head(reps[attack_mask]),-1)
            # Apply per-class thresholds — vectorized (same logic as K04 cell 7)
            thresh_t=torch.tensor([per_cls_thr.get(UNIFIED[i],0.5) for i in range(N_CLASSES)],device=device)
            max_probs,max_cls=multi_probs.max(dim=-1)
            pred_thresh=thresh_t[max_cls]
            below=(max_probs<pred_thresh)
            if below.any():
                # For below-threshold samples: find highest-prob class that meets its own threshold
                qualified=(multi_probs>=thresh_t.unsqueeze(0))  # (N,11) bool
                masked_p=multi_probs.clone();masked_p[~qualified]=-1.0
                fallback=masked_p.argmax(dim=-1)  # if none qualify, argmax returns 0 (Benign)
                max_cls[below]=fallback[below]
            preds[attack_mask]=max_cls
        test_preds.extend(preds.cpu().tolist());test_targets.extend(g.y.cpu().tolist())
in_domain_f1=f1_score(test_targets,test_preds,average='macro')
print(f"In-domain Macro-F1: {in_domain_f1:.4f}")
for i,cn in enumerate(UNIFIED):
    cm=test_targets==i
    if cm.sum()>0:print(f"  {cn:25s}: F1={f1_score(cm,np.array(test_preds)==i):.4f}")

# Tab05
report=classification_report(test_targets,test_preds,target_names=UNIFIED,output_dict=True,zero_division=0)
tab05=[{'class':c,'f1_score':round(report[c]['f1-score'],4),'recall':round(report[c]['recall'],4)} for c in UNIFIED if c in report and isinstance(report[c],dict)]
tab05.append({'class':'MACRO AVG','f1_score':round(in_domain_f1,4)})
pd.DataFrame(tab05).to_csv(TABS_DIR/'tab05_main_results.csv',index=False)

# %% [cell 5] Cross-Dataset Blind Test
gc.collect(); torch.cuda.empty_cache()
print("\n"+"="*50+"\nCROSS-DATASET BLIND TEST\n"+"="*50)
cross_results=[]
for ds in ['NF-ToN-IoT','NF-BoT-IoT']:
    graphs=load_graphs(ds,'test')
    if not graphs:print(f"  {ds}: SKIPPED (no preprocessed graphs)");cross_results.append({'dataset':ds,'macro_f1':'N/A'});continue
    p,t=[],[]
    with torch.no_grad():
        for g in graphs:
            g=g.to(device);reps=encoder(encode(g.edge_attr,g.edge_time,g.edge_index,g.num_nodes))
            preds=multi_head(reps).argmax(-1);p.extend(preds.cpu().tolist());t.extend(g.y.cpu().tolist())
    mf1=f1_score(t,p,average='macro')
    print(f"  {ds}: Macro-F1={mf1:.4f}");cross_results.append({'dataset':ds,'schema':'in-schema','macro_f1':round(float(mf1),4)})
cross_results.append({'dataset':'CIC-DDoS2019','schema':'out-of-schema','macro_f1':'N/A'})
cross_results.append({'dataset':'CIC-Darknet2020','schema':'out-of-schema','macro_f1':'N/A'})
pd.DataFrame(cross_results).to_csv(TABS_DIR/'tab06_cross_dataset_results.csv',index=False)

# Fig 11: Cross-dataset bar chart
plot_data=[r for r in cross_results if r['macro_f1']!='N/A']
if plot_data:
    fig,ax=plt.subplots(figsize=(10,5))
    names=['In-Domain']+[r['dataset'] for r in plot_data]
    vals=[in_domain_f1]+[r['macro_f1'] for r in plot_data]
    ax.bar(range(len(names)),vals,color=['#4CAF50']+['#2196F3']*len(plot_data),edgecolor='black')
    ax.set_xticks(range(len(names)));ax.set_xticklabels(names,fontsize=8);ax.set_ylabel('Macro-F1');ax.set_ylim(0,1.05)
    for i,v in enumerate(vals):ax.text(i,v+0.01,f'{v:.3f}',ha='center',fontweight='bold')
    ax.set_title('Cross-Dataset Generalization');ax.grid(axis='y',alpha=0.3)
    plt.tight_layout();plt.savefig(FIGS_DIR/'fig11_cross_dataset_bar_chart.png',dpi=300);plt.show()

# %% [cell 6] Adversarial Robustness
gc.collect(); torch.cuda.empty_cache()
print("\n"+"="*50+"\nADVERSARIAL ROBUSTNESS\n"+"="*50)
G_test_sub=[];n_per=2000
for g in G_test[:5]:
    g = g.clone()
    if g.edge_index.shape[1]>n_per:
        idx=torch.randperm(g.edge_index.shape[1])[:n_per];g.edge_index=g.edge_index[:,idx];g.edge_attr=g.edge_attr[idx];g.edge_time=g.edge_time[idx];g.y=g.y[idx]
    G_test_sub.append(g)

fmins=torch.cat([torch.full((41,),-4.0),torch.full((17,),-4.0)]).to(device)
fmaxs=torch.cat([torch.full((41,),4.0),torch.full((17,),4.0)]).to(device)
rob_results=[]

for eps in [0.0,0.01,0.03,0.05]:
    ap,at=[],[]
    for g in G_test_sub:
        g=g.to(device)
        ea58=encode(g.edge_attr,g.edge_time,g.edge_index,g.num_nodes).edge_attr
        if eps>0:
            ea58_orig=ea58.clone()
            for _ in range(7):
                ea58=ea58.clone().detach().requires_grad_(True)
                d=Data(edge_index=g.edge_index,edge_attr=ea58,num_nodes=g.num_nodes)
                loss=-F.cross_entropy(multi_head(encoder(d)),g.y)
                grad=torch.autograd.grad(loss,ea58)[0]
                ea58=ea58.detach()+0.01*grad.sign()
                delta=torch.clamp(ea58-ea58_orig,-eps,eps);ea58=torch.clamp(ea58_orig+delta,fmins,fmaxs)
        d=Data(edge_index=g.edge_index,edge_attr=ea58,num_nodes=g.num_nodes)
        preds=multi_head(encoder(d)).argmax(-1);ap.extend(preds.cpu().tolist());at.extend(g.y.cpu().tolist())
    mf1=f1_score(at,ap,average='macro');rob_results.append({'epsilon':eps,'macro_f1':round(float(mf1),4)})
    print(f"  eps={eps:.2f}: Macro-F1={mf1:.4f}")

pd.DataFrame(rob_results).to_csv(TABS_DIR/'tab08_adversarial_robustness.csv',index=False)
fig,ax=plt.subplots(figsize=(8,5))
ax.plot([r['epsilon'] for r in rob_results],[r['macro_f1'] for r in rob_results],'o-',color='#E53935',lw=2,ms=10)
ax.set_xlabel('PGD epsilon');ax.set_ylabel('Macro-F1');ax.set_ylim(0,1.05);ax.grid(alpha=0.3)
for r in rob_results:ax.annotate(f"{r['macro_f1']:.3f}",(r['epsilon'],r['macro_f1']),textcoords="offset points",xytext=(0,10),ha='center')
ax.set_title('Adversarial Robustness Curve')
plt.tight_layout();plt.savefig(FIGS_DIR/'fig10_adversarial_robustness_curve.png',dpi=300);plt.show()

# %% [cell 7] t-SNE Embeddings (Fig 09)
gc.collect(); torch.cuda.empty_cache()
print("\n"+"="*50+"\nt-SNE VISUALIZATION\n"+"="*50)
# Sample embeddings
sample_embs,sample_lbls=[],[]
with torch.no_grad():
    for g in G_test[:3]:
        g=g.clone().to(device)
        if g.edge_index.shape[1]>1000:
            idx=torch.randperm(g.edge_index.shape[1])[:1000];g.edge_index=g.edge_index[:,idx];g.edge_attr=g.edge_attr[idx];g.edge_time=g.edge_time[idx];g.y=g.y[idx]
        reps=encoder(encode(g.edge_attr,g.edge_time,g.edge_index,g.num_nodes))
        sample_embs.append(reps.cpu());sample_lbls.append(g.y.cpu())
sample_embs=torch.cat(sample_embs);sample_lbls=torch.cat(sample_lbls)
if sample_embs.shape[0]>2000:
    idx=torch.randperm(sample_embs.shape[0])[:2000];sample_embs=sample_embs[idx];sample_lbls=sample_lbls[idx]

tsne=TSNE(n_components=2,random_state=SEED,perplexity=30,n_iter=1000)
emb_2d=tsne.fit_transform(sample_embs.numpy())
fig,ax=plt.subplots(figsize=(10,8))
colors=plt.cm.tab10(np.linspace(0,1,N_CLASSES))
for i,cn in enumerate(UNIFIED):
    mask=sample_lbls.numpy()==i
    if mask.sum()>0:ax.scatter(emb_2d[mask,0],emb_2d[mask,1],c=[colors[i]],label=cn,alpha=0.5,s=5)
ax.legend(fontsize=7,markerscale=3);ax.set_xlabel('t-SNE 1');ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE of Flow Embeddings (Post-Multiclass)')
plt.tight_layout();plt.savefig(FIGS_DIR/'fig09_tsne_embeddings.png',dpi=300);plt.show()

# %% [cell 8] Inference Latency (Tab 10)
print("\n"+"="*50+"\nLATENCY BENCHMARK\n"+"="*50)
test_g=G_test[0].clone().to(device)
if test_g.edge_index.shape[1]>1024:
    idx=torch.randperm(test_g.edge_index.shape[1])[:1024];test_g.edge_index=test_g.edge_index[:,idx];test_g.edge_attr=test_g.edge_attr[idx];test_g.edge_time=test_g.edge_time[idx]

# Warmup
for _ in range(50):
    _=multi_head(encoder(encode(test_g.edge_attr,test_g.edge_time,test_g.edge_index,test_g.num_nodes)))

# Benchmark
n_runs=200;torch.cuda.synchronize();t0=time.time()
for _ in range(n_runs):
    reps=encoder(encode(test_g.edge_attr,test_g.edge_time,test_g.edge_index,test_g.num_nodes))
    _=bin_head(reps);_=multi_head(reps)
torch.cuda.synchronize()
batch_lat=(time.time()-t0)/n_runs*1000;per_flow=batch_lat/test_g.edge_index.shape[1]
n_params=sum(p.numel() for p in list(t2v.parameters())+list(encoder.parameters())+list(bin_head.parameters())+list(multi_head.parameters()))
print(f"Batch ({test_g.edge_index.shape[1]} flows): {batch_lat:.2f} ms ({per_flow:.4f} ms/flow)")
print(f"Params: {n_params:,}")

latency=[{'metric':'Batch latency (ms)','value':round(batch_lat,4)},
         {'metric':'Per-flow (ms)','value':round(per_flow,4)},
         {'metric':'Parameters','value':f'{n_params:,}'},
         {'metric':'Model size (MB,fp32)','value':round(n_params*4/1e6,2)}]
pd.DataFrame(latency).to_csv(TABS_DIR/'tab10_inference_latency.csv',index=False)

# %% [cell 9] XAI: Attention Visualization (Fig 14)
print("\n"+"="*50+"\nATTENTION VISUALIZATION\n"+"="*50)
g_attn=G_test[0].clone().to(device)
if g_attn.edge_index.shape[1]>100:
    idx=torch.randperm(g_attn.edge_index.shape[1])[:100];g_attn.edge_index=g_attn.edge_index[:,idx];g_attn.edge_attr=g_attn.edge_attr[idx];g_attn.edge_time=g_attn.edge_time[idx]
d_attn=encode(g_attn.edge_attr,g_attn.edge_time,g_attn.edge_index,g_attn.num_nodes)
with torch.no_grad():_,attn_weights=encoder(d_attn,return_attn=True)
final_attn=attn_weights[-1]
if isinstance(final_attn,tuple):
    attn_edges,attn_scores=final_attn;avg_attn=attn_scores.mean(dim=1).cpu().numpy()
    ei_np=attn_edges.cpu().numpy();nn=g_attn.num_nodes
    pos={i:(np.cos(2*np.pi*i/nn)+np.random.randn()*0.05,np.sin(2*np.pi*i/nn)+np.random.randn()*0.05) for i in range(nn)}
    fig,ax=plt.subplots(figsize=(10,8))
    for i in range(ei_np.shape[1]):
        u,v=ei_np[0,i],ei_np[1,i];w=avg_attn[i];alpha=min(w/avg_attn.max(),1.0)*0.8+0.2
        ax.plot([pos[u][0],pos[v][0]],[pos[u][1],pos[v][1]],'gray',alpha=alpha,lw=w*3)
    for node in range(nn):ax.scatter(pos[node][0],pos[node][1],s=50,c='#2196F3',edgecolors='black',lw=0.5,zorder=3)
    ax.set_title('Attention Visualization — Final E-GATv2 Layer');ax.axis('off')
    plt.tight_layout();plt.savefig(FIGS_DIR/'fig14_attention_visualization.png',dpi=300);plt.show()

# %% [cell 10] XAI: SHAP (Fig 13 + Tab 11)
print("\n"+"="*50+"\nSHAP FEATURE ATTRIBUTION\n"+"="*50)
try:
    import shap
    # SHAP operates on 768-dim encoder embeddings (not raw features) because the
    # full pipeline requires graph structure which GradientExplainer cannot capture.
    # This explains which embedding dimensions the classification head relies on.
    G_val=load_graphs('NF-CICIDS2018','val')[:3]+load_graphs('NF-UNSW-NB15','val')[:3]
    val_embs,val_lbls=[],[]
    with torch.no_grad():
        for g in G_val:
            g=g.clone().to(device)
            if g.edge_index.shape[1]>2000:
                idx=torch.randperm(g.edge_index.shape[1])[:2000];g.edge_index=g.edge_index[:,idx];g.edge_attr=g.edge_attr[idx];g.edge_time=g.edge_time[idx];g.y=g.y[idx]
            reps=encoder(encode(g.edge_attr,g.edge_time,g.edge_index,g.num_nodes))
            val_embs.append(reps.cpu());val_lbls.append(g.y.cpu())
    bg_emb=torch.cat(val_embs)[:100].numpy()   # background: 100 x 768
    X_emb=torch.cat(val_embs)[:500].numpy()     # test: 500 x 768
    print(f"SHAP (embedding-space): bg={bg_emb.shape}, samples={X_emb.shape}")

    # GradientExplainer on the multiclass head (768 -> 11)
    explainer=shap.GradientExplainer(
        lambda x:F.softmax(multi_head(torch.tensor(x,dtype=torch.float32,device=device)),dim=-1).cpu().numpy(),
        bg_emb[:50])
    shap_vals=explainer.shap_values(X_emb[:200])
    print(f"SHAP values shape: {[sv.shape for sv in shap_vals[:3]]}")

    # Tab 11: Top-5 SHAP embedding dimensions per class
    tab11=[]
    for cls_i in range(min(N_CLASSES,len(shap_vals))):
        mean_shap=np.abs(shap_vals[cls_i]).mean(axis=0)
        top5=np.argsort(mean_shap)[-5:][::-1]
        for rank,fi in enumerate(top5):
            tab11.append({'class':UNIFIED[cls_i],'rank':rank+1,
                          'feature':f'Emb_{fi}','mean_shap':round(float(mean_shap[fi]),6)})
    pd.DataFrame(tab11).to_csv(TABS_DIR/'tab11_shap_top_features.csv',index=False)
    pd.DataFrame(tab11).to_markdown(TABS_DIR/'tab11_shap_top_features.md',index=False)
    print(f"Saved: tab11 ({len(tab11)} rows)")

    # Fig 13: Top-8 SHAP embedding dims per class
    fig,axes=plt.subplots(4,3,figsize=(18,16));axes=axes.flatten()
    for cls_i in range(min(N_CLASSES,len(axes))):
        ax=axes[cls_i]
        if cls_i<len(shap_vals):
            ms=np.abs(shap_vals[cls_i]).mean(axis=0);t8=np.argsort(ms)[-8:]
            names=[f'Emb_{i}' for i in t8]
            ax.barh(range(8),ms[t8],color='#2196F3',edgecolor='black',lw=0.5)
            ax.set_yticks(range(8));ax.set_yticklabels(names,fontsize=7);ax.set_title(UNIFIED[cls_i][:15],fontsize=9)
            ax.grid(axis='x',alpha=0.3)
        else:ax.axis('off')
    fig.suptitle('Top-8 SHAP Embedding Dimensions per Class (Head-Level)',fontweight='bold',fontsize=13)
    plt.tight_layout();plt.savefig(FIGS_DIR/'fig13_shap_summary.png',dpi=300);plt.show()
except Exception as e:
    print(f"SHAP failed: {e}")

# %% [cell 11] RESULTS_SUMMARY.md + Verification
summary=f"""# RESULTS SUMMARY — Graph-NIDS
## In-Domain
- Macro-F1: {in_domain_f1:.4f}
- Dataset: CICIDS2018 + UNSW-NB15 (chronological test split)

## Cross-Dataset
{chr(10).join(f'- {r["dataset"]}: {r["macro_f1"]}' for r in cross_results)}

## Adversarial Robustness
{chr(10).join(f'- epsilon={r["epsilon"]}: F1={r["macro_f1"]}' for r in rob_results)}

## Inference
- Per-flow: {per_flow:.4f} ms
- Params: {n_params:,}
"""
with open(WORKING/'RESULTS_SUMMARY.md','w') as f:f.write(summary)

# Output verification (figs/tabs produced across K01-K06 notebooks)
missing=[]
for d,stem,exts in [(FIGS_DIR,f'fig{i:02d}',['.png','_architecture_diagram.png']) for i in range(1,17)] + \
                     [(TABS_DIR,f'tab{i:02d}',['.csv','.md']) for i in range(1,13)]:
    if not any((d/(stem+ext)).exists() for ext in exts):
        missing.append(f'{stem} ({", ".join(exts)})')
print("\n"+"="*50+"\nOUTPUT VERIFICATION\n"+"="*50)
if missing:
    print(f"WARNING: {len(missing)} expected outputs NOT FOUND (may be in earlier notebooks):")
    for m in missing: print(f"  - {m}")
else:
    print("All 16 figures + 12 tables verified present.")
print(f"RESULTS_SUMMARY.md saved to {WORKING/'RESULTS_SUMMARY.md'}")

log={'notebook':'K06','stages':['I','J'],'in_domain_f1':float(in_domain_f1),
     'cross_dataset':cross_results,'adversarial':rob_results,'per_flow_ms':float(per_flow)}
with open(LOGS_DIR/'k06_log.json','w') as f:json.dump(log,f,indent=2)
print(f"\nK06 COMPLETE. Pipeline end-to-end finished.")
