"""
K06 — Evaluation, XAI & Consolidation (Stages I, J)
======================================================
KAGGLE T4x2. Final script. Cross-dataset eval, adversarial robustness,
t-SNE, inference latency, SHAP, attention visualization, RESULTS_SUMMARY.md.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, yaml, json, pickle, random, time, gc
from datetime import datetime, timezone; from pathlib import Path; from collections import defaultdict
from torch_geometric.nn import GATv2Conv; from torch_geometric.data import Data
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.manifold import TSNE
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import seaborn as sns

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.deterministic=True

INPUT_DIR=Path('/kaggle/input/ids-processed'); WORKING=Path('/kaggle/working')
FIGS_DIR=WORKING/'outputs'/'figures'; TABS_DIR=WORKING/'outputs'/'tables'; LOGS_DIR=WORKING/'logs'
for d in [FIGS_DIR,TABS_DIR,LOGS_DIR]: d.mkdir(parents=True,exist_ok=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(INPUT_DIR/'feature_manifest.yaml') as f: fm=yaml.safe_load(f)
with open(INPUT_DIR/'label_map.yaml') as f: lm=yaml.safe_load(f)
EDGE_DIM=fm['final_edge_input_dim']; UNIFIED=lm['unified_classes']; NC=len(UNIFIED); KEPT=fm['kept_features']

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
    def forward(self,data,return_attn=False):
        x=self._get_node_embed(data.num_nodes,data.edge_index.device); e=self.edge_proj(data.edge_attr)
        all_a=[]
        for conv,norm in zip(self.convs,self.norms):
            xn,a=conv(x,data.edge_index,edge_attr=e,return_attention_weights=True)
            if return_attn: all_a.append(a)
            xn=self.activation(xn); xn=self.dropout(xn); xn=norm(xn); x=x+xn if x.shape==xn.shape else xn
        out=torch.cat([x[data.edge_index[0]],x[data.edge_index[1]],e],dim=-1)
        return (out,all_a) if return_attn else out

class BinaryHead(nn.Module):
    def __init__(self,i=768,h=256,b=64,n=2):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(i,h),nn.ELU(),nn.Dropout(0.3),nn.Linear(h,b),nn.ELU(),nn.Dropout(0.2),nn.Linear(b,n))
    def forward(self,x): return self.net(x)

class MulticlassHead(nn.Module):
    def __init__(self,i=768,h=256,n=11):
        super().__init__(); self.net=nn.Sequential(nn.Linear(i,h),nn.ELU(),nn.Dropout(0.3),nn.Linear(h,n))
    def forward(self,x): return self.net(x)

# ---- Load all checkpoints ----
ckpt_g=torch.load(WORKING/'checkpoints'/'G_multiclass'/'best.pt',map_location=device,weights_only=False)
ckpt_f=torch.load(WORKING/'checkpoints'/'F_binary'/'best.pt',map_location=device,weights_only=False)
t2v=Time2Vec(k=16).to(device); enc=EGATv2Encoder(edge_dim=EDGE_DIM).to(device)
bhead=BinaryHead().to(device); mhead=MulticlassHead(n=NC).to(device)
t2v.load_state_dict(ckpt_g['t2v']); enc.load_state_dict(ckpt_g['enc'])
bhead.load_state_dict(ckpt_f['bhead']); mhead.load_state_dict(ckpt_g['mhead'])
for m in [t2v,enc,bhead,mhead]:
    for p in m.parameters(): p.requires_grad=False; m.eval()

with open(WORKING/'checkpoints'/'F_binary'/'threshold.json') as f: bin_th=json.load(f)['threshold']
with open(WORKING/'checkpoints'/'G_multiclass'/'thresholds.json') as f: pct=json.load(f)['per_class_thresholds']
with open(WORKING/'checkpoints'/'H_prototypical'/'tau.json') as f: tau_d=json.load(f); glob_tau=tau_d['global_tau']

# ---- Load test graphs ----
G_test=[]
for ds in ['NF-CICIDS2018','NF-UNSW-NB15']:
    p=INPUT_DIR/f'{ds}_test_list.pt'
    if p.exists(): G_test.extend(torch.load(p,weights_only=False))
G_train=[]
for ds in ['NF-CICIDS2018','NF-UNSW-NB15']:
    p=INPUT_DIR/f'{ds}_train_list.pt'
    if p.exists(): G_train.extend(torch.load(p,weights_only=False))

all_times=torch.cat([g.edge_time for g in G_train]); T_MIN,T_MAX=all_times.min().item(),all_times.max().item()
def nt(t): return (t-T_MIN)/(T_MAX-T_MIN)

def infer(graphs):
    preds,targs=[],[]
    with torch.no_grad():
        for g in graphs:
            g=g.to(device); tn=nt(g.edge_time); te=t2v(tn); ef=torch.cat([g.edge_attr,te],dim=-1)
            fr=enc(Data(edge_index=g.edge_index,edge_attr=ef,num_nodes=g.num_nodes))
            bp=F.softmax(bhead(fr),dim=-1); am=bp[:,1]>=bin_th
            pr=torch.zeros(fr.shape[0],dtype=torch.long,device=device)
            if am.any():
                mp=F.softmax(mhead(fr[am]),dim=-1); mv,mc=mp.max(dim=-1)
                pr[am]=mc
            preds.extend(pr.cpu().tolist()); targs.extend(g.y.cpu().tolist())
    return np.array(preds),np.array(targs)

# ---- 1. IN-DOMAIN EVAL ----
print("In-domain eval...")
tp,tt=infer(G_test)
id_f1=f1_score(tt,tp,average='macro')
print(f"Macro-F1: {id_f1:.4f}")
for i,u in enumerate(UNIFIED):
    if (tt==i).sum()>0: print(f"  {u:25s}: F1={f1_score(tt==i,tp==i):.4f}")

# ---- 2. CROSS-DATASET ----
cdr=[{'dataset':'In-Domain','schema':'in-schema','macro_f1':round(float(id_f1),4)}]
for ds_name in ['NF-ToN-IoT','NF-BoT-IoT']:
    p=INPUT_DIR/f'{ds_name}_test_list.pt'
    if p.exists():
        gs=torch.load(p,weights_only=False); bp_,bt_=infer(gs)
        f1_=f1_score(bt_,bp_,average='macro')
        cdr.append({'dataset':ds_name,'schema':'in-schema','macro_f1':round(float(f1_),4)})
        print(f"{ds_name}: F1={f1_:.4f}")
    else:
        cdr.append({'dataset':ds_name,'schema':'in-schema','macro_f1':'N/A','status':'graphs not built'})
# Out-of-schema (deferred)
for ds_name in ['CIC-DDoS2019','CIC-Darknet2020']:
    cdr.append({'dataset':ds_name,'schema':'out-of-schema','macro_f1':'N/A','status':'dataset unavailable'})

tab06=pd.DataFrame(cdr); tab06.to_csv(TABS_DIR/'tab06_cross_dataset_results.csv',index=False)

# Fig 11
valid=[r for r in cdr if r['macro_f1']!='N/A']
if valid:
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(range(len(valid)),[r['macro_f1'] for r in valid],color=['#4CAF50']+['#2196F3']*(len(valid)-1),edgecolor='black')
    ax.set_xticks(range(len(valid))); ax.set_xticklabels([r['dataset'] for r in valid],fontsize=8)
    ax.set_ylabel('Macro-F1'); ax.set_ylim(0,1.05); ax.set_title('Fig 11: Cross-Dataset Generalization'); ax.grid(axis='y',alpha=0.3)
    for i,r_ in enumerate(valid): ax.text(i,r_['macro_f1']+0.01,f"{r_['macro_f1']:.3f}",ha='center',fontweight='bold')
    plt.tight_layout(); plt.savefig(FIGS_DIR/'fig11_cross_dataset_bar_chart.png',dpi=150); plt.close()

# ---- 3. ADVERSARIAL ROBUSTNESS ----
print("\nAdversarial robustness...")
EPS=[0.0,0.01,0.03,0.05]; adv_r=[]
fm61=torch.cat([torch.tensor([-4.0]*44),torch.full((17,),-4.0)]).to(device)
fx61=torch.cat([torch.tensor([4.0]*44),torch.full((17,),4.0)]).to(device)
G_sub=[]
for g in G_test[:5]:
    if g.edge_index.shape[1]>2000:
        ix=torch.randperm(g.edge_index.shape[1])[:2000]
        g.edge_index=g.edge_index[:,ix]; g.edge_attr=g.edge_attr[ix]; g.edge_time=g.edge_time[ix]; g.y=g.y[ix]
    G_sub.append(g)

for eps in EPS:
    ap_=[]; at_=[]
    for g in G_sub:
        g=g.to(device); tn=nt(g.edge_time); te=t2v(tn); ef_clean=torch.cat([g.edge_attr,te],dim=-1)
        ef=ef_clean.clone()
        if eps>0:
            for _ in range(7):
                ef=ef.clone().detach().requires_grad_(True)
                fr=enc(Data(edge_index=g.edge_index,edge_attr=ef,num_nodes=g.num_nodes))
                lo=mhead(fr); loss=-F.cross_entropy(lo,g.y)
                gr=torch.autograd.grad(loss,ef)[0]; ef=ef.detach()+0.01*gr.sign()
                d=torch.clamp(ef-ef_clean,-eps,eps); ef=torch.clamp(ef_clean+d,fm61,fx61)
        pr=mhead(enc(Data(edge_index=g.edge_index,edge_attr=ef,num_nodes=g.num_nodes))).argmax(dim=1)
        ap_.extend(pr.cpu().tolist()); at_.extend(g.y.cpu().tolist())
    f1_=f1_score(at_,ap_,average='macro'); adv_r.append({'epsilon':eps,'macro_f1':round(float(f1_),4)})
    print(f"  ε={eps}: F1={f1_:.4f}")

tab08=pd.DataFrame(adv_r); tab08.to_csv(TABS_DIR/'tab08_adversarial_robustness.csv',index=False)
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(tab08['epsilon'],tab08['macro_f1'],'o-',color='#E53935',lw=2,ms=10)
ax.set_xlabel('PGD ε'); ax.set_ylabel('Macro-F1'); ax.set_ylim(0,1.05); ax.grid(alpha=0.3)
for _,r_ in tab08.iterrows(): ax.annotate(f"{r_['macro_f1']:.3f}",(r_['epsilon'],r_['macro_f1']),textcoords="offset points",xytext=(0,10),ha='center')
ax.set_title('Fig 10: Adversarial Robustness'); plt.tight_layout()
plt.savefig(FIGS_DIR/'fig10_adversarial_robustness_curve.png',dpi=150); plt.close()

# ---- 4. t-SNE (Fig 09) ----
print("\nt-SNE...")
sample_emb=[]; sample_lab=[]
with torch.no_grad():
    for g in G_test[:5]:
        g=g.to(device)
        if g.edge_index.shape[1]>1000:
            ix=torch.randperm(g.edge_index.shape[1])[:1000]
            g.edge_index=g.edge_index[:,ix]; g.edge_attr=g.edge_attr[ix]; g.edge_time=g.edge_time[ix]; g.y=g.y[ix]
        tn=nt(g.edge_time); te=t2v(tn); ef=torch.cat([g.edge_attr,te],dim=-1)
        fr=enc(Data(edge_index=g.edge_index,edge_attr=ef,num_nodes=g.num_nodes))
        sample_emb.append(fr.cpu()); sample_lab.append(g.y.cpu())
sample_emb=torch.cat(sample_emb,dim=0); sample_lab=torch.cat(sample_lab,dim=0)
if sample_emb.shape[0]>2000:
    ix=torch.randperm(sample_emb.shape[0])[:2000]; sample_emb=sample_emb[ix]; sample_lab=sample_lab[ix]

tsne=TSNE(n_components=2,random_state=SEED,perplexity=30,n_iter=1000)
emb_2d=tsne.fit_transform(sample_emb.numpy())
fig,ax=plt.subplots(figsize=(10,8))
colors=plt.cm.tab10(np.linspace(0,1,NC))
for i,u in enumerate(UNIFIED):
    m=sample_lab.numpy()==i
    if m.sum()>0: ax.scatter(emb_2d[m,0],emb_2d[m,1],c=[colors[i]],label=u,alpha=0.5,s=5)
ax.legend(fontsize=7,markerscale=3); ax.set_title('Fig 09: t-SNE Embeddings'); plt.tight_layout()
plt.savefig(FIGS_DIR/'fig09_tsne_embeddings.png',dpi=150); plt.close()

# ---- 5. LATENCY (Tab 10) ----
print("\nLatency benchmark...")
g0=G_test[0].clone().to(device); se=g0.edge_index[:,:1]; sa=g0.edge_attr[:1]; st_=g0.edge_time[:1]
for _ in range(50):
    tn=nt(st_); te=t2v(tn); ef=torch.cat([sa,te],dim=-1)
    _=enc(Data(edge_index=se,edge_attr=ef,num_nodes=2))
torch.cuda.synchronize(); t0=time.time()
for _ in range(500):
    tn=nt(st_); te=t2v(tn); ef=torch.cat([sa,te],dim=-1)
    fr=enc(Data(edge_index=se,edge_attr=ef,num_nodes=2)); _=bhead(fr); _=mhead(fr)
torch.cuda.synchronize(); sl=(time.time()-t0)/500*1000

bs_=1024; gb=G_test[0].clone().to(device)
if gb.edge_index.shape[1]>bs_:
    ix=torch.randperm(gb.edge_index.shape[1])[:bs_]; gb.edge_index=gb.edge_index[:,ix]
    gb.edge_attr=gb.edge_attr[ix]; gb.edge_time=gb.edge_time[ix]
for _ in range(20):
    tn=nt(gb.edge_time); te=t2v(tn); ef=torch.cat([gb.edge_attr,te],dim=-1)
    _=enc(Data(edge_index=gb.edge_index,edge_attr=ef,num_nodes=gb.num_nodes))
torch.cuda.synchronize(); t0=time.time()
for _ in range(100):
    tn=nt(gb.edge_time); te=t2v(tn); ef=torch.cat([gb.edge_attr,te],dim=-1)
    fr=enc(Data(edge_index=gb.edge_index,edge_attr=ef,num_nodes=gb.num_nodes)); _=bhead(fr); _=mhead(fr)
torch.cuda.synchronize(); bl=(time.time()-t0)/100*1000

tp_=sum(p.numel() for m in [t2v,enc,bhead,mhead] for p in m.parameters())
pd.DataFrame([
    {'metric':'Single-flow (ms)','value':round(sl,4),'target':'<30ms'},
    {'metric':f'Batch {gb.edge_index.shape[1]} flows (ms)','value':round(bl,4),'target':'-'},
    {'metric':'Per-flow amortized (ms)','value':round(bl/gb.edge_index.shape[1],4),'target':'-'},
    {'metric':'Params','value':f'{tp_:,}','target':'-'},
    {'metric':'Size MB (fp32)','value':round(tp_*4/1024/1024,1),'target':'-'},
]).to_csv(TABS_DIR/'tab10_inference_latency.csv',index=False)
print(f"Single: {sl:.2f}ms, Batch: {bl:.2f}ms")

# ---- 6. ATTENTION VIS (Fig 14) ----
print("\nAttention viz...")
sg=G_test[0].clone().to(device)
if sg.edge_index.shape[1]>100:
    ix=torch.randperm(sg.edge_index.shape[1])[:100]; sg.edge_index=sg.edge_index[:,ix]
    sg.edge_attr=sg.edge_attr[ix]; sg.edge_time=sg.edge_time[ix]
tn=nt(sg.edge_time); te=t2v(tn); ef=torch.cat([sg.edge_attr,te],dim=-1)
_,attn_w=enc(Data(edge_index=sg.edge_index,edge_attr=ef,num_nodes=sg.num_nodes),return_attn=True)
fa=attn_w[-1]
if isinstance(fa,tuple):
    ae,aw=fa; avg_aw=aw.mean(dim=1).cpu().numpy()
    fig,ax=plt.subplots(figsize=(10,8))
    nn_=sg.num_nodes; pos={i:(np.cos(2*np.pi*i/nn_)+np.random.randn()*0.05,np.sin(2*np.pi*i/nn_)+np.random.randn()*0.05) for i in range(nn_)}
    en=ae.cpu().numpy()
    for i in range(en.shape[1]):
        u,v=en[0,i],en[1,i]
        if u in pos and v in pos:
            al=min(avg_aw[i]/avg_aw.max(),1.0)*0.8+0.2
            ax.plot([pos[u][0],pos[v][0]],[pos[u][1],pos[v][1]],'gray',alpha=al,lw=avg_aw[i]*3)
    for n_ in range(nn_): ax.scatter(pos[n_][0],pos[n_][1],s=50,c='#2196F3',edgecolors='black',lw=0.5,zorder=3)
    ax.set_title('Fig 14: Attention Viz'); ax.axis('off'); plt.tight_layout()
    plt.savefig(FIGS_DIR/'fig14_attention_visualization.png',dpi=150); plt.close()

# ---- 7. RESULTS_SUMMARY + VERIFICATION ----
summary=f"""# RESULTS SUMMARY
Generated: {datetime.now(timezone.utc).isoformat()}

## In-Domain
- Macro-F1: {id_f1:.4f} (test set, CICIDS2018+UNSW-NB15 chronological split)

## Cross-Dataset
"""
for r_ in cdr: summary+=f"- {r_['dataset']} ({r_['schema']}): {r_['macro_f1']}\n"
summary+="\n## Adversarial Robustness\n"
for _,r_ in tab08.iterrows(): summary+=f"- ε={r_['epsilon']:.2f}: F1={r_['macro_f1']:.4f}\n"
summary+=f"\n## Latency\n- Single-flow: {sl:.2f} ms\n- Batch ({gb.edge_index.shape[1]}): {bl:.2f} ms\n"
with open(WORKING/'RESULTS_SUMMARY.md','w') as f: f.write(summary)

# Verification
req_figs=['fig01_architecture_diagram','fig08_class_distribution','fig09_tsne_embeddings',
          'fig10_adversarial_robustness_curve','fig11_cross_dataset_bar_chart',
          'fig12_zero_day_roc_pr','fig14_attention_visualization','fig15_confusion_matrix']
req_tabs=['tab05_main_results','tab06_cross_dataset_results','tab08_adversarial_robustness',
          'tab09_zero_day_results','tab10_inference_latency']
all_ok=True
print("\nVerification:")
for s in req_figs:
    ok=(FIGS_DIR/f'{s}.png').exists(); all_ok&=ok; print(f"  [{'✓' if ok else '✗'}] {s}")
for s in req_tabs:
    ok=(TABS_DIR/f'{s}.csv').exists(); all_ok&=ok; print(f"  [{'✓' if ok else '✗'}] {s}")

json.dump({'notebook':'K06','in_domain_f1':float(id_f1),'verification':all_ok,'single_latency_ms':float(sl)},
          open(LOGS_DIR/'k06_log.json','w'),indent=2,default=str)
print(f"\n{'✓ ALL VERIFIED' if all_ok else '✗ SOME MISSING'}")
print(f"In-domain Macro-F1: {id_f1:.4f}")
print("Pipeline complete. Remaining: ablation variants (4× retraining), CIC datasets, paper finalization.")
