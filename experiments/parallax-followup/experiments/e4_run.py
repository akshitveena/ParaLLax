import sys, json, numpy as np, torch
sys.path.insert(0,'main'); sys.path.insert(0,'experiments/parallax-followup/experiments')
from sdae_prm import StepSDAE_PRM
from train_sdae import StepDS, collate
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import e4_nonlinear_control as e4
recs = torch.load('data/step_cache.pt', weights_only=False)
meta = {json.loads(l)['record_id']:json.loads(l) for l in open('data/processed_pb/candidates.jsonl') if l.strip()}
ychain = np.array([r['chain'] for r in recs]); y=(ychain=='A').astype(int)
L,LA,NS,DS=[],[],[],[]
for r in recs:
    t=meta.get(r['id'],{}).get('response_text') or meta.get(r['id'],{}).get('full_text') or ""
    L.append(np.log1p(len(t.split()))); LA.append((t.count(chr(92))+t.count('$'))/max(len(t.split()),1))
    NS.append(len(r['steps_text'])); DS.append(r['split'])
Doh=OneHotEncoder(sparse_output=False,handle_unknown='ignore').fit_transform(np.array(DS).reshape(-1,1))
Z=np.c_[np.array(L),np.array(LA),np.array(NS,float),Doh]
m=StepSDAE_PRM(); m.load_state_dict(torch.load('results/results_multiseed/ckpts/frozen_seed0/sdae_best.pt',map_location='cpu')); m.eval()
Zc=[]
with torch.no_grad():
    for X_,SL,pad,ch in DataLoader(StepDS(recs),batch_size=64,collate_fn=collate,shuffle=False):
        _,_,_,pl=m(X_,pad,None); Zc.append(pl.numpy())
X=np.concatenate(Zc,0)
rng=np.random.RandomState(0); idx=np.arange(len(y)); rng.shuffle(idx); cut=int(.8*len(y)); tr,va=idx[:cut],idx[cut:]
split={'train':(X[tr],Z[tr],y[tr]),'val':(X[va],Z[va],y[va])}
e4.load_representation_confounds_labels=lambda s: split[s]
e4.f1b_at_paper_threshold=lambda scores,yv: f1_score((yv==0).astype(int),(scores<0.5).astype(int))
print("X",X.shape,"Z",Z.shape,flush=True)
res=e4.run(kinds=("linear","krr"))
print(json.dumps(res,indent=2))
