"""Torch-only isolated Trial98 fit/replay worker; intentionally no sklearn tree imports."""
from __future__ import annotations

import sys
import hashlib
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

SEED = 42
WIDTH = 32
EPOCHS = 60
MASK_RATE = 0.25
BASE_COUNT = 172
GRAPH_COUNT = 5


class GatedMean(nn.Module):
    def __init__(self):
        super().__init__(); w=WIDTH
        self.w_value=nn.Parameter(torch.empty(BASE_COUNT,w))
        self.w_avail=nn.Parameter(torch.empty(BASE_COUNT,w))
        self.graph_proj=nn.Linear(GRAPH_COUNT*2,w,bias=False)
        self.bias=nn.Parameter(torch.zeros(w))
        self.missing_token=nn.Parameter(torch.zeros(BASE_COUNT))
        self.self_proj=nn.Linear(w,w,bias=False)
        self.in_proj=nn.Linear(w,w,bias=False)
        self.out_proj=nn.Linear(w,w,bias=False)
        self.gates=nn.Linear(w*3,w*2)
        self.norm=nn.LayerNorm(w)
        self.dropout=nn.Dropout(.1)
        self.head=nn.Linear(w,3)
        nn.init.xavier_uniform_(self.w_value); nn.init.xavier_uniform_(self.w_avail)

    def initial(self, xb, mb, xg, mg, attr):
        h=torch.empty((len(xb),WIDTH),dtype=xb.dtype)
        ids=torch.nonzero(attr,as_tuple=True)[0]
        token_projection=self.missing_token @ self.w_value
        if ids.numel():
            observed=torch.where(mb[ids]>0,xb[ids],self.missing_token[None,:])
            h[ids]=observed @ self.w_value + mb[ids] @ self.w_avail
        missing_ids=torch.nonzero(~attr,as_tuple=True)[0]
        if missing_ids.numel(): h[missing_ids]=token_projection
        return h+self.graph_proj(torch.cat([xg,mg],dim=1))+self.bias

    def forward(self,xb,mb,xg,mg,attr,ain,aout):
        h0=self.initial(xb,mb,xg,mg,attr)
        inc=torch.sparse.mm(ain,h0); out=torch.sparse.mm(aout,h0)
        hs=self.self_proj(h0); hi=self.in_proj(inc); ho=self.out_proj(out)
        gates=torch.sigmoid(self.gates(torch.cat([hs,hi,ho],dim=1)))
        gi,go=gates.chunk(2,dim=1)
        h=self.dropout(self.norm(torch.relu(hs+gi*hi+go*ho)))
        return self.head(h)


def _tensor_inputs(partition, mean, scale):
    raw=partition["raw"].astype(np.float64,copy=False)
    x=np.clip((raw-mean)/scale,-5,5).astype(np.float32)
    attr=partition["is_attr"].astype(bool,copy=False)
    x[np.ix_(~attr,np.arange(BASE_COUNT))]=0.0
    xb=torch.as_tensor(x[:,:BASE_COUNT],dtype=torch.float32)
    xg=torch.as_tensor(x[:,BASE_COUNT:],dtype=torch.float32)
    mb=torch.as_tensor(np.repeat(attr[:,None],BASE_COUNT,axis=1),dtype=torch.float32)
    mg=torch.ones((len(x),GRAPH_COUNT),dtype=torch.float32)
    attr_t=torch.as_tensor(attr,dtype=torch.bool)
    def sparse(v):
        coo=v.tocoo(); ix=torch.as_tensor(np.vstack([coo.row,coo.col]),dtype=torch.long)
        return torch.sparse_coo_tensor(ix,torch.as_tensor(coo.data,dtype=torch.float32),coo.shape).coalesce()
    return xb,mb,xg,mg,attr_t,sparse(partition["incoming"]),sparse(partition["outgoing"])


def _score(prob):
    return prob[:,1]/np.clip(prob[:,0]+prob[:,1],1e-12,None)


def fit(job):
    torch.set_num_threads(1); torch.manual_seed(SEED); np.random.seed(SEED)
    hist, future=job["history"],job["future"]
    if hist["raw"].shape[1]!=BASE_COUNT+GRAPH_COUNT or future["raw"].shape[1]!=BASE_COUNT+GRAPH_COUNT:
        raise ValueError("Trial98 worker expected 172 base plus five graph columns")
    if job["columns"][-GRAPH_COUNT:] != job["graph_columns"] or len(job["columns"]) != BASE_COUNT+GRAPH_COUNT:
        raise ValueError("Trial98 feature order must be 172 base columns followed by graph5")
    scaler=StandardScaler().fit(hist["raw"][hist["is_attr"]])
    train_x=_tensor_inputs(hist,scaler.mean_,scaler.scale_)
    predict_x=_tensor_inputs(future,scaler.mean_,scaler.scale_)
    y=torch.as_tensor(hist["labels"],dtype=torch.long)
    weights=torch.as_tensor(hist["weights"],dtype=torch.float32)
    train_idx=torch.as_tensor(np.flatnonzero(hist["labels"]>=0),dtype=torch.long)
    attr_idx=torch.as_tensor(np.flatnonzero(hist["is_attr"]),dtype=torch.long)
    model=GatedMean(); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    losses=[]; masked=[]
    for _ in range(EPOCHS):
        model.train(); aug=train_x[1].clone()
        keep=(torch.rand((len(attr_idx),1))>=MASK_RATE).to(torch.float32).expand(-1,BASE_COUNT)
        aug[attr_idx]=keep; masked.append(int((keep==0).sum()))
        logits=model(train_x[0],aug,*train_x[2:])
        per=nn.functional.cross_entropy(logits[train_idx],y[train_idx],reduction="none")
        w=weights[train_idx]; loss=(per*w).sum()/w.sum()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        logits=model(predict_x[0],predict_x[1],*predict_x[2:])
        prob=torch.softmax(logits,dim=1).cpu().numpy()
    if not np.isfinite(prob).all() or not np.allclose(prob.sum(1),1,rtol=0,atol=1e-6):
        raise ValueError("Trial98 probability output invalid")
    state={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
    state_hash=hashlib.sha256(b"".join(k.encode()+state[k].tobytes() for k in sorted(state))).hexdigest()
    bundle={"state_dict_numpy":state,"state_dict_sha256":state_hash,
      "scaler_mean":scaler.mean_.copy(),"scaler_scale":scaler.scale_.copy(),"configuration":job["configuration"],
      "columns":job["columns"],"origin":job["origin"],"stop":job["stop"],"stage":job["stage"],
      "seed":SEED,"epochs":EPOCHS}
    return {"probabilities":prob,"score":_score(prob),"bundle":bundle,
      "diagnostics":{"epochs":EPOCHS,"loss_first":losses[0],"loss_last":losses[-1],
      "masked_base_cells_by_epoch":masked,"masked_row_count_mean":float(np.mean(masked)/BASE_COUNT),
      "history_nodes":len(hist["ids"]),"history_attributed_nodes":int(hist["is_attr"].sum()),
      "history_graph_only_nodes":int((~hist["is_attr"]).sum()),"future_nodes":len(future["ids"]),
      "future_attributed_nodes":int(future["is_attr"].sum()),"future_graph_only_nodes":int((~future["is_attr"]).sum()),
      "history_edge_count":hist["edge_count"],"future_edge_count":future["edge_count"],
      "history_max_period":int(hist["periods"].max()),"future_min_period":int(future["periods"].min()),
      "train_status_counts":np.bincount(hist["labels"][hist["labels"]>=0],minlength=3).tolist(),
      "train_status_weight_mass":[float(hist["weights"][hist["labels"]==k].sum()) for k in range(3)],
      "unknown_status_weight_share":float(hist["weights"][hist["labels"]==2].sum()/hist["weights"].sum()),
      "torch_threads":torch.get_num_threads(),"seed":SEED}}


def replay(job):
    torch.set_num_threads(1)
    bundle=job["bundle"]; model=GatedMean()
    model.load_state_dict({k:torch.as_tensor(v) for k,v in bundle["state_dict_numpy"].items()}); model.eval()
    x=_tensor_inputs(job["future"],np.asarray(bundle["scaler_mean"]),np.asarray(bundle["scaler_scale"]))
    with torch.no_grad(): prob=torch.softmax(model(*x),dim=1).cpu().numpy()
    if not np.isfinite(prob).all() or not np.allclose(prob.sum(1),1,rtol=0,atol=1e-6):
        raise ValueError("Trial98 replay probabilities invalid")
    return {"probabilities":prob,"score":_score(prob)}


if __name__=="__main__":
    mode,input_path,output_path=sys.argv[1:4]
    job=joblib.load(input_path)
    result=fit(job) if mode=="fit" else replay(job) if mode=="replay" else None
    if result is None: raise SystemExit("mode must be fit or replay")
    joblib.dump(result,output_path)
