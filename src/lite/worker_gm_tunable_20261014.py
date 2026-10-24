"""Torch-only isolated Trial120 hybrid fit/replay worker; intentionally no sklearn tree imports."""
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
GRAPH_COUNT = 9


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
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
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


class GatedMeanBlock(nn.Module):
    def __init__(self):
        super().__init__(); w=WIDTH
        self.self_proj=nn.Linear(w,w,bias=False)
        self.in_proj=nn.Linear(w,w,bias=False)
        self.out_proj=nn.Linear(w,w,bias=False)
        self.gates=nn.Linear(w*3,w*2)
        self.norm=nn.LayerNorm(w)
        self.out=nn.Linear(w,w,bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self,h,ain,aout):
        inc=torch.sparse.mm(ain,h); out=torch.sparse.mm(aout,h)
        hs=self.self_proj(h); hi=self.in_proj(inc); ho=self.out_proj(out)
        gates=torch.sigmoid(self.gates(torch.cat([hs,hi,ho],dim=1)))
        gi,go=gates.chunk(2,dim=1)
        return h+self.out(self.norm(torch.relu(hs+gi*hi+go*ho)))


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
        return torch.sparse_coo_tensor(ix,torch.as_tensor(coo.data,dtype=torch.float32),coo.shape,check_invariants=True).coalesce()
    return xb,mb,xg,mg,attr_t,sparse(partition["incoming"]),sparse(partition["outgoing"])


def _score(prob):
    return prob[:,1]/np.clip(prob[:,0]+prob[:,1],1e-12,None)


def _dropedge_adjacencies(ain, aout, drop, epoch):
    gen=torch.Generator().manual_seed(SEED+15000+epoch)
    n=ain.shape[0]
    idx=ain.indices()
    vals=ain.values()
    keep=(torch.rand(vals.shape[0],generator=gen)>=drop)
    rows=idx[0][keep]; cols=idx[1][keep]
    if rows.numel()==0:
        ain_new=torch.sparse_coo_tensor(torch.zeros((2,0),dtype=torch.long),
            torch.zeros((0,),dtype=torch.float32),(n,n),check_invariants=True).coalesce()
        aout_new=torch.sparse_coo_tensor(torch.zeros((2,0),dtype=torch.long),
            torch.zeros((0,),dtype=torch.float32),(n,n),check_invariants=True).coalesce()
        return ain_new,aout_new
    deg_in=torch.zeros(n,dtype=torch.float32)
    deg_in.index_add_(0,rows,torch.ones(rows.shape[0],dtype=torch.float32))
    w_in=torch.ones(rows.shape[0],dtype=torch.float32)/deg_in[rows].clamp_min(1e-12)
    ain_new=torch.sparse_coo_tensor(torch.stack([rows,cols]),w_in,(n,n),
        check_invariants=True).coalesce()
    src=cols; tgt=rows
    deg_out=torch.zeros(n,dtype=torch.float32)
    deg_out.index_add_(0,src,torch.ones(src.shape[0],dtype=torch.float32))
    w_out=torch.ones(src.shape[0],dtype=torch.float32)/deg_out[src].clamp_min(1e-12)
    aout_new=torch.sparse_coo_tensor(torch.stack([src,tgt]),w_out,(n,n),
        check_invariants=True).coalesce()
    return ain_new,aout_new


def fit(job):
    torch.set_num_threads(1); torch.manual_seed(SEED); np.random.seed(SEED)
    ex=job.get("extra",{}) or {}
    drop=float(ex.get("drop",0.0))
    mask_rate=float(ex.get("mask",MASK_RATE))
    frac=float(ex.get("frac",1.0))
    layers=int(ex.get("layers",1))
    epochs=int(ex.get("epochs",EPOCHS))
    hist, future=job["history"],job["future"]
    if hist["raw"].shape[1]!=BASE_COUNT+GRAPH_COUNT or future["raw"].shape[1]!=BASE_COUNT+GRAPH_COUNT:
        raise ValueError("Trial120 worker expected 172 base plus nine graph columns")
    if job["columns"][-GRAPH_COUNT:] != job["graph_columns"] or len(job["columns"]) != BASE_COUNT+GRAPH_COUNT:
        raise ValueError("Trial120 feature order must be 172 base columns followed by graph9")
    hist_raw=hist["raw"]
    future_raw=future["raw"]
    if frac<1.0:
        rng=np.random.default_rng(SEED+31337)
        keep=sorted(rng.choice(BASE_COUNT,size=int(round(frac*BASE_COUNT)),replace=False))
        keep_set=set(keep)
        drop_cols=[c for c in range(BASE_COUNT) if c not in keep_set]
        hist_raw=hist_raw.copy()
        future_raw=future_raw.copy()
        hist_attr=hist["is_attr"].astype(bool,copy=False)
        if hist_attr.any():
            hist_means=hist_raw[hist_attr][:,drop_cols].mean(axis=0)
        else:
            hist_means=np.zeros(len(drop_cols),dtype=hist_raw.dtype)
        hist_raw[np.ix_(np.arange(len(hist_raw)),drop_cols)]=hist_means[None,:]
        future_raw[np.ix_(np.arange(len(future_raw)),drop_cols)]=hist_means[None,:]
    scaler=StandardScaler().fit(hist_raw[hist["is_attr"]])
    hist_part={"raw":hist_raw,"is_attr":hist["is_attr"],"incoming":hist["incoming"],"outgoing":hist["outgoing"]}
    future_part={"raw":future_raw,"is_attr":future["is_attr"],"incoming":future["incoming"],"outgoing":future["outgoing"]}
    train_x=_tensor_inputs(hist_part,scaler.mean_,scaler.scale_)
    predict_x=_tensor_inputs(future_part,scaler.mean_,scaler.scale_)
    y=torch.as_tensor(hist["labels"],dtype=torch.long)
    weights=torch.as_tensor(hist["weights"],dtype=torch.float32)
    train_idx=torch.as_tensor(np.flatnonzero(hist["labels"]>=0),dtype=torch.long)
    attr_idx=torch.as_tensor(np.flatnonzero(hist["is_attr"]),dtype=torch.long)
    model=GatedMean()
    if layers==2:
        rng_state=torch.get_rng_state()
        block=GatedMeanBlock()
        torch.set_rng_state(rng_state)
    else:
        block=None
    params=list(model.parameters())
    if block is not None:
        params=params+list(block.parameters())
    optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=.001)
    losses=[]; masked=[]
    for epoch in range(epochs):
        model.train()
        if block is not None: block.train()
        aug=train_x[1].clone()
        keep=(torch.rand((len(attr_idx),1))>=mask_rate).to(torch.float32).expand(-1,BASE_COUNT)
        aug[attr_idx]=keep; masked.append(int((keep==0).sum()))
        if drop>0.0:
            ain_ep,aout_ep=_dropedge_adjacencies(train_x[5],train_x[6],drop,epoch)
        else:
            ain_ep,aout_ep=train_x[5],train_x[6]
        h0=model.initial(train_x[0],aug,train_x[2],train_x[3],train_x[4])
        inc=torch.sparse.mm(ain_ep,h0); out=torch.sparse.mm(aout_ep,h0)
        hs=model.self_proj(h0); hi=model.in_proj(inc); ho=model.out_proj(out)
        gates=torch.sigmoid(model.gates(torch.cat([hs,hi,ho],dim=1)))
        gi,go=gates.chunk(2,dim=1)
        h=model.norm(torch.relu(hs+gi*hi+go*ho))
        if block is not None:
            h=block(h,ain_ep,aout_ep)
        h=model.dropout(h)
        logits=torch.as_tensor(hist["teacher_logits"],dtype=torch.float32)+torch.tanh(model.head(h))
        per=nn.functional.cross_entropy(logits[train_idx],y[train_idx],reduction="none")
        w=weights[train_idx]; loss=(per*w).sum()/w.sum()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
    model.eval()
    if block is not None: block.eval()
    with torch.no_grad():
        h0=model.initial(predict_x[0],predict_x[1],predict_x[2],predict_x[3],predict_x[4])
        inc=torch.sparse.mm(predict_x[5],h0); out=torch.sparse.mm(predict_x[6],h0)
        hs=model.self_proj(h0); hi=model.in_proj(inc); ho=model.out_proj(out)
        gates=torch.sigmoid(model.gates(torch.cat([hs,hi,ho],dim=1)))
        gi,go=gates.chunk(2,dim=1)
        h=model.norm(torch.relu(hs+gi*hi+go*ho))
        if block is not None:
            h=block(h,predict_x[5],predict_x[6])
        h=model.dropout(h)
        logits=torch.as_tensor(future["teacher_logits"],dtype=torch.float32)+torch.tanh(model.head(h))
        prob=torch.softmax(logits,dim=1).cpu().numpy()
    if not np.isfinite(prob).all() or not np.allclose(prob.sum(1),1,rtol=0,atol=1e-6):
        raise ValueError("Trial120 probability output invalid")
    state={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
    if block is not None:
        for k,v in block.state_dict().items():
            state["block."+k]=v.detach().cpu().numpy().copy()
    state_hash=hashlib.sha256(b"".join(k.encode()+state[k].tobytes() for k in sorted(state))).hexdigest()
    bundle={"state_dict_numpy":state,"state_dict_sha256":state_hash,
      "scaler_mean":scaler.mean_.copy(),"scaler_scale":scaler.scale_.copy(),"configuration":job["configuration"],
      "columns":job["columns"],"origin":job["origin"],"stop":job["stop"],"stage":job["stage"],
      "seed":SEED,"epochs":epochs,"layers":layers,"drop":drop,"mask":mask_rate,"frac":frac}
    return {"probabilities":prob,"score":_score(prob),"bundle":bundle,
      "diagnostics":{"epochs":epochs,"loss_first":losses[0],"loss_last":losses[-1],
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
    layers=int(bundle.get("layers",1))
    block=None
    if layers==2:
        block=GatedMeanBlock()
    state={}
    block_state={}
    for k,v in bundle["state_dict_numpy"].items():
        if k.startswith("block."):
            block_state[k[len("block."):]]=torch.as_tensor(v)
        else:
            state[k]=torch.as_tensor(v)
    model.load_state_dict(state)
    if block is not None:
        block.load_state_dict(block_state)
    model.eval()
    if block is not None: block.eval()
    x=_tensor_inputs(job["future"],np.asarray(bundle["scaler_mean"]),np.asarray(bundle["scaler_scale"]))
    with torch.no_grad():
        h0=model.initial(x[0],x[1],x[2],x[3],x[4])
        inc=torch.sparse.mm(x[5],h0); out=torch.sparse.mm(x[6],h0)
        hs=model.self_proj(h0); hi=model.in_proj(inc); ho=model.out_proj(out)
        gates=torch.sigmoid(model.gates(torch.cat([hs,hi,ho],dim=1)))
        gi,go=gates.chunk(2,dim=1)
        h=model.norm(torch.relu(hs+gi*hi+go*ho))
        if block is not None:
            h=block(h,x[5],x[6])
        h=model.dropout(h)
        prob=torch.softmax(torch.as_tensor(job["future"]["teacher_logits"],dtype=torch.float32)+torch.tanh(model.head(h)),dim=1).cpu().numpy()
    if not np.isfinite(prob).all() or not np.allclose(prob.sum(1),1,rtol=0,atol=1e-6):
        raise ValueError("Trial120 replay probabilities invalid")
    return {"probabilities":prob,"score":_score(prob)}


if __name__=="__main__":
    mode,input_path,output_path=sys.argv[1:4]
    job=joblib.load(input_path)
    result=fit(job) if mode=="fit" else replay(job) if mode=="replay" else None
    if result is None: raise SystemExit("mode must be fit or replay")
    joblib.dump(result,output_path)
