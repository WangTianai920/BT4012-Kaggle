"""Trial120: directed missing-aware graph network on successful181 inputs, exact public117 control."""
import csv,hashlib,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
import os,subprocess
from scipy.sparse import coo_matrix
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from src.combined_challengers import load_features
from src.horizon_hgb_20260917 import historical_columns
from src.block_cycle2_public115_20261008 import GRAPH5,CUT2,BLOCK2,attach_block2,attach_cut2,attach_graph5,GRAPH5_PATH,CUT2_PATH,GRAPH_FEATURES_PATH as BLOCK_PATH
from src.label_status_multiclass_20260929 import label_status
from src.status_mass40_lr03_iter600_20261003 import training_weights_40

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';P='directed_block_gnn120_public117_20261008';SOURCE=ROOT/'src'/f'{P}.py';PLAN=R/f'{P}_plan.json';AUTH=R/f'{P}_root_authorization.json';MODELS=ROOT/'models'/P
ORIGINS={14:28,21:35,17:24,24:31,28:35};CONF={17:31,18:32,20:34};KEY=['origin','txId','time_step','label']
CONTROLS=('trial115_selected_blend','trial116_selected_blend','trial117_standalone');STANDALONE='directed_block_gnn120';FALLBACK='blend_75_25_trial117_gnn120'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def config():return dict(architecture='frozen HGB teacher plus tanh-bounded directed gated mean GraphSAGE residual',width=32,epochs=60,seed=42,mask_rate=.25,base_count=172,graph_count=9,optimizer='AdamW',learning_rate=.001,weight_decay=.001,torch_threads=1)
WORKER=ROOT/'src/directed_block_gnn120_worker_20261008.py'
GRAPH9=GRAPH5+CUT2+BLOCK2
NODEMAP=R/'terminal_coverage4_public99_20261008_graph_node_map.pkl'
EDGEFILE=R/'terminal_coverage4_public99_20261008_deduplicated_edges.pkl'
def partition(frame,periods,map_,edges,graph9,cols,labels=None,weights=None):
    nodes=map_[map_.period.isin(periods)].copy();ids=nodes.txId.to_numpy(np.int64);n=len(ids);pos=np.searchsorted(ids,frame.txId.to_numpy(np.int64));assert np.array_equal(ids[pos],frame.txId)
    raw=np.zeros((n,181),np.float64);raw[:,172:]=graph9.set_index('txId').loc[ids,GRAPH9].to_numpy(float);raw[pos,:]=frame[cols].to_numpy(float)
    attr=np.zeros(n,bool);attr[pos]=True;y=np.full(n,-1,np.int64);w=np.zeros(n,float)
    if labels is not None:y[pos]=labels;w[pos]=weights
    sel=edges[edges.txId1.isin(ids)&edges.txId2.isin(ids)];src=np.searchsorted(ids,sel.txId1);dst=np.searchsorted(ids,sel.txId2)
    assert np.array_equal(nodes.period.to_numpy()[src],nodes.period.to_numpy()[dst])
    def adjacency(row,col):
        deg=np.bincount(row,minlength=n);return coo_matrix((1/deg[row],(row,col)),shape=(n,n)).tocsr()
    return dict(raw=raw,is_attr=attr,labels=y,weights=w,ids=ids,periods=nodes.period.to_numpy(),incoming=adjacency(dst,src),outgoing=adjacency(src,dst),edge_count=len(sel),attributed_positions=pos)
def launch(job,mode,ip,op):
    joblib.dump(job,ip,compress=3);env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run([sys.executable,str(WORKER),mode,str(ip),str(op)],env=env,check=True,cwd=ROOT);return joblib.load(op)

def blend(a,b):return expit(.75*logit(np.clip(a,1e-12,1-1e-12))+.25*logit(np.clip(b,1e-12,1-1e-12)))
def stage_metrics(frame,stage):
    from sklearn.metrics import roc_auc_score,average_precision_score
    rows=[]
    for o,f in frame.groupby('origin',sort=False):
        slices={'first7':f[f.time_step<=o+7]}
        if f.time_step.max()==o+14:slices.update(full14=f,second7=f[f.time_step>o+7])
        for name,q in slices.items():
            for v in (*CONTROLS,STANDALONE,FALLBACK):rows.append(dict(stage=stage,origin=int(o),slice=name,variant=v,n=len(q),positives=int(q.label.sum()),auc=roc_auc_score(q.label,q[v]),average_precision=average_precision_score(q.label,q[v])))
    return pd.DataFrame(rows)
def summary(r,stage):
    # Independent gate algorithm reused on renamed scores; no threshold search.
    from src import verify_recursive_core5_trial118_20261008 as audit
    x=r.copy();x['variant']=x['variant'].replace({STANDALONE:audit.STANDALONE,FALLBACK:audit.FALLBACK})
    out=audit.independent_summary(x,stage);out['variant']=out['variant'].replace({audit.STANDALONE:STANDALONE,audit.FALLBACK:FALLBACK});return out

def run():
    csv.field_size_limit(20000000);p=json.loads(PLAN.read_text());a=json.loads(AUTH.read_text())
    assert p['source_sha256']==sha(SOURCE) and p['worker_sha256']==sha(WORKER) and p['configuration']==config() and p['trial']==120
    assert a['authorized'] and a['worker_sha256']==sha(WORKER) and a['source_sha256']==sha(SOURCE) and a['plan_sha256']==sha(PLAN) and a['registry_rows']==p['registry_rows']
    for rel,h in p['pins'].items():assert sha(ROOT/rel)==h,rel
    rows=list(csv.DictReader((ROOT/'experiment_log.csv').open(newline='')));assert len(rows)==len({r['experiment_id'] for r in rows})==p['registry_rows']
    assert not MODELS.exists() and not list(R.glob(f'{P}_original_*')) and not list(R.glob(f'{P}_confirmation_*'))
    train,test,sample,groups=load_features()
    for path,attach in [(GRAPH5_PATH,attach_graph5),(CUT2_PATH,attach_cut2),(BLOCK_PATH,attach_block2)]:train,test=attach(train,test,pd.read_pickle(path))
    full=list(groups['regularized_two_hop'])+GRAPH5;map_=pd.read_pickle(NODEMAP);edges=pd.read_pickle(EDGEFILE);graph9=pd.read_pickle(GRAPH5_PATH).merge(pd.read_pickle(CUT2_PATH),on='txId',validate='one_to_one').merge(pd.read_pickle(BLOCK_PATH),on='txId',validate='one_to_one');MODELS.mkdir();details=[];allmetrics=[];allsummaries=[];allknown=[]
    def fit(stage,o,end,future):
        h=train[train.time_step<=o];cols=historical_columns(train,full,o)+CUT2+BLOCK2;assert len(cols)==181
        s=label_status(h.label);w,factor=training_weights_40(h.time_step,s,o);assert np.isclose(w[s==2].sum()/w.sum(),.4,atol=1e-12,rtol=0)
        assert future.time_step.min()>o and not np.intersect1d(h.txId,future.txId).size
        stem=f'{stage}_{o}_{end}';ip=R/f'{P}_{stem}_inputs.joblib';mp=MODELS/f'{stem}.joblib';fp=R/f'{P}_{stem}_forecast.pkl'
        x=h[cols];v=future[cols];assert np.isfinite(x.to_numpy()).all() and np.isfinite(v.to_numpy()).all()
        hp=partition(h,range(1,o+1),map_,edges,graph9,cols,s,w);fp_=partition(future,range(o+1,end+1),map_,edges,graph9,cols)
        assert hp['periods'].max()<=o and fp_['periods'].min()>o and np.intersect1d(hp['ids'],fp_['ids']).size==0
        teacher_path=ROOT/f'models/block_cycle2_public115_20261008/{stage}_{o}_{end}_model.joblib';teacher=joblib.load(teacher_path)['model']
        for part in (hp,fp_):
            probabilities=teacher.predict_proba(pd.DataFrame(part['raw'],columns=cols));teacher_logits=np.log(np.clip(probabilities,1e-12,1));teacher_logits[~part['is_attr']]=0.;part['teacher_logits']=teacher_logits.astype(np.float32)
        inputs=dict(teacher_file=str(teacher_path.relative_to(ROOT)),teacher_sha256=sha(teacher_path),history=hp,future=fp_,columns=cols,graph_columns=GRAPH9,origin=o,stop=end,stage=stage,configuration=config(),source_sha256=sha(SOURCE),worker_sha256=sha(WORKER),plan_sha256=sha(PLAN))
        output=R/f'{P}_{stem}_worker_result.joblib';result=launch(inputs,'fit',ip,output);joblib.dump(result['bundle'],mp)
        attrpos=fp_['attributed_positions'];proba=result['probabilities'][attrpos];score=result['score'][attrpos]
        assert np.isfinite(proba).all() and np.allclose(proba.sum(1),1,atol=1e-6,rtol=0)
        frame=future[KEY[1:]].copy().reset_index(drop=True);frame.insert(0,'origin',o)
        for j,c in enumerate(['p0','p1','p2']):frame[c]=proba[:,j]
        frame[STANDALONE]=score;frame.to_pickle(fp)
        details.append(dict(stage=stage,origin=o,end=end,inputs=str(ip.relative_to(ROOT)),inputs_sha=sha(ip),model=str(mp.relative_to(ROOT)),model_sha=sha(mp),forecast=str(fp.relative_to(ROOT)),forecast_sha=sha(fp),unknown_factor=float(factor),worker_result=str(output.relative_to(ROOT)),worker_result_sha=sha(output),diagnostics=result['diagnostics']))
        print(f'Trial120 {stage} {o}->{end} done',flush=True);return frame
    def evaluate(stage,origins):
        ctl=pd.read_pickle(R/f'block_cycle2_public115_20261008_{stage}_predictions.pkl')[KEY+['trial115_control','blend_75_25_logodds','block_cycle2_hgb']].rename(columns=dict(zip(['trial115_control','blend_75_25_logodds','block_cycle2_hgb'],CONTROLS)))
        known=[]
        for o,end in origins.items():
            f=fit(stage,o,end,train[train.time_step.between(o+1,end)]);k=f[f.label.notna()].copy();joined=k[KEY].merge(ctl[ctl.origin==o],on=KEY,how='left',validate='one_to_one',sort=False);assert np.array_equal(joined.txId,k.txId) and not joined[list(CONTROLS)].isna().any().any()
            for c in CONTROLS:k[c]=joined[c].to_numpy()
            k[FALLBACK]=blend(k[CONTROLS[-1]],k[STANDALONE]);known.append(k)
        k=pd.concat(known,ignore_index=True);r=stage_metrics(k,stage);s=summary(r,stage);k.to_pickle(R/f'{P}_{stage}_predictions.pkl');r.to_csv(R/f'{P}_{stage}_results.csv',index=False);s.to_csv(R/f'{P}_{stage}_summary.csv',index=False);allknown.append(k);allmetrics.append(r);allsummaries.append(s);return s
    so=evaluate('original',ORIGINS);winner=None;meta=dict(trial=120,source_sha256=sha(SOURCE),plan_sha256=sha(PLAN),original_pass=bool(so.qualified.any()),confirmation_run=False,candidate_created=False,kaggle_upload=False)
    if so.qualified.any():
        sc=evaluate('confirmation',CONF);meta['confirmation_run']=True;eligible=set(so[so.qualified].variant)&set(sc[sc.qualified].variant);winner=STANDALONE if STANDALONE in eligible else FALLBACK if FALLBACK in eligible else None
        if winner:
            f=test.copy();assert 'label' not in f or f.label.isna().all();f['label']=np.nan;frame=fit('final',35,49,f);ctl=pd.read_csv(ROOT/'submissions/candidate_block_cycle_standalone117_20261008.csv');ids=sample['index'].to_numpy();assert np.array_equal(ctl['index'],ids) and np.array_equal(f['index'] if 'index' in f else f.txId,ids)
            score=frame[STANDALONE].to_numpy();score=score if winner==STANDALONE else blend(ctl.target,score);candidate=ROOT/'submissions'/f'candidate_{P}.csv';pd.DataFrame({'index':ids,'target':score}).to_csv(candidate,index=False);meta.update(candidate_created=True,candidate=str(candidate.relative_to(ROOT)),candidate_sha=sha(candidate))
    meta['winner']=winner;pd.concat(allmetrics,ignore_index=True).to_csv(R/f'{P}_results.csv',index=False);pd.concat(allsummaries,ignore_index=True).to_csv(R/f'{P}_summary.csv',index=False);(R/f'{P}_fits.json').write_text(json.dumps(details,indent=2,sort_keys=True)+'\n');(R/f'{P}_metadata.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':run()
