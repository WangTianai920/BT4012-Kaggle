"""Rebuild the two selected BT4012 submissions from raw competition CSV files."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,hashlib,json,subprocess,sys,time
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=None,help='Competition CSV folder; defaults to repository data/ or BT4012_DATA_DIR')
    ap.add_argument('--model',choices=['both','1019-6','0929-1'],default='both')
    ap.add_argument('--work-dir',type=Path,default=ROOT/'work')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    ap.add_argument('--reference-dir',type=Path,help='Optional directory containing original 1019-6.csv and 0929-1.csv')
    a=ap.parse_args(); t0=time.time()
    from final.submission_io import resolve_data_dir, repository_path, check_input_files, write_submissions, SELECTED_MODELS
    a.data_dir=resolve_data_dir(a.data_dir)
    raw_sha256=check_input_files(a.data_dir)
    a.work_dir=repository_path(a.work_dir); a.output_dir=repository_path(a.output_dir)
    if a.reference_dir is not None: a.reference_dir=repository_path(a.reference_dir)
    a.work_dir=a.work_dir.resolve();a.output_dir=a.output_dir.resolve()
    a.work_dir.mkdir(parents=True,exist_ok=True);a.output_dir.mkdir(parents=True,exist_ok=True)
    import src.competition as competition
    competition.DATA_DIR=a.data_dir.resolve()
    from final.regenerate_features import regenerate
    from final import pipeline
    regenerate(a.work_dir)
    inp=pipeline.load_inputs(a.work_dir)
    train,test,sample=inp['train'],inp['test'],inp['sample']
    results={};receipt={'raw_sha256':raw_sha256,'requested_models':list(SELECTED_MODELS) if a.model=='both' else [a.model]}
    if a.model in ('both','0929-1'):
        print('Rebuilding 0929-1 binary HGB and historical risk specialists',flush=True)
        from src.combined_challengers import load_features
        from src.horizon_hgb_20260917 import predict_stack,historical_columns,BASE
        from src.score_level_alignment_20260929 import align
        _,_,edges,_=competition.load_competition_data()
        tr,te,_,groups=load_features()
        cols=historical_columns(tr,groups['regularized_two_hop'],35)
        stack,diag=predict_stack(tr,te,edges,groups['regularized_two_hop'],cols,35,BASE)
        score,med,ref=align(stack['bagged'],te.time_step,.10)
        results['0929-1']=score
        receipt['0929-1']={'features':len(cols),'diagnostics':diag,'period_medians':med,'reference_median':ref}
    if a.model in ('both','1019-6'):
        print('Building 1019-6 graph partitions',flush=True)
        h,cols,status,w=pipeline.history_view(inp,35)
        future=test.copy();future['label']=np.nan
        hp=pipeline.partition(h,range(1,36),inp['node_map'],inp['edges'],inp['graph9'],cols,status,w)
        fp=pipeline.partition(future,range(36,50),inp['node_map'],inp['edges'],inp['graph9'],cols)
        # Fit the tree before importing Torch; mixed OpenMP runtimes can stall on macOS.
        from sklearn.ensemble import HistGradientBoostingClassifier
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
        soft_weights=hp['weights'].astype(float).copy()
        old=hp['periods']<18
        soft_weights[old]*=0.5**((18-hp['periods'][old])/3.0)
        unknown=hp['labels']==2;known=(hp['labels']==0)|(hp['labels']==1)
        soft_weights[unknown]*=(.4/.6)*soft_weights[known].sum()/max(soft_weights[unknown].sum(),1e-12)
        hp['weights']=soft_weights
        attr=hp['is_attr'];selected=(hp['labels'][attr]>=0)&(soft_weights[attr]>0)
        Xh=pd.DataFrame(hp['raw'][attr],columns=cols)
        print('Fitting soft-window HGB teacher in the tree-only process',flush=True)
        teacher=HistGradientBoostingClassifier(l2_regularization=2.0,learning_rate=.03,max_features=.8,max_iter=600,max_leaf_nodes=15,min_samples_leaf=40,early_stopping=False,random_state=42)
        teacher.fit(Xh[selected],hp['labels'][attr][selected],sample_weight=soft_weights[attr][selected])
        for part in (hp,fp):
            part['teacher_logits']=np.zeros((len(part['ids']),3),dtype=np.float32)
            probabilities=teacher.predict_proba(pd.DataFrame(part['raw'][part['is_attr']],columns=cols))
            part['teacher_logits'][part['is_attr']]=np.log(np.clip(probabilities,1e-12,1)).astype(np.float32)
        print('Soft-window teacher completed',flush=True)
        job=dict(history=hp,future=fp,columns=cols,graph_columns=pipeline.GRAPH9,origin=35,stop=49,stage='final',configuration=pipeline.gnn_config(),extra={'min_period':18,'soft_hl':3.0})
        ip=a.work_dir/'softwindow_inputs.joblib';joblib.dump(job,ip,compress=3)
        logits=[]
        for seed in [42,101,202,303,404]:
            print(f'Fitting 1019-6 seed {seed}',flush=True)
            op=a.work_dir/f'softwindow_seed{seed}.joblib'
            code="import sys,joblib;import src.lite.worker_gm_tunable_20261014 as G;G.SEED=int(sys.argv[3]);joblib.dump(G.fit(joblib.load(sys.argv[1])),sys.argv[2])"
            subprocess.run([sys.executable,'-c',code,str(ip),str(op),str(seed)],cwd=ROOT,check=True)
            result=joblib.load(op);sc=np.asarray(result['score'],float)[fp['attributed_positions']]
            logits.append(logit(np.clip(sc,1e-12,1-1e-12)))
        results['1019-6']=expit(np.mean(logits,axis=0))
        receipt['1019-6']={'seeds':[42,101,202,303,404],'features':len(cols),'min_period':18,'soft_half_life':3,'history_status_counts':np.bincount(status,minlength=3).tolist()}
    expected=set(SELECTED_MODELS) if a.model=='both' else {a.model}
    if set(results)!=expected:
        raise RuntimeError('Not all requested models produced predictions.')
    summary=write_submissions(results,sample,a.output_dir,a.reference_dir)
    for name,item in summary.items():
        receipt[name].update(item)
        print(f"{name}: {item['rows']} rows, integrity PASS -> {a.output_dir / item['file']}",flush=True)
    receipt['runtime_seconds']=time.time()-t0
    receipt['complete']=True
    (a.output_dir/'reproduction_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('Completed in',round(time.time()-t0),'seconds',flush=True)
if __name__=='__main__':main()
