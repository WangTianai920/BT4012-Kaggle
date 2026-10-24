"""Nested temporal HGB tuning and frozen fourteen-step forecast stress tests.

Run with OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m
src.horizon_hgb_20260917. Existing submissions are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from src.combined_challengers import load_features
from src.competition import PROJECT_ROOT, REPORT_DIR, RANDOM_SEED, load_competition_data
from src.targeted_hgb_challengers import _model, _variants, _weights
from src.refined_historical_neighbor_risk import _generate_block, aggregate_refined_neighbor_risk
from src.bagged_neighbor_risk_20260916 import bootstrap_positions, SEEDS

PREFIX = 'horizon_hgb_20260917'
CACHE = PROJECT_ROOT / 'models' / PREFIX
BASE = dict(max_iter=300, learning_rate=.06, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=2., max_features=1., half_life=20.)
ORIGINS = {14: 28, 21: 35, 17: 24, 24: 31, 28: 35}
SHAPES = [(15,40,2.), (7,40,2.), (7,80,5.), (15,20,.5), (15,80,5.),
          (15,120,10.), (31,40,2.), (31,80,5.), (31,120,10.)]


def inner_windows(end):
    if end < 14:
        return []
    return [(7,14)] if end == 14 else [(end-14,end),(end-7,end)]


def choose_config(end, selections):
    available = [int(k) for k in selections if int(k) <= end]
    return selections[str(max(available))]['config'].copy() if available else BASE.copy()


def weights(steps, end, half_life):
    return np.ones(len(steps)) if half_life is None else _weights(steps,end,half_life)


def model(config):
    args = {k:v for k,v in config.items() if k != 'half_life'}
    return HistGradientBoostingClassifier(**args, early_stopping=False, random_state=RANDOM_SEED)


def historical_columns(train, full_columns, end):
    """Replicate incumbent pruning, restricting feature selection to known history."""
    raw = [c for c in full_columns if c.startswith('feat_')]
    early = train.loc[train.time_step.le(min(14,end))]
    med = early.groupby('time_step')[raw].median()
    score = ((med.max()-med.min())/early[raw].std(ddof=0).replace(0,np.nan)).fillna(0)
    ranking = pd.DataFrame({'feature':raw,'score':score.to_numpy()}).sort_values(
        ['score','feature'],ascending=[False,True])
    drop = set(ranking.feature.iloc[:10])
    return [c for c in full_columns if c not in drop]


def fit_base(train, columns, end, config):
    history = train.loc[train.time_step.le(end) & train.label.notna()]
    assert history.time_step.max() <= end and history.label.nunique() == 2
    fitted = model(config)
    fitted.fit(history[columns],history.label.astype(int),
               sample_weight=weights(history.time_step,end,config['half_life']))
    return fitted


def check_probabilities(values):
    values = np.asarray(values)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Invalid probabilities')


def inner_trial(train, full_columns, start, stop, config, checkpoints):
    key = hashlib.sha256(json.dumps([start,stop,config,checkpoints],sort_keys=True).encode()).hexdigest()[:20]
    path = CACHE / ('inner_'+key+'.json')
    if path.exists():
        return json.loads(path.read_text())
    columns = historical_columns(train,full_columns,start)
    future = train.loc[train.time_step.between(start+1,stop) & train.label.notna()]
    fitted = fit_base(train,columns,start,config)
    late = future.time_step.gt(start+(stop-start)//2).to_numpy()
    output = []
    for iteration,p in enumerate(fitted.staged_predict_proba(future[columns]),1):
        if iteration not in checkpoints:
            continue
        check_probabilities(p[:,1])
        full = roc_auc_score(future.label,p[:,1])
        tail = roc_auc_score(future.loc[late,'label'],p[late,1])
        output.append(dict(iteration=iteration,full_auc=float(full),late_auc=float(tail),
                           objective=float((full+tail)/2),train_end=start,valid_end=stop,
                           feature_selection_end=min(14,start),feature_count=len(columns)))
    path.write_text(json.dumps(output,indent=2))
    return output


def tune(train, full_columns, end):
    selection_path = CACHE / f'selection_{end}.json'
    if selection_path.exists():
        return json.loads(selection_path.read_text())
    trials = []
    def evaluate(config, checkpoints, stage):
        by_iteration = {n:[] for n in checkpoints}
        for start,stop in inner_windows(end):
            for result in inner_trial(train,full_columns,start,stop,config,checkpoints):
                by_iteration[result['iteration']].append(result)
        for n,results in by_iteration.items():
            chosen = dict(config,max_iter=n)
            trials.append(dict(stage=stage,config=chosen,objective=float(np.mean([r['objective'] for r in results])),
                               inner_results=results))
    def best():
        return sorted(trials,key=lambda x:(-x['objective'],x['config']['max_leaf_nodes'],
                                          x['config']['max_iter'],json.dumps(x['config'],sort_keys=True)))[0]
    for leaves,minimum,l2 in SHAPES:
        print(f'Tune through {end}: structure {leaves}/{minimum}/{l2}',flush=True)
        evaluate(dict(BASE,max_leaf_nodes=leaves,min_samples_leaf=minimum,l2_regularization=l2),[300],'structure')
    shape = best()['config'].copy()
    for rate in (.03,.06,.08):
        print(f'Tune through {end}: learning rate {rate}',flush=True)
        evaluate(dict(shape,learning_rate=rate,max_iter=600),[150,300,450,600],'learning_rate_iterations')
    training = best()['config'].copy()
    for half in (12.,35.,None):
        print(f'Tune through {end}: half-life {half}',flush=True)
        evaluate(dict(training,half_life=half),[training['max_iter']],'recency')
    for fraction in (.8,.6):
        print(f'Tune through {end}: feature fraction {fraction}',flush=True)
        evaluate(dict(training,max_features=fraction),[training['max_iter']],'feature_subsampling')
    result = dict(end=end,config=best()['config'],objective=best()['objective'],trials=trials,
                  inner_windows=inner_windows(end),outer_labels_used=False)
    selection_path.write_text(json.dumps(result,indent=2))
    print(f'Selected through {end}: '+json.dumps(result['config']),flush=True)
    return result


def predict_stack(train, future, edges, full_columns, fixed_columns, end, config, selections=None):
    """Freeze all fitted models at end; do not retrain inside the forecast window."""
    labelled = train.loc[train.label.notna()]
    first = fit_base(train,fixed_columns,end,config)
    vf,legacy,enhanced = aggregate_refined_neighbor_risk(future,edges,first.predict_proba(future[fixed_columns])[:,1])
    if selections is None:
        meta,lc,ec = _generate_block(labelled,train,edges,fixed_columns,20.,end-6,end)
        seed_config = BASE.copy()
    else:
        seed_end = end-7
        seed_config = choose_config(seed_end,selections)
        seed_cols = historical_columns(train,full_columns,seed_end)
        seed = fit_base(train,seed_cols,seed_end,seed_config)
        block = train.loc[train.time_step.between(end-6,end)]
        mf,lc,ec = aggregate_refined_neighbor_risk(block,edges,seed.predict_proba(block[seed_cols])[:,1])
        meta = block.loc[block.label.notna(),['txId','time_step','label']].merge(
            mf,on=['txId','time_step'],validate='one_to_one',sort=False)
    assert legacy == lc and enhanced == ec
    singles = {}
    for kind,cols in [('original',legacy),('enhanced',legacy+enhanced)]:
        specialist = _model()
        specialist.fit(meta[cols],meta.label.astype(int),sample_weight=_weights(meta.time_step,end,20.))
        singles[kind] = specialist.predict_proba(vf[cols])[:,1]
    simple = .9*vf.hist_self_risk.to_numpy()+.1*singles['original']
    anchor = .975*simple+.025*singles['enhanced']
    members = []
    for seed in SEEDS:
        sampled = meta.iloc[bootstrap_positions(meta,seed)]
        specialist = _model()
        specialist.fit(sampled[legacy+enhanced],sampled.label.astype(int),
                       sample_weight=_weights(sampled.time_step,end,20.))
        members.append(specialist.predict_proba(vf[legacy+enhanced])[:,1])
    return dict(base=vf.hist_self_risk.to_numpy(),simple=simple,
                bagged=.975*anchor+.025*np.mean(members,axis=0)), dict(
                    first_stage_config=config,meta_seed_end=end-7,meta_seed_config=seed_config,
                    specialist_train_start=end-6,specialist_train_end=end)


def metrics(predictions):
    output = []
    variants = [c for c in predictions if c not in ['origin','txId','time_step','label']]
    for end,g in predictions.groupby('origin',sort=False):
        slices = {'first7':g.loc[g.time_step.le(end+7)]}
        if g.time_step.max() == end+14:
            slices.update(full14=g,second7=g.loc[g.time_step.gt(end+7)])
        for part,rows in slices.items():
            for variant in variants:
                if rows[variant].isna().any():
                    continue
                output.append(dict(origin=int(end),slice=part,variant=variant,n=len(rows),
                                   positives=int(rows.label.sum()),auc=float(roc_auc_score(rows.label,rows[variant])),
                                   ap=float(average_precision_score(rows.label,rows[variant]))))
    return pd.DataFrame(output)


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    plan = dict(experiment=PREFIX,outer_origins=ORIGINS,selection_origins=[14,21,28],
        inner_windows={n:inner_windows(n) for n in [14,21,28]},base=BASE,shapes=SHAPES,
        rate_grid=[.03,.06,.08],iteration_grid=[150,300,450,600],half_life_grid=[12,20,35,None],
        feature_fraction_grid=[1,.8,.6],selection_objective='mean over inner windows of (full AUC + last-half AUC)/2',
        procedure='structure, then learning rate/iterations, then separate recency and feature-subset refinements; retain earlier best',
        reused_historical_periods=True,untouched_holdout=False,step24_selection_target=False,
        specialist_rebuild_gate='tuned base improves mean full14 and mean second7 vs base; no horizon slice loses more than .002',
        final_gate='beat both controls on mean full14 and second7; primary and rolling seven-step means do not fall; no measured slice loses more than .001',
        test_labels_used=False,kaggle_upload=False)
    plan_path = REPORT_DIR / (PREFIX+'_plan.json')
    serialized = json.dumps(plan,indent=2,sort_keys=True)
    if plan_path.exists() and plan_path.read_text() != serialized:
        raise RuntimeError('Plan changed; use a new experiment prefix')
    plan_path.write_text(serialized)
    sources = sorted((PROJECT_ROOT/'src').glob('*.py'))
    inputs = sources+[PROJECT_ROOT/'data/train.csv',PROJECT_ROOT/'data/test.csv',PROJECT_ROOT/'data/txs_edgelist.csv',
                      REPORT_DIR/'time_relative_early_drift_ranking.csv',plan_path]
    identity = {str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    identity['sklearn_version'] = sklearn.__version__
    identity_path = CACHE/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError('Cached source/data identity changed')
    identity_path.write_text(json.dumps(identity,indent=2))
    train,test,sample,groups = load_features()
    _,_,edges,_ = load_competition_data()
    full_columns = groups['regularized_two_hop']
    columns,_ = _variants(groups)['regularized_drop_top10_drift']
    assert historical_columns(train,full_columns,14) == columns
    saved = pd.read_csv(REPORT_DIR/'development_20260917_oof_predictions.csv')
    fold_names = {14:'fold_0',21:'fold_1',28:'fold_2',17:'rolling_17',24:'rolling_24'}
    outputs,diagnostics = [],{}
    # Complete Step 1 before looking at any tuning results.
    for end,stop in ORIGINS.items():
        path = CACHE/f'control_{end}.joblib'
        future = train.loc[train.time_step.between(end+1,stop)].copy()
        if path.exists():
            result,diag = joblib.load(path)
        else:
            print(f'Frozen controls through {end}, forecasting {end+1}-{stop}',flush=True)
            pred,diag = predict_stack(train,future,edges,full_columns,columns,end,BASE)
            result = future.loc[future.label.notna(),['txId','time_step','label']].copy()
            known = future.label.notna().to_numpy()
            for key,name in [('base','base_0915_1'),('simple','0915-4'),('bagged','0916-2')]:
                result[name] = pred[key][known]
            result.insert(0,'origin',end)
            joblib.dump((result,diag),path)
        previous = saved.loc[saved.fold.eq(fold_names[end])]
        joined = previous.merge(result,on=['txId','time_step','label'],validate='one_to_one',suffixes=('_old','_new'))
        assert len(joined)==len(previous)
        for name in ['0915-4','0916-2']:
            assert np.allclose(joined[name+'_old'],joined[name+'_new'],rtol=0,atol=1e-10)
        outputs.append(result);diagnostics[str(end)] = diag
        metrics(pd.concat(outputs)).to_csv(REPORT_DIR/(PREFIX+'_control_results.csv'),index=False)
    predictions = pd.concat(outputs,ignore_index=True)
    predictions.to_csv(REPORT_DIR/(PREFIX+'_predictions.csv'),index=False)
    selections = {}
    for end in (14,21,28):
        selections[str(end)] = tune(train,full_columns,end)
        (REPORT_DIR/(PREFIX+'_selections.json')).write_text(json.dumps(selections,indent=2))
    for end,stop in ORIGINS.items():
        config = choose_config(end,selections)
        part = predictions.origin.eq(end)
        rows = train.set_index('txId').loc[predictions.loc[part,'txId']]
        fitted = fit_base(train,columns,end,config)
        predictions.loc[part,'tuned_base'] = fitted.predict_proba(rows[columns])[:,1]
    results = metrics(predictions)
    horizon = results.loc[results.origin.isin([14,21])].pivot(index=['origin','slice'],columns='variant',values='auc')
    d = horizon.tuned_base-horizon.base_0915_1
    rebuild = bool(d.xs('full14',level='slice').mean()>0 and d.xs('second7',level='slice').mean()>0 and d.min()>=-.002)
    if rebuild:
        print('Base passed predeclared horizon gate; rebuilding historical specialists',flush=True)
        for end,stop in ORIGINS.items():
            path = CACHE/f'tuned_stack_{end}.joblib'
            future = train.loc[train.time_step.between(end+1,stop)]
            if path.exists():
                pred,diag = joblib.load(path)
            else:
                pred,diag = predict_stack(train,future,edges,full_columns,columns,end,choose_config(end,selections),selections)
                joblib.dump((pred,diag),path)
            known = future.label.notna().to_numpy(); part = predictions.origin.eq(end)
            for key,name in [('simple','tuned_simple'),('bagged','tuned_bagged')]:
                predictions.loc[part,name] = pred[key][known]
            diagnostics[str(end)]['tuned'] = diag
    predictions.to_csv(REPORT_DIR/(PREFIX+'_predictions.csv'),index=False)
    results = metrics(predictions)
    results.to_csv(REPORT_DIR/(PREFIX+'_results.csv'),index=False)
    summary = []
    pivot = results.pivot(index=['origin','slice'],columns='variant',values='auc')
    for name in pivot:
        row = dict(variant=name,full14_mean=float(pivot.loc[[(14,'full14'),(21,'full14')],name].mean()),
                   second7_mean=float(pivot.loc[[(14,'second7'),(21,'second7')],name].mean()),
                   primary7_mean=float(pivot.loc[[(e,'first7') for e in [14,21,28]],name].mean()),
                   rolling7_mean=float(pivot.loc[[(e,'first7') for e in [17,24]],name].mean()))
        passed = name.startswith('tuned')
        for control in ['0915-4','0916-2']:
            delta = pivot[name]-pivot[control]
            full = float(delta.loc[[(14,'full14'),(21,'full14')]].mean())
            late = float(delta.loc[[(14,'second7'),(21,'second7')]].mean())
            primary = float(delta.loc[[(e,'first7') for e in [14,21,28]]].mean())
            rolling = float(delta.loc[[(e,'first7') for e in [17,24]]].mean())
            row.update({f'full14_delta_{control}':full,f'second7_delta_{control}':late,
                        f'primary7_delta_{control}':primary,f'rolling7_delta_{control}':rolling,
                        f'max_slice_loss_{control}':float(max(0,-delta.min()))})
            passed &= full>0 and late>0 and primary>=0 and rolling>=0 and delta.min()>=-.001
        row['qualified'] = bool(passed)
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(REPORT_DIR/(PREFIX+'_summary.csv'),index=False)
    metadata = dict(plan=plan,identity=identity,selections=selections,diagnostics=diagnostics,
                    base_rebuild_gate_passed=rebuild,eligible=summary.loc[summary.qualified,'variant'].tolist(),
                    candidate_created=False,kaggle_upload_performed=False)
    (REPORT_DIR/(PREFIX+'_metadata.json')).write_text(json.dumps(metadata,indent=2))
    print(summary.to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
