"""Independent Trial118 all-node core, saved-model, control, and gate audit.

Importing this module does not load competition data. --pure-toy uses only
synthetic graphs. The full audit requires the root's frozen authorization.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import deque
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.combined_challengers import load_features
from src.competition import PROJECT_ROOT, REPORT_DIR, load_competition_data
from src.verify_articulation_cut2_public99_20261008 import hard_status_weights
from src.verify_block_cycle_standalone117_20261008 import verify_all as replay_trial117
from src.horizon_hgb_20260917 import historical_columns

ROOT, REPORTS=PROJECT_ROOT, REPORT_DIR
P="recursive_core5_trial118_20261008"
SOURCE=ROOT/"src"/"recursive_core5_trial118_runner_20261008.py"
FEATURE_SOURCE=ROOT/"src"/f"{P}.py"
PLAN=REPORTS/f"{P}_plan.json"
AUTH=REPORTS/f"{P}_root_authorization.json"
META=REPORTS/f"{P}_metadata.json"
FLAT=REPORTS/f"{P}_fit_diagnostics.json"
AUDIT=REPORTS/f"{P}_parent_verification.json"
GRAPH=REPORTS/f"{P}_graph_features.pkl"
NODE_MAP=REPORTS/f"{P}_graph_node_map.pkl"
GRAPH_META=REPORTS/f"{P}_graph_metadata.json"
EDGE_CATALOG=REPORTS/"terminal_coverage4_public99_20261008_deduplicated_edges.pkl"
PRIOR_MAP=REPORTS/"terminal_coverage4_public99_20261008_graph_node_map.pkl"
GRAPH5_CACHE=REPORTS/"directed_chain_position5_public55_20261005_graph_features.pkl"
CUT2_CACHE=REPORTS/"articulation_cut2_public99_20261008_graph_features.pkl"
BLOCK2_CACHE=REPORTS/"block_cycle2_public115_20261008_graph_features.pkl"
PREV116="block_cycle2_public115_20261008"
PREV117="block_cycle_standalone117_20261008"
PREV117_AUDIT=REPORTS/f"{PREV117}_audit.json"
PREV117_CANDIDATE=ROOT/"submissions"/f"candidate_{PREV117}.csv"
LOG=ROOT/"experiment_log.csv"
GRAPH5=["dag_log1p_scc_size","dag_log1p_source_depth","dag_log1p_sink_depth",
        "dag_log1p_chain_span","dag_chain_position"]
CUT2=["graph_cut_pair_disconnect_fraction","graph_cut_dominant_remainder_loss"]
BLOCK2=["graph_block_log1p_largest_cycle_excess","graph_block_cycle_membership_count"]
CORE5=["graph_core_number","graph_core_log1p_peel_wave",
       "graph_core_higher_neighbor_fraction","graph_core_same_neighbor_fraction",
       "graph_core_log1p_shell_size"]
KEY=["origin","txId","time_step","label"]
PUBLIC115="trial115_selected_blend"
BLEND116="trial116_selected_blend"
RAW117="trial117_standalone"
CONTROLS=(PUBLIC115,BLEND116,RAW117)
STANDALONE="recursive_core5_hgb"
FALLBACK="blend_75_25_trial117_trial118"
ORIGINS={14:28,21:35,17:24,24:31,28:35}
CONFIRMATION={17:31,18:32,20:34}
SOURCE_SHA="abef524e06eb1997781fcb7ccd71b925f9b927c908e26f6412f9f51ba771f3ca"
FEATURE_SOURCE_SHA="590b1b1f17822e3620d30af94d987718088c7a99abf3f10ad50cd6460efe332d"
PLAN_SHA="b379ddb070597c574dbcab31219e77521e164b27eaa021011318e80fe637d54b"
PREFLIGHT=15808


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prefix_sha(path:Path,count:int)->str:
    raw=path.read_bytes()
    with io.StringIO(raw.decode("utf-8-sig"),newline="") as stream:
        reader=csv.reader(stream)
        next(reader)
        for _ in range(count):next(reader)
        physical_lines=reader.line_num
    return hashlib.sha256(b"".join(raw.splitlines(keepends=True)[:physical_lines])).hexdigest()


def row_sha(row:dict)->str:
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def fixed_blend(control,challenger):
    a,b=np.asarray(control,float),np.asarray(challenger,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError("Trial118 blend inputs invalid")
    return expit(.75*logit(np.clip(a,1e-12,1-1e-12))+
                 .25*logit(np.clip(b,1e-12,1-1e-12)))


def reference_core(ids,edge_pairs,components):
    """NetworkX Batagelj-Zaversnik core plus separate synchronous shell waves."""
    graph=nx.Graph()
    graph.add_nodes_from(int(v) for v in ids)
    graph.add_edges_from((int(a),int(b)) for a,b in edge_pairs)
    if set(graph)!=set(map(int,ids)):
        raise AssertionError("Reference graph endpoint outside all-node universe")
    core_by_id=nx.core_number(graph)
    ordered=np.asarray(ids,np.int64)
    core=np.asarray([core_by_id[int(v)] for v in ordered],np.int32)
    wave=np.full(len(ids),-1,np.int32)
    id_to_pos={int(v):i for i,v in enumerate(ordered)}
    adjacency=[[id_to_pos[int(u)] for u in graph.neighbors(int(v))] for v in ordered]
    # At shell k, the active graph contains exactly vertices with core >= k.
    # Recompute its initial degrees, then remove only the k shell in rounds.
    for k in np.unique(core):
        active=core>=k
        degree=np.asarray([sum(active[u] for u in neighbors) if active[i] else 0
                           for i,neighbors in enumerate(adjacency)],np.int32)
        shell=np.flatnonzero(core==k)
        remaining=np.zeros(len(ids),bool);remaining[shell]=True
        frontier=deque(int(v) for v in shell if degree[v]<=k)
        layer=0
        while remaining[shell].any():
            if not frontier:
                raise AssertionError("Independent synchronous core shell stalled")
            current=list(frontier);frontier.clear()
            for v in current:
                if not remaining[v]:continue
                remaining[v]=False;active[v]=False;wave[v]=layer
            for v in current:
                for u in adjacency[v]:
                    if active[u]:
                        degree[u]-=1
            next_frontier=[int(v) for v in shell if remaining[v] and degree[v]<=k]
            frontier.extend(next_frontier)
            layer+=1
    if (wave<0).any():raise AssertionError("Independent peel waves incomplete")
    degree=np.asarray([len(neighbors) for neighbors in adjacency],np.int32)
    higher=np.asarray([sum(core[u]>core[v] for u in neighbors)
                       for v,neighbors in enumerate(adjacency)],np.int32)
    same=np.asarray([sum(core[u]==core[v] for u in neighbors)
                     for v,neighbors in enumerate(adjacency)],np.int32)
    component=np.asarray(components,np.int32)
    shell_size=np.empty(len(ids),np.int32)
    for c in np.unique(component):
        members=np.flatnonzero(component==c)
        _,inverse,counts=np.unique(core[members],return_inverse=True,return_counts=True)
        shell_size[members]=counts[inverse]
    values=np.column_stack((core,np.log1p(wave),
                            np.divide(higher,degree,out=np.zeros(len(ids),float),where=degree>0),
                            np.divide(same,degree,out=np.zeros(len(ids),float),where=degree>0),
                            np.log1p(shell_size))).astype(np.float32)
    return graph,values,core,wave,shell_size


def pure_toy():
    from src.recursive_core5_trial118_20261008 import build_recursive_core_features
    cases=[nx.empty_graph(3),nx.path_graph(9),nx.cycle_graph(9),nx.complete_graph(7),
           nx.barbell_graph(5,3),nx.lollipop_graph(6,4)]
    cases += [nx.gnp_random_graph(18,.18,seed=i) for i in range(12)]
    cases.append(nx.path_graph(1201))
    for graph in cases:
        ids=np.asarray(sorted(graph),np.int64)
        components=sorted(nx.connected_components(graph),key=lambda c:min(c))
        comp_by_id={v:i for i,part in enumerate(components) for v in part}
        prior=pd.DataFrame({"txId":ids,
                            "weak_component":[comp_by_id[int(v)] for v in ids],
                            "period":[comp_by_id[int(v)]+1 for v in ids]})
        edges=pd.DataFrame(sorted((min(a,b),max(a,b)) for a,b in graph.edges()),
                           columns=["txId1","txId2"],dtype=np.int64)
        saved,node_map,meta=build_recursive_core_features(prior,edges)
        _,values,core,wave,shell=reference_core(ids,graph.edges(),prior.weak_component)
        assert np.array_equal(saved[CORE5].to_numpy(),values)
        assert np.array_equal(node_map.core_number,core)
        assert np.array_equal(node_map.core_peel_wave,wave)
        assert np.array_equal(node_map.core_shell_size,shell)
        assert meta["graph_node_count"]==len(ids)
    print("Trial118 independent NetworkX core and shell-wave toys passed")


def independent_metrics(frame,stage):
    rows=[]
    for origin,part in frame.groupby('origin',sort=False):
        slices={'first7':part[part.time_step<=origin+7]}
        if part.time_step.max()==origin+14:
            slices.update(full14=part,second7=part[part.time_step>origin+7])
        for name,q in slices.items():
            for variant in (*CONTROLS,STANDALONE,FALLBACK):
                rows.append(dict(stage=stage,origin=int(origin),slice=name,variant=variant,n=len(q),positives=int(q.label.sum()),auc=roc_auc_score(q.label,q[variant]),average_precision=average_precision_score(q.label,q[variant])))
    return pd.DataFrame(rows)


def independent_summary(results,stage):
    pivot=results.pivot(index=['origin','slice'],columns='variant',values='auc')
    groups=({'full14':[(14,'full14'),(21,'full14')],'second7':[(14,'second7'),(21,'second7')],'primary7':[(o,'first7') for o in (14,21,28)],'rolling7':[(o,'first7') for o in (17,24)]} if stage=='original' else {name:[(o,name) for o in CONFIRMATION] for name in ('full14','second7','first7')})
    rows=[]
    for v in (STANDALONE,FALLBACK):
        r={'stage':stage,'variant':v};valid=True
        for c in CONTROLS:
            for name,keys in groups.items():
                r[f'{c}_{name}_mean_auc']=float(pivot.loc[keys,v].mean())
                r[f'{c}_{name}_mean_delta']=float((pivot.loc[keys,v]-pivot.loc[keys,c]).mean())
            r[f'{c}_worst_slice_delta']=float((pivot[v]-pivot[c]).min())
            full=r[f'{c}_full14_mean_delta'];late=r[f'{c}_second7_mean_delta']
            gate=(full>=.0005 and late>=.001) if c==RAW117 else (full>0 and late>0)
            gate=gate and (r[f'{c}_primary7_mean_delta']>0 and r[f'{c}_rolling7_mean_delta']>0 if stage=='original' else r[f'{c}_first7_mean_delta']>=0)
            valid=valid and gate and r[f'{c}_worst_slice_delta']>=-.001
        r['qualified']=bool(valid);rows.append(r)
    return pd.DataFrame(rows)


def verify_all():
    csv.field_size_limit(20000000)
    plan=json.loads(PLAN.read_text());auth=json.loads(AUTH.read_text());meta=json.loads(META.read_text());details=json.loads(FLAT.read_text())
    assert sha(SOURCE)==SOURCE_SHA and sha(FEATURE_SOURCE)==FEATURE_SOURCE_SHA and sha(PLAN)==PLAN_SHA
    assert auth['authorized'] and auth['source_sha256']==SOURCE_SHA and auth['feature_source_sha256']==FEATURE_SOURCE_SHA and auth['plan_sha256']==PLAN_SHA and auth['registry_rows']==PREFLIGHT
    for flag in ('trial117_registered_and_closed','root_trial118_core_model_and_material_gates_reviewed','no_test_labels_or_external_data'):assert auth[flag] is True
    for rel,h in plan['input_sha256'].items():assert (prefix_sha(LOG,PREFLIGHT) if rel=='experiment_log.csv' else sha(ROOT/rel))==h,rel
    rows=list(csv.DictReader(LOG.open(newline='')));assert len(rows)==len({r['experiment_id'] for r in rows}) and len(rows)>=PREFLIGHT
    byid={r['experiment_id']:r for r in rows}
    for ident,h in plan['trial117_exact_rows'].items():assert row_sha(byid[ident])==h
    before=sha(PREV117_AUDIT);parent=replay_trial117();assert sha(PREV117_AUDIT)==before
    assert parent['parent_saved_three_status_model_replays']==9 and parent['parent_trial115_control_model_replays']==9 and parent['parent_public99_control_member_replays']==27
    train,test,sample,groups=load_features();_,_,official_edges,_=load_competition_data()
    catalog=pd.read_pickle(EDGE_CATALOG);prior=pd.read_pickle(PRIOR_MAP);ids=prior.txId.to_numpy(np.int64)
    canonical=official_edges[['txId1','txId2']].astype(np.int64).drop_duplicates();canonical=canonical[canonical.txId1!=canonical.txId2].sort_values(['txId1','txId2'],kind='stable').reset_index(drop=True)
    pd.testing.assert_frame_equal(catalog,canonical,check_exact=True)
    assert len(ids)==203769 and len(train)+len(test)==157101
    graph,values,core,wave,shell=reference_core(ids,catalog.to_numpy(),prior.weak_component)
    comps=list(nx.connected_components(graph));assert len(comps)==49 and max(map(len,comps))==7880
    for comp in comps:
        pos=np.searchsorted(ids,np.fromiter(comp,np.int64));assert len(np.unique(prior.period.to_numpy()[pos]))==len(np.unique(prior.weak_component.to_numpy()[pos]))==1
    saved=pd.read_pickle(GRAPH);mapped=pd.read_pickle(NODE_MAP);gm=json.loads(GRAPH_META.read_text())
    assert saved.columns.tolist()==['txId',*CORE5] and np.array_equal(saved.txId,ids) and np.array_equal(saved[CORE5],values)
    pd.testing.assert_frame_equal(mapped[prior.columns],prior,check_exact=True)
    for col,x in [('core_number',core),('core_peel_wave',wave),('core_shell_size',shell)]:assert np.array_equal(mapped[col],x)
    assert np.array_equal(mapped[CORE5],values)
    for period,part in mapped.groupby('period',sort=True):
        d=gm['per_period_feature_diagnostics'][str(int(period))];assert d['node_count']==len(part)
        for name in CORE5:
            x=part[name].to_numpy(float);q=np.quantile(x,[0,.05,.5,.95,.99,1]);expect=dict(zip(['min','p05','median','p95','p99','max'],map(float,q)));expect['nonzero_count']=int((x>0).sum());assert d[name]==expect
    assert meta['graph_metadata_sha256']==sha(GRAPH_META)
    for name,path in [('source_sha256',SOURCE),('feature_source_sha256',FEATURE_SOURCE),('plan_sha256',PLAN),('graph_features_sha256',GRAPH),('graph_node_map_sha256',NODE_MAP),('deduplicated_edges_sha256',EDGE_CATALOG)]:assert gm[name]==sha(path)
    enriched=[]
    for frame in (train,test):
        old=frame.txId.to_numpy();f=frame
        for path in (GRAPH5_CACHE,CUT2_CACHE,BLOCK2_CACHE,GRAPH):f=f.merge(pd.read_pickle(path),on='txId',how='left',validate='one_to_one',sort=False)
        assert np.array_equal(f.txId,old);enriched.append(f)
    train,test=enriched;full=list(groups['regularized_two_hop'])+GRAPH5
    stages={'original':ORIGINS}
    if meta['confirmation_run']:stages['confirmation']=CONFIRMATION
    folds={(d['stage'],d['origin'],d['forecast_end']):d for d in details};expected={(s,o,e) for s,os in stages.items() for o,e in os.items()}
    if meta['candidate_created']:expected.add(('final',35,49))
    assert set(folds)==expected and len(folds)==len(details)
    results=[];summaries=[];predictions=[];replays=0
    def replay_fold(stage,o,end):
        nonlocal replays
        h=train[train.time_step<=o];f=test.copy() if stage=='final' else train[train.time_step.between(o+1,end)].copy()
        if stage=='final':f['label']=np.nan
        cols=historical_columns(train,full,o)+CUT2+BLOCK2+CORE5;assert len(cols)==186
        status=np.where(h.label.isna(),2,h.label.fillna(2)).astype(np.int64);weights,factor=hard_status_weights(h.time_step,status,o)
        d=folds[(stage,o,end)]
        for hashkey,pathkey in [('model_sha256','model_file'),('model_inputs_sha256','model_inputs_file'),('forecast_sha256','forecast_file')]:assert d[hashkey]==sha(ROOT/d[pathkey])
        b=joblib.load(ROOT/d['model_file']);i=joblib.load(ROOT/d['model_inputs_file']);model=b['model']
        assert b['configuration']==plan['model_config'] and b['columns']==i['columns']==cols
        assert model.n_iter_==600 and np.array_equal(model.classes_,[0,1,2]) and all(model.get_params()[k]==v for k,v in plan['model_config'].items())
        for snapshot in (b,i):
            assert snapshot['source_sha256']==SOURCE_SHA and snapshot['feature_source_sha256']==FEATURE_SOURCE_SHA and snapshot['plan_sha256']==PLAN_SHA
        for key,x in [('history_ids',h.txId),('history_times',h.time_step),('forecast_ids',f.txId),('forecast_times',f.time_step),('status',status),('raw_weights',weights)]:assert np.array_equal(i[key],x),key
        assert np.array_equal(i['history_labels'],h.label,equal_nan=True)
        pd.testing.assert_frame_equal(i['history_features'],h[cols],check_exact=True);pd.testing.assert_frame_equal(i['forecast_features'],f[cols],check_exact=True)
        assert np.isclose(weights[status==2].sum()/weights.sum(),.4,atol=1e-12,rtol=0)
        assert b['unknown_weight_factor']==factor and b['status_counts']==np.bincount(status,minlength=3).tolist() and b['status_weight_mass']==np.bincount(status,weights=weights,minlength=3).tolist()
        p=model.predict_proba(f[cols]);den=p[:,0]+p[:,1];score=np.divide(p[:,1],den,out=np.zeros(len(p)),where=den>0)
        out=f[['txId','time_step','label']].copy().reset_index(drop=True);out.insert(0,'origin',o)
        for j,name in enumerate(('p_known_licit','p_known_illicit','p_unknown')):out[name]=p[:,j]
        out['p_known_denominator']=den;out[STANDALONE]=score
        pd.testing.assert_frame_equal(out,pd.read_pickle(ROOT/d['forecast_file']),check_exact=False,rtol=0,atol=1e-12);replays+=1
        return out
    for stage,os in stages.items():
        ctl=pd.read_pickle(REPORTS/f'{PREV116}_{stage}_predictions.pkl')[KEY+['trial115_control','blend_75_25_logodds','block_cycle2_hgb']].rename(columns={'trial115_control':PUBLIC115,'blend_75_25_logodds':BLEND116,'block_cycle2_hgb':RAW117})
        allforecasts=[];known=[]
        for o,e in os.items():
            out=replay_fold(stage,o,e);allforecasts.append(out);k=out[out.label.notna()].copy();aligned=k[KEY].merge(ctl[ctl.origin==o],on=KEY,how='left',validate='one_to_one',sort=False)
            assert np.array_equal(aligned.txId,k.txId) and not aligned[list(CONTROLS)].isna().any().any()
            for c in CONTROLS:k[c]=aligned[c].to_numpy()
            k[FALLBACK]=fixed_blend(k[RAW117],k[STANDALONE]);known.append(k)
        forecast=pd.concat(allforecasts,ignore_index=True);k=pd.concat(known,ignore_index=True)
        pd.testing.assert_frame_equal(forecast,pd.read_pickle(REPORTS/f'{P}_{stage}_forecast_probabilities.pkl'),check_exact=False,rtol=0,atol=1e-12)
        pd.testing.assert_frame_equal(k,pd.read_pickle(REPORTS/f'{P}_{stage}_predictions.pkl'),check_exact=False,rtol=0,atol=1e-12)
        r=independent_metrics(k,stage);s=independent_summary(r,stage)
        for suffix,x in [('results',r),('summary',s)]:pd.testing.assert_frame_equal(pd.read_csv(REPORTS/f'{P}_{stage}_{suffix}.csv'),x,check_dtype=False,check_exact=False,rtol=0,atol=1e-12)
        assert meta[f'{stage}_gate_passed']==bool(s.qualified.any());results.append(r);summaries.append(s);predictions.append(k)
    eligible=sorted(set(summaries[0].loc[summaries[0].qualified,'variant']) & set(summaries[1].loc[summaries[1].qualified,'variant'])) if len(summaries)==2 else []
    winner=STANDALONE if STANDALONE in eligible else FALLBACK if FALLBACK in eligible else None
    assert meta['winner']==winner and meta['two_stage_qualified_variants']==eligible and meta['candidate_created']==bool(winner) and meta['same_variant_passed_both_stages']==bool(winner)
    for suffix,x in [('results.csv',pd.concat(results,ignore_index=True)),('summary.csv',pd.concat(summaries,ignore_index=True))]:pd.testing.assert_frame_equal(pd.read_csv(REPORTS/f'{P}_{suffix}'),x,check_dtype=False,check_exact=False,rtol=0,atol=1e-12)
    pd.testing.assert_frame_equal(pd.read_pickle(REPORTS/f'{P}_predictions.pkl'),pd.concat(predictions,ignore_index=True),check_exact=False,rtol=0,atol=1e-12)
    final={'candidate_created':False}
    if winner:
        out=replay_fold('final',35,49);candidate=pd.read_csv(ROOT/meta['candidate_file']);ids=sample['index'].to_numpy();ctl=pd.read_csv(PREV117_CANDIDATE)
        assert candidate.columns.tolist()==['index','target'] and len(candidate)==15329 and np.array_equal(candidate['index'],ids) and np.array_equal(ctl['index'],ids)
        scores=out[STANDALONE].to_numpy();scores=scores if winner==STANDALONE else fixed_blend(ctl.target,scores)
        assert np.isfinite(candidate.target).all() and candidate.target.between(0,1).all() and np.allclose(candidate.target,scores,atol=1e-12,rtol=0)
        assert sha(ROOT/meta['candidate_file'])==meta['candidate_sha256'];final={'candidate_created':True,'candidate_rows':len(candidate),'candidate_sha256':meta['candidate_sha256']}
    else:assert not (ROOT/f'submissions/candidate_{P}.csv').exists()
    audit={'trial_number':118,'variant':P,'source_sha256':SOURCE_SHA,'feature_source_sha256':FEATURE_SOURCE_SHA,'plan_sha256':PLAN_SHA,'pinned_input_count':len(plan['input_sha256']),'prior117_rows_verified':len(plan['trial117_exact_rows']),'all_node_core_features_verified':True,'raw_weights_cutoffs_and_inputs_verified':True,'control_ancestry_verified':True,'saved_model_replays':replays,'metric_rows_recomputed':sum(map(len,results)),'eligible_variants':eligible,'strict_material_gates_verified':True,'final_artifact_checks':final,'kaggle_upload':False,'independent_holdout_claim':False}
    AUDIT.write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n');return audit


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--pure-toy']:pure_toy()
    else:print(json.dumps(verify_all(),indent=2,sort_keys=True))
