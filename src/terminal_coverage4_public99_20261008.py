"""Trial112: distinct reachable DAG source/sink terminal coverage plus HGB.

Graph construction is label-free but reads competition edges only after exact
root authorization and input pin checks. Each weak component is period-pure.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,roc_auc_score

from src.combined_challengers import load_features
from src.competition import PROJECT_ROOT,REPORT_DIR,load_competition_data
from src.horizon_hgb_20260917 import historical_columns
from src.label_status_multiclass_20260929 import label_status
from src.status_mass40_lr03_iter600_20261003 import training_weights_40
from src.feature_propagation16_public99_20261007 import attach_graph5,load_public99_control
from src.missing_aware_directed_gnn_public90_20261006 import weak_component_period_map

PREFIX='terminal_coverage4_public99_20261008'
PLAN_PATH=REPORT_DIR/f'{PREFIX}_plan.json'
AUTH_PATH=REPORT_DIR/f'{PREFIX}_root_authorization.json'
SOURCE_PATH=PROJECT_ROOT/'src'/f'{PREFIX}.py'
MODEL_DIR=PROJECT_ROOT/'models'/PREFIX
GRAPH5_PATH=REPORT_DIR/'directed_chain_position5_public55_20261005_graph_features.pkl'
GRAPH_FEATURES_PATH=REPORT_DIR/f'{PREFIX}_graph_features.pkl'
GRAPH_NODE_MAP_PATH=REPORT_DIR/f'{PREFIX}_graph_node_map.pkl'
GRAPH_EDGES_PATH=REPORT_DIR/f'{PREFIX}_deduplicated_edges.pkl'
GRAPH_METADATA_PATH=REPORT_DIR/f'{PREFIX}_graph_metadata.json'
PUBLIC99_CANDIDATE=PROJECT_ROOT/'submissions/candidate_chain_position_seedensemble3_public90_20261007.csv'
ORIGINS={14:28,21:35,17:24,24:31,28:35}
CONFIRMATION={17:31,18:32,20:34}
KEY=['origin','txId','time_step','label']
GRAPH5=['dag_log1p_scc_size','dag_log1p_source_depth','dag_log1p_sink_depth',
        'dag_log1p_chain_span','dag_chain_position']
TERMINAL4=['dagterminal_log1p_sources','dagterminal_log1p_sinks',
        'dagterminal_source_fraction','dagterminal_sink_fraction']
STANDALONE='terminal_coverage4_hgb'
BLEND='blend_75_25_logodds'
VARIANTS=(STANDALONE,BLEND)
EPS=1e-12
GRAPH_NODE_COUNT=203769
ATTRIBUTED_NODE_COUNT=157101
GRAPH_ONLY_NODE_COUNT=46668
WEAK_COMPONENT_COUNT=49
MAX_COMPONENT_NODES=7880


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_config():
    return dict(max_leaf_nodes=15,min_samples_leaf=40,learning_rate=.03,
                max_iter=600,l2_regularization=2.,max_features=.8,
                random_state=42,early_stopping=False)


def verify_frozen_inputs(plan):
    if (plan.get('trial_number')!=112 or plan.get('variant')!=PREFIX
            or plan.get('source_sha256')!=sha(SOURCE_PATH)
            or plan.get('model_config')!=model_config()
            or plan.get('graph_feature_columns')!=TERMINAL4):
        raise RuntimeError('Trial112 source or frozen configuration changed')
    auth=json.loads(AUTH_PATH.read_text())
    if (auth.get('authorized') is not True or auth.get('source_sha256')!=sha(SOURCE_PATH)
            or auth.get('plan_sha256')!=sha(PLAN_PATH)
            or auth.get('registry_rows')!=plan['registry_preflight']['unique_rows']):
        raise RuntimeError('Exact root authorization is required before data load')
    for rel,digest in plan['input_sha256'].items():
        if sha(PROJECT_ROOT/rel)!=digest:raise RuntimeError(f'Frozen input changed: {rel}')
    csv.field_size_limit(10_000_000)
    with (PROJECT_ROOT/'experiment_log.csv').open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    ids=[r['experiment_id'] for r in rows]
    if len(rows)!=plan['registry_preflight']['unique_rows'] or len(ids)!=len(set(ids)):
        raise RuntimeError('Experiment register unique-ID preflight changed')
    by_id={r['experiment_id']:r for r in rows}
    for experiment_id,expected in plan['trial111_exact_rows'].items():
        row=by_id.get(experiment_id)
        actual=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest() if row else None
        if actual!=expected:raise RuntimeError(f'Trial111 exact registered row changed: {experiment_id}')


def assert_clean_outputs():
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):raise RuntimeError('Refusing to overwrite Trial112 models')
    for path in (GRAPH_FEATURES_PATH,GRAPH_NODE_MAP_PATH,GRAPH_EDGES_PATH,GRAPH_METADATA_PATH):
        if path.exists():raise RuntimeError(f'Refusing to overwrite graph artifact: {path}')
    for stage in ('original','confirmation','final'):
        if any(REPORT_DIR.glob(f'{PREFIX}_{stage}_*')):raise RuntimeError('Refusing to overwrite Trial112 stage artifacts')
    for suffix in ('predictions.pkl','results.csv','summary.csv','fit_diagnostics.json','metadata.json'):
        if (REPORT_DIR/f'{PREFIX}_{suffix}').exists():raise RuntimeError('Refusing to overwrite Trial112 reports')
    if (PROJECT_ROOT/'submissions'/f'candidate_{PREFIX}.csv').exists():
        raise RuntimeError('Refusing to overwrite Trial112 candidate')


def build_terminal_coverage_features(graph_ids,edges,attributed_time,*,expected_components=None,
                                     expected_max_size=None,expect_no_self=False):
    """Count distinct reachable source and sink nodes in period-pure weak DAGs.

    A source contributes its own bit (and only a source does); forward OR merges
    source sets once per distinct terminal despite multiple routes. Reverse OR
    does the analogous calculation for sinks. Component-local bitsets are freed.
    """
    ids=np.asarray(graph_ids,dtype=np.int64)
    if ids.ndim!=1 or len(ids)==0 or len(np.unique(ids))!=len(ids) or not np.array_equal(ids,np.sort(ids)):
        raise ValueError('Graph IDs must be nonempty, unique, sorted int64')
    if not {'txId1','txId2'}.issubset(edges.columns):raise ValueError('Directed edge endpoints missing')
    if (attributed_time.txId.duplicated().any() or attributed_time.empty
            or len(attributed_time)>len(ids)):
        raise ValueError('Attributed period anchors missing or duplicated')
    attributed_ids=attributed_time.txId.to_numpy(dtype=np.int64)
    if not np.isin(attributed_ids,ids).all():raise ValueError('Attributed transaction absent from all-node graph')
    component_count=(expected_components if expected_components is not None
                     else int(_count_weak_components(ids,edges)))
    labels,node_period,component_meta=weak_component_period_map(
        ids,edges,attributed_time,expected_components=component_count)
    count=int(component_meta['weak_component_count'])
    minima=np.full(count,np.iinfo(np.int64).max,dtype=np.int64)
    np.minimum.at(minima,labels,ids)
    order=np.argsort(minima,kind='stable')
    remap=np.empty(count,dtype=np.int32);remap[order]=np.arange(count,dtype=np.int32)
    components=remap[labels]
    sizes=np.bincount(components,minlength=count)
    if expected_max_size is not None and int(sizes.max())!=expected_max_size:
        raise ValueError('Weak component maximum size differs from frozen expectation')
    if (node_period<=0).any():raise ValueError('All graph nodes require positive inferred periods')
    loc=pd.Index(ids)
    src=loc.get_indexer(edges.txId1.to_numpy(dtype=np.int64))
    dst=loc.get_indexer(edges.txId2.to_numpy(dtype=np.int64))
    if (src<0).any() or (dst<0).any():raise ValueError('Edge endpoint missing from graph IDs')
    pairs=np.unique(np.column_stack((src,dst)),axis=0) if len(src) else np.empty((0,2),dtype=np.int64)
    duplicate_rows=int(len(src)-len(pairs))
    self_mask=pairs[:,0]==pairs[:,1] if len(pairs) else np.zeros(0,dtype=bool)
    self_edges=int(self_mask.sum())
    if expect_no_self and self_edges:raise ValueError('Unexpected self-loop in frozen DAG')
    pairs=pairs[~self_mask]
    if len(pairs) and ((components[pairs[:,0]]!=components[pairs[:,1]]).any()
                       or (node_period[pairs[:,0]]!=node_period[pairs[:,1]]).any()):
        raise ValueError('Edge crosses a weak component or inferred time period')
    source_reach=np.zeros(len(ids),dtype=np.int32)
    sink_reach=np.zeros(len(ids),dtype=np.int32)
    component_sources=np.zeros(count,dtype=np.int32)
    component_sinks=np.zeros(count,dtype=np.int32)
    is_source=np.zeros(len(ids),dtype=bool)
    is_sink=np.zeros(len(ids),dtype=bool)
    local_index=np.full(len(ids),-1,dtype=np.int32)
    max_edge_count=0
    topo_hash=hashlib.sha256()
    for component in range(count):
        nodes=np.flatnonzero(components==component)
        n=len(nodes)
        local_index[nodes]=np.arange(n,dtype=np.int32)
        group=pairs[components[pairs[:,0]]==component] if len(pairs) else np.empty((0,2),dtype=np.int64)
        max_edge_count=max(max_edge_count,len(group))
        successors=[[] for _ in range(n)]
        indegree=np.zeros(n,dtype=np.int32)
        outdegree=np.zeros(n,dtype=np.int32)
        for gu,gv in group:
            u=int(local_index[gu]);v=int(local_index[gv])
            successors[u].append(v);indegree[v]+=1;outdegree[u]+=1
        for neighbours in successors:neighbours.sort()
        sources=np.flatnonzero(indegree==0)
        sinks=np.flatnonzero(outdegree==0)
        component_sources[component]=len(sources)
        component_sinks[component]=len(sinks)
        is_source[nodes[sources]]=True
        is_sink[nodes[sinks]]=True
        ready=[int(i) for i in sources];heapq.heapify(ready)
        degree=indegree.copy();topo=[]
        while ready:
            u=heapq.heappop(ready);topo.append(u)
            for v in successors[u]:
                degree[v]-=1
                if degree[v]==0:heapq.heappush(ready,v)
        if len(topo)!=n:
            raise ValueError(f'Directed cycle/SCC with more than one node in weak component {component}')
        if len(sources)==0 or len(sinks)==0:
            raise ValueError('Acyclic weak component lacks a source or sink terminal')
        topo_hash.update(ids[nodes[np.asarray(topo,dtype=np.int64)]].tobytes())
        source_bits=[0]*n
        for rank,u in enumerate(sources):source_bits[int(u)]=1<<rank
        for u in topo:
            bits=source_bits[u]
            for v in successors[u]:source_bits[v] |= bits
        source_reach[nodes]=[bits.bit_count() for bits in source_bits]
        del source_bits
        sink_bits=[0]*n
        for rank,u in enumerate(sinks):sink_bits[int(u)]=1<<rank
        for u in reversed(topo):
            for v in successors[u]:sink_bits[u] |= sink_bits[v]
        sink_reach[nodes]=[bits.bit_count() for bits in sink_bits]
        del sink_bits
        local_index[nodes]=-1
    source_total=component_sources[components]
    sink_total=component_sinks[components]
    if ((source_reach<1).any() or (sink_reach<1).any()
            or (source_reach>source_total).any() or (sink_reach>sink_total).any()
            or not np.array_equal(source_reach[is_source],np.ones(int(is_source.sum()),dtype=np.int32))
            or not np.array_equal(sink_reach[is_sink],np.ones(int(is_sink.sum()),dtype=np.int32))):
        raise ValueError('Distinct terminal reachability bounds/self counts failed')
    sc=source_reach.astype(np.float64);tc=sink_reach.astype(np.float64)
    values=np.column_stack((np.log1p(sc),np.log1p(tc),sc/source_total,tc/sink_total)).astype(np.float32)
    if not np.isfinite(values).all() or (values[:,2:]<0).any() or (values[:,2:]>1).any():
        raise ValueError('Nonfinite or out-of-range terminal coverage feature')
    features=pd.DataFrame(values,columns=TERMINAL4);features.insert(0,'txId',ids)
    node_map=pd.DataFrame({'txId':ids,'weak_component':components,'period':node_period,
                           'is_source':is_source,'is_sink':is_sink,
                           'reachable_source_count':source_reach,'reachable_sink_count':sink_reach,
                           'component_source_count':source_total,'component_sink_count':sink_total})
    meta={'graph_node_count':len(ids),'attributed_node_count':len(attributed_time),
          'attributed_ids_covered':int(np.isin(attributed_ids,ids).sum()),
          'graph_only_node_count':len(ids)-len(attributed_time),'weak_component_count':count,
          'max_component_nodes':int(sizes.max()),'max_component_edges':int(max_edge_count),
          'input_edge_rows':int(len(src)),'unique_edge_rows_before_self_omission':int(len(pairs)+self_edges),
          'duplicate_edge_rows':duplicate_rows,'self_loop_edges_omitted':self_edges,
          'directed_edges_after_dedup_self_omission':int(len(pairs)),
          'deduplicated_edge_index_sha256':hashlib.sha256(pairs.tobytes()).hexdigest(),
          'topological_node_id_sequence_sha256':topo_hash.hexdigest(),
          'topological_acyclic':True,'singleton_scc_count':len(ids),
          'cross_period_edge_count':0,'component_period_count':int(len(np.unique(node_period))),
          'source_terminal_count':int(is_source.sum()),'sink_terminal_count':int(is_sink.sum()),
          'max_component_source_count':int(component_sources.max()),
          'max_component_sink_count':int(component_sinks.max()),
          'max_reachable_source_count':int(source_reach.max()),
          'max_reachable_sink_count':int(sink_reach.max()),
          'sum_reachable_source_count':int(source_reach.sum()),
          'sum_reachable_sink_count':int(sink_reach.sum())}
    edge_catalog=pd.DataFrame({'txId1':ids[pairs[:,0]],'txId2':ids[pairs[:,1]]})
    return features,node_map,edge_catalog,meta


def _count_weak_components(graph_ids,edges):
    # Only for small synthetic checks without a frozen expected component count.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ids=np.asarray(graph_ids,dtype=np.int64);loc=pd.Index(ids)
    src=loc.get_indexer(edges.txId1.to_numpy(dtype=np.int64));dst=loc.get_indexer(edges.txId2.to_numpy(dtype=np.int64))
    if (src<0).any() or (dst<0).any():raise ValueError('Toy graph edge endpoint missing')
    graph=coo_matrix((np.ones(2*len(src),dtype=np.int8),
                      (np.r_[src,dst],np.r_[dst,src])),shape=(len(ids),len(ids))).tocsr()
    return connected_components(graph,directed=False,return_labels=False)


def attach_terminal4(train,test,features):
    if (features.columns.tolist()!=['txId',*TERMINAL4] or features.txId.duplicated().any()
            or not np.array_equal(features.txId.to_numpy(),np.sort(features.txId.to_numpy()))
            or features[TERMINAL4].isna().any().any()
            or not np.isfinite(features[TERMINAL4].to_numpy(float)).all()):
        raise ValueError('Frozen graph terminal coverage schema changed')
    outputs=[]
    for frame in (train,test):
        ids=frame.txId.to_numpy()
        joined=frame.merge(features,on='txId',how='left',validate='one_to_one',sort=False)
        if not np.array_equal(joined.txId.to_numpy(),ids) or joined[TERMINAL4].isna().any().any():
            raise ValueError('Terminal coverage join lost or reordered attributed rows')
        outputs.append(joined)
    return outputs


def selected_columns(train,full_columns,cutoff):
    base=historical_columns(train,full_columns,cutoff)
    if (len(full_columns)!=187 or len(base)!=177 or base[-5:]!=GRAPH5
            or len([c for c in base if c.startswith('feat_')])!=155):
        raise ValueError('Fixed 172 historical plus Trial90 graph5 selection changed')
    return base+TERMINAL4


def conditional_score(probability,classes):
    p=np.asarray(probability,dtype=np.float64)
    if (p.ndim!=2 or p.shape[1]!=3 or not np.array_equal(classes,[0,1,2])
            or not np.isfinite(p).all() or (p<0).any() or (p>1).any()):
        raise ValueError('Three-status HGB probability/class contract changed')
    den=p[:,0]+p[:,1]
    score=np.divide(p[:,1],den,out=np.zeros(len(p)),where=den>0)
    if not np.isfinite(score).all() or (score<0).any() or (score>1).any():
        raise ValueError('Conditional fraud score invalid')
    return score,den


def blend_score(control,challenger):
    a=np.asarray(control,dtype=float);b=np.asarray(challenger,dtype=float)
    if (a.ndim!=1 or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all()
            or (a<0).any() or (a>1).any() or (b<0).any() or (b>1).any()):
        raise ValueError('Blend requires aligned probabilities')
    return expit(.75*logit(np.clip(a,EPS,1-EPS))+.25*logit(np.clip(b,EPS,1-EPS)))


def fit_one(train,future,full_columns,cutoff,stop,stage,control=None):
    history=train.loc[train.time_step.le(cutoff)].copy()
    if (history.empty or history.time_step.max()!=cutoff or future.empty
            or future.time_step.min()<=cutoff or future.time_step.max()>stop
            or history.txId.duplicated().any() or future.txId.duplicated().any()
            or np.intersect1d(history.txId,future.txId).size):
        raise ValueError('Historical or forecast cutoff population changed')
    cols=selected_columns(train,full_columns,cutoff)
    status=label_status(history.label)
    weights,factor=training_weights_40(history.time_step,status,cutoff)
    if (len(cols)!=181 or not np.array_equal(np.unique(status),[0,1,2])
            or not np.isclose(weights[status==2].sum()/weights.sum(),.4,rtol=0,atol=1e-12)):
        raise ValueError('181-column three-status/raw-weight contract changed')
    x=history[cols].copy();v=future[cols].copy()
    if (x.isna().any().any() or v.isna().any().any()
            or not np.isfinite(x.to_numpy(float)).all() or not np.isfinite(v.to_numpy(float)).all()):
        raise ValueError('HGB model input contains nonfinite values')
    stem=f'{stage}_{cutoff}_{stop}'
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    model_path=MODEL_DIR/f'{stem}_model.joblib'
    input_path=REPORT_DIR/f'{PREFIX}_{stem}_model_inputs.joblib'
    forecast_path=REPORT_DIR/f'{PREFIX}_{stem}_forecast_probabilities.pkl'
    if any(p.exists() for p in (model_path,input_path,forecast_path)):
        raise FileExistsError(f'Trial112 artifact already exists for {stem}')
    joblib.dump({'history_features':x,'forecast_features':v,
                 'history_ids':history.txId.to_numpy(),'history_times':history.time_step.to_numpy(),
                 'history_labels':history.label.to_numpy(),
                 'forecast_ids':future.txId.to_numpy(),'forecast_times':future.time_step.to_numpy(),
                 'status':status,'raw_weights':weights,'columns':cols,
                 'cutoff':cutoff,'forecast_end':stop,'stage':stage,'plan_sha256':sha(PLAN_PATH),
                 'source_sha256':sha(SOURCE_PATH),
                 'graph_features_sha256':sha(GRAPH_FEATURES_PATH),
                 'graph_node_map_sha256':sha(GRAPH_NODE_MAP_PATH),
                 'deduplicated_edges_sha256':sha(GRAPH_EDGES_PATH)},input_path,compress=3)
    fitted=HistGradientBoostingClassifier(**model_config())
    fitted.fit(x,status,sample_weight=weights)
    if fitted.n_iter_!=600 or not np.array_equal(fitted.classes_,[0,1,2]):
        raise ValueError('Fixed 600-iteration three-status HGB contract changed')
    proba=fitted.predict_proba(v);score,den=conditional_score(proba,fitted.classes_)
    forecast=future[['txId','time_step','label']].copy().reset_index(drop=True)
    forecast.insert(0,'origin',cutoff)
    for i,name in enumerate(('p_known_licit','p_known_illicit','p_unknown')):
        forecast[name]=proba[:,i]
    forecast['p_known_denominator']=den;forecast[STANDALONE]=score
    known=forecast.loc[forecast.label.notna()].copy()
    if control is not None:
        ctl=control.loc[control.origin.eq(cutoff),KEY+['public99_control']]
        joined=known[KEY].merge(ctl,on=KEY,how='left',validate='one_to_one',sort=False,indicator=True)
        if (len(joined)!=len(known) or not joined._merge.eq('both').all()
                or not np.array_equal(joined.txId.to_numpy(),known.txId.to_numpy())):
            raise ValueError('Exact keyed Public99 control alignment failed')
        known['public99_control']=joined.public99_control.to_numpy(float)
        known[BLEND]=blend_score(known.public99_control,known[STANDALONE])
    bundle={'model':fitted,'columns':cols,'configuration':model_config(),
            'stage':stage,'origin':cutoff,'forecast_end':stop,'class_order':[0,1,2],
            'history_rows':len(history),'status_counts':np.bincount(status,minlength=3).tolist(),
            'status_weight_mass':np.bincount(status,weights=weights,minlength=3).tolist(),
            'unknown_weight_factor':float(factor),'raw_weight_sum':float(weights.sum()),
            'plan_sha256':sha(PLAN_PATH),'source_sha256':sha(SOURCE_PATH),
            'graph_features_sha256':sha(GRAPH_FEATURES_PATH),
            'graph_node_map_sha256':sha(GRAPH_NODE_MAP_PATH),
            'deduplicated_edges_sha256':sha(GRAPH_EDGES_PATH),
            'model_inputs_file':str(input_path.relative_to(PROJECT_ROOT)),
            'model_inputs_sha256':sha(input_path)}
    joblib.dump(bundle,model_path)
    replay=joblib.load(model_path)['model'].predict_proba(v)
    if not np.allclose(replay,proba,rtol=0,atol=1e-12):
        raise ValueError('Saved three-status HGB probability replay failed')
    forecast.to_pickle(forecast_path)
    diag={'stage':stage,'origin':cutoff,'forecast_end':stop,
          'history_rows':len(history),'forecast_rows':len(future),'known_forecast_rows':len(known),
          'feature_count':181,'columns':cols,'class_order':[0,1,2],
          'status_counts':bundle['status_counts'],'status_weight_mass':bundle['status_weight_mass'],
          'unknown_weight_factor':float(factor),'unknown_raw_weight_share':float(weights[status==2].sum()/weights.sum()),
          'iterations':int(fitted.n_iter_),'zero_known_denominator_rows':int((den==0).sum()),
          'graph_features_sha256':sha(GRAPH_FEATURES_PATH),
          'graph_node_map_sha256':sha(GRAPH_NODE_MAP_PATH),
          'deduplicated_edges_sha256':sha(GRAPH_EDGES_PATH),
          'model_file':str(model_path.relative_to(PROJECT_ROOT)),'model_sha256':sha(model_path),
          'model_inputs_file':str(input_path.relative_to(PROJECT_ROOT)),'model_inputs_sha256':sha(input_path),
          'forecast_file':str(forecast_path.relative_to(PROJECT_ROOT)),'forecast_sha256':sha(forecast_path)}
    return known,forecast,diag


def run_stage(train,full_columns,origins,stage,control):
    known=[];forecasts=[];diags=[]
    for cutoff,stop in origins.items():
        future=train.loc[train.time_step.between(cutoff+1,stop)].copy()
        p,f,d=fit_one(train,future,full_columns,cutoff,stop,stage,control)
        known.append(p);forecasts.append(f);diags.append(d)
        print(f'Trial112 {stage} {cutoff}->{stop} completed',flush=True)
    return pd.concat(known,ignore_index=True),pd.concat(forecasts,ignore_index=True),diags


def evaluate(predictions,stage):
    rows=[]
    for origin,frame in predictions.groupby('origin',sort=False):
        slices={'first7':frame.loc[frame.time_step.le(origin+7)]}
        if frame.time_step.max()==origin+14:
            slices.update(full14=frame,second7=frame.loc[frame.time_step.gt(origin+7)])
        for name,part in slices.items():
            for variant in ('public99_control',*VARIANTS):
                rows.append({'stage':stage,'origin':int(origin),'slice':name,'variant':variant,
                             'n':len(part),'positives':int(part.label.sum()),
                             'auc':float(roc_auc_score(part.label,part[variant])),
                             'average_precision':float(average_precision_score(part.label,part[variant]))})
    return pd.DataFrame(rows)


def summarize(results,stage):
    pivot=results.pivot(index=['origin','slice'],columns='variant',values='auc')
    if stage=='original':
        groups={'full14':[(14,'full14'),(21,'full14')],
                'second7':[(14,'second7'),(21,'second7')],
                'primary7':[(o,'first7') for o in (14,21,28)],
                'rolling7':[(o,'first7') for o in (17,24)]}
    else:
        groups={'full14':[(o,'full14') for o in CONFIRMATION],
                'second7':[(o,'second7') for o in CONFIRMATION],
                'first7':[(o,'first7') for o in CONFIRMATION]}
    rows=[]
    for variant in VARIANTS:
        row={'stage':stage,'variant':variant}
        for name,keys in groups.items():
            delta=pivot.loc[keys,variant]-pivot.loc[keys,'public99_control']
            row[f'{name}_mean_auc']=float(pivot.loc[keys,variant].mean())
            row[f'{name}_mean_delta']=float(delta.mean())
        row['worst_slice_delta']=float((pivot[variant]-pivot.public99_control).min())
        gate=(all(row[f'{name}_mean_delta']>0 for name in groups) if stage=='original' else
              row['full14_mean_delta']>0 and row['second7_mean_delta']>0 and row['first7_mean_delta']>=0)
        row['qualified']=bool(gate and row['worst_slice_delta']>=-.001)
        rows.append(row)
    return pd.DataFrame(rows)


def select_winner(original,confirmation):
    common=set(original.loc[original.qualified,'variant']) & set(confirmation.loc[confirmation.qualified,'variant'])
    return (BLEND if BLEND in common else STANDALONE if STANDALONE in common else None),sorted(common)


def create_candidate(train,test,full_columns,sample,winner):
    control=pd.read_csv(PUBLIC99_CANDIDATE)
    expected=sample['index'].to_numpy()
    test_ids=test['index'].to_numpy() if 'index' in test.columns else test.txId.to_numpy()
    if (len(expected)!=15329 or not np.array_equal(test_ids,expected)
            or not np.array_equal(control['index'].to_numpy(),expected)):
        raise ValueError('Official sample/test/Public99 order changed')
    future=test.copy()
    if 'label' in future and future.label.notna().any():raise ValueError('Final test contains labels')
    future['label']=np.nan
    _,forecast,diag=fit_one(train,future,full_columns,35,49,'final')
    score=forecast[STANDALONE].to_numpy(float)
    blend=blend_score(control.target.to_numpy(float),score)
    components=pd.DataFrame({'index':expected,'public99_control':control.target.to_numpy(float),
                             STANDALONE:score,BLEND:blend,
                             'p_known_licit':forecast.p_known_licit,'p_known_illicit':forecast.p_known_illicit,
                             'p_unknown':forecast.p_unknown,'p_known_denominator':forecast.p_known_denominator})
    components_path=REPORT_DIR/f'{PREFIX}_final_components.pkl'
    if components_path.exists():raise FileExistsError(components_path)
    components.to_pickle(components_path)
    candidate_path=PROJECT_ROOT/'submissions'/f'candidate_{PREFIX}.csv'
    pd.DataFrame({'index':expected,'target':blend if winner==BLEND else score}).to_csv(candidate_path,index=False)
    read=pd.read_csv(candidate_path)
    if (len(read)!=15329 or not np.array_equal(read['index'].to_numpy(),expected)
            or not np.isfinite(read.target).all() or (read.target<0).any() or (read.target>1).any()):
        raise ValueError('Trial112 candidate readback invalid')
    return {'candidate_created':True,'final_model_fit_performed':True,'winner':winner,
            'candidate_file':str(candidate_path.relative_to(PROJECT_ROOT)),'candidate_sha256':sha(candidate_path),
            'final_components_file':str(components_path.relative_to(PROJECT_ROOT)),
            'final_components_sha256':sha(components_path)},diag


def verify_pure_toys():
    def graph(ids,pairs,anchors=None):
        edges=pd.DataFrame(pairs,columns=['txId1','txId2'])
        if anchors is None:anchors=ids
        times=pd.DataFrame({'txId':anchors,'time_step':[1]*len(anchors)})
        return build_terminal_coverage_features(np.array(ids,dtype=np.int64),edges,times)
    chain,m,edgecat,meta=graph([1,2,3,4],[(1,2),(2,3),(3,4)])
    assert m.reachable_source_count.tolist()==[1]*4 and m.reachable_sink_count.tolist()==[1]*4
    assert m.is_source.tolist()==[True,False,False,False]
    assert m.is_sink.tolist()==[False,False,False,True]
    assert np.allclose(chain[TERMINAL4[2:]],1.)
    diamond,m,_,_=graph([1,2,3,4],[(1,2),(1,3),(2,4),(3,4),(1,2)])
    assert m.reachable_source_count.tolist()==[1]*4 and m.reachable_sink_count.tolist()==[1]*4
    assert meta['duplicate_edge_rows']==0 and len(edgecat)==3
    merger,mm,_,_=graph([1,2,3,4],[(1,3),(2,3),(3,4)])
    assert mm.reachable_source_count.tolist()==[1,1,2,2]
    assert mm.reachable_sink_count.tolist()==[1,1,1,1]
    assert np.isclose(merger.loc[merger.txId.eq(3),TERMINAL4[2]].iloc[0],1.)
    split,sm,_,_=graph([1,2,3,4,5],[(1,3),(3,4),(3,5)])
    assert sm.reachable_source_count.tolist()==[1]*5
    assert sm.reachable_sink_count.tolist()==[2,1,2,1,1]
    both,bm,_,_=graph([1,2,3,4,5],[(1,3),(2,3),(3,4),(3,5)])
    assert bm.loc[bm.txId.eq(3),'reachable_source_count'].iloc[0]==2
    assert bm.loc[bm.txId.eq(3),'reachable_sink_count'].iloc[0]==2
    assert np.isclose(both.loc[both.txId.eq(3),TERMINAL4[2]].iloc[0],1.)
    assert np.isclose(both.loc[both.txId.eq(3),TERMINAL4[3]].iloc[0],1.)
    repeated,mr,ec,mrmeta=graph([1,2,3,4,5],[(1,2),(1,3),(2,4),(3,4),(1,2)],anchors=[1,5])
    assert mr.reachable_source_count.tolist()==[1]*5 and mr.reachable_sink_count.tolist()==[1]*5
    assert mrmeta['duplicate_edge_rows']==1 and len(ec)==4
    assert mr.loc[mr.txId.eq(5),'is_source'].iloc[0] and mr.loc[mr.txId.eq(5),'is_sink'].iloc[0]
    assert np.allclose(repeated.loc[repeated.txId.eq(5),TERMINAL4].to_numpy(float),
                       [np.log1p(1),np.log1p(1),1.,1.],atol=1e-6)
    with_self,ws,_,self_meta=graph([1,2,3],[(1,2),(2,2)],anchors=[1,3])
    assert self_meta['self_loop_edges_omitted']==1 and ws.reachable_source_count.tolist()==[1,1,1]
    reordered,_,_,_=graph([1,2,3,4],[(3,4),(1,3),(1,2),(2,4),(1,2)])
    assert diamond.equals(reordered)
    try:graph([1,2,3],[(1,2),(2,3),(3,1)])
    except ValueError as e:assert 'cycle' in str(e).lower() or 'terminal' in str(e).lower()
    else:raise AssertionError('Directed cycle was accepted')
    rng=np.random.default_rng(42)
    for n in range(2,11):
        pairs=[(i+1,j+1) for i in range(n) for j in range(i+1,n) if rng.random()<.35]
        f,m,_,_=graph(list(range(1,n+1)),pairs)
        successors={i:set() for i in range(1,n+1)}
        predecessors={i:set() for i in range(1,n+1)}
        for u,v in pairs:successors[u].add(v);predecessors[v].add(u)
        source_nodes={i for i in successors if not predecessors[i]}
        sink_nodes={i for i in successors if not successors[i]}
        def traverse(start,adj):
            found={start};stack=[start]
            while stack:
                for v in adj[stack.pop()]:
                    if v not in found:found.add(v);stack.append(v)
            return found
        for node in range(1,n+1):
            upstream=traverse(node,predecessors)&source_nodes
            downstream=traverse(node,successors)&sink_nodes
            row=m.loc[m.txId.eq(node)].iloc[0]
            assert row.reachable_source_count==len(upstream)
            assert row.reachable_sink_count==len(downstream)
            weak=traverse(node,{i:successors[i]|predecessors[i] for i in successors})
            assert row.component_source_count==len(weak&source_nodes)
            assert row.component_sink_count==len(weak&sink_nodes)
            feat=f.loc[f.txId.eq(node)].iloc[0]
            assert np.isclose(feat[TERMINAL4[2]],len(upstream)/len(weak&source_nodes),atol=1e-7)
            assert np.isclose(feat[TERMINAL4[3]],len(downstream)/len(weak&sink_nodes),atol=1e-7)
    assert np.allclose(blend_score([.5,.5],[.5,.5]),.5)
    assert model_config()=={'max_leaf_nodes':15,'min_samples_leaf':40,'learning_rate':.03,
                            'max_iter':600,'l2_regularization':2.,'max_features':.8,
                            'random_state':42,'early_stopping':False}
    print('Trial112 exact distinct terminal coverage, dedup, cycle and scoring toys passed')


def main():
    plan=json.loads(PLAN_PATH.read_text())
    verify_frozen_inputs(plan)
    assert_clean_outputs()
    # Competition data and edges first load only after exact root authorization.
    train,test,sample,groups=load_features()
    _,_,edges,_=load_competition_data()
    graph5=pd.read_pickle(GRAPH5_PATH)
    train,test=attach_graph5(train,test,graph5)
    graph_ids=graph5.txId.to_numpy(dtype=np.int64)
    attributed=pd.concat([train[['txId','time_step']],test[['txId','time_step']]],ignore_index=True)
    if (len(graph_ids)!=GRAPH_NODE_COUNT or len(attributed)!=ATTRIBUTED_NODE_COUNT
            or len(graph_ids)-len(attributed)!=GRAPH_ONLY_NODE_COUNT
            or attributed.txId.duplicated().any()):
        raise ValueError('Frozen all-node/attributed graph populations changed')
    graph,graph_map,graph_edges,graph_meta=build_terminal_coverage_features(
        graph_ids,edges,attributed,expected_components=WEAK_COMPONENT_COUNT,
        expected_max_size=MAX_COMPONENT_NODES,expect_no_self=True)
    graph.to_pickle(GRAPH_FEATURES_PATH);graph_map.to_pickle(GRAPH_NODE_MAP_PATH)
    graph_edges.to_pickle(GRAPH_EDGES_PATH)
    graph_meta.update(source_sha256=sha(SOURCE_PATH),plan_sha256=sha(PLAN_PATH),
                      graph_features_file=str(GRAPH_FEATURES_PATH.relative_to(PROJECT_ROOT)),
                      graph_features_sha256=sha(GRAPH_FEATURES_PATH),
                      graph_node_map_file=str(GRAPH_NODE_MAP_PATH.relative_to(PROJECT_ROOT)),
                      graph_node_map_sha256=sha(GRAPH_NODE_MAP_PATH),
                      deduplicated_edges_file=str(GRAPH_EDGES_PATH.relative_to(PROJECT_ROOT)),
                      deduplicated_edges_sha256=sha(GRAPH_EDGES_PATH))
    GRAPH_METADATA_PATH.write_text(json.dumps(graph_meta,indent=2,sort_keys=True)+'\n')
    train,test=attach_terminal4(train,test,graph)
    full_columns=list(groups['regularized_two_hop'])+GRAPH5
    if len(full_columns)!=187:raise ValueError('Expected 182 candidates plus fixed Trial90 graph5')
    metadata={'trial_number':112,'variant':PREFIX,'source_sha256':sha(SOURCE_PATH),
              'plan_sha256':sha(PLAN_PATH),'graph_metadata_sha256':sha(GRAPH_METADATA_PATH),
              'original_gate_passed':False,'confirmation_run':False,
              'confirmation_gate_passed':False,'same_variant_passed_both_stages':False,
              'two_stage_qualified_variants':[],'winner':None,'candidate_created':False,
              'final_model_fit_performed':False,'kaggle_upload':False}
    predictions=[];results=[];summaries=[];diags=[]
    control=load_public99_control('original',ORIGINS)
    original,forecast,detail=run_stage(train,full_columns,ORIGINS,'original',control)
    original.to_pickle(REPORT_DIR/f'{PREFIX}_original_predictions.pkl')
    forecast.to_pickle(REPORT_DIR/f'{PREFIX}_original_forecast_probabilities.pkl')
    ro=evaluate(original,'original');so=summarize(ro,'original')
    ro.to_csv(REPORT_DIR/f'{PREFIX}_original_results.csv',index=False)
    so.to_csv(REPORT_DIR/f'{PREFIX}_original_summary.csv',index=False)
    metadata['original_gate_passed']=bool(so.qualified.any())
    predictions.append(original);results.append(ro);summaries.append(so);diags.extend(detail)
    if metadata['original_gate_passed']:
        control_c=load_public99_control('confirmation',CONFIRMATION)
        confirm,forecast_c,detail_c=run_stage(train,full_columns,CONFIRMATION,'confirmation',control_c)
        confirm.to_pickle(REPORT_DIR/f'{PREFIX}_confirmation_predictions.pkl')
        forecast_c.to_pickle(REPORT_DIR/f'{PREFIX}_confirmation_forecast_probabilities.pkl')
        rc=evaluate(confirm,'confirmation');sc=summarize(rc,'confirmation')
        rc.to_csv(REPORT_DIR/f'{PREFIX}_confirmation_results.csv',index=False)
        sc.to_csv(REPORT_DIR/f'{PREFIX}_confirmation_summary.csv',index=False)
        winner,intersection=select_winner(so,sc)
        metadata.update(confirmation_run=True,confirmation_gate_passed=bool(sc.qualified.any()),
                        same_variant_passed_both_stages=winner is not None,
                        two_stage_qualified_variants=intersection,winner=winner)
        predictions.append(confirm);results.append(rc);summaries.append(sc);diags.extend(detail_c)
        if winner:
            final_meta,final_diag=create_candidate(train,test,full_columns,sample,winner)
            metadata.update(final_meta);diags.append(final_diag)
    pd.concat(predictions,ignore_index=True).to_pickle(REPORT_DIR/f'{PREFIX}_predictions.pkl')
    pd.concat(results,ignore_index=True).to_csv(REPORT_DIR/f'{PREFIX}_results.csv',index=False)
    pd.concat(summaries,ignore_index=True).to_csv(REPORT_DIR/f'{PREFIX}_summary.csv',index=False)
    (REPORT_DIR/f'{PREFIX}_fit_diagnostics.json').write_text(json.dumps(diags,indent=2,sort_keys=True)+'\n')
    (REPORT_DIR/f'{PREFIX}_metadata.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='--fit-free-checks':
        verify_pure_toys()
    elif len(sys.argv)==1:
        main()
    else:
        raise SystemExit('Usage: source.py [--fit-free-checks]')
