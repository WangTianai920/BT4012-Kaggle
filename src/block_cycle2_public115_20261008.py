"""Trial116: two undirected cyclic biconnected-block membership features.

Only root authorization permits loading Trial112's frozen all-node graph or fitting.
"""
from __future__ import annotations

import csv
import hashlib
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
from src.competition import PROJECT_ROOT,REPORT_DIR
from src.horizon_hgb_20260917 import historical_columns
from src.label_status_multiclass_20260929 import label_status
from src.status_mass40_lr03_iter600_20261003 import training_weights_40
from src.feature_propagation16_public99_20261007 import attach_graph5
from src.articulation_cut2_public99_20261008 import attach_cut2

PREFIX='block_cycle2_public115_20261008'
PLAN_PATH=REPORT_DIR/f'{PREFIX}_plan.json'
AUTH_PATH=REPORT_DIR/f'{PREFIX}_root_authorization.json'
SOURCE_PATH=PROJECT_ROOT/'src'/f'{PREFIX}.py'
MODEL_DIR=PROJECT_ROOT/'models'/PREFIX
GRAPH5_PATH=REPORT_DIR/'directed_chain_position5_public55_20261005_graph_features.pkl'
GRAPH_FEATURES_PATH=REPORT_DIR/f'{PREFIX}_graph_features.pkl'
CUT2_PATH=REPORT_DIR/'articulation_cut2_public99_20261008_graph_features.pkl'
CONTROL_PREFIX='articulation_cut2_public99_20261008'
CONTROL_CANDIDATE=PROJECT_ROOT/'submissions/candidate_articulation_cut2_public99_20261008.csv'
GRAPH_NODE_MAP_PATH=REPORT_DIR/f'{PREFIX}_graph_node_map.pkl'
GRAPH_EDGES_PATH=REPORT_DIR/'terminal_coverage4_public99_20261008_deduplicated_edges.pkl'
PRIOR_NODE_MAP_PATH=REPORT_DIR/'terminal_coverage4_public99_20261008_graph_node_map.pkl'
GRAPH_METADATA_PATH=REPORT_DIR/f'{PREFIX}_graph_metadata.json'

ORIGINS={14:28,21:35,17:24,24:31,28:35}
CONFIRMATION={17:31,18:32,20:34}
KEY=['origin','txId','time_step','label']
GRAPH5=['dag_log1p_scc_size','dag_log1p_source_depth','dag_log1p_sink_depth',
        'dag_log1p_chain_span','dag_chain_position']
CUT2=['graph_cut_pair_disconnect_fraction','graph_cut_dominant_remainder_loss']
BLOCK2=['graph_block_log1p_largest_cycle_excess','graph_block_cycle_membership_count']
CONTROL='trial115_control'
STANDALONE='block_cycle2_hgb'
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
    if (plan.get('trial_number')!=116 or plan.get('variant')!=PREFIX
            or plan.get('source_sha256')!=sha(SOURCE_PATH)
            or plan.get('model_config')!=model_config()
            or plan.get('graph_feature_columns')!=BLOCK2):
        raise RuntimeError('Trial116 source or frozen configuration changed')
    auth=json.loads(AUTH_PATH.read_text())
    if (auth.get('authorized') is not True or auth.get('source_sha256')!=sha(SOURCE_PATH)
            or auth.get('plan_sha256')!=sha(PLAN_PATH)
            or auth.get('registry_rows')!=plan['registry_preflight']['unique_rows']
            or auth.get('root_trial116_block_cycle_allnode_weights_and_gates_reviewed') is not True
            or auth.get('trial115_public_result_and_closeout_acknowledged') is not True
            or auth.get('no_test_labels_or_external_data') is not True):
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
    for experiment_id,expected in plan['trial115_exact_rows'].items():
        row=by_id.get(experiment_id)
        actual=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest() if row else None
        if actual!=expected:raise RuntimeError(f'Trial115 exact registered row changed: {experiment_id}')


def assert_clean_outputs():
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):raise RuntimeError('Refusing to overwrite Trial116 models')
    for path in (GRAPH_FEATURES_PATH,GRAPH_NODE_MAP_PATH,GRAPH_METADATA_PATH):
        if path.exists():raise RuntimeError(f'Refusing to overwrite graph artifact: {path}')
    for stage in ('original','confirmation','final'):
        if any(REPORT_DIR.glob(f'{PREFIX}_{stage}_*')):raise RuntimeError('Refusing to overwrite Trial116 stage artifacts')
    for suffix in ('predictions.pkl','results.csv','summary.csv','fit_diagnostics.json','metadata.json'):
        if (REPORT_DIR/f'{PREFIX}_{suffix}').exists():raise RuntimeError('Refusing to overwrite Trial116 reports')
    if (PROJECT_ROOT/'submissions'/f'candidate_{PREFIX}.csv').exists():
        raise RuntimeError('Refusing to overwrite Trial116 candidate')


def build_block_cycle_features(prior_node_map, edges, *, expected_components=None,
                               expected_max_size=None):
    """Deterministic iterative Tarjan edge-stack biconnected blocks, all nodes.

    A bridge-only edge forms a size-two block. Only blocks with >=3 vertices
    contribute: log1p(max(size-2)) and count of such block memberships.
    No labels, node attributes, training period, or future targets enter.
    """
    required={'txId','weak_component','period','is_source','is_sink',
              'reachable_source_count','reachable_sink_count',
              'component_source_count','component_sink_count'}
    if not required.issubset(prior_node_map.columns):
        raise ValueError('Trial112 node-map schema changed')
    m=prior_node_map.copy()
    ids=m.txId.to_numpy(dtype=np.int64);n=len(ids)
    if n==0 or not np.array_equal(ids,np.sort(np.unique(ids))) or (m.period.to_numpy(int)<=0).any():
        raise ValueError('Trial112 node IDs or periods changed')
    components=m.weak_component.to_numpy(dtype=np.int32)
    k=int(components.max())+1
    if (components.min()!=0 or set(np.unique(components))!=set(range(k)) or
        (expected_components is not None and k!=expected_components)):
        raise ValueError('Trial112 weak components changed')
    sizes=np.bincount(components,minlength=k)
    if expected_max_size is not None and int(sizes.max())!=expected_max_size:
        raise ValueError('Trial112 maximum component size changed')
    period=m.period.to_numpy(dtype=np.int32)
    if any(np.unique(period[components==c]).size!=1 for c in range(k)):
        raise ValueError('Weak component crosses inferred periods')
    if (edges.columns.tolist()!=['txId1','txId2'] or edges.duplicated().any() or
        (edges.txId1==edges.txId2).any()):
        raise ValueError('Trial112 directed edge catalog changed')
    loc=pd.Index(ids)
    src=loc.get_indexer(edges.txId1.to_numpy(dtype=np.int64))
    dst=loc.get_indexer(edges.txId2.to_numpy(dtype=np.int64))
    if (src<0).any() or (dst<0).any() or np.any(components[src]!=components[dst]) or np.any(period[src]!=period[dst]):
        raise ValueError('Trial112 edge endpoint/component/period changed')
    directed_index=np.column_stack((src,dst)).astype(np.int64)
    undirected=np.unique(np.column_stack((np.minimum(src,dst),np.maximum(src,dst))).astype(np.int64),axis=0)
    adjacency=[[] for _ in range(n)]
    for a,b in undirected:
        adjacency[int(a)].append(int(b));adjacency[int(b)].append(int(a))
    for neighbors in adjacency:neighbors.sort()
    discovery=np.full(n,-1,dtype=np.int32);low=np.full(n,-1,dtype=np.int32)
    parent=np.full(n,-1,dtype=np.int32);edge_stack=[]
    largest_cycle_excess=np.zeros(n,dtype=np.int32)
    cycle_memberships=np.zeros(n,dtype=np.int32)
    block_memberships=np.zeros(n,dtype=np.int32)
    roots_per_component=np.zeros(k,dtype=np.int32)
    block_count=cycle_block_count=bridge_block_count=0;tick=0
    for root in range(n):
        if discovery[root]>=0:continue
        roots_per_component[components[root]]+=1
        discovery[root]=low[root]=tick;tick+=1
        stack=[[root,0]]
        while stack:
            u,j=stack[-1]
            if j<len(adjacency[u]):
                v=adjacency[u][j];stack[-1][1]+=1
                if discovery[v]<0:
                    parent[v]=u;edge_stack.append((u,v))
                    discovery[v]=low[v]=tick;tick+=1
                    stack.append([v,0])
                elif v!=parent[u] and discovery[v]<discovery[u]:
                    edge_stack.append((u,v));low[u]=min(low[u],discovery[v])
            else:
                stack.pop();p=int(parent[u])
                if p>=0:
                    low[p]=min(low[p],low[u])
                    if low[u]>=discovery[p]:
                        members=set()
                        while True:
                            if not edge_stack:raise ValueError('Tarjan block edge stack underflow')
                            a,b=edge_stack.pop();members.add(a);members.add(b)
                            if a==p and b==u:break
                        block_count+=1
                        if len(members)<2:raise ValueError('Degenerate biconnected block')
                        cyclic=len(members)>=3
                        cycle_block_count+=int(cyclic)
                        bridge_block_count+=int(not cyclic)
                        for v in members:
                            block_memberships[v]+=1
                            if cyclic:
                                cycle_memberships[v]+=1
                                largest_cycle_excess[v]=max(largest_cycle_excess[v],len(members)-2)
        if edge_stack:raise ValueError('Biconnected edge stack not empty at component root')
    if tick!=n or not np.array_equal(roots_per_component,np.ones(k,dtype=np.int32)):
        raise ValueError('Frozen weak component disconnected in undirected projection')
    if bridge_block_count+cycle_block_count!=block_count or np.any(cycle_memberships>block_memberships):
        raise ValueError('Invalid biconnected block accounting')
    values=np.column_stack((np.log1p(largest_cycle_excess),cycle_memberships)).astype(np.float32)
    features=pd.DataFrame(values,columns=BLOCK2);features.insert(0,'txId',ids)
    node_map=m.copy()
    node_map['block_membership_count']=block_memberships
    node_map['cycle_block_membership_count']=cycle_memberships
    node_map['largest_cycle_block_excess']=largest_cycle_excess
    for name in BLOCK2:node_map[name]=features[name].to_numpy()
    meta={'graph_node_count':n,'weak_component_count':k,'max_component_nodes':int(sizes.max()),
          'directed_edges_input':len(directed_index),'undirected_edges_after_dedup':len(undirected),
          'deduplicated_edge_index_sha256':hashlib.sha256(directed_index.tobytes()).hexdigest(),
          'undirected_projection_index_sha256':hashlib.sha256(undirected.tobytes()).hexdigest(),
          'cross_period_edge_count':0,'block_count':block_count,
          'cyclic_block_count':cycle_block_count,'bridge_block_count':bridge_block_count,
          'nodes_in_cyclic_blocks':int((cycle_memberships>0).sum()),
          'max_cycle_memberships':int(cycle_memberships.max()),
          'max_cycle_block_excess':int(largest_cycle_excess.max()),
          'dfs_roots_per_component':roots_per_component.tolist()}
    return features,node_map,meta


def attach_block2(train,test,features):
    if (features.columns.tolist()!=['txId',*BLOCK2] or features.txId.duplicated().any()
        or not np.array_equal(features.txId.to_numpy(),np.sort(features.txId.to_numpy()))
        or features[BLOCK2].isna().any().any()
        or not np.isfinite(features[BLOCK2].to_numpy(float)).all()):
        raise ValueError('Frozen block feature schema changed')
    outputs=[]
    for frame in (train,test):
        ids=frame.txId.to_numpy()
        joined=frame.merge(features,on='txId',how='left',validate='one_to_one',sort=False)
        if not np.array_equal(joined.txId.to_numpy(),ids) or joined[BLOCK2].isna().any().any():
            raise ValueError('Block feature join lost or reordered attributed rows')
        outputs.append(joined)
    return outputs


def load_trial115_control(stage,origins):
    if stage not in ('original','confirmation'):raise ValueError('Unsupported control stage')
    path=REPORT_DIR/f'{CONTROL_PREFIX}_{stage}_predictions.pkl'
    return validate_trial115_control_frame(pd.read_pickle(path),origins)


def validate_trial115_control_frame(frame,origins):
    required=KEY+['public99_control','articulation_cut2_hgb','blend_75_25_logodds']
    if (not set(required).issubset(frame.columns) or frame[KEY].duplicated().any()
        or frame[required].isna().any().any()):
        raise ValueError('Trial115 cached selected-blend control schema changed')
    forecast=[]
    for origin,stop in origins.items():
        part=frame.loc[frame.origin.eq(origin)&frame.time_step.between(origin+1,stop)].copy()
        if part.empty or part.time_step.min()!=origin+1 or part.time_step.max()!=stop:
            raise ValueError('Trial115 control historical window missing')
        exact=blend_score(part.public99_control.to_numpy(float),part.articulation_cut2_hgb.to_numpy(float))
        if not np.allclose(exact,part.blend_75_25_logodds.to_numpy(float),atol=1e-12,rtol=0):
            raise ValueError('Trial115 saved selected blend fails ancestor replay')
        forecast.append(part[KEY+['blend_75_25_logodds']].rename(columns={'blend_75_25_logodds':CONTROL}))
    out=pd.concat(forecast,ignore_index=True)
    if out[KEY].duplicated().any() or not np.isfinite(out[CONTROL].to_numpy(float)).all():
        raise ValueError('Trial115 control exact keyed alignment failed')
    return out


def selected_columns(train,full_columns,cutoff):
    base=historical_columns(train,full_columns,cutoff)
    if (len(full_columns)!=187 or len(base)!=177 or base[-5:]!=GRAPH5
            or len([c for c in base if c.startswith('feat_')])!=155):
        raise ValueError('Fixed 172 historical plus Trial90 graph5 selection changed')
    return base+CUT2+BLOCK2


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
        raise FileExistsError(f'Trial116 artifact already exists for {stem}')
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
        ctl=control.loc[control.origin.eq(cutoff),KEY+[CONTROL]]
        joined=known[KEY].merge(ctl,on=KEY,how='left',validate='one_to_one',sort=False,indicator=True)
        if (len(joined)!=len(known) or not joined._merge.eq('both').all()
                or not np.array_equal(joined.txId.to_numpy(),known.txId.to_numpy())):
            raise ValueError('Exact keyed Trial115 control alignment failed')
        known[CONTROL]=joined[CONTROL].to_numpy(float)
        known[BLEND]=blend_score(known[CONTROL],known[STANDALONE])
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
        print(f'Trial116 {stage} {cutoff}->{stop} completed',flush=True)
    return pd.concat(known,ignore_index=True),pd.concat(forecasts,ignore_index=True),diags


def evaluate(predictions,stage):
    rows=[]
    for origin,frame in predictions.groupby('origin',sort=False):
        slices={'first7':frame.loc[frame.time_step.le(origin+7)]}
        if frame.time_step.max()==origin+14:
            slices.update(full14=frame,second7=frame.loc[frame.time_step.gt(origin+7)])
        for name,part in slices.items():
            for variant in (CONTROL,*VARIANTS):
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
            delta=pivot.loc[keys,variant]-pivot.loc[keys,CONTROL]
            row[f'{name}_mean_auc']=float(pivot.loc[keys,variant].mean())
            row[f'{name}_mean_delta']=float(delta.mean())
        row['worst_slice_delta']=float((pivot[variant]-pivot[CONTROL]).min())
        gate=(all(row[f'{name}_mean_delta']>0 for name in groups) if stage=='original' else
              row['full14_mean_delta']>0 and row['second7_mean_delta']>0 and row['first7_mean_delta']>=0)
        row['qualified']=bool(gate and row['worst_slice_delta']>=-.001)
        rows.append(row)
    return pd.DataFrame(rows)


def select_winner(original,confirmation):
    common=set(original.loc[original.qualified,'variant']) & set(confirmation.loc[confirmation.qualified,'variant'])
    return (BLEND if BLEND in common else STANDALONE if STANDALONE in common else None),sorted(common)


def create_candidate(train,test,full_columns,sample,winner):
    control=pd.read_csv(CONTROL_CANDIDATE)
    expected=sample['index'].to_numpy()
    test_ids=test['index'].to_numpy() if 'index' in test.columns else test.txId.to_numpy()
    if (len(expected)!=15329 or not np.array_equal(test_ids,expected)
            or not np.array_equal(control['index'].to_numpy(),expected)):
        raise ValueError('Official sample/test/Trial115 control order changed')
    future=test.copy()
    if 'label' in future and future.label.notna().any():raise ValueError('Final test contains labels')
    future['label']=np.nan
    _,forecast,diag=fit_one(train,future,full_columns,35,49,'final')
    score=forecast[STANDALONE].to_numpy(float)
    blend=blend_score(control.target.to_numpy(float),score)
    components=pd.DataFrame({'index':expected,CONTROL:control.target.to_numpy(float),
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
        raise ValueError('Trial116 candidate readback invalid')
    return {'candidate_created':True,'final_model_fit_performed':True,'winner':winner,
            'candidate_file':str(candidate_path.relative_to(PROJECT_ROOT)),'candidate_sha256':sha(candidate_path),
            'final_components_file':str(components_path.relative_to(PROJECT_ROOT)),
            'final_components_sha256':sha(components_path)},diag


def verify_pure_toys():
    from src.terminal_coverage4_public99_20261008 import build_terminal_coverage_features
    import networkx as nx
    def graph(ids,pairs):
        ids=sorted(ids)
        raw=pd.DataFrame([(min(a,b),max(a,b)) for a,b in pairs],columns=['txId1','txId2'])
        times=pd.DataFrame({'txId':ids,'time_step':[1]*len(ids)})
        _,prior,directed,prior_meta=build_terminal_coverage_features(np.asarray(ids,dtype=np.int64),raw,times)
        features,node_map,meta=build_block_cycle_features(prior,directed)
        assert meta['deduplicated_edge_index_sha256']==prior_meta['deduplicated_edge_index_sha256']
        return features,node_map,meta,prior,directed
    cases=[([1,2,3,4,5],[(1,2),(1,3),(1,4),(1,5)]),
           ([1,2,3,4],[(1,2),(2,3),(3,4),(1,4)]),
           ([1,2,3,4,5],[(1,2),(2,3),(1,3),(3,4),(4,5)]),
           ([1,2,3,4,5],[(i,j) for i in range(1,6) for j in range(i+1,6)]),
           ([1,2,3,4,5],[(i,i+1) for i in range(1,5)]),
           ([1,2,3,4],[(1,2)]),
           ([1,2,3,4],[(1,2),(1,3),(2,4),(3,4)])]
    rng=np.random.default_rng(42)
    for n in range(2,11):
        for _ in range(10):
            cases.append((list(range(1,n+1)),[(i,j) for i in range(1,n+1)
                 for j in range(i+1,n+1) if rng.random()<.35]))
    for ids,pairs in cases:
        f,m,meta,prior,_=graph(ids,pairs)
        g=nx.Graph();g.add_nodes_from(ids);g.add_edges_from(pairs)
        blocks=list(nx.biconnected_components(g))
        for row in m.itertuples(index=False):
            cyclic=[len(b)-2 for b in blocks if len(b)>=3 and row.txId in b]
            assert np.isclose(row.graph_block_log1p_largest_cycle_excess,
                              np.log1p(max(cyclic,default=0)),atol=1e-7)
            assert row.graph_block_cycle_membership_count==len(cyclic)
            assert row.block_membership_count==sum(row.txId in b for b in blocks)
        assert meta['cyclic_block_count']==sum(len(b)>=3 for b in blocks)
        assert f.equals(graph(ids,list(reversed(pairs)))[0])
    star,_,_,_,_=graph(*cases[0]);assert np.allclose(star[BLOCK2],0.)
    cycle,_,_,_,_=graph(*cases[1]);assert np.allclose(cycle[BLOCK2].to_numpy(float),[np.log1p(2),1])
    long,_,meta,_,_=graph(list(range(1,1302)),[(i,i+1) for i in range(1,1301)])
    assert meta['bridge_block_count']==1300 and np.allclose(long[BLOCK2],0.)
    _,_,_,prior,directed=graph([1,2,3],[(1,2),(2,3)])
    crossed=prior.copy();crossed.loc[crossed.txId.eq(3),'period']=2
    try:build_block_cycle_features(crossed,directed)
    except ValueError as e:assert 'period' in str(e).lower()
    else:raise AssertionError('Cross-period edge was accepted')
    assert np.allclose(blend_score([.5,.5],[.5,.5]),.5)
    cached=pd.DataFrame({'origin':[14,14],'txId':[1,2],'time_step':[15,15],
                         'label':[0.,1.],'public99_control':[.2,.7],
                         'articulation_cut2_hgb':[.4,.8]})
    cached['blend_75_25_logodds']=blend_score(cached.public99_control,
                                              cached.articulation_cut2_hgb)
    replay=validate_trial115_control_frame(cached,{14:15})
    assert replay.columns.tolist()==KEY+[CONTROL] and len(replay)==2
    assert np.allclose(replay[CONTROL],cached.blend_75_25_logodds)
    corrupted=cached.copy();corrupted.loc[0,'blend_75_25_logodds']+=.01
    try:validate_trial115_control_frame(corrupted,{14:15})
    except ValueError as e:assert 'ancestor replay' in str(e)
    else:raise AssertionError('Corrupted parent control accepted')
    print('Trial116 synthetic block reference, edge order, long-chain and scoring checks passed')


def main():
    plan=json.loads(PLAN_PATH.read_text())
    verify_frozen_inputs(plan)
    assert_clean_outputs()
    # Competition-derived data first load only after exact root authorization.
    train,test,sample,groups=load_features()
    edges=pd.read_pickle(GRAPH_EDGES_PATH)
    prior_map=pd.read_pickle(PRIOR_NODE_MAP_PATH)
    graph5=pd.read_pickle(GRAPH5_PATH)
    cut2=pd.read_pickle(CUT2_PATH)
    train,test=attach_graph5(train,test,graph5)
    train,test=attach_cut2(train,test,cut2)
    attributed=pd.concat([train[['txId','time_step']],test[['txId','time_step']]],ignore_index=True)
    if (len(prior_map)!=GRAPH_NODE_COUNT or len(attributed)!=ATTRIBUTED_NODE_COUNT
            or len(prior_map)-len(attributed)!=GRAPH_ONLY_NODE_COUNT
            or not np.array_equal(graph5.txId.to_numpy(dtype=np.int64),prior_map.txId.to_numpy(dtype=np.int64))
            or attributed.txId.duplicated().any()):
        raise ValueError('Frozen all-node/attributed graph populations changed')
    graph,graph_map,graph_meta=build_block_cycle_features(
        prior_map,edges,expected_components=WEAK_COMPONENT_COUNT,
        expected_max_size=MAX_COMPONENT_NODES)
    prior_meta=json.loads((REPORT_DIR/'terminal_coverage4_public99_20261008_graph_metadata.json').read_text())
    for key in ('deduplicated_edge_index_sha256','weak_component_count','max_component_nodes',
                'cross_period_edge_count'):
        if graph_meta[key]!=prior_meta[key]:raise ValueError(f'Trial112 graph invariant changed: {key}')
    if graph_meta['directed_edges_input']!=prior_meta['directed_edges_after_dedup_self_omission']:
        raise ValueError('Trial112 deduplicated directed edge count changed')
    graph_meta['per_period_feature_diagnostics']={}
    for period,part in graph_map.groupby('period',sort=True):
        arr=part[BLOCK2].to_numpy(dtype=np.float64)
        record={'node_count':len(part),'nonzero_both_count':int(((arr[:,0]>0)&(arr[:,1]>0)).sum())}
        for i,name in enumerate(BLOCK2):
            q=np.quantile(arr[:,i],[0,.05,.5,.95,.99,1])
            record[name]={'nonzero_count':int((arr[:,i]>0).sum()),
                          'min':float(q[0]),'p05':float(q[1]),'median':float(q[2]),
                          'p95':float(q[3]),'p99':float(q[4]),'max':float(q[5])}
        graph_meta['per_period_feature_diagnostics'][str(int(period))]=record
    graph.to_pickle(GRAPH_FEATURES_PATH);graph_map.to_pickle(GRAPH_NODE_MAP_PATH)
    graph_meta.update(source_sha256=sha(SOURCE_PATH),plan_sha256=sha(PLAN_PATH),
                      prior_node_map_sha256=sha(PRIOR_NODE_MAP_PATH),
                      graph_features_file=str(GRAPH_FEATURES_PATH.relative_to(PROJECT_ROOT)),
                      graph_features_sha256=sha(GRAPH_FEATURES_PATH),
                      graph_node_map_file=str(GRAPH_NODE_MAP_PATH.relative_to(PROJECT_ROOT)),
                      graph_node_map_sha256=sha(GRAPH_NODE_MAP_PATH),
                      deduplicated_edges_file=str(GRAPH_EDGES_PATH.relative_to(PROJECT_ROOT)),
                      deduplicated_edges_sha256=sha(GRAPH_EDGES_PATH))
    GRAPH_METADATA_PATH.write_text(json.dumps(graph_meta,indent=2,sort_keys=True)+'\n')
    train,test=attach_block2(train,test,graph)
    full_columns=list(groups['regularized_two_hop'])+GRAPH5
    if len(full_columns)!=187:raise ValueError('Expected 182 candidates plus fixed Trial90 graph5')
    metadata={'trial_number':116,'variant':PREFIX,'source_sha256':sha(SOURCE_PATH),
              'plan_sha256':sha(PLAN_PATH),'graph_metadata_sha256':sha(GRAPH_METADATA_PATH),
              'original_gate_passed':False,'confirmation_run':False,
              'confirmation_gate_passed':False,'same_variant_passed_both_stages':False,
              'two_stage_qualified_variants':[],'winner':None,'candidate_created':False,
              'final_model_fit_performed':False,'kaggle_upload':False}
    predictions=[];results=[];summaries=[];diags=[]
    control=load_trial115_control('original',ORIGINS)
    original,forecast,detail=run_stage(train,full_columns,ORIGINS,'original',control)
    original.to_pickle(REPORT_DIR/f'{PREFIX}_original_predictions.pkl')
    forecast.to_pickle(REPORT_DIR/f'{PREFIX}_original_forecast_probabilities.pkl')
    ro=evaluate(original,'original');so=summarize(ro,'original')
    ro.to_csv(REPORT_DIR/f'{PREFIX}_original_results.csv',index=False)
    so.to_csv(REPORT_DIR/f'{PREFIX}_original_summary.csv',index=False)
    metadata['original_gate_passed']=bool(so.qualified.any())
    predictions.append(original);results.append(ro);summaries.append(so);diags.extend(detail)
    if metadata['original_gate_passed']:
        control_c=load_trial115_control('confirmation',CONFIRMATION)
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
