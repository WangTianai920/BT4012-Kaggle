"""Independent biconnected block-incidence forest and saved-model audit for frozen Trial115.

The root runs the real audit only after fitting. Import and --pure-toy do not
read competition data or execute model evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from collections import deque
import heapq

csv.field_size_limit(10_000_000)

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,roc_auc_score

from src.competition import PROJECT_ROOT,REPORT_DIR,load_competition_data
from src.horizon_hgb_20260917 import historical_columns
from src.verify_feature_propagation16_public99_20261007 import (
    GRAPH5,KEY,rebuild_public99_stage,rebuild_public99_final,
)

ROOT,REPORTS=PROJECT_ROOT,REPORT_DIR
P="articulation_cut2_public99_20261008"
SOURCE=ROOT/"src"/f"{P}.py"
PLAN=REPORTS/f"{P}_plan.json"
AUTH=REPORTS/f"{P}_root_authorization.json"
META=REPORTS/f"{P}_metadata.json"
FLAT=REPORTS/f"{P}_fit_diagnostics.json"
AUDIT=REPORTS/f"{P}_parent_verification.json"
LOG=ROOT/"experiment_log.csv"
GRAPH5_CACHE=REPORTS/"directed_chain_position5_public55_20261005_graph_features.pkl"
GRAPH=REPORTS/f"{P}_graph_features.pkl"
NODE_MAP=REPORTS/f"{P}_graph_node_map.pkl"
GRAPH_META=REPORTS/f"{P}_graph_metadata.json"
EDGE_CATALOG=REPORTS/"terminal_coverage4_public99_20261008_deduplicated_edges.pkl"
PRIOR112_MAP=REPORTS/"terminal_coverage4_public99_20261008_graph_node_map.pkl"
PRIOR112_GRAPH_META=REPORTS/"terminal_coverage4_public99_20261008_graph_metadata.json"
PIN_COUNT=866
PRIOR114="native_lgbm_ce3_public99_20261008"
PRIOR114_AUDIT=REPORTS/f"{PRIOR114}_parent_verification.json"
PRIOR114_AUDIT_ID=f"S114_{PRIOR114}_independent_parent_audit"
PRIOR114_ELIGIBILITY_ID=f"S114_{PRIOR114}_final_eligibility"
CUT2=["graph_cut_pair_disconnect_fraction","graph_cut_dominant_remainder_loss"]
CONTROL,STANDALONE,BLEND="public99_control","articulation_cut2_hgb","blend_75_25_logodds"
ORIGINS={14:28,21:35,17:24,24:31,28:35}
CONFIRMATION={17:31,18:32,20:34}
EPS=1e-12
SOURCE_SHA="60bd2660c0baf595d1770ec44b2c5ba17fbd2a6d0874d5d80972dd861186ac18"
PLAN_SHA="91d0ad4e1deb7c29c6d23e9be9402a979fcdcad3582fe1afab59f5cfbc0a8291"
PREFLIGHT_ROWS=15523
PRIOR114_AUDIT_SHA="db61e63f3b519f445e20bae03c685acd395df7fa0c26b646edd3cc13ced4a5f0"
PRIOR114_AUDIT_ROW_SHA="027ecc616b96944509440391345f722653e2d6d799689846ee73ae59d750166d"
PRIOR114_ELIGIBILITY_ROW_SHA="0589d2dc91a81e7aa27ce6f10f9a54ea95a8b4f25be91adb6b42965bec9e67f2"


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry_prefix_sha(path:Path,n_rows:int)->str:
    raw=path.read_bytes()
    with io.StringIO(raw.decode("utf-8-sig"),newline="") as stream:
        reader=csv.reader(stream)
        next(reader)
        for _ in range(n_rows):next(reader)
        physical_lines=reader.line_num
    return hashlib.sha256(b"".join(raw.splitlines(keepends=True)[:physical_lines])).hexdigest()


def row_sha(row:dict)->str:
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def fixed_blend(control,challenger):
    a,b=np.asarray(control,float),np.asarray(challenger,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError("Trial115 blend inputs invalid")
    return expit(.75*logit(np.clip(a,EPS,1-EPS))+.25*logit(np.clip(b,EPS,1-EPS)))


def hard_status_weights(steps,status,cutoff):
    base=np.power(.5,(float(cutoff)-np.asarray(steps,float))/20.)
    unknown=np.asarray(status)==2
    known_mass,unknown_mass=base[~unknown].sum(),base[unknown].sum()
    if known_mass<=0 or unknown_mass<=0:raise AssertionError("Trial115 historical statuses absent")
    factor=known_mass*.4/(.6*unknown_mass)
    weights=base.copy();weights[unknown]*=factor
    return weights,float(factor)



def metrics(stage,frame):
    rows=[]
    for origin,part in frame.groupby("origin",sort=False):
        slices=[("first7",part.loc[part.time_step.le(origin+7)])]
        if int(part.time_step.max())==int(origin)+14:
            slices += [("full14",part),("second7",part.loc[part.time_step.gt(origin+7)])]
        for name,subset in slices:
            for variant in (CONTROL,STANDALONE,BLEND):
                rows.append({"stage":stage,"origin":int(origin),"slice":name,"variant":variant,
                             "n":len(subset),"positives":int(subset.label.sum()),
                             "auc":float(roc_auc_score(subset.label,subset[variant])),
                             "average_precision":float(average_precision_score(subset.label,subset[variant]))})
    return pd.DataFrame(rows)


def summary(stage,results):
    pivot=results.pivot(index=["origin","slice"],columns="variant",values="auc")
    groups=({"full14":[(14,"full14"),(21,"full14")],
             "second7":[(14,"second7"),(21,"second7")],
             "primary7":[(o,"first7") for o in (14,21,28)],
             "rolling7":[(o,"first7") for o in (17,24)]} if stage=="original" else
            {"full14":[(o,"full14") for o in CONFIRMATION],
             "second7":[(o,"second7") for o in CONFIRMATION],
             "first7":[(o,"first7") for o in CONFIRMATION]})
    rows=[]
    for variant in (STANDALONE,BLEND):
        row={"stage":stage,"variant":variant}
        for name,keys in groups.items():
            delta=pivot.loc[keys,variant]-pivot.loc[keys,CONTROL]
            row[f"{name}_mean_auc"]=float(pivot.loc[keys,variant].mean())
            row[f"{name}_mean_delta"]=float(delta.mean())
        row["worst_slice_delta"]=float((pivot[variant]-pivot[CONTROL]).min())
        gate=(all(row[f"{name}_mean_delta"]>0 for name in groups) if stage=="original" else
              row["full14_mean_delta"]>0 and row["second7_mean_delta"]>0 and row["first7_mean_delta"]>=0)
        row["qualified"]=bool(gate and row["worst_slice_delta"]>=-.001)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_csv(path,expected):
    pd.testing.assert_frame_equal(pd.read_csv(path).reset_index(drop=True),expected.reset_index(drop=True),
                                  check_dtype=False,check_exact=False,rtol=0,atol=1e-12)



def check_frozen():
    if sha(SOURCE)!=SOURCE_SHA or sha(PLAN)!=PLAN_SHA or sha(PRIOR114_AUDIT)!=PRIOR114_AUDIT_SHA:
        raise AssertionError("Trial115 frozen source, plan, or Trial114 audit changed")
    plan,auth,meta,flat=[json.loads(path.read_text()) for path in (PLAN,AUTH,META,FLAT)]
    config={"max_leaf_nodes":15,"min_samples_leaf":40,"learning_rate":.03,
            "max_iter":600,"l2_regularization":2.,"max_features":.8,
            "random_state":42,"early_stopping":False}
    if (plan.get("trial_number")!=115 or plan.get("variant")!=P
            or plan.get("source_sha256")!=SOURCE_SHA or plan.get("model_config")!=config
            or plan.get("graph_feature_columns")!=CUT2
            or auth.get("authorized") is not True or auth.get("source_sha256")!=SOURCE_SHA
            or auth.get("plan_sha256")!=PLAN_SHA or auth.get("registry_rows")!=PREFLIGHT_ROWS
            or auth.get("root_trial115_undirected_articulation_allnode_weights_and_gates_reviewed") is not True
            or auth.get("trial114_negative_closeout_acknowledged") is not True
            or auth.get("no_test_labels_or_external_data") is not True
            or meta.get("trial_number")!=115 or meta.get("variant")!=P
            or meta.get("source_sha256")!=SOURCE_SHA or meta.get("plan_sha256")!=PLAN_SHA
            or not isinstance(flat,list)):
        raise AssertionError("Trial115 plan/auth/metadata identity or configuration changed")
    pins=plan.get("input_sha256",{})
    if len(pins)!=PIN_COUNT or "experiment_log.csv" not in pins:
        raise AssertionError("Trial115 immutable pin inventory changed")
    for rel,digest in pins.items():
        actual=registry_prefix_sha(LOG,PREFLIGHT_ROWS) if rel=="experiment_log.csv" else sha(ROOT/rel)
        if actual!=digest:raise AssertionError(f"Trial115 frozen input changed: {rel}")
    with LOG.open(newline="",encoding="utf-8-sig") as stream:rows=list(csv.DictReader(stream))
    ids=[r.get("experiment_id","") for r in rows]
    if len(rows)<PREFLIGHT_ROWS or not all(ids) or len(ids)!=len(set(ids)):
        raise AssertionError("Trial115 unique-ID registry preflight changed")
    by_id={r["experiment_id"]:r for r in rows}
    prior_rows=plan.get("trial114_exact_rows",{})
    if (len(prior_rows)!=68 or prior_rows.get(PRIOR114_AUDIT_ID)!=PRIOR114_AUDIT_ROW_SHA
            or prior_rows.get(PRIOR114_ELIGIBILITY_ID)!=PRIOR114_ELIGIBILITY_ROW_SHA):
        raise AssertionError("Trial115 exact Trial114 closeout inventory changed")
    for eid,digest in prior_rows.items():
        if eid not in by_id or row_sha(by_id[eid])!=digest:
            raise AssertionError(f"Trial114 registered closeout row changed: {eid}")
    prior=json.loads(PRIOR114_AUDIT.read_text())
    if (prior.get("saved_three_status_model_replays")!=5 or prior.get("metric_rows_recomputed")!=27
            or prior.get("public99_control_member_replays")!=15
            or prior.get("native_multiclass_model_verified") is not True
            or prior.get("inherited_graph5_provenance_verified") is not True
            or prior.get("eligible_variants")!=[]
            or prior.get("final_artifact_checks",{}).get("candidate_created") is not False):
        raise AssertionError("Trial114 independently audited negative closeout changed")
    return plan,auth,meta,flat


def block_forest_reference(graph_ids,edge_pairs):
    """Independent weighted biconnected incidence forest; no lowlink reuse.

    Every original vertex has weight1, each biconnected block has weight0.
    Removing original vertex v separates its forest-neighbor branches; their
    exact integer weights are the connected-component sizes after deleting v.
    """
    ids=[int(x) for x in graph_ids]
    graph=nx.Graph();graph.add_nodes_from(ids)
    graph.add_edges_from((int(u),int(v)) for u,v in edge_pairs if int(u)!=int(v))
    if set(graph)!=set(ids):raise AssertionError("Undirected edge endpoint outside all-node universe")
    components=sorted((sorted(c) for c in nx.connected_components(graph)),key=lambda c:c[0])
    records=[]
    for component,members in enumerate(components):
        local=graph.subgraph(members);tree=nx.Graph()
        tree.add_nodes_from((('v',v),{'weight':1}) for v in members)
        for b,block in enumerate(nx.biconnected_components(local)):
            bnode=('b',b);tree.add_node(bnode,weight=0)
            tree.add_edges_from((bnode,('v',v)) for v in block)
        if not nx.is_tree(tree):raise AssertionError("Block incidence graph must be a tree per component")
        root=('v',members[0]);parent={root:None};order=[root]
        for node in order:
            for neighbor in tree[node]:
                if neighbor==parent[node]:continue
                if neighbor in parent:raise AssertionError("Block incidence traversal encountered cycle")
                parent[neighbor]=node;order.append(neighbor)
        subtree={node:int(tree.nodes[node]['weight']) for node in order}
        for node in reversed(order[1:]):subtree[parent[node]]+=subtree[node]
        n=len(members)
        if subtree[root]!=n:raise AssertionError("Block forest vertex mass changed")
        for v in members:
            node=('v',v);pieces=[]
            for neighbor in tree[node]:
                size=(n-subtree[node] if neighbor==parent[node] else subtree[neighbor])
                if size<=0:raise AssertionError("Every removal branch must contain original vertices")
                pieces.append(size)
            remaining=n-1;sum_square=sum(s*s for s in pieces)
            ordered_cut=remaining*remaining-sum_square
            largest=max(pieces,default=0)
            if sum(pieces)!=remaining or ordered_cut<0 or ordered_cut%2:
                raise AssertionError("Exact removal branch/pair accounting invalid")
            # Unordered cross-piece pairs divided by all remaining unordered pairs.
            pair_fraction=(ordered_cut//2)/((remaining*(remaining-1))//2) if remaining>=2 else 0.
            mass_loss=(remaining-largest)/remaining if remaining>0 else 0.
            records.append({'txId':v,'weak_component':component,'component_node_count':n,
                            'remaining_piece_count':len(pieces),'remaining_piece_square_sum':sum_square,
                            'largest_remaining_component_size':largest,
                            'ordered_pair_disconnect_count':ordered_cut,'is_articulation':len(pieces)>1,
                            CUT2[0]:pair_fraction,CUT2[1]:mass_loss})
    return pd.DataFrame(records).sort_values('txId',kind='stable').reset_index(drop=True)


def deletion_bfs_reference(graph):
    """Synthetic-only slow reference: explicitly remove each original vertex."""
    output=[]
    for component in nx.connected_components(graph):
        n=len(component)
        for v in component:
            remainder=graph.subgraph(component-{v})
            pieces=[len(c) for c in nx.connected_components(remainder)]
            r=n-1;ordered=r*r-sum(s*s for s in pieces);largest=max(pieces,default=0)
            output.append((v,ordered,largest,len(pieces)>1,
                (ordered//2)/((r*(r-1))//2) if r>=2 else 0.,(r-largest)/r if r else 0.))
    return sorted(output)


def pure_toy():
    graphs=[nx.path_graph(1),nx.path_graph(2),nx.path_graph(8),nx.star_graph(7),
            nx.cycle_graph(8),nx.complete_graph(7),nx.lollipop_graph(5,6)]
    graphs.append(nx.disjoint_union(nx.path_graph(5),nx.complete_graph(4)))
    for seed in range(30):graphs.append(nx.gnp_random_graph(15,.15,seed=seed))
    for graph in graphs:
        reference=block_forest_reference(sorted(graph),graph.edges())
        actual=[(int(r.txId),int(r.ordered_pair_disconnect_count),
                 int(r.largest_remaining_component_size),bool(r.is_articulation),
                 float(getattr(r,CUT2[0])),float(getattr(r,CUT2[1])))
                for r in reference.itertuples(index=False)]
        assert actual==deletion_bfs_reference(graph)
    # More than the normal recursion limit, while all production/reference walks are iterative.
    graph=nx.path_graph(1501);reference=block_forest_reference(sorted(graph),graph.edges())
    assert reference.is_articulation.sum()==1499
    for v in (0,1,750,1499,1500):
        r=reference.loc[reference.txId.eq(v)].iloc[0]
        assert r.ordered_pair_disconnect_count==2*v*(1500-v)
        assert r.largest_remaining_component_size==max(v,1500-v)
    assert np.isclose(fixed_blend([.5],[.5])[0],.5)
    print("Trial115 independent block forest vs direct deletion BFS toys and1501-node chain passed")


def verify_graph(train,test,edges,meta):
    graph5=pd.read_pickle(GRAPH5_CACHE)
    if (graph5.columns.tolist()!=['txId',*GRAPH5] or len(graph5)!=203769
            or graph5.txId.duplicated().any() or not graph5.txId.is_monotonic_increasing
            or graph5[GRAPH5].isna().any().any()):
        raise AssertionError("Trial115 inherited all-node graph5 cache changed")
    prior_map=pd.read_pickle(PRIOR112_MAP)
    canonical=edges[['txId1','txId2']].astype(np.int64).drop_duplicates()
    canonical=canonical.loc[canonical.txId1.ne(canonical.txId2)].sort_values(
        ['txId1','txId2'],kind='stable').reset_index(drop=True)
    inherited_edges=pd.read_pickle(EDGE_CATALOG)
    if (not canonical.equals(inherited_edges) or len(canonical)!=234355
            or len(prior_map)!=203769 or not np.array_equal(prior_map.txId,graph5.txId)):
        raise AssertionError("Trial115 official directed edges/inherited all-node map changed")
    directed_index=np.column_stack((np.searchsorted(graph5.txId,canonical.txId1),
                                    np.searchsorted(graph5.txId,canonical.txId2))).astype(np.int64)
    pairs=np.sort(directed_index,axis=1)
    pairs=np.unique(pairs,axis=0).astype(np.int64)
    ids=graph5.txId.to_numpy(np.int64)
    undirected=np.column_stack((ids[pairs[:,0]],ids[pairs[:,1]]))
    reference=block_forest_reference(ids,undirected)
    if (not np.array_equal(reference.txId,ids)
            or not np.array_equal(reference.weak_component,prior_map.weak_component)):
        raise AssertionError("Trial115 independently reconstructed weak-component universe changed")
    saved,raw=pd.read_pickle(GRAPH),pd.read_pickle(NODE_MAP)
    pd.testing.assert_frame_equal(raw[prior_map.columns],prior_map,check_exact=True)
    extra=['dfs_discovery','dfs_lowlink','dfs_parent_txId','dfs_subtree_size',
           'separated_child_count','separated_child_size_sum','separated_child_size_square_sum',
           'separated_child_size_max','residual_component_size','largest_remaining_component_size',
           'ordered_pair_disconnect_count','is_articulation',*CUT2]
    if (raw.columns.tolist()!=[*prior_map.columns,*extra]
            or saved.columns.tolist()!=['txId',*CUT2] or len(saved)!=203769
            or not np.array_equal(saved.txId,ids)
            or saved[CUT2].dtypes.astype(str).tolist()!=['float32']*2
            or not np.isfinite(saved[CUT2].to_numpy()).all()
            or ((saved[CUT2].to_numpy()<0)|(saved[CUT2].to_numpy()>1)).any()):
        raise AssertionError("Trial115 all-node cut feature/map schema, order or bounds changed")
    for col in ['largest_remaining_component_size','ordered_pair_disconnect_count','is_articulation']:
        if not np.array_equal(raw[col],reference[col]):
            raise AssertionError(f"Trial115 independent block forest removal aggregate differs: {col}")
    expected=reference[CUT2].to_numpy(float).astype(np.float32)
    if (not np.array_equal(saved[CUT2].to_numpy(),expected)
            or not np.array_equal(raw[CUT2].to_numpy(),saved[CUT2].to_numpy())):
        raise AssertionError("Trial115 all-node float32 cut features differ from exact block-forest pairs/masses")
    parent_ids=raw.dfs_parent_txId.to_numpy(np.int64)
    roots=parent_ids==-1
    if int(roots.sum())!=49:raise AssertionError("Trial115 DFS requires one root per weak component")
    parent_index=np.searchsorted(ids,parent_ids[~roots])
    if (np.any(parent_index>=len(ids)) or not np.array_equal(ids[parent_index],parent_ids[~roots])
            or not np.array_equal(raw.weak_component.to_numpy()[parent_index],raw.weak_component.to_numpy()[~roots])
            or not raw.loc[roots].weak_component.is_unique
            or (raw.loc[roots,'residual_component_size']!=0).any()
            or not np.array_equal(raw.loc[roots,'dfs_subtree_size'],reference.loc[roots,'component_node_count'])):
        raise AssertionError("Trial115 persisted DFS parent/root/subtree component accounting changed")
    undirected_pairs={tuple(row) for row in pairs}
    vertex_index=np.flatnonzero(~roots)
    if any(tuple(sorted((int(i),int(p)))) not in undirected_pairs for i,p in zip(vertex_index,parent_index)):
        raise AssertionError("Trial115 DFS parent is not an observed undirected edge")
    discovery=raw.dfs_discovery.to_numpy(np.int64)
    if (not np.array_equal(np.sort(discovery),np.arange(len(ids)))
            or np.any(discovery[parent_index]>=discovery[~roots])):
        raise AssertionError("Trial115 global DFS discovery/parent order changed")
    replay_subtree=np.ones(len(ids),np.int64)
    parent_of=np.full(len(ids),-1,np.int64);parent_of[~roots]=parent_index
    for i in np.argsort(discovery)[::-1]:
        if parent_of[i]>=0:replay_subtree[parent_of[i]]+=replay_subtree[i]
    if not np.array_equal(replay_subtree,raw.dfs_subtree_size):
        raise AssertionError("Trial115 persisted DFS parent-tree subtree masses changed")
    remaining=reference.component_node_count.to_numpy(np.int64)-1
    residual=raw.residual_component_size.to_numpy(np.int64)
    count=raw.separated_child_count.to_numpy(np.int64)
    size_sum=raw.separated_child_size_sum.to_numpy(np.int64)
    square_sum=raw.separated_child_size_square_sum.to_numpy(np.int64)
    maximum=raw.separated_child_size_max.to_numpy(np.int64)
    if (np.any(residual<0) or np.any(count<0) or np.any(size_sum<0) or np.any(square_sum<0)
            or np.any(maximum<0) or not np.array_equal(size_sum+residual,remaining)
            or not np.array_equal(square_sum+residual*residual,reference.remaining_piece_square_sum)
            or not np.array_equal(count+(residual>0),reference.remaining_piece_count)
            or not np.array_equal(np.maximum(maximum,residual),reference.largest_remaining_component_size)
            or np.any(raw.dfs_lowlink.to_numpy()<0)
            or np.any(raw.dfs_lowlink.to_numpy()>raw.dfs_discovery.to_numpy())
            or np.any(raw.dfs_subtree_size.to_numpy()<1)
            or np.any(raw.dfs_subtree_size.to_numpy()>reference.component_node_count.to_numpy())):
        raise AssertionError("Trial115 persisted DFS branch/residual accounting differs from independent block forest")
    gm=json.loads(GRAPH_META.read_text())
    expected_diag={'graph_node_count':len(ids),'weak_component_count':int(reference.weak_component.nunique()),
        'max_component_nodes':int(reference.component_node_count.max()),'directed_edges_input':len(canonical),
        'undirected_edges_after_dedup':len(pairs),
        'undirected_projection_index_sha256':hashlib.sha256(pairs.tobytes()).hexdigest(),
        'deduplicated_edge_index_sha256':hashlib.sha256(directed_index.tobytes()).hexdigest(),
        'max_dfs_subtree_size':int(reference.component_node_count.max()),
        'dfs_roots_per_component':[1]*int(reference.weak_component.nunique()),
        'cross_period_edge_count':int(np.sum(prior_map.period.to_numpy()[pairs[:,0]]!=prior_map.period.to_numpy()[pairs[:,1]])),
        'articulation_count':int(reference.is_articulation.sum()),
        'positive_pair_disconnect_nodes':int((reference[CUT2[0]]>0).sum()),
        'max_pair_disconnect_fraction':float(reference[CUT2[0]].max()),
        'max_dominant_remainder_loss':float(reference[CUT2[1]].max())}
    by_period={}
    for period in sorted(prior_map.period.unique()):
        selected=prior_map.period.eq(period).to_numpy()
        arr=expected[selected].astype(np.float64)
        record={'node_count':int(selected.sum()),
                'nonzero_both_count':int(((arr[:,0]>0)&(arr[:,1]>0)).sum())}
        for i,name in enumerate(CUT2):
            q=np.quantile(arr[:,i],[0,.05,.5,.95,.99,1])
            record[name]={'nonzero_count':int((arr[:,i]>0).sum()),
                          'min':float(q[0]),'p05':float(q[1]),'median':float(q[2]),
                          'p95':float(q[3]),'p99':float(q[4]),'max':float(q[5])}
        by_period[str(int(period))]=record
    expected_diag['per_period_feature_diagnostics']=by_period
    for key,value in expected_diag.items():
        if gm.get(key)!=value:raise AssertionError(f"Trial115 exact block-forest graph diagnostic differs: {key}")
    prior_audit=json.loads((REPORTS/'terminal_coverage4_public99_20261008_parent_verification.json').read_text())
    prior_gm=json.loads(PRIOR112_GRAPH_META.read_text())
    for key,value in prior_audit['graph_diagnostics_verified'].items():
        if prior_gm.get(key)!=value:raise AssertionError(f"Trial112 inherited graph provenance changed: {key}")
    if (gm.get('source_sha256')!=SOURCE_SHA or gm.get('plan_sha256')!=PLAN_SHA
            or gm.get('graph_features_sha256')!=sha(GRAPH)
            or gm.get('graph_node_map_sha256')!=sha(NODE_MAP)
            or gm.get('deduplicated_edges_sha256')!=sha(EDGE_CATALOG)
            or gm.get('graph_features_file')!=str(GRAPH.relative_to(ROOT))
            or gm.get('graph_node_map_file')!=str(NODE_MAP.relative_to(ROOT))
            or gm.get('deduplicated_edges_file')!=str(EDGE_CATALOG.relative_to(ROOT))
            or gm.get('prior_node_map_sha256')!=sha(PRIOR112_MAP)
            or meta.get('graph_metadata_sha256')!=sha(GRAPH_META)):
        raise AssertionError("Trial115 graph source/plan/artifact provenance hashes changed")
    joined=[]
    for frame in (train,test):
        original=frame.txId.to_numpy()
        enriched=frame.merge(graph5,on='txId',how='left',validate='one_to_one',sort=False)
        enriched=enriched.merge(saved,on='txId',how='left',validate='one_to_one',sort=False)
        if not np.array_equal(enriched.txId,original) or enriched[GRAPH5+CUT2].isna().any().any():
            raise AssertionError("Trial115 attributed graph join changed")
        joined.append(enriched)
    return joined[0],joined[1],{'independent_block_incidence_forest_verified':True,**expected_diag}



def verify_fold(stage,origin,stop,detail,train,test,full_columns,plan,control):
    history=train.loc[train.time_step.le(origin)]
    future=(test.copy() if stage=="final" else train.loc[train.time_step.between(origin+1,stop)])
    if stage=="final":
        if "label" in future and future.label.notna().any():
            raise AssertionError("Trial115 final test labels present")
        future["label"]=np.nan
    columns=historical_columns(train,full_columns,origin)+CUT2
    if (len(columns)!=179 or columns[-7:-2]!=GRAPH5 or columns[-2:]!=CUT2
            or len([c for c in columns if c.startswith("feat_")])!=155):
        raise AssertionError("Trial115 historical177 plus undirected articulation2 feature selection changed")
    status=np.where(history.label.isna().to_numpy(),2,history.label.fillna(2).to_numpy()).astype(np.int64)
    weights,factor=hard_status_weights(history.time_step,status,origin)
    if (not np.array_equal(np.unique(status),[0,1,2])
            or not np.isclose(weights[status==2].sum()/weights.sum(),.4,rtol=0,atol=1e-12)):
        raise AssertionError("Trial115 three statuses/raw 40% mass changed")
    x,v=history[columns],future[columns]
    if x.isna().any().any() or v.isna().any().any():
        raise AssertionError("Trial115 model features missing")
    input_path=ROOT/detail["model_inputs_file"]
    model_path=ROOT/detail["model_file"]
    forecast_path=ROOT/detail["forecast_file"]
    for kind,path in (("model_inputs",input_path),("model",model_path),("forecast",forecast_path)):
        if sha(path)!=detail[f"{kind}_sha256"]:
            raise AssertionError(f"Trial115 saved {kind} artifact changed")
    saved=joblib.load(input_path);bundle=joblib.load(model_path)
    if (saved.get("stage")!=stage or saved.get("cutoff")!=origin
            or saved.get("forecast_end")!=stop or saved.get("plan_sha256")!=PLAN_SHA
            or saved.get("source_sha256")!=SOURCE_SHA
            or saved.get("graph_features_sha256")!=sha(GRAPH)
            or saved.get("graph_node_map_sha256")!=sha(NODE_MAP)
            or saved.get("deduplicated_edges_sha256")!=sha(EDGE_CATALOG)
            or saved.get("columns")!=columns or bundle.get("columns")!=columns
            or bundle.get("configuration")!=plan["model_config"]
            or bundle.get("stage")!=stage or bundle.get("origin")!=origin
            or bundle.get("forecast_end")!=stop or bundle.get("class_order")!=[0,1,2]
            or bundle.get("source_sha256")!=SOURCE_SHA or bundle.get("plan_sha256")!=PLAN_SHA
            or bundle.get("graph_features_sha256")!=sha(GRAPH)
            or bundle.get("graph_node_map_sha256")!=sha(NODE_MAP)
            or bundle.get("deduplicated_edges_sha256")!=sha(EDGE_CATALOG)
            or bundle.get("model_inputs_file")!=detail["model_inputs_file"]
            or bundle.get("model_inputs_sha256")!=detail["model_inputs_sha256"]):
        raise AssertionError("Trial115 saved model/input contract changed")
    for key,expected in (("history_ids",history.txId),("history_times",history.time_step),
                         ("forecast_ids",future.txId),("forecast_times",future.time_step),
                         ("status",status)):
        if not np.array_equal(np.asarray(saved[key]),np.asarray(expected)):
            raise AssertionError(f"Trial115 saved population/status changed: {key}")
    if not np.array_equal(np.asarray(saved["history_labels"],float),history.label.to_numpy(float),equal_nan=True):
        raise AssertionError("Trial115 saved historical labels changed")
    if not np.array_equal(np.asarray(saved["raw_weights"]),weights):
        raise AssertionError("Trial115 saved raw status weights changed")
    pd.testing.assert_frame_equal(saved["history_features"],x,check_exact=True)
    pd.testing.assert_frame_equal(saved["forecast_features"],v,check_exact=True)
    model=bundle["model"]
    if (not isinstance(model,HistGradientBoostingClassifier)
            or any(model.get_params().get(k)!=val for k,val in plan["model_config"].items())
            or model.n_iter_!=600 or not np.array_equal(model.classes_,[0,1,2])):
        raise AssertionError("Trial115 saved HGB shape/iterations/class order changed")
    counts=np.bincount(status,minlength=3).tolist()
    masses=np.bincount(status,weights=weights,minlength=3).tolist()
    for item in (bundle,detail):
        if (item.get("status_counts")!=counts or item.get("status_weight_mass")!=masses
                or item.get("unknown_weight_factor")!=factor):
            raise AssertionError("Trial115 saved historical class counts/raw masses changed")
    if (bundle.get("history_rows")!=len(history) or bundle.get("raw_weight_sum")!=float(weights.sum())
            or detail.get("history_rows")!=len(history) or detail.get("forecast_rows")!=len(future)
            or detail.get("known_forecast_rows")!=int(future.label.notna().sum())
            or detail.get("feature_count")!=179 or detail.get("columns")!=columns
            or detail.get("class_order")!=[0,1,2] or detail.get("iterations")!=600
            or detail.get("graph_features_sha256")!=sha(GRAPH)
            or detail.get("graph_node_map_sha256")!=sha(NODE_MAP)
            or detail.get("deduplicated_edges_sha256")!=sha(EDGE_CATALOG)
            or not np.isclose(detail.get("unknown_raw_weight_share"),.4,rtol=0,atol=1e-12)):
        raise AssertionError("Trial115 fold diagnostic/model population changed")
    p=np.asarray(model.predict_proba(v),float)
    den=p[:,0]+p[:,1]
    score=np.divide(p[:,1],den,out=np.zeros(len(p)),where=den>0)
    if (p.shape!=(len(future),3) or not np.isfinite(p).all()
            or not np.allclose(p.sum(axis=1),1,rtol=0,atol=1e-12)
            or detail.get("zero_known_denominator_rows")!=int((den==0).sum())):
        raise AssertionError("Trial115 saved model three-class probability invalid")
    expected=future[["txId","time_step","label"]].copy().reset_index(drop=True)
    expected.insert(0,"origin",origin)
    for k,name in enumerate(("p_known_licit","p_known_illicit","p_unknown")):expected[name]=p[:,k]
    expected["p_known_denominator"]=den;expected[STANDALONE]=score
    pd.testing.assert_frame_equal(pd.read_pickle(forecast_path).reset_index(drop=True),expected,
                                  check_exact=False,rtol=0,atol=1e-12)
    known=expected.loc[expected.label.notna()].copy()
    if control is not None:
        ctl=control.loc[control.origin.eq(origin),KEY+[CONTROL]]
        joined=known[KEY].merge(ctl,on=KEY,how="left",validate="one_to_one",sort=False,indicator=True)
        if (len(joined)!=len(known) or not joined._merge.eq("both").all()
                or not np.array_equal(joined.txId,known.txId)):
            raise AssertionError("Trial115 public99 control key alignment changed")
        known[CONTROL]=joined[CONTROL].to_numpy(float)
        known[BLEND]=fixed_blend(known[CONTROL],known[STANDALONE])
    return known,expected


def verify_all():
    from src.combined_challengers import load_features
    plan,auth,meta,flat=check_frozen()
    train,test,sample,groups=load_features()
    _,_,edges,_=load_competition_data()
    train,test,graph_diag=verify_graph(train,test,edges,meta)
    full_columns=list(groups["regularized_two_hop"])+GRAPH5
    if len(full_columns)!=187:raise AssertionError("Trial115 historical candidate feature inventory changed")
    stages={"original":ORIGINS}
    if meta.get("confirmation_run") is True:stages["confirmation"]=CONFIRMATION
    expected={(stage,origin,stop) for stage,origins in stages.items() for origin,stop in origins.items()}
    if meta.get("candidate_created") is True:expected.add(("final",35,49))
    if len(flat)!=len(expected) or {(d.get("stage"),d.get("origin"),d.get("forecast_end")) for d in flat}!=expected:
        raise AssertionError("Trial115 original/confirmation/final fold population changed")
    details={(d["stage"],d["origin"],d["forecast_end"]):d for d in flat}
    known_parts=[];result_parts=[];summary_parts=[]
    metric_rows=replays=control_replays=0
    for stage,origins in stages.items():
        control=rebuild_public99_stage(stage,origins,train,full_columns)
        control_replays+=3*len(origins)
        fold_known=[];fold_forecast=[]
        for origin,stop in origins.items():
            known,forecast=verify_fold(stage,origin,stop,details[(stage,origin,stop)],
                                       train,test,full_columns,plan,control)
            fold_known.append(known);fold_forecast.append(forecast);replays+=1
        known=pd.concat(fold_known,ignore_index=True)
        forecast=pd.concat(fold_forecast,ignore_index=True)
        pd.testing.assert_frame_equal(pd.read_pickle(REPORTS/f"{P}_{stage}_predictions.pkl").reset_index(drop=True),
                                      known.reset_index(drop=True),check_exact=False,rtol=0,atol=1e-12)
        pd.testing.assert_frame_equal(pd.read_pickle(REPORTS/f"{P}_{stage}_forecast_probabilities.pkl").reset_index(drop=True),
                                      forecast.reset_index(drop=True),check_exact=False,rtol=0,atol=1e-12)
        result=metrics(stage,known);gate=summary(stage,result)
        compare_csv(REPORTS/f"{P}_{stage}_results.csv",result)
        compare_csv(REPORTS/f"{P}_{stage}_summary.csv",gate)
        metric_rows+=len(result)
        known_parts.append(known);result_parts.append(result);summary_parts.append(gate)
        if meta.get(f"{stage}_gate_passed") is not bool(gate.qualified.any()):
            raise AssertionError(f"Trial115 {stage} gate conclusion changed")
    pd.testing.assert_frame_equal(pd.read_pickle(REPORTS/f"{P}_predictions.pkl").reset_index(drop=True),
                                  pd.concat(known_parts,ignore_index=True).reset_index(drop=True),
                                  check_exact=False,rtol=0,atol=1e-12)
    compare_csv(REPORTS/f"{P}_results.csv",pd.concat(result_parts,ignore_index=True))
    compare_csv(REPORTS/f"{P}_summary.csv",pd.concat(summary_parts,ignore_index=True))
    eligible=(sorted(set(summary_parts[0].loc[summary_parts[0].qualified,"variant"]) &
                     set(summary_parts[1].loc[summary_parts[1].qualified,"variant"])) if len(summary_parts)==2 else [])
    winner=next((v for v in (BLEND,STANDALONE) if v in eligible),None)
    if (meta.get("two_stage_qualified_variants")!=eligible
            or meta.get("same_variant_passed_both_stages") is not bool(eligible)
            or meta.get("winner")!=winner or meta.get("candidate_created") is not bool(winner)
            or meta.get("final_model_fit_performed") is not bool(winner)):
        raise AssertionError("Trial115 strict same-variant intersection/final gate changed")
    final={"winner":winner,"candidate_created":False}
    if winner:
        _,forecast=verify_fold("final",35,49,details[("final",35,49)],train,test,full_columns,plan,None)
        replays+=1
        control=rebuild_public99_final(train,test,sample,full_columns);control_replays+=3
        expected_index=sample["index"].to_numpy()
        score=forecast[STANDALONE].to_numpy(float);blend=fixed_blend(control,score)
        candidate,components=ROOT/meta["candidate_file"],ROOT/meta["final_components_file"]
        if sha(candidate)!=meta["candidate_sha256"] or sha(components)!=meta["final_components_sha256"]:
            raise AssertionError("Trial115 final candidate/component hash changed")
        csv_candidate=pd.read_csv(candidate);frame=pd.read_pickle(components)
        chosen=blend if winner==BLEND else score
        if (csv_candidate.columns.tolist()!=["index","target"] or len(csv_candidate)!=15329
                or not np.array_equal(csv_candidate["index"],expected_index)
                or not np.isfinite(csv_candidate.target).all()
                or (csv_candidate.target<0).any() or (csv_candidate.target>1).any()
                or np.max(np.abs(csv_candidate.target.to_numpy()-chosen))>1e-12
                or not np.array_equal(frame["index"],expected_index)
                or np.max(np.abs(frame[CONTROL].to_numpy()-control))>1e-12
                or np.max(np.abs(frame[STANDALONE].to_numpy()-score))>1e-12
                or np.max(np.abs(frame[BLEND].to_numpy()-blend))>1e-12):
            raise AssertionError("Trial115 final sample-order/candidate/blend differs")
        for col in ("p_known_licit","p_known_illicit","p_unknown","p_known_denominator"):
            if np.max(np.abs(frame[col].to_numpy()-forecast[col].to_numpy()))>1e-12:
                raise AssertionError(f"Trial115 final {col} component changed")
        final={"winner":winner,"candidate_created":True,"candidate_rows":len(csv_candidate),
               "candidate_sha256":meta["candidate_sha256"]}
    elif (ROOT/"submissions"/f"candidate_{P}.csv").exists():
        raise AssertionError("Trial115 ineligible experiment wrote candidate")
    audit={"trial_number":115,"variant":P,"source_sha256":SOURCE_SHA,"plan_sha256":PLAN_SHA,
           "frozen_hashes_verified":True,"root_authorization_verified":True,
           "pinned_input_count":len(plan["input_sha256"]),
           "immutable_registry_unique_ids_preflight":PREFLIGHT_ROWS,
           "trial114_negative_audit_sha256":PRIOR114_AUDIT_SHA,
           "trial114_negative_exact_rows_verified":len(plan["trial114_exact_rows"]),
           "all_node_exact_articulation_cut_verified":True,
           "period_pure_weak_components_verified":True,
           "graph_diagnostics_verified":graph_diag,
           "saved_three_status_model_replays":replays,
           "raw_status_mass_and_cutoff_verified":True,
           "public99_control_member_replays":control_replays,
           "public99_exact_ancestry_verified":True,
           "metric_rows_recomputed":metric_rows,
           "strict_same_variant_intersection_verified":True,
           "eligible_variants":eligible,"final_artifact_checks":final,
           "independent_holdout_claim":False,"kaggle_upload":False}
    AUDIT.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    return audit


if __name__=="__main__":
    import sys
    if len(sys.argv)==2 and sys.argv[1]=="--pure-toy":pure_toy()
    elif len(sys.argv)==1:print(json.dumps(verify_all(),indent=2,sort_keys=True))
    else:raise SystemExit("Usage: verifier.py [--pure-toy]")
