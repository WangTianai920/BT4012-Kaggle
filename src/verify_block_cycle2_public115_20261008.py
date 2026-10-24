"""Independent Trial116 graph, ancestry, model, and gate audit.

Importing this module and running --pure-toy do not read competition data.
The full audit is run by the root only after the authorized fit completes.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
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
from src.horizon_hgb_20260917 import historical_columns
from src.verify_feature_propagation16_public99_20261007 import (
    rebuild_public99_stage, rebuild_public99_final,
)
from src.verify_articulation_cut2_public99_20261008 import hard_status_weights

ROOT, REPORTS = PROJECT_ROOT, REPORT_DIR
P = "block_cycle2_public115_20261008"
PREV = "articulation_cut2_public99_20261008"
GRAPH5 = ["dag_log1p_scc_size", "dag_log1p_source_depth", "dag_log1p_sink_depth",
          "dag_log1p_chain_span", "dag_chain_position"]
CUT2 = ["graph_cut_pair_disconnect_fraction", "graph_cut_dominant_remainder_loss"]
BLOCK2 = ["graph_block_log1p_largest_cycle_excess", "graph_block_cycle_membership_count"]
KEY = ["origin", "txId", "time_step", "label"]
CONTROL, STANDALONE, BLEND = "trial115_control", "block_cycle2_hgb", "blend_75_25_logodds"
ORIGINS = {14:28, 21:35, 17:24, 24:31, 28:35}
CONFIRMATION = {17:31, 18:32, 20:34}
SOURCE = ROOT / "src" / f"{P}.py"
PLAN = REPORTS / f"{P}_plan.json"
AUTH = REPORTS / f"{P}_root_authorization.json"
META = REPORTS / f"{P}_metadata.json"
FLAT = REPORTS / f"{P}_fit_diagnostics.json"
AUDIT = REPORTS / f"{P}_parent_verification.json"
GRAPH = REPORTS / f"{P}_graph_features.pkl"
NODE_MAP = REPORTS / f"{P}_graph_node_map.pkl"
GRAPH_META = REPORTS / f"{P}_graph_metadata.json"
EDGE_CATALOG = REPORTS / "terminal_coverage4_public99_20261008_deduplicated_edges.pkl"
PRIOR_MAP = REPORTS / "terminal_coverage4_public99_20261008_graph_node_map.pkl"
GRAPH5_CACHE = REPORTS / "directed_chain_position5_public55_20261005_graph_features.pkl"
CUT2_CACHE = REPORTS / f"{PREV}_graph_features.pkl"
PREV_AUDIT = REPORTS / f"{PREV}_parent_verification.json"
PREV_RESULT = REPORTS / f"{PREV}_submission_result.json"
PREV_CANDIDATE = ROOT / "submissions" / f"candidate_{PREV}.csv"
LOG = ROOT / "experiment_log.csv"
PREFLIGHT = 15631
SOURCE_SHA = "7f2d183e9c512c7986998f2175b2cd4fcf84e1d24dcaf2df3bcc7849b11d00c5"
PLAN_SHA = "c11935e24d7e322807df8c8291996af199e3033ad5634be509c569301c9e8552"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prefix_sha(path: Path, count: int) -> str:
    raw = path.read_bytes()
    with io.StringIO(raw.decode("utf-8-sig"), newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for _ in range(count):
            next(reader)
        physical_lines = reader.line_num
    return hashlib.sha256(b"".join(raw.splitlines(keepends=True)[:physical_lines])).hexdigest()


def row_sha(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def blend(control, challenger):
    a, b = np.asarray(control, float), np.asarray(challenger, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError("Trial116 blend inputs invalid")
    return expit(.75 * logit(np.clip(a, 1e-12, 1-1e-12)) +
                 .25 * logit(np.clip(b, 1e-12, 1-1e-12)))


def check_frozen():
    csv.field_size_limit(10_000_000)
    plan, auth, meta, details = [json.loads(p.read_text()) for p in (PLAN, AUTH, META, FLAT)]
    if (plan.get("trial_number") != 116 or plan.get("variant") != P
            or sha(SOURCE) != SOURCE_SHA or sha(PLAN) != PLAN_SHA
            or plan.get("source_sha256") != SOURCE_SHA
            or auth.get("authorized") is not True
            or auth.get("source_sha256") != sha(SOURCE)
            or auth.get("plan_sha256") != sha(PLAN)
            or auth.get("registry_rows") != PREFLIGHT
            or auth.get("root_trial116_block_cycle_allnode_weights_and_gates_reviewed") is not True
            or auth.get("trial115_public_result_and_closeout_acknowledged") is not True
            or auth.get("no_test_labels_or_external_data") is not True
            or meta.get("trial_number") != 116 or meta.get("variant") != P
            or meta.get("source_sha256") != sha(SOURCE) or meta.get("plan_sha256") != sha(PLAN)
            or not isinstance(details, list)):
        raise AssertionError("Trial116 frozen identity, authorization, or metadata changed")
    pins = plan.get("input_sha256", {})
    if len(pins) != 912 or "experiment_log.csv" not in pins or plan["registry_preflight"]["unique_rows"] != PREFLIGHT:
        raise AssertionError("Trial116 preflight inventory changed")
    for rel, digest in pins.items():
        actual = prefix_sha(LOG, PREFLIGHT) if rel == "experiment_log.csv" else sha(ROOT / rel)
        if actual != digest:
            raise AssertionError(f"Trial116 input changed: {rel}")
    with LOG.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    ids = [r.get("experiment_id", "") for r in rows]
    if len(rows) < PREFLIGHT or not all(ids) or len(ids) != len(set(ids)):
        raise AssertionError("Trial116 registry lost unique preflight rows")
    by_id = {r["experiment_id"]: r for r in rows}
    previous = plan.get("trial115_exact_rows", {})
    if len(previous) != 108:
        raise AssertionError("Trial115 exact row inventory changed")
    for eid, digest in previous.items():
        if eid not in by_id or row_sha(by_id[eid]) != digest:
            raise AssertionError(f"Trial115 registered row changed: {eid}")
    prior_audit = json.loads(PREV_AUDIT.read_text())
    prior_result = json.loads(PREV_RESULT.read_text())
    if (prior_audit.get("trial_number") != 115 or prior_audit.get("eligible_variants") !=
            ["articulation_cut2_hgb", "blend_75_25_logodds"]
            or prior_audit.get("saved_three_status_model_replays") != 9
            or prior_audit.get("public99_control_member_replays") != 27
            or prior_result.get("submission_name") != "1008-3"
            or prior_result.get("public_roc_auc") != .95311
            or prior_result.get("private_roc_auc") is not None
            or prior_result.get("candidate_sha256") != sha(PREV_CANDIDATE)):
        raise AssertionError("Trial115 independently audited/public control changed")
    return plan, meta, details


def reference_blocks(ids, edge_pairs):
    """NetworkX vertex-biconnected blocks; independent of production edge stack."""
    graph = nx.Graph()
    graph.add_nodes_from(int(v) for v in ids)
    graph.add_edges_from((int(a), int(b)) for a, b in edge_pairs)
    if set(graph) != set(map(int, ids)):
        raise AssertionError("Graph edge outside frozen node universe")
    largest = {int(v): 0 for v in ids}
    count = {int(v): 0 for v in ids}
    all_count = {int(v): 0 for v in ids}
    block_count = cyclic_count = bridge_count = 0
    for block in nx.biconnected_components(graph):
        size = len(block)
        if size < 2:
            raise AssertionError("Reference found degenerate block")
        block_count += 1
        cyclic = size >= 3
        cyclic_count += int(cyclic)
        bridge_count += int(not cyclic)
        for v in block:
            all_count[v] += 1
            if cyclic:
                count[v] += 1
                largest[v] = max(largest[v], size-2)
    ordered = np.asarray(ids, np.int64)
    excess = np.asarray([largest[int(v)] for v in ordered], np.int32)
    memberships = np.asarray([count[int(v)] for v in ordered], np.int32)
    all_memberships = np.asarray([all_count[int(v)] for v in ordered], np.int32)
    values = np.column_stack((np.log1p(excess), memberships)).astype(np.float32)
    return graph, values, excess, memberships, all_memberships, (block_count, cyclic_count, bridge_count)


def pure_toy():
    from src.block_cycle2_public115_20261008 import build_block_cycle_features
    from src.terminal_coverage4_public99_20261008 import build_terminal_coverage_features
    rng = np.random.default_rng(116)
    examples = [nx.path_graph(1), nx.path_graph(9), nx.cycle_graph(7), nx.star_graph(8),
                nx.complete_graph(6), nx.disjoint_union(nx.cycle_graph(5), nx.path_graph(4)),
                nx.lollipop_graph(5, 5)]
    examples += [nx.gnp_random_graph(15, .2, seed=i) for i in range(20)]
    for example in examples:
        ids = np.asarray(sorted(example), np.int64)
        directed = pd.DataFrame([(min(a,b), max(a,b)) for a,b in example.edges()],
                                columns=["txId1", "txId2"])
        anchors = pd.DataFrame({"txId": ids, "time_step": np.ones(len(ids), np.int64)})
        _, prior, catalog, _ = build_terminal_coverage_features(ids, directed, anchors)
        actual, node_map, meta = build_block_cycle_features(prior, catalog)
        graph, values, excess, count, all_count, blocks = reference_blocks(ids, example.edges())
        assert np.array_equal(actual[BLOCK2].to_numpy(), values)
        assert np.array_equal(node_map["largest_cycle_block_excess"], excess)
        assert np.array_equal(node_map["cycle_block_membership_count"], count)
        assert np.array_equal(node_map["block_membership_count"], all_count)
        assert (meta["block_count"], meta["cyclic_block_count"], meta["bridge_block_count"]) == blocks
        reversed_catalog = catalog.iloc[::-1].reset_index(drop=True)
        repeated = build_block_cycle_features(prior, reversed_catalog)[0]
        pd.testing.assert_frame_equal(actual, repeated, check_exact=True)
    graph = nx.path_graph(1501)
    ids = np.asarray(sorted(graph), np.int64)
    directed = pd.DataFrame(list(graph.edges()), columns=["txId1", "txId2"])
    anchors = pd.DataFrame({"txId": ids, "time_step": np.ones(len(ids), np.int64)})
    _, prior, catalog, _ = build_terminal_coverage_features(ids, directed, anchors)
    actual, _, meta = build_block_cycle_features(prior, catalog)
    assert meta["bridge_block_count"] == 1500 and not actual[BLOCK2].to_numpy().any()
    assert np.isclose(blend([.5], [.5])[0], .5)
    print("Trial116 independent NetworkX block toys and 1501-node chain passed")


def verify_graph(train, test, official_edges, meta):
    graph5, cut2, prior_map = [pd.read_pickle(path) for path in
                               (GRAPH5_CACHE, CUT2_CACHE, PRIOR_MAP)]
    directed = official_edges[["txId1", "txId2"]].astype(np.int64).drop_duplicates()
    directed = directed.loc[directed.txId1.ne(directed.txId2)].sort_values(
        ["txId1", "txId2"], kind="stable").reset_index(drop=True)
    catalog = pd.read_pickle(EDGE_CATALOG)
    pd.testing.assert_frame_equal(directed, catalog, check_exact=True)
    ids = prior_map.txId.to_numpy(np.int64)
    if (len(ids) != 203769 or not np.array_equal(ids, graph5.txId)
            or not np.array_equal(ids, cut2.txId)
            or len(train) + len(test) != 157101 or len(catalog) != 234355):
        raise AssertionError("Trial116 inherited all-node graph population changed")
    src = np.searchsorted(ids, catalog.txId1.to_numpy(np.int64))
    dst = np.searchsorted(ids, catalog.txId2.to_numpy(np.int64))
    directed_index = np.column_stack((src,dst)).astype(np.int64)
    if (not np.array_equal(ids[src], catalog.txId1) or
            not np.array_equal(ids[dst], catalog.txId2)):
        raise AssertionError("Directed edge catalog endpoint outside node map")
    pairs = np.unique(np.sort(directed_index, axis=1), axis=0)
    graph, values, excess, memberships, all_memberships, blocks = reference_blocks(
        ids, np.column_stack((ids[pairs[:,0]], ids[pairs[:,1]])))
    component_sets = list(nx.connected_components(graph))
    if len(component_sets) != 49 or max(map(len, component_sets)) != 7880:
        raise AssertionError("Independent weak components differ")
    for component in component_sets:
        index = np.searchsorted(ids, np.fromiter(component, np.int64))
        if (np.unique(prior_map.weak_component.to_numpy()[index]).size != 1 or
                np.unique(prior_map.period.to_numpy()[index]).size != 1):
            raise AssertionError("Weak component or period ancestry differs")
    saved, node_map, gm = pd.read_pickle(GRAPH), pd.read_pickle(NODE_MAP), json.loads(GRAPH_META.read_text())
    pd.testing.assert_frame_equal(node_map[prior_map.columns], prior_map, check_exact=True)
    expected_cols = [*prior_map.columns, "block_membership_count", "cycle_block_membership_count",
                     "largest_cycle_block_excess", *BLOCK2]
    if (node_map.columns.tolist() != expected_cols or saved.columns.tolist() != ["txId", *BLOCK2]
            or not np.array_equal(saved.txId, ids)
            or saved[BLOCK2].dtypes.astype(str).tolist() != ["float32", "float32"]
            or not np.array_equal(saved[BLOCK2].to_numpy(), values)
            or not np.array_equal(node_map[BLOCK2].to_numpy(), values)
            or not np.array_equal(node_map.block_membership_count, all_memberships)
            or not np.array_equal(node_map.cycle_block_membership_count, memberships)
            or not np.array_equal(node_map.largest_cycle_block_excess, excess)):
        raise AssertionError("Independent NetworkX blocks differ from persisted graph features")
    diagnostic = {"graph_node_count": len(ids), "weak_component_count":len(component_sets),
                  "max_component_nodes":max(map(len,component_sets)),
                  "directed_edges_input":len(catalog), "undirected_edges_after_dedup":len(pairs),
                  "deduplicated_edge_index_sha256":hashlib.sha256(directed_index.tobytes()).hexdigest(),
                  "undirected_projection_index_sha256":hashlib.sha256(pairs.tobytes()).hexdigest(),
                  "cross_period_edge_count":0, "block_count":blocks[0],
                  "cyclic_block_count":blocks[1], "bridge_block_count":blocks[2],
                  "nodes_in_cyclic_blocks":int((memberships>0).sum()),
                  "max_cycle_memberships":int(memberships.max()),
                  "max_cycle_block_excess":int(excess.max()),
                  "dfs_roots_per_component":[1]*len(component_sets)}
    by_period = {}
    for period in sorted(prior_map.period.unique()):
        arr = values[prior_map.period.to_numpy()==period].astype(float)
        item = {"node_count":len(arr), "nonzero_both_count":int(((arr[:,0]>0)&(arr[:,1]>0)).sum())}
        for j,name in enumerate(BLOCK2):
            q = np.quantile(arr[:,j], [0,.05,.5,.95,.99,1])
            item[name] = {"nonzero_count":int((arr[:,j]>0).sum()), "min":float(q[0]),
                          "p05":float(q[1]), "median":float(q[2]), "p95":float(q[3]),
                          "p99":float(q[4]), "max":float(q[5])}
        by_period[str(int(period))] = item
    diagnostic["per_period_feature_diagnostics"] = by_period
    for key,value in diagnostic.items():
        if gm.get(key) != value:
            raise AssertionError(f"Independent graph diagnostic differs: {key}")
    for key,path in (("source_sha256",SOURCE),("plan_sha256",PLAN),
                     ("graph_features_sha256",GRAPH),("graph_node_map_sha256",NODE_MAP),
                     ("deduplicated_edges_sha256",EDGE_CATALOG),
                     ("prior_node_map_sha256",PRIOR_MAP)):
        if gm.get(key) != sha(path):
            raise AssertionError(f"Trial116 graph provenance changed: {key}")
    if meta.get("graph_metadata_sha256") != sha(GRAPH_META):
        raise AssertionError("Trial116 metadata graph hash changed")
    enriched = []
    for frame in (train,test):
        ids_before = frame.txId.to_numpy()
        joined = frame.merge(graph5,on="txId",how="left",validate="one_to_one",sort=False)
        joined = joined.merge(cut2,on="txId",how="left",validate="one_to_one",sort=False)
        joined = joined.merge(saved,on="txId",how="left",validate="one_to_one",sort=False)
        if (not np.array_equal(joined.txId, ids_before) or
                joined[GRAPH5+CUT2+BLOCK2].isna().any().any()):
            raise AssertionError("Attributed all-node graph join changed")
        enriched.append(joined)
    return enriched[0], enriched[1], diagnostic


def compare_csv(path, expected):
    pd.testing.assert_frame_equal(pd.read_csv(path).reset_index(drop=True),
                                  expected.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=0, atol=1e-12)


def metrics(stage, frame):
    rows=[]
    for origin, part in frame.groupby("origin",sort=False):
        slices=[("first7",part.loc[part.time_step.le(origin+7)])]
        if int(part.time_step.max())==int(origin)+14:
            slices += [("full14",part),("second7",part.loc[part.time_step.gt(origin+7)])]
        for name, subset in slices:
            for variant in (CONTROL,STANDALONE,BLEND):
                rows.append({"stage":stage,"origin":int(origin),"slice":name,"variant":variant,
                             "n":len(subset),"positives":int(subset.label.sum()),
                             "auc":float(roc_auc_score(subset.label,subset[variant])),
                             "average_precision":float(average_precision_score(subset.label,subset[variant]))})
    return pd.DataFrame(rows)


def summary(stage, results):
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


def verify_prior_control(stage, origins, train, full_columns):
    public99 = rebuild_public99_stage(stage, origins, train, full_columns)
    saved = pd.read_pickle(REPORTS/f"{PREV}_{stage}_predictions.pkl")
    if saved[KEY].duplicated().any():
        raise AssertionError("Trial115 control has duplicate keys")
    merged = saved[KEY+["public99_control","articulation_cut2_hgb","blend_75_25_logodds"]].merge(
        public99[KEY+["public99_control"]],on=KEY,how="left",validate="one_to_one",sort=False,
        suffixes=("_saved","_replayed"),indicator=True)
    if (len(merged)!=len(saved) or not merged._merge.eq("both").all() or
            not np.array_equal(merged.txId,saved.txId) or
            np.max(np.abs(merged.public99_control_saved-merged.public99_control_replayed))>1e-12 or
            np.max(np.abs(merged.blend_75_25_logodds-
                          blend(merged.public99_control_replayed,merged.articulation_cut2_hgb)))>1e-12):
        raise AssertionError("Trial115 selected blend/Public99 ancestry changed")
    for origin,stop in origins.items():
        bundle=joblib.load(ROOT/f"models/{PREV}/{stage}_{origin}_{stop}_model.joblib")
        inputs=joblib.load(REPORTS/f"{PREV}_{stage}_{origin}_{stop}_model_inputs.joblib")
        forecast=pd.read_pickle(REPORTS/f"{PREV}_{stage}_{origin}_{stop}_forecast_probabilities.pkl")
        prob=bundle["model"].predict_proba(inputs["forecast_features"])
        score=np.divide(prob[:,1],prob[:,0]+prob[:,1],out=np.zeros(len(prob)),
                        where=(prob[:,0]+prob[:,1])>0)
        if (not np.allclose(prob,forecast[["p_known_licit","p_known_illicit","p_unknown"]],
                            rtol=0,atol=1e-12) or
                not np.allclose(score,forecast.articulation_cut2_hgb,rtol=0,atol=1e-12)):
            raise AssertionError("Trial115 saved model control replay changed")
    return merged[KEY+["blend_75_25_logodds"]].rename(columns={"blend_75_25_logodds":CONTROL})


def verify_prior_final_control(train, test, sample, full_columns):
    bundle=joblib.load(ROOT/f"models/{PREV}/final_35_49_model.joblib")
    inputs=joblib.load(REPORTS/f"{PREV}_final_35_49_model_inputs.joblib")
    forecast=pd.read_pickle(REPORTS/f"{PREV}_final_35_49_forecast_probabilities.pkl")
    prob=bundle["model"].predict_proba(inputs["forecast_features"])
    den=prob[:,0]+prob[:,1]
    score=np.divide(prob[:,1],den,out=np.zeros(len(prob)),where=den>0)
    if (not np.allclose(prob,forecast[["p_known_licit","p_known_illicit","p_unknown"]],
                        rtol=0,atol=1e-12) or
            not np.allclose(score,forecast.articulation_cut2_hgb,rtol=0,atol=1e-12)):
        raise AssertionError("Trial115 final saved model replay changed")
    public99=rebuild_public99_final(train,test,sample,full_columns)
    selected=blend(public99,score)
    candidate=pd.read_csv(PREV_CANDIDATE)
    if (candidate.columns.tolist()!=["index","target"] or
            not np.array_equal(candidate["index"],sample["index"]) or
            np.max(np.abs(candidate.target.to_numpy()-selected))>1e-12):
        raise AssertionError("Trial115 final Public99 ancestry/candidate changed")
    return selected


def verify_fold(stage, origin, stop, detail, train, test, full_columns, plan, control=None):
    history=train.loc[train.time_step.le(origin)]
    future=(test.copy() if stage=="final" else train.loc[train.time_step.between(origin+1,stop)])
    if stage=="final":
        if "label" in future and future.label.notna().any():
            raise AssertionError("Final test labels present")
        future["label"]=np.nan
    columns=historical_columns(train,full_columns,origin)+CUT2+BLOCK2
    if (len(columns)!=181 or columns[-9:-4]!=GRAPH5 or columns[-4:-2]!=CUT2 or
            columns[-2:]!=BLOCK2 or len([c for c in columns if c.startswith("feat_")])!=155):
        raise AssertionError("Trial116 historical177/cut2/block2 feature selection changed")
    status=np.where(history.label.isna().to_numpy(),2,history.label.fillna(2).to_numpy()).astype(np.int64)
    weights,factor=hard_status_weights(history.time_step,status,origin)
    if (not np.array_equal(np.unique(status),[0,1,2]) or
            not np.isclose(weights[status==2].sum()/weights.sum(),.4,rtol=0,atol=1e-12)):
        raise AssertionError("Trial116 raw 40% unknown status mass changed")
    x,v=history[columns],future[columns]
    input_path,model_path,forecast_path=[ROOT/detail[k] for k in
                                          ("model_inputs_file","model_file","forecast_file")]
    for key,path in (("model_inputs_sha256",input_path),("model_sha256",model_path),
                     ("forecast_sha256",forecast_path)):
        if sha(path)!=detail[key]:raise AssertionError(f"Trial116 saved artifact changed: {key}")
    saved,bundle=joblib.load(input_path),joblib.load(model_path)
    if (saved.get("columns")!=columns or bundle.get("columns")!=columns or
            bundle.get("configuration")!=plan["model_config"] or
            saved.get("plan_sha256")!=sha(PLAN) or bundle.get("plan_sha256")!=sha(PLAN) or
            saved.get("source_sha256")!=sha(SOURCE) or bundle.get("source_sha256")!=sha(SOURCE)):
        raise AssertionError("Trial116 saved model/input contract changed")
    for key,expected in (("history_ids",history.txId),("history_times",history.time_step),
                         ("forecast_ids",future.txId),("forecast_times",future.time_step),
                         ("status",status),("raw_weights",weights)):
        if not np.array_equal(np.asarray(saved[key]),np.asarray(expected)):
            raise AssertionError(f"Trial116 saved model population changed: {key}")
    if not np.array_equal(np.asarray(saved["history_labels"],float),history.label.to_numpy(float),equal_nan=True):
        raise AssertionError("Trial116 historical labels changed")
    pd.testing.assert_frame_equal(saved["history_features"],x,check_exact=True)
    pd.testing.assert_frame_equal(saved["forecast_features"],v,check_exact=True)
    model=bundle["model"]
    if (not isinstance(model,HistGradientBoostingClassifier) or model.n_iter_!=600 or
            not np.array_equal(model.classes_,[0,1,2]) or
            any(model.get_params().get(k)!=value for k,value in plan["model_config"].items())):
        raise AssertionError("Trial116 saved three-status HGB changed")
    masses=np.bincount(status,weights=weights,minlength=3).tolist()
    counts=np.bincount(status,minlength=3).tolist()
    for item in (bundle,detail):
        if (item.get("status_weight_mass")!=masses or item.get("status_counts")!=counts or
                item.get("unknown_weight_factor")!=factor):
            raise AssertionError("Trial116 raw class mass/count diagnostics changed")
    p=np.asarray(model.predict_proba(v),float)
    den=p[:,0]+p[:,1]
    score=np.divide(p[:,1],den,out=np.zeros(len(p)),where=den>0)
    expected=future[["txId","time_step","label"]].copy().reset_index(drop=True)
    expected.insert(0,"origin",origin)
    for j,name in enumerate(("p_known_licit","p_known_illicit","p_unknown")):expected[name]=p[:,j]
    expected["p_known_denominator"]=den;expected[STANDALONE]=score
    pd.testing.assert_frame_equal(pd.read_pickle(forecast_path).reset_index(drop=True),expected,
                                  check_exact=False,rtol=0,atol=1e-12)
    known=expected.loc[expected.label.notna()].copy()
    if control is not None:
        aligned=known[KEY].merge(control.loc[control.origin.eq(origin),KEY+[CONTROL]],on=KEY,
                                 how="left",validate="one_to_one",sort=False,indicator=True)
        if len(aligned)!=len(known) or not aligned._merge.eq("both").all():
            raise AssertionError("Trial116 historical control key alignment changed")
        known[CONTROL]=aligned[CONTROL].to_numpy(float)
        known[BLEND]=blend(known[CONTROL],known[STANDALONE])
    return known,expected


def verify_all():
    plan,meta,details=check_frozen()
    train,test,sample,groups=load_features()
    _,_,official_edges,_=load_competition_data()
    train,test,graph_diag=verify_graph(train,test,official_edges,meta)
    full_columns=list(groups["regularized_two_hop"])+GRAPH5
    if len(full_columns)!=187:raise AssertionError("Trial116 historical candidate inventory changed")
    stages={"original":ORIGINS}
    if meta.get("confirmation_run") is True:stages["confirmation"]=CONFIRMATION
    expected={(stage,origin,stop) for stage,origins in stages.items() for origin,stop in origins.items()}
    if meta.get("candidate_created") is True:expected.add(("final",35,49))
    if len(details)!=len(expected) or {(d.get("stage"),d.get("origin"),d.get("forecast_end")) for d in details}!=expected:
        raise AssertionError("Trial116 saved fold inventory changed")
    by_key={(d["stage"],d["origin"],d["forecast_end"]):d for d in details}
    known_parts=[];result_parts=[];summary_parts=[]
    metric_rows=challenger_replays=prior_replays=public99_replays=0
    for stage,origins in stages.items():
        control=verify_prior_control(stage,origins,train,full_columns)
        prior_replays+=len(origins);public99_replays+=3*len(origins)
        folds=[];forecasts=[]
        for origin,stop in origins.items():
            known,forecast=verify_fold(stage,origin,stop,by_key[(stage,origin,stop)],
                                       train,test,full_columns,plan,control)
            folds.append(known);forecasts.append(forecast);challenger_replays+=1
        known=pd.concat(folds,ignore_index=True)
        forecast=pd.concat(forecasts,ignore_index=True)
        pd.testing.assert_frame_equal(pd.read_pickle(REPORTS/f"{P}_{stage}_predictions.pkl").reset_index(drop=True),
                                      known.reset_index(drop=True),check_exact=False,rtol=0,atol=1e-12)
        pd.testing.assert_frame_equal(pd.read_pickle(REPORTS/f"{P}_{stage}_forecast_probabilities.pkl").reset_index(drop=True),
                                      forecast.reset_index(drop=True),check_exact=False,rtol=0,atol=1e-12)
        results=metrics(stage,known);gate=summary(stage,results)
        compare_csv(REPORTS/f"{P}_{stage}_results.csv",results)
        compare_csv(REPORTS/f"{P}_{stage}_summary.csv",gate)
        if meta.get(f"{stage}_gate_passed") is not bool(gate.qualified.any()):
            raise AssertionError("Trial116 gate differs from recomputed metrics")
        known_parts.append(known);result_parts.append(results);summary_parts.append(gate)
        metric_rows+=len(results)
    pd.testing.assert_frame_equal(pd.read_pickle(REPORTS/f"{P}_predictions.pkl").reset_index(drop=True),
                                  pd.concat(known_parts,ignore_index=True).reset_index(drop=True),
                                  check_exact=False,rtol=0,atol=1e-12)
    compare_csv(REPORTS/f"{P}_results.csv",pd.concat(result_parts,ignore_index=True))
    compare_csv(REPORTS/f"{P}_summary.csv",pd.concat(summary_parts,ignore_index=True))
    eligible=(sorted(set(summary_parts[0].loc[summary_parts[0].qualified,"variant"]) &
                     set(summary_parts[1].loc[summary_parts[1].qualified,"variant"])) if len(summary_parts)==2 else [])
    winner=next((v for v in (BLEND,STANDALONE) if v in eligible),None)
    if (meta.get("two_stage_qualified_variants")!=eligible or
            meta.get("same_variant_passed_both_stages") is not bool(eligible) or
            meta.get("winner")!=winner or meta.get("candidate_created") is not bool(winner) or
            meta.get("final_model_fit_performed") is not bool(winner)):
        raise AssertionError("Trial116 strict same-variant gate/final state changed")
    final={"winner":winner,"candidate_created":False}
    if winner:
        _,forecast=verify_fold("final",35,49,by_key[("final",35,49)],
                               train,test,full_columns,plan,None)
        challenger_replays+=1
        expected_index=sample["index"].to_numpy()
        control=pd.read_csv(PREV_CANDIDATE)
        replayed_control=verify_prior_final_control(train,test,sample,full_columns)
        prior_replays+=1;public99_replays+=3
        chosen=forecast[STANDALONE].to_numpy(float)
        blended=blend(control.target.to_numpy(float),chosen)
        candidate=ROOT/meta["candidate_file"]
        components=ROOT/meta["final_components_file"]
        if sha(candidate)!=meta["candidate_sha256"] or sha(components)!=meta["final_components_sha256"]:
            raise AssertionError("Trial116 candidate/component hash changed")
        output=pd.read_csv(candidate);frame=pd.read_pickle(components)
        if (output.columns.tolist()!=["index","target"] or len(output)!=15329 or
                not np.array_equal(output["index"],expected_index) or
                not np.array_equal(control["index"],expected_index) or
                not np.array_equal(frame["index"],expected_index) or
                not np.isfinite(output.target).all() or (output.target<0).any() or (output.target>1).any() or
                np.max(np.abs(output.target.to_numpy()-(blended if winner==BLEND else chosen)))>1e-12 or
                np.max(np.abs(frame[CONTROL].to_numpy()-replayed_control))>1e-12 or
                np.max(np.abs(frame[STANDALONE].to_numpy()-chosen))>1e-12 or
                np.max(np.abs(frame[BLEND].to_numpy()-blended))>1e-12):
            raise AssertionError("Trial116 final sample order/candidate/blend changed")
        final={"winner":winner,"candidate_created":True,"candidate_rows":15329,
               "candidate_sha256":meta["candidate_sha256"]}
    elif (ROOT/"submissions"/f"candidate_{P}.csv").exists():
        raise AssertionError("Ineligible Trial116 wrote a candidate")
    audit={"trial_number":116,"variant":P,"source_sha256":sha(SOURCE),"plan_sha256":sha(PLAN),
           "frozen_hashes_verified":True,"root_authorization_verified":True,
           "pinned_input_count":len(plan["input_sha256"]),
           "immutable_registry_unique_ids_preflight":PREFLIGHT,
           "trial115_exact_rows_verified":len(plan["trial115_exact_rows"]),
           "trial115_public_result_verified":True,
           "all_node_independent_biconnected_blocks_verified":True,
           "period_pure_weak_components_verified":True,
           "graph_diagnostics_verified":graph_diag,
           "saved_three_status_model_replays":challenger_replays,
           "trial115_control_model_replays":prior_replays,
           "public99_control_member_replays":public99_replays,
           "raw_status_mass_and_cutoff_verified":True,
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
