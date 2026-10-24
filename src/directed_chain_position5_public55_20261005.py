"""Prepared Trial90: deterministic SCC-condensed chain-position features.

The graph feature builder is pure and label-free. `main` requires an exact root
authorization bound to the frozen plan and Trial89 closeout before loading any
competition data, reading the edge list, fitting graph structure, or evaluating.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

from src.combined_challengers import load_features
from src.competition import DATA_DIR, PROJECT_ROOT, REPORT_DIR
from src.dual_steps_logodds_20261002 import conditional_score
from src.horizon_hgb_20260917 import historical_columns, model
from src.label_status_multiclass_20260929 import label_status
from src.lr04_increment_logodds_20261002 import CONFIRMATION, ORIGINS
from src.status_mass40_leaf31_public55_20261004 import stage_gate_pass
from src.status_mass40_lr03_iter600_20261003 import config as learner_config, training_weights_40

PREFIX = "directed_chain_position5_public55_20261005"
PLAN_PATH = REPORT_DIR / f"{PREFIX}_plan.json"
ROOT_AUTHORIZATION = REPORT_DIR / f"{PREFIX}_root_authorization.json"
TRIAL55_PREFIX = "status_mass40_logodds_blend_20261003"
TRIAL55_VARIANT = "blend_75_25_logodds"
TRIAL55_CONTROL = "public55_control"
TRIAL55_CANDIDATE = PROJECT_ROOT / "submissions" / f"candidate_{TRIAL55_PREFIX}.csv"
TRIAL55_RESULT = REPORT_DIR / f"{TRIAL55_PREFIX}_submission_result.json"
TRIAL55_META = REPORT_DIR / f"{TRIAL55_PREFIX}_metadata.json"
TRIAL55_AUDIT = REPORT_DIR / f"{TRIAL55_PREFIX}_parent_verification.json"
TRIAL89_PREFIX = "conservative_residual4_15_public55_20261004"
ABORTED_PREFIX = "directed_chain_position5_public55_20261004"
ABORT_RECORD_PATH = REPORT_DIR / f"{ABORTED_PREFIX}_engineering_abort.json"
TRIAL89_SOURCE = PROJECT_ROOT / "src" / f"{TRIAL89_PREFIX}.py"
TRIAL89_PLAN = REPORT_DIR / f"{TRIAL89_PREFIX}_plan.json"
TRIAL89_META = REPORT_DIR / f"{TRIAL89_PREFIX}_metadata.json"
TRIAL89_AUDIT = REPORT_DIR / f"{TRIAL89_PREFIX}_parent_verification.json"
TRIAL89_RESULT = REPORT_DIR / f"{TRIAL89_PREFIX}_submission_result.json"
TRIAL89_CANDIDATE = PROJECT_ROOT / "submissions" / f"candidate_{TRIAL89_PREFIX}.csv"
REGISTRY_PATH = PROJECT_ROOT / "experiment_log.csv"
MODEL_DIR = PROJECT_ROOT / "models" / PREFIX
GRAPH_FEATURES_PATH = REPORT_DIR / f"{PREFIX}_graph_features.pkl"
GRAPH_NODE_MAP_PATH = REPORT_DIR / f"{PREFIX}_graph_node_map.pkl"
GRAPH_METADATA_PATH = REPORT_DIR / f"{PREFIX}_graph_metadata.json"
CANDIDATE_PATH = PROJECT_ROOT / "submissions" / f"candidate_{PREFIX}.csv"
COMPONENTS_PATH = REPORT_DIR / f"{PREFIX}_final_components.pkl"
KEY = ["origin", "txId", "time_step", "label"]
CLIP_EPSILON = 1e-12
UNKNOWN_SHARE = 0.40
HALF_LIFE = 20.0
BASE_FEATURE_COUNT = 172
CHAIN_FEATURE_COUNT = 5
MODEL_FEATURE_COUNT = BASE_FEATURE_COUNT + CHAIN_FEATURE_COUNT
STANDALONE = "directed_chain_position5_hgb"
BLEND = "blend_75_25_logodds"
BLEND_WEIGHT = 0.25
GRAPH_FEATURE_COLUMNS = [
    "dag_log1p_scc_size",
    "dag_log1p_source_depth",
    "dag_log1p_sink_depth",
    "dag_log1p_chain_span",
    "dag_chain_position",
]
OUTPUT_PATHS = [REPORT_DIR / f"{PREFIX}_{suffix}" for suffix in (
    "graph_features.pkl", "graph_node_map.pkl", "graph_metadata.json",
    "original_predictions.pkl", "original_results.csv", "original_summary.csv",
    "confirmation_predictions.pkl", "confirmation_results.csv", "confirmation_summary.csv",
    "predictions.pkl", "results.csv", "summary.csv", "fit_diagnostics.json", "metadata.json")]


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_config() -> dict:
    return {**learner_config(), "random_seed": 42, "early_stopping": False,
            "unknown_status_mass": UNKNOWN_SHARE, "base_feature_count": BASE_FEATURE_COUNT,
            "chain_feature_count": CHAIN_FEATURE_COUNT, "total_feature_count": MODEL_FEATURE_COUNT,
            "row_triplication": False, "label_smoothing_alpha": 0.0,
            "target_classes": {"0": "known licit", "1": "known illicit",
                               "2": "unknown observation status, not pseudo-fraud"}}


def candidate_scores(control, challenger):
    a, b = np.asarray(control, dtype=float), np.asarray(challenger, dtype=float)
    if (a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all()
            or not np.isfinite(b).all() or ((a < 0) | (a > 1)).any()
            or ((b < 0) | (b > 1)).any()):
        raise ValueError("Trial55 and Trial90 scores must be aligned valid probabilities")
    blended = expit((1.0 - BLEND_WEIGHT) * logit(np.clip(a, CLIP_EPSILON, 1 - CLIP_EPSILON))
                    + BLEND_WEIGHT * logit(np.clip(b, CLIP_EPSILON, 1 - CLIP_EPSILON)))
    if not np.isfinite(blended).all() or ((blended < 0) | (blended > 1)).any():
        raise ValueError("Trial90 fixed log-odds blend is invalid")
    return b, blended


def _integer_ids(values, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional ID vector")
    try:
        as_float = arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite integer transaction IDs") from exc
    if (not np.isfinite(as_float).all() or not np.equal(as_float, np.floor(as_float)).all()
            or np.max(np.abs(as_float)) > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must contain finite int64 transaction IDs")
    return as_float.astype(np.int64)


def build_chain_position_features(edges: pd.DataFrame, node_ids) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Return deterministic SCC-condensed chain features, metadata, and node map.

    Duplicate directed edges are counted then collapsed. Self-loops are counted
    then omitted; they do not connect distinct SCCs or add path length. All
    supplied node IDs, including isolates and graph-only endpoints, are retained.
    Depths count condensed-DAG edges on longest paths from any source/to any sink.
    """
    if not {"txId1", "txId2"}.issubset(edges.columns):
        raise ValueError("Directed edge table must contain txId1 and txId2")
    raw_nodes = _integer_ids(node_ids, "node_ids")
    if np.unique(raw_nodes).size != raw_nodes.size:
        raise ValueError("node_ids must be unique")
    src = _integer_ids(edges["txId1"].to_numpy(), "txId1") if len(edges) else np.array([], dtype=np.int64)
    dst = _integer_ids(edges["txId2"].to_numpy(), "txId2") if len(edges) else np.array([], dtype=np.int64)
    if src.shape != dst.shape:
        raise ValueError("Directed edge endpoints are misaligned")
    ids = np.sort(np.unique(np.concatenate([raw_nodes, src, dst])))
    if len(ids) != len(raw_nodes):
        raise ValueError("node_ids must already include every directed edge endpoint")
    pos_src = np.searchsorted(ids, src)
    pos_dst = np.searchsorted(ids, dst)
    edge_pairs = np.column_stack((pos_src, pos_dst)).astype(np.int64, copy=False)
    input_edge_count = int(len(edge_pairs))
    unique_pairs = np.unique(edge_pairs, axis=0) if input_edge_count else np.empty((0, 2), dtype=np.int64)
    duplicate_edge_rows = input_edge_count - int(len(unique_pairs))
    self_loop_mask = unique_pairs[:, 0] == unique_pairs[:, 1] if len(unique_pairs) else np.zeros(0, dtype=bool)
    self_loop_edge_rows = int(self_loop_mask.sum())
    path_pairs = unique_pairs[~self_loop_mask]
    n = int(len(ids))
    adjacency = coo_matrix((np.ones(len(path_pairs), dtype=np.int8),
                            (path_pairs[:, 0], path_pairs[:, 1])), shape=(n, n)).tocsr()
    adjacency.sort_indices()
    component_count, labels = connected_components(adjacency, directed=True,
                                                   connection="strong", return_labels=True)
    component_min = np.full(component_count, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(component_min, labels, ids)
    canonical_order = np.argsort(component_min, kind="stable")
    canonical_labels = np.empty(component_count, dtype=np.int64)
    canonical_labels[canonical_order] = np.arange(component_count, dtype=np.int64)
    component = canonical_labels[labels]
    component_size = np.bincount(component, minlength=component_count).astype(np.int64)
    component_min = component_min[canonical_order]

    if len(path_pairs):
        condensed = np.column_stack((component[path_pairs[:, 0]], component[path_pairs[:, 1]]))
        condensed = condensed[condensed[:, 0] != condensed[:, 1]]
        condensed = np.unique(condensed, axis=0) if len(condensed) else np.empty((0, 2), dtype=np.int64)
    else:
        condensed = np.empty((0, 2), dtype=np.int64)
    dag_edges = int(len(condensed))
    successors: list[list[int]] = [[] for _ in range(component_count)]
    indegree = np.zeros(component_count, dtype=np.int64)
    for u, v in condensed:
        successors[int(u)].append(int(v))
        indegree[int(v)] += 1
    for neighbours in successors:
        neighbours.sort()
    ready = [int(c) for c in np.flatnonzero(indegree == 0)]
    heapq.heapify(ready)
    topo: list[int] = []
    degree = indegree.copy()
    while ready:
        u = heapq.heappop(ready)
        topo.append(u)
        for v in successors[u]:
            degree[v] -= 1
            if degree[v] == 0:
                heapq.heappush(ready, v)
    topo_acyclic = len(topo) == component_count
    if not topo_acyclic:
        raise RuntimeError("SCC condensation did not produce a DAG")

    source_depth = np.zeros(component_count, dtype=np.int64)
    for u in topo:
        for v in successors[u]:
            source_depth[v] = max(source_depth[v], source_depth[u] + 1)
    sink_depth = np.zeros(component_count, dtype=np.int64)
    for u in reversed(topo):
        for v in successors[u]:
            sink_depth[u] = max(sink_depth[u], sink_depth[v] + 1)
    span = source_depth + sink_depth
    position = source_depth / (span + 1.0)
    node_sizes = component_size[component]
    node_src_depth = source_depth[component]
    node_sink_depth = sink_depth[component]
    node_span = span[component]
    values = np.column_stack((np.log1p(node_sizes), np.log1p(node_src_depth),
                              np.log1p(node_sink_depth), np.log1p(node_span),
                              node_src_depth / (node_span + 1.0))).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Chain-position features contain nonfinite values")
    features = pd.DataFrame(values, columns=GRAPH_FEATURE_COLUMNS)
    features.insert(0, "txId", ids)
    node_map = pd.DataFrame({"txId": ids, "scc_id": component,
                             "scc_min_txId": component_min[component], "scc_size": node_sizes,
                             "source_depth": node_src_depth, "sink_depth": node_sink_depth,
                             "chain_span": node_span})
    meta = {
        "method": "label-free directed SCC condensation and longest-path depths",
        "uses_labels": False, "uses_tabular_feature_values": False,
        "uses_full_supplied_graph": True, "all_graph_nodes": n,
        "input_edge_rows": input_edge_count, "unique_directed_edges_before_self_loop_removal": int(len(unique_pairs)),
        "duplicate_edge_rows_collapsed": int(duplicate_edge_rows),
        "unique_self_loop_edges_omitted_from_paths": self_loop_edge_rows,
        "path_edges_after_deduplication_and_self_loop_removal": int(len(path_pairs)),
        "strongly_connected_components": int(component_count),
        "nontrivial_scc_count": int((component_size > 1).sum()),
        "largest_scc_size": int(component_size.max(initial=0)),
        "condensed_dag_edges": dag_edges, "topological_order_nodes": len(topo),
        "condensed_graph_acyclic": bool(topo_acyclic),
        "maximum_source_depth": int(source_depth.max(initial=0)),
        "maximum_sink_depth": int(sink_depth.max(initial=0)),
        "maximum_chain_span": int(span.max(initial=0)),
        "feature_columns": list(GRAPH_FEATURE_COLUMNS), "feature_count": len(GRAPH_FEATURE_COLUMNS),
        "node_ids_sorted_ascending": True,
        "depth_definition": "number of condensed-DAG edges on the longest path from any source/to any sink",
        "isolated_node_behavior": "singleton SCC with zero source and sink depths",
        "feature_formula": {
            GRAPH_FEATURE_COLUMNS[0]: "log1p SCC node count",
            GRAPH_FEATURE_COLUMNS[1]: "log1p longest source-to-SCC edge count",
            GRAPH_FEATURE_COLUMNS[2]: "log1p longest SCC-to-sink edge count",
            GRAPH_FEATURE_COLUMNS[3]: "log1p(source_depth + sink_depth)",
            GRAPH_FEATURE_COLUMNS[4]: "source_depth / (source_depth + sink_depth + 1)",
        },
    }
    return features, meta, node_map


def attach_chain_features(train: pd.DataFrame, test: pd.DataFrame,
                          features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if features["txId"].duplicated().any() or not set(["txId", *GRAPH_FEATURE_COLUMNS]).issubset(features.columns):
        raise ValueError("Chain feature artifact key or schema changed")
    outputs = []
    for frame, name in ((train, "train"), (test, "test")):
        if frame["txId"].duplicated().any():
            raise ValueError(f"{name} transaction IDs are not unique")
        enriched = frame.merge(features, on="txId", how="left", validate="one_to_one", sort=False)
        if (len(enriched) != len(frame) or not np.array_equal(enriched.txId.to_numpy(), frame.txId.to_numpy())
                or enriched[GRAPH_FEATURE_COLUMNS].isna().any().any()
                or not np.isfinite(enriched[GRAPH_FEATURE_COLUMNS].to_numpy(dtype=float)).all()):
            raise ValueError(f"{name} chain feature join changed row order or contains missing values")
        outputs.append(enriched)
    return outputs[0], outputs[1]


def verify_graph_toys() -> None:
    # Pure synthetic graphs only. No project data, edge list, model, or forecasts are read.
    edges = pd.DataFrame({"txId1": [1, 2, 3, 3, 6, 7, 7],
                          "txId2": [2, 3, 1, 4, 7, 6, 7]})
    feats, meta, node_map = build_chain_position_features(edges, [1, 2, 3, 4, 5, 6, 7])
    got = feats.set_index("txId")
    # 1-2-3 is an SCC with edge 3->4; 4 and 5 are sink/isolated components.
    expected = {1: (3, 0, 1, 0.0), 2: (3, 0, 1, 0.0), 3: (3, 0, 1, 0.0),
                4: (1, 1, 0, 1 / 2), 5: (1, 0, 0, 0.0),
                6: (2, 0, 0, 0.0), 7: (2, 0, 0, 0.0)}
    for txid, (size, src_depth, sink_depth, position) in expected.items():
        row = got.loc[txid]
        if (not np.isclose(row[GRAPH_FEATURE_COLUMNS[0]], np.log1p(size))
                or not np.isclose(row[GRAPH_FEATURE_COLUMNS[1]], np.log1p(src_depth))
                or not np.isclose(row[GRAPH_FEATURE_COLUMNS[2]], np.log1p(sink_depth))
                or not np.isclose(row[GRAPH_FEATURE_COLUMNS[4]], position)):
            raise AssertionError(f"Synthetic chain feature mismatch at transaction {txid}")
    # The hand-written 1->2->3 cycle has a single SCC; the condensation 3->4 has depth one.
    if (meta["strongly_connected_components"] != 4 or meta["nontrivial_scc_count"] != 2
            or meta["largest_scc_size"] != 3 or not meta["condensed_graph_acyclic"]
            or not np.array_equal(node_map.txId.to_numpy(), np.arange(1, 8))):
        raise AssertionError("Synthetic SCC/DAG diagnostics changed")
    # The real 187-column input (182 historical candidates + five graph columns)
    # must prune ten historical raw features and preserve all five graph columns.
    toy_values = {"time_step": np.tile(np.arange(1, 16), 2)}
    full_columns = [f"feat_{i:03d}" for i in range(182)] + list(GRAPH_FEATURE_COLUMNS)
    for i, column in enumerate(full_columns[:182]):
        toy_values[column] = np.tile(np.arange(15, dtype=float) * (i + 1), 2)
    for column in GRAPH_FEATURE_COLUMNS:
        toy_values[column] = np.zeros(30, dtype=float)
    toy_train = pd.DataFrame(toy_values)
    selected = historical_columns(toy_train, full_columns, 14)
    if len(full_columns) != 187 or len(selected) != MODEL_FEATURE_COUNT or not set(GRAPH_FEATURE_COLUMNS).issubset(selected):
        raise AssertionError("Synthetic historical_columns 187-to-177 contract changed")

    # A branched DAG checks longest paths when one path is shorter than another.
    branch_edges = pd.DataFrame({"txId1": [10, 10, 11, 12, 13], "txId2": [11, 12, 13, 13, 14]})
    branch_features, _, _ = build_chain_position_features(branch_edges, [10, 11, 12, 13, 14, 15])
    branch = branch_features.set_index("txId")
    if (not np.isclose(branch.loc[13, GRAPH_FEATURE_COLUMNS[1]], np.log1p(2))
            or not np.isclose(branch.loc[10, GRAPH_FEATURE_COLUMNS[2]], np.log1p(3))
            or not np.isclose(branch.loc[15, GRAPH_FEATURE_COLUMNS[0]], np.log1p(1))):
        raise AssertionError("Synthetic branched-DAG longest-path guard changed")

    duplicated = pd.concat([edges, edges.iloc[[0]]], ignore_index=True)
    duplicate_features, duplicate_meta, _ = build_chain_position_features(duplicated, [1, 2, 3, 4, 5, 6, 7])
    if (duplicate_meta["duplicate_edge_rows_collapsed"] != 1
            or not duplicate_features.equals(feats)):
        raise AssertionError("Synthetic duplicate-edge handling changed")


def verify_gate_logic() -> None:
    original = {"full14_mean_delta": .01, "second7_mean_delta": .01,
                "primary7_mean_delta": .01, "rolling7_mean_delta": .01, "worst_slice_delta": -.001}
    confirmation = {"full14_mean_delta": .01, "second7_mean_delta": .01,
                    "first7_mean_delta": 0., "worst_slice_delta": -.001}
    if not stage_gate_pass("original", original) or not stage_gate_pass("confirmation", confirmation):
        raise AssertionError("Trial90 strict gate contract changed")
    for key in ("full14_mean_delta", "second7_mean_delta", "primary7_mean_delta", "rolling7_mean_delta"):
        bad = dict(original); bad[key] = 0.
        if stage_gate_pass("original", bad):
            raise AssertionError(f"Original gate accepted {key}=0")
    for key in ("full14_mean_delta", "second7_mean_delta"):
        bad = dict(confirmation); bad[key] = 0.
        if stage_gate_pass("confirmation", bad):
            raise AssertionError(f"Confirmation gate accepted {key}=0")
    bad = dict(confirmation); bad["worst_slice_delta"] = -.0010001
    if stage_gate_pass("confirmation", bad):
        raise AssertionError("Confirmation gate accepted worst-slice miss")
    control = np.array([.01, .2, .5, .99]); challenger = np.array([.9, .4, .1, .8])
    standalone, blend = candidate_scores(control, challenger)
    expected = expit(.75 * logit(np.clip(control, CLIP_EPSILON, 1 - CLIP_EPSILON))
                     + .25 * logit(np.clip(challenger, CLIP_EPSILON, 1 - CLIP_EPSILON)))
    if not np.array_equal(standalone, challenger) or not np.allclose(blend, expected, rtol=0, atol=1e-15):
        raise AssertionError("Synthetic Trial90 fixed candidate formula changed")


def verify_static_contract(plan: dict) -> None:
    if (plan.get("trial_number") != 90 or plan.get("variant") != PREFIX
            or plan.get("source_sha256") != sha(PROJECT_ROOT / "src" / f"{PREFIX}.py")):
        raise ValueError("Trial90 source/plan identity mismatch")
    expected_config = frozen_config()
    if plan.get("configuration") != expected_config:
        raise ValueError("Trial90 learner or 177-feature configuration changed")
    if plan.get("graph_feature_contract") != {
        "algorithm": "SciPy strongly connected components followed by canonical SCC IDs and deterministic Kahn topological order",
        "input": "all distinct directed transaction edges in txs_edgelist.csv; no labels or tabular feature values",
        "nodes": "union of train IDs, test IDs, and all edge endpoints, sorted ascending int64",
        "cycle_handling": "condense every SCC to one node; self-loop edges omitted; duplicate directed edges collapsed",
        "depth_unit": "one edge in the SCC-condensation DAG",
        "depths": "longest edge-count path from any source and longest edge-count path to any sink",
        "features": [
            {"name": GRAPH_FEATURE_COLUMNS[0], "formula": "log1p(SCC size)"},
            {"name": GRAPH_FEATURE_COLUMNS[1], "formula": "log1p(longest source depth)"},
            {"name": GRAPH_FEATURE_COLUMNS[2], "formula": "log1p(longest sink depth)"},
            {"name": GRAPH_FEATURE_COLUMNS[3], "formula": "log1p(source depth + sink depth)"},
            {"name": GRAPH_FEATURE_COLUMNS[4], "formula": "source depth / (source depth + sink depth + 1)"},
        ],
        "output_dtype": "float32", "output_order": "ascending transaction ID",
        "missing_or_isolated_nodes": "retained; absent edges yield singleton SCC and zero source/sink depths",
        "model_input": "172 historical columns plus five graph-only chain-position columns = 177 columns",
    }:
        raise ValueError("Trial90 graph feature definition changed")
    if plan.get("candidate") != {
        "standalone_variant": STANDALONE, "blend_variant": BLEND,
        "blend_formula": "sigmoid(.75*logit(PUBLICBEST55)+.25*logit(Trial90 standalone)); input clip epsilon=1e-12",
        "blend_weight_trial90": .25, "blend_weight_public55": .75,
        "winner_preference": [BLEND, STANDALONE], "sweeps": False, "upload": False,
    }:
        raise ValueError("Trial90 fixed candidate set or winner preference changed")
    if plan.get("input_sha256", {}).get("experiment_log.csv"):
        raise ValueError("Trial90 must use exact audit-row hashes, not pin a mutable experiment log")
    if len(plan.get("input_sha256", {})) < 250:
        raise ValueError("Trial90 ancestry pins are incomplete")
    verify_graph_toys()
    verify_gate_logic()


def verify_root_authorization(plan: dict) -> dict:
    if not ROOT_AUTHORIZATION.is_file():
        raise RuntimeError("Matching Trial90 root authorization required before graph/data loading or evaluation")
    auth = json.loads(ROOT_AUTHORIZATION.read_text())
    plan_hash = sha(PLAN_PATH)
    if (auth.get("authorized") is not True or auth.get("source_sha256") != plan.get("source_sha256")
            or auth.get("plan_sha256") != plan_hash
            or auth.get("trial89_public_miss_acknowledged") is not True):
        raise RuntimeError("Root authorization does not bind the frozen Trial90 source/plan and Trial89 public miss")
    return auth


def verify_trial89_closeout(plan: dict, auth: dict) -> dict:
    contract = plan.get("trial89_closeout_contract", {})
    paths = {"source_sha256": TRIAL89_SOURCE, "plan_sha256": TRIAL89_PLAN,
             "metadata_sha256": TRIAL89_META, "parent_audit_sha256": TRIAL89_AUDIT,
             "public_result_sha256": TRIAL89_RESULT, "candidate_sha256": TRIAL89_CANDIDATE}
    hashes = {key: sha(path) for key, path in paths.items()}
    for key, digest in hashes.items():
        if contract.get(key) != digest or auth.get(f"trial89_{key}") != digest:
            raise RuntimeError(f"Root authorization does not bind Trial89 {key}")
    if (auth.get("trial89_positive_local_audit_registered") is not True
            or auth.get("trial89_registry_id") != contract.get("registry_id")
            or auth.get("trial89_public_registry_id") != contract.get("public_registry_id")
            or auth.get("trial89_registry_row_sha256") != contract.get("registry_row_sha256")
            or auth.get("trial89_public_registry_row_sha256") != contract.get("public_registry_row_sha256")
            or int(auth.get("trial89_registry_unique_ids", 0)) < int(contract.get("registry_unique_ids_snapshot", 1))):
        raise RuntimeError("Trial89 exact local-audit/public-result registrations are not root-authorized")
    meta = json.loads(TRIAL89_META.read_text())
    audit = json.loads(TRIAL89_AUDIT.read_text())
    result = json.loads(TRIAL89_RESULT.read_text())
    if (meta.get("trial_number") != 89 or meta.get("variant") != TRIAL89_PREFIX
            or meta.get("source_sha256") != hashes["source_sha256"]
            or meta.get("plan_sha256") != hashes["plan_sha256"]
            or meta.get("original_gate_passed") is not True or meta.get("confirmation_run") is not True
            or meta.get("confirmation_gate_passed") is not True
            or meta.get("same_variant_passed_both_stages") is not True
            or meta.get("qualified_variants_both_stages") != ["conservative_residual4_15_hgb_blend"]
            or meta.get("candidate_sha256") != hashes["candidate_sha256"]
            or meta.get("kaggle_upload") is not False
            or audit.get("eligible_variants") != ["conservative_residual4_15_hgb_blend"]
            or audit.get("strict_same_variant_intersection_verified") is not True
            or audit.get("original_gate_passed") is not True or audit.get("confirmation_gate_passed") is not True
            or audit.get("saved_model_replays") != 1 or audit.get("metric_rows_recomputed") != 36
            or audit.get("source_sha256") != hashes["source_sha256"]
            or audit.get("plan_sha256") != hashes["plan_sha256"]
            or audit.get("frozen_hashes_verified") is not True
            or result.get("submission_name") != "1004-3" or result.get("public_roc_auc") != 0.95229
            or result.get("incumbent_public_roc_auc") != 0.95255 or result.get("delta_vs_incumbent") != -0.00026
            or result.get("private_roc_auc") is not None or result.get("source") != "user_reported"
            or result.get("candidate_sha256") != hashes["candidate_sha256"]):
        raise RuntimeError("Trial89 local-positive/public-miss contract changed")
    csv.field_size_limit(10_000_000)
    with REGISTRY_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row.get("experiment_id", "") for row in rows]
    snapshot = int(contract["registry_unique_ids_snapshot"])
    if len(ids) != len(set(ids)) or len(set(ids)) < max(snapshot, int(auth["trial89_registry_unique_ids"])):
        raise RuntimeError("Trial89 authorization registry snapshot is not present")
    for row_id_key, hash_key, expected_source, expected_hash in (
        ("registry_id", "registry_row_sha256", f"reports/{TRIAL89_PREFIX}_parent_verification.json", hashes["parent_audit_sha256"]),
        ("public_registry_id", "public_registry_row_sha256", f"reports/{TRIAL89_PREFIX}_submission_result.json", hashes["public_result_sha256"]),
    ):
        matched = [row for row in rows if row.get("experiment_id") == contract[row_id_key]]
        row_hash = hashlib.sha256(json.dumps(matched[0], sort_keys=True, separators=(",", ":")).encode()).hexdigest() if len(matched) == 1 else None
        if (row_hash != contract[hash_key] or row_hash != auth.get(f"trial89_{hash_key}")
                or matched[0].get("study") != TRIAL89_PREFIX
                or matched[0].get("source_file") != expected_source
                or matched[0].get("source_sha256") != expected_hash):
            raise RuntimeError(f"Trial89 exact registered row changed: {row_id_key}")
    return {**hashes, "registry_id": contract["registry_id"],
            "public_registry_id": contract["public_registry_id"],
            "registry_row_sha256": contract["registry_row_sha256"],
            "public_registry_row_sha256": contract["public_registry_row_sha256"],
            "registry_unique_ids_snapshot": snapshot}


def verify_engineering_abort(plan: dict) -> dict:
    contract = plan.get("engineering_restart_contract", {})
    if contract.get("abort_record_path") != str(ABORT_RECORD_PATH.relative_to(PROJECT_ROOT)):
        raise RuntimeError("Trial90 engineering restart lineage path changed")
    abort = json.loads(ABORT_RECORD_PATH.read_text())
    preserved = abort.get("preserved_artifacts", {})
    if (abort.get("trial_number") != 90
            or abort.get("status") != "engineering_abort_before_any_auc"
            or abort.get("performance_metrics_computed") is not False
            or abort.get("candidate_created") is not False
            or abort.get("kaggle_upload") is not False
            or abort.get("design_change_for_restart") is not False
            or abort.get("saved_model_count") != 1
            or contract.get("abort_record_sha256") != sha(ABORT_RECORD_PATH)
            or contract.get("preserved_artifacts") != preserved
            or any(plan.get("input_sha256", {}).get(path) != digest for path, digest in preserved.items())
            or any(sha(PROJECT_ROOT / path) != digest for path, digest in preserved.items())):
        raise RuntimeError("Trial90 restart is not bound to the verified engineering-only abort")
    return {"abort_record_sha256": sha(ABORT_RECORD_PATH),
            "performance_metrics_computed": False, "saved_model_count": 1,
            "preserved_artifact_count": len(preserved)}


def verify_public55(plan: dict) -> None:
    control = json.loads(TRIAL55_RESULT.read_text())
    meta = json.loads(TRIAL55_META.read_text())
    audit = json.loads(TRIAL55_AUDIT.read_text())
    if (control.get("submission_name") != "1003-2" or control.get("public_roc_auc") != 0.95255
            or control.get("private_roc_auc") is not None
            or meta.get("original_gate_passed") is not True or meta.get("confirmation_run") is not True
            or meta.get("submission_created") is not True or meta.get("kaggle_upload") is not False
            or audit.get("eligible_variants") != [TRIAL55_VARIANT]
            or meta.get("candidate_sha256") != sha(TRIAL55_CANDIDATE)
            or plan.get("public55_control", {}).get("candidate_sha256") != sha(TRIAL55_CANDIDATE)):
        raise RuntimeError("Trial55 public-best control ancestry changed")


def verify_pins(plan: dict) -> None:
    for name, digest in plan["input_sha256"].items():
        if sha(PROJECT_ROOT / name) != digest:
            raise RuntimeError(f"Pinned Trial90 input changed: {name}")


def merge_and_score(trial: pd.DataFrame, control: pd.DataFrame, stage: str) -> pd.DataFrame:
    if trial.duplicated(KEY).any() or control.duplicated(KEY).any():
        raise ValueError(f"{stage} Trial90 or Trial55 keys are not unique")
    if not {STANDALONE}.issubset(trial.columns) or TRIAL55_CONTROL not in control.columns:
        raise ValueError(f"{stage} Trial90 predictions or Trial55 control schema changed")
    # Outer validation proves both populations have identical keys and cardinality.
    # `trial` is the fresh forecast and therefore has no saved control column.
    check = trial[KEY].merge(control[KEY], on=KEY, how="outer", validate="one_to_one",
                             indicator=True, sort=False)
    if (len(check) != len(trial) or len(control) != len(trial)
            or not check._merge.eq("both").all()):
        raise ValueError(f"{stage} Trial90 and Trial55 keys or row counts differ")
    joined = trial.merge(control[KEY + [TRIAL55_CONTROL]], on=KEY, how="inner",
                         validate="one_to_one", sort=False)
    if (len(joined) != len(trial)
            or not np.array_equal(joined.txId.to_numpy(), trial.txId.to_numpy())):
        raise ValueError(f"{stage} keyed control merge changed row order or cardinality")
    standalone, blend = candidate_scores(joined[TRIAL55_CONTROL], joined[STANDALONE])
    joined[STANDALONE] = standalone
    joined[BLEND] = blend
    return joined


def verify_merge_toy() -> None:
    toy = pd.DataFrame({"origin": [1, 1], "txId": [11, 12], "time_step": [2, 2], "label": [0, 1],
                        STANDALONE: [.2, .8]})
    ctrl = pd.DataFrame({"origin": [1, 1], "txId": [11, 12], "time_step": [2, 2], "label": [0, 1],
                         TRIAL55_CONTROL: [.3, .7]})
    result = merge_and_score(toy, ctrl, "toy")
    if len(result) != 2 or not result[BLEND].between(0, 1).all():
        raise AssertionError("Synthetic Trial90 keyed merge failed")


def load_control(stage: str, origins: dict[int, int]) -> pd.DataFrame:
    frame = pd.read_pickle(REPORT_DIR / f"{TRIAL55_PREFIX}_{stage}_predictions.pkl")
    if (not set(KEY + [TRIAL55_VARIANT]).issubset(frame.columns) or frame.duplicated(KEY).any()
            or set(frame.origin.unique()) != set(origins)):
        raise ValueError(f"Trial55 {stage} saved control keys changed")
    for origin, stop in origins.items():
        part = frame.loc[frame.origin.eq(origin)]
        if part.empty or not part.time_step.between(origin + 1, stop).all() or int(part.time_step.max()) != stop:
            raise ValueError(f"Trial55 {stage} saved forecast horizon changed at {origin}")
    return frame[KEY + [TRIAL55_VARIANT]].rename(columns={TRIAL55_VARIANT: TRIAL55_CONTROL})


def fit_origin(train: pd.DataFrame, full_columns: list[str], origin: int, stop: int,
               stage: str, control_frame: pd.DataFrame):
    history = train.loc[train.time_step.le(origin)]
    future = train.loc[train.time_step.between(origin + 1, stop)]
    columns = historical_columns(train, full_columns, origin)
    if len(columns) != MODEL_FEATURE_COUNT or not set(GRAPH_FEATURE_COLUMNS).issubset(columns):
        raise ValueError("Trial90 177-column cutoff-safe feature set changed")
    status = label_status(history.label)
    weights, factor = training_weights_40(history.time_step, status, origin)
    counts = np.bincount(status, minlength=3).astype(int)
    masses = np.bincount(status, weights=weights, minlength=3).astype(float)
    share = float(masses[2] / masses.sum())
    if (not np.array_equal(np.unique(status), [0, 1, 2])
            or not np.isclose(share, UNKNOWN_SHARE, rtol=0, atol=1e-12)
            or not np.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError("Trial90 observed-status classes or raw 40% status mass changed")
    fitted = model(learner_config())
    fitted.fit(history[columns], status, sample_weight=weights)
    if fitted.n_iter_ != 600:
        raise ValueError("Trial90 HGB did not reach its fixed 600 iterations")
    all_scores = conditional_score(fitted, future, columns)
    if len(all_scores) != len(future) or not np.isfinite(all_scores).all() or ((all_scores < 0) | (all_scores > 1)).any():
        raise ValueError("Trial90 conditional forecasts are invalid or misaligned")
    mask = future.label.notna().to_numpy()
    known = future.loc[mask, KEY[1:]].copy()
    known[STANDALONE] = all_scores[mask]
    known.insert(0, "origin", origin)
    model_path = MODEL_DIR / f"{stage}_{origin}_{stop}_model.joblib"
    if model_path.exists():
        raise RuntimeError("Preserve existing Trial90 model output")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": fitted, "base_columns": [c for c in columns if c not in GRAPH_FEATURE_COLUMNS],
              "chain_columns": list(GRAPH_FEATURE_COLUMNS), "columns": list(columns),
              "configuration": frozen_config(), "stage": stage, "origin": int(origin),
              "forecast_end": int(stop), "raw_hgb_weights_mean": float(np.mean(weights)),
              "original_class_counts": counts.tolist(), "original_class_weight_mass": masses.tolist(),
              "unknown_weight_share": share, "unknown_weight_factor": float(factor),
              "source_sha256": sha(PROJECT_ROOT / "src" / f"{PREFIX}.py"),
              "plan_sha256": sha(PLAN_PATH), "graph_features_sha256": sha(GRAPH_FEATURES_PATH)}
    joblib.dump(bundle, model_path)
    replay = conditional_score(joblib.load(model_path)["model"], future, columns)
    if not np.allclose(replay[mask], all_scores[mask], rtol=0, atol=1e-12):
        raise ValueError("Trial90 saved outer model replay differs")
    ctrl = control_frame.loc[control_frame.origin.eq(origin), KEY + [TRIAL55_CONTROL]]
    part = merge_and_score(known, ctrl, stage)
    diag = {"stage": stage, "origin": int(origin), "forecast_end": int(stop),
            "history_rows": int(len(history)), "future_rows": int(len(future)),
            "known_forecast_rows": int(len(known)), "history_max_step": int(history.time_step.max()),
            "base_columns": bundle["base_columns"], "chain_columns": list(GRAPH_FEATURE_COLUMNS),
            "columns": list(columns), "base_feature_count": BASE_FEATURE_COUNT,
            "chain_feature_count": CHAIN_FEATURE_COUNT, "feature_count": len(columns),
            "original_class_counts": counts.tolist(), "original_class_weight_mass": masses.tolist(),
            "raw_hgb_weights_mean": float(np.mean(weights)), "unknown_weight_share": share,
            "unknown_weight_factor": float(factor), "configuration": frozen_config(),
            "iteration_count": int(fitted.n_iter_), "model_file": str(model_path.relative_to(PROJECT_ROOT)),
            "model_sha256": sha(model_path), "graph_features_sha256": sha(GRAPH_FEATURES_PATH)}
    return part, diag


def run_stage(train, full_columns, origins: dict[int, int], stage: str):
    control = load_control(stage, origins)
    parts, diagnostics = [], []
    for origin, stop in origins.items():
        part, diag = fit_origin(train, full_columns, origin, stop, stage, control)
        expected = train.loc[train.time_step.between(origin + 1, stop) & train.label.notna(), "txId"]
        if not np.array_equal(part.txId.to_numpy(), expected.to_numpy()):
            raise ValueError(f"{stage} known validation row order changed at {origin}")
        parts.append(part); diagnostics.append(diag)
        print(f"Completed Trial90 {stage} origin {origin}->{stop}: 177 features, status mass 40%", flush=True)
    return pd.concat(parts, ignore_index=True), diagnostics


def evaluate(predictions: pd.DataFrame, stage: str) -> pd.DataFrame:
    rows = []
    for origin, frame in predictions.groupby("origin", sort=False):
        slices = {"first7": frame.loc[frame.time_step.le(origin + 7)]}
        if int(frame.time_step.max()) == int(origin) + 14:
            slices.update(full14=frame, second7=frame.loc[frame.time_step.gt(origin + 7)])
        for name, part in slices.items():
            for variant in (TRIAL55_CONTROL, STANDALONE, BLEND):
                rows.append({"stage": stage, "origin": int(origin), "slice": name,
                             "variant": variant, "n": int(len(part)), "positives": int(part.label.sum()),
                             "auc": float(roc_auc_score(part.label, part[variant])),
                             "average_precision": float(average_precision_score(part.label, part[variant]))})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, stage: str) -> pd.DataFrame:
    pivot = results.pivot(index=["origin", "slice"], columns="variant", values="auc")
    rows = []
    if stage == "original":
        groups = {"full14": [(14, "full14"), (21, "full14")],
                  "second7": [(14, "second7"), (21, "second7")],
                  "primary7": [(o, "first7") for o in (14, 21, 28)],
                  "rolling7": [(o, "first7") for o in (17, 24)]}
    else:
        groups = {"full14": [(o, "full14") for o in CONFIRMATION],
                  "second7": [(o, "second7") for o in CONFIRMATION],
                  "first7": [(o, "first7") for o in CONFIRMATION]}
    for variant in (STANDALONE, BLEND):
        delta = pivot[variant] - pivot[TRIAL55_CONTROL]
        row = {"stage": stage, "variant": variant, "control_variant": TRIAL55_CONTROL}
        for name, keys in groups.items():
            row[f"{name}_mean_auc"] = float(pivot.loc[keys, variant].mean())
            row[f"{name}_mean_delta"] = float(delta.loc[keys].mean())
            row[f"{name}_control_auc"] = float(pivot.loc[keys, TRIAL55_CONTROL].mean())
        row["worst_slice_delta"] = float(delta.min())
        row["qualified"] = bool(stage_gate_pass(stage, row))
        rows.append(row)
    return pd.DataFrame(rows)


def fit_final(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame,
              full_columns: list[str], winner: str) -> dict:
    history = train.loc[train.time_step.le(35)]
    columns = historical_columns(train, full_columns, 35)
    if len(columns) != MODEL_FEATURE_COUNT:
        raise ValueError("Trial90 final cutoff-safe feature count changed")
    status = label_status(history.label)
    weights, factor = training_weights_40(history.time_step, status, 35)
    counts = np.bincount(status, minlength=3).astype(int)
    masses = np.bincount(status, weights=weights, minlength=3).astype(float)
    share = float(masses[2] / masses.sum())
    if not np.isclose(share, UNKNOWN_SHARE, rtol=0, atol=1e-12):
        raise ValueError("Trial90 final observed-status mass differs from fixed 40%")
    fitted = model(learner_config())
    fitted.fit(history[columns], status, sample_weight=weights)
    if fitted.n_iter_ != 600:
        raise ValueError("Trial90 final HGB did not reach its fixed 600 iterations")
    standalone = conditional_score(fitted, test, columns)
    if len(standalone) != len(test) or not np.isfinite(standalone).all() or ((standalone < 0) | (standalone > 1)).any():
        raise ValueError("Trial90 final probabilities are invalid")
    model_path = MODEL_DIR / "final_through35_model.joblib"
    if model_path.exists() or COMPONENTS_PATH.exists() or CANDIDATE_PATH.exists():
        raise RuntimeError("Preserve existing Trial90 final outputs")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": fitted, "base_columns": [c for c in columns if c not in GRAPH_FEATURE_COLUMNS],
              "chain_columns": list(GRAPH_FEATURE_COLUMNS), "columns": list(columns),
              "configuration": frozen_config(), "stage": "final", "origin": 35,
              "forecast_end": "test", "raw_hgb_weights_mean": float(np.mean(weights)),
              "original_class_counts": counts.tolist(), "original_class_weight_mass": masses.tolist(),
              "unknown_weight_share": share, "unknown_weight_factor": float(factor),
              "source_sha256": sha(PROJECT_ROOT / "src" / f"{PREFIX}.py"),
              "plan_sha256": sha(PLAN_PATH), "graph_features_sha256": sha(GRAPH_FEATURES_PATH)}
    joblib.dump(bundle, model_path)
    replay = conditional_score(joblib.load(model_path)["model"], test, columns)
    if not np.allclose(replay, standalone, rtol=0, atol=1e-12):
        raise ValueError("Trial90 final saved-model replay differs")
    control = pd.read_csv(TRIAL55_CANDIDATE)
    expected = sample["index"].to_numpy()
    if (len(expected) != 15329 or not np.array_equal(test["index"].to_numpy(), expected)
            or not np.array_equal(control["index"].to_numpy(), expected)):
        raise ValueError("Trial90 sample/test/public55 candidate order changed")
    _, blend = candidate_scores(control.target.to_numpy(float), standalone)
    chosen = blend if winner == BLEND else standalone
    components = pd.DataFrame({"index": expected, TRIAL55_CONTROL: control.target.to_numpy(float),
                               STANDALONE: standalone, BLEND: blend})
    pd.DataFrame({"index": expected, "target": chosen}).to_csv(CANDIDATE_PATH, index=False)
    components.to_pickle(COMPONENTS_PATH)
    exported = pd.read_csv(CANDIDATE_PATH)
    if (len(exported) != 15329 or not np.array_equal(exported["index"].to_numpy(), expected)
            or not np.isfinite(exported.target).all() or not exported.target.between(0, 1).all()
            or not np.allclose(exported.target.to_numpy(), chosen, rtol=0, atol=1e-15)):
        raise ValueError("Trial90 exported candidate validation failed")
    diagnostic = {"stage": "final", "origin": 35, "forecast_end": "test",
                  "base_columns": bundle["base_columns"], "chain_columns": list(GRAPH_FEATURE_COLUMNS),
                  "columns": list(columns), "class_counts": counts.tolist(), "class_weight_mass": masses.tolist(),
                  "raw_hgb_weights_mean": float(np.mean(weights)), "unknown_weight_share": share,
                  "unknown_weight_factor": float(factor), "iteration_count": int(fitted.n_iter_),
                  "configuration": frozen_config(), "graph_features_sha256": sha(GRAPH_FEATURES_PATH),
                  "model_file": str(model_path.relative_to(PROJECT_ROOT)), "model_sha256": sha(model_path)}
    return {"winner": winner, "candidate_file": str(CANDIDATE_PATH.relative_to(PROJECT_ROOT)),
            "candidate_sha256": sha(CANDIDATE_PATH), "candidate_rows": len(exported),
            "final_components_file": str(COMPONENTS_PATH.relative_to(PROJECT_ROOT)),
            "final_components_sha256": sha(COMPONENTS_PATH),
            "final_model_file": str(model_path.relative_to(PROJECT_ROOT)),
            "final_model_sha256": sha(model_path), "final_fit_diagnostic": diagnostic}


def main() -> None:
    plan = json.loads(PLAN_PATH.read_text())
    verify_static_contract(plan)
    auth = verify_root_authorization(plan)
    prior89 = verify_trial89_closeout(plan, auth)
    verify_engineering_abort(plan)
    verify_public55(plan)
    verify_pins(plan)
    if (MODEL_DIR.exists() and any(MODEL_DIR.iterdir()) or any(path.exists() for path in OUTPUT_PATHS)
            or COMPONENTS_PATH.exists() or CANDIDATE_PATH.exists()):
        raise RuntimeError("Preserve pre-existing Trial90 outputs")

    # No competition feature or edge table is opened before root authorization above.
    train, test, sample, groups = load_features()
    edges = pd.read_csv(DATA_DIR / "txs_edgelist.csv")
    if len(groups["regularized_two_hop"]) != 182:
        raise ValueError("Established 182-column raw/structural feature set changed")
    all_ids = np.unique(np.concatenate([train.txId.to_numpy(), test.txId.to_numpy(),
                                        edges.txId1.to_numpy(), edges.txId2.to_numpy()]))
    graph_features, graph_meta, node_map = build_chain_position_features(edges, all_ids)
    if graph_meta["all_graph_nodes"] != 203769 or len(graph_features) != 203769:
        raise ValueError("Trial90 full graph node population differs from the frozen 203769-node contract")
    train, test = attach_chain_features(train, test, graph_features)
    graph_meta["feature_artifact_sha256"] = "PENDING_WRITE"
    graph_meta["source_sha256"] = sha(PROJECT_ROOT / "src" / f"{PREFIX}.py")
    graph_meta["plan_sha256"] = sha(PLAN_PATH)
    graph_features.to_pickle(GRAPH_FEATURES_PATH)
    node_map.to_pickle(GRAPH_NODE_MAP_PATH)
    graph_meta["feature_artifact_sha256"] = sha(GRAPH_FEATURES_PATH)
    graph_meta["node_map_sha256"] = sha(GRAPH_NODE_MAP_PATH)
    GRAPH_METADATA_PATH.write_text(json.dumps(graph_meta, indent=2, sort_keys=True) + "\n")
    full_columns = list(groups["regularized_two_hop"]) + list(GRAPH_FEATURE_COLUMNS)
    if len(full_columns) != 187:
        raise ValueError("Trial90 historical_columns must receive 187 candidates before cutoff-safe pruning")
    diagnostics = [{"stage": "graph_feature_build", **graph_meta}]
    original, original_diag = run_stage(train, full_columns, ORIGINS, "original")
    original_results = evaluate(original, "original")
    original_summary = summarize(original_results, "original")
    original.to_pickle(REPORT_DIR / f"{PREFIX}_original_predictions.pkl")
    original_results.to_csv(REPORT_DIR / f"{PREFIX}_original_results.csv", index=False)
    original_summary.to_csv(REPORT_DIR / f"{PREFIX}_original_summary.csv", index=False)
    diagnostics.extend(original_diag)
    result_frames = [original_results]
    summary_frames = [original_summary]
    metadata = {"trial_number": 90, "variant": PREFIX,
                "source_sha256": sha(PROJECT_ROOT / "src" / f"{PREFIX}.py"),
                "plan_sha256": sha(PLAN_PATH), "input_sha256": plan["input_sha256"],
                "trial89_closeout": prior89, "graph_features_sha256": sha(GRAPH_FEATURES_PATH),
                "graph_metadata_sha256": sha(GRAPH_METADATA_PATH),
                "original_gate_passed": bool(original_summary.qualified.any()),
                "confirmation_run": False, "confirmation_gate_passed": False,
                "same_variant_passed_both_stages": False, "qualified_variants_both_stages": [],
                "submission_created": False, "final_model_fit_performed": False,
                "kaggle_upload": False, "fit_diagnostics": diagnostics}
    if metadata["original_gate_passed"]:
        confirmation, confirmation_diag = run_stage(train, full_columns, CONFIRMATION, "confirmation")
        confirmation_results = evaluate(confirmation, "confirmation")
        confirmation_summary = summarize(confirmation_results, "confirmation")
        confirmation.to_pickle(REPORT_DIR / f"{PREFIX}_confirmation_predictions.pkl")
        confirmation_results.to_csv(REPORT_DIR / f"{PREFIX}_confirmation_results.csv", index=False)
        confirmation_summary.to_csv(REPORT_DIR / f"{PREFIX}_confirmation_summary.csv", index=False)
        result_frames.append(confirmation_results); summary_frames.append(confirmation_summary)
        diagnostics.extend(confirmation_diag)
        metadata["confirmation_run"] = True
        metadata["confirmation_gate_passed"] = bool(confirmation_summary.qualified.any())
        winners = set(original_summary.loc[original_summary.qualified, "variant"]) & set(
            confirmation_summary.loc[confirmation_summary.qualified, "variant"])
        metadata["qualified_variants_both_stages"] = sorted(winners)
        metadata["same_variant_passed_both_stages"] = bool(winners)
        if winners:
            winner = next(name for name in (BLEND, STANDALONE) if name in winners)
            final = fit_final(train, test, sample, full_columns, winner)
            diagnostics.append(final.pop("final_fit_diagnostic")); metadata.update(final)
            metadata["submission_created"] = True
            metadata["final_model_fit_performed"] = True
    metadata["fit_diagnostics"] = diagnostics
    pd.concat(result_frames, ignore_index=True).to_csv(REPORT_DIR / f"{PREFIX}_results.csv", index=False)
    pd.concat(summary_frames, ignore_index=True).to_csv(REPORT_DIR / f"{PREFIX}_summary.csv", index=False)
    (REPORT_DIR / f"{PREFIX}_fit_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    (REPORT_DIR / f"{PREFIX}_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: metadata[key] for key in (
        "original_gate_passed", "confirmation_run", "confirmation_gate_passed",
        "same_variant_passed_both_stages", "submission_created")}, indent=2))


if __name__ == "__main__":
    main()
