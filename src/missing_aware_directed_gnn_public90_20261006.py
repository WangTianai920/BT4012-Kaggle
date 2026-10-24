"""Prepared Trial98: missing-aware directed GraphSAGE over Trial90 graph5/177 inputs.

The controller never imports Torch. Each fit runs sequentially in an isolated
worker process so Torch/OpenMP cannot share a runtime with tree libraries.
No data, graph, scaler, or model access occurs until the exact root authorization
and all predecessor closeout checks pass.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

from src.combined_challengers import load_features
from src.competition import DATA_DIR, PROJECT_ROOT, REPORT_DIR
from src.directed_chain_position5_public55_20261005 import GRAPH_FEATURE_COLUMNS as GRAPH5
from src.horizon_hgb_20260917 import historical_columns
from src.label_status_multiclass_20260929 import label_status
from src.lr04_increment_logodds_20261002 import CONFIRMATION, ORIGINS
from src.status_mass40_lr03_iter600_20261003 import training_weights_40

PREFIX = "missing_aware_directed_gnn_public90_20261006"
PLAN_PATH = REPORT_DIR / f"{PREFIX}_plan.json"
ROOT_AUTH_PATH = REPORT_DIR / f"{PREFIX}_root_authorization.json"
MODEL_DIR = PROJECT_ROOT / "models" / PREFIX
GRAPH5_CACHE = REPORT_DIR / "directed_chain_position5_public55_20261005_graph_features.pkl"
GRAPH_NODE_MAP = REPORT_DIR / "directed_chain_position5_public55_20261005_graph_node_map.pkl"
TRIAL90_PREFIX = "directed_chain_position5_public55_20261005"
TRIAL90_CANDIDATE = PROJECT_ROOT / "submissions" / f"candidate_{TRIAL90_PREFIX}.csv"
TRIAL90_COMPONENTS = REPORT_DIR / f"{TRIAL90_PREFIX}_final_components.pkl"
TRIAL90_META = REPORT_DIR / f"{TRIAL90_PREFIX}_metadata.json"
TRIAL90_AUDIT = REPORT_DIR / f"{TRIAL90_PREFIX}_parent_verification.json"
TRIAL90_PLAN = REPORT_DIR / f"{TRIAL90_PREFIX}_plan.json"
TRIAL97_PREFIX = "chain_position_feature80_public90_20261005"
TRIAL97_SOURCE = PROJECT_ROOT / "src" / f"{TRIAL97_PREFIX}.py"
TRIAL97_PLAN = REPORT_DIR / f"{TRIAL97_PREFIX}_plan.json"
TRIAL97_META = REPORT_DIR / f"{TRIAL97_PREFIX}_metadata.json"
TRIAL97_AUDIT = REPORT_DIR / f"{TRIAL97_PREFIX}_parent_verification.json"
TRIAL97_RESULT = REPORT_DIR / f"{TRIAL97_PREFIX}_submission_result.json"
TRIAL97_CANDIDATE = PROJECT_ROOT / "submissions" / f"candidate_{TRIAL97_PREFIX}.csv"
TRANSFER_PREFIX = "public_transfer_diagnostic_20261006"
TRANSFER_REPORT = REPORT_DIR / f"{TRANSFER_PREFIX}_report.json"
REGISTRY_PATH = PROJECT_ROOT / "experiment_log.csv"
PUBLIC_RESULT_ID = f"S97_{TRIAL97_PREFIX}_public_1005_3"
PUBLIC_ALIAS_ID = f"S97_{TRIAL97_PREFIX}_public_1005_2"
PREPARED_REGISTRY_ROWS = 14305
KEY = ["origin", "txId", "time_step", "label"]
CONTROL = "public90_control"
CHALLENGER = "missing_aware_directed_gnn"
BLEND = "blend_75_25_logodds"
CLIP_EPSILON = 1e-12
UNKNOWN_SHARE = 0.40
HALF_LIFE = 20.0
SEED = 42
EPOCHS = 60
WIDTH = 32
ATTRIBUTE_MASK_RATE = 0.25
GRAPH_NODE_COUNT = 203769
WEAK_COMPONENT_COUNT = 49
BASE_FULL_COUNT = 182
BASE_SELECTED_COUNT = 172
GRAPH_FEATURE_COUNT = 5
MODEL_INPUT_COUNT = 177

STAGE_PREDICTION_PATHS = {
    "original": REPORT_DIR / f"{TRIAL90_PREFIX}_original_predictions.pkl",
    "confirmation": REPORT_DIR / f"{TRIAL90_PREFIX}_confirmation_predictions.pkl",
}
OUTPUT_SUFFIXES = (
    "original_predictions.pkl", "original_results.csv", "original_summary.csv",
    "confirmation_predictions.pkl", "confirmation_results.csv", "confirmation_summary.csv",
    "predictions.pkl", "results.csv", "summary.csv", "fit_diagnostics.json", "metadata.json",
    "final_components.pkl", "submission_manifest.json",
    "original_forecast_probabilities.pkl", "confirmation_forecast_probabilities.pkl",
)


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config() -> dict:
    return {
        "architecture": "one directed gated-mean GraphSAGE layer, separate self/incoming/outgoing transforms and gates",
        "hidden_width": WIDTH, "message_layers": 1, "classifier_classes": 3,
        "input_attributes": MODEL_INPUT_COUNT, "input_availability_masks": MODEL_INPUT_COUNT,
        "input_dimension": 2 * MODEL_INPUT_COUNT, "dropout": 0.1,
        "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.001,
        "epochs": EPOCHS, "early_stopping": False, "seed": SEED,
        "half_life": HALF_LIFE, "unknown_status_mass": UNKNOWN_SHARE,
        "attribute_mask_rate_per_epoch": ATTRIBUTE_MASK_RATE,
        "mask_scope": "one Bernoulli(0.25) draw per attributed historical row per epoch masks its entire 172-column base block; graph5 never masked",
        "target_classes": {"0": "known licit", "1": "known illicit", "2": "unknown observation status"},
        "score": "p(known illicit)/(p(known licit)+p(known illicit))",
        "standardization": "StandardScaler fitted only to historical attributed rows through each cutoff; clip +/-5",
        "input_contract": "172 cutoff-safe historical columns from 182 candidates plus exact frozen Trial90 graph5 = 177",
        "graph_node_count": GRAPH_NODE_COUNT, "weak_component_count": WEAK_COMPONENT_COUNT,
        "graph_only_policy": "include all graph-only nodes with base value zero, base availability zero, learned missing token, and standardized graph5",
        "sparse_projection": "exact linear factorization; dense 172-value and 172-mask projections only for attributed nodes; shared missing-token projection for graph-only nodes; graph5 and its 5 availability bits for all nodes",
        "training_graph": "only components with inferred period <= cutoff; no future nodes/features/messages/loss",
        "inference_graph": "only feature and structure rows for the independent forecast period window; no fraud labels",
        "component_period": "infer from supplied train/test time_step; assert every weak component has one period",
        "period_graphs": "components are period-homogeneous; reject any cross-period edge",
        "torch_isolation": "one sequential subprocess per fit; controller imports no torch",
    }


def candidate_scores(control, challenger):
    a, b = np.asarray(control, dtype=float), np.asarray(challenger, dtype=float)
    if (a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all()
            or not np.isfinite(b).all() or ((a < 0) | (a > 1)).any()
            or ((b < 0) | (b > 1)).any()):
        raise ValueError("Trial90 and Trial98 scores must be aligned finite probabilities")
    blended = expit(.75 * logit(np.clip(a, CLIP_EPSILON, 1-CLIP_EPSILON))
                    + .25 * logit(np.clip(b, CLIP_EPSILON, 1-CLIP_EPSILON)))
    return b, blended


def stage_gate_pass(stage: str, row: dict) -> bool:
    if stage == "original":
        required = ("full14", "second7", "primary7", "rolling7")
    elif stage == "confirmation":
        required = ("full14", "second7")
    else:
        raise ValueError(f"Unknown stage {stage}")
    if any(float(row.get(f"{name}_mean_delta", float("-inf"))) <= 0 for name in required):
        return False
    if stage == "confirmation" and float(row.get("first7_mean_delta", float("-inf"))) < 0:
        return False
    return float(row.get("worst_slice_delta", float("-inf"))) >= -0.001


def select_winner(original: pd.DataFrame, confirmation: pd.DataFrame) -> tuple[str | None, list[str]]:
    a = set(original.loc[original.qualified, "variant"])
    b = set(confirmation.loc[confirmation.qualified, "variant"])
    intersection = a & b
    for preferred in (BLEND, CHALLENGER):
        if preferred in intersection:
            return preferred, sorted(intersection)
    return None, sorted(intersection)


def weak_component_period_map(graph_ids, edges: pd.DataFrame, attributed_time: pd.DataFrame, expected_components: int = WEAK_COMPONENT_COUNT):
    """Fit-free topology helper; caller supplies data only after root authorization."""
    ids = np.asarray(graph_ids, dtype=np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or not np.array_equal(ids, np.sort(ids)):
        raise ValueError("Graph node IDs must be unique, sorted int64 values")
    loc = pd.Index(ids)
    if not {"txId1", "txId2"}.issubset(edges.columns):
        raise ValueError("Directed edges lack endpoint columns")
    src = loc.get_indexer(edges.txId1.to_numpy(dtype=np.int64))
    dst = loc.get_indexer(edges.txId2.to_numpy(dtype=np.int64))
    if (src < 0).any() or (dst < 0).any():
        raise ValueError("Frozen graph node universe omits an edge endpoint")
    pairs = np.unique(np.column_stack((src, dst)), axis=0)
    undirected = coo_matrix((np.ones(2*len(pairs), dtype=np.int8),
        (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
        shape=(len(ids), len(ids))).tocsr()
    n_components, labels = connected_components(undirected, directed=False, return_labels=True)
    if n_components != expected_components:
        raise ValueError(f"Expected {expected_components} weak components; got {n_components}")
    attr_ids = attributed_time.txId.to_numpy(dtype=np.int64)
    attr_time = attributed_time.time_step.to_numpy(dtype=np.int16)
    if attributed_time.txId.duplicated().any():
        raise ValueError("Attributed time map contains duplicate transaction IDs")
    attr_pos = loc.get_indexer(attr_ids)
    if (attr_pos < 0).any():
        raise ValueError("An attributed transaction is missing from the frozen graph universe")
    periods = np.full(n_components, -1, dtype=np.int16)
    for component, period in zip(labels[attr_pos], attr_time):
        prior = periods[component]
        if prior not in (-1, period):
            raise ValueError("A weak component spans more than one supplied time period")
        periods[component] = period
    if (periods < 0).any():
        raise ValueError("Every weak component must have an attributed train/test time-step anchor")
    return labels.astype(np.int32), periods[labels].astype(np.int16), {
        "weak_component_count": int(n_components), "component_period_count": int(len(np.unique(periods))),
        "unanchored_component_count": int((periods < 0).sum()), "node_count": int(len(ids)),
        "cross_period_edge_count": 0,
    }


def mean_adjacencies(node_ids, period_ids, edges: pd.DataFrame):
    """Return row-normalized incoming/outgoing sparse means for a period slice."""
    ids = np.asarray(node_ids, dtype=np.int64)
    periods = np.asarray(period_ids)
    if ids.ndim != 1 or periods.shape != ids.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("Adjacency node IDs/periods are not aligned and unique")
    loc = pd.Index(ids)
    src = loc.get_indexer(edges.txId1.to_numpy(dtype=np.int64))
    dst = loc.get_indexer(edges.txId2.to_numpy(dtype=np.int64))
    keep = (src >= 0) & (dst >= 0) & (src != dst)
    src, dst = src[keep], dst[keep]
    if len(src) and (periods[src] != periods[dst]).any():
        raise ValueError("Future/history graph has an edge crossing time periods")
    pairs = np.unique(np.column_stack((src, dst)), axis=0) if len(src) else np.empty((0, 2), dtype=np.int64)
    # Incoming: target row, source column. Outgoing: source row, target column.
    in_count = np.bincount(pairs[:, 1], minlength=len(ids)) if len(pairs) else np.zeros(len(ids), dtype=int)
    out_count = np.bincount(pairs[:, 0], minlength=len(ids)) if len(pairs) else np.zeros(len(ids), dtype=int)
    incoming = coo_matrix((1/np.maximum(in_count[pairs[:, 1]], 1), (pairs[:, 1], pairs[:, 0])),
                          shape=(len(ids), len(ids))).tocsr() if len(pairs) else csr_matrix((len(ids),len(ids)))
    outgoing = coo_matrix((1/np.maximum(out_count[pairs[:, 0]], 1), (pairs[:, 0], pairs[:, 1])),
                          shape=(len(ids), len(ids))).tocsr() if len(pairs) else csr_matrix((len(ids),len(ids)))
    return incoming.astype(np.float32), outgoing.astype(np.float32), int(len(pairs))


def verify_factorized_projection_toy() -> None:
    """Prove sparse graph-only input projection equals the dense token construction."""
    rng = np.random.default_rng(SEED)
    n_attr, n_graph_only, base, graph, width = 4, 2, 3, 2, 5
    x = rng.normal(size=(n_attr, base)); mask = rng.integers(0, 2, size=(n_attr, base)).astype(float)
    token = rng.normal(size=base); g = rng.normal(size=(n_attr+n_graph_only, graph))
    wg = rng.normal(size=(graph*2, width)); wv = rng.normal(size=(base, width)); wm = rng.normal(size=(base, width)); bias = rng.normal(size=width)
    x_eff = np.where(mask == 1, x, token)
    dense_values = np.concatenate([x_eff, np.repeat(token[None, :], n_graph_only, axis=0)], axis=0)
    dense_mask = np.concatenate([mask, np.zeros((n_graph_only, base))], axis=0)
    dense = np.concatenate([dense_values, dense_mask,
                           g, np.ones_like(g)], axis=1)
    wdense = np.concatenate([wv, wm, wg], axis=0)
    expected = dense @ wdense + bias
    fact = np.empty_like(expected)
    fact[:n_attr] = x_eff @ wv + mask @ wm + np.concatenate([g[:n_attr], np.ones_like(g[:n_attr])], axis=1) @ wg + bias
    # A featureless graph node gets the shared learned missing-value token.
    fact[n_attr:] = token @ wv + np.concatenate([g[n_attr:], np.ones_like(g[n_attr:])], axis=1) @ wg + bias
    if not np.allclose(expected, fact, rtol=0, atol=1e-12):
        raise AssertionError("Factorized projection differs from dense learned-token input")


def verify_graph_toys() -> None:
    edges = pd.DataFrame({"txId1": [1, 2, 3], "txId2": [2, 1, 4]})
    ids = np.arange(1, 6, dtype=np.int64)
    attr = pd.DataFrame({"txId": ids, "time_step": [1, 1, 2, 2, 3]})
    labels, period, meta = weak_component_period_map(ids, edges, attr, expected_components=3)
    if meta["weak_component_count"] != 3 or period.tolist() != [1, 1, 2, 2, 3]:
        raise AssertionError("Synthetic component-to-period propagation changed")
    ai, ao, nedge = mean_adjacencies(ids[:4], period[:4], edges)
    if nedge != 3 or not np.isclose(ai[1, 0], 1) or not np.isclose(ao[0, 1], 1):
        raise AssertionError("Synthetic directed means lost incoming/outgoing direction")
    try:
        weak_component_period_map(ids, edges, pd.DataFrame({"txId": [1,2,3,4,5], "time_step": [1,2,2,2,3]}), expected_components=3)
    except ValueError as exc:
        if "spans more than one" not in str(exc):
            raise
    else:
        raise AssertionError("Period-mixed weak component passed")
    try:
        mean_adjacencies(ids[:4], np.array([1,1,2,3]), edges)
    except ValueError as exc:
        if "crossing time periods" not in str(exc):
            raise
    else:
        raise AssertionError("Cross-period edge passed the graph boundary")
    verify_factorized_projection_toy()


def verify_weight_score_gate_toys() -> None:
    status = np.array([0, 1, 2, 2], dtype=np.int8)
    steps = np.array([10, 11, 12, 13])
    weights, _ = training_weights_40(pd.Series(steps), status, cutoff=13)
    if not np.isclose(weights[status == 2].sum()/weights.sum(), UNKNOWN_SHARE, atol=1e-12):
        raise AssertionError("Synthetic unknown status mass drifted")
    p = np.array([[.8, .1, .1], [.2, .7, .1]])
    q = p[:, 1] / (p[:, 0] + p[:, 1])
    if not np.allclose(q, [1/9, 7/9], rtol=0, atol=1e-15):
        raise AssertionError("Conditional fraud score changed")
    good_o = {"full14_mean_delta":.01,"second7_mean_delta":.01,"primary7_mean_delta":.01,
              "rolling7_mean_delta":.01,"worst_slice_delta":-.001}
    good_c = {"full14_mean_delta":.01,"second7_mean_delta":.01,"first7_mean_delta":0.,"worst_slice_delta":-.001}
    if not stage_gate_pass("original",good_o) or not stage_gate_pass("confirmation",good_c):
        raise AssertionError("Synthetic passing gates failed")
    bad = dict(good_c); bad["first7_mean_delta"]=-1e-12
    if stage_gate_pass("confirmation",bad): raise AssertionError("Negative first7 passed")
    if select_winner(pd.DataFrame({"variant":[CHALLENGER,BLEND],"qualified":[True,True]}),
                     pd.DataFrame({"variant":[CHALLENGER,BLEND],"qualified":[True,True]}))[0] != BLEND:
        raise AssertionError("Predeclared blend-first tie-break changed")
    candidate_scores(np.array([.1,.9]),np.array([.3,.7]))


def verify_trial97_closeout(plan: dict) -> None:
    meta = json.loads(TRIAL97_META.read_text())
    audit = json.loads(TRIAL97_AUDIT.read_text())
    result = json.loads(TRIAL97_RESULT.read_text())
    if (meta.get("original_gate_passed") is not True or meta.get("confirmation_gate_passed") is not True
            or meta.get("same_variant_passed_both_stages") is not True or meta.get("winner") != BLEND
            or meta.get("candidate_sha256") != sha(TRIAL97_CANDIDATE)
            or result.get("submission_name") != "1005-3" or result.get("public_roc_auc") != .95236
            or result.get("candidate_sha256") != sha(TRIAL97_CANDIDATE)
            or audit.get("frozen_hashes_verified") is not True
            or audit.get("strict_same_variant_intersection_verified") is not True
            or audit.get("final_artifact_checks", {}).get("winner") != BLEND
            or audit.get("pinned_input_count") != 406):
        raise ValueError("Trial97 public miss or independent closeout is not verified")
    if plan.get("trial97_closeout", {}).get("public_result_id") != PUBLIC_RESULT_ID:
        raise ValueError("Trial97 canonical public registry ID changed")
    csv.field_size_limit(10_000_000)
    with REGISTRY_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    ids = [r.get("experiment_id", "") for r in rows]
    if len(rows) != PREPARED_REGISTRY_ROWS or len(ids) != len(set(ids)):
        raise ValueError("Registry snapshot changed since Trial98 preparation")
    records = {r["experiment_id"]: r for r in rows if r.get("experiment_id") in (PUBLIC_RESULT_ID, PUBLIC_ALIAS_ID)}
    if set(records) != {PUBLIC_RESULT_ID, PUBLIC_ALIAS_ID}:
        raise ValueError("Both original Trial97 public ID and corrected alias must be retained")
    canonical, alias = records[PUBLIC_RESULT_ID], records[PUBLIC_ALIAS_ID]
    expected_details = json.loads(canonical["details_json"])
    if (canonical.get("variant") != "1005-3" or expected_details.get("public_roc_auc") != .95236
            or expected_details.get("incumbent_public_roc_auc") != .95273
            or expected_details.get("candidate_sha256") != sha(TRIAL97_CANDIDATE)
            or json.loads(alias["details_json"]) != expected_details):
        raise ValueError("Trial97 exact canonical public result/alias payload changed")
    expected_row_hash = plan.get("trial97_closeout", {}).get("canonical_public_registry_row_sha256")
    if _row_sha(canonical) != expected_row_hash:
        raise ValueError("Trial97 canonical public registry row hash changed")
    if _row_sha(alias) != plan.get("trial97_closeout", {}).get("public_alias_registry_row_sha256"):
        raise ValueError("Trial97 public alias registry row hash changed")


def verify_trial90_control() -> None:
    meta=json.loads(TRIAL90_META.read_text()); audit=json.loads(TRIAL90_AUDIT.read_text())
    if (meta.get("original_gate_passed") is not True or meta.get("confirmation_gate_passed") is not True
            or meta.get("same_variant_passed_both_stages") is not True or meta.get("winner") != BLEND
            or meta.get("candidate_sha256") != sha(TRIAL90_CANDIDATE)
            or audit.get("frozen_hashes_verified") is not True
            or audit.get("strict_same_variant_intersection_verified") is not True
            or audit.get("eligible_variants") != [BLEND]):
        raise ValueError("Trial90 public-best control did not pass its audited gates")
    if sha(GRAPH5_CACHE) != audit.get("graph_features_sha256"):
        raise ValueError("Frozen Trial90 graph5 cache does not match the parent audit")
    plan97=json.loads(TRIAL97_PLAN.read_text()); contract=plan97.get("trial90_closeout_contract",{})
    csv.field_size_limit(10_000_000)
    with REGISTRY_PATH.open(newline="",encoding="utf-8-sig") as handle:
        registry={r["experiment_id"]:r for r in csv.DictReader(handle)}
    for id_key,hash_key in (("registry_id","registry_row_sha256"),("public_registry_id","public_registry_row_sha256")):
        row=registry.get(contract.get(id_key))
        if row is None or _row_sha(row)!=contract.get(hash_key):
            raise ValueError("Trial90 canonical audit/public registry row changed")
    transfer=json.loads(TRANSFER_REPORT.read_text())
    if (transfer.get("diagnostic_id")!=TRANSFER_PREFIX or transfer.get("model_fits")!=0
            or transfer.get("uses_test_labels") is not False
            or transfer.get("public_results_user_reported",{}).get("97",{}).get("public_roc_auc")!=.95236):
        raise ValueError("Frozen local/public transfer diagnostic changed")


def _row_sha(row: dict) -> str:
    # DictReader preserves the registered column order when iterated by header.
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()


def verify_frozen_inputs(plan: dict) -> None:
    if plan.get("trial_number") != 98 or plan.get("variant") != PREFIX or plan.get("configuration") != config():
        raise ValueError("Trial98 frozen plan/configuration mismatch")
    if sha(PROJECT_ROOT / "src" / f"{PREFIX}.py") != plan.get("source_sha256"):
        raise ValueError("Trial98 source changed after freeze")
    for name, digest in plan.get("input_sha256", {}).items():
        if sha(PROJECT_ROOT / name) != digest:
            raise ValueError(f"Frozen Trial98 input changed: {name}")
    verify_graph_toys(); verify_weight_score_gate_toys(); verify_trial90_control(); verify_trial97_closeout(plan)
    if not ROOT_AUTH_PATH.exists():
        raise RuntimeError("Exact root authorization is required before reading project data or graph")
    auth = json.loads(ROOT_AUTH_PATH.read_text())
    if (auth.get("authorized") is not True or auth.get("source_sha256") != plan["source_sha256"]
            or auth.get("plan_sha256") != sha(PLAN_PATH)
            or auth.get("registry_rows") != PREPARED_REGISTRY_ROWS
            or auth.get("trial97_public_miss_acknowledged") is not True
            or auth.get("trial90_positive_audit_acknowledged") is not True
            or auth.get("transfer_diagnostic_reviewed") is not True):
        raise RuntimeError("Trial98 root authorization does not match the frozen preparation")


def assert_clean_outputs() -> None:
    existing = [str(p) for p in [*(REPORT_DIR / f"{PREFIX}_{s}" for s in OUTPUT_SUFFIXES),
                                  MODEL_DIR, PROJECT_ROOT / "submissions" / f"candidate_{PREFIX}.csv"] if p.exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite Trial98 outputs: {existing}")


def stage_control(stage: str, origins: dict[int, int]) -> pd.DataFrame:
    frame = pd.read_pickle(STAGE_PREDICTION_PATHS[stage])
    variant = "blend_75_25_logodds"
    if not set(KEY + [variant]).issubset(frame.columns) or frame.duplicated(KEY).any():
        raise ValueError(f"Trial90 {stage} control keys/schema changed")
    for origin, stop in origins.items():
        part = frame.loc[frame.origin.eq(origin)]
        if part.empty or not part.time_step.between(origin+1,stop).all() or int(part.time_step.max()) != stop:
            raise ValueError(f"Trial90 control horizon changed at {origin}")
    return frame[KEY + [variant]].rename(columns={variant: CONTROL})


def attach_graph5(train: pd.DataFrame, test: pd.DataFrame, graph5: pd.DataFrame):
    cols = ["txId", *GRAPH5]
    if graph5.columns.tolist() != cols or graph5.txId.duplicated().any():
        raise ValueError("Trial90 graph5 cache schema changed")
    out=[]
    for frame, name in ((train,"train"),(test,"test")):
        joined=frame.merge(graph5,on="txId",how="left",validate="one_to_one",sort=False)
        if (len(joined)!=len(frame) or not np.array_equal(joined.txId,frame.txId)
                or joined[GRAPH5].isna().any().any() or not np.isfinite(joined[GRAPH5].to_numpy(float)).all()):
            raise ValueError(f"Trial90 graph5 merge invalid for {name}")
        out.append(joined)
    return out[0],out[1]


def make_partition(attr: pd.DataFrame, graph_ids: np.ndarray, graph5: pd.DataFrame,
                   node_period: np.ndarray, edges: pd.DataFrame, columns: list[str],
                   first_step: int, last_step: int, history: bool):
    periods = node_period
    take=(periods<=last_step) if history else ((periods>=first_step)&(periods<=last_step))
    ids=graph_ids[take]; local_period=periods[take]
    if history and (local_period>last_step).any(): raise AssertionError("future graph node entered training partition")
    if not history and ((local_period<first_step).any() or (local_period>last_step).any()):
        raise AssertionError("out-of-window node entered forecast graph")
    subedges=edges.loc[edges.txId1.isin(ids)&edges.txId2.isin(ids),["txId1","txId2"]]
    in_adj,out_adj,edge_count=mean_adjacencies(ids,local_period,subedges)
    selected=attr.loc[attr.time_step.between(int(local_period.min(initial=first_step)),last_step)]
    selected=selected.loc[selected.txId.isin(ids)]
    if history:
        selected=selected.loc[selected.time_step<=last_step]
    else:
        selected=selected.loc[selected.time_step.between(first_step,last_step)]
    if selected.txId.duplicated().any(): raise ValueError("Attributed partition has duplicate txId")
    position=pd.Index(ids)
    attr_loc=position.get_indexer(selected.txId.to_numpy(dtype=np.int64))
    if (attr_loc<0).any(): raise AssertionError("Attributed nodes disappeared from graph partition")
    raw=np.zeros((len(ids),len(columns)),dtype=np.float32)
    if columns[-GRAPH_FEATURE_COUNT:] != GRAPH5:
        raise ValueError("Trial98 requires the 172 historical columns followed by frozen graph5")
    graph_position=pd.Index(graph5.txId.to_numpy(dtype=np.int64)).get_indexer(ids)
    if (graph_position<0).any(): raise AssertionError("Trial90 graph5 cache lost a graph node")
    graph_columns=[c for c in GRAPH5 if c in columns]
    raw[:,[columns.index(c) for c in graph_columns]]=graph5.iloc[graph_position][graph_columns].to_numpy(dtype=np.float32)
    is_attr=np.zeros(len(ids),dtype=bool)
    raw[attr_loc]=selected[columns].to_numpy(dtype=np.float32)
    is_attr[attr_loc]=True
    # Graph-only nodes have exact base-zero / base-unavailable values, while graph5 remains available.
    base_columns=[c for c in columns if c not in GRAPH5]
    base_positions=[columns.index(c) for c in base_columns]
    raw[np.ix_(~is_attr,base_positions)]=0.0
    labels=np.full(len(ids),-1,dtype=np.int8)
    weights=np.zeros(len(ids),dtype=np.float64)
    if history:
        status=label_status(selected.label)
        labels[attr_loc]=status
        w,_=training_weights_40(selected.time_step,status,last_step)
        weights[attr_loc]=w
    edge_bytes=np.ascontiguousarray(subedges.sort_values(["txId1","txId2"]).to_numpy(dtype=np.int64)).tobytes()
    return {"ids":ids,"periods":local_period,"raw":raw,"is_attr":is_attr,"labels":labels,
            "weights":weights,"incoming":in_adj,"outgoing":out_adj,"edge_count":edge_count,
            "node_ids_sha256":hashlib.sha256(ids.tobytes()).hexdigest(),
            "active_edge_sha256":hashlib.sha256(edge_bytes).hexdigest(),
            "attribute_rows":int(is_attr.sum()),"graph_only_rows":int((~is_attr).sum()),
            "history":bool(history),"first_step":int(first_step),"last_step":int(last_step)}


def build_partition_inputs(train: pd.DataFrame, test: pd.DataFrame, graph5: pd.DataFrame,
                           node_period: np.ndarray, edges: pd.DataFrame, full_columns: list[str],
                           origin: int, stop: int):
    history_frame=train.loc[train.time_step<=origin]
    columns=historical_columns(history_frame,full_columns,origin)
    if (len(columns)!=MODEL_INPUT_COUNT or columns[-GRAPH_FEATURE_COUNT:]!=GRAPH5
            or any(c in GRAPH5 for c in columns[:BASE_SELECTED_COUNT])):
        raise ValueError("Trial98 must retain cutoff-safe 172 historical columns plus graph5")
    graph_ids=graph5.txId.to_numpy(dtype=np.int64)
    # Only the history <= cutoff enters the training graph. Future partitions are built separately.
    hist=make_partition(train,graph_ids,graph5,node_period,edges,columns,1,origin,True)
    future_source=pd.concat([train.drop(columns=["label"]),test],ignore_index=True,sort=False)
    future=make_partition(future_source,graph_ids,graph5,node_period,edges,columns,origin+1,stop,False)
    if hist["periods"].max(initial=0)>origin or future["periods"].min(initial=stop+1)<origin+1:
        raise AssertionError("History/future period boundary changed")
    return columns,hist,future


def launch_worker(job: dict, mode: str = "fit") -> dict:
    worker = PROJECT_ROOT / "src" / "missing_aware_directed_gnn_worker_20261006.py"
    with tempfile.TemporaryDirectory(prefix="trial98_worker_") as td:
        inp=Path(td)/"job.joblib"; out=Path(td)/"result.joblib"
        joblib.dump(job,inp,compress=0)
        env=os.environ.copy()
        env["OMP_NUM_THREADS"]="1"; env["MKL_NUM_THREADS"]="1"; env["OPENBLAS_NUM_THREADS"]="1"
        subprocess.run([sys.executable,str(worker),mode,str(inp),str(out)],check=True,env=env,cwd=PROJECT_ROOT)
        return joblib.load(out)


def worker_main(input_path: str, output_path: str, mode: str = "fit") -> None:
    job=joblib.load(input_path)
    result=launch_worker(job,mode)
    joblib.dump(result,output_path)


def load_control(stage: str, origins: dict[int,int]) -> pd.DataFrame:
    return stage_control(stage,origins)


def fit_origin(train,test,graph5,node_period,edges,full_columns,origin,stop,stage,controls):
    columns,hist,future=build_partition_inputs(train,test,graph5,node_period,edges,full_columns,origin,stop)
    attr_rows=train.loc[train.time_step.between(origin+1,stop),KEY[1:]].copy()
    job={"history":hist,"future":future,"columns":columns,"graph_columns":GRAPH5,
         "origin":origin,"stop":stop,"stage":stage,"configuration":config()}
    result=launch_worker(job)
    future_attr=future["is_attr"]
    attr_pos=pd.Index(future["ids"][future_attr]).get_indexer(attr_rows.txId.to_numpy(dtype=np.int64))
    if (attr_pos<0).any(): raise ValueError("Known forecast rows missing from Trial98 inference partition")
    future_labels=dict(zip(attr_rows.txId.to_numpy(dtype=np.int64),attr_rows.label.to_numpy()))
    forecast=pd.DataFrame({"origin":origin,"txId":future["ids"],"time_step":future["periods"],
                           "label":[future_labels.get(int(txid),np.nan) for txid in future["ids"]],
                           "has_feature_row":future_attr,"p_licit":result["probabilities"][:,0],
                           "p_illicit":result["probabilities"][:,1],"p_unknown":result["probabilities"][:,2],
                           CHALLENGER:result["score"]})
    forecast["graph_node_ids_sha256"]=future["node_ids_sha256"]
    pred=forecast.loc[forecast.has_feature_row & forecast.label.notna(),KEY+[CHALLENGER]].copy()
    control=controls.loc[controls.origin.eq(origin),KEY+[CONTROL]]
    known=pred.loc[pred.label.notna()].copy()
    merged=known.merge(control,on=KEY,how="inner",validate="one_to_one",sort=False)
    if len(merged)!=len(known) or not np.array_equal(merged.txId.to_numpy(),known.txId.to_numpy()):
        raise ValueError("Trial98/Trial90 exact keyed prediction alignment failed")
    _,merged[BLEND]=candidate_scores(merged[CONTROL],merged[CHALLENGER])
    file=MODEL_DIR/f"{stage}_{origin}_{stop}.joblib"
    file.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(result["bundle"],file)
    diag={**result["diagnostics"],"stage":stage,"origin":origin,"forecast_end":stop,
          "history_max_step":origin,"selected_columns":columns,"feature_count":len(columns),
          "scaler_fit_rows":hist["attribute_rows"],"scaler_fit_cutoff":origin,
          "history_graph_node_ids_sha256":hist["node_ids_sha256"],
          "future_graph_node_ids_sha256":future["node_ids_sha256"],
          "history_active_edge_sha256":hist["active_edge_sha256"],
          "future_active_edge_sha256":future["active_edge_sha256"],
          "scaler_fit_txid_sha256":hashlib.sha256(hist["ids"][hist["is_attr"]].tobytes()).hexdigest(),
          "model_state_dict_sha256":result["bundle"]["state_dict_sha256"],
          "forecast_row_order_sha256":hashlib.sha256(forecast.txId.to_numpy(dtype=np.int64).tobytes()).hexdigest(),
          "model_file":str(file.relative_to(PROJECT_ROOT)),"model_sha256":sha(file),
          "control":"exact keyed public90 Trial90 blend","same_period_graph_only":True}
    return merged,forecast,diag


def run_stage(train,test,graph5,node_period,edges,full_columns,origins,stage):
    controls=load_control(stage,origins); parts=[]; forecasts=[]; diags=[]
    for origin,stop in origins.items():
        frame,forecast,diag=fit_origin(train,test,graph5,node_period,edges,full_columns,origin,stop,stage,controls)
        parts.append(frame); forecasts.append(forecast); diags.append(diag)
        print(f"Completed Trial98 {stage} {origin}->{stop}",flush=True)
    return pd.concat(parts,ignore_index=True),pd.concat(forecasts,ignore_index=True),diags


def evaluate(predictions: pd.DataFrame,stage: str) -> pd.DataFrame:
    rows=[]
    for origin,frame in predictions.groupby("origin",sort=False):
        slices={"first7":frame.loc[frame.time_step.le(origin+7)]}
        if frame.time_step.max()==origin+14:
            slices.update(full14=frame,second7=frame.loc[frame.time_step.gt(origin+7)])
        for name,part in slices.items():
            for variant in (CONTROL,CHALLENGER,BLEND):
                rows.append({"stage":stage,"origin":int(origin),"slice":name,"variant":variant,
                    "n":len(part),"positives":int(part.label.sum()),"auc":roc_auc_score(part.label,part[variant]),
                    "average_precision":average_precision_score(part.label,part[variant])})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame,stage: str) -> pd.DataFrame:
    pivot=results.pivot(index=["origin","slice"],columns="variant",values="auc"); rows=[]
    if stage=="original":
        groups={"full14":[(14,"full14"),(21,"full14")],"second7":[(14,"second7"),(21,"second7")],
                "primary7":[(o,"first7") for o in (14,21,28)],"rolling7":[(o,"first7") for o in (17,24)]}
    else:
        groups={"full14":[(o,"full14") for o in CONFIRMATION],"second7":[(o,"second7") for o in CONFIRMATION],
                "first7":[(o,"first7") for o in CONFIRMATION]}
    for variant in (CHALLENGER,BLEND):
        delta=pivot[variant]-pivot[CONTROL]; row={"stage":stage,"variant":variant,"control_variant":CONTROL}
        for name,keys in groups.items():
            row[f"{name}_mean_auc"]=float(pivot.loc[keys,variant].mean())
            row[f"{name}_mean_delta"]=float(delta.loc[keys].mean())
            row[f"{name}_control_auc"]=float(pivot.loc[keys,CONTROL].mean())
        row["worst_slice_delta"]=float(delta.min()); row["qualified"]=stage_gate_pass(stage,row); rows.append(row)
    return pd.DataFrame(rows)


def fit_final(train,test,graph5,node_period,edges,full_columns,sample,winner):
    origin,stop=35,49
    columns,hist,future=build_partition_inputs(train,test,graph5,node_period,edges,full_columns,origin,stop)
    # Final output worker trains only through period35 and predicts test periods36-49.
    result=launch_worker({"history":hist,"future":future,"columns":columns,"graph_columns":GRAPH5,
                          "origin":origin,"stop":stop,"stage":"final","configuration":config()})
    test_ids=future["ids"][future["is_attr"]]; scores=result["score"][future["is_attr"]]
    test_position=pd.Index(test_ids).get_indexer(test.txId.to_numpy(dtype=np.int64))
    if (test_position<0).any(): raise ValueError("Final GNN prediction rows do not match test rows")
    challenger=scores[test_position]
    incumbent=pd.read_csv(TRIAL90_CANDIDATE)
    if list(incumbent.columns)!=["index","target"] or not np.array_equal(incumbent["index"],sample["index"]):
        raise ValueError("Trial90 candidate/sample schema or order changed")
    if winner==CHALLENGER: target=challenger
    elif winner==BLEND: target=candidate_scores(incumbent.target.to_numpy(),challenger)[1]
    else: raise ValueError("Unknown final Trial98 winner")
    out=pd.DataFrame({"index":sample["index"].to_numpy(),"target":target})
    if len(out)!=len(sample) or not np.isfinite(target).all() or ((target<0)|(target>1)).any():
        raise ValueError("Final Trial98 candidate is invalid")
    path=PROJECT_ROOT/"submissions"/f"candidate_{PREFIX}.csv"; out.to_csv(path,index=False)
    components=REPORT_DIR/f"{PREFIX}_final_components.pkl"
    pd.DataFrame({"index":sample["index"].to_numpy(),"public90_control":incumbent.target.to_numpy(),
                  CHALLENGER:challenger,BLEND:candidate_scores(incumbent.target.to_numpy(),challenger)[1],
                  "winner":target}).to_pickle(components)
    model_path=MODEL_DIR/"final_through35.joblib"; joblib.dump(result["bundle"],model_path)
    return {"winner":winner,"final_model_file":str(model_path.relative_to(PROJECT_ROOT)),
            "final_model_sha256":sha(model_path),"final_components_file":str(components.relative_to(PROJECT_ROOT)),
            "final_components_sha256":sha(components),"candidate_file":str(path.relative_to(PROJECT_ROOT)),
            "candidate_sha256":sha(path),"candidate_rows":len(out),"candidate_created":True,
            "final_model_fit_performed":True,"submission_created":True,"kaggle_upload":False}


def main() -> None:
    plan=json.loads(PLAN_PATH.read_text())
    verify_frozen_inputs(plan)
    assert_clean_outputs()
    # First competition-data/graph load is below the exact authorization boundary.
    train,test,sample,groups=load_features()
    from src.competition import load_competition_data
    _,_,edges,_=load_competition_data()
    graph5=pd.read_pickle(GRAPH5_CACHE)
    train,test=attach_graph5(train,test,graph5)
    raw_full=groups["regularized_two_hop"]+list(GRAPH5)
    if len(raw_full)!=BASE_FULL_COUNT+GRAPH_FEATURE_COUNT: raise ValueError("Trial98 full feature universe changed")
    attrs=pd.concat([train[["txId","time_step"]],test[["txId","time_step"]]],ignore_index=True)
    graph_ids=graph5.txId.to_numpy(dtype=np.int64)
    component,period,component_meta=weak_component_period_map(graph_ids,edges,attrs,expected_components=WEAK_COMPONENT_COUNT)
    if len(graph_ids)!=GRAPH_NODE_COUNT: raise ValueError("Trial90 frozen graph node universe changed")
    original,forecast_o,diag_o=run_stage(train,test,graph5,period,edges,raw_full,ORIGINS,"original")
    original.to_pickle(REPORT_DIR/f"{PREFIX}_original_predictions.pkl")
    forecast_o.to_pickle(REPORT_DIR/f"{PREFIX}_original_forecast_probabilities.pkl")
    results_o=evaluate(original,"original"); summary_o=summarize(results_o,"original")
    results_o.to_csv(REPORT_DIR/f"{PREFIX}_original_results.csv",index=False)
    summary_o.to_csv(REPORT_DIR/f"{PREFIX}_original_summary.csv",index=False)
    metadata={"trial_number":98,"variant":PREFIX,"plan_sha256":sha(PLAN_PATH),
              "source_sha256":sha(PROJECT_ROOT/"src"/f"{PREFIX}.py"),"input_sha256":plan["input_sha256"],
              "original_gate_passed":bool(summary_o.qualified.any()),"confirmation_run":False,
              "confirmation_gate_passed":False,"same_variant_passed_both_stages":False,
              "two_stage_qualified_variants":[],"winner":None,"candidate_created":False,
              "final_model_fit_performed":False,"submission_created":False,"kaggle_upload":False,
              "weak_component_period_map":component_meta,"original_fit_diagnostics":diag_o}
    preds=[original]; results=[results_o]; summaries=[summary_o]; diags=list(diag_o)
    if metadata["original_gate_passed"]:
        confirm,forecast_c,diag_c=run_stage(train,test,graph5,period,edges,raw_full,CONFIRMATION,"confirmation")
        confirm.to_pickle(REPORT_DIR/f"{PREFIX}_confirmation_predictions.pkl")
        forecast_c.to_pickle(REPORT_DIR/f"{PREFIX}_confirmation_forecast_probabilities.pkl")
        results_c=evaluate(confirm,"confirmation"); summary_c=summarize(results_c,"confirmation")
        results_c.to_csv(REPORT_DIR/f"{PREFIX}_confirmation_results.csv",index=False)
        summary_c.to_csv(REPORT_DIR/f"{PREFIX}_confirmation_summary.csv",index=False)
        winner,intersection=select_winner(summary_o,summary_c)
        metadata.update(confirmation_run=True,confirmation_gate_passed=bool(summary_c.qualified.any()),
            same_variant_passed_both_stages=bool(intersection),two_stage_qualified_variants=intersection,
            winner=winner,confirmation_fit_diagnostics=diag_c)
        preds.append(confirm);results.append(results_c);summaries.append(summary_c);diags.extend(diag_c)
        if winner: metadata.update(fit_final(train,test,graph5,period,edges,raw_full,sample,winner))
    pd.concat(preds,ignore_index=True).to_pickle(REPORT_DIR/f"{PREFIX}_predictions.pkl")
    pd.concat(results,ignore_index=True).to_csv(REPORT_DIR/f"{PREFIX}_results.csv",index=False)
    pd.concat(summaries,ignore_index=True).to_csv(REPORT_DIR/f"{PREFIX}_summary.csv",index=False)
    (REPORT_DIR/f"{PREFIX}_fit_diagnostics.json").write_text(json.dumps(diags,indent=2,sort_keys=True)+"\n")
    metadata["fit_diagnostics"]=diags
    (REPORT_DIR/f"{PREFIX}_metadata.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__":
    if len(sys.argv)==4 and sys.argv[1] in ("--worker","--replay"):
        worker_main(sys.argv[2],sys.argv[3],"fit" if sys.argv[1]=="--worker" else "replay")
    elif len(sys.argv)==1:
        main()
    elif sys.argv[1]=="--fit-free-checks":
        verify_graph_toys();verify_weight_score_gate_toys()
        print("Trial98 pure synthetic checks passed")
    else:
        raise SystemExit("Usage: source.py [--fit-free-checks | --worker INPUT OUTPUT]")
