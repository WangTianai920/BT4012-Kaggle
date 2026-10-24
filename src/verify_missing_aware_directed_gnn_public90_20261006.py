"""Independent, fit-free Trial98 verification. Run only after root authorization.

The replay subprocess evaluates the saved state tensors without importing the
training worker or calling an optimizer. This file never trains a model.
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

csv.field_size_limit(10_000_000)

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.special import expit, logit

# Keep top-level imports Torch-compatible. Tree/metric modules are loaded only
# in the parent audit path, never in the isolated replay process.
GRAPH5 = ["dag_log1p_scc_size", "dag_log1p_source_depth", "dag_log1p_sink_depth",
          "dag_log1p_chain_span", "dag_chain_position"]

P = "missing_aware_directed_gnn_public90_20261006"
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SOURCE = ROOT / "src" / f"{P}.py"
WORKER = ROOT / "src" / "missing_aware_directed_gnn_worker_20261006.py"
PLAN = REPORTS / f"{P}_plan.json"
AUTH = REPORTS / f"{P}_root_authorization.json"
META = REPORTS / f"{P}_metadata.json"
FLAT = REPORTS / f"{P}_fit_diagnostics.json"
AUDIT = REPORTS / f"{P}_parent_verification.json"
LOG = ROOT / "experiment_log.csv"
GRAPH = REPORTS / "directed_chain_position5_public55_20261005_graph_features.pkl"
GRAPH_MAP = REPORTS / "directed_chain_position5_public55_20261005_graph_node_map.pkl"
GRAPH_META = REPORTS / "directed_chain_position5_public55_20261005_graph_metadata.json"
CONTROL_PREFIX = "directed_chain_position5_public55_20261005"
TRIAL97 = "chain_position_feature80_public90_20261005"
KEY = ["origin", "txId", "time_step", "label"]
CONTROL, CHALLENGER, BLEND = "public90_control", "missing_aware_directed_gnn", "blend_75_25_logodds"
EXPECTED_SOURCE_SHA = "f210dc67f8ea886e3d83d139eda5aeab03dde4019dc876b621af2451d92afa47"
EXPECTED_WORKER_SHA = "e0f3326449cbe86941e7276f6ab738e2bdf907683f53f3087ed08a5f1a8192b7"
EXPECTED_PLAN_SHA = "8894d547506857dab77e4b06f4f73166ae8f941cacd7460ff3af7708a4c47160"
PREPARED_REGISTRY_ROWS = 14305
EPS = 1e-12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_row(row: dict) -> bytes:
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")


def fixed_blend(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError("Invalid fixed-blend arrays")
    if ((a < 0) | (a > 1) | (b < 0) | (b > 1)).any():
        raise AssertionError("Fixed-blend inputs leave [0,1]")
    return expit(.75 * logit(np.clip(a, EPS, 1-EPS)) + .25 * logit(np.clip(b, EPS, 1-EPS)))


def check_identity() -> tuple[dict, dict, dict]:
    if "PENDING" in (EXPECTED_SOURCE_SHA + EXPECTED_WORKER_SHA + EXPECTED_PLAN_SHA):
        raise RuntimeError("Trial98 independent verifier awaits the final source, worker, and plan hashes")
    if (sha(SOURCE), sha(WORKER), sha(PLAN)) != (EXPECTED_SOURCE_SHA, EXPECTED_WORKER_SHA, EXPECTED_PLAN_SHA):
        raise AssertionError("Trial98 frozen source, worker, or plan changed")
    if not AUTH.exists() or not META.exists():
        raise RuntimeError("Trial98 root authorization and completed metadata are required")
    plan, auth, meta = [json.loads(p.read_text()) for p in (PLAN, AUTH, META)]
    if (plan.get("trial_number") != 98 or plan.get("variant") != P
            or plan.get("source_sha256") != EXPECTED_SOURCE_SHA
            or plan.get("worker_sha256") != EXPECTED_WORKER_SHA
            or auth.get("authorized") is not True
            or auth.get("source_sha256") != EXPECTED_SOURCE_SHA
            or auth.get("worker_sha256") != EXPECTED_WORKER_SHA
            or auth.get("plan_sha256") != EXPECTED_PLAN_SHA
            or auth.get("registry_unique_ids_preflight", auth.get("registry_rows")) != PREPARED_REGISTRY_ROWS
            or auth.get("trial97_public_miss_acknowledged") is not True
            or auth.get("trial90_positive_audit_acknowledged") is not True
            or auth.get("transfer_diagnostic_reviewed") is not True
            or meta.get("trial_number") != 98 or meta.get("variant") != P
            or meta.get("source_sha256") != EXPECTED_SOURCE_SHA
            or meta.get("plan_sha256") != EXPECTED_PLAN_SHA
            or meta.get("input_sha256") != plan.get("input_sha256")):
        raise AssertionError("Trial98 plan, authorization, and metadata disagree")
    pins = plan.get("input_sha256")
    if not isinstance(pins, dict) or "experiment_log.csv" in pins or len(pins) != 415:
        raise AssertionError("Trial98 immutable input pin inventory is invalid")
    for name, digest in pins.items():
        if sha(ROOT / name) != digest:
            raise AssertionError(f"Trial98 frozen input changed: {name}")
    if str(WORKER.relative_to(ROOT)) not in pins or pins[str(WORKER.relative_to(ROOT))] != EXPECTED_WORKER_SHA:
        raise AssertionError("Independent Torch worker was not frozen in the plan")
    with LOG.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    identities = [r["experiment_id"] for r in rows]
    if len(rows) < PREPARED_REGISTRY_ROWS or not all(identities) or len(identities) != len(set(identities)):
        raise AssertionError("Experiment register has lost the immutable Trial98 preflight snapshot")
    canonical = next((r for r in rows if r["experiment_id"] == f"S97_{TRIAL97}_public_1005_3"), None)
    if canonical is None or json.loads(canonical["details_json"]).get("public_roc_auc") != .95236:
        raise AssertionError("Trial97 public miss is not registered under the corrected 1005-3 ID")
    expected = plan.get("trial97_closeout", {}).get("canonical_public_registry_row_sha256")
    if expected and hashlib.sha256(canonical_row(canonical)).hexdigest() != expected:
        raise AssertionError("Trial97 public registry row changed")
    return plan, auth, meta


def independent_period_map(ids: np.ndarray, edges: pd.DataFrame, anchors: pd.DataFrame):
    """Use NetworkX weak connectivity, independent of the controller's SciPy path."""
    import networkx as nx
    ids = np.asarray(ids, np.int64)
    if len(ids) != 203769 or not np.array_equal(ids, np.unique(ids)):
        raise AssertionError("Frozen all-node graph universe changed")
    graph = nx.Graph()
    graph.add_nodes_from(map(int, ids))
    graph.add_edges_from(edges[["txId1", "txId2"]].itertuples(index=False, name=None))
    if graph.number_of_nodes() != len(ids):
        raise AssertionError("Official graph endpoint missing from Trial90 cache")
    components = list(nx.connected_components(graph))
    if len(components) != 49:
        raise AssertionError("Weak-component count changed")
    periods = {}
    anchor = dict(zip(anchors.txId.astype(int), anchors.time_step.astype(int)))
    if len(anchor) != len(anchors):
        raise AssertionError("Attributed anchor IDs duplicate")
    for members in components:
        inferred = {anchor[n] for n in members if n in anchor}
        if len(inferred) != 1:
            raise AssertionError("Graph component lacks one unique attributed time period")
        period = inferred.pop()
        periods.update((n, period) for n in members)
    period_array = np.fromiter((periods[int(n)] for n in ids), np.int16, len(ids))
    if any(periods[int(u)] != periods[int(v)] for u, v in edges[["txId1", "txId2"]].itertuples(index=False, name=None)):
        raise AssertionError("An official edge crosses forecast periods")
    return period_array, len(components)


def independent_partition(attrs: pd.DataFrame, graph: pd.DataFrame, periods: np.ndarray,
                          edges: pd.DataFrame, columns: list[str], first: int, last: int, history: bool):
    """Create label-free period graph and cutoff-only raw/mask rows from first principles."""
    from scipy.sparse import csr_matrix
    full_ids = graph.txId.to_numpy(np.int64)
    use = periods <= last if history else (periods >= first) & (periods <= last)
    ids = full_ids[use]
    local_period = periods[use]
    local = pd.Index(ids)
    base = columns[:172]
    if len(columns) != 177 or columns[172:] != list(GRAPH5) or any(c in GRAPH5 for c in base):
        raise AssertionError("Saved model must use 172 historical attributes then exact Trial90 graph5")
    selected = attrs.loc[attrs.time_step.between(first, last)].copy()
    selected = selected.loc[selected.txId.isin(ids)]
    if selected.txId.duplicated().any():
        raise AssertionError("Attributed partition IDs duplicate")
    positions = local.get_indexer(selected.txId.to_numpy(np.int64))
    if (positions < 0).any():
        raise AssertionError("Attributed row missing from period graph")
    raw = np.zeros((len(ids), 177), np.float32)
    raw[:, 172:] = graph.loc[use, list(GRAPH5)].to_numpy(np.float32)
    raw[positions] = selected[columns].to_numpy(np.float32)
    attr = np.zeros(len(ids), bool); attr[positions] = True
    labels = np.full(len(ids), -1, np.int8)
    weights = np.zeros(len(ids), np.float64)
    if history:
        from src.label_status_multiclass_20260929 import label_status
        from src.status_mass40_lr03_iter600_20261003 import training_weights_40
        labels[positions] = label_status(selected.label)
        w, _ = training_weights_40(selected.time_step, labels[positions], last)
        weights[positions] = w
    endpoint = edges[["txId1", "txId2"]].to_numpy(np.int64)
    source = local.get_indexer(endpoint[:, 0]); target = local.get_indexer(endpoint[:, 1])
    active = edges.loc[(source >= 0) & (target >= 0), ["txId1", "txId2"]]
    active_bytes = np.ascontiguousarray(active.sort_values(["txId1", "txId2"]).to_numpy(np.int64)).tobytes()
    keep = (source >= 0) & (target >= 0) & (source != target)
    pair = np.unique(np.column_stack((source[keep], target[keep])), axis=0)
    if len(pair) and (local_period[pair[:, 0]] != local_period[pair[:, 1]]).any():
        raise AssertionError("Partition edge crosses a time boundary")
    n = len(ids)
    indegree = np.bincount(pair[:, 1], minlength=n) if len(pair) else np.zeros(n, int)
    outdegree = np.bincount(pair[:, 0], minlength=n) if len(pair) else np.zeros(n, int)
    incoming = (coo_matrix((1 / indegree[pair[:, 1]], (pair[:, 1], pair[:, 0])), shape=(n, n)).tocsr()
                if len(pair) else csr_matrix((n, n)))
    outgoing = (coo_matrix((1 / outdegree[pair[:, 0]], (pair[:, 0], pair[:, 1])), shape=(n, n)).tocsr()
                if len(pair) else csr_matrix((n, n)))
    return {"ids": ids, "periods": local_period, "raw": raw, "is_attr": attr,
            "labels": labels, "weights": weights, "incoming": incoming.astype(np.float32),
            "outgoing": outgoing.astype(np.float32), "edge_count": len(pair),
            "node_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "active_edge_sha256": hashlib.sha256(active_bytes).hexdigest()}


def independent_replay_worker(job: dict):
    """Evaluate checkpoint tensor equations directly; no training worker import."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    bundle, part = job["bundle"], job["future"]
    state = {k: torch.as_tensor(v) for k, v in bundle["state_dict_numpy"].items()}
    raw = part["raw"].astype(np.float64)
    mean = np.asarray(bundle["scaler_mean"], np.float64)
    scale = np.asarray(bundle["scaler_scale"], np.float64)
    x = np.clip((raw - mean) / scale, -5, 5).astype(np.float32)
    attr = np.asarray(part["is_attr"], bool)
    xb = torch.as_tensor(x[:, :172]); xg = torch.as_tensor(x[:, 172:])
    mask = torch.as_tensor(np.repeat(attr[:, None], 172, axis=1), dtype=torch.float32)
    token = state["missing_token"]
    h = torch.empty((len(x), 32), dtype=torch.float32)
    attr_ids = torch.as_tensor(np.flatnonzero(attr), dtype=torch.long)
    if len(attr_ids):
        effective = torch.where(mask[attr_ids] > 0, xb[attr_ids], token[None, :])
        h[attr_ids] = effective @ state["w_value"] + mask[attr_ids] @ state["w_avail"]
    missing_ids = torch.as_tensor(np.flatnonzero(~attr), dtype=torch.long)
    if len(missing_ids):
        h[missing_ids] = token @ state["w_value"]
    h = h + F.linear(torch.cat((xg, torch.ones_like(xg)), dim=1), state["graph_proj.weight"]) + state["bias"]
    def sparse_times(path):
        a = part[path].tocoo()
        index = torch.as_tensor(np.vstack((a.row, a.col)), dtype=torch.long)
        value = torch.as_tensor(a.data, dtype=torch.float32)
        return torch.sparse.mm(torch.sparse_coo_tensor(index, value, a.shape).coalesce(), h)
    hs = F.linear(h, state["self_proj.weight"])
    hi = F.linear(sparse_times("incoming"), state["in_proj.weight"])
    ho = F.linear(sparse_times("outgoing"), state["out_proj.weight"])
    gates = torch.sigmoid(F.linear(torch.cat((hs, hi, ho), dim=1), state["gates.weight"], state["gates.bias"]))
    gi, go = gates.chunk(2, dim=1)
    z = F.layer_norm(torch.relu(hs + gi * hi + go * ho), (32,),
                     state["norm.weight"], state["norm.bias"])
    logits = F.linear(z, state["head.weight"], state["head.bias"])
    prob = torch.softmax(logits, dim=1).numpy()
    return {"probabilities": prob, "score": prob[:, 1] / np.clip(prob[:, 0] + prob[:, 1], 1e-12, None)}


def replay_subprocess(bundle: dict, part: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="trial98_independent_replay_") as td:
        inp, out = Path(td) / "job.joblib", Path(td) / "result.joblib"
        joblib.dump({"bundle": bundle, "future": part}, inp, compress=0)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--independent-replay", str(inp), str(out)],
                       check=True, cwd=ROOT, env=env)
        return joblib.load(out)


def compare_table(path: Path, expected: pd.DataFrame):
    saved = pd.read_csv(path)
    if saved.columns.tolist() != expected.columns.tolist():
        raise AssertionError(f"Saved table columns changed: {path.name}")
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), expected.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=0, atol=1e-12)


def verify_all() -> dict:
    """Read actual artifacts only after source/plan/root authorization checks pass."""
    from sklearn.preprocessing import StandardScaler
    from src.combined_challengers import load_features
    from src.competition import load_competition_data
    from src.horizon_hgb_20260917 import historical_columns
    from src.lr04_increment_logodds_20261002 import CONFIRMATION, ORIGINS
    from src.verify_chain_position_feature80_public90_20261005 import (
        check_control_ancestry, recompute_metrics, recompute_summary,
        reference_graph_features, hard_status_weights,
    )
    plan, auth, meta = check_identity()
    flat = json.loads(FLAT.read_text())
    if flat != meta.get("fit_diagnostics") or not isinstance(flat, list):
        raise AssertionError("Trial98 fit diagnostics differ from metadata")
    train, test, sample, groups = load_features()
    _, _, edges, _ = load_competition_data()
    graph = pd.read_pickle(GRAPH)
    node_map = pd.read_pickle(GRAPH_MAP)
    if graph.columns.tolist() != ["txId", *GRAPH5] or len(graph) != 203769:
        raise AssertionError("Trial90 all-node graph5 schema changed")
    full_ids = np.sort(np.unique(np.r_[train.txId.to_numpy(np.int64), test.txId.to_numpy(np.int64),
                                      edges.txId1.to_numpy(np.int64), edges.txId2.to_numpy(np.int64)]))
    if not np.array_equal(graph.txId.to_numpy(np.int64), full_ids):
        raise AssertionError("Trial90 cache omits a train, test, or graph-only endpoint")
    ref, _, raw_ref, graph_diagnostics = reference_graph_features(edges, full_ids)
    if (not np.array_equal(graph[GRAPH5].to_numpy(), ref[GRAPH5].to_numpy())
            or not np.array_equal(node_map.to_numpy(), raw_ref[node_map.columns].to_numpy())):
        raise AssertionError("Frozen graph5 differs from independent NetworkX SCC reconstruction")
    graph_meta = json.loads(GRAPH_META.read_text())
    for name, value in graph_diagnostics.items():
        if graph_meta.get(name) != value:
            raise AssertionError(f"Trial90 graph metadata differs: {name}")
    anchors = pd.concat((train[["txId", "time_step"]], test[["txId", "time_step"]]), ignore_index=True)
    periods, components = independent_period_map(full_ids, edges, anchors)
    if meta.get("weak_component_period_map", {}).get("weak_component_count") != components:
        raise AssertionError("Trial98 weak-component coverage differs from independent NetworkX map")
    enriched = []
    for name, frame in (("train", train), ("test", test)):
        joined = frame.merge(graph, on="txId", how="left", validate="one_to_one", sort=False)
        if (len(joined) != len(frame) or not np.array_equal(joined.txId, frame.txId)
                or joined[GRAPH5].isna().any().any()):
            raise AssertionError(f"Trial98 {name} graph join lost rows or order")
        enriched.append(joined)
    train, test = enriched
    full_columns = groups["regularized_two_hop"] + GRAPH5
    if len(full_columns) != 187:
        raise AssertionError("Trial98 full candidate input schema changed")
    replay_errors = []
    metric_rows = 0
    expected_models = 0
    stage_predictions = {}
    for stage, origins in (("original", ORIGINS), ("confirmation", CONFIRMATION)):
        run = stage == "original" or meta.get("confirmation_run") is True
        paths = [REPORTS / f"{P}_{stage}_{s}" for s in
                 ("predictions.pkl", "forecast_probabilities.pkl", "results.csv", "summary.csv")]
        if not run:
            if any(p.exists() for p in paths):
                raise AssertionError("Trial98 non-run confirmation produced artifacts")
            continue
        if not all(p.exists() for p in paths):
            raise AssertionError(f"Trial98 {stage} artifacts incomplete")
        prediction, forecast = pd.read_pickle(paths[0]), pd.read_pickle(paths[1])
        if prediction.duplicated(KEY).any() or forecast.duplicated(["origin", "txId"]).any():
            raise AssertionError(f"Trial98 {stage} forecast keys duplicate")
        stage_predictions[stage] = prediction
        if len(forecast) == 0 or not set(KEY+["has_feature_row", "p_licit", "p_illicit", "p_unknown", CHALLENGER,
                                             "graph_node_ids_sha256"]).issubset(forecast.columns):
            raise AssertionError(f"Trial98 {stage} full-node probability cache incomplete")
        for origin, stop in origins.items():
            cutoff_train = train.loc[train.time_step.le(origin)]
            cols = historical_columns(cutoff_train, full_columns, origin)
            if len(cols) != 177 or cols[172:] != GRAPH5:
                raise AssertionError(f"Trial98 {stage} {origin} has wrong cutoff-safe columns")
            hist = independent_partition(train, graph, periods, edges, cols, 1, origin, True)
            future_source = pd.concat((train.drop(columns=["label"]), test), ignore_index=True, sort=False)
            future = independent_partition(future_source, graph, periods, edges, cols, origin+1, stop, False)
            if (hist["periods"].max(initial=0) > origin or future["periods"].min(initial=stop+1) < origin+1):
                raise AssertionError("Future nodes entered historical graph or vice versa")
            detail = next((d for d in flat if d.get("stage") == stage and d.get("origin") == origin), None)
            if detail is None or detail.get("forecast_end") != stop:
                raise AssertionError(f"Trial98 {stage} {origin} diagnostic missing")
            path = ROOT / detail["model_file"]
            if sha(path) != detail.get("model_sha256"):
                raise AssertionError("Trial98 saved model file hash differs")
            bundle = joblib.load(path)
            state = bundle.get("state_dict_numpy")
            if not isinstance(state, dict) or len(state) < 12:
                raise AssertionError("Trial98 saved Torch state tensors incomplete")
            state_digest = hashlib.sha256(b"".join(k.encode() + np.ascontiguousarray(state[k]).tobytes()
                                                  for k in sorted(state))).hexdigest()
            if (state_digest != bundle.get("state_dict_sha256")
                    or state_digest != detail.get("model_state_dict_sha256")
                    or bundle.get("configuration") != plan.get("configuration")
                    or bundle.get("columns") != cols or bundle.get("origin") != origin
                    or bundle.get("stop") != stop or bundle.get("stage") != stage
                    or bundle.get("seed") != 42 or bundle.get("epochs") != 60):
                raise AssertionError("Trial98 saved checkpoint/configuration identity changed")
            scaler = StandardScaler().fit(hist["raw"][hist["is_attr"]])
            if (not np.allclose(bundle["scaler_mean"], scaler.mean_, rtol=0, atol=1e-9)
                    or not np.allclose(bundle["scaler_scale"], scaler.scale_, rtol=0, atol=1e-9)):
                raise AssertionError("Trial98 scaler was not fitted on cutoff-only attributed rows")
            status = hist["labels"][hist["is_attr"]]
            w, _ = hard_status_weights(hist["periods"][hist["is_attr"]], status, origin)
            masses = np.bincount(status, weights=w, minlength=3)
            expected_diag = {
                "feature_count": 177, "history_nodes": len(hist["ids"]), "future_nodes": len(future["ids"]),
                "history_attributed_nodes": int(hist["is_attr"].sum()),
                "history_graph_only_nodes": int((~hist["is_attr"]).sum()),
                "future_attributed_nodes": int(future["is_attr"].sum()),
                "future_graph_only_nodes": int((~future["is_attr"]).sum()),
                "history_edge_count": hist["edge_count"], "future_edge_count": future["edge_count"],
                "history_graph_node_ids_sha256": hist["node_ids_sha256"],
                "future_graph_node_ids_sha256": future["node_ids_sha256"],
                "history_active_edge_sha256": hist["active_edge_sha256"],
                "future_active_edge_sha256": future["active_edge_sha256"],
                "scaler_fit_txid_sha256": hashlib.sha256(hist["ids"][hist["is_attr"]].tobytes()).hexdigest(),
                "train_status_counts": np.bincount(status, minlength=3).tolist(),
            }
            for field, value in expected_diag.items():
                if detail.get(field) != value:
                    raise AssertionError(f"Trial98 {stage} {origin} diagnostic differs: {field}")
            if (not np.allclose(detail["train_status_weight_mass"], masses, rtol=1e-12, atol=1e-12)
                    or not np.isclose(detail["unknown_status_weight_share"], .4, rtol=0, atol=1e-12)
                    or detail.get("epochs") != 60 or detail.get("scaler_fit_rows") != int(hist["is_attr"].sum())
                    or len(detail.get("masked_base_cells_by_epoch", [])) != 60
                    or any(int(v) % 172 for v in detail["masked_base_cells_by_epoch"])):
                raise AssertionError("Trial98 3-status weights or whole-row augmentation changed")
            result = replay_subprocess(bundle, future)
            prob = result["probabilities"]
            if (prob.shape != (len(future["ids"]), 3) or not np.isfinite(prob).all()
                    or not np.allclose(prob.sum(axis=1), 1, rtol=0, atol=1e-6)):
                raise AssertionError("Independent Trial98 replay probabilities invalid")
            saved = forecast.loc[forecast.origin.eq(origin)]
            if (len(saved) != len(future["ids"])
                    or not np.array_equal(saved.txId.to_numpy(np.int64), future["ids"])
                    or not np.array_equal(saved.time_step.to_numpy(np.int16), future["periods"])
                    or not np.array_equal(saved.has_feature_row.to_numpy(bool), future["is_attr"])
                    or not saved.graph_node_ids_sha256.eq(future["node_ids_sha256"]).all()):
                raise AssertionError("Trial98 full-node forecast coverage, order, or availability differs")
            expected_p = saved[["p_licit", "p_illicit", "p_unknown"]].to_numpy(float)
            error = float(np.max(np.abs(expected_p - prob)))
            if error > 1e-7 or np.max(np.abs(saved[CHALLENGER].to_numpy(float)-result["score"])) > 1e-7:
                raise AssertionError(f"Trial98 {stage} {origin} independent checkpoint replay differs: {error}")
            known = saved.loc[saved.has_feature_row & saved.label.notna(), KEY+[CHALLENGER]]
            actual = prediction.loc[prediction.origin.eq(origin), KEY+[CHALLENGER]]
            if not known.reset_index(drop=True).equals(actual.reset_index(drop=True)):
                raise AssertionError("Trial98 known-row prediction cache differs from full-node probabilities")
            replay_errors.append(error); expected_models += 1
        check_control_ancestry(stage, prediction, CONTROL)
        incumbent = pd.read_pickle(REPORTS / f"{CONTROL_PREFIX}_{stage}_predictions.pkl")
        keyed = prediction[KEY+[CONTROL,CHALLENGER,BLEND]].merge(
            incumbent[KEY+[BLEND]].rename(columns={BLEND: "official90"}), on=KEY,
            how="outer", validate="one_to_one", indicator=True)
        if (not keyed._merge.eq("both").all() or len(keyed) != len(prediction)
                or not np.array_equal(keyed[CONTROL], keyed.official90)
                or not np.allclose(keyed[BLEND], fixed_blend(keyed[CONTROL], keyed[CHALLENGER]),
                                   rtol=0, atol=1e-15)):
            raise AssertionError("Trial98 keyed Trial90 control or fixed blend differs")
        metrics = recompute_metrics(stage, prediction, (CONTROL, CHALLENGER, BLEND))
        summary = recompute_summary(stage, metrics, CONTROL, (CHALLENGER, BLEND))
        compare_table(paths[2], metrics)
        compare_table(paths[3], summary)
        metric_rows += len(metrics)
        if stage == "original" and bool(summary.qualified.any()) != bool(meta.get("original_gate_passed")):
            raise AssertionError("Trial98 original gate differs")
        if stage == "confirmation" and bool(summary.qualified.any()) != bool(meta.get("confirmation_gate_passed")):
            raise AssertionError("Trial98 confirmation gate differs")
    combined = pd.concat(stage_predictions.values(), ignore_index=True)
    if not pd.read_pickle(REPORTS / f"{P}_predictions.pkl").equals(combined):
        raise AssertionError("Trial98 combined prediction cache differs")
    # Rebuild stage metric/summary concatenations with NaN-aware DataFrame checks.
    all_metrics = pd.concat([pd.read_csv(REPORTS / f"{P}_{s}_results.csv") for s in stage_predictions], ignore_index=True)
    all_summary = pd.concat([pd.read_csv(REPORTS / f"{P}_{s}_summary.csv") for s in stage_predictions], ignore_index=True)
    compare_table(REPORTS / f"{P}_results.csv", all_metrics)
    compare_table(REPORTS / f"{P}_summary.csv", all_summary)
    original = pd.read_csv(REPORTS / f"{P}_original_summary.csv")
    confirmation = pd.read_csv(REPORTS / f"{P}_confirmation_summary.csv") if "confirmation" in stage_predictions else None
    eligible = (sorted(set(original.loc[original.qualified, "variant"]) &
                       set(confirmation.loc[confirmation.qualified, "variant"])) if confirmation is not None else [])
    winner = next((v for v in (BLEND, CHALLENGER) if v in eligible), None)
    if (meta.get("two_stage_qualified_variants") != eligible
            or meta.get("same_variant_passed_both_stages") is not bool(eligible)
            or meta.get("winner") != winner):
        raise AssertionError("Trial98 strict same-variant two-stage selection differs")
    final_checks = {"winner": winner, "candidate_created": False}
    if winner:
        cutoff, stop = 35, 49
        cols = historical_columns(train.loc[train.time_step.le(cutoff)], full_columns, cutoff)
        hist = independent_partition(train, graph, periods, edges, cols, 1, cutoff, True)
        future_source = pd.concat((train.drop(columns=["label"]), test), ignore_index=True, sort=False)
        future = independent_partition(future_source, graph, periods, edges, cols, 36, 49, False)
        model_path = ROOT / meta["final_model_file"]
        if sha(model_path) != meta.get("final_model_sha256"):
            raise AssertionError("Trial98 final saved model hash differs")
        bundle = joblib.load(model_path)
        if (bundle.get("columns") != cols or bundle.get("origin") != 35 or bundle.get("stop") != 49
                or bundle.get("stage") != "final" or bundle.get("configuration") != plan.get("configuration")):
            raise AssertionError("Trial98 final checkpoint schema differs")
        scaler = StandardScaler().fit(hist["raw"][hist["is_attr"]])
        if not np.allclose(bundle["scaler_mean"], scaler.mean_, rtol=0, atol=1e-9):
            raise AssertionError("Trial98 final scaler fitted outside history")
        replay = replay_subprocess(bundle, future)
        test_pos = pd.Index(future["ids"][future["is_attr"]]).get_indexer(test.txId.to_numpy(np.int64))
        if (test_pos < 0).any():
            raise AssertionError("Trial98 final inference misses test IDs")
        challenger = replay["score"][future["is_attr"]][test_pos]
        incumbent = pd.read_csv(ROOT / "submissions" / f"candidate_{CONTROL_PREFIX}.csv")
        candidate = pd.read_csv(ROOT / meta["candidate_file"])
        expected = challenger if winner == CHALLENGER else fixed_blend(incumbent.target, challenger)
        components_frame = pd.read_pickle(ROOT / meta["final_components_file"])
        if (candidate.columns.tolist() != ["index", "target"] or len(candidate) != len(sample)
                or not np.array_equal(candidate["index"], sample["index"])
                or not np.array_equal(incumbent["index"], sample["index"])
                or np.max(np.abs(candidate.target.to_numpy()-expected)) > 1e-7
                or np.max(np.abs(components_frame[CHALLENGER].to_numpy()-challenger)) > 1e-7
                or sha(ROOT / meta["candidate_file"]) != meta.get("candidate_sha256")
                or sha(ROOT / meta["final_components_file"]) != meta.get("final_components_sha256")):
            raise AssertionError("Trial98 final candidate/component replay differs")
        expected_models += 1
        final_checks = {"winner": winner, "candidate_created": True,
                        "candidate_rows": len(candidate), "candidate_sha256": sha(ROOT / meta["candidate_file"]),
                        "final_model_sha256": sha(model_path)}
    elif (meta.get("candidate_created") is not False or meta.get("final_model_fit_performed") is not False
          or (ROOT / "submissions" / f"candidate_{P}.csv").exists()):
        raise AssertionError("Trial98 non-eligible experiment created a final candidate")
    result = {"trial_number": 98, "variant": P, "source_sha256": EXPECTED_SOURCE_SHA,
              "worker_sha256": EXPECTED_WORKER_SHA, "plan_sha256": EXPECTED_PLAN_SHA,
              "frozen_hashes_verified": True, "root_authorization_verified": True,
              "immutable_registry_unique_ids_preflight": PREPARED_REGISTRY_ROWS,
              "pinned_input_count": len(plan["input_sha256"]),
              "graph_node_count_verified": len(graph), "weak_components_verified": components,
              "cutoff_graph_and_availability_verified": True, "three_status_mass_verified": True,
              "saved_model_replays": expected_models, "maximum_probability_replay_error": max(replay_errors),
              "metric_rows_recomputed": metric_rows, "public90_exact_keyed_ancestry_verified": True,
              "strict_same_variant_intersection_verified": True,
              "eligible_variants": eligible, "final_artifact_checks": final_checks,
              "independent_holdout_claim": False, "kaggle_upload": False}
    AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def pure_replay_toy():
    """Validate direct checkpoint arithmetic against the saved-worker architecture on tiny data."""
    import torch
    from src.missing_aware_directed_gnn_worker_20261006 import GatedMean, _tensor_inputs
    torch.manual_seed(42)
    n = 5
    pair = np.array([[0, 1], [1, 2], [3, 4]])
    a = coo_matrix((np.ones(len(pair)), (pair[:, 1], pair[:, 0])), shape=(n, n)).tocsr()
    b = a.T.tocsr()
    part = {"raw": np.arange(n*177, dtype=np.float32).reshape(n,177)/100,
            "is_attr": np.array([True, False, True, False, True]), "incoming": a, "outgoing": b}
    bundle = {"state_dict_numpy": {k: v.numpy().copy() for k, v in GatedMean().state_dict().items()}, "scaler_mean": np.zeros(177),
              "scaler_scale": np.ones(177)}
    model = GatedMean(); model.load_state_dict({k: torch.as_tensor(v) for k, v in bundle["state_dict_numpy"].items()}); model.eval()
    with torch.no_grad(): expected = torch.softmax(model(*_tensor_inputs(part, bundle["scaler_mean"], bundle["scaler_scale"])),1).numpy()
    actual = independent_replay_worker({"bundle": bundle, "future": part})["probabilities"]
    if not np.allclose(expected, actual, rtol=0, atol=1e-7):
        raise AssertionError(f"Independent GNN arithmetic toy differs from worker architecture: {np.max(np.abs(expected-actual))}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--independent-replay":
        joblib.dump(independent_replay_worker(joblib.load(sys.argv[2])), sys.argv[3])
    elif len(sys.argv) == 2 and sys.argv[1] == "--pure-toy":
        pure_replay_toy(); print("Trial98 independent replay toy passed")
    elif len(sys.argv) == 1:
        print(json.dumps(verify_all(), indent=2, sort_keys=True))
    else:
        raise SystemExit("Usage: verifier.py [--pure-toy | --independent-replay INPUT OUTPUT]")
