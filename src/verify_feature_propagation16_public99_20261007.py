"""Independent fit-free Trial100 feature propagation and model audit."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

csv.field_size_limit(10_000_000)

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

from src.competition import PROJECT_ROOT, REPORT_DIR
from src.horizon_hgb_20260917 import historical_columns
from src.lr04_increment_logodds_20261002 import CONFIRMATION, ORIGINS
from src.label_status_multiclass_20260929 import label_status
from src.verify_chain_position_feature80_public90_20261005 import hard_status_weights, reference_graph_features
from src.verify_missing_aware_directed_gnn_public90_20261006 import independent_period_map

ROOT, REPORTS = PROJECT_ROOT, REPORT_DIR
P = "feature_propagation16_public99_20261007"
SOURCE, PLAN = ROOT / "src" / f"{P}.py", REPORTS / f"{P}_plan.json"
AUTH, META = REPORTS / f"{P}_root_authorization.json", REPORTS / f"{P}_metadata.json"
AUDIT, FLAT = REPORTS / f"{P}_parent_verification.json", REPORTS / f"{P}_fit_diagnostics.json"
LOG = ROOT / "experiment_log.csv"
PRIOR99 = "chain_position_seedensemble3_public90_20261007"
PRIOR99_AUDIT = REPORTS / f"{PRIOR99}_parent_verification.json"
PRIOR99_RESULT = REPORTS / f"{PRIOR99}_submission_result.json"
PRIOR99_CANDIDATE = ROOT / "submissions" / f"candidate_{PRIOR99}.csv"
PRIOR99_AUDIT_ID = f"S99_{PRIOR99}_independent_parent_audit"
PRIOR99_PUBLIC_ID = f"S99_{PRIOR99}_public_1007_1"
TRIAL90 = "directed_chain_position5_public55_20261005"
TRIAL90_CANDIDATE = ROOT / "submissions" / f"candidate_{TRIAL90}.csv"
GRAPH5 = ["dag_log1p_scc_size", "dag_log1p_source_depth", "dag_log1p_sink_depth",
          "dag_log1p_chain_span", "dag_chain_position"]
GRAPH5_CACHE = REPORTS / f"{TRIAL90}_graph_features.pkl"
SEEDS = (7, 2026, 31415)
CONTROL = "public99_control"
STANDALONE, BLEND = "feature_propagation_hgb", "blend_75_25_logodds"
KEY = ["origin", "txId", "time_step", "label"]
EXTRA = [f"fp_in_{i:02d}" for i in range(16)] + [f"fp_out_{i:02d}" for i in range(16)] + ["frac_graph_only_in", "frac_graph_only_out"]
EXPECTED_SOURCE_SHA = "7878a3e726b86e3f33239db1281acd4377b73375ade4357be4891b3138952f89"
EXPECTED_PLAN_SHA = "e8606fa7d7fc22747914ed37be0b74e3fe86cbf8769d949d013a162620bd2ae2"
EXPECTED_PRIOR99_AUDIT_SHA = "35505788be1af242ee2d351ccb80c26f60ea854553db57bdd1d76b293ba38c13"
EXPECTED_PRIOR99_RESULT_SHA = "e92e14b4df1b2111d0e6d44b660448aaeafb95830b683aefcd1934f298493b44"
EXPECTED_PRIOR99_CANDIDATE_SHA = "a6913ff1127d47a12e2434010f6d5be9729d6fc1eca5e0fcdc3e3253bdc31ef1"
EXPECTED_PRIOR99_AUDIT_ROW_SHA = "5ce74c4171a184406514ffc095b009f8b754985164871dd9d4ddf028a62f6fb3"
EXPECTED_PRIOR99_PUBLIC_ROW_SHA = "78760d3cacb8d41bf13f8deb193021962bb3f9500a5678fb95208a8d16073204"
PREPARED_REGISTRY_ROWS = 14460
EPS = 1e-12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry_preflight_prefix_sha(path: Path, n_rows: int) -> str:
    """Hash exact frozen CSV bytes through the n-th data row, including header."""
    raw = path.read_bytes()
    with io.StringIO(raw.decode("utf-8-sig"), newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for _ in range(n_rows):
            next(reader)
        physical_lines = reader.line_num
    lines = raw.splitlines(keepends=True)
    if physical_lines > len(lines):
        raise AssertionError("Trial100 registry snapshot line count exceeds file length")
    return hashlib.sha256(b"".join(lines[:physical_lines])).hexdigest()


def score_logit(p):
    return logit(np.clip(np.asarray(p, float), EPS, 1-EPS))


def fixed_ensemble(member_scores):
    matrix = np.asarray(member_scores, float)
    if matrix.ndim != 2 or matrix.shape[1] != 3 or not np.isfinite(matrix).all():
        raise AssertionError("Trial99 control requires three finite conditional-score vectors")
    return expit(score_logit(matrix).mean(axis=1))


def fixed_blend(control, challenger):
    a, b = np.asarray(control, float), np.asarray(challenger, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError("Fixed logit blend inputs are not aligned")
    return expit(.75*score_logit(a) + .25*score_logit(b))


def check_prior99_registry(plan: dict, auth: dict) -> dict:
    if (sha(PRIOR99_AUDIT) != EXPECTED_PRIOR99_AUDIT_SHA
            or sha(PRIOR99_RESULT) != EXPECTED_PRIOR99_RESULT_SHA
            or sha(PRIOR99_CANDIDATE) != EXPECTED_PRIOR99_CANDIDATE_SHA):
        raise AssertionError("Trial99 positive audit/public result/candidate changed")
    prior_audit = json.loads(PRIOR99_AUDIT.read_text())
    prior_result = json.loads(PRIOR99_RESULT.read_text())
    if (prior_audit.get("saved_model_replays") != 27
            or prior_audit.get("metric_rows_recomputed") != 54
            or prior_audit.get("final_artifact_checks", {}).get("winner") != "blend_75_25_logodds"
            or prior_audit.get("strict_same_variant_intersection_verified") is not True
            or prior_result.get("submission_name") != "1007-1"
            or prior_result.get("public_roc_auc") != .95280
            or prior_result.get("private_roc_auc") is not None
            or prior_result.get("candidate_sha256") != EXPECTED_PRIOR99_CANDIDATE_SHA):
        raise AssertionError("Trial99 positive public control status changed")
    with LOG.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ids = [r.get("experiment_id", "") for r in rows]
    if len(rows) < PREPARED_REGISTRY_ROWS or not all(ids) or len(ids) != len(set(ids)):
        raise AssertionError("Trial100 immutable registry snapshot lost rows or unique identities")
    selected = {r["experiment_id"]: r for r in rows if r["experiment_id"] in (PRIOR99_AUDIT_ID, PRIOR99_PUBLIC_ID)}
    if set(selected) != {PRIOR99_AUDIT_ID, PRIOR99_PUBLIC_ID}:
        raise AssertionError("Trial99 audited and public registry identities are missing")
    hashes = {name: hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
              for name, row in selected.items()}
    if (hashes[PRIOR99_AUDIT_ID] != EXPECTED_PRIOR99_AUDIT_ROW_SHA
            or hashes[PRIOR99_PUBLIC_ID] != EXPECTED_PRIOR99_PUBLIC_ROW_SHA):
        raise AssertionError("Trial99 exact audit/public registry rows changed")
    return hashes


def check_identity() -> tuple[dict, dict, dict, dict]:
    if "PENDING" in EXPECTED_SOURCE_SHA + EXPECTED_PLAN_SHA:
        raise RuntimeError("Trial100 independent verifier awaits final source and plan hashes")
    if sha(SOURCE) != EXPECTED_SOURCE_SHA or sha(PLAN) != EXPECTED_PLAN_SHA:
        raise AssertionError("Trial100 source or plan changed after freeze")
    if not AUTH.exists() or not META.exists():
        raise RuntimeError("Trial100 root authorization and completed metadata are required")
    plan, auth, meta = [json.loads(p.read_text()) for p in (PLAN, AUTH, META)]
    if (plan.get("trial_number") != 100 or plan.get("variant") != P
            or plan.get("source_sha256") != EXPECTED_SOURCE_SHA
            or auth.get("authorized") is not True or auth.get("source_sha256") != EXPECTED_SOURCE_SHA
            or auth.get("plan_sha256") != EXPECTED_PLAN_SHA
            or auth.get("registry_rows") != PREPARED_REGISTRY_ROWS
            or auth.get("root_historical_pca_clamped_diffusion_graph_only_fractions_weights_gates_reviewed") is not True
            or auth.get("trial99_positive_public_control_acknowledged") is not True
            or meta.get("trial_number") != 100 or meta.get("variant") != P
            or meta.get("source_sha256") != EXPECTED_SOURCE_SHA
            or meta.get("plan_sha256") != EXPECTED_PLAN_SHA):
        raise AssertionError("Trial100 frozen plan/authorization/metadata identity changed")
    cfg = plan.get("configuration", {})
    if cfg != plan.get("model"):
        raise AssertionError("Trial100 duplicated frozen HGB configuration differs")
    for name, expected in {"max_iter": 600, "learning_rate": .03, "max_leaf_nodes": 15,
                           "min_samples_leaf": 40, "l2_regularization": 2.,
                           "max_features": .8, "unknown_status_mass": .4,
                           "half_life": 20., "random_state": 42, "early_stopping": False}.items():
        if cfg.get(name) != expected:
            raise AssertionError(f"Trial100 frozen HGB/status configuration changed: {name}")
    pins = plan.get("input_sha256")
    if not isinstance(pins, dict) or len(pins) != 476 or "experiment_log.csv" not in pins:
        raise AssertionError("Trial100 input pin inventory is invalid")
    for rel, digest in pins.items():
        actual = (registry_preflight_prefix_sha(LOG, PREPARED_REGISTRY_ROWS)
                  if rel == "experiment_log.csv" else sha(ROOT / rel))
        if actual != digest:
            raise AssertionError(f"Trial100 frozen input changed: {rel}")
    predecessor = check_prior99_registry(plan, auth)
    return plan, auth, meta, predecessor


def compare_table(path: Path, expected: pd.DataFrame):
    saved = pd.read_csv(path)
    if saved.columns.tolist() != expected.columns.tolist():
        raise AssertionError(f"Trial100 table columns changed: {path.name}")
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), expected.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=0, atol=1e-12)


def reference_metrics(stage: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for origin, part in frame.groupby("origin", sort=False):
        slices = [("first7", part.loc[part.time_step.le(origin+7)])]
        if int(part.time_step.max()) == int(origin)+14:
            slices += [("full14", part), ("second7", part.loc[part.time_step.gt(origin+7)])]
        for name, subset in slices:
            for variant in (CONTROL, STANDALONE, BLEND):
                y = subset.label.to_numpy(int)
                rows.append({"stage": stage, "origin": int(origin), "slice": name,
                             "variant": variant, "n": len(subset), "positives": int(y.sum()),
                             "auc": float(roc_auc_score(y, subset[variant])),
                             "average_precision": float(average_precision_score(y, subset[variant]))})
    return pd.DataFrame(rows)


def reference_summary(stage: str, metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(index=["origin", "slice"], columns="variant", values="auc")
    groups = ({"full14": [(14,"full14"),(21,"full14")],
               "second7": [(14,"second7"),(21,"second7")],
               "primary7": [(o,"first7") for o in (14,21,28)],
               "rolling7": [(o,"first7") for o in (17,24)]} if stage == "original" else
              {name: [(o,name) for o in CONFIRMATION] for name in ("full14","second7","first7")})
    rows = []
    for variant in (STANDALONE, BLEND):
        rec = {"stage": stage, "variant": variant}
        for name, keys in groups.items():
            delta = pivot.loc[keys,variant] - pivot.loc[keys,CONTROL]
            rec[f"{name}_mean_auc"] = float(pivot.loc[keys,variant].mean())
            rec[f"{name}_mean_delta"] = float(delta.mean())
        rec["worst_slice_delta"] = float((pivot[variant]-pivot[CONTROL]).min())
        gate = (all(rec[f"{name}_mean_delta"] > 0 for name in groups) if stage == "original" else
                rec["full14_mean_delta"] > 0 and rec["second7_mean_delta"] > 0 and rec["first7_mean_delta"] >= 0)
        rec["qualified"] = bool(gate and rec["worst_slice_delta"] >= -.001)
        rows.append(rec)
    return pd.DataFrame(rows)


def rebuild_public99_stage(stage: str, origins: dict[int, int], train: pd.DataFrame,
                           full_columns: list[str]) -> pd.DataFrame:
    """Replay every saved Trial99 seed member, then rebuild 90→99 control on exact keys."""
    trial99 = pd.read_pickle(REPORTS / f"{PRIOR99}_{stage}_predictions.pkl")
    member_cache = pd.read_pickle(REPORTS / f"{PRIOR99}_{stage}_member_forecasts.pkl")
    trial90 = pd.read_pickle(REPORTS / f"{TRIAL90}_{stage}_predictions.pkl")
    chunks = []
    for origin, stop in origins.items():
        future = train.loc[train.time_step.between(origin+1, stop)]
        columns = historical_columns(train.loc[train.time_step.le(origin)], full_columns, origin)
        if len(columns) != 177 or columns[-5:] != GRAPH5:
            raise AssertionError("Trial99 control lost its 172+5 cutoff-safe feature selection")
        scores = []
        for seed in SEEDS:
            model_path = ROOT / "models" / PRIOR99 / f"{stage}_{origin}_{stop}_seed{seed}_model.joblib"
            bundle = joblib.load(model_path)
            if bundle.get("seed") != seed or bundle.get("columns") != columns:
                raise AssertionError("Trial99 saved seed-control bundle changed")
            p = bundle["model"].predict_proba(future[columns])
            score = p[:, 1] / (p[:, 0] + p[:, 1])
            scores.append(score)
        matrix = np.column_stack(scores)
        saved = member_cache.loc[member_cache.origin.eq(origin)]
        if len(saved) != len(future) or not np.array_equal(saved.txId.to_numpy(), future.txId.to_numpy()):
            raise AssertionError("Trial99 saved member forecast rows changed")
        for col, values in zip((f"seed_{seed}_conditional_score" for seed in SEEDS), scores):
            if not np.allclose(saved[col].to_numpy(), values, rtol=0, atol=1e-12):
                raise AssertionError(f"Trial99 saved {col} differs from model replay")
        ensemble = fixed_ensemble(matrix)
        if not np.allclose(saved.chain_position_seedensemble3_conditional_logit, ensemble, rtol=0, atol=1e-12):
            raise AssertionError("Trial99 three-member ensemble differs")
        known = saved.loc[saved.label.notna(), KEY].copy()
        known["replayed99_ensemble"] = ensemble[saved.label.notna().to_numpy()]
        baseline = trial90.loc[trial90.origin.eq(origin), KEY+["blend_75_25_logodds"]]
        joined = known.merge(baseline, on=KEY, how="outer", validate="one_to_one", indicator=True, sort=False)
        if len(joined) != len(known) or not joined._merge.eq("both").all():
            raise AssertionError("Trial90→Trial99 control key population changed")
        joined[CONTROL] = fixed_blend(joined.blend_75_25_logodds, joined.replayed99_ensemble)
        original = trial99.loc[trial99.origin.eq(origin)]
        if (len(original) != len(joined) or not np.array_equal(original.txId.to_numpy(), joined.txId.to_numpy())
                or not np.allclose(original.blend_75_25_logodds, joined[CONTROL], rtol=0, atol=1e-15)):
            raise AssertionError("Trial99 exact stage control differs from model/public90 reconstruction")
        chunks.append(joined[KEY+[CONTROL]])
    return pd.concat(chunks, ignore_index=True)


def rebuild_public99_final(train: pd.DataFrame, test: pd.DataFrame,
                           sample: pd.DataFrame, full_columns: list[str]) -> np.ndarray:
    """Rebuild the exact 1007-1 control from three final Trial99 seed models."""
    columns = historical_columns(train.loc[train.time_step.le(35)], full_columns, 35)
    scores = []
    for seed in SEEDS:
        path = ROOT / "models" / PRIOR99 / f"final_through35_seed{seed}_model.joblib"
        bundle = joblib.load(path)
        if bundle.get("seed") != seed or bundle.get("stage") != "final" or bundle.get("origin") != 35:
            raise AssertionError("Trial99 final seed-control bundle changed")
        if bundle["columns"] != columns:
            raise AssertionError("Trial99 final seed members use different historical columns")
        if len(columns) != 177 or columns[-5:] != GRAPH5:
            raise AssertionError("Trial99 final cutoff feature schema changed")
        p = bundle["model"].predict_proba(test[columns])
        scores.append(p[:, 1] / (p[:, 0] + p[:, 1]))
    ensemble = fixed_ensemble(np.column_stack(scores))
    prior90 = pd.read_csv(TRIAL90_CANDIDATE)
    prior99 = pd.read_csv(PRIOR99_CANDIDATE)
    if (prior90.columns.tolist() != ["index", "target"]
            or prior99.columns.tolist() != ["index", "target"]
            or not np.array_equal(prior90["index"], sample["index"])
            or not np.array_equal(prior99["index"], sample["index"])):
        raise AssertionError("Trial99 final public control/sample order changed")
    rebuilt = fixed_blend(prior90.target, ensemble)
    if not np.allclose(prior99.target, rebuilt, rtol=0, atol=1e-12):
        raise AssertionError("Trial99 final public control differs from three-member + Trial90 blend")
    return rebuilt


def independent_diffusion(graph_ids: np.ndarray, periods: np.ndarray, edges: pd.DataFrame,
                          observed_ids: np.ndarray, observed_pca: np.ndarray,
                          steps: int = 20) -> tuple[np.ndarray, np.ndarray, dict]:
    """NetworkX-built symmetric operator; NumPy directed aggregation and clamp."""
    from scipy.sparse import diags
    ids = np.asarray(graph_ids, np.int64)
    period = np.asarray(periods, np.int16)
    observed_ids = np.asarray(observed_ids, np.int64)
    observed = np.asarray(observed_pca, np.float64)
    if (ids.ndim != 1 or not np.array_equal(ids, np.unique(ids)) or period.shape != ids.shape
            or observed.shape != (len(observed_ids), 16) or not np.isfinite(observed).all()):
        raise AssertionError("Trial100 independent diffusion input schema changed")
    loc = pd.Index(ids)
    observed_pos = loc.get_indexer(observed_ids)
    if (observed_pos < 0).any() or len(np.unique(observed_pos)) != len(observed_pos):
        raise AssertionError("Trial100 observed PCA nodes duplicate or leave graph")
    observed_mask = np.zeros(len(ids), bool); observed_mask[observed_pos] = True
    values = np.zeros((len(ids), 16), np.float64); values[observed_pos] = observed
    result = values.copy()
    feature = np.zeros((len(ids), 34), np.float64)
    all_edges = edges[["txId1", "txId2"]].drop_duplicates()
    all_edges = all_edges.loc[all_edges.txId1.ne(all_edges.txId2)]
    src = loc.get_indexer(all_edges.txId1.to_numpy(np.int64))
    dst = loc.get_indexer(all_edges.txId2.to_numpy(np.int64))
    if (src < 0).any() or (dst < 0).any() or (period[src] != period[dst]).any():
        raise AssertionError("Trial100 graph edge endpoint/period is invalid")
    in_degree = np.zeros(len(ids), np.int64)
    out_degree = np.zeros(len(ids), np.int64)
    for p in np.unique(period):
        nodes = np.flatnonzero(period == p)
        index = np.full(len(ids), -1, np.int64); index[nodes] = np.arange(len(nodes))
        take = period[src] == p
        u, v = index[src[take]], index[dst[take]]
        graph = nx.Graph()
        graph.add_nodes_from(range(len(nodes)))
        graph.add_edges_from(zip(u.tolist(), v.tolist()))
        adj = nx.to_scipy_sparse_array(graph, nodelist=list(range(len(nodes))),
                                       weight=None, dtype=np.float64, format="csr")
        degree = np.asarray(adj.sum(axis=1)).ravel()
        inverse = np.divide(1., np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
        operator = diags(inverse) @ adj @ diags(inverse)
        clamped = observed_mask[nodes]
        local = values[nodes].copy()
        for _ in range(steps):
            local = np.asarray(operator @ local)
            local[clamped] = values[nodes][clamped]
        result[nodes] = local
        incoming = np.zeros((len(nodes), 16), np.float64)
        outgoing = np.zeros_like(incoming)
        np.add.at(incoming, v, local[u])
        np.add.at(outgoing, u, local[v])
        di = np.bincount(v, minlength=len(nodes))
        do = np.bincount(u, minlength=len(nodes))
        in_degree[nodes] = di; out_degree[nodes] = do
        feature[nodes, :16] = np.divide(incoming, di[:, None], out=np.zeros_like(incoming), where=di[:, None] > 0)
        feature[nodes, 16:32] = np.divide(outgoing, do[:, None], out=np.zeros_like(outgoing), where=do[:, None] > 0)
        absent = (~clamped).astype(np.float64)
        missing_in = np.bincount(v, weights=absent[u], minlength=len(nodes))
        missing_out = np.bincount(u, weights=absent[v], minlength=len(nodes))
        feature[nodes, 32] = np.divide(missing_in, di, out=np.zeros(len(nodes)), where=di > 0)
        feature[nodes, 33] = np.divide(missing_out, do, out=np.zeros(len(nodes)), where=do > 0)
    if not np.isfinite(feature).all() or not np.isfinite(result).all():
        raise AssertionError("Trial100 independent propagation produced nonfinite features")
    diagnostics = {"graph_node_count": len(ids), "attributed_node_count": int(observed_mask.sum()),
                   "graph_only_node_count": int((~observed_mask).sum()),
                   "component_count": len(np.unique(period)), "propagation_steps": steps,
                   "incoming_supported": int((in_degree > 0).sum()),
                   "outgoing_supported": int((out_degree > 0).sum()),
                   "incoming_zero_mean_rows": int((in_degree == 0).sum()),
                   "outgoing_zero_mean_rows": int((out_degree == 0).sum()),
                   "mean_graph_only_incoming_neighbor_fraction": float(feature[:, 32].mean()),
                   "mean_graph_only_outgoing_neighbor_fraction": float(feature[:, 33].mean())}
    return result, feature, diagnostics


def reference_feature_cache(train: pd.DataFrame, test: pd.DataFrame,
                            graph_ids: np.ndarray, periods: np.ndarray,
                            edges: pd.DataFrame, raw_columns: list[str],
                            scaler: StandardScaler, pca: PCA,
                            origin: int, stop: int) -> tuple[pd.DataFrame, list[dict]]:
    parts, diagnostics = [], []
    for step in range(1, stop+1):
        nodes = graph_ids[periods == step]
        if not len(nodes):
            continue
        if step <= origin:
            observed = train.loc[train.time_step.eq(step)]
        else:
            observed = pd.concat((train.loc[train.time_step.eq(step)],
                                  test.loc[test.time_step.eq(step)]), ignore_index=True)
        if observed.empty or observed.txId.duplicated().any():
            raise AssertionError(f"Trial100 period {step} has missing/duplicate attributed nodes")
        edge_subset = edges.loc[edges.txId1.isin(nodes) & edges.txId2.isin(nodes), ["txId1", "txId2"]]
        z = pca.transform(scaler.transform(observed[raw_columns].to_numpy(np.float64)))
        propagated, extra, summary = independent_diffusion(
            nodes, np.full(len(nodes), step, np.int16), edge_subset,
            observed.txId.to_numpy(np.int64), z, steps=20)
        if not np.array_equal(propagated[pd.Index(nodes).get_indexer(observed.txId.to_numpy(np.int64))], z):
            raise AssertionError("Trial100 attributed PCA observations were not clamped")
        part = pd.DataFrame(extra, columns=EXTRA)
        part.insert(0, "txId", nodes)
        part["time_step"] = step
        parts.append(part)
        diagnostics.append({"period": int(step), **summary,
                            "observed_attributed_count": len(observed)})
    return pd.concat(parts, ignore_index=True), diagnostics


def check_feature_diagnostics(saved: list[dict], reference: list[dict]) -> bool:
    if not isinstance(saved, list) or len(saved) != len(reference):
        return False
    for left, right in zip(saved, reference):
        if set(left) != set(right):
            return False
        for key in left:
            a, b = left[key], right[key]
            if isinstance(a, float) or isinstance(b, float):
                if not np.isclose(a, b, rtol=0, atol=1e-9):
                    return False
            elif a != b:
                return False
    return True


def pure_toy():
    ids = np.arange(1, 7, dtype=np.int64)
    period = np.array([1,1,1,1,2,2], dtype=np.int16)
    edge = pd.DataFrame({"txId1": [1,2,3,4,5], "txId2": [2,3,2,1,6]})
    observed_ids = np.array([1,3,5], dtype=np.int64)
    observed = np.zeros((3,16), float); observed[:,0] = [2.,4.,8.]
    h, features, diag = independent_diffusion(ids, period, edge, observed_ids, observed)
    if (not np.array_equal(h[[0,2,4],0], [2.,4.,8.]) or features.shape != (6,34)
            or not np.isclose(features[0,32], 1.)  # node1 has graph-only predecessor node4.
            or not np.isclose(features[0,33], 1.)  # node1 has graph-only successor node2.
            or not np.isclose(features[1,32], 0.)
            or not np.isclose(features[5,32], 0.)
            or diag["graph_node_count"] != 6):
        raise AssertionError("Trial100 independent clamp/fraction toy differs")


def verify_all() -> dict:
    from src.combined_challengers import load_features
    from src.competition import load_competition_data
    plan, auth, meta, predecessor = check_identity()
    flat = json.loads(FLAT.read_text())
    if not isinstance(flat, list) or flat != meta.get("fit_diagnostics", []):
        raise AssertionError("Trial100 flat fit diagnostics differ from metadata")
    train, test, sample, groups = load_features()
    _, _, edges, _ = load_competition_data()
    graph5 = pd.read_pickle(GRAPH5_CACHE)
    graph_map = pd.read_pickle(REPORTS / f"{TRIAL90}_graph_node_map.pkl")
    graph_meta = json.loads((REPORTS / f"{TRIAL90}_graph_metadata.json").read_text())
    ids = np.sort(np.unique(np.r_[train.txId.to_numpy(np.int64), test.txId.to_numpy(np.int64),
                               edges.txId1.to_numpy(np.int64), edges.txId2.to_numpy(np.int64)]))
    ref_graph5, _, ref_map, ref_graph_diag = reference_graph_features(edges, ids)
    if (len(ids) != 203769 or graph5.columns.tolist() != ["txId", *GRAPH5]
            or not np.array_equal(graph5.txId.to_numpy(np.int64), ids)
            or not np.array_equal(graph5[GRAPH5].to_numpy(), ref_graph5[GRAPH5].to_numpy())
            or not np.array_equal(graph_map.to_numpy(), ref_map[graph_map.columns].to_numpy())):
        raise AssertionError("Trial100 inherited all-node graph5 differs from NetworkX reference")
    for name, value in ref_graph_diag.items():
        if graph_meta.get(name) != value:
            raise AssertionError(f"Trial100 Trial90 graph metadata differs: {name}")
    enriched = []
    for name, frame in (("train", train), ("test", test)):
        joined = frame.merge(graph5, on="txId", how="left", validate="one_to_one", sort=False)
        if len(joined) != len(frame) or not np.array_equal(joined.txId, frame.txId) or joined[GRAPH5].isna().any().any():
            raise AssertionError(f"Trial100 {name} graph join lost rows/order")
        enriched.append(joined)
    train, test = enriched
    full_columns = list(groups["regularized_two_hop"]) + GRAPH5
    if len(full_columns) != 187:
        raise AssertionError("Trial100 full feature universe changed")
    anchors = pd.concat((train[["txId", "time_step"]], test[["txId", "time_step"]]), ignore_index=True)
    periods, n_component = independent_period_map(ids, edges, anchors)
    if n_component != 49 or meta.get("weak_component_period_map", {}).get("weak_component_count") != 49:
        raise AssertionError("Trial100 component-period coverage changed")
    replay_errors, metric_rows, model_replays = [], 0, 0
    prior99_member_replays = 0
    stage_prediction = {}
    for stage, origins in (("original", ORIGINS), ("confirmation", CONFIRMATION)):
        run = stage == "original" or meta.get("confirmation_run") is True
        stage_paths = [REPORTS / f"{P}_{stage}_{name}" for name in
                       ("predictions.pkl", "forecast_probabilities.pkl", "results.csv", "summary.csv")]
        if not run:
            if any(path.exists() for path in stage_paths):
                raise AssertionError("Trial100 non-run confirmation artifacts exist")
            continue
        if not all(path.exists() for path in stage_paths):
            raise AssertionError(f"Trial100 {stage} artifact set incomplete")
        control99 = rebuild_public99_stage(stage, origins, train, full_columns)
        prior99_member_replays += 3*len(origins)
        saved_pred, stage_forecast = pd.read_pickle(stage_paths[0]), pd.read_pickle(stage_paths[1])
        if saved_pred.duplicated(KEY).any() or stage_forecast.duplicated(KEY).any():
            raise AssertionError("Trial100 keyed stage predictions duplicate")
        stage_prediction[stage] = saved_pred
        for origin, stop in origins.items():
            base_columns = historical_columns(train.loc[train.time_step.le(origin)], full_columns, origin)
            raw_columns = [name for name in base_columns if name.startswith("feat_")]
            model_columns = base_columns + EXTRA
            if len(base_columns) != 177 or base_columns[-5:] != GRAPH5 or len(raw_columns) != 155 or len(model_columns) != 211:
                raise AssertionError("Trial100 172+5/155 raw/34 propagated feature contract changed")
            history = train.loc[train.time_step.le(origin)]
            scaler = StandardScaler().fit(history[raw_columns].to_numpy(np.float64))
            pca = PCA(n_components=16, svd_solver="full", whiten=False).fit(
                scaler.transform(history[raw_columns].to_numpy(np.float64)))
            detail = next((d for d in flat if d.get("stage") == stage and d.get("origin") == origin), None)
            if detail is None or detail.get("forecast_end") != stop:
                raise AssertionError(f"Trial100 {stage} {origin} model diagnostic missing")
            prep_path = ROOT / detail["preprocess_file"]
            model_path = ROOT / detail["model_file"]
            per_forecast_path = ROOT / detail["forecast_file"]
            feature_path = ROOT / detail["graph_features_file"]
            for path, field in ((prep_path,"preprocess_sha256"),(model_path,"model_sha256"),
                                (per_forecast_path,"forecast_sha256"),(feature_path,"graph_features_sha256")):
                if sha(path) != detail[field]:
                    raise AssertionError(f"Trial100 {stage} {origin} saved artifact changed: {field}")
            prep = joblib.load(prep_path)
            if (prep.get("raw_columns") != raw_columns or prep.get("model_columns") != model_columns
                    or prep.get("origin") != origin or prep.get("forecast_end") != stop
                    or prep.get("plan_sha256") != EXPECTED_PLAN_SHA):
                raise AssertionError("Trial100 saved historical preprocessing schema changed")
            saved_scaler, saved_pca = prep.get("scaler"), prep.get("pca")
            if not isinstance(saved_scaler, StandardScaler) or not isinstance(saved_pca, PCA):
                raise AssertionError("Trial100 saved preprocessing estimator type changed")
            for name in ("mean_", "var_", "scale_"):
                if not np.allclose(getattr(saved_scaler,name), getattr(scaler,name), rtol=0, atol=1e-9):
                    raise AssertionError(f"Trial100 cutoff-only StandardScaler differs: {name}")
            for name in ("components_", "mean_", "singular_values_", "explained_variance_"):
                if not np.allclose(getattr(saved_pca,name), getattr(pca,name), rtol=1e-10, atol=1e-8):
                    raise AssertionError(f"Trial100 cutoff-only PCA16 differs: {name}")
            reference_features, feature_diag = reference_feature_cache(
                train, test, ids, periods, edges, raw_columns, scaler, pca, origin, stop)
            saved_features = pd.read_pickle(feature_path)
            if (saved_features.columns.tolist() != ["txId", *EXTRA, "time_step"]
                    or len(saved_features) != len(reference_features)
                    or not np.array_equal(saved_features.txId.to_numpy(), reference_features.txId.to_numpy())
                    or not np.array_equal(saved_features.time_step.to_numpy(), reference_features.time_step.to_numpy())):
                raise AssertionError("Trial100 saved full-node propagation population/order differs")
            feature_error = float(np.max(np.abs(saved_features[EXTRA].to_numpy()-reference_features[EXTRA].to_numpy())))
            if feature_error > 1e-8 or not check_feature_diagnostics(detail.get("feature_diagnostics"), feature_diag):
                raise AssertionError(f"Trial100 independent diffusion differs: {feature_error}")
            train_features = reference_features.loc[reference_features.txId.isin(train.txId)].set_index("txId")[EXTRA]
            augmented_train = train.join(train_features, on="txId", validate="one_to_one")
            future = augmented_train.loc[augmented_train.time_step.between(origin+1,stop)]
            if history.index.isin(augmented_train.index).sum() != len(history):
                raise AssertionError("Trial100 historical row population changed")
            hist = augmented_train.loc[augmented_train.time_step.le(origin)]
            if hist[model_columns].isna().any().any() or future[model_columns].isna().any().any():
                raise AssertionError("Trial100 independently reconstructed 211-column model matrix has gaps")
            status = label_status(hist.label)
            weights, factor = hard_status_weights(hist.time_step, status, origin)
            if (not np.isclose(detail.get("unknown_weight_share"), .4, rtol=0, atol=1e-12)
                    or abs(detail.get("unknown_weight_factor", -1)-factor) > 1e-12
                    or detail.get("history_rows") != len(hist)
                    or detail.get("forecast_rows") != len(future)
                    or detail.get("known_forecast_rows") != int(future.label.notna().sum())
                    or detail.get("feature_count") != 211 or detail.get("raw_pca_attributes") != 155
                    or detail.get("pca_components") != 16 or detail.get("propagation_steps") != 20
                    or detail.get("iterations") != 600 or detail.get("class_order") != [0,1,2]):
                raise AssertionError("Trial100 historical status weights or diagnostic counts differ")
            bundle = joblib.load(model_path)
            if (bundle.get("columns") != model_columns or bundle.get("configuration") != plan.get("configuration")
                    or bundle.get("stage") != stage or bundle.get("origin") != origin
                    or bundle.get("forecast_end") != stop or bundle.get("plan_sha256") != EXPECTED_PLAN_SHA):
                raise AssertionError("Trial100 saved HGB 211-column bundle identity changed")
            fitted = bundle.get("model")
            settings = {"max_iter":600,"learning_rate":.03,"max_leaf_nodes":15,
                        "min_samples_leaf":40,"l2_regularization":2.,"max_features":.8,
                        "random_state":42,"early_stopping":False}
            if (not isinstance(fitted, HistGradientBoostingClassifier)
                    or any(fitted.get_params().get(name) != value for name,value in settings.items())
                    or fitted.n_iter_ != 600 or not np.array_equal(fitted.classes_, [0,1,2])):
                raise AssertionError("Trial100 saved HGB learner settings/classes/iterations changed")
            probability = fitted.predict_proba(future[model_columns])
            denom = probability[:,0] + probability[:,1]
            score = np.divide(probability[:,1], denom, out=np.zeros(len(denom)), where=denom>0)
            per_forecast = pd.read_pickle(per_forecast_path)
            if (per_forecast.columns.tolist() != KEY + ["p_known_licit","p_known_illicit","p_unknown",
                                                          "p_known_denominator",STANDALONE]
                    or len(per_forecast) != len(future)
                    or not np.array_equal(per_forecast.txId.to_numpy(), future.txId.to_numpy())
                    or not np.array_equal(per_forecast.time_step.to_numpy(), future.time_step.to_numpy())
                    or not per_forecast.origin.eq(origin).all()):
                raise AssertionError("Trial100 full attributed forecast key/schema changed")
            replay = np.column_stack((probability,denom,score))
            saved_replay = per_forecast[["p_known_licit","p_known_illicit","p_unknown",
                                         "p_known_denominator",STANDALONE]].to_numpy(float)
            error = float(np.max(np.abs(saved_replay-replay)))
            if error > 1e-12:
                raise AssertionError(f"Trial100 saved model probability replay differs: {error}")
            stage_chunk = stage_forecast.loc[stage_forecast.origin.eq(origin)]
            if not per_forecast.reset_index(drop=True).equals(stage_chunk.reset_index(drop=True)):
                raise AssertionError("Trial100 stage-wide forecast cache differs from per-origin file")
            known = per_forecast.loc[per_forecast.label.notna(), KEY+[STANDALONE]]
            control = control99.loc[control99.origin.eq(origin), KEY+[CONTROL]]
            merged = known.merge(control, on=KEY, how="outer", validate="one_to_one", indicator=True)
            if len(merged) != len(known) or not merged._merge.eq("both").all():
                raise AssertionError("Trial100 known-row control key population changed")
            merged[BLEND] = fixed_blend(merged[CONTROL], merged[STANDALONE])
            saved_known = saved_pred.loc[saved_pred.origin.eq(origin), KEY+[STANDALONE,CONTROL,BLEND]]
            check = merged[KEY+[STANDALONE,CONTROL,BLEND]].merge(saved_known,on=KEY,how="outer",
                       validate="one_to_one",indicator=True,suffixes=("_ref","_saved"))
            if len(check) != len(merged) or not check._merge.eq("both").all():
                raise AssertionError("Trial100 saved known-row key population changed")
            for variant in (STANDALONE,CONTROL,BLEND):
                if not np.allclose(check[f"{variant}_ref"], check[f"{variant}_saved"], rtol=0, atol=1e-12):
                    raise AssertionError(f"Trial100 known-row variant differs: {variant}")
            replay_errors.append(error); model_replays += 1
        metrics = reference_metrics(stage, saved_pred)
        summary = reference_summary(stage, metrics)
        compare_table(stage_paths[2], metrics)
        compare_table(stage_paths[3], summary)
        metric_rows += len(metrics)
        if stage == "original" and bool(summary.qualified.any()) != bool(meta.get("original_gate_passed")):
            raise AssertionError("Trial100 original selection gate changed")
        if stage == "confirmation" and bool(summary.qualified.any()) != bool(meta.get("confirmation_gate_passed")):
            raise AssertionError("Trial100 confirmation selection gate changed")
    combined = pd.concat(stage_prediction.values(), ignore_index=True)
    if not pd.read_pickle(REPORTS / f"{P}_predictions.pkl").equals(combined):
        raise AssertionError("Trial100 combined known-row predictions differ")
    compare_table(REPORTS / f"{P}_results.csv", pd.concat(
        (pd.read_csv(REPORTS / f"{P}_{s}_results.csv") for s in stage_prediction),ignore_index=True))
    compare_table(REPORTS / f"{P}_summary.csv", pd.concat(
        (pd.read_csv(REPORTS / f"{P}_{s}_summary.csv") for s in stage_prediction),ignore_index=True))
    original_summary = pd.read_csv(REPORTS / f"{P}_original_summary.csv")
    confirmation_summary = (pd.read_csv(REPORTS / f"{P}_confirmation_summary.csv")
                            if "confirmation" in stage_prediction else None)
    eligible = (sorted(set(original_summary.loc[original_summary.qualified,"variant"]) &
                       set(confirmation_summary.loc[confirmation_summary.qualified,"variant"]))
                if confirmation_summary is not None else [])
    winner = next((v for v in (BLEND,STANDALONE) if v in eligible),None)
    if (meta.get("two_stage_qualified_variants") != eligible
            or meta.get("same_variant_passed_both_stages") is not bool(eligible)
            or meta.get("winner") != winner):
        raise AssertionError("Trial100 strict same-variant two-stage selection changed")
    final_checks = {"winner": None, "candidate_created": False}
    if winner:
        reference_control = rebuild_public99_final(train,test,sample,full_columns)
        raw_columns = [c for c in historical_columns(train.loc[train.time_step.le(35)],full_columns,35) if c.startswith("feat_")]
        scaler = StandardScaler().fit(train.loc[train.time_step.le(35),raw_columns].to_numpy(np.float64))
        pca = PCA(n_components=16,svd_solver="full",whiten=False).fit(
            scaler.transform(train.loc[train.time_step.le(35),raw_columns].to_numpy(np.float64)))
        expected_features, feature_diag = reference_feature_cache(train,test,ids,periods,edges,raw_columns,scaler,pca,35,49)
        saved_features = pd.read_pickle(ROOT / meta["final_graph_features_file"])
        if (saved_features.columns.tolist() != ["txId",*EXTRA,"time_step"]
                or len(saved_features) != len(expected_features)
                or not np.array_equal(saved_features.txId.to_numpy(),expected_features.txId.to_numpy())
                or np.max(np.abs(saved_features[EXTRA].to_numpy()-expected_features[EXTRA].to_numpy())) > 1e-8
                or not check_feature_diagnostics(meta.get("final_feature_diagnostics"), feature_diag)):
            raise AssertionError("Trial100 final all-node diffusion feature artifact differs")
        prep = joblib.load(ROOT / meta["final_preprocess_file"])
        bundle = joblib.load(ROOT / meta["final_model_file"])
        columns = historical_columns(train.loc[train.time_step.le(35)],full_columns,35)+EXTRA
        if (prep.get("cutoff") != 35 or prep.get("plan_sha256") != EXPECTED_PLAN_SHA
                or prep.get("model_columns") != columns
                or bundle.get("columns") != columns or bundle.get("configuration") != plan.get("configuration")
                or bundle.get("plan_sha256") != EXPECTED_PLAN_SHA):
            raise AssertionError("Trial100 final preprocessing/model bundle changed")
        for name in ("mean_", "var_", "scale_"):
            if not np.allclose(getattr(prep["scaler"],name),getattr(scaler,name),rtol=0,atol=1e-9):
                raise AssertionError(f"Trial100 final historical StandardScaler differs: {name}")
        for name in ("components_", "mean_", "singular_values_", "explained_variance_"):
            if not np.allclose(getattr(prep["pca"],name),getattr(pca,name),rtol=1e-10,atol=1e-8):
                raise AssertionError(f"Trial100 final historical PCA differs: {name}")
        final_model = bundle.get("model")
        settings = {"max_iter":600,"learning_rate":.03,"max_leaf_nodes":15,
                    "min_samples_leaf":40,"l2_regularization":2.,"max_features":.8,
                    "random_state":42,"early_stopping":False}
        if (not isinstance(final_model, HistGradientBoostingClassifier)
                or any(final_model.get_params().get(name) != value for name,value in settings.items())
                or final_model.n_iter_ != 600 or not np.array_equal(final_model.classes_,[0,1,2])):
            raise AssertionError("Trial100 final HGB parameters/classes/iterations changed")
        test_features = expected_features.loc[expected_features.txId.isin(test.txId)].set_index("txId")[EXTRA]
        test_matrix = test.join(test_features,on="txId",validate="one_to_one")
        probability = final_model.predict_proba(test_matrix[columns])
        denom = probability[:,0]+probability[:,1]
        challenger = np.divide(probability[:,1],denom,out=np.zeros(len(denom)),where=denom>0)
        blend = fixed_blend(reference_control,challenger)
        chosen = blend if winner == BLEND else challenger
        candidate = pd.read_csv(ROOT / meta["candidate_file"])
        components = pd.read_pickle(ROOT / meta["final_components_file"])
        if (candidate.columns.tolist() != ["index","target"]
                or not np.array_equal(candidate["index"],sample["index"])
                or np.max(np.abs(candidate.target.to_numpy()-chosen)) > 1e-12
                or np.max(np.abs(components[STANDALONE].to_numpy()-challenger)) > 1e-12
                or np.max(np.abs(components[CONTROL].to_numpy()-reference_control)) > 1e-12
                or np.max(np.abs(components[BLEND].to_numpy()-blend)) > 1e-12
                or np.max(np.abs(components[["p_known_licit","p_known_illicit","p_unknown"]].to_numpy()-probability)) > 1e-12
                or np.max(np.abs(components.p_known_denominator.to_numpy()-denom)) > 1e-12
                or sha(ROOT / meta["candidate_file"]) != meta["candidate_sha256"]):
            raise AssertionError("Trial100 final candidate/fixed blend replay differs")
        model_replays += 1
        prior99_member_replays += 3
        final_checks = {"winner": winner, "candidate_created": True,
                        "candidate_rows": len(candidate), "candidate_sha256": meta["candidate_sha256"]}
    elif (meta.get("candidate_created") is not False or meta.get("final_model_fit_performed") is not False
          or (ROOT / "submissions" / f"candidate_{P}.csv").exists()):
        raise AssertionError("Trial100 ineligible experiment created a final candidate")
    audit = {"trial_number":100,"variant":P,"source_sha256":EXPECTED_SOURCE_SHA,
             "plan_sha256":EXPECTED_PLAN_SHA,"frozen_hashes_verified":True,
             "root_authorization_verified":True,"pinned_input_count":len(plan["input_sha256"]),
             "immutable_registry_unique_ids_preflight":PREPARED_REGISTRY_ROWS,
             "prior99_audit_sha256":EXPECTED_PRIOR99_AUDIT_SHA,
             "prior99_result_sha256":EXPECTED_PRIOR99_RESULT_SHA,
             "prior99_audit_registry_row_sha256":EXPECTED_PRIOR99_AUDIT_ROW_SHA,
             "prior99_public_registry_row_sha256":EXPECTED_PRIOR99_PUBLIC_ROW_SHA,
             "graph_period_and_diffusion_verified":True,"historical_scaler_pca_verified":True,
             "public99_exact_member_ancestry_verified":True,"three_status_mass_verified":True,
             "saved_model_replays":model_replays,"prior99_member_replays":prior99_member_replays,
             "maximum_probability_replay_error":max(replay_errors),"metric_rows_recomputed":metric_rows,
             "strict_same_variant_intersection_verified":True,"eligible_variants":eligible,
             "final_artifact_checks":final_checks,"independent_holdout_claim":False,"kaggle_upload":False}
    AUDIT.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    return audit


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--pure-toy"]:
        pure_toy(); print("Trial100 independent diffusion toy passed")
    elif not sys.argv[1:]:
        print(json.dumps(verify_all(),indent=2,sort_keys=True))
    else:
        raise SystemExit("Usage: verifier.py [--pure-toy]")
