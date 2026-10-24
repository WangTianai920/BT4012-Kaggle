"""Independent Trial97 graph, forecast, and artifact audit; never fits a model."""
from __future__ import annotations

import csv
csv.field_size_limit(10_000_000)
import hashlib
import json
import math
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.combined_challengers import load_features
from src.competition import PROJECT_ROOT, REPORT_DIR, load_competition_data
from src.horizon_hgb_20260917 import historical_columns
from src.label_status_multiclass_20260929 import label_status
from src.lr04_increment_logodds_20261002 import CONFIRMATION, ORIGINS

ROOT, REPORTS = PROJECT_ROOT, REPORT_DIR
P = "chain_position_feature80_public90_20261005"
SOURCE, PLAN = ROOT / "src" / f"{P}.py", REPORTS / f"{P}_plan.json"
META, AUTH = REPORTS / f"{P}_metadata.json", REPORTS / f"{P}_root_authorization.json"
AUDIT, FLAT = REPORTS / f"{P}_parent_verification.json", REPORTS / f"{P}_fit_diagnostics.json"
LOG = ROOT / "experiment_log.csv"
GRAPH = REPORTS / "directed_chain_position5_public55_20261005_graph_features.pkl"
NODE_MAP = REPORTS / "directed_chain_position5_public55_20261005_graph_node_map.pkl"
GRAPH_META = REPORTS / "directed_chain_position5_public55_20261005_graph_metadata.json"
PRIOR96_PREFIX = "chain_position_minleaf80_public90_20261005"
PRIOR96_META = REPORTS / f"{PRIOR96_PREFIX}_metadata.json"
PRIOR96_AUDIT = REPORTS / f"{PRIOR96_PREFIX}_parent_verification.json"
PRIOR96_REGISTRY_ID = f"S96_{PRIOR96_PREFIX}_independent_parent_audit"
CONTROL_PREFIX = "directed_chain_position5_public55_20261005"
CONTROL_VARIANT = "blend_75_25_logodds"
CONTROL_CANDIDATE = ROOT / "submissions" / f"candidate_{CONTROL_PREFIX}.csv"
CONTROL_RESULT = REPORTS / f"{CONTROL_PREFIX}_submission_result.json"
CONTROL_AUDIT = REPORTS / f"{CONTROL_PREFIX}_parent_verification.json"
CONTROL_GRAPH = REPORTS / f"{CONTROL_PREFIX}_graph_features.pkl"
CONTROL_NODE_MAP = REPORTS / f"{CONTROL_PREFIX}_graph_node_map.pkl"
CONTROL_GRAPH_META = REPORTS / f"{CONTROL_PREFIX}_graph_metadata.json"
PUBLIC55_PREFIX = "status_mass40_logodds_blend_20261003"
PUBLIC55_CANDIDATE = ROOT / "submissions" / f"candidate_{PUBLIC55_PREFIX}.csv"
CONTROL_REGISTRY_AUDIT_ID = f"S90_{CONTROL_PREFIX}_independent_parent_audit"
CONTROL_REGISTRY_PUBLIC_ID = f"S90_{CONTROL_PREFIX}_public_1005_1"
STANDALONE = "chain_position_feature80_hgb"
BLEND = "blend_75_25_logodds"
CONTROL = "public90_control"
OLD_CONTROL = "public55_control"
CHAIN_COLUMNS = ["dag_log1p_scc_size", "dag_log1p_source_depth", "dag_log1p_sink_depth",
                 "dag_log1p_chain_span", "dag_chain_position"]
BOUNDARY_COLUMNS = ["dag_log1p_nearest_source_hops", "dag_log1p_nearest_sink_hops",
                    "dag_log1p_source_shortest_path_count", "dag_log1p_sink_shortest_path_count"]
GRAPH_COLUMNS = CHAIN_COLUMNS
KEY = ["origin", "txId", "time_step", "label"]
UNKNOWN_SHARE, HALF_LIFE, EPS = .4, 20.0, 1e-12
EXPECTED_SOURCE_SHA = "7d40521d11d8ecf2d2adf00173e5536f7aba89d8f6b468015d60222ae42bd241"
EXPECTED_PLAN_SHA = "d8669e2228a60f1327c65becdd6fd191fa422dc4a3e6f1fc8c948641de6b9606"
EXPECTED_CONTROL_RESULT_SHA = "a3130a96c06cfd7b38806df38709e61f43e9abe61598dfa2a3dff353c45464ce"
EXPECTED_CONTROL_AUDIT_SHA = "04ca4679ad6071c6667ae4a2001422f15f1cf5f605aa8d3e73fff5c565af3d1b"
EXPECTED_CONTROL_CANDIDATE_SHA = "96703d114be7b126e5374c7483d1f799160a55acc4825e1b5c48930b5dabe65e"
EXPECTED_CONTROL_AUDIT_ROW_SHA = "025f0cea2cbee2dbd38795b911ea1e93b5487f8bf162c9da1668b614c08c0130"
EXPECTED_CONTROL_PUBLIC_ROW_SHA = "0e0982e038879d6b7192858c1130b765470b64743e70cec28ca4a0a61fc27416"
EXPECTED_PRIOR96_AUDIT_SHA = "d11230353ee6a26a0ad00461c09dc76ddbc27802990c37e518778ad7d54a9482"
EXPECTED_PRIOR96_AUDIT_ROW_SHA = "b6cd4c72977b0246a66781b905b8492750c2f284c472ff1b8015c35f2c87b02b"

def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_logit(values):
    p = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def fixed_blend(control, challenger):
    a, b = np.asarray(control, float), np.asarray(challenger, float)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError("Invalid Trial90 blend inputs")
    if ((a < 0) | (a > 1) | (b < 0) | (b > 1)).any():
        raise AssertionError("Trial90 blend input is outside [0,1]")
    return expit(.75 * score_logit(a) + .25 * score_logit(b))


def hard_status_weights(steps, status, cutoff):
    status = np.asarray(status, dtype=np.int64)
    raw = np.power(2.0, (np.asarray(steps, float) - float(cutoff)) / HALF_LIFE)
    unknown = status == 2
    known_mass, unknown_mass = raw[~unknown].sum(), raw[unknown].sum()
    if known_mass <= 0 or unknown_mass <= 0:
        raise AssertionError("Empty known or unknown observation status mass")
    factor = float(known_mass * UNKNOWN_SHARE / ((1.0 - UNKNOWN_SHARE) * unknown_mass))
    weights = raw.copy()
    weights[unknown] *= factor
    if abs(weights[unknown].sum() / weights.sum() - UNKNOWN_SHARE) >= 1e-12:
        raise AssertionError("Unknown observation status does not have 40% raw mass")
    return weights, factor


def canonical_row(row: dict) -> bytes:
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")


def check_prior_public_history(plan: dict, auth: dict) -> dict:
    """Bind the live control to its exact positive public result and registry rows."""
    with LOG.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row.get("experiment_id", "") for row in rows]
    if not ids or not all(ids) or len(ids) != len(set(ids)):
        raise AssertionError("Experiment registry has blank or duplicate identities")
    if int(auth.get("registry_unique_ids_preflight", -1)) != 14208 or len(ids) < 14208:
        raise AssertionError("Trial97 immutable preflight registry snapshot changed")
    result, audit = json.loads(CONTROL_RESULT.read_text()), json.loads(CONTROL_AUDIT.read_text())
    if (sha(CONTROL_RESULT) != EXPECTED_CONTROL_RESULT_SHA
            or sha(CONTROL_AUDIT) != EXPECTED_CONTROL_AUDIT_SHA
            or sha(CONTROL_CANDIDATE) != EXPECTED_CONTROL_CANDIDATE_SHA
            or result.get("submission_name") != "1005-1"
            or result.get("public_roc_auc") != .95273
            or result.get("private_roc_auc") is not None
            or result.get("candidate_sha256") != EXPECTED_CONTROL_CANDIDATE_SHA
            or audit.get("eligible_variants") != [CONTROL_VARIANT]
            or audit.get("strict_same_variant_intersection_verified") is not True
            or audit.get("metric_rows_recomputed") != 54
            or audit.get("saved_model_replays") != 9):
        raise AssertionError("Trial90 positive public control changed")
    control = plan.get("public90_control", {})
    closeout = plan.get("trial90_closeout_contract", {})
    if (control.get("submission_name") != "1005-1"
            or control.get("public_roc_auc_user_reported") != .95273
            or control.get("candidate_sha256") != EXPECTED_CONTROL_CANDIDATE_SHA):
        raise AssertionError("Trial97 public90 control metadata changed")
    for key, path in (("candidate_sha256", CONTROL_CANDIDATE),
                      ("public_result_sha256", CONTROL_RESULT),
                      ("parent_audit_sha256", CONTROL_AUDIT)):
        if closeout.get(key) != sha(path) or auth.get(f"trial90_{key}") != sha(path):
            raise AssertionError(f"Trial97 plan does not bind Trial90 {key}")
    if (closeout.get("registry_id") != CONTROL_REGISTRY_AUDIT_ID
            or closeout.get("public_registry_id") != CONTROL_REGISTRY_PUBLIC_ID
            or closeout.get("registry_row_sha256") != EXPECTED_CONTROL_AUDIT_ROW_SHA
            or closeout.get("public_registry_row_sha256") != EXPECTED_CONTROL_PUBLIC_ROW_SHA
            or closeout.get("registry_unique_ids_snapshot") != 13883):
        raise AssertionError("Trial97 plan does not bind exact Trial90 registry rows")
    for identifier, digest, path in (
        (CONTROL_REGISTRY_AUDIT_ID, EXPECTED_CONTROL_AUDIT_ROW_SHA, CONTROL_AUDIT),
        (CONTROL_REGISTRY_PUBLIC_ID, EXPECTED_CONTROL_PUBLIC_ROW_SHA, CONTROL_RESULT),
    ):
        found = [row for row in rows if row["experiment_id"] == identifier]
        if (len(found) != 1 or hashlib.sha256(canonical_row(found[0])).hexdigest() != digest
                or found[0].get("source_file") != str(path.relative_to(ROOT))
                or found[0].get("source_sha256") != sha(path)):
            raise AssertionError(f"Exact registered Trial90 predecessor differs: {identifier}")
    prior96_meta, prior96_audit = (json.loads(path.read_text()) for path in (PRIOR96_META, PRIOR96_AUDIT))
    prior96_rows = [row for row in rows if row["experiment_id"] == PRIOR96_REGISTRY_ID]
    if (sha(PRIOR96_AUDIT) != EXPECTED_PRIOR96_AUDIT_SHA
            or len(prior96_rows) != 1
            or hashlib.sha256(canonical_row(prior96_rows[0])).hexdigest() != EXPECTED_PRIOR96_AUDIT_ROW_SHA
            or prior96_rows[0].get("source_file") != str(PRIOR96_AUDIT.relative_to(ROOT))
            or prior96_rows[0].get("source_sha256") != EXPECTED_PRIOR96_AUDIT_SHA
            or prior96_meta.get("original_gate_passed") is not False
            or prior96_meta.get("confirmation_run") is not False
            or prior96_meta.get("submission_created") is not False
            or prior96_audit.get("eligible_variants") != []
            or prior96_audit.get("metric_rows_recomputed") != 27
            or prior96_audit.get("saved_model_replays") != 5):
        raise AssertionError("Trial96 exact locally negative closeout or registry row changed")
    closeout96 = plan.get("trial96_negative_closeout_contract", {})
    if (closeout96.get("parent_audit_sha256") != EXPECTED_PRIOR96_AUDIT_SHA
            or closeout96.get("registry_id") != PRIOR96_REGISTRY_ID
            or closeout96.get("registry_row_sha256") != EXPECTED_PRIOR96_AUDIT_ROW_SHA
            or closeout96.get("registry_unique_ids_snapshot") != 14208
            or auth.get("trial96_negative_audit_registered") is not True
            or auth.get("trial96_registry_id") != PRIOR96_REGISTRY_ID
            or auth.get("trial96_registry_row_sha256") != EXPECTED_PRIOR96_AUDIT_ROW_SHA
            or int(auth.get("trial96_registry_unique_ids", 0)) < 14208):
        raise AssertionError("Trial97 plan does not bind Trial96 negative closeout")
    for key, path in (("source_sha256", ROOT / "src" / f"{PRIOR96_PREFIX}.py"),
                      ("plan_sha256", REPORTS / f"{PRIOR96_PREFIX}_plan.json"),
                      ("root_authorization_sha256", REPORTS / f"{PRIOR96_PREFIX}_root_authorization.json"),
                      ("metadata_sha256", PRIOR96_META),
                      ("parent_audit_sha256", PRIOR96_AUDIT),
                      ("original_predictions_sha256", REPORTS / f"{PRIOR96_PREFIX}_original_predictions.pkl"),
                      ("original_results_sha256", REPORTS / f"{PRIOR96_PREFIX}_original_results.csv"),
                      ("original_summary_sha256", REPORTS / f"{PRIOR96_PREFIX}_original_summary.csv"),
                      ("fit_diagnostics_sha256", REPORTS / f"{PRIOR96_PREFIX}_fit_diagnostics.json")):
        if closeout96.get(key) != sha(path) or auth.get(f"trial96_{key}") != sha(path):
            raise AssertionError(f"Trial97 Trial96 negative closeout does not bind {key}")
    models = {path.name: sha(path) for path in sorted((ROOT / "models" / PRIOR96_PREFIX).glob("*.joblib"))}
    if (len(models) != 5 or closeout96.get("outer_model_sha256") != models
            or auth.get("trial96_outer_model_sha256") != models):
        raise AssertionError("Trial97 Trial96 closeout does not bind five original models")
    return {"trial90_public_result_sha256": EXPECTED_CONTROL_RESULT_SHA,
            "trial90_parent_audit_sha256": EXPECTED_CONTROL_AUDIT_SHA,
            "trial90_candidate_sha256": EXPECTED_CONTROL_CANDIDATE_SHA,
            "trial90_registry_audit_row_sha256": EXPECTED_CONTROL_AUDIT_ROW_SHA,
            "trial90_registry_public_row_sha256": EXPECTED_CONTROL_PUBLIC_ROW_SHA,
            "trial96_parent_audit_sha256": EXPECTED_PRIOR96_AUDIT_SHA,
            "trial96_registry_audit_row_sha256": EXPECTED_PRIOR96_AUDIT_ROW_SHA,
            "immutable_registry_unique_ids_preflight": 14208}


def reference_graph_features(edges: pd.DataFrame, node_ids) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Rebuild longest/shortest SCC-DAG coordinates with NetworkX and integer path counts."""
    if not {"txId1", "txId2"}.issubset(edges.columns) or edges[["txId1", "txId2"]].isna().any().any():
        raise AssertionError("Official directed edge schema or IDs changed")
    unique_nodes = np.sort(np.unique(np.asarray(node_ids, dtype=np.int64)))
    if len(unique_nodes) != len(node_ids):
        raise AssertionError("Node universe contains duplicate IDs")
    pairs = edges[["txId1", "txId2"]].astype(np.int64).drop_duplicates()
    if not set(pairs.txId1).union(pairs.txId2).issubset(set(unique_nodes)):
        raise AssertionError("Node universe omits graph endpoints")
    loop_count = int(pairs.txId1.eq(pairs.txId2).sum())
    path_pairs = pairs.loc[pairs.txId1.ne(pairs.txId2)]
    graph = nx.DiGraph()
    graph.add_nodes_from(int(x) for x in unique_nodes)
    graph.add_edges_from(path_pairs.itertuples(index=False, name=None))
    components = list(nx.strongly_connected_components(graph))
    component_of = {node: i for i, members in enumerate(components) for node in members}
    if len(component_of) != graph.number_of_nodes():
        raise AssertionError("SCC assignment lost graph nodes")
    dag = nx.DiGraph()
    dag.add_nodes_from(range(len(components)))
    dag.add_edges_from((component_of[u], component_of[v]) for u, v in graph.edges()
                       if component_of[u] != component_of[v])
    if not nx.is_directed_acyclic_graph(dag):
        raise AssertionError("Condensed SCC graph is cyclic")
    order = list(nx.topological_sort(dag))
    n = len(components)
    source_depth = np.zeros(n, dtype=np.int64)
    sink_depth = np.zeros(n, dtype=np.int64)
    source_hops = [None] * n
    sink_hops = [None] * n
    source_counts = [0] * n
    sink_counts = [0] * n
    for u in order:
        if dag.in_degree(u) == 0:
            source_hops[u], source_counts[u] = 0, 1
        for v in dag.successors(u):
            source_depth[v] = max(source_depth[v], source_depth[u] + 1)
            d = source_hops[u] + 1
            if source_hops[v] is None or d < source_hops[v]:
                source_hops[v], source_counts[v] = d, source_counts[u]
            elif d == source_hops[v]:
                source_counts[v] += source_counts[u]
    for u in reversed(order):
        if dag.out_degree(u) == 0:
            sink_hops[u], sink_counts[u] = 0, 1
        for v in dag.successors(u):
            sink_depth[u] = max(sink_depth[u], sink_depth[v] + 1)
            d = sink_hops[v] + 1
            if sink_hops[u] is None or d < sink_hops[u]:
                sink_hops[u], sink_counts[u] = d, sink_counts[v]
            elif d == sink_hops[u]:
                sink_counts[u] += sink_counts[v]
    if any(x is None for x in source_hops + sink_hops) or not all(source_counts) or not all(sink_counts):
        raise AssertionError("Every DAG component must reach source and sink")
    sizes = np.fromiter((len(x) for x in components), dtype=np.int64)
    ci = np.fromiter((component_of[int(node)] for node in unique_nodes), dtype=np.int64)
    sd, td = source_depth[ci], sink_depth[ci]
    total = sd + td
    component_min = np.fromiter((min(members) for members in components), dtype=np.int64)
    ordered = np.argsort(component_min, kind="stable")
    canonical = np.empty(n, dtype=np.int64)
    canonical[ordered] = np.arange(n, dtype=np.int64)
    longest_values = np.column_stack((np.log1p(sizes[ci]), np.log1p(sd), np.log1p(td),
                                      np.log1p(total), sd / (total + 1.0))).astype(np.float32)
    inherited = pd.DataFrame(longest_values, columns=CHAIN_COLUMNS)
    inherited.insert(0, "txId", unique_nodes)
    source_hops_a = np.asarray(source_hops, dtype=np.int64)[ci]
    sink_hops_a = np.asarray(sink_hops, dtype=np.int64)[ci]
    source_log_a = np.fromiter((math.log(source_counts[i]) for i in ci), dtype=np.float64)
    sink_log_a = np.fromiter((math.log(sink_counts[i]) for i in ci), dtype=np.float64)
    values = np.column_stack((np.log1p(source_hops_a), np.log1p(sink_hops_a),
                              np.logaddexp(0.0, source_log_a),
                              np.logaddexp(0.0, sink_log_a))).astype(np.float32)
    boundary = pd.DataFrame(values, columns=BOUNDARY_COLUMNS)
    boundary.insert(0, "txId", unique_nodes)
    node_map = pd.DataFrame({"txId": unique_nodes, "scc_id": canonical[ci],
                             "scc_min_txId": component_min[ci], "scc_size": sizes[ci],
                             "source_depth": sd, "sink_depth": td, "chain_span": total,
                             "nearest_source_hops": source_hops_a,
                             "nearest_sink_hops": sink_hops_a,
                             "source_shortest_path_logcount": source_log_a,
                             "sink_shortest_path_logcount": sink_log_a})
    diagnostics = {"all_graph_nodes": graph.number_of_nodes(),
                   "input_edge_rows": len(edges),
                   "unique_directed_edges_before_self_loop_removal": len(pairs),
                   "duplicate_edge_rows_collapsed": len(edges) - len(pairs),
                   "unique_self_loop_edges_omitted_from_paths": loop_count,
                   "path_edges_after_deduplication_and_self_loop_removal": len(path_pairs),
                   "strongly_connected_components": n,
                   "nontrivial_scc_count": int((sizes > 1).sum()),
                   "largest_scc_size": int(sizes.max()),
                   "condensed_dag_edges": dag.number_of_edges(),
                   "topological_order_nodes": len(order),
                   "condensed_graph_acyclic": True,
                   "maximum_source_depth": int(source_depth.max()),
                   "maximum_sink_depth": int(sink_depth.max()),
                   "maximum_chain_span": int((source_depth + sink_depth).max())}
    return inherited, boundary, node_map, diagnostics


def exact_join(left: pd.DataFrame, right: pd.DataFrame, names: list[str], stage: str) -> pd.DataFrame:
    if left.duplicated(KEY).any() or right.duplicated(KEY).any():
        raise AssertionError(f"{stage} duplicate control ancestry keys")
    merged = left.merge(right[KEY + names], on=KEY, how="outer", validate="one_to_one", indicator=True)
    if len(merged) != len(left) or not merged._merge.eq("both").all():
        raise AssertionError(f"{stage} control ancestry key population changed")
    return merged.drop(columns="_merge")


def check_control_ancestry(stage: str, forecast: pd.DataFrame, control_col: str):
    """Independently rebuild public55 then Trial90 control on exact held-out keys."""
    shrink = pd.read_pickle(REPORTS / f"shrinkage_status_20261001_{stage}_predictions.pkl")
    intermediate = pd.read_pickle(REPORTS / f"intermediate_steps_status_20261002_{stage}_predictions.pkl")
    mass40 = pd.read_pickle(REPORTS / f"status_mass40_lr03_iter600_20261003_{stage}_predictions.pkl")
    merged = shrink[KEY + ["incumbent", "lr03_iter600"]].rename(
        columns={"incumbent": "incumbent_20260929"})
    merged = exact_join(merged, intermediate, ["lr04_iter450"], stage)
    merged = exact_join(merged, mass40, ["status_mass40_lr03_iter600"], stage)
    t39 = expit(.75 * score_logit(merged.incumbent_20260929) +
                .125 * score_logit(merged.lr03_iter600) + .125 * score_logit(merged.lr04_iter450))
    t44 = expit(.95 * score_logit(t39) + .05 * score_logit(merged.lr04_iter450))
    merged["rebuilt55"] = expit(.75 * score_logit(t44) +
                                .25 * score_logit(merged.status_mass40_lr03_iter600))
    official55 = pd.read_pickle(REPORTS / f"{PUBLIC55_PREFIX}_{stage}_predictions.pkl")
    merged = exact_join(merged, official55.rename(columns={BLEND: "official55"}), ["official55"], stage)
    if not np.allclose(merged.rebuilt55, merged.official55, rtol=0, atol=1e-15):
        raise AssertionError(f"{stage} public55 ancestry differs")
    official90 = pd.read_pickle(REPORTS / f"{CONTROL_PREFIX}_{stage}_predictions.pkl")
    merged = exact_join(merged, official90.rename(columns={CONTROL_VARIANT: "official90",
                                                    "directed_chain_position5_hgb": "standalone90"}),
                        ["official90", "standalone90"], stage)
    rebuilt90 = fixed_blend(merged.official55, merged.standalone90)
    if not np.allclose(merged.official90, rebuilt90, rtol=0, atol=1e-15):
        raise AssertionError(f"{stage} Trial90 control differs from public55 plus standalone90")
    checked = exact_join(forecast[KEY + [control_col]],
                         merged[KEY + ["official90"]], ["official90"], stage)
    if not np.allclose(checked[control_col], checked.official90, rtol=0, atol=0):
        raise AssertionError(f"{stage} Trial97 public90 keyed control differs")


def recompute_metrics(stage: str, forecast: pd.DataFrame, variants: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for origin, part in forecast.groupby("origin", sort=False):
        first = part.loc[part.time_step.le(origin + 7)]
        slices = [("first7", first)]
        if int(part.time_step.max()) == int(origin) + 14:
            slices += [("full14", part), ("second7", part.loc[part.time_step.gt(origin + 7)])]
        for name, data in slices:
            y = data.label.to_numpy(int)
            for variant in variants:
                rows.append({"stage": stage, "origin": int(origin), "slice": name,
                             "variant": variant, "n": len(data), "positives": int(y.sum()),
                             "auc": float(roc_auc_score(y, data[variant])),
                             "average_precision": float(average_precision_score(y, data[variant]))})
    return pd.DataFrame(rows)


def recompute_summary(stage: str, metrics: pd.DataFrame, control: str,
                      candidates: tuple[str, ...]) -> pd.DataFrame:
    pivot = metrics.pivot(index=["origin", "slice"], columns="variant", values="auc")
    groups = ({"full14": [(14, "full14"), (21, "full14")],
               "second7": [(14, "second7"), (21, "second7")],
               "primary7": [(o, "first7") for o in (14, 21, 28)],
               "rolling7": [(o, "first7") for o in (17, 24)]} if stage == "original" else
              {s: [(o, s) for o in CONFIRMATION] for s in ("full14", "second7", "first7")})
    rows = []
    for variant in candidates:
        delta = pivot[variant] - pivot[control]
        row = {"stage": stage, "variant": variant, "control_variant": control}
        for name, keys in groups.items():
            row[f"{name}_mean_auc"] = float(pivot.loc[keys, variant].mean())
            row[f"{name}_mean_delta"] = float(delta.loc[keys].mean())
            row[f"{name}_control_auc"] = float(pivot.loc[keys, control].mean())
        row["worst_slice_delta"] = float(delta.min())
        good = (all(row[f"{name}_mean_delta"] > 0 for name in groups) if stage == "original" else
                row["full14_mean_delta"] > 0 and row["second7_mean_delta"] > 0 and
                row["first7_mean_delta"] >= 0)
        row["qualified"] = bool(good and row["worst_slice_delta"] >= -.001)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_table(path: Path, expected: pd.DataFrame):
    """Compare full table including NaN cells, column order, and row order."""
    saved = pd.read_csv(path)
    if saved.columns.tolist() != expected.columns.tolist():
        raise AssertionError(f"Saved table columns differ: {path.name}")
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), expected.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=0, atol=1e-12)


def check_frozen_identity():
    """Refuse data/cache/model reads until source, plan, and authorization agree."""
    if EXPECTED_SOURCE_SHA == "PENDING_FINAL_FREEZE" or EXPECTED_PLAN_SHA == "PENDING_FINAL_FREEZE":
        raise RuntimeError("Trial97 verifier awaits final frozen source and plan hashes")
    source_hash, plan_hash = sha(SOURCE), sha(PLAN)
    if source_hash != EXPECTED_SOURCE_SHA or plan_hash != EXPECTED_PLAN_SHA:
        raise AssertionError("Trial97 frozen source/plan hashes changed")
    if not AUTH.exists():
        raise RuntimeError("Trial97 root authorization must exist before audit loads")
    plan, auth, meta = (json.loads(path.read_text()) for path in (PLAN, AUTH, META))
    if (plan.get("trial_number") != 97 or plan.get("variant") != P
            or plan.get("source_sha256") != source_hash
            or auth.get("authorized") is not True
            or auth.get("source_sha256") != source_hash or auth.get("plan_sha256") != plan_hash
            or auth.get("trial90_positive_audit_acknowledged") is not True
            or auth.get("trial96_negative_audit_acknowledged") is not True
            or meta.get("trial_number") != 97 or meta.get("variant") != P
            or meta.get("source_sha256") != source_hash or meta.get("plan_sha256") != plan_hash
            or meta.get("input_sha256") != plan.get("input_sha256")):
        raise AssertionError("Trial97 plan, authorization, or metadata identity changed")
    config = plan.get("configuration", {})
    for key, value in {"learning_rate": .03, "max_iter": 600, "max_leaf_nodes": 15,
                       "min_samples_leaf": 40, "max_features": .8, "l2_regularization": 2.0,
                       "half_life": 20.0, "unknown_status_mass": .4,
                       "base_feature_count": 172, "chain_feature_count": 5,
                       "total_feature_count": 177, "row_triplication": False,
                       "label_smoothing_alpha": 0.0}.items():
        if config.get(key) != value:
            raise AssertionError(f"Trial97 frozen HGB/status contract changed: {key}")
    graph_contract = plan.get("graph_feature_contract", {})
    if (graph_contract.get("columns") != GRAPH_COLUMNS
            or graph_contract.get("selection") != "retain all five Trial90 graph columns by exact name"
            or graph_contract.get("model_input") !=
               "172 cutoff-safe historical columns plus five frozen Trial90 graph columns = 177 columns"
            or graph_contract.get("output_dtype") != "float32"
            or graph_contract.get("node_universe") !=
               "all 203769 Trial90 graph nodes including graph-only edge endpoints; preserve sorted txId order"
            or plan.get("candidate", {}).get("standalone_variant") != STANDALONE
            or plan.get("candidate", {}).get("blend_variant") != BLEND
            or plan.get("candidate", {}).get("winner_preference") != [BLEND, STANDALONE]
            or plan.get("candidate", {}).get("blend_weight_trial97") != .25
            or plan.get("candidate", {}).get("blend_weight_public90") != .75
            or plan.get("prior_max_features_comparison") != {
                "historical_value_tested": .8,
                "closest_historical_configuration": {
                    "learning_rate": .03, "max_iter": 600, "l2_regularization": 2.0,
                    "max_leaf_nodes": 7, "min_samples_leaf": 40, "feature_count": 172},
                "current_fixed_configuration": {
                    "max_features": .8, "max_leaf_nodes": 15, "min_samples_leaf": 40,
                    "feature_count": 177, "unknown_status_mass": .4,
                    "trial90_chain_features": True},
                "novelty_claim":
                    "test max_features=0.8 in the exact Trial90 five-feature/status-mass40 177-input setup; do not claim the parameter value itself is novel"}):
        raise AssertionError("Trial97 fixed graph, blend, or historical comparison contract changed")
    pins = plan.get("input_sha256")
    if not isinstance(pins, dict) or len(pins) != 406 or "experiment_log.csv" in pins:
        raise AssertionError("Trial97 ancestry input pin inventory changed")
    for relative_path, digest in pins.items():
        if sha(ROOT / relative_path) != digest:
            raise AssertionError(f"Trial97 pinned input changed: {relative_path}")
    predecessor = check_prior_public_history(plan, auth)
    if json.loads(FLAT.read_text()) != meta.get("fit_diagnostics", []):
        raise AssertionError("Trial97 flat fit diagnostics differ from metadata")
    return plan, auth, meta, predecessor


def verify_graph_artifact(train: pd.DataFrame, test: pd.DataFrame, edges: pd.DataFrame,
                          meta: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Independently reconstruct the pinned Trial90 five-column graph cache."""
    all_ids = np.sort(np.unique(np.concatenate((
        train.txId.to_numpy(np.int64), test.txId.to_numpy(np.int64),
        edges.txId1.to_numpy(np.int64), edges.txId2.to_numpy(np.int64)))))
    inherited, _, raw_reference, diagnostics = reference_graph_features(edges, all_ids)
    saved, raw_saved = pd.read_pickle(GRAPH), pd.read_pickle(NODE_MAP)
    raw_fields = ["txId", "scc_id", "scc_min_txId", "scc_size",
                  "source_depth", "sink_depth", "chain_span"]
    if (saved.columns.tolist() != ["txId", *GRAPH_COLUMNS]
            or raw_saved.columns.tolist() != raw_fields
            or len(saved) != 203769 or len(raw_saved) != 203769
            or not np.array_equal(saved.txId.to_numpy(), all_ids)
            or not np.array_equal(raw_saved.txId.to_numpy(), all_ids)
            or not np.array_equal(raw_saved[raw_fields].to_numpy(),
                                  raw_reference[raw_fields].to_numpy())
            or saved[GRAPH_COLUMNS].dtypes.astype(str).tolist() != ["float32"] * 5
            or not np.array_equal(saved[GRAPH_COLUMNS].to_numpy(),
                                  inherited[GRAPH_COLUMNS].to_numpy())):
        raise AssertionError("Trial97 inherited all-node SCC graph cache differs from NetworkX reference")
    graph_meta = json.loads(GRAPH_META.read_text())
    for key, value in diagnostics.items():
        if graph_meta.get(key) != value:
            raise AssertionError(f"Trial90 graph diagnostic differs: {key}")
    if (graph_meta.get("feature_columns") != GRAPH_COLUMNS
            or graph_meta.get("feature_count") != 5
            or graph_meta.get("uses_labels") is not False
            or graph_meta.get("uses_tabular_feature_values") is not False
            or graph_meta.get("feature_artifact_sha256") != sha(GRAPH)
            or graph_meta.get("node_map_sha256") != sha(NODE_MAP)
            or meta.get("graph_features_sha256") != sha(GRAPH)
            or meta.get("graph_node_map_sha256") != sha(NODE_MAP)
            or meta.get("graph_metadata_sha256") != sha(GRAPH_META)):
        raise AssertionError("Trial97 reused graph cache hash/label-free contract differs")
    joined = []
    for name, frame in (("train", train), ("test", test)):
        if frame.txId.duplicated().any():
            raise AssertionError(f"Trial97 {name} transaction IDs duplicate")
        enriched = frame.merge(saved, on="txId", how="left", validate="one_to_one", sort=False)
        if (len(enriched) != len(frame)
                or not np.array_equal(enriched.txId.to_numpy(), frame.txId.to_numpy())
                or enriched[GRAPH_COLUMNS].isna().any().any()):
            raise AssertionError(f"Trial97 {name} inherited graph join lost rows, order, or features")
        joined.append(enriched)
    return joined[0], joined[1], diagnostics


def replay_model(detail: dict, train: pd.DataFrame, test: pd.DataFrame,
                 full_columns: list[str], plan: dict, predictions: pd.DataFrame | None = None):
    stage = detail.get("stage")
    if stage not in ("original", "confirmation", "final"):
        raise AssertionError("Trial97 model diagnostic stage is invalid")
    cutoff = int(detail["origin"])
    stop = detail["forecast_end"]
    history = train.loc[train.time_step.le(cutoff)]
    future = test if stage == "final" else train.loc[train.time_step.between(cutoff + 1, int(stop))]
    columns = historical_columns(train, full_columns, cutoff)
    chain_columns = GRAPH_COLUMNS
    base_columns = [col for col in columns if col not in chain_columns]
    if (len(columns) != 177 or len(base_columns) != 172
            or columns[-5:] != chain_columns or not np.isfinite(future[columns]).all().all()):
        raise AssertionError("Trial97 cutoff-safe 172+5 model input changed")
    status = label_status(history.label)
    weights, factor = hard_status_weights(history.time_step, status, cutoff)
    counts = np.bincount(status, minlength=3).tolist()
    masses = np.bincount(status, weights=weights, minlength=3)
    path = ROOT / detail["model_file"]
    if sha(path) != detail["model_sha256"]:
        raise AssertionError("Trial97 cached HGB model hash changed")
    bundle = joblib.load(path)
    exact_fields = {"base_columns": base_columns, "chain_columns": chain_columns,
                    "columns": columns, "configuration": plan["configuration"],
                    "stage": stage, "origin": cutoff, "forecast_end": stop,
                    "original_class_counts": counts,
                    "source_sha256": sha(SOURCE), "plan_sha256": sha(PLAN),
                    "graph_features_sha256": sha(GRAPH)}
    for key, expected in exact_fields.items():
        if bundle.get(key) != expected:
            raise AssertionError(f"Trial97 cached HGB bundle differs: {key}")
    for item in (bundle, detail):
        if (item.get("base_columns") != base_columns
                or item.get("chain_columns") != chain_columns
                or item.get("columns") != columns
                or item.get("configuration") != plan["configuration"]
                or item.get("graph_features_sha256") != sha(GRAPH)
                or not np.allclose(item.get("original_class_weight_mass", item.get("class_weight_mass")),
                                   masses, rtol=1e-12, atol=1e-12)
                or abs(float(item["raw_hgb_weights_mean"]) - weights.mean()) > 1e-12
                or abs(float(item["unknown_weight_share"]) - .4) > 1e-12
                or abs(float(item["unknown_weight_factor"]) - factor) > 1e-12):
            raise AssertionError("Trial97 cached HGB status weights or feature diagnostics differ")
    expected_counts = detail.get("class_counts") if stage == "final" else detail.get("original_class_counts")
    if expected_counts != counts or (stage != "final" and (detail.get("feature_count") != 177
             or detail.get("iteration_count") != 600
             or detail.get("history_rows") != len(history)
             or detail.get("future_rows") != len(future)
             or detail.get("history_max_step") != cutoff)):
        raise AssertionError("Trial97 cached diagnostic counts/horizon differ")
    estimator = bundle.get("model")
    if not isinstance(estimator, HistGradientBoostingClassifier):
        raise AssertionError("Trial97 cached learner is not HGB")
    parameters = {"max_iter": 600, "learning_rate": .03, "max_leaf_nodes": 15,
                  "min_samples_leaf": 40, "l2_regularization": 2.0, "max_features": .8,
                  "early_stopping": False, "random_state": 42}
    if (any(estimator.get_params().get(key) != expected for key, expected in parameters.items())
            or not np.array_equal(estimator.classes_, [0, 1, 2])
            or int(estimator.n_iter_) != 600 or int(detail["iteration_count"]) != 600):
        raise AssertionError("Trial97 cached HGB parameters/class order/iterations differ")
    probability = estimator.predict_proba(future[columns])
    if probability.shape != (len(future), 3) or not np.isfinite(probability).all():
        raise AssertionError("Trial97 replayed class probabilities are invalid")
    denominator = probability[:, 0] + probability[:, 1]
    if (denominator <= 0).any():
        raise AssertionError("Trial97 conditional fraud denominator vanished")
    score = probability[:, 1] / denominator
    replay_error = 0.0
    if stage != "final":
        if predictions is None:
            raise AssertionError("Trial97 outer replay needs exact saved forecast rows")
        expected = future.loc[future.label.notna()]
        actual = predictions.loc[predictions.origin.eq(cutoff)]
        if (len(actual) != len(expected)
                or detail.get("known_forecast_rows") != len(expected)
                or not np.array_equal(actual.txId.to_numpy(), expected.txId.to_numpy())
                or not np.array_equal(actual.time_step.to_numpy(), expected.time_step.to_numpy())
                or not np.array_equal(actual.label.to_numpy(), expected.label.to_numpy())):
            raise AssertionError("Trial97 cached forecast row population differs")
        replay_error = float(np.max(np.abs(actual[STANDALONE].to_numpy() -
                                               score[future.label.notna().to_numpy()])))
        if replay_error > 1e-12:
            raise AssertionError("Trial97 cached forecast differs from saved HGB replay")
    return score, replay_error


def verify_all() -> dict:
    plan, auth, meta, predecessor = check_frozen_identity()
    if any((REPORTS / f"{P}_{suffix}").exists() for suffix in
           ("graph_features.pkl", "graph_node_map.pkl", "graph_metadata.json")):
        raise AssertionError("Trial97 unexpectedly created a new graph artifact")
    train, test, sample, groups = load_features()
    edges = pd.read_csv(ROOT / "data" / "txs_edgelist.csv")
    train, test, graph_diagnostics = verify_graph_artifact(train, test, edges, meta)
    flat = meta["fit_diagnostics"]
    graph_meta = json.loads(GRAPH_META.read_text())
    expected_cache_diag = {"stage": "trial90_cached_chain_position_features",
                           "source_graph_features_sha256": sha(GRAPH),
                           "source_graph_node_map_sha256": sha(NODE_MAP),
                           "source_graph_metadata_sha256": sha(GRAPH_META),
                           "selected_columns": GRAPH_COLUMNS, "node_count": 203769}
    if flat[:1] != [expected_cache_diag]:
        raise AssertionError("Trial97 saved inherited-graph diagnostic changed")
    full_columns = list(groups["regularized_two_hop"]) + list(graph_meta["feature_columns"])
    if len(groups["regularized_two_hop"]) != 182 or len(full_columns) != 187:
        raise AssertionError("Trial97 182 source plus five inherited graph columns changed")
    variants = (CONTROL, STANDALONE, BLEND)
    candidates = variants[1:]
    stage_frames, results, summaries, gates = {}, [], [], {}
    replay_count, replay_max_error = 0, 0.0
    for stage, origins in (("original", ORIGINS), ("confirmation", CONFIRMATION)):
        if stage == "confirmation" and not gates["original"].qualified.any():
            for suffix in ("predictions.pkl", "results.csv", "summary.csv"):
                if (REPORTS / f"{P}_confirmation_{suffix}").exists():
                    raise AssertionError("Trial97 original miss has confirmation artifacts")
            continue
        path = REPORTS / f"{P}_{stage}_predictions.pkl"
        frame = pd.read_pickle(path)
        if (set(frame.columns) != set(KEY + list(variants)) or frame.duplicated(KEY).any()
                or set(frame.origin.unique()) != set(origins)):
            raise AssertionError(f"Trial97 {stage} forecast keys, columns, or origins changed")
        expected_keys = []
        for origin, stop in origins.items():
            heldout = train.loc[train.time_step.between(origin + 1, stop) & train.label.notna()]
            expected_keys.extend([(origin, row.txId, row.time_step, row.label)
                                  for row in heldout.itertuples()])
        if list(frame[KEY].itertuples(index=False, name=None)) != expected_keys:
            raise AssertionError(f"Trial97 {stage} held-out forecast rows differ")
        check_control_ancestry(stage, frame, variants[0])
        blended = fixed_blend(frame[variants[0]], frame[variants[1]])
        if not np.allclose(frame[variants[2]], blended, rtol=0, atol=1e-15):
            raise AssertionError(f"Trial97 {stage} fixed 75/25 logit blend changed")
        details = [d for d in flat if d.get("stage") == stage]
        if [(int(d["origin"]), int(d["forecast_end"])) for d in details] != list(origins.items()):
            raise AssertionError(f"Trial97 {stage} saved model diagnostics are incomplete or out of order")
        for detail in details:
            _, error = replay_model(detail, train, test, full_columns, plan, frame)
            replay_max_error = max(replay_max_error, error)
            replay_count += 1
        computed = recompute_metrics(stage, frame, variants)
        compare_table(REPORTS / f"{P}_{stage}_results.csv", computed)
        gate = recompute_summary(stage, computed, variants[0], candidates)
        compare_table(REPORTS / f"{P}_{stage}_summary.csv", gate)
        stage_frames[stage], gates[stage] = frame, gate
        results.append(computed)
        summaries.append(gate)
    original_pass = bool(gates["original"].qualified.any())
    confirmation_run = "confirmation" in gates
    confirmation_pass = bool(gates["confirmation"].qualified.any()) if confirmation_run else False
    eligible = (sorted(set(gates["original"].loc[gates["original"].qualified, "variant"]) &
                       set(gates["confirmation"].loc[gates["confirmation"].qualified, "variant"]))
                if confirmation_run else [])
    if (meta.get("original_gate_passed") is not original_pass
            or meta.get("confirmation_run") is not confirmation_run
            or meta.get("confirmation_gate_passed") is not confirmation_pass
            or meta.get("qualified_variants_both_stages") != eligible
            or meta.get("same_variant_passed_both_stages") is not bool(eligible)
            or meta.get("submission_created") is not bool(eligible)
            or meta.get("final_model_fit_performed") is not bool(eligible)
            or meta.get("kaggle_upload") is not False):
        raise AssertionError("Trial97 metadata differs from strict same-variant gates")
    compare_table(REPORTS / f"{P}_results.csv", pd.concat(results, ignore_index=True))
    compare_table(REPORTS / f"{P}_summary.csv", pd.concat(summaries, ignore_index=True))
    final_checks = {"submission_created": False, "candidate_rows": 0}
    if eligible:
        winner = next(name for name in (variants[2], variants[1]) if name in eligible)
        if meta.get("winner") != winner:
            raise AssertionError("Trial97 winner differs from fixed blend-first rule")
        final_details = [d for d in flat if d.get("stage") == "final"]
        if len(final_details) != 1:
            raise AssertionError("Trial97 eligible trial lacks one final model")
        standalone, error = replay_model(final_details[0], train, test, full_columns, plan)
        replay_max_error = max(replay_max_error, error)
        replay_count += 1
        control = pd.read_csv(CONTROL_CANDIDATE)
        old_control = pd.read_csv(PUBLIC55_CANDIDATE)
        prior_components = pd.read_pickle(REPORTS / f"{CONTROL_PREFIX}_final_components.pkl")
        components = pd.read_pickle(ROOT / meta["final_components_file"])
        candidate = pd.read_csv(ROOT / meta["candidate_file"])
        index = sample["index"].to_numpy()
        if (len(index) != 15329 or not np.array_equal(test["index"].to_numpy(), index)
                or control.columns.tolist() != ["index", "target"]
                or old_control.columns.tolist() != ["index", "target"]
                or prior_components.columns.tolist() != ["index", OLD_CONTROL,
                                                          "directed_chain_position5_hgb", BLEND]
                or components.columns.tolist() != ["index", *variants]
                or candidate.columns.tolist() != ["index", "target"]
                or any(not np.array_equal(frame["index"].to_numpy(), index)
                       for frame in (old_control, prior_components, control, components, candidate))):
            raise AssertionError("Trial97 final components/candidate/sample order differ")
        rebuilt90 = fixed_blend(old_control.target, prior_components.directed_chain_position5_hgb)
        if (not np.allclose(prior_components[OLD_CONTROL], old_control.target, rtol=0, atol=1e-15)
                or not np.allclose(prior_components[BLEND], rebuilt90, rtol=0, atol=1e-12)
                or not np.allclose(control.target, rebuilt90, rtol=0, atol=1e-12)):
            raise AssertionError("Trial90 final public control differs from frozen public55 plus standalone")
        blend = fixed_blend(control.target, standalone)
        selected = blend if winner == variants[2] else standalone
        if (not np.allclose(components[variants[0]], control.target, rtol=0, atol=1e-15)
                or not np.allclose(components[variants[1]], standalone, rtol=0, atol=1e-12)
                or not np.allclose(components[variants[2]], blend, rtol=0, atol=1e-12)
                or not np.allclose(candidate.target, selected, rtol=0, atol=1e-12)
                or not np.isfinite(candidate.target).all()
                or not candidate.target.between(0, 1).all()
                or meta.get("candidate_rows") != len(candidate)
                or sha(ROOT / meta["candidate_file"]) != meta.get("candidate_sha256")
                or sha(ROOT / meta["final_components_file"]) != meta.get("final_components_sha256")
                or sha(ROOT / meta["final_model_file"]) != meta.get("final_model_sha256")):
            raise AssertionError("Trial97 final replay, component export, or candidate hash differs")
        final_checks = {"submission_created": True, "winner": winner,
                        "candidate_rows": len(candidate), "candidate_sha256": meta["candidate_sha256"],
                        "final_components_sha256": meta["final_components_sha256"],
                        "final_model_sha256": meta["final_model_sha256"]}
    else:
        if (any(d.get("stage") == "final" for d in flat)
                or (ROOT / "submissions" / f"candidate_{P}.csv").exists()
                or (REPORTS / f"{P}_final_components.pkl").exists()
                or (ROOT / "models" / P / "final_through35_model.joblib").exists()):
            raise AssertionError("Trial97 negative result has final model/candidate artifacts")
    audit = {"trial_number": 97, "variant": P, "source_sha256": sha(SOURCE),
             "plan_sha256": sha(PLAN), "pinned_input_count": len(plan["input_sha256"]),
             "frozen_hashes_verified": True, "root_authorization_verified": True,
             "prior_public_history_and_registry_verified": predecessor,
             "graph_features_verified": True, "graph_node_count": graph_diagnostics["all_graph_nodes"],
             "graph_features_sha256": sha(GRAPH), "graph_node_map_sha256": sha(NODE_MAP),
             "graph_metadata_sha256": sha(GRAPH_META),
             "public90_exact_keyed_ancestry_verified": True,
             "original_gate_passed": original_pass, "confirmation_gate_passed": confirmation_pass,
             "eligible_variants": eligible, "strict_same_variant_intersection_verified": True,
             "metric_rows_recomputed": sum(len(frame) for frame in results),
             "saved_model_replays": replay_count, "max_model_replay_abs_error": replay_max_error,
             "reused_overlapping_windows_acknowledged": True,
             "independent_holdout_claim": False, "final_artifact_checks": final_checks,
             "kaggle_upload": False}
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


if __name__ == "__main__":
    print(json.dumps(verify_all(), indent=2, sort_keys=True))
