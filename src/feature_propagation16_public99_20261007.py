"""Frozen Trial100 preparation: cutoff-safe PCA feature propagation plus HGB.

This module defines the preparation contract and reusable transformations. It does
not load competition data or fit a model on import. The execution entry point is
locked until a separate root authorization matches the frozen source and plan.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import csv
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.combined_challengers import load_features
from src.competition import PROJECT_ROOT, REPORT_DIR, load_competition_data
from src.horizon_hgb_20260917 import historical_columns, model as historical_model, weights as recency_weights
from src.label_status_multiclass_20260929 import label_status
from src.status_mass40_lr03_iter600_20261003 import config as learner_config, training_weights_40
from src.missing_aware_directed_gnn_public90_20261006 import weak_component_period_map

PREFIX = "feature_propagation16_public99_20261007"
PLAN_PATH = REPORT_DIR / f"{PREFIX}_plan.json"
AUTH_PATH = REPORT_DIR / f"{PREFIX}_root_authorization.json"
TRIAL90 = "directed_chain_position5_public55_20261005"
TRIAL99 = "chain_position_seedensemble3_public90_20261007"
TRIAL99_VARIANT = "blend_75_25_logodds"
TRIAL99_ENSEMBLE = "chain_position_seedensemble3_conditional_logit"
PUBLIC_NAME = "1007-1"
PUBLIC_AUC = 0.95280
PREVIOUS_PUBLIC_NAME = "1005-1"
PREVIOUS_PUBLIC_AUC = 0.95273
N_COMPONENTS = 16
PROPAGATION_STEPS = 20
GRAPH_NODE_COUNT = 203_769
WEAK_COMPONENT_COUNT = 49
GRAPH5 = ["dag_log1p_scc_size", "dag_log1p_source_depth", "dag_log1p_sink_depth",
          "dag_log1p_chain_span", "dag_chain_position"]
BASE_COUNT = 177
PROPAGATED_COUNT = 32
SUPPORT_COUNT = 2
MODEL_FEATURE_COUNT = 211
EPS = 1e-12
ORIGINS = {14: 28, 21: 35, 17: 24, 24: 31, 28: 35}
CONFIRMATION = {17: 31, 18: 32, 20: 34}
ORIGINAL_VARIANTS = ("feature_propagation_hgb", "blend_75_25_logodds")
KEY = ["origin", "txId", "time_step", "label"]
MODEL_DIR = PROJECT_ROOT / "models" / PREFIX
TRIAL90_GRAPH = REPORT_DIR / f"{TRIAL90}_graph_features.pkl"
TRIAL99_CANDIDATE = PROJECT_ROOT / "submissions" / f"candidate_{TRIAL99}.csv"
TRIAL99_STAGE_PATHS = {stage: REPORT_DIR / f"{TRIAL99}_{stage}_member_forecasts.pkl"
                       for stage in ("original", "confirmation")}
TRIAL90_STAGE_PATHS = {stage: REPORT_DIR / f"{TRIAL90}_{stage}_predictions.pkl"
                       for stage in ("original", "confirmation")}
OUTPUT_NAMES = [f"{PREFIX}_{suffix}" for suffix in (
    "original_predictions.pkl", "original_forecast_probabilities.pkl", "original_results.csv",
    "original_summary.csv", "confirmation_predictions.pkl", "confirmation_forecast_probabilities.pkl",
    "confirmation_results.csv", "confirmation_summary.csv", "predictions.pkl", "results.csv",
    "summary.csv", "fit_diagnostics.json", "metadata.json", "final_components.pkl")]
MODEL_DIR = PROJECT_ROOT / "models" / PREFIX
TRIAL90_GRAPH = REPORT_DIR / f"{TRIAL90}_graph_features.pkl"
TRIAL99_DIR = PROJECT_ROOT / "models" / TRIAL99
TRIAL99_CANDIDATE = PROJECT_ROOT / "submissions" / f"candidate_{TRIAL99}.csv"
TRIAL99_STAGE_PATHS = {stage: REPORT_DIR / f"{TRIAL99}_{stage}_member_forecasts.pkl"
                       for stage in ("original", "confirmation")}
TRIAL90_STAGE_PATHS = {stage: REPORT_DIR / f"{TRIAL90}_{stage}_predictions.pkl"
                       for stage in ("original", "confirmation")}
OUTPUT_NAMES = [f"{PREFIX}_{suffix}" for suffix in (
    "original_predictions.pkl", "original_forecast_probabilities.pkl", "original_results.csv",
    "original_summary.csv", "confirmation_predictions.pkl", "confirmation_forecast_probabilities.pkl",
    "confirmation_results.csv", "confirmation_summary.csv", "predictions.pkl", "results.csv",
    "summary.csv", "fit_diagnostics.json", "metadata.json", "final_components.pkl")]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditional_score(probability: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return conditional fraud probability and the known-status denominator."""
    p = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(classes)
    if p.ndim != 2 or p.shape[1] != len(labels):
        raise ValueError("Probability matrix and class order disagree")
    if not np.array_equal(labels, np.array([0, 1, 2])):
        raise ValueError("Expected fixed class order [licit, illicit, unknown]")
    denominator = p[:, 0] + p[:, 1]
    score = np.divide(p[:, 1], denominator, out=np.zeros(len(p)), where=denominator > 0)
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Conditional fraud score is invalid")
    return score, denominator


def select_winner(original: pd.DataFrame, confirmation: pd.DataFrame) -> tuple[str | None, list[str]]:
    a = set(original.loc[original.qualified, "variant"])
    b = set(confirmation.loc[confirmation.qualified, "variant"])
    common = a & b
    for variant in ("blend_75_25_logodds", "feature_propagation_hgb"):
        if variant in common:
            return variant, sorted(common)
    return None, sorted(common)


def propagated_and_directional_features(
    graph_ids: np.ndarray,
    graph_periods: np.ndarray,
    edges: pd.DataFrame,
    attributed_ids: np.ndarray,
    attributed_pca: np.ndarray,
    *,
    steps: int = PROPAGATION_STEPS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return propagated values and incoming/outgoing means for full graph IDs.

    Each edge txId1->txId2 is used undirected for the symmetric normalized
    clamped propagation. Direction is retained separately for the two means.
    Graph-only initial values are zero and stay free; observed PCA vectors are
    reset after every propagation update. Components are processed by period.
    """
    ids = np.asarray(graph_ids, dtype=np.int64)
    periods = np.asarray(graph_periods, dtype=np.int16)
    known_ids = np.asarray(attributed_ids, dtype=np.int64)
    known = np.asarray(attributed_pca, dtype=np.float64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or not np.array_equal(ids, np.sort(ids)):
        raise ValueError("Graph node IDs must be unique sorted int64 values")
    if periods.shape != ids.shape or known.shape != (len(known_ids), N_COMPONENTS):
        raise ValueError("Graph periods or PCA observations are not aligned")
    if len(np.unique(known_ids)) != len(known_ids) or not np.isfinite(known).all():
        raise ValueError("Attributed PCA node IDs/vectors are invalid")
    if not {"txId1", "txId2"}.issubset(edges.columns):
        raise ValueError("Edge frame must contain txId1 and txId2")
    pos = pd.Index(ids)
    src_all = pos.get_indexer(edges.txId1.to_numpy(dtype=np.int64))
    dst_all = pos.get_indexer(edges.txId2.to_numpy(dtype=np.int64))
    if (src_all < 0).any() or (dst_all < 0).any():
        raise ValueError("Graph node universe omits an edge endpoint")
    keep = src_all != dst_all
    src_all, dst_all = src_all[keep], dst_all[keep]
    # Treat the supplied directed edge list as an undirected simple graph for FP.
    pairs = np.unique(np.column_stack((src_all, dst_all)), axis=0)
    src, dst = pairs[:, 0], pairs[:, 1]
    if len(src) and (periods[src] != periods[dst]).any():
        raise ValueError("A graph edge crosses the inferred time period")
    known_pos = pos.get_indexer(known_ids)
    if (known_pos < 0).any() or len(np.unique(known_pos)) != len(known_pos):
        raise ValueError("An attributed node is missing or duplicated in graph IDs")
    obs_mask = np.zeros(len(ids), dtype=bool)
    obs_mask[known_pos] = True
    x = np.zeros((len(ids), N_COMPONENTS), dtype=np.float64)
    x[known_pos] = known
    h = x.copy()
    in_mean = np.zeros_like(h)
    out_mean = np.zeros_like(h)
    in_degree = np.zeros(len(ids), dtype=np.int64)
    out_degree = np.zeros(len(ids), dtype=np.int64)
    in_graph_only = np.zeros(len(ids), dtype=np.float64)
    out_graph_only = np.zeros(len(ids), dtype=np.float64)
    for period in np.unique(periods):
        nodes = np.flatnonzero(periods == period)
        local = np.full(len(ids), -1, dtype=np.int64)
        local[nodes] = np.arange(len(nodes))
        e_mask = (periods[src] == period)
        u, v = local[src[e_mask]], local[dst[e_mask]]
        n = len(nodes)
        if len(u):
            a = coo_matrix((np.ones(2*len(u), dtype=np.float64),
                (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()
            a.data[:] = 1.0
            a.eliminate_zeros()
        else:
            a = coo_matrix((n, n), dtype=np.float64).tocsr()
        degree = np.asarray(a.sum(axis=1)).ravel()
        invroot = np.divide(1.0, np.sqrt(degree), out=np.zeros(n), where=degree > 0)
        norm = diags(invroot) @ a @ diags(invroot)
        local_obs = obs_mask[nodes]
        h_local = x[nodes].copy()
        for _ in range(steps):
            h_local = np.asarray(norm @ h_local)
            h_local[local_obs] = x[nodes][local_obs]
        h[nodes] = h_local
        # Directed neighbor means use reconstructed values for every node.
        gsrc, gdst = local[src[e_mask]], local[dst[e_mask]]
        if len(gsrc):
            incoming_sum = coo_matrix((np.ones(len(gsrc)), (gdst, gsrc)), shape=(n,n)).tocsr() @ h_local
            outgoing_sum = coo_matrix((np.ones(len(gsrc)), (gsrc, gdst)), shape=(n,n)).tocsr() @ h_local
            di = np.bincount(gdst, minlength=n).astype(np.int64)
            do = np.bincount(gsrc, minlength=n).astype(np.int64)
            in_local = np.divide(incoming_sum, di[:, None], out=np.zeros_like(incoming_sum), where=di[:, None] > 0)
            out_local = np.divide(outgoing_sum, do[:, None], out=np.zeros_like(outgoing_sum), where=do[:, None] > 0)
            missing_neighbors = (~obs_mask[nodes]).astype(np.float64)
            incoming_missing = np.bincount(gdst, weights=missing_neighbors[gsrc], minlength=n)
            outgoing_missing = np.bincount(gsrc, weights=missing_neighbors[gdst], minlength=n)
            in_fraction = np.divide(incoming_missing, di, out=np.zeros(n), where=di > 0)
            out_fraction = np.divide(outgoing_missing, do, out=np.zeros(n), where=do > 0)
        else:
            di = do = np.zeros(n, dtype=np.int64)
            in_local = out_local = np.zeros((n, N_COMPONENTS), dtype=np.float64)
            in_fraction = out_fraction = np.zeros(n, dtype=np.float64)
        in_mean[nodes], out_mean[nodes] = in_local, out_local
        in_degree[nodes], out_degree[nodes] = di, do
        in_graph_only[nodes], out_graph_only[nodes] = in_fraction, out_fraction
    features = np.column_stack((in_mean, out_mean, in_graph_only, out_graph_only))
    if features.shape != (len(ids), MODEL_FEATURE_COUNT - BASE_COUNT) or not np.isfinite(features).all():
        raise ValueError("Directional propagated feature matrix is invalid")
    diag = {"graph_node_count": int(len(ids)), "attributed_node_count": int(obs_mask.sum()),
            "graph_only_node_count": int((~obs_mask).sum()), "component_count": int(len(np.unique(periods))),
            "propagation_steps": int(steps), "incoming_supported": int((in_degree > 0).sum()),
            "outgoing_supported": int((out_degree > 0).sum()),
            "incoming_zero_mean_rows": int((in_degree == 0).sum()),
            "outgoing_zero_mean_rows": int((out_degree == 0).sum()),
            "mean_graph_only_incoming_neighbor_fraction": float(in_graph_only.mean()),
            "mean_graph_only_outgoing_neighbor_fraction": float(out_graph_only.mean())}
    return h, features, diag


def fit_pca(raw_history: pd.DataFrame, raw_columns: list[str]) -> tuple[StandardScaler, PCA]:
    """Fit on all attributed history at/before the cutoff, independent of labels."""
    if len(raw_columns) != 155 or raw_history.empty:
        raise ValueError("Trial100 requires exactly 155 retained raw PCA attributes")
    values = raw_history[raw_columns].to_numpy(dtype=np.float64)
    if values.shape[1] != 155 or not np.isfinite(values).all():
        raise ValueError("PCA fit population contains missing/nonfinite raw values")
    scaler = StandardScaler().fit(values)
    pca = PCA(n_components=N_COMPONENTS, svd_solver="full", whiten=False).fit(scaler.transform(values))
    return scaler, pca


def fit_config() -> dict:
    base = learner_config()
    return {**base, "max_leaf_nodes": 15, "min_samples_leaf": 40,
            "learning_rate": .03, "max_iter": 600, "l2_regularization": 2.0,
            "max_features": .8, "unknown_status_mass": .40, "half_life": 20.0,
            "random_state": 42, "early_stopping": False,
            "input_contract": "177 cutoff-safe historical columns plus 32 directed means and 2 graph-only neighbor fractions = 211"}


PLAN = {
    "trial_number": 100,
    "variant": PREFIX,
    "date": "2026-10-07",
    "status": "prepared_source_and_plan_only_waiting_for_root_review_and_authorization",
    "hypothesis": "Clamped label-free diffusion can reconstruct compact attribute vectors for graph-only transactions; directional means over observed and reconstructed vectors may add relationship context that attributed-node-only filters omit.",
    "research_context": [
        {"url": "https://arxiv.org/abs/2111.12128", "finding": "Feature Propagation iteratively applies a normalized graph operator and restores known node features after each step.", "relevance": "This is a deterministic fixed-PCA HGB adaptation, not a reproduction of the paper's benchmark pipeline."},
        {"url": "https://blog.x.com/engineering/en_us/topics/insights/2022/graph-machine-learning-with-missing-node-features", "finding": "The authors describe zero initialization for unknown values and about 40 steps as sufficient in their experiments; they also identify homophily as a limitation.", "relevance": "Trial100 freezes 20 steps and measures no homophily claim; transfer to transaction fraud remains uncertain."}
    ],
    "public_context": {"reported_best_name": PUBLIC_NAME, "reported_best_public_roc_auc": PUBLIC_AUC,
                       "previous_name": PREVIOUS_PUBLIC_NAME, "previous_public_roc_auc": PREVIOUS_PUBLIC_AUC,
                       "private_score": None, "source": "user_reported_trial99_submission_result"},
    "novelty_scope": "Trials27 and29 computed directed PCA filters/residuals after excluding graph-only endpoints. Trial98 retained graph-only nodes but used a learned missing-token neural model rather than deterministic clamped feature reconstruction. This trial changes the representation and graph-only endpoint coverage; it does not claim a new propagation algorithm.",
    "input_contract": {"historical_candidates": 182, "cutoff_safe_base_columns": 172, "trial90_graph_columns": GRAPH5,
                       "full_base_count": BASE_COUNT, "retained_raw_pca_attributes": 155,
                       "pca_components": N_COMPONENTS},
    "preprocessing": {"scaler": "StandardScaler", "pca": "PCA(n_components=16, svd_solver='full', whiten=False)",
                      "fit_population": "Every attributed training row (known licit, known illicit, and unknown status) with time_step <= that fold's fit cutoff; labels are not read for scaler/PCA fitting.",
                      "cutoff": "The outer/confirmation origin for validation, 35 for the final model.",
                      "transform": "Transform attributed nodes in each historical or forecast period using only that cutoff's scaler/PCA."},
    "graph_contract": {"node_count": GRAPH_NODE_COUNT, "weak_components": WEAK_COMPONENT_COUNT,
                        "component_period": "Infer each graph-only node's period from attributed node times in its weak component; assert every component maps to exactly one period.",
                        "edges": "Treat unique directed edge rows as undirected for propagation; remove self loops; use binary symmetric adjacency, no self loops.",
                        "propagation_operator": "D^-1/2 A D^-1/2, symmetric normalized binary adjacency, no self loops.",
                        "initialization": "PCA vectors for observed attributed nodes; all-zero vectors for graph-only nodes.",
                        "update": "Exactly 20 iterations per period component: H <- A_norm H, then restore every observed PCA vector to its original value.",
                        "directional_means": "From the 20-step reconstructed vectors, compute source-to-target incoming and outgoing arithmetic neighbor means on original directed edges; include graph-only endpoints; return 16 zeros when a direction has no edge.",
                        "support": "For each direction, add the fraction of adjacent nodes that are graph-only; zero when the row has no edge in that direction.",
                        "partition": "Historical periods <= cutoff and each forecast period are processed as separate components. Only attributed nodes from that same period are clamped; no historical or other future period nodes enter forecast propagation."},
    "features": {"existing": "The same 172 cutoff-safe historical columns plus exact frozen Trial90 graph5 = 177.",
                 "added": "16 reconstructed-PCA incoming means + 16 reconstructed-PCA outgoing means + incoming/outgoing graph-only neighbor fractions.",
                 "support_features": "fractions of incoming/outgoing graph neighbors that are graph-only, zero when the direction has no edges.",
                 "total": MODEL_FEATURE_COUNT, "column_order": "177 existing columns in established order, then fp_in_00..15, fp_out_00..15, frac_graph_only_in, frac_graph_only_out."},
    "model": fit_config(),
    "target_and_weights": "Three statuses: 0 known licit, 1 known illicit, 2 unknown. Use half-life 20 and raw recency weights, scaled so unknown rows carry exactly 40% of total weight and known rows 60%; no mean-one weight normalization.",
    "score": "p(known illicit)/(p(known licit)+p(known illicit)); preserve all three class probabilities and denominator.",
    "public99_control": {"trial": TRIAL99, "submission_name": PUBLIC_NAME, "public_roc_auc_user_reported": PUBLIC_AUC,
                         "candidate": f"submissions/candidate_{TRIAL99}.csv", "control_variant": TRIAL99_VARIANT,
                         "validation_rebuild": "Rebuild Trial99 on exact keyed Trial90 forecasts plus all three saved Trial99 conditional-logit member forecasts; independently check against Trial99 saved outputs before computing new deltas.",
                         "blend": "sigmoid(0.75*logit(clip(PUBLIC90,1e-12,1-1e-12)) + 0.25*logit(clip(mean-seed-logit Trial99 ensemble,1e-12,1-1e-12)))"},
    "eligible_variants": list(ORIGINAL_VARIANTS),
    "original_origins": ORIGINS, "confirmation_origins": CONFIRMATION,
    "original_gate": "For each same variant: full14, second7, primary7, and rolling7 mean AUC deltas > 0 against reconstructed Trial99; worst individual slice delta >= -0.001.",
    "confirmation_gate": "Only for original-stage qualifiers; same variant full14 and second7 mean deltas > 0, first7 mean delta >= 0, worst individual slice delta >= -0.001.",
    "winner": "Strict same-variant intersection only; prefer the fixed 75/25 log-odds blend over standalone when both qualify.",
    "known_uncertainty": "Feature propagation assumes neighboring feature similarity. Fraud graphs may be heterophilic, and temporal validation windows are reused/overlapping and stop at period35 while test covers later periods.",
    "safety": {"label_used_for_feature_build": False, "test_labels_used": False, "external_data_used": False,
               "fit_before_root_authorization": False, "confirmation_before_original_pass": False,
               "final_fit_before_same_variant_two_stage_pass": False, "kaggle_upload": False},
    "source_sha256": None,
    "registry_preflight": {"unique_rows": 14460, "source": "experiment_log.csv at Trial100 preparation", "runner_may_mutate_register": False},
    "artifacts": {"source": f"src/{PREFIX}.py", "plan": f"reports/{PREFIX}_plan.json",
                  "candidate": f"submissions/candidate_{PREFIX}.csv", "models": f"models/{PREFIX}/",
                  "forecast_and_feature_diagnostics": f"reports/{PREFIX}_fit_diagnostics.json"},
}


def verify_frozen_inputs(plan: dict) -> None:
    if (plan.get("trial_number") != 100 or plan.get("variant") != PREFIX
            or plan.get("configuration") != fit_config()
            or plan.get("source_sha256") != sha(PROJECT_ROOT / "src" / f"{PREFIX}.py")):
        raise RuntimeError("Trial100 source, plan, or fixed configuration changed")
    for rel, digest in plan.get("input_sha256", {}).items():
        if sha(PROJECT_ROOT / rel) != digest:
            raise RuntimeError(f"Frozen Trial100 input changed: {rel}")
    auth = json.loads(AUTH_PATH.read_text())
    if (auth.get("authorized") is not True or auth.get("source_sha256") != plan["source_sha256"]
            or auth.get("plan_sha256") != sha(PLAN_PATH)
            or auth.get("registry_rows") != plan["registry_preflight"]["unique_rows"]):
        raise RuntimeError("Matching root authorization for the exact Trial100 source and plan is required")
    csv.field_size_limit(10_000_000)
    with (PROJECT_ROOT / "experiment_log.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ids = [r.get("experiment_id", "") for r in rows]
    if len(rows) != plan["registry_preflight"]["unique_rows"] or len(ids) != len(set(ids)):
        raise RuntimeError("Experiment register unique-ID preflight changed")
    by_id={r.get("experiment_id",""):r for r in rows}
    for experiment_id,expected_hash in plan.get("trial99_direct",{}).get("registered_rows",{}).items():
        row=by_id.get(experiment_id)
        actual=(hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":")).encode()).hexdigest()
                if row is not None else None)
        if actual!=expected_hash:
            raise RuntimeError(f"Trial99 exact registered row changed: {experiment_id}")


def assert_clean_outputs() -> None:
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        raise RuntimeError("Refusing to overwrite existing Trial100 model files")
    if any((REPORT_DIR / f"{PREFIX}_{name}").exists() for name in (
            "original_predictions.pkl", "original_forecast_probabilities.pkl", "original_results.csv",
            "original_summary.csv", "confirmation_predictions.pkl", "confirmation_forecast_probabilities.pkl",
            "confirmation_results.csv", "confirmation_summary.csv", "predictions.pkl", "results.csv",
            "summary.csv", "fit_diagnostics.json", "metadata.json", "final_components.pkl")):
        raise RuntimeError("Refusing to overwrite existing Trial100 reports")
    if (PROJECT_ROOT / "submissions" / f"candidate_{PREFIX}.csv").exists():
        raise RuntimeError("Refusing to overwrite an existing Trial100 candidate")
    if (REPORT_DIR / f"{PREFIX}_final_forecast_probabilities.pkl").exists() or (REPORT_DIR / f"{PREFIX}_final_graph_features.pkl").exists():
        raise RuntimeError("Refusing to overwrite existing Trial100 final forecast/features")


def rebuild_public99_control(members: pd.DataFrame, public90: pd.DataFrame,
                             saved: pd.DataFrame, origins: dict[int, int]) -> pd.DataFrame:
    member_columns = KEY + [f"seed_{seed}_conditional_score" for seed in (7, 2026, 31415)]
    if (not set(member_columns).issubset(members.columns) or members.duplicated(KEY).any()
            or not set(KEY + [TRIAL99_VARIANT]).issubset(public90.columns) or public90.duplicated(KEY).any()
            or set(members.origin.unique()) != set(origins) or set(public90.origin.unique()) != set(origins)):
        raise ValueError("Trial99 member/Trial90 forecast schema or keys changed")
    member_scores=[]
    for seed in (7, 2026, 31415):
        p=members[f"seed_{seed}_conditional_score"].to_numpy(dtype=np.float64)
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError(f"Invalid Trial99 seed {seed} forecast")
        member_scores.append(p)
    ensemble=expit(np.mean(logit(np.clip(np.vstack(member_scores),EPS,1-EPS)),axis=0))
    keyed=members[KEY].copy();keyed[TRIAL99_ENSEMBLE]=ensemble
    keyed=keyed.loc[keyed.label.notna()].copy()
    out=public90[KEY+[TRIAL99_VARIANT]].merge(keyed,on=KEY,how="inner",validate="one_to_one",sort=False)
    if len(out)!=len(public90) or not np.array_equal(out[KEY].to_numpy(),public90[KEY].to_numpy()):
        raise ValueError("Trial90 and Trial99 member forecast populations differ")
    p90=out[TRIAL99_VARIANT].to_numpy(dtype=np.float64)
    ensemble_known=out[TRIAL99_ENSEMBLE].to_numpy(dtype=np.float64)
    out["public99_control"]=expit(.75*logit(np.clip(p90,EPS,1-EPS))+.25*logit(np.clip(ensemble_known,EPS,1-EPS)))
    check=out[KEY+['public99_control']].merge(saved[KEY+[TRIAL99_VARIANT]],on=KEY,how="inner",validate="one_to_one",sort=False)
    if len(check)!=len(saved) or not np.array_equal(check[KEY].to_numpy(),saved[KEY].to_numpy()) or not np.allclose(check.public99_control,saved[TRIAL99_VARIANT],rtol=0,atol=1e-12):
        raise ValueError("Independent Trial99 member/control reconstruction failed")
    return out[KEY+['public99_control']]


def load_public99_control(stage: str, origins: dict[int, int]) -> pd.DataFrame:
    return rebuild_public99_control(pd.read_pickle(TRIAL99_STAGE_PATHS[stage]),
        pd.read_pickle(TRIAL90_STAGE_PATHS[stage]),
        pd.read_pickle(REPORT_DIR/f"{TRIAL99}_{stage}_predictions.pkl"), origins)


def attach_graph5(train: pd.DataFrame, test: pd.DataFrame, graph5: pd.DataFrame):
    if (graph5.columns.tolist()!=['txId',*GRAPH5] or len(graph5)!=GRAPH_NODE_COUNT
            or graph5.txId.duplicated().any() or not np.array_equal(graph5.txId.to_numpy(),np.sort(graph5.txId.to_numpy()))
            or graph5[GRAPH5].isna().any().any() or not np.isfinite(graph5[GRAPH5].to_numpy(dtype=float)).all()):
        raise ValueError("Frozen Trial90 graph5 cache changed")
    out=[]
    for frame in (train,test):
        ids=frame.txId.to_numpy();joined=frame.merge(graph5,on='txId',how='left',validate='one_to_one',sort=False)
        if not np.array_equal(joined.txId.to_numpy(),ids) or joined[GRAPH5].isna().any().any():
            raise ValueError("Graph5 join changed feature row population/order")
        out.append(joined)
    return out


def build_stage_features(train,test,graph_ids,node_period,edges,full_columns,origin,stop):
    columns=historical_columns(train,full_columns,origin)
    raw_columns=[c for c in columns if c.startswith('feat_')]
    if len(columns)!=BASE_COUNT or columns[-5:]!=GRAPH5 or len(raw_columns)!=155:
        raise ValueError("Trial100 cutoff-safe 177 columns or 155 raw PCA inputs changed")
    history=train.loc[train.time_step.le(origin)]
    scaler,pca=fit_pca(history,raw_columns)
    ids=np.asarray(graph_ids,dtype=np.int64);node_period=np.asarray(node_period,dtype=np.int16)
    if ids.shape!=node_period.shape:raise ValueError("Graph period map is misaligned")
    derived=[];graph_derived=[];diagnostics=[]
    # Each period is processed separately. This keeps each future graph separate
    # from both historical nodes and other forecast periods.
    for period in range(1,stop+1):
        nodes=ids[node_period==period]
        if not len(nodes):continue
        if period<=origin:
            attributed=train.loc[train.time_step.eq(period)]
        else:
            attributed=pd.concat([train.loc[train.time_step.eq(period)],test.loc[test.time_step.eq(period)]],ignore_index=True)
        if attributed.empty or attributed.txId.duplicated().any() or not set(attributed.txId).issubset(set(nodes)):
            raise ValueError(f"Invalid same-period attributed-node set at period {period}")
        z=pca.transform(scaler.transform(attributed[raw_columns].to_numpy(dtype=np.float64)))
        period_edges=edges.loc[edges.txId1.isin(nodes)&edges.txId2.isin(nodes),['txId1','txId2']]
        h,extra,diag=propagated_and_directional_features(nodes,np.full(len(nodes),period,dtype=np.int16),period_edges,
            attributed.txId.to_numpy(dtype=np.int64),z,steps=PROPAGATION_STEPS)
        names=[f"fp_in_{i:02d}" for i in range(16)]+[f"fp_out_{i:02d}" for i in range(16)]+['frac_graph_only_in','frac_graph_only_out']
        part=pd.DataFrame(extra,columns=names);part.insert(0,'txId',nodes);part['time_step']=period
        graph_derived.append(part)
        derived.append(attributed[['txId','time_step']].merge(part,on=['txId','time_step'],how='left',validate='one_to_one',sort=False))
        diagnostics.append({'period':int(period),**diag,'observed_attributed_count':int(len(attributed))})
    extra=pd.concat(derived,ignore_index=True);graph_derived=pd.concat(graph_derived,ignore_index=True)
    names=[f"fp_in_{i:02d}" for i in range(16)]+[f"fp_out_{i:02d}" for i in range(16)]+['frac_graph_only_in','frac_graph_only_out']
    train_extra=extra.loc[extra.txId.isin(train.txId)].set_index('txId')[names]
    test_extra=extra.loc[extra.txId.isin(test.txId)].set_index('txId')[names]
    train_out=train.join(train_extra,on='txId',validate='one_to_one')
    test_out=test.join(test_extra,on='txId',validate='one_to_one')
    model_columns=columns+names
    fit_rows=train_out.loc[train_out.time_step.le(stop),model_columns]
    test_rows=test_out.loc[test_out.time_step.le(stop),model_columns]
    if (len(model_columns)!=MODEL_FEATURE_COUNT or fit_rows.isna().any().any() or test_rows.isna().any().any()
            or not np.isfinite(fit_rows.to_numpy(dtype=float)).all() or not np.isfinite(test_rows.to_numpy(dtype=float)).all()):
        raise ValueError("Trial100 211-column transformed feature matrix has missing/nonfinite values")
    return train_out,test_out,model_columns,scaler,pca,graph_derived,diagnostics


def fit_one(train,test,graph_ids,node_period,edges,full_columns,origin,stop,stage,control):
    tr,te,columns,scaler,pca,gfeatures,gdiag=build_stage_features(train,test,graph_ids,node_period,edges,full_columns,origin,stop)
    history=tr.loc[tr.time_step.le(origin)];future=tr.loc[tr.time_step.between(origin+1,stop)]
    status=label_status(history.label);weights,factor=training_weights_40(history.time_step,status,origin)
    if not np.array_equal(np.unique(status),np.array([0,1,2])) or not np.isclose(weights[status==2].sum()/weights.sum(),.40,rtol=0,atol=1e-12):
        raise ValueError("Trial100 status or class-mass weighting changed")
    cfg={k:v for k,v in fit_config().items() if k not in ('half_life','unknown_status_mass','input_contract','random_state','early_stopping')}
    fitted=historical_model(cfg).set_params(random_state=42)
    fitted.fit(history[columns],status,sample_weight=weights)
    if fitted.n_iter_!=600 or not np.array_equal(fitted.classes_,[0,1,2]) or len(fitted.feature_names_in_)!=MODEL_FEATURE_COUNT:
        raise ValueError("Trial100 HGB model contract changed")
    proba=fitted.predict_proba(future[columns]);score,denom=conditional_score(proba,fitted.classes_)
    forecast=future[['txId','time_step','label']].copy()
    forecast['p_known_licit']=proba[:,0];forecast['p_known_illicit']=proba[:,1];forecast['p_unknown']=proba[:,2]
    forecast['p_known_denominator']=denom;forecast['feature_propagation_hgb']=score
    forecast.insert(0,'origin',origin)
    known=forecast.loc[forecast.label.notna()].copy();ctl=control.loc[control.origin.eq(origin),KEY+['public99_control']]
    joined=known.merge(ctl,on=KEY,how='inner',validate='one_to_one',sort=False)
    if len(joined)!=len(known) or not np.array_equal(joined.txId.to_numpy(),known.txId.to_numpy()):raise ValueError("Trial99 keyed control differs")
    known['public99_control']=joined.public99_control.to_numpy()
    known['blend_75_25_logodds']=expit(.75*logit(np.clip(known.public99_control,EPS,1-EPS))+.25*logit(np.clip(known.feature_propagation_hgb,EPS,1-EPS)))
    MODEL_DIR.mkdir(parents=True,exist_ok=True);stem=f'{stage}_{origin}_{stop}'
    prep_path=MODEL_DIR/f'{stem}_scaler_pca.joblib'
    joblib.dump({'scaler':scaler,'pca':pca,'raw_columns':[c for c in columns if c.startswith('feat_')],
                 'model_columns':columns,'origin':origin,'forecast_end':stop,'plan_sha256':sha(PLAN_PATH)},prep_path)
    model_path=MODEL_DIR/f'{stem}_model.joblib'
    joblib.dump({'model':fitted,'columns':columns,'configuration':fit_config(),'stage':stage,
                 'origin':origin,'forecast_end':stop,'plan_sha256':sha(PLAN_PATH)},model_path)
    forecast_path=REPORT_DIR/f'{PREFIX}_{stem}_forecast_probabilities.pkl';forecast.to_pickle(forecast_path)
    feature_path=REPORT_DIR/f'{PREFIX}_{stem}_graph_features.pkl';gfeatures.to_pickle(feature_path)
    replay=joblib.load(model_path)['model'].predict_proba(future[columns])
    if not np.allclose(replay,proba,rtol=0,atol=1e-12):raise ValueError('Saved Trial100 HGB replay mismatch')
    diag={'stage':stage,'origin':int(origin),'forecast_end':int(stop),'history_rows':len(history),'forecast_rows':len(future),
          'known_forecast_rows':len(known),'feature_count':len(columns),'raw_pca_attributes':155,'pca_components':16,
          'pca_solver':'full','propagation_steps':20,'iterations':int(fitted.n_iter_),
          'class_order':fitted.classes_.tolist(),'unknown_weight_share':float(weights[status==2].sum()/weights.sum()),
          'unknown_weight_factor':float(factor),'zero_known_denominator_rows':int((denom==0).sum()),
          'feature_diagnostics':gdiag,'preprocess_file':str(prep_path.relative_to(PROJECT_ROOT)),'preprocess_sha256':sha(prep_path),
          'model_file':str(model_path.relative_to(PROJECT_ROOT)),'model_sha256':sha(model_path),
          'forecast_file':str(forecast_path.relative_to(PROJECT_ROOT)),'forecast_sha256':sha(forecast_path),
          'graph_features_file':str(feature_path.relative_to(PROJECT_ROOT)),'graph_features_sha256':sha(feature_path)}
    return known,forecast,diag


def run_stage(train,test,graph_ids,node_period,edges,full_columns,origins,stage,control):
    known=[];forecasts=[];diagnostics=[]
    for origin,stop in origins.items():
        p,f,d=fit_one(train,test,graph_ids,node_period,edges,full_columns,origin,stop,stage,control)
        known.append(p);forecasts.append(f);diagnostics.append(d)
        print(f'Completed Trial100 {stage} {origin}->{stop}',flush=True)
    return pd.concat(known,ignore_index=True),pd.concat(forecasts,ignore_index=True),diagnostics


def evaluate(predictions: pd.DataFrame,stage: str) -> pd.DataFrame:
    rows=[]
    for origin,frame in predictions.groupby('origin',sort=False):
        slices={'first7':frame.loc[frame.time_step.le(origin+7)]}
        if int(frame.time_step.max())==int(origin+14):slices.update(full14=frame,second7=frame.loc[frame.time_step.gt(origin+7)])
        for name,part in slices.items():
            for variant in ('public99_control',*ORIGINAL_VARIANTS):
                rows.append({'stage':stage,'origin':int(origin),'slice':name,'variant':variant,'n':len(part),
                    'positives':int(part.label.sum()),'auc':float(roc_auc_score(part.label,part[variant])),
                    'average_precision':float(average_precision_score(part.label,part[variant]))})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame,stage: str) -> pd.DataFrame:
    pivot=results.pivot(index=['origin','slice'],columns='variant',values='auc')
    if stage=='original':
        groups={'full14':[(14,'full14'),(21,'full14')],'second7':[(14,'second7'),(21,'second7')],
                'primary7':[(o,'first7') for o in (14,21,28)],'rolling7':[(o,'first7') for o in (17,24)]}
    else:
        groups={'full14':[(o,'full14') for o in CONFIRMATION],
                'second7':[(o,'second7') for o in CONFIRMATION],'first7':[(o,'first7') for o in CONFIRMATION]}
    rows=[]
    for variant in ORIGINAL_VARIANTS:
        row={'stage':stage,'variant':variant}
        for name,keys in groups.items():
            delta=pivot.loc[keys,variant]-pivot.loc[keys,'public99_control']
            row[f'{name}_mean_auc']=float(pivot.loc[keys,variant].mean());row[f'{name}_mean_delta']=float(delta.mean())
        row['worst_slice_delta']=float((pivot[variant]-pivot.public99_control).min())
        gate=(all(row[f'{name}_mean_delta']>0 for name in groups) if stage=='original' else
              row['full14_mean_delta']>0 and row['second7_mean_delta']>0 and row['first7_mean_delta']>=0)
        row['qualified']=bool(gate and row['worst_slice_delta']>=-.001);rows.append(row)
    return pd.DataFrame(rows)


def create_candidate(train,test,graph_ids,node_period,edges,full_columns,sample,winner,plan_sha):
    control=pd.read_csv(TRIAL99_CANDIDATE)
    expected=sample['index'].to_numpy()
    test_ids=test['index'].to_numpy() if 'index' in test.columns else test.txId.to_numpy()
    if len(expected)!=15329 or not np.array_equal(test_ids,expected) or not np.array_equal(control['index'].to_numpy(),expected):
        raise ValueError("Official sample/test order or Trial99 candidate changed")
    tr,te,columns,scaler,pca,gfeatures,gdiag=build_stage_features(train,test,graph_ids,node_period,edges,full_columns,35,49)
    history=tr.loc[tr.time_step.le(35)];status=label_status(history.label);weights,factor=training_weights_40(history.time_step,status,35)
    cfg={k:v for k,v in fit_config().items() if k not in ('half_life','unknown_status_mass','input_contract','random_state','early_stopping')}
    fitted=historical_model(cfg).set_params(random_state=42);fitted.fit(history[columns],status,sample_weight=weights)
    if fitted.n_iter_!=600 or not np.array_equal(fitted.classes_,[0,1,2]):raise ValueError("Final Trial100 HGB configuration changed")
    proba=fitted.predict_proba(te[columns]);score,denom=conditional_score(proba,fitted.classes_)
    blend=expit(.75*logit(np.clip(control.target.to_numpy(float),EPS,1-EPS))+.25*logit(np.clip(score,EPS,1-EPS)))
    selected=blend if winner=='blend_75_25_logodds' else score
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    model_path=MODEL_DIR/'final_through35_model.joblib';prep_path=MODEL_DIR/'final_through35_scaler_pca.joblib'
    joblib.dump({'model':fitted,'columns':columns,'configuration':fit_config(),'plan_sha256':plan_sha},model_path)
    joblib.dump({'scaler':scaler,'pca':pca,'raw_columns':[c for c in columns if c.startswith('feat_')],
                 'model_columns':columns,'cutoff':35,'plan_sha256':plan_sha},prep_path)
    components=REPORT_DIR/f'{PREFIX}_final_components.pkl'
    pd.DataFrame({'index':expected,'public99_control':control.target.to_numpy(float),'feature_propagation_hgb':score,
        'p_known_licit':proba[:,0],'p_known_illicit':proba[:,1],'p_unknown':proba[:,2],
        'p_known_denominator':denom,'blend_75_25_logodds':blend}).to_pickle(components)
    feature_path=REPORT_DIR/f'{PREFIX}_final_graph_features.pkl';gfeatures.to_pickle(feature_path)
    forecast_path=REPORT_DIR/f'{PREFIX}_final_forecast_probabilities.pkl'
    pd.DataFrame({'index':expected,'txId':te.txId.to_numpy(),'time_step':te.time_step.to_numpy(),
        'p_known_licit':proba[:,0],'p_known_illicit':proba[:,1],'p_unknown':proba[:,2],
        'p_known_denominator':denom,'feature_propagation_hgb':score,'blend_75_25_logodds':blend}).to_pickle(forecast_path)
    candidate=PROJECT_ROOT/'submissions'/f'candidate_{PREFIX}.csv'
    pd.DataFrame({'index':expected,'target':selected}).to_csv(candidate,index=False)
    read=pd.read_csv(candidate)
    if len(read)!=15329 or not np.array_equal(read['index'].to_numpy(),expected) or not np.isfinite(read.target).all() or ((read.target<0)|(read.target>1)).any():
        raise ValueError("Trial100 candidate readback failed")
    return {'candidate_created':True,'final_model_fit_performed':True,'winner':winner,
        'candidate_file':str(candidate.relative_to(PROJECT_ROOT)),'candidate_sha256':sha(candidate),
        'final_model_file':str(model_path.relative_to(PROJECT_ROOT)),'final_model_sha256':sha(model_path),
        'final_preprocess_file':str(prep_path.relative_to(PROJECT_ROOT)),'final_preprocess_sha256':sha(prep_path),
        'final_components_file':str(components.relative_to(PROJECT_ROOT)),'final_components_sha256':sha(components),
        'final_graph_features_file':str(feature_path.relative_to(PROJECT_ROOT)),'final_graph_features_sha256':sha(feature_path),
        'final_forecast_file':str(forecast_path.relative_to(PROJECT_ROOT)),'final_forecast_sha256':sha(forecast_path),
        'final_unknown_weight_share':float(weights[status==2].sum()/weights.sum()),
        'final_unknown_weight_factor':float(factor),'final_feature_diagnostics':gdiag}


def verify_pure_toys() -> None:
    ids=np.array([1,2,3,10],dtype=np.int64)
    edges=pd.DataFrame({'txId1':[1,2],'txId2':[2,3]})
    observed=np.array([1,3],dtype=np.int64)
    z=np.zeros((2,16));z[0,0]=1.0
    h,features,_=propagated_and_directional_features(ids,np.full(4,2,dtype=np.int16),edges,observed,z)
    if not np.isclose(h[1,0],1/np.sqrt(2),rtol=0,atol=1e-12):
        raise AssertionError("Clamped propagation chain reconstruction changed")
    if not np.allclose(features[[0,2,3],-2:],[[0,1],[1,0],[0,0]],rtol=0,atol=0):
        raise AssertionError("Graph-only directional neighbor fractions changed")
    members=pd.DataFrame({'origin':[1,1,1],'txId':[102,101,103],'time_step':[3,2,4],
        'label':[1,0,np.nan],'seed_7_conditional_score':[.7,.2,.4],
        'seed_2026_conditional_score':[.65,.25,.45],'seed_31415_conditional_score':[.75,.15,.35]})
    public90=pd.DataFrame({'origin':[1,1],'txId':[101,102],'time_step':[2,3],'label':[0,1],
        TRIAL99_VARIANT:[.3,.8]})
    seed_stack=np.array([[.2,.7],[.25,.65],[.15,.75]])
    ensemble=expit(logit(seed_stack).mean(axis=0))
    blend=expit(.75*logit(np.array([.3,.8]))+.25*logit(ensemble))
    saved=public90[KEY].copy();saved[TRIAL99_VARIANT]=blend
    rebuilt=rebuild_public99_control(members,public90,saved,{1:3})
    if len(rebuilt)!=2 or not np.allclose(rebuilt.public99_control.to_numpy(),blend,rtol=0,atol=1e-15):
        raise AssertionError("Known-row Trial99 keyed control reconstruction changed")


def main() -> None:
    plan=json.loads(PLAN_PATH.read_text())
    verify_frozen_inputs(plan)
    assert_clean_outputs()
    # First competition-data load occurs only after every pin and root approval.
    train,test,sample,groups=load_features()
    _,_,edges,_=load_competition_data()
    graph5=pd.read_pickle(TRIAL90_GRAPH)
    train,test=attach_graph5(train,test,graph5)
    full_columns=list(groups['regularized_two_hop'])+GRAPH5
    if len(full_columns)!=187:raise ValueError("Trial100 requires 182 candidates plus Trial90 graph5")
    graph_ids=graph5.txId.to_numpy(dtype=np.int64)
    attrs=pd.concat([train[['txId','time_step']],test[['txId','time_step']]],ignore_index=True)
    _,node_period,component_meta=weak_component_period_map(graph_ids,edges,attrs,expected_components=WEAK_COMPONENT_COUNT)
    if len(graph_ids)!=GRAPH_NODE_COUNT or component_meta['weak_component_count']!=WEAK_COMPONENT_COUNT:
        raise ValueError("Trial100 frozen graph universe/component map changed")
    metadata={'trial_number':100,'variant':PREFIX,'source_sha256':sha(PROJECT_ROOT/'src'/f'{PREFIX}.py'),
        'plan_sha256':sha(PLAN_PATH),'original_gate_passed':False,'confirmation_run':False,
        'confirmation_gate_passed':False,'same_variant_passed_both_stages':False,
        'two_stage_qualified_variants':[],'winner':None,'candidate_created':False,
        'final_model_fit_performed':False,'kaggle_upload':False,'weak_component_period_map':component_meta}
    predictions=[];results=[];summaries=[];diagnostics=[]
    control=load_public99_control('original',ORIGINS)
    original,forecast,diag=run_stage(train,test,graph_ids,node_period,edges,full_columns,ORIGINS,'original',control)
    original.to_pickle(REPORT_DIR/f'{PREFIX}_original_predictions.pkl')
    forecast.to_pickle(REPORT_DIR/f'{PREFIX}_original_forecast_probabilities.pkl')
    ro=evaluate(original,'original');so=summarize(ro,'original')
    ro.to_csv(REPORT_DIR/f'{PREFIX}_original_results.csv',index=False);so.to_csv(REPORT_DIR/f'{PREFIX}_original_summary.csv',index=False)
    metadata['original_gate_passed']=bool(so.qualified.any())
    predictions.append(original);results.append(ro);summaries.append(so);diagnostics.extend(diag)
    if metadata['original_gate_passed']:
        control_c=load_public99_control('confirmation',CONFIRMATION)
        confirm,forecast_c,diag_c=run_stage(train,test,graph_ids,node_period,edges,full_columns,CONFIRMATION,'confirmation',control_c)
        confirm.to_pickle(REPORT_DIR/f'{PREFIX}_confirmation_predictions.pkl')
        forecast_c.to_pickle(REPORT_DIR/f'{PREFIX}_confirmation_forecast_probabilities.pkl')
        rc=evaluate(confirm,'confirmation');sc=summarize(rc,'confirmation')
        rc.to_csv(REPORT_DIR/f'{PREFIX}_confirmation_results.csv',index=False);sc.to_csv(REPORT_DIR/f'{PREFIX}_confirmation_summary.csv',index=False)
        winner,intersection=select_winner(so,sc)
        metadata.update(confirmation_run=True,confirmation_gate_passed=bool(sc.qualified.any()),
            same_variant_passed_both_stages=winner is not None,two_stage_qualified_variants=intersection,winner=winner)
        predictions.append(confirm);results.append(rc);summaries.append(sc);diagnostics.extend(diag_c)
        if winner:metadata.update(create_candidate(train,test,graph_ids,node_period,edges,full_columns,sample,winner,sha(PLAN_PATH)))
    pd.concat(predictions,ignore_index=True).to_pickle(REPORT_DIR/f'{PREFIX}_predictions.pkl')
    pd.concat(results,ignore_index=True).to_csv(REPORT_DIR/f'{PREFIX}_results.csv',index=False)
    pd.concat(summaries,ignore_index=True).to_csv(REPORT_DIR/f'{PREFIX}_summary.csv',index=False)
    metadata['fit_diagnostics']=diagnostics
    (REPORT_DIR/f'{PREFIX}_fit_diagnostics.json').write_text(json.dumps(diagnostics,indent=2,sort_keys=True)+'\n')
    (REPORT_DIR/f'{PREFIX}_metadata.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')


if __name__ == "__main__":
    if len(sys.argv)==2 and sys.argv[1]=="--fit-free-checks":
        verify_pure_toys()
        print("Trial100 pure synthetic checks passed")
    elif len(sys.argv)==1:
        main()
    else:
        raise SystemExit("Usage: source.py [--fit-free-checks]")
