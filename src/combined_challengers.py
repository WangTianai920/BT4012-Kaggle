"""Controlled follow-up experiments on existing temporal fraud-detection candidates.

Run from the project root with ``.venv/bin/python -m src.combined_challengers``.
No Kaggle upload is performed. Existing candidate files are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.competition import PROJECT_ROOT, RANDOM_SEED, REPORT_DIR, load_competition_data
from src.graph_experiments import FOLDS, make_feature_sets, merge_graph_features
from src.graph_features import load_or_build_graph_features
from src.graph_structure_features import load_or_build_graph_structure_features
from src.model_comparison import MODEL_CONFIGURATIONS
from src.time_relative_features import merge_time_relative_ranks


SUBMISSION_DIR = PROJECT_ROOT / "submissions"
VARIANTS = {
    "recency_control": "submission_02_recency_weighted_hgb.csv",
    "rank_40": "submission_03_time_relative_hgb.csv",
    "two_hop": "submission_04_new_graph_structure_hgb.csv",
    "combined_rank40_two_hop": "submission_05_combined_rank_graph_hgb.csv",
    "regularized_two_hop": "submission_06_regularized_graph_hgb.csv",
    "blend_half_rank_half_graph": "submission_07_rank_graph_blend.csv",
}
HALF_LIFE = 20.0


def _model(regularized: bool = False) -> HistGradientBoostingClassifier:
    configuration = MODEL_CONFIGURATIONS["Histogram gradient boosting"].copy()
    if regularized:
        configuration.update(max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=2.0)
    return HistGradientBoostingClassifier(**configuration, random_state=RANDOM_SEED)


def _weights(steps: pd.Series, reference_time: int) -> np.ndarray:
    return np.power(0.5, (reference_time - steps.to_numpy(dtype=float)) / HALF_LIFE)


def _read_rank_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    ranking = pd.read_csv(REPORT_DIR / "time_relative_early_drift_ranking.csv")
    ranked = ranking["feature"].tolist()
    if len(ranked) != 165 or len(set(ranked)) != 165 or not ranking["early_window_only"].all():
        raise ValueError("Early-only drift ranking is missing or invalid")
    rank_features = [f"time_rank_{name}" for name in ranked[:40]]
    derived = pd.read_pickle(PROJECT_ROOT / "data" / "derived_time_relative_ranks.pkl")
    train_enriched, test_enriched = merge_time_relative_ranks(train, test, derived)
    return train_enriched, test_enriched, rank_features


def load_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    train, test, edges, sample = load_competition_data()
    if not sample["index"].equals(test["index"]):
        raise ValueError("Sample submission index does not match test order")
    original, original_groups, _ = load_or_build_graph_features(train, test, edges)
    graph2, graph2_groups, _ = load_or_build_graph_structure_features(train, test, edges)
    train_graph, test_graph = merge_graph_features(train, test, original)
    train_graph, test_graph = merge_graph_features(train_graph, test_graph, graph2)
    train_all, test_all, rank_features = _read_rank_features(train_graph, test_graph)
    base = make_feature_sets(train, original_groups)["tabular_plus_structural"]
    two_hop = graph2_groups["two_hop_neighborhood"]
    groups = {
        "recency_control": base,
        "rank_40": base + rank_features,
        "two_hop": base + two_hop,
        "combined_rank40_two_hop": base + rank_features + two_hop,
        "regularized_two_hop": base + two_hop,
    }
    if [len(groups[name]) for name in ["recency_control", "rank_40", "two_hop", "combined_rank40_two_hop"]] != [176, 216, 182, 222]:
        raise ValueError("Feature counts differ from the established controls")
    for name, columns in groups.items():
        if len(columns) != len(set(columns)) or train_all[columns].isna().any().any() or test_all[columns].isna().any().any():
            raise ValueError(f"Invalid feature set: {name}")
    return train_all, test_all, sample, groups


def _check_control(results: pd.DataFrame) -> None:
    sources = {
        "recency_control": ("recency_weighting_results.csv", "half_life_20"),
        "rank_40": ("time_relative_results.csv", "recency_hgb_top40_time_ranks"),
        "two_hop": ("new_graph_structure_results.csv", "recency_hgb_two_hop_only"),
    }
    for name, (filename, old_name) in sources.items():
        old = pd.read_csv(REPORT_DIR / filename).query("variant == @old_name").set_index("fold")["roc_auc"]
        new = results.query("variant == @name").set_index("fold")["roc_auc"]
        if not old.index.equals(new.index) or not np.allclose(old.to_numpy(), new.to_numpy(), rtol=0, atol=1e-9):
            raise ValueError(f"The {name} control did not reproduce the earlier fold scores")


def evaluate(train: pd.DataFrame, groups: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labelled = train.loc[train["label"].notna()].copy()
    rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    for fold in FOLDS:
        development = labelled.loc[labelled["time_step"].between(fold["train_start"], fold["train_end"])]
        validation = labelled.loc[labelled["time_step"].between(fold["valid_start"], fold["valid_end"])]
        weights = _weights(development["time_step"], fold["train_end"])
        truth = validation["label"].astype(int).to_numpy()
        predictions: dict[str, np.ndarray] = {}
        for name, columns in groups.items():
            started = time.perf_counter()
            model = _model(regularized=name == "regularized_two_hop")
            model.fit(development[columns], development["label"].astype(int), sample_weight=weights)
            probability = model.predict_proba(validation[columns])[:, 1]
            predictions[name] = probability
            rows.append({**fold, "variant": name, "feature_count": len(columns),
                         "roc_auc": roc_auc_score(truth, probability),
                         "average_precision": average_precision_score(truth, probability),
                         "runtime_seconds": time.perf_counter() - started})
        blend = 0.5 * predictions["rank_40"] + 0.5 * predictions["two_hop"]
        predictions["blend_half_rank_half_graph"] = blend
        rows.append({**fold, "variant": "blend_half_rank_half_graph", "feature_count": 2,
                     "roc_auc": roc_auc_score(truth, blend),
                     "average_precision": average_precision_score(truth, blend),
                     "runtime_seconds": 0.0})
        prediction_rows.append(pd.DataFrame({"fold": fold["fold"], "txId": validation["txId"].to_numpy(),
                                             "time_step": validation["time_step"].to_numpy(),
                                             "label": truth, **predictions}))
    results = pd.DataFrame(rows)
    _check_control(results)
    summary = results.groupby("variant", as_index=False).agg(
        mean_roc_auc=("roc_auc", "mean"), weakest_roc_auc=("roc_auc", "min"),
        mean_average_precision=("average_precision", "mean"),
        feature_count=("feature_count", "first"))
    latest = results.query("fold == 'fold_2'").set_index("variant")["roc_auc"]
    summary["latest_roc_auc"] = summary["variant"].map(latest)
    summary = summary.sort_values(["mean_roc_auc", "weakest_roc_auc"], ascending=False).reset_index(drop=True)
    return results, summary, pd.concat(prediction_rows, ignore_index=True)


def _write_submission(name: str, probability: np.ndarray, sample: pd.DataFrame) -> dict:
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError(f"Invalid probabilities for {name}")
    submission = sample.copy()
    if submission.columns.tolist() != ["index", "target"]:
        raise ValueError("Unexpected sample submission columns")
    submission["target"] = probability
    path = SUBMISSION_DIR / VARIANTS[name]
    if path.exists():
        existing = pd.read_csv(path)
        if (existing.columns.tolist() != submission.columns.tolist()
            or not existing["index"].equals(submission["index"])
            or not np.allclose(existing["target"], probability, rtol=0, atol=1e-15)):
            raise FileExistsError(f"Existing candidate differs; refusing overwrite: {path}")
    else:
        submission.to_csv(path, index=False)
    check = {"variant": name, "file": path.name, "rows": len(submission),
             "columns_match_sample": submission.columns.tolist() == sample.columns.tolist(),
             "order_matches_sample": submission["index"].equals(sample["index"]),
             "all_finite": bool(np.isfinite(submission["target"]).all()),
             "in_unit_interval": bool(submission["target"].between(0, 1).all()),
             "unique_probabilities": int(submission["target"].nunique()),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    check["ready"] = (check["rows"] == len(sample) and check["columns_match_sample"]
                      and check["order_matches_sample"] and check["all_finite"]
                      and check["in_unit_interval"] and check["unique_probabilities"] > 1)
    if not check["ready"]:
        raise ValueError(f"Candidate failed independent validation: {name}")
    return check


def fit_candidates(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame,
                   groups: dict[str, list[str]]) -> pd.DataFrame:
    labelled = train.loc[train["label"].notna()]
    weights = _weights(labelled["time_step"], 35)
    outputs: dict[str, np.ndarray] = {}
    for name in ["rank_40", "two_hop", "combined_rank40_two_hop", "regularized_two_hop"]:
        model = _model(regularized=name == "regularized_two_hop")
        model.fit(labelled[groups[name]], labelled["label"].astype(int), sample_weight=weights)
        outputs[name] = model.predict_proba(test[groups[name]])[:, 1]
    outputs["blend_half_rank_half_graph"] = 0.5 * outputs["rank_40"] + 0.5 * outputs["two_hop"]
    checks = [_write_submission(name, outputs[name], sample)
              for name in ["combined_rank40_two_hop", "regularized_two_hop", "blend_half_rank_half_graph"]]
    return pd.DataFrame(checks)


def main() -> None:
    train, test, sample, groups = load_features()
    results, summary, predictions = evaluate(train, groups)
    checks = fit_candidates(train, test, sample, groups)
    REPORT_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORT_DIR / "combined_challenger_results.csv", index=False)
    summary.to_csv(REPORT_DIR / "combined_challenger_summary.csv", index=False)
    predictions.to_csv(REPORT_DIR / "combined_challenger_oof_predictions.csv", index=False)
    checks.to_csv(REPORT_DIR / "combined_challenger_submission_checks.csv", index=False)
    metadata = {"uses_test_labels": False, "uses_external_data": False,
                "time_rank_candidates": ["combined_rank40_two_hop", "blend_half_rank_half_graph"],
                "graph_only_candidate": "regularized_two_hop",
                "time_rank_candidates_use_unlabelled_test_rows": True,
                "uses_full_supplied_graph": True, "kaggle_upload_performed": False,
                "selection_warning": "Repeated use of three folds can overfit model selection; public and private test results are unknown here."}
    (REPORT_DIR / "combined_challenger_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(checks[["variant", "file", "ready"]].to_string(index=False))


if __name__ == "__main__":
    main()
