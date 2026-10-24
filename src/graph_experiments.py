from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.competition import FIGURE_DIR, PROJECT_ROOT, RANDOM_SEED, REPORT_DIR, feature_columns
from src.model_comparison import MODEL_CONFIGURATIONS


MODEL_DIR = PROJECT_ROOT / "models"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
FOLDS = [
    {"fold": "fold_0", "train_start": 1, "train_end": 14, "valid_start": 15, "valid_end": 21},
    {"fold": "fold_1", "train_start": 1, "train_end": 21, "valid_start": 22, "valid_end": 28},
    {"fold": "fold_2", "train_start": 1, "train_end": 28, "valid_start": 29, "valid_end": 35},
]


def merge_graph_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    graph_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_enriched = train.merge(graph_features, on="txId", how="left", validate="one_to_one", sort=False)
    test_enriched = test.merge(graph_features, on="txId", how="left", validate="one_to_one", sort=False)
    if not train_enriched["txId"].equals(train["txId"]):
        raise ValueError("Graph merge changed training row order")
    if not test_enriched["txId"].equals(test["txId"]):
        raise ValueError("Graph merge changed test row order")
    return train_enriched, test_enriched


def make_feature_sets(train: pd.DataFrame, groups: dict[str, list[str]]) -> dict[str, list[str]]:
    base = feature_columns(train)
    structural = groups["structural"]
    coverage = groups["coverage"]
    aggregates = groups["predecessor_aggregates"]
    return {
        "tabular": base,
        "tabular_plus_structural": base + structural,
        "tabular_plus_structural_coverage": base + structural + coverage,
        "tabular_plus_directed_predecessors": base + structural + coverage + aggregates,
    }


def _histogram_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        **MODEL_CONFIGURATIONS["Histogram gradient boosting"],
        random_state=RANDOM_SEED,
    )


def _xgboost_model(n_estimators: int, early_stopping_rounds: int | None = None) -> XGBClassifier:
    configuration = MODEL_CONFIGURATIONS["XGBoost"].copy()
    configuration["n_estimators"] = n_estimators
    configuration.pop("early_stopping_rounds", None)
    if early_stopping_rounds is not None:
        configuration["early_stopping_rounds"] = early_stopping_rounds
    return XGBClassifier(
        **configuration,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=6,
        random_state=RANDOM_SEED,
    )


def fit_xgboost_for_fold(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    development_end: int,
) -> tuple[XGBClassifier, np.ndarray, int]:
    early_stop_start = development_end - 3
    tuning_train = development.loc[development["time_step"] < early_stop_start]
    tuning_valid = development.loc[development["time_step"].between(early_stop_start, development_end)]
    if tuning_train["label"].nunique() != 2 or tuning_valid["label"].nunique() != 2:
        raise ValueError("XGBoost internal temporal tuning split lacks both classes")
    tuning_model = _xgboost_model(n_estimators=1_500, early_stopping_rounds=75)
    tuning_model.fit(
        tuning_train[features].to_numpy(dtype=np.float32),
        tuning_train["label"].to_numpy(dtype=np.int8),
        eval_set=[
            (
                tuning_valid[features].to_numpy(dtype=np.float32),
                tuning_valid["label"].to_numpy(dtype=np.int8),
            )
        ],
        verbose=False,
    )
    selected_iterations = int(tuning_model.best_iteration) + 1
    final_model = _xgboost_model(n_estimators=selected_iterations)
    final_model.fit(
        development[features].to_numpy(dtype=np.float32),
        development["label"].to_numpy(dtype=np.int8),
    )
    probabilities = final_model.predict_proba(validation[features].to_numpy(dtype=np.float32))[:, 1]
    return final_model, probabilities, selected_iterations


def _summarize(results: pd.DataFrame, group_column: str) -> pd.DataFrame:
    return (
        results.groupby(group_column)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            min_roc_auc=("roc_auc", "min"),
            max_roc_auc=("roc_auc", "max"),
            standard_deviation=("roc_auc", "std"),
            total_runtime_seconds=("runtime_seconds", "sum"),
        )
        .reset_index()
        .sort_values(["mean_roc_auc", "min_roc_auc"], ascending=False)
        .reset_index(drop=True)
    )


def _append_log(rows: pd.DataFrame, prefix: str) -> None:
    log_path = PROJECT_ROOT / "experiment_log.csv"
    new_rows = rows.copy()
    new_rows.insert(0, "experiment_id", [f"{prefix}_{index + 1:02d}" for index in range(len(new_rows))])
    new_rows.insert(1, "date", "2026-09-11")
    new_rows["feature_set"] = new_rows.get("feature_group", new_rows.get("candidate", "graph_experiment"))
    new_rows["model"] = new_rows.get("model_name", "probability_blend")
    new_rows["variant"] = prefix
    new_rows["notes"] = "Three expanding temporal folds; labels excluded from graph features"
    if log_path.exists():
        existing = pd.read_csv(log_path)
        existing = existing.loc[~existing["experiment_id"].astype(str).str.startswith(f"{prefix}_")]
        new_rows = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    new_rows.to_csv(log_path, index=False)


def _plot_ablation(summary: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    plotting = summary.sort_values("mean_roc_auc", ascending=True)
    figure, axis = plt.subplots(figsize=(9.5, 5.3))
    axis.barh(plotting["feature_group"], plotting["mean_roc_auc"], color="#26547C")
    axis.scatter(plotting["min_roc_auc"], plotting["feature_group"], color="#D08C24", zorder=3, label="Weakest fold")
    lower = max(0.5, float(plotting[["mean_roc_auc", "min_roc_auc"]].min().min()) - 0.02)
    upper = min(1.0, float(plotting["mean_roc_auc"].max()) + 0.02)
    axis.set(
        title="Histogram gradient boosting graph-feature ablation",
        xlabel="ROC-AUC (bars: three-fold mean; dots: weakest fold)",
        ylabel="Feature group",
        xlim=(lower, upper),
    )
    axis.legend(frameon=False)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.4f", padding=3, fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "graph_feature_ablation.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_candidates(summary: pd.DataFrame) -> None:
    plotting = summary.sort_values("mean_roc_auc", ascending=True)
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    colors = ["#D08C24" if selected else "#26547C" for selected in plotting["selected"]]
    axis.barh(plotting["candidate"], plotting["mean_roc_auc"], color=colors)
    axis.scatter(plotting["min_roc_auc"], plotting["candidate"], color="#30343B", zorder=3, label="Weakest fold")
    lower = max(0.5, float(plotting[["mean_roc_auc", "min_roc_auc"]].min().min()) - 0.02)
    upper = min(1.0, float(plotting["mean_roc_auc"].max()) + 0.02)
    axis.set(
        title="Final candidate comparison across three temporal folds",
        xlabel="ROC-AUC (bars: mean; dots: weakest fold)",
        ylabel="Candidate",
        xlim=(lower, upper),
    )
    axis.legend(frameon=False)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.4f", padding=3, fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "submission_candidate_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def run_graph_experiments(
    train_enriched: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labelled = train_enriched.loc[train_enriched["label"].notna()].copy()
    ablation_rows = []
    histogram_predictions: dict[tuple[str, str], np.ndarray] = {}
    fold_labels: dict[str, np.ndarray] = {}

    for fold in FOLDS:
        development = labelled.loc[labelled["time_step"].between(fold["train_start"], fold["train_end"])]
        validation = labelled.loc[labelled["time_step"].between(fold["valid_start"], fold["valid_end"])]
        fold_labels[fold["fold"]] = validation["label"].to_numpy(dtype=np.int8)
        for feature_group, features in feature_sets.items():
            started = time.perf_counter()
            model = _histogram_model()
            model.fit(development[features], development["label"].astype(int))
            probabilities = model.predict_proba(validation[features])[:, 1]
            elapsed = time.perf_counter() - started
            histogram_predictions[(feature_group, fold["fold"])] = probabilities
            ablation_rows.append(
                {
                    **fold,
                    "model_name": "Histogram gradient boosting",
                    "feature_group": feature_group,
                    "feature_count": len(features),
                    "train_rows": len(development),
                    "validation_rows": len(validation),
                    "roc_auc": float(roc_auc_score(validation["label"].astype(int), probabilities)),
                    "runtime_seconds": elapsed,
                    "random_seed": RANDOM_SEED,
                }
            )
    ablation_results = pd.DataFrame(ablation_rows)
    ablation_summary = _summarize(ablation_results, "feature_group")
    feature_counts = ablation_results[["feature_group", "feature_count"]].drop_duplicates()
    ablation_summary = ablation_summary.merge(feature_counts, on="feature_group", how="left")
    ablation_summary = ablation_summary.sort_values(
        ["mean_roc_auc", "min_roc_auc", "feature_count"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ablation_summary["selected"] = False
    ablation_summary.loc[0, "selected"] = True
    selected_group = ablation_summary.loc[0, "feature_group"]

    xgboost_rows = []
    xgboost_predictions: dict[tuple[str, str], np.ndarray] = {}
    xgboost_groups = ["tabular"]
    if selected_group != "tabular":
        xgboost_groups.append(selected_group)
    for fold in FOLDS:
        development = labelled.loc[labelled["time_step"].between(fold["train_start"], fold["train_end"])]
        validation = labelled.loc[labelled["time_step"].between(fold["valid_start"], fold["valid_end"])]
        for feature_group in xgboost_groups:
            features = feature_sets[feature_group]
            started = time.perf_counter()
            _, probabilities, iterations = fit_xgboost_for_fold(
                development, validation, features, fold["train_end"]
            )
            elapsed = time.perf_counter() - started
            xgboost_predictions[(feature_group, fold["fold"])] = probabilities
            xgboost_rows.append(
                {
                    **fold,
                    "model_name": "XGBoost",
                    "feature_group": feature_group,
                    "feature_count": len(features),
                    "train_rows": len(development),
                    "validation_rows": len(validation),
                    "roc_auc": float(roc_auc_score(validation["label"].astype(int), probabilities)),
                    "runtime_seconds": elapsed,
                    "iterations": iterations,
                    "random_seed": RANDOM_SEED,
                }
            )
    xgboost_results = pd.DataFrame(xgboost_rows)

    candidate_rows = []
    for hgb_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
        candidate = (
            "XGBoost"
            if hgb_weight == 0.0
            else "Histogram gradient boosting"
            if hgb_weight == 1.0
            else f"Blend {hgb_weight:.0%} HGB / {1-hgb_weight:.0%} XGB"
        )
        for fold in FOLDS:
            hgb_probability = histogram_predictions[(selected_group, fold["fold"])]
            xgb_probability = xgboost_predictions[(selected_group, fold["fold"])]
            blended = hgb_weight * hgb_probability + (1 - hgb_weight) * xgb_probability
            candidate_rows.append(
                {
                    "candidate": candidate,
                    "hgb_weight": hgb_weight,
                    "xgb_weight": 1 - hgb_weight,
                    "feature_group": selected_group,
                    "fold": fold["fold"],
                    "roc_auc": float(roc_auc_score(fold_labels[fold["fold"]], blended)),
                    "runtime_seconds": 0.0,
                }
            )
    candidate_results = pd.DataFrame(candidate_rows)
    candidate_summary = _summarize(candidate_results, "candidate")
    weights = candidate_results[["candidate", "hgb_weight", "xgb_weight", "feature_group"]].drop_duplicates()
    candidate_summary = candidate_summary.merge(weights, on="candidate", how="left")
    candidate_summary["selected"] = False
    candidate_summary.loc[0, "selected"] = True

    ablation_results.to_csv(REPORT_DIR / "graph_ablation_results.csv", index=False)
    ablation_summary.to_csv(REPORT_DIR / "graph_ablation_summary.csv", index=False)
    xgboost_results.to_csv(REPORT_DIR / "graph_xgboost_results.csv", index=False)
    candidate_results.to_csv(REPORT_DIR / "candidate_blend_results.csv", index=False)
    candidate_summary.to_csv(REPORT_DIR / "candidate_blend_summary.csv", index=False)
    _append_log(ablation_results, "S8_HGB")
    _append_log(xgboost_results, "S8_XGB")
    _append_log(candidate_results, "S9_BLEND")
    _plot_ablation(ablation_summary)
    _plot_candidates(candidate_summary)
    return ablation_results, ablation_summary, xgboost_results, candidate_summary


def _validate_submission(path: Path, submission: pd.DataFrame, sample: pd.DataFrame) -> dict:
    target = submission["target"].to_numpy(dtype=float)
    checks = {
        "path": path.name,
        "row_count": int(len(submission)),
        "columns": submission.columns.tolist(),
        "columns_match_sample": submission.columns.tolist() == sample.columns.tolist(),
        "index_matches_sample": bool(submission["index"].equals(sample["index"])),
        "all_predictions_finite": bool(np.isfinite(target).all()),
        "all_predictions_between_zero_and_one": bool(((target >= 0) & (target <= 1)).all()),
        "prediction_min": float(target.min()),
        "prediction_max": float(target.max()),
        "prediction_mean": float(target.mean()),
        "prediction_standard_deviation": float(target.std()),
        "unique_prediction_count": int(pd.Series(target).nunique()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    checks["ready"] = bool(
        checks["row_count"] == len(sample)
        and checks["columns_match_sample"]
        and checks["index_matches_sample"]
        and checks["all_predictions_finite"]
        and checks["all_predictions_between_zero_and_one"]
        and checks["unique_prediction_count"] > 1
    )
    return checks


def fit_final_models_and_create_submissions(
    train_enriched: pd.DataFrame,
    test_enriched: pd.DataFrame,
    sample_submission: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[dict, pd.DataFrame]:
    candidate_summary = pd.read_csv(REPORT_DIR / "candidate_blend_summary.csv")
    selected = candidate_summary.loc[candidate_summary["selected"]].iloc[0]
    feature_group = str(selected["feature_group"])
    features = feature_sets[feature_group]
    labelled = train_enriched.loc[train_enriched["label"].notna()].copy()

    histogram_model = _histogram_model()
    histogram_model.fit(labelled[features], labelled["label"].astype(int))
    histogram_probability = histogram_model.predict_proba(test_enriched[features])[:, 1]

    tuning_train = labelled.loc[labelled["time_step"] <= 31]
    tuning_valid = labelled.loc[labelled["time_step"].between(32, 35)]
    tuning_model = _xgboost_model(n_estimators=1_500, early_stopping_rounds=75)
    tuning_model.fit(
        tuning_train[features].to_numpy(dtype=np.float32),
        tuning_train["label"].to_numpy(dtype=np.int8),
        eval_set=[
            (
                tuning_valid[features].to_numpy(dtype=np.float32),
                tuning_valid["label"].to_numpy(dtype=np.int8),
            )
        ],
        verbose=False,
    )
    final_xgboost_iterations = int(tuning_model.best_iteration) + 1
    xgboost_model = _xgboost_model(n_estimators=final_xgboost_iterations)
    xgboost_model.fit(
        labelled[features].to_numpy(dtype=np.float32),
        labelled["label"].to_numpy(dtype=np.int8),
    )
    xgboost_probability = xgboost_model.predict_proba(test_enriched[features].to_numpy(dtype=np.float32))[:, 1]

    hgb_weight = float(selected["hgb_weight"])
    xgb_weight = float(selected["xgb_weight"])
    primary_probability = hgb_weight * histogram_probability + xgb_weight * xgboost_probability
    probabilities = {
        "submission_primary.csv": primary_probability,
        "submission_histogram_gradient_boosting.csv": histogram_probability,
        "submission_xgboost.csv": xgboost_probability,
    }
    SUBMISSION_DIR.mkdir(exist_ok=True)
    validations = []
    for filename, probability in probabilities.items():
        submission = sample_submission.copy()
        submission["target"] = probability
        path = SUBMISSION_DIR / filename
        submission.to_csv(path, index=False)
        validations.append(_validate_submission(path, submission, sample_submission))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": histogram_model, "features": features, "feature_group": feature_group},
        MODEL_DIR / "histogram_gradient_boosting_final.joblib",
    )
    joblib.dump(
        {
            "model": xgboost_model,
            "features": features,
            "feature_group": feature_group,
            "n_estimators": final_xgboost_iterations,
        },
        MODEL_DIR / "xgboost_final.joblib",
    )

    validation_frame = pd.DataFrame(validations)
    validation_frame.to_csv(REPORT_DIR / "submission_validation.csv", index=False)
    metadata = {
        "selected_candidate": str(selected["candidate"]),
        "selected_feature_group": feature_group,
        "feature_count": len(features),
        "hgb_weight": hgb_weight,
        "xgb_weight": xgb_weight,
        "final_xgboost_iterations": final_xgboost_iterations,
        "labelled_training_rows": int(len(labelled)),
        "test_rows": int(len(test_enriched)),
        "primary_submission": "submission_primary.csv",
        "all_submission_files_ready": bool(validation_frame["ready"].all()),
        "kaggle_upload_performed": False,
    }
    with (REPORT_DIR / "final_candidate_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata, validation_frame
