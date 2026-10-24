from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.competition import FIGURE_DIR, PROJECT_ROOT, RANDOM_SEED, REPORT_DIR, feature_columns


FOLDS = [
    {"fold": "fold_1", "train_start": 1, "train_end": 21, "valid_start": 22, "valid_end": 28},
    {"fold": "fold_2", "train_start": 1, "train_end": 28, "valid_start": 29, "valid_end": 35},
]


MODEL_CONFIGURATIONS = {
    "XGBoost": {
        "n_estimators": 1_500,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
        "early_stopping_rounds": 75,
    },
    "Histogram gradient boosting": {
        "max_iter": 300,
        "learning_rate": 0.06,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "early_stopping": False,
    },
    "Extra Trees": {
        "n_estimators": 600,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "class_weight": None,
    },
    "Neural network": {
        "hidden_layer_sizes": [128, 64],
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": 256,
        "learning_rate_init": 0.0005,
        "max_iter": 60,
        "early_stopping": False,
    },
}


def _build_model(model_name: str, xgboost_n_estimators: int | None = None):
    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        configuration = MODEL_CONFIGURATIONS[model_name].copy()
        if xgboost_n_estimators is not None:
            configuration["n_estimators"] = xgboost_n_estimators
            configuration.pop("early_stopping_rounds", None)
        return XGBClassifier(
            **configuration,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=6,
            random_state=RANDOM_SEED,
        )
    if model_name == "Histogram gradient boosting":
        return HistGradientBoostingClassifier(
            **MODEL_CONFIGURATIONS[model_name],
            random_state=RANDOM_SEED,
        )
    if model_name == "Extra Trees":
        return ExtraTreesClassifier(
            **MODEL_CONFIGURATIONS[model_name],
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    if model_name == "Neural network":
        neural_config = MODEL_CONFIGURATIONS[model_name].copy()
        neural_config["hidden_layer_sizes"] = tuple(neural_config["hidden_layer_sizes"])
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(**neural_config, random_state=RANDOM_SEED)),
            ]
        )
    raise KeyError(model_name)


def _fit_and_predict(model_name: str, model, x_train, y_train, x_valid):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_valid)[:, 1]
    convergence_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)

    if model_name == "Neural network":
        iteration = int(model.named_steps["model"].n_iter_)
        converged = not convergence_warning
    elif hasattr(model, "n_iter_"):
        iteration = int(model.n_iter_)
        converged = True
    else:
        iteration = None
        converged = True
    return probabilities, iteration, converged


def _append_experiment_log(results: pd.DataFrame) -> None:
    log_path = PROJECT_ROOT / "experiment_log.csv"
    new_log = results.copy()
    new_log.insert(0, "experiment_id", [f"S7_{row.model_slug}_{row.fold}" for row in new_log.itertuples()])
    new_log.insert(1, "date", "2026-09-11")
    new_log.insert(2, "feature_set", "feat_1_to_feat_165")
    new_log.rename(columns={"model_name": "model"}, inplace=True)
    new_log["variant"] = "step_7_fixed_configuration"
    new_log["notes"] = "Expanding-window temporal validation; unknown labels excluded; no random internal validation"

    if log_path.exists():
        existing = pd.read_csv(log_path)
        if "experiment_id" in existing.columns:
            existing = existing.loc[~existing["experiment_id"].astype(str).str.startswith("S7_")]
        combined = pd.concat([existing, new_log], ignore_index=True, sort=False)
    else:
        combined = new_log
    combined.to_csv(log_path, index=False)


def _make_figures(
    combined_results: pd.DataFrame,
    combined_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    per_time_results: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    model_order = combined_summary.sort_values("mean_roc_auc", ascending=True)["model_name"].tolist()
    pivot = combined_results.pivot(index="model_name", columns="fold", values="roc_auc").loc[model_order]

    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    pivot.plot(kind="barh", ax=axis, color=["#26547C", "#D08C24"], width=0.72)
    lower = max(0.5, float(pivot.min().min()) - 0.035)
    upper = min(1.0, float(pivot.max().max()) + 0.025)
    axis.set(
        title="Model ROC-AUC across expanding temporal folds",
        xlabel="ROC-AUC (focused scale for model comparison)",
        ylabel="Model",
        xlim=(lower, upper),
    )
    axis.legend(title="Validation fold", frameon=False, loc="lower right")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.4f", padding=3, fontsize=8.5)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "step7_model_auc_by_fold.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    matrix = correlations.to_numpy()
    image = axis.imshow(matrix, vmin=0.5, vmax=1.0, cmap="YlGnBu")
    axis.set_xticks(range(len(correlations.columns)), correlations.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(correlations.index)), correlations.index)
    axis.set_title("Latest-fold prediction correlation among Step 7 models")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > 0.82 else "#20252B"
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", color=color)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson correlation")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "step7_prediction_correlation.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    colors = {
        "XGBoost": "#26547C",
        "Histogram gradient boosting": "#D08C24",
        "Extra Trees": "#8B5E83",
        "Neural network": "#6B7D3A",
    }
    markers = {"XGBoost": "o", "Histogram gradient boosting": "s", "Extra Trees": "^", "Neural network": "D"}
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    for model_name in MODEL_CONFIGURATIONS:
        subset = per_time_results.loc[per_time_results["model_name"] == model_name]
        axis.plot(
            subset["time_step"],
            subset["roc_auc"],
            label=model_name,
            color=colors[model_name],
            marker=markers[model_name],
            linewidth=1.6,
            markersize=4.5,
        )
    axis.axvline(28.5, color="#30343B", linestyle="--", linewidth=1.1, label="Fold boundary")
    axis.set(
        title="Within-time-step ROC-AUC across validation periods",
        xlabel="Validation time step",
        ylabel="ROC-AUC",
        ylim=(max(0.45, float(per_time_results["roc_auc"].min()) - 0.05), 1.01),
    )
    axis.set_xticks(sorted(per_time_results["time_step"].unique()))
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "step7_auc_by_time_step.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def run_model_comparison(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train four model families on the same two expanding temporal folds."""
    REPORT_DIR.mkdir(exist_ok=True)
    features = feature_columns(train)
    labelled = train.loc[train["label"].notna()].copy()
    results: list[dict] = []
    per_time_rows: list[dict] = []
    latest_predictions: dict[str, np.ndarray] = {}

    with (REPORT_DIR / "model_configurations.json").open("w", encoding="utf-8") as handle:
        json.dump(MODEL_CONFIGURATIONS, handle, indent=2)

    for fold in FOLDS:
        development = labelled.loc[labelled["time_step"].between(fold["train_start"], fold["train_end"])]
        validation = labelled.loc[labelled["time_step"].between(fold["valid_start"], fold["valid_end"])]
        if development["label"].nunique() != 2 or validation["label"].nunique() != 2:
            raise ValueError(f"{fold['fold']} lacks both classes")

        x_train = development[features].to_numpy(dtype=np.float32)
        y_train = development["label"].to_numpy(dtype=np.int8)
        x_valid = validation[features].to_numpy(dtype=np.float32)
        y_valid = validation["label"].to_numpy(dtype=np.int8)

        for model_name in MODEL_CONFIGURATIONS:
            started = time.perf_counter()
            early_stop_start = None
            early_stop_end = None
            if model_name == "XGBoost":
                early_stop_start = fold["train_end"] - 3
                early_stop_end = fold["train_end"]
                tuning_train = development.loc[development["time_step"] < early_stop_start]
                tuning_valid = development.loc[development["time_step"].between(early_stop_start, early_stop_end)]
                tuning_model = _build_model(model_name)
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
                iteration = int(tuning_model.best_iteration) + 1
                model = _build_model(model_name, xgboost_n_estimators=iteration)
                model.fit(x_train, y_train)
                probabilities = model.predict_proba(x_valid)[:, 1]
                converged = True
            else:
                model = _build_model(model_name)
                probabilities, iteration, converged = _fit_and_predict(
                    model_name, model, x_train, y_train, x_valid
                )
            elapsed = time.perf_counter() - started
            results.append(
                {
                    **fold,
                    "model_name": model_name,
                    "model_slug": model_name.lower().replace(" ", "_").replace("-", "_"),
                    "train_rows": int(len(development)),
                    "validation_rows": int(len(validation)),
                    "train_illicit_rate": float(y_train.mean()),
                    "validation_illicit_rate": float(y_valid.mean()),
                    "roc_auc": float(roc_auc_score(y_valid, probabilities)),
                    "runtime_seconds": float(elapsed),
                    "iterations": iteration,
                    "converged": bool(converged),
                    "early_stop_start": early_stop_start,
                    "early_stop_end": early_stop_end,
                    "random_seed": RANDOM_SEED,
                }
            )
            validation_diagnostics = pd.DataFrame(
                {"time_step": validation["time_step"].to_numpy(), "label": y_valid, "probability": probabilities}
            )
            for time_step, time_group in validation_diagnostics.groupby("time_step"):
                if time_group["label"].nunique() == 2:
                    per_time_rows.append(
                        {
                            "fold": fold["fold"],
                            "model_name": model_name,
                            "time_step": int(time_step),
                            "rows": int(len(time_group)),
                            "illicit_rows": int(time_group["label"].sum()),
                            "illicit_rate": float(time_group["label"].mean()),
                            "roc_auc": float(roc_auc_score(time_group["label"], time_group["probability"])),
                        }
                    )
            if fold["fold"] == "fold_2":
                latest_predictions[model_name] = probabilities

    result_frame = pd.DataFrame(results)
    per_time_results = pd.DataFrame(per_time_rows)
    per_time_summary = (
        per_time_results.groupby("model_name")["roc_auc"]
        .agg(mean_within_time_auc="mean", min_time_step_auc="min")
        .reset_index()
    )
    step7_summary = (
        result_frame.groupby("model_name")
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            min_roc_auc=("roc_auc", "min"),
            max_roc_auc=("roc_auc", "max"),
            standard_deviation=("roc_auc", "std"),
            total_runtime_seconds=("runtime_seconds", "sum"),
            all_runs_converged=("converged", "all"),
        )
        .reset_index()
        .sort_values(["mean_roc_auc", "min_roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    step7_summary = step7_summary.merge(per_time_summary, on="model_name", how="left")
    step7_summary["selected_step7_model"] = False
    step7_summary.loc[0, "selected_step7_model"] = True

    baseline_results = pd.read_csv(REPORT_DIR / "validation_results.csv")
    baseline_results = baseline_results.loc[baseline_results["variant"] == "unweighted", ["fold", "roc_auc"]]
    baseline_results["model_name"] = "Logistic regression"
    combined_results = pd.concat(
        [baseline_results, result_frame[["fold", "model_name", "roc_auc"]]], ignore_index=True
    )
    combined_summary = (
        combined_results.groupby("model_name")["roc_auc"]
        .agg(mean_roc_auc="mean", min_roc_auc="min", max_roc_auc="max")
        .reset_index()
        .sort_values(["mean_roc_auc", "min_roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    combined_summary["selected_overall"] = False
    combined_summary.loc[0, "selected_overall"] = True

    correlation_frame = pd.DataFrame(latest_predictions).corr()
    result_frame.to_csv(REPORT_DIR / "step7_validation_results.csv", index=False)
    per_time_results.to_csv(REPORT_DIR / "step7_auc_by_time_step.csv", index=False)
    step7_summary.to_csv(REPORT_DIR / "step7_model_summary.csv", index=False)
    combined_results.to_csv(REPORT_DIR / "model_comparison_results.csv", index=False)
    combined_summary.to_csv(REPORT_DIR / "model_comparison_summary.csv", index=False)
    correlation_frame.to_csv(REPORT_DIR / "step7_latest_fold_prediction_correlation.csv")
    _append_experiment_log(result_frame)
    _make_figures(combined_results, combined_summary, correlation_frame, per_time_results)
    return result_frame, step7_summary, combined_summary, correlation_frame, per_time_results
