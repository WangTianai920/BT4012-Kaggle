from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = PROJECT_ROOT / "figures"
RANDOM_SEED = 42


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return anonymized feature columns in numeric order."""
    columns = [column for column in frame.columns if column.startswith("feat_")]
    return sorted(columns, key=lambda value: int(value.split("_")[1]))


def load_competition_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four competition files from the local data directory."""
    train = pd.read_csv(DATA_DIR / "train.csv", low_memory=False)
    test = pd.read_csv(DATA_DIR / "test.csv", low_memory=False)
    edges = pd.read_csv(DATA_DIR / "txs_edgelist.csv")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return train, test, edges, sample_submission


def _finite_feature_profile(frame: pd.DataFrame, columns: list[str]) -> dict[str, int | float]:
    values = frame[columns].to_numpy(dtype="float64", copy=False)
    return {
        "missing_cells": int(np.isnan(values).sum()),
        "missing_rate": float(np.isnan(values).mean()),
        "infinite_cells": int(np.isinf(values).sum()),
    }


def run_data_audit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    edges: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Run high-value quality, integrity, temporal, and drift checks."""
    REPORT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    features = feature_columns(train)
    labelled = train.loc[train["label"].notna()].copy()

    train_profile = _finite_feature_profile(train, features)
    test_profile = _finite_feature_profile(test, features)
    constant_train = [column for column in features if train[column].nunique(dropna=False) <= 1]

    all_known_ids = set(train["txId"]).union(test["txId"])
    edge_nodes = set(edges["txId1"]).union(edges["txId2"])
    orphan_edge_nodes = edge_nodes.difference(all_known_ids)

    label_counts = train["label"].value_counts(dropna=False)
    class_by_time = (
        labelled.groupby("time_step")["label"]
        .agg(total="size", illicit="sum", illicit_rate="mean")
        .reset_index()
    )
    rows_by_time = pd.concat(
        [
            train.groupby("time_step").size().rename("rows").reset_index().assign(dataset="train"),
            test.groupby("time_step").size().rename("rows").reset_index().assign(dataset="test"),
        ],
        ignore_index=True,
    )

    train_means = labelled[features].mean()
    test_means = test[features].mean()
    pooled_scale = labelled[features].std(ddof=0).replace(0, np.nan)
    standardized_shift = ((test_means - train_means) / pooled_scale).abs().sort_values(ascending=False)
    drift_table = standardized_shift.rename("absolute_standardized_mean_shift").reset_index()
    drift_table.columns = ["feature", "absolute_standardized_mean_shift"]

    summary = {
        "intended_grain": "one row per transaction",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(len(features)),
        "labelled_train_rows": int(len(labelled)),
        "unknown_label_rows": int(train["label"].isna().sum()),
        "licit_rows": int(label_counts.get(0.0, 0)),
        "illicit_rows": int(label_counts.get(1.0, 0)),
        "illicit_rate_among_labelled": float(labelled["label"].mean()),
        "train_time_min": int(train["time_step"].min()),
        "train_time_max": int(train["time_step"].max()),
        "test_time_min": int(test["time_step"].min()),
        "test_time_max": int(test["time_step"].max()),
        "duplicate_train_txid": int(train["txId"].duplicated().sum()),
        "duplicate_test_txid": int(test["txId"].duplicated().sum()),
        "train_test_txid_overlap": int(len(set(train["txId"]).intersection(test["txId"]))),
        "duplicate_edges": int(edges.duplicated().sum()),
        "self_loop_edges": int((edges["txId1"] == edges["txId2"]).sum()),
        "edge_rows": int(len(edges)),
        "edge_unique_nodes": int(len(edge_nodes)),
        "edge_nodes_absent_from_train_and_test": int(len(orphan_edge_nodes)),
        "model_transaction_ids_absent_from_edge_list": int(len(all_known_ids.difference(edge_nodes))),
        "sample_rows": int(len(sample_submission)),
        "sample_columns": sample_submission.columns.tolist(),
        "sample_index_matches_test": bool(sample_submission["index"].equals(test["index"])),
        "train_feature_profile": train_profile,
        "test_feature_profile": test_profile,
        "constant_train_features": constant_train,
    }

    with (REPORT_DIR / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    class_by_time.to_csv(REPORT_DIR / "class_by_time.csv", index=False)
    drift_table.to_csv(REPORT_DIR / "feature_drift.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(10, 4.8))
    for dataset, color in [("train", "#26547C"), ("test", "#D08C24")]:
        subset = rows_by_time.loc[rows_by_time["dataset"] == dataset]
        axis.bar(subset["time_step"], subset["rows"], label=dataset.title(), color=color, alpha=0.9)
    axis.axvline(35.5, color="#30343B", linestyle="--", linewidth=1.2, label="Train/test boundary")
    axis.set(title="Transactions by time step", xlabel="Time step", ylabel="Number of transactions")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "transactions_by_time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(
        class_by_time["time_step"],
        class_by_time["illicit_rate"],
        color="#9C3D54",
        marker="o",
        linewidth=1.8,
    )
    axis.set(
        title="Illicit share among labelled transactions by time step",
        xlabel="Time step",
        ylabel="Illicit share",
        ylim=(0, max(0.25, class_by_time["illicit_rate"].max() * 1.1)),
    )
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "illicit_rate_by_time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return summary, class_by_time, drift_table


def make_logistic_pipeline(class_weight: str | None) -> Pipeline:
    """Create the deterministic baseline pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight=class_weight,
                    max_iter=2_000,
                    random_state=RANDOM_SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def run_temporal_baselines(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate unweighted and balanced logistic regression on expanding time folds."""
    REPORT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    features = feature_columns(train)
    labelled = train.loc[train["label"].notna()].copy()
    folds = [
        {"fold": "fold_1", "train_start": 1, "train_end": 21, "valid_start": 22, "valid_end": 28},
        {"fold": "fold_2", "train_start": 1, "train_end": 28, "valid_start": 29, "valid_end": 35},
    ]
    variants = [("unweighted", None), ("balanced", "balanced")]
    results: list[dict] = []
    fitted_models: dict[tuple[str, str], Pipeline] = {}

    for fold in folds:
        train_mask = labelled["time_step"].between(fold["train_start"], fold["train_end"])
        valid_mask = labelled["time_step"].between(fold["valid_start"], fold["valid_end"])
        development = labelled.loc[train_mask]
        validation = labelled.loc[valid_mask]
        if development["label"].nunique() != 2 or validation["label"].nunique() != 2:
            raise ValueError(f"{fold['fold']} does not contain both classes in training and validation")

        for variant_name, class_weight in variants:
            started = time.perf_counter()
            pipeline = make_logistic_pipeline(class_weight)
            pipeline.fit(development[features], development["label"].astype(int))
            probabilities = pipeline.predict_proba(validation[features])[:, 1]
            elapsed = time.perf_counter() - started
            auc = roc_auc_score(validation["label"].astype(int), probabilities)
            results.append(
                {
                    **fold,
                    "variant": variant_name,
                    "train_rows": int(len(development)),
                    "validation_rows": int(len(validation)),
                    "train_illicit_rate": float(development["label"].mean()),
                    "validation_illicit_rate": float(validation["label"].mean()),
                    "roc_auc": float(auc),
                    "runtime_seconds": float(elapsed),
                    "random_seed": RANDOM_SEED,
                }
            )
            fitted_models[(fold["fold"], variant_name)] = pipeline

    result_frame = pd.DataFrame(results)
    summary = (
        result_frame.groupby("variant")["roc_auc"]
        .agg(mean_roc_auc="mean", min_roc_auc="min", max_roc_auc="max", fold_count="count")
        .reset_index()
        .sort_values(["mean_roc_auc", "min_roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    selected_variant = summary.iloc[0]["variant"]
    summary["selected"] = summary["variant"].eq(selected_variant)

    result_frame.to_csv(REPORT_DIR / "validation_results.csv", index=False)
    summary.to_csv(REPORT_DIR / "baseline_summary.csv", index=False)

    log = result_frame.copy()
    log.insert(0, "experiment_id", [f"LR_{row.variant}_{row.fold}" for row in log.itertuples()])
    log.insert(1, "date", "2026-09-11")
    log.insert(2, "feature_set", "feat_1_to_feat_165")
    log.insert(3, "model", "median_imputer_standard_scaler_logistic_regression")
    log["notes"] = "Expanding-window temporal validation; unknown labels excluded"
    log_path = PROJECT_ROOT / "experiment_log.csv"
    if log_path.exists():
        existing_log = pd.read_csv(log_path)
        if "experiment_id" in existing_log.columns:
            existing_log = existing_log.loc[
                ~existing_log["experiment_id"].astype(str).str.startswith("LR_")
            ]
        log = pd.concat([existing_log, log], ignore_index=True, sort=False)
    log.to_csv(log_path, index=False)

    latest_pipeline = fitted_models[("fold_2", selected_variant)]
    coefficients = latest_pipeline.named_steps["model"].coef_[0]
    coefficient_table = pd.DataFrame(
        {"feature": features, "coefficient": coefficients, "absolute_coefficient": np.abs(coefficients)}
    ).sort_values("absolute_coefficient", ascending=False)
    coefficient_table.to_csv(REPORT_DIR / "baseline_coefficients_latest_fold.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    pivot = result_frame.pivot(index="fold", columns="variant", values="roc_auc").loc[["fold_1", "fold_2"]]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    pivot.plot(kind="bar", ax=axis, color=["#26547C", "#D08C24"], width=0.72)
    axis.set(title="Logistic-regression ROC-AUC by temporal fold", xlabel="Temporal fold", ylabel="ROC-AUC")
    axis.set_ylim(max(0.5, float(pivot.min().min()) - 0.04), min(1.0, float(pivot.max().max()) + 0.04))
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Class weighting", frameon=False)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.4f", padding=3, fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "baseline_auc_by_fold.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    return result_frame, summary
