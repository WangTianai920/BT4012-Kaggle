"""Leakage-safe historical-model neighbour-risk aggregation experiment."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.combined_challengers import load_features
from src.competition import PROJECT_ROOT, REPORT_DIR, load_competition_data
from src.graph_experiments import FOLDS
from src.targeted_hgb_challengers import _model, _variants, _weights


CONTROL = "incumbent"
META_WINDOW = 7
BLEND_WEIGHTS = (0.10, 0.25)
PAYLOAD_PATH = REPORT_DIR / "historical_neighbor_risk_submission_payload.json"


def aggregate_neighbor_risk(
    rows: pd.DataFrame,
    edges: pd.DataFrame,
    probability: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate model-estimated risk; no observed neighbour labels are used."""
    if len(rows) != len(probability) or rows["txId"].duplicated().any():
        raise ValueError("Risk rows must have one prediction per unique transaction")
    risk = pd.Series(probability, index=rows["txId"].to_numpy(), name="risk")
    annotated = edges.copy()
    annotated["source_risk"] = annotated["txId1"].map(risk)
    annotated["target_risk"] = annotated["txId2"].map(risk)
    incoming = annotated.loc[annotated["source_risk"].notna()].groupby("txId2")["source_risk"].agg(
        ["count", "mean", "max", "std"]
    ).rename(columns=lambda name: f"hist_pred_in_{name}")
    outgoing = annotated.loc[annotated["target_risk"].notna()].groupby("txId1")["target_risk"].agg(
        ["count", "mean", "max", "std"]
    ).rename(columns=lambda name: f"hist_pred_out_{name}")
    result = rows[["txId", "time_step"]].copy().set_index("txId")
    result["hist_self_risk"] = risk
    result = result.join(incoming, how="left").join(outgoing, how="left")
    for direction in ("in", "out"):
        count = f"hist_pred_{direction}_count"
        mean = f"hist_pred_{direction}_mean"
        maximum = f"hist_pred_{direction}_max"
        standard_deviation = f"hist_pred_{direction}_std"
        result[count] = result[count].fillna(0)
        result[f"hist_pred_{direction}_log_count"] = np.log1p(result[count])
        result[f"hist_pred_{direction}_present"] = result[count].gt(0).astype(float)
        result[mean] = result[mean].fillna(result["hist_self_risk"])
        result[maximum] = result[maximum].fillna(result["hist_self_risk"])
        result[standard_deviation] = result[standard_deviation].fillna(0)
        result[f"hist_pred_{direction}_minus_self"] = result[mean] - result["hist_self_risk"]
        result[f"hist_pred_{direction}_mean_rank"] = result.groupby("time_step")[mean].rank(
            method="average", pct=True
        )
    result["hist_self_risk_rank"] = result.groupby("time_step")["hist_self_risk"].rank(
        method="average", pct=True
    )
    result = result.drop(columns=["hist_pred_in_count", "hist_pred_out_count"])
    result = result.reset_index()
    feature_columns = [column for column in result if column not in {"txId", "time_step"}]
    if not np.isfinite(result[feature_columns].to_numpy()).all():
        raise ValueError("Historical neighbour-risk features contain non-finite values")
    return result, feature_columns


def _fit_first_stage(
    labelled: pd.DataFrame,
    columns: list[str],
    reference: int,
    half_life: float,
):
    if labelled["label"].nunique() != 2:
        raise ValueError("First-stage historical window lacks both classes")
    model = _model()
    model.fit(
        labelled[columns],
        labelled["label"].astype(int),
        sample_weight=_weights(labelled["time_step"], reference, half_life),
    )
    return model


def _score(frame: pd.DataFrame, probability: np.ndarray, fold: str, variant: str, step: object) -> dict:
    truth = frame["label"].astype(int)
    return {
        "fold": fold,
        "time_step": step,
        "variant": variant,
        "rows": len(frame),
        "roc_auc": roc_auc_score(truth, probability),
        "average_precision": average_precision_score(truth, probability),
    }


def main() -> None:
    train, test, sample, groups = load_features()
    _, _, edges, _ = load_competition_data()
    base_columns, half_life = _variants(groups)["regularized_drop_top10_drift"]
    stored = pd.read_csv(REPORT_DIR / "targeted_hgb_oof_predictions.csv")
    labelled = train.loc[train["label"].notna()].copy()
    records: list[dict] = []
    oof_parts: list[pd.DataFrame] = []
    fold_design: dict[str, dict] = {}
    runtimes: dict[str, float] = {}
    for fold in FOLDS:
        train_end = fold["train_end"]
        seed_end = train_end - META_WINDOW
        meta_start = seed_end + 1
        development = labelled.loc[labelled["time_step"].between(fold["train_start"], train_end)]
        seed = labelled.loc[labelled["time_step"].between(fold["train_start"], seed_end)]
        meta_labelled = labelled.loc[labelled["time_step"].between(meta_start, train_end)]
        meta_all = train.loc[train["time_step"].between(meta_start, train_end)]
        validation = labelled.loc[labelled["time_step"].between(fold["valid_start"], fold["valid_end"])]
        validation_all = train.loc[train["time_step"].between(fold["valid_start"], fold["valid_end"])]
        fold_design[fold["fold"]] = {
            "first_stage_training_steps_for_meta": [fold["train_start"], seed_end],
            "second_stage_training_steps": [meta_start, train_end],
            "first_stage_training_steps_for_validation": [fold["train_start"], train_end],
            "validation_steps": [fold["valid_start"], fold["valid_end"]],
        }

        started = time.perf_counter()
        meta_first_stage = _fit_first_stage(seed, base_columns, seed_end, half_life)
        meta_risk, neighbor_columns = aggregate_neighbor_risk(
            meta_all,
            edges,
            meta_first_stage.predict_proba(meta_all[base_columns])[:, 1],
        )
        validation_first_stage = _fit_first_stage(development, base_columns, train_end, half_life)
        validation_risk, validation_columns = aggregate_neighbor_risk(
            validation_all,
            edges,
            validation_first_stage.predict_proba(validation_all[base_columns])[:, 1],
        )
        if neighbor_columns != validation_columns:
            raise ValueError("Training and validation neighbour-risk schemas differ")
        runtimes[f"{fold['fold']}:feature_generation"] = time.perf_counter() - started

        meta = meta_labelled.merge(meta_risk, on=["txId", "time_step"], validate="one_to_one", sort=False)
        valid = validation.merge(validation_risk, on=["txId", "time_step"], validate="one_to_one", sort=False)
        current = valid[["txId", "time_step", "label"]].merge(
            stored.loc[stored["fold"].eq(fold["fold"]), [
                "txId", "time_step", "label", "regularized_drop_top10_drift"
            ]],
            on=["txId", "time_step", "label"],
            validate="one_to_one",
            sort=False,
        ).rename(columns={"regularized_drop_top10_drift": CONTROL})

        models = {
            "recent_base_control": base_columns,
            "plus_historical_neighbor_risk": base_columns + neighbor_columns,
            "historical_neighbor_risk_only": neighbor_columns,
        }
        predictions: dict[str, np.ndarray] = {CONTROL: current[CONTROL].to_numpy()}
        for name, columns in models.items():
            started = time.perf_counter()
            model = _model()
            model.fit(
                meta[columns],
                meta["label"].astype(int),
                sample_weight=_weights(meta["time_step"], train_end, half_life),
            )
            predictions[name] = model.predict_proba(valid[columns])[:, 1]
            runtimes[f"{fold['fold']}:{name}"] = time.perf_counter() - started
        for name in ("plus_historical_neighbor_risk", "historical_neighbor_risk_only"):
            for weight in BLEND_WEIGHTS:
                predictions[f"blend_{name}_{int(weight * 100)}"] = (
                    (1 - weight) * predictions[CONTROL] + weight * predictions[name]
                )
        for name, probability in predictions.items():
            current[name] = probability
            records.append(_score(current, probability, fold["fold"], name, "all"))
            for step, group in current.groupby("time_step"):
                records.append(_score(group, group[name].to_numpy(), fold["fold"], name, int(step)))
        current.insert(0, "fold", fold["fold"])
        oof_parts.append(current)

    results = pd.DataFrame(records)
    whole = results.loc[results["time_step"].eq("all")]
    pivot = whole.pivot(index="variant", columns="fold", values="roc_auc")
    step24 = results.loc[(results["fold"].eq("fold_1")) & results["time_step"].eq(24)].set_index("variant")["roc_auc"]
    summary = whole.groupby("variant", as_index=False).agg(
        mean_roc_auc=("roc_auc", "mean"),
        weakest_roc_auc=("roc_auc", "min"),
        mean_average_precision=("average_precision", "mean"),
    )
    for fold, label in [("fold_0", "fold_0_roc_auc"), ("fold_1", "middle_fold_roc_auc"), ("fold_2", "latest_fold_roc_auc")]:
        summary[label] = summary["variant"].map(pivot[fold])
    summary["step24_roc_auc"] = summary["variant"].map(step24)
    base = summary.loc[summary["variant"].eq(CONTROL)].iloc[0]
    summary["improved_fold_count"] = summary.apply(
        lambda row: sum(row[column] > base[column] for column in ["fold_0_roc_auc", "middle_fold_roc_auc", "latest_fold_roc_auc"]),
        axis=1,
    )
    summary["max_other_fold_loss"] = summary.apply(
        lambda row: max(base["fold_0_roc_auc"] - row["fold_0_roc_auc"], base["latest_fold_roc_auc"] - row["latest_fold_roc_auc"]),
        axis=1,
    )
    summary["passes_guardrail"] = (
        summary["variant"].ne(CONTROL)
        & summary["mean_roc_auc"].gt(base["mean_roc_auc"])
        & summary["weakest_roc_auc"].gt(base["weakest_roc_auc"])
        & summary["middle_fold_roc_auc"].gt(base["middle_fold_roc_auc"])
        & summary["step24_roc_auc"].gt(base["step24_roc_auc"])
        & summary["max_other_fold_loss"].le(0.001)
    )
    summary = summary.sort_values(["passes_guardrail", "mean_roc_auc", "weakest_roc_auc"], ascending=False).reset_index(drop=True)
    REPORT_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORT_DIR / "historical_neighbor_risk_results.csv", index=False)
    summary.to_csv(REPORT_DIR / "historical_neighbor_risk_summary.csv", index=False)
    pd.concat(oof_parts, ignore_index=True).to_csv(REPORT_DIR / "historical_neighbor_risk_oof_predictions.csv", index=False)
    eligible = summary.loc[summary["passes_guardrail"]]
    candidate = {"candidate_created": False, "candidate_payload_created": False}
    if not eligible.empty:
        selected = str(eligible.iloc[0]["variant"])
        if selected != "blend_historical_neighbor_risk_only_10":
            raise ValueError(f"Unsupported selected final strategy: {selected}")
        seed = labelled.loc[labelled["time_step"].between(1, 28)]
        meta_labelled = labelled.loc[labelled["time_step"].between(29, 35)]
        meta_all = train.loc[train["time_step"].between(29, 35)]
        first_meta = _fit_first_stage(seed, base_columns, 28, half_life)
        meta_risk, final_neighbor_columns = aggregate_neighbor_risk(
            meta_all, edges, first_meta.predict_proba(meta_all[base_columns])[:, 1]
        )
        final_history = _fit_first_stage(labelled, base_columns, 35, half_life)
        test_risk, test_neighbor_columns = aggregate_neighbor_risk(
            test, edges, final_history.predict_proba(test[base_columns])[:, 1]
        )
        if final_neighbor_columns != neighbor_columns or test_neighbor_columns != neighbor_columns:
            raise ValueError("Final neighbour-risk schema differs from temporal evaluation")
        meta = meta_labelled.merge(meta_risk, on=["txId", "time_step"], validate="one_to_one", sort=False)
        test_final = test.merge(test_risk, on=["txId", "time_step"], validate="one_to_one", sort=False)
        specialist = _model()
        specialist.fit(
            meta[neighbor_columns],
            meta["label"].astype(int),
            sample_weight=_weights(meta["time_step"], 35, half_life),
        )
        specialist_probability = specialist.predict_proba(test_final[neighbor_columns])[:, 1]
        incumbent_submission = pd.read_csv(
            PROJECT_ROOT / "submissions" / "candidate_targeted_hgb_20260914.csv"
        )
        probability = 0.90 * incumbent_submission["target"].to_numpy() + 0.10 * specialist_probability
        payload = {
            "method": "90% 0915-1 + 10% leakage-safe historical-model neighbour-risk specialist",
            "selected_variant": selected,
            "uses_current_or_validation_labels_in_neighbor_features": False,
            "records": [
                {"index": index, "target": float(target)}
                for index, target in zip(sample["index"].tolist(), probability)
            ],
        }
        PAYLOAD_PATH.write_text(json.dumps(payload, indent=2))
        candidate = {
            "candidate_created": False,
            "candidate_payload_created": True,
            "selected_variant": selected,
            "candidate_filename": "candidate_historical_neighbor_risk_hgb_20260915.csv",
        }
    metadata = {
        "method": "blocked historical first-stage risk prediction followed by same-period directed neighbour aggregation",
        "direct_observed_neighbor_labels_used": False,
        "current_or_validation_labels_used_in_neighbor_features": False,
        "first_stage_uses_only_earlier_labelled_steps": True,
        "meta_window_steps": META_WINDOW,
        "fold_design": fold_design,
        "neighbor_feature_columns": neighbor_columns,
        "blend_weights": BLEND_WEIGHTS,
        "runtimes_seconds": runtimes,
        "eligible_variants": summary.loc[summary["passes_guardrail"], "variant"].tolist(),
        "kaggle_upload_performed": False,
        **candidate,
    }
    (REPORT_DIR / "historical_neighbor_risk_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps({key: value for key, value in metadata.items() if key not in {"fold_design", "neighbor_feature_columns", "runtimes_seconds"}}, indent=2))


if __name__ == "__main__":
    main()
