"""Refine the leakage-safe historical neighbour-risk specialist behind 0915-4."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.combined_challengers import load_features
from src.competition import PROJECT_ROOT, RANDOM_SEED, REPORT_DIR, load_competition_data
from src.graph_experiments import FOLDS
from src.historical_neighbor_risk import META_WINDOW, aggregate_neighbor_risk, _fit_first_stage
from src.targeted_hgb_challengers import _model, _variants, _weights


BASE = "base_0915_1"
CONTROL = "incumbent_0915_4"
BLEND_WEIGHTS = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20)
CONTROL_BLEND_WEIGHTS = (0.025, 0.05, 0.075, 0.10)
LOGIT_WEIGHTS = (0.05, 0.10, 0.15)
PAYLOAD_PATH = REPORT_DIR / "refined_historical_neighbor_risk_submission_payload.json"


@dataclass(frozen=True)
class Strategy:
    anchor: str
    specialist: str
    weight: float
    blend: str


def _named_aggregates(frame: pd.DataFrame, node: str, prefix: str) -> pd.DataFrame:
    grouped = frame.groupby(node, sort=False)
    result = grouped.agg(
        count=("neighbor_risk", "count"),
        mean=("neighbor_risk", "mean"),
        max=("neighbor_risk", "max"),
        std=("neighbor_risk", "std"),
        median=("neighbor_risk", "median"),
        rank_mean=("neighbor_rank", "mean"),
        rank_max=("neighbor_rank", "max"),
        high90_share=("high90", "mean"),
        high99_share=("high99", "mean"),
        above_self_share=("above_self", "mean"),
        abs_diff_mean=("abs_diff", "mean"),
    )
    result["q90"] = grouped["neighbor_risk"].quantile(0.90)
    return result.rename(columns=lambda column: f"ref_{prefix}_{column}")


def aggregate_refined_neighbor_risk(
    rows: pd.DataFrame,
    edges: pd.DataFrame,
    probability: np.ndarray,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add robust directed risk-composition features without observed labels."""
    legacy, legacy_columns = aggregate_neighbor_risk(rows, edges, probability)
    node = rows[["txId", "time_step"]].copy()
    node["risk"] = np.asarray(probability, dtype=float)
    node["risk_rank"] = node.groupby("time_step")["risk"].rank(method="average", pct=True)
    risk = node.set_index("txId")["risk"]
    rank = node.set_index("txId")["risk_rank"]

    annotated = edges.copy()
    annotated["source_risk"] = annotated["txId1"].map(risk)
    annotated["target_risk"] = annotated["txId2"].map(risk)
    annotated["source_rank"] = annotated["txId1"].map(rank)
    annotated["target_rank"] = annotated["txId2"].map(rank)

    incoming = annotated.loc[annotated["source_risk"].notna() & annotated["target_risk"].notna()].copy()
    incoming["neighbor_risk"] = incoming["source_risk"]
    incoming["neighbor_rank"] = incoming["source_rank"]
    incoming["self_risk"] = incoming["target_risk"]
    incoming["high90"] = incoming["neighbor_rank"].ge(0.90).astype(float)
    incoming["high99"] = incoming["neighbor_rank"].ge(0.99).astype(float)
    incoming["above_self"] = incoming["neighbor_risk"].gt(incoming["self_risk"]).astype(float)
    incoming["abs_diff"] = (incoming["neighbor_risk"] - incoming["self_risk"]).abs()

    outgoing = annotated.loc[annotated["source_risk"].notna() & annotated["target_risk"].notna()].copy()
    outgoing["neighbor_risk"] = outgoing["target_risk"]
    outgoing["neighbor_rank"] = outgoing["target_rank"]
    outgoing["self_risk"] = outgoing["source_risk"]
    outgoing["high90"] = outgoing["neighbor_rank"].ge(0.90).astype(float)
    outgoing["high99"] = outgoing["neighbor_rank"].ge(0.99).astype(float)
    outgoing["above_self"] = outgoing["neighbor_risk"].gt(outgoing["self_risk"]).astype(float)
    outgoing["abs_diff"] = (outgoing["neighbor_risk"] - outgoing["self_risk"]).abs()

    incoming_features = _named_aggregates(incoming, "txId2", "in")
    outgoing_features = _named_aggregates(outgoing, "txId1", "out")
    combined = pd.concat([
        incoming.rename(columns={"txId2": "node"})[["node", "neighbor_risk", "neighbor_rank", "high90", "high99", "above_self", "abs_diff"]],
        outgoing.rename(columns={"txId1": "node"})[["node", "neighbor_risk", "neighbor_rank", "high90", "high99", "above_self", "abs_diff"]],
    ], ignore_index=True)
    combined_features = _named_aggregates(combined, "node", "all")

    refined = node[["txId", "risk"]].set_index("txId")
    refined = refined.join(incoming_features, how="left").join(outgoing_features, how="left").join(combined_features, how="left")
    for prefix in ("ref_in", "ref_out", "ref_all"):
        count = f"{prefix}_count"
        refined[f"{prefix}_log_count"] = np.log1p(refined[count].fillna(0))
        refined[f"{prefix}_present"] = refined[count].notna().astype(float)
        for statistic in ("mean", "max", "median", "q90"):
            column = f"{prefix}_{statistic}"
            refined[column] = refined[column].fillna(refined["risk"])
        for statistic in ("std", "rank_mean", "rank_max", "high90_share", "high99_share", "above_self_share", "abs_diff_mean"):
            column = f"{prefix}_{statistic}"
            refined[column] = refined[column].fillna(0)
        refined = refined.drop(columns=[count])
    refined["ref_in_out_mean_gap"] = refined["ref_in_mean"] - refined["ref_out_mean"]
    refined["ref_in_out_max_gap"] = refined["ref_in_max"] - refined["ref_out_max"]
    refined["ref_all_mean_minus_self"] = refined["ref_all_mean"] - refined["risk"]
    refined = refined.drop(columns=["risk"]).reset_index()
    enhanced_columns = [column for column in refined if column != "txId"]
    output = legacy.merge(refined, on="txId", validate="one_to_one", sort=False)
    if not np.isfinite(output[legacy_columns + enhanced_columns].to_numpy()).all():
        raise ValueError("Refined neighbour-risk features contain non-finite values")
    return output, legacy_columns, enhanced_columns


def _specialist_model(kind: str) -> HistGradientBoostingClassifier:
    if kind == "default":
        return _model()
    if kind == "shallow":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=7,
            min_samples_leaf=60,
            l2_regularization=4.0,
            random_state=RANDOM_SEED,
        )
    raise KeyError(kind)


def _generate_block(
    labelled: pd.DataFrame,
    all_rows: pd.DataFrame,
    edges: pd.DataFrame,
    base_columns: list[str],
    half_life: float,
    start: int,
    end: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    history = labelled.loc[labelled["time_step"].between(1, start - 1)]
    block_all = all_rows.loc[all_rows["time_step"].between(start, end)]
    block_labelled = labelled.loc[labelled["time_step"].between(start, end)]
    first_stage = _fit_first_stage(history, base_columns, start - 1, half_life)
    features, legacy_columns, enhanced_columns = aggregate_refined_neighbor_risk(
        block_all, edges, first_stage.predict_proba(block_all[base_columns])[:, 1]
    )
    meta = block_labelled.merge(features, on=["txId", "time_step"], validate="one_to_one", sort=False)
    return meta, legacy_columns, enhanced_columns


def _blend(anchor: np.ndarray, specialist: np.ndarray, weight: float, mode: str) -> np.ndarray:
    if mode == "probability":
        return (1.0 - weight) * anchor + weight * specialist
    if mode == "logit":
        epsilon = 1e-6
        return expit((1.0 - weight) * logit(np.clip(anchor, epsilon, 1 - epsilon))
                     + weight * logit(np.clip(specialist, epsilon, 1 - epsilon)))
    raise KeyError(mode)


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


def _strategy_name(anchor: str, specialist: str, weight: float, blend: str) -> str:
    return f"{anchor}__{specialist}__{blend}_{int(round(weight * 1000)):03d}"


def main() -> None:
    train, test, sample, groups = load_features()
    _, _, edges, _ = load_competition_data()
    base_columns, half_life = _variants(groups)["regularized_drop_top10_drift"]
    labelled = train.loc[train["label"].notna()].copy()
    stored = pd.read_csv(REPORT_DIR / "historical_neighbor_risk_oof_predictions.csv")
    records: list[dict] = []
    oof_parts: list[pd.DataFrame] = []
    strategies: dict[str, Strategy] = {}
    runtimes: dict[str, float] = {}

    for fold in FOLDS:
        train_end = fold["train_end"]
        validation = labelled.loc[labelled["time_step"].between(fold["valid_start"], fold["valid_end"])]
        validation_all = train.loc[train["time_step"].between(fold["valid_start"], fold["valid_end"])]
        current = validation[["txId", "time_step", "label"]].merge(
            stored.loc[stored["fold"].eq(fold["fold"]), ["txId", "time_step", "label", "incumbent", "blend_historical_neighbor_risk_only_10"]],
            on=["txId", "time_step", "label"], validate="one_to_one", sort=False,
        ).rename(columns={"incumbent": BASE, "blend_historical_neighbor_risk_only_10": CONTROL})

        started = time.perf_counter()
        meta_parts = []
        for start in range(8, train_end + 1, META_WINDOW):
            end = min(start + META_WINDOW - 1, train_end)
            part, legacy_columns, enhanced_columns = _generate_block(
                labelled, train, edges, base_columns, half_life, start, end
            )
            meta_parts.append(part)
        rolling_meta = pd.concat(meta_parts, ignore_index=True)
        recent_meta = meta_parts[-1]
        validation_first_stage = _fit_first_stage(
            labelled.loc[labelled["time_step"].between(1, train_end)], base_columns, train_end, half_life
        )
        validation_features, validation_legacy, validation_enhanced = aggregate_refined_neighbor_risk(
            validation_all, edges, validation_first_stage.predict_proba(validation_all[base_columns])[:, 1]
        )
        if legacy_columns != validation_legacy or enhanced_columns != validation_enhanced:
            raise ValueError("Refined feature schemas differ between meta and validation blocks")
        valid = validation.merge(validation_features, on=["txId", "time_step"], validate="one_to_one", sort=False)
        runtimes[f"{fold['fold']}:feature_generation"] = time.perf_counter() - started

        relational_columns = [column for column in legacy_columns + enhanced_columns
                              if column not in {"hist_self_risk", "hist_self_risk_rank"}]
        specifications = {
            "legacy_recent": (recent_meta, legacy_columns, "default"),
            "enhanced_recent": (recent_meta, legacy_columns + enhanced_columns, "default"),
            "enhanced_rolling": (rolling_meta, legacy_columns + enhanced_columns, "default"),
            "enhanced_rolling_shallow": (rolling_meta, legacy_columns + enhanced_columns, "shallow"),
            "relational_rolling": (rolling_meta, relational_columns, "default"),
        }
        predictions: dict[str, np.ndarray] = {
            BASE: current[BASE].to_numpy(float),
            CONTROL: current[CONTROL].to_numpy(float),
        }
        for name, (meta, columns, model_kind) in specifications.items():
            started = time.perf_counter()
            model = _specialist_model(model_kind)
            model.fit(meta[columns], meta["label"].astype(int),
                      sample_weight=_weights(meta["time_step"], train_end, half_life))
            predictions[name] = model.predict_proba(valid[columns])[:, 1]
            runtimes[f"{fold['fold']}:{name}"] = time.perf_counter() - started

        for specialist in specifications:
            for weight in BLEND_WEIGHTS:
                name = _strategy_name(BASE, specialist, weight, "probability")
                predictions[name] = _blend(predictions[BASE], predictions[specialist], weight, "probability")
                strategies[name] = Strategy(BASE, specialist, weight, "probability")
            for weight in LOGIT_WEIGHTS:
                name = _strategy_name(BASE, specialist, weight, "logit")
                predictions[name] = _blend(predictions[BASE], predictions[specialist], weight, "logit")
                strategies[name] = Strategy(BASE, specialist, weight, "logit")
        for specialist in ("enhanced_recent", "enhanced_rolling", "enhanced_rolling_shallow", "relational_rolling"):
            for weight in CONTROL_BLEND_WEIGHTS:
                name = _strategy_name(CONTROL, specialist, weight, "probability")
                predictions[name] = _blend(predictions[CONTROL], predictions[specialist], weight, "probability")
                strategies[name] = Strategy(CONTROL, specialist, weight, "probability")

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
        mean_roc_auc=("roc_auc", "mean"), weakest_roc_auc=("roc_auc", "min"),
        mean_average_precision=("average_precision", "mean"),
    )
    for fold, label in [("fold_0", "fold_0_roc_auc"), ("fold_1", "middle_fold_roc_auc"), ("fold_2", "latest_fold_roc_auc")]:
        summary[label] = summary["variant"].map(pivot[fold])
    summary["step24_roc_auc"] = summary["variant"].map(step24)
    control = summary.loc[summary["variant"].eq(CONTROL)].iloc[0]
    summary["max_other_fold_loss"] = summary.apply(
        lambda row: max(control["fold_0_roc_auc"] - row["fold_0_roc_auc"],
                        control["latest_fold_roc_auc"] - row["latest_fold_roc_auc"]), axis=1
    )
    summary["passes_guardrail"] = (
        summary["variant"].ne(CONTROL)
        & summary["mean_roc_auc"].gt(control["mean_roc_auc"])
        & summary["weakest_roc_auc"].gt(control["weakest_roc_auc"])
        & summary["middle_fold_roc_auc"].gt(control["middle_fold_roc_auc"])
        & summary["step24_roc_auc"].gt(control["step24_roc_auc"])
        & summary["max_other_fold_loss"].le(0.001)
    )
    summary = summary.sort_values(["passes_guardrail", "mean_roc_auc", "step24_roc_auc"], ascending=False).reset_index(drop=True)
    REPORT_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORT_DIR / "refined_historical_neighbor_risk_results.csv", index=False)
    summary.to_csv(REPORT_DIR / "refined_historical_neighbor_risk_summary.csv", index=False)
    pd.concat(oof_parts, ignore_index=True).to_csv(REPORT_DIR / "refined_historical_neighbor_risk_oof_predictions.csv", index=False)

    eligible = summary.loc[summary["passes_guardrail"] & summary["variant"].isin(strategies)]
    candidate = {"candidate_payload_created": False}
    if not eligible.empty:
        selected = str(eligible.iloc[0]["variant"])
        strategy = strategies[selected]
        meta_parts = []
        for start in range(8, 36, META_WINDOW):
            part, legacy_columns, enhanced_columns = _generate_block(
                labelled, train, edges, base_columns, half_life, start, min(start + META_WINDOW - 1, 35)
            )
            meta_parts.append(part)
        rolling_meta = pd.concat(meta_parts, ignore_index=True)
        recent_meta = meta_parts[-1]
        final_history = _fit_first_stage(labelled, base_columns, 35, half_life)
        test_features, test_legacy, test_enhanced = aggregate_refined_neighbor_risk(
            test, edges, final_history.predict_proba(test[base_columns])[:, 1]
        )
        relational_columns = [column for column in test_legacy + test_enhanced
                              if column not in {"hist_self_risk", "hist_self_risk_rank"}]
        final_specs = {
            "legacy_recent": (recent_meta, test_legacy, "default"),
            "enhanced_recent": (recent_meta, test_legacy + test_enhanced, "default"),
            "enhanced_rolling": (rolling_meta, test_legacy + test_enhanced, "default"),
            "enhanced_rolling_shallow": (rolling_meta, test_legacy + test_enhanced, "shallow"),
            "relational_rolling": (rolling_meta, relational_columns, "default"),
        }
        meta, columns, model_kind = final_specs[strategy.specialist]
        specialist = _specialist_model(model_kind)
        specialist.fit(meta[columns], meta["label"].astype(int),
                       sample_weight=_weights(meta["time_step"], 35, half_life))
        specialist_probability = specialist.predict_proba(test_features[columns])[:, 1]
        anchor_file = ("candidate_targeted_hgb_20260914.csv" if strategy.anchor == BASE
                       else "candidate_historical_neighbor_risk_hgb_20260915.csv")
        anchor_probability = pd.read_csv(PROJECT_ROOT / "submissions" / anchor_file)["target"].to_numpy(float)
        probability = _blend(anchor_probability, specialist_probability, strategy.weight, strategy.blend)
        payload = {
            "method": f"{strategy.anchor} plus {strategy.weight:.3f} {strategy.specialist} using {strategy.blend} blend",
            "selected_variant": selected,
            "uses_current_or_validation_labels_in_neighbor_features": False,
            "records": [{"index": index, "target": float(target)}
                        for index, target in zip(sample["index"].tolist(), probability)],
        }
        PAYLOAD_PATH.write_text(json.dumps(payload, indent=2))
        candidate = {"candidate_payload_created": True, "selected_variant": selected,
                     "candidate_filename": "candidate_refined_historical_neighbor_risk_20260915.csv"}

    metadata = {
        "control": CONTROL,
        "control_user_reported_public_roc_auc": 0.94954,
        "direct_observed_neighbor_labels_used": False,
        "first_stage_uses_only_earlier_labelled_steps": True,
        "rolling_meta_blocks": True,
        "legacy_feature_count": len(legacy_columns),
        "enhanced_feature_count": len(enhanced_columns),
        "strategies_tested": len(strategies),
        "eligible_variants": eligible["variant"].tolist(),
        "runtimes_seconds": runtimes,
        "kaggle_upload_performed": False,
        **candidate,
    }
    (REPORT_DIR / "refined_historical_neighbor_risk_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.head(20).to_string(index=False))
    print(json.dumps({key: value for key, value in metadata.items() if key != "runtimes_seconds"}, indent=2))


if __name__ == "__main__":
    main()
