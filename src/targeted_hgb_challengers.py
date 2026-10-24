"""Targeted, one-change-at-a-time follow-ups to submitted HGB Model 06.

Run from project root: .venv/bin/python -m src.targeted_hgb_challengers
No Kaggle upload is performed. New files do not overwrite previous candidates.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.combined_challengers import load_features
from src.competition import PROJECT_ROOT, RANDOM_SEED, REPORT_DIR
from src.graph_experiments import FOLDS
from src.model_comparison import MODEL_CONFIGURATIONS


CONTROL = "regularized_two_hop"
HALF_LIFE = 20.0
CANDIDATE_FILENAME = "candidate_targeted_hgb_20260914.csv"


def _model() -> HistGradientBoostingClassifier:
    config = MODEL_CONFIGURATIONS["Histogram gradient boosting"].copy()
    config.update(max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=2.0)
    return HistGradientBoostingClassifier(**config, random_state=RANDOM_SEED)


def _weights(steps: pd.Series, reference: int, half_life: float) -> np.ndarray:
    return np.power(0.5, (reference - steps.to_numpy(dtype=float)) / half_life)


def _variants(groups: dict[str, list[str]]) -> dict[str, tuple[list[str], float]]:
    base = groups[CONTROL]
    ranking = pd.read_csv(REPORT_DIR / "time_relative_early_drift_ranking.csv")
    if not ranking["early_window_only"].all() or len(ranking) != 165:
        raise ValueError("Early-only, label-free drift ranking is missing")
    ranked = ranking["feature"].tolist()
    rank20 = [f"time_rank_{name}" for name in ranked[:20]]
    rank40 = [f"time_rank_{name}" for name in ranked[:40]]
    drop10 = set(ranked[:10])
    variants = {
        "regularized_plus_rank20": (base + rank20, HALF_LIFE),
        "regularized_plus_rank40": (base + rank40, HALF_LIFE),
        "regularized_drop_top10_drift": ([name for name in base if name not in drop10], HALF_LIFE),
        "regularized_half_life12": (base, 12.0),
    }
    expected = [202, 222, 172, 182]
    if [len(columns) for columns, _ in variants.values()] != expected:
        raise ValueError("Targeted feature counts differ from the fixed design")
    if any(len(columns) != len(set(columns)) for columns, _ in variants.values()):
        raise ValueError("Targeted feature lists contain duplicates")
    return variants


def _stored_control() -> pd.DataFrame:
    stored = pd.read_csv(REPORT_DIR / "combined_challenger_oof_predictions.csv")
    stored = stored[["fold", "txId", "time_step", "label", CONTROL]]
    if stored.duplicated(["fold", "txId"]).any():
        raise ValueError("Duplicated saved control predictions")
    expected = pd.read_csv(REPORT_DIR / "combined_challenger_results.csv")
    expected = expected.loc[expected["variant"] == CONTROL].set_index("fold")["roc_auc"]
    actual = stored.groupby("fold").apply(lambda g: roc_auc_score(g["label"], g[CONTROL]))
    if not expected.index.equals(actual.index) or not np.allclose(expected, actual, rtol=0, atol=1e-10):
        raise ValueError("Saved Model 06 predictions do not reproduce its recorded fold scores")
    return stored


def evaluate(train: pd.DataFrame, variants: dict[str, tuple[list[str], float]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labelled = train.loc[train["label"].notna()].copy()
    rows: list[dict] = []
    oof_parts: list[pd.DataFrame] = []
    for fold in FOLDS:
        development = labelled.loc[labelled["time_step"].between(fold["train_start"], fold["train_end"])]
        validation = labelled.loc[labelled["time_step"].between(fold["valid_start"], fold["valid_end"])]
        truth = validation["label"].astype(int).to_numpy()
        current = pd.DataFrame({"fold": fold["fold"], "txId": validation["txId"].to_numpy(),
                                "time_step": validation["time_step"].to_numpy(), "label": truth})
        for name, (columns, half_life) in variants.items():
            started = time.perf_counter()
            model = _model()
            model.fit(development[columns], development["label"].astype(int),
                      sample_weight=_weights(development["time_step"], fold["train_end"], half_life))
            probability = model.predict_proba(validation[columns])[:, 1]
            current[name] = probability
            rows.append({**fold, "variant": name, "feature_count": len(columns),
                         "half_life": half_life, "roc_auc": roc_auc_score(truth, probability),
                         "average_precision": average_precision_score(truth, probability),
                         "runtime_seconds": time.perf_counter() - started})
        oof_parts.append(current)
    oof = pd.concat(oof_parts, ignore_index=True).merge(
        _stored_control(), on=["fold", "txId", "time_step", "label"], validate="one_to_one", sort=False)
    for fold, group in oof.groupby("fold", sort=False):
        truth = group["label"].to_numpy(dtype=int)
        baseline = group[CONTROL].to_numpy(dtype=float)
        rows.append({"fold": fold, "variant": CONTROL, "feature_count": 182,
                     "half_life": HALF_LIFE, "roc_auc": roc_auc_score(truth, baseline),
                     "average_precision": average_precision_score(truth, baseline),
                     "runtime_seconds": 0.0})
    results = pd.DataFrame(rows)
    summary = results.groupby("variant", as_index=False).agg(
        mean_roc_auc=("roc_auc", "mean"), weakest_roc_auc=("roc_auc", "min"),
        mean_average_precision=("average_precision", "mean"),
        feature_count=("feature_count", "first"), half_life=("half_life", "first"))
    latest = results.loc[results["fold"] == "fold_2"].set_index("variant")["roc_auc"]
    summary["latest_roc_auc"] = summary["variant"].map(latest)
    control_fold = results.loc[results["variant"] == CONTROL].set_index("fold")["roc_auc"]
    summary["improved_fold_count"] = [int((results.loc[results["variant"] == name].set_index("fold")["roc_auc"]
                                           > control_fold).sum()) for name in summary["variant"]]
    control = summary.loc[summary["variant"] == CONTROL].iloc[0]
    summary["passes_guardrail"] = (summary["variant"].ne(CONTROL)
                                    & summary["mean_roc_auc"].gt(control["mean_roc_auc"])
                                    & summary["weakest_roc_auc"].gt(control["weakest_roc_auc"])
                                    & summary["improved_fold_count"].ge(2))
    summary = summary.sort_values(["mean_roc_auc", "weakest_roc_auc"], ascending=False).reset_index(drop=True)
    return results, summary, oof


def _check_submission(path, sample: pd.DataFrame) -> dict:
    submission = pd.read_csv(path)
    probability = submission["target"].to_numpy(dtype=float)
    check = {"file": path.name, "rows": len(submission),
             "columns_match_sample": submission.columns.tolist() == sample.columns.tolist(),
             "order_matches_sample": bool(submission["index"].equals(sample["index"])),
             "all_finite": bool(np.isfinite(probability).all()),
             "in_unit_interval": bool(((probability >= 0) & (probability <= 1)).all()),
             "unique_probabilities": int(pd.Series(probability).nunique()),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    check["ready"] = bool(check["rows"] == len(sample) and check["columns_match_sample"]
                          and check["order_matches_sample"] and check["all_finite"]
                          and check["in_unit_interval"] and check["unique_probabilities"] > 1)
    if not check["ready"]:
        raise ValueError("Targeted local candidate failed submission checks")
    return check


def fit_if_supported(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame,
                     variants: dict[str, tuple[list[str], float]], summary: pd.DataFrame) -> dict:
    eligible = summary.loc[summary["passes_guardrail"]]
    if eligible.empty:
        return {"candidate_created": False, "reason": "No variant improved mean, weakest fold and at least two folds"}
    selected = str(eligible.iloc[0]["variant"])
    columns, half_life = variants[selected]
    labelled = train.loc[train["label"].notna()]
    model = _model()
    model.fit(labelled[columns], labelled["label"].astype(int),
              sample_weight=_weights(labelled["time_step"], 35, half_life))
    probability = model.predict_proba(test[columns])[:, 1]
    submission = sample.copy()
    submission["target"] = probability
    path = PROJECT_ROOT / "submissions" / CANDIDATE_FILENAME
    if path.exists():
        existing = pd.read_csv(path)
        if (existing.columns.tolist() != submission.columns.tolist()
            or not existing["index"].equals(submission["index"])
            or not np.allclose(existing["target"], probability, rtol=0, atol=1e-15)):
            raise FileExistsError(f"Existing local candidate differs: {path}")
    else:
        submission.to_csv(path, index=False)
    return {"candidate_created": True, "selected_variant": selected,
            "uses_time_ranks": selected.startswith("regularized_plus_rank"),
            "half_life": half_life, "validation": _check_submission(path, sample)}


def main() -> None:
    train, test, sample, groups = load_features()
    variants = _variants(groups)
    results, summary, oof = evaluate(train, variants)
    candidate = fit_if_supported(train, test, sample, variants, summary)
    REPORT_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORT_DIR / "targeted_hgb_results.csv", index=False)
    summary.to_csv(REPORT_DIR / "targeted_hgb_summary.csv", index=False)
    oof.to_csv(REPORT_DIR / "targeted_hgb_oof_predictions.csv", index=False)
    metadata = {"uses_test_labels": False, "uses_external_data": False,
                "uses_full_supplied_graph": True,
                "rank_variants_use_unlabelled_test_values": True,
                "early_drift_selection": "unsupervised and limited to time steps 1-14",
                "kaggle_upload_performed": False, **candidate}
    (REPORT_DIR / "targeted_hgb_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(candidate, indent=2))


if __name__ == "__main__":
    main()
