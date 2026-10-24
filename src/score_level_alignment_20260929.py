"""Test label-free period score-level alignment on frozen temporal forecasts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

from src.combined_challengers import load_features
from src.competition import PROJECT_ROOT, REPORT_DIR, load_competition_data
from src.horizon_hgb_20260917 import BASE, ORIGINS, predict_stack
from src.targeted_hgb_challengers import _variants


PREFIX = "score_level_alignment_20260929"
CACHE = PROJECT_ROOT / "models" / PREFIX
CONTROL_PATH = REPORT_DIR / "horizon_hgb_20260917_predictions.csv"
ALPHAS = (0.10, 0.25, 0.50)


def align(scores: np.ndarray, periods: np.ndarray, alpha: float):
    """Shrink each period's log-odds median toward the median forecast period."""
    scores = np.asarray(scores, dtype=float)
    periods = np.asarray(periods, dtype=int)
    if len(scores) != len(periods) or not np.isfinite(scores).all():
        raise ValueError("Invalid score array")
    if np.any((scores < 0) | (scores > 1)) or not 0 <= alpha <= 1:
        raise ValueError("Invalid probabilities or correction strength")
    odds = logit(np.clip(scores, 1e-8, 1 - 1e-8))
    unique, codes = np.unique(periods, return_inverse=True)
    medians = np.array([np.median(odds[codes == i]) for i in range(len(unique))])
    reference = float(np.median(medians))
    aligned = expit(odds - alpha * (medians[codes] - reference))
    return aligned, {str(int(step)): float(median) for step, median in zip(unique, medians)}, reference


def slices(frame: pd.DataFrame, origin: int):
    output = {"first7": frame.loc[frame.time_step.le(origin + 7)]}
    if frame.time_step.max() == origin + 14:
        output["full14"] = frame
        output["second7"] = frame.loc[frame.time_step.gt(origin + 7)]
    return output


def evaluate(predictions: pd.DataFrame):
    variants = ["0915-4", "0916-2"] + [f"align_{a:.2f}" for a in ALPHAS]
    records = []
    for origin, whole in predictions.groupby("origin", sort=False):
        for slice_name, subset in slices(whole, int(origin)).items():
            truth = subset.label.to_numpy(dtype=int)
            for variant in variants:
                score = subset[variant].to_numpy(dtype=float)
                records.append({"origin": int(origin), "slice": slice_name,
                                "variant": variant, "n": len(subset),
                                "positives": int(truth.sum()),
                                "auc": float(roc_auc_score(truth, score)),
                                "average_precision": float(average_precision_score(truth, score))})
    result = pd.DataFrame(records)
    pivot = result.pivot(index=["origin", "slice"], columns="variant", values="auc")
    groups = {"full14": [(14, "full14"), (21, "full14")],
              "second7": [(14, "second7"), (21, "second7")],
              "primary7": [(origin, "first7") for origin in (14, 21, 28)],
              "rolling7": [(origin, "first7") for origin in (17, 24)]}
    summaries = []
    for variant in variants:
        row = {"variant": variant}
        for group, keys in groups.items():
            row[f"{group}_mean"] = float(pivot.loc[keys, variant].mean())
            row[f"{group}_delta_0916_2"] = float((pivot.loc[keys, variant]
                                                  - pivot.loc[keys, "0916-2"]).mean())
        delta = pivot[variant] - pivot["0916-2"]
        row["worst_slice_delta_0916_2"] = float(delta.min())
        row["qualified"] = bool(variant not in ("0915-4", "0916-2")
                                and all(row[f"{name}_mean"] > float(pivot.loc[keys, "0916-2"].mean())
                                        for name, keys in groups.items())
                                and delta.min() >= -0.001)
        summaries.append(row)
    return result, pd.DataFrame(summaries)


def main():
    plan = {
        "hypothesis": "period-specific score level drift harms cross-period ranking",
        "date": "2026-09-29", "controls": ["0915-4", "0916-2"],
        "origins": {str(k): v for k, v in ORIGINS.items()}, "alphas": list(ALPHAS),
        "method": "subtract alpha times each forecast period's all-row median logit minus the equal-period median of those medians",
        "score_population": "all supplied rows in each forecast period, including unknown labels",
        "test_labels_used": False, "reused_historical_windows": True,
        "gate": "beat 0916-2 on mean full14, second7, primary7, rolling7; no slice loss greater than 0.001",
        "test_fit": False, "submission_generation": False, "kaggle_upload": False,
    }
    plan_path = REPORT_DIR / f"{PREFIX}_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Frozen plan changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    CACHE.mkdir(parents=True, exist_ok=True)
    train, test, sample, groups = load_features()
    _, _, edges, _ = load_competition_data()
    columns, _ = _variants(groups)["regularized_drop_top10_drift"]
    full_columns = groups["regularized_two_hop"]
    saved = pd.read_csv(CONTROL_PATH)
    parts, diagnostics = [], {}
    for origin, stop in ORIGINS.items():
        cache_path = CACHE / f"origin_{origin}.joblib"
        future = train.loc[train.time_step.between(origin + 1, stop)].copy()
        if cache_path.exists():
            all_rows = joblib.load(cache_path)
        else:
            print(f"Predicting all rows for origin {origin}, periods {origin + 1}-{stop}", flush=True)
            stack, _ = predict_stack(train, future, edges, full_columns, columns, origin, BASE)
            all_rows = future[["txId", "time_step", "label"]].copy()
            all_rows["0916-2"] = stack["bagged"]
            joblib.dump(all_rows, cache_path)
        if not all_rows.txId.equals(future.txId) or not all_rows.time_step.equals(future.time_step):
            raise ValueError("Cached row order differs from source")
        known = all_rows.loc[all_rows.label.notna(), ["txId", "time_step", "label", "0916-2"]]
        control = saved.loc[saved.origin.eq(origin), ["txId", "time_step", "label", "0915-4", "0916-2"]]
        joined = known.merge(control, on=["txId", "time_step", "label"], validate="one_to_one",
                             suffixes=("_recomputed", "_saved"), sort=False)
        if len(joined) != len(control) or not np.allclose(joined["0916-2_recomputed"],
                                                          joined["0916-2_saved"], atol=1e-10, rtol=0):
            raise ValueError(f"Frozen control mismatch at origin {origin}")
        current = control.copy()
        current.insert(0, "origin", origin)
        diag = {"all_rows": len(all_rows), "labelled_rows": len(current), "periods": {}}
        for alpha in ALPHAS:
            adjusted, medians, reference = align(all_rows["0916-2"], all_rows.time_step, alpha)
            correction = all_rows[["txId"]].copy()
            correction[f"align_{alpha:.2f}"] = adjusted
            current = current.merge(correction, on="txId", validate="one_to_one", sort=False)
            diag["periods"] = medians
            diag["reference_logit_median"] = reference
        for period, group in current.groupby("time_step"):
            base = group["0916-2"].to_numpy()
            for alpha in ALPHAS:
                candidate = group[f"align_{alpha:.2f}"].to_numpy()
                if not np.array_equal(np.argsort(base, kind="stable"),
                                      np.argsort(candidate, kind="stable")):
                    raise ValueError("Within-period ranking changed")
        parts.append(current)
        diagnostics[str(origin)] = diag
        print(f"Completed origin {origin}", flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    if len(predictions) != len(saved):
        raise ValueError("Validation row count changed")
    results, summary = evaluate(predictions)
    predictions.to_pickle(REPORT_DIR / f"{PREFIX}_predictions.pkl")
    (REPORT_DIR / f"{PREFIX}_results.json").write_text(
        results.to_json(orient="records", indent=2, double_precision=15) + "\n")
    (REPORT_DIR / f"{PREFIX}_summary.json").write_text(
        summary.to_json(orient="records", indent=2, double_precision=15) + "\n")
    metadata = {"plan": plan, "control_sha256": hashlib.sha256(CONTROL_PATH.read_bytes()).hexdigest(),
                "diagnostics": diagnostics, "candidate_qualified": bool(summary.qualified.any()),
                "test_predictions_created": False, "submission_created": False, "kaggle_upload": False}
    eligible = summary.loc[summary.qualified, "variant"].tolist()
    if eligible:
        chosen = min(eligible, key=lambda value: float(value.split("_")[1]))
        alpha = float(chosen.split("_")[1])
        anchor_path = PROJECT_ROOT / "submissions" / "candidate_exploratory_bagged_neighbor_risk_20260916.csv"
        anchor = pd.read_csv(anchor_path)
        if not sample["index"].equals(test["index"]) or not sample["index"].equals(anchor["index"]):
            raise ValueError("Actual-test index order differs from sample or anchor")
        if len(test) != len(anchor) or sorted(test.time_step.unique().tolist()) != list(range(36, 50)):
            raise ValueError("Unexpected test period population")
        corrected, medians, reference = align(anchor.target, test.time_step, alpha)
        if not np.isfinite(corrected).all() or np.any((corrected < 0) | (corrected > 1)):
            raise ValueError("Invalid test probabilities")
        payload = {"filename": "candidate_score_level_alignment_20260929.csv",
                   "anchor": anchor_path.name, "anchor_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
                   "alpha": alpha, "method": plan["method"],
                   "test_period_logit_medians": medians, "test_reference_logit_median": reference,
                   "records": [{"index": int(index), "target": float(score)}
                               for index, score in zip(sample["index"], corrected)]}
        (REPORT_DIR / f"{PREFIX}_submission_payload.json").write_text(json.dumps(payload) + "\n")
        metadata["candidate_filename"] = payload["filename"]
        metadata["candidate_alpha"] = alpha
        metadata["test_predictions_created"] = True
        metadata["test_anchor_sha256"] = payload["anchor_sha256"]
    (REPORT_DIR / f"{PREFIX}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
