"""Test label-observation status as an auxiliary third outcome, not a fraud label."""
from __future__ import annotations

import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.combined_challengers import load_features
from src.competition import PROJECT_ROOT, REPORT_DIR
from src.horizon_hgb_20260917 import BASE, ORIGINS, model, weights
from src.targeted_hgb_challengers import _variants


PREFIX = "label_status_multiclass_20260929"
CACHE = PROJECT_ROOT / "models" / PREFIX
ALIGNMENT = REPORT_DIR / "score_level_alignment_20260929_predictions.pkl"
HORIZON = REPORT_DIR / "horizon_hgb_20260917_predictions.csv"
UNKNOWN_SHARE = .25


def label_status(labels: pd.Series) -> np.ndarray:
    """0=known licit, 1=known illicit, 2=unlabelled observation status."""
    value = labels.to_numpy(dtype=float)
    known = np.isfinite(value)
    if not np.isin(value[known], [0., 1.]).all():
        raise ValueError("Unexpected known fraud label")
    return np.where(known, value, 2).astype(np.int8)


def training_weights(steps: pd.Series, classes: np.ndarray, cutoff: int):
    recency = weights(steps, cutoff, BASE["half_life"])
    known = classes != 2
    known_mass = float(recency[known].sum())
    unknown_mass = float(recency[~known].sum())
    if known_mass <= 0 or unknown_mass <= 0:
        raise ValueError("Both known and unknown histories are required")
    factor = known_mass * UNKNOWN_SHARE / (1 - UNKNOWN_SHARE) / unknown_mass
    output = recency.copy()
    output[~known] *= factor
    if not np.isclose(output[~known].sum() / output.sum(), UNKNOWN_SHARE, atol=1e-12):
        raise ValueError("Unknown status weight share changed")
    return output, factor


def fraud_scores(probabilities: np.ndarray, classes: np.ndarray):
    classes = np.asarray(classes)
    if not np.array_equal(classes, [0, 1, 2]):
        raise ValueError("Unexpected class order")
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("Invalid multiclass probabilities")
    denominator = p[:, 0] + p[:, 1]
    conditional = np.divide(p[:, 1], denominator, out=np.zeros(len(p)), where=denominator > 0)
    return {"fraud_joint": p[:, 1], "fraud_given_labelled": conditional,
            "predicted_labelled": denominator}


def evaluate(predictions: pd.DataFrame):
    variants = ("0929-1", "base_0915_1", "fraud_joint", "fraud_given_labelled")
    records = []
    for origin, frame in predictions.groupby("origin", sort=False):
        slices = {"first7": frame.loc[frame.time_step.le(origin + 7)]}
        if frame.time_step.max() == origin + 14:
            slices.update(full14=frame, second7=frame.loc[frame.time_step.gt(origin + 7)])
        for name, rows in slices.items():
            target = rows.label.to_numpy(dtype=int)
            for variant in variants:
                score = rows[variant].to_numpy(dtype=float)
                records.append({"origin": int(origin), "slice": name, "variant": variant,
                                "n": len(rows), "positives": int(target.sum()),
                                "auc": float(roc_auc_score(target, score)),
                                "average_precision": float(average_precision_score(target, score))})
    result = pd.DataFrame(records)
    pivot = result.pivot(index=["origin", "slice"], columns="variant", values="auc")
    groups = {"full14": [(14, "full14"), (21, "full14")],
              "second7": [(14, "second7"), (21, "second7")],
              "primary7": [(origin, "first7") for origin in (14, 21, 28)],
              "rolling7": [(origin, "first7") for origin in (17, 24)]}
    summary = []
    for variant in variants:
        row = {"variant": variant}
        for name, keys in groups.items():
            row[f"{name}_mean"] = float(pivot.loc[keys, variant].mean())
            row[f"{name}_delta_0929_1"] = float((pivot.loc[keys, variant] - pivot.loc[keys, "0929-1"]).mean())
        row["worst_slice_delta_0929_1"] = float((pivot[variant] - pivot["0929-1"]).min())
        row["qualified"] = bool(variant not in ("0929-1", "base_0915_1")
                                and all(row[f"{name}_delta_0929_1"] > 0 for name in groups)
                                and row["worst_slice_delta_0929_1"] >= -.001)
        summary.append(row)
    return result, pd.DataFrame(summary)


def main():
    plan = {"date": "2026-09-29", "hypothesis": "label-observation status may provide an auxiliary signal about transaction subpopulations without guessing unknown fraud labels",
            "state_mapping": {"0": "known licit", "1": "known illicit", "2": "unlabelled status, not a fraud outcome"},
            "model": BASE, "feature_set": "established 172 columns", "unknown_weight_share": UNKNOWN_SHARE,
            "readouts": ["joint P(known illicit)", "conditional P(illicit | labelled)"],
            "origins": {str(k): v for k, v in ORIGINS.items()},
            "gate": "beat submitted 0929-1 on full14, second7, primary7 and rolling7 mean AUC; worst individual slice loss >= -0.001",
            "no_blend_grid": True, "historical_windows_reused": True,
            "test_labels_used": False, "test_fit": False, "submission_generation": False, "kaggle_upload": False}
    plan_path = REPORT_DIR / f"{PREFIX}_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Frozen plan changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    CACHE.mkdir(parents=True, exist_ok=True)
    train, _, _, groups = load_features()
    columns, _ = _variants(groups)["regularized_drop_top10_drift"]
    if len(columns) != 172:
        raise ValueError("Established feature list changed")
    alignment = pd.read_pickle(ALIGNMENT)
    horizon = pd.read_csv(HORIZON)
    all_parts, diagnostics = [], []
    for origin, stop in ORIGINS.items():
        start = time.perf_counter()
        historical = train.loc[train.time_step.le(origin)]
        forecast = train.loc[train.time_step.between(origin + 1, stop)]
        cache_path = CACHE / f"origin_{origin}.joblib"
        if cache_path.exists():
            all_forecast, detail = joblib.load(cache_path)
        else:
            classes = label_status(historical.label)
            sample_weight, unknown_factor = training_weights(historical.time_step, classes, origin)
            fitted = model(BASE)
            fitted.fit(historical[columns], classes, sample_weight=sample_weight)
            probability = fitted.predict_proba(forecast[columns])
            scores = fraud_scores(probability, fitted.classes_)
            all_forecast = forecast[["txId", "time_step", "label"]].copy()
            for name, value in scores.items():
                all_forecast[name] = value
            knownness_auc = float(roc_auc_score(forecast.label.notna(), scores["predicted_labelled"]))
            detail = {"origin": origin, "forecast_end": stop,
                      "historical_rows": len(historical), "forecast_rows": len(forecast),
                      "known_licit": int((classes == 0).sum()),
                      "known_illicit": int((classes == 1).sum()),
                      "unlabelled": int((classes == 2).sum()),
                      "unknown_weight_factor": unknown_factor,
                      "knownness_auc_on_all_forecast_rows": knownness_auc}
            joblib.dump((all_forecast, detail), cache_path)
        if not all_forecast.txId.equals(forecast.txId) or not all_forecast.time_step.equals(forecast.time_step):
            raise ValueError("Forecast score order changed")
        known = all_forecast.loc[all_forecast.label.notna()].copy()
        anchor = alignment.loc[alignment.origin.eq(origin), ["txId", "align_0.10"]]
        joined = known[["txId"]].merge(anchor, on="txId", validate="one_to_one", sort=False)
        if len(joined) != len(known):
            raise ValueError("0929-1 control population changed")
        known["0929-1"] = joined["align_0.10"].to_numpy()
        base = horizon.loc[horizon.origin.eq(origin), ["txId", "base_0915_1"]]
        joined_base = known[["txId"]].merge(base, on="txId", validate="one_to_one", sort=False)
        if len(joined_base) != len(known):
            raise ValueError("Base control population changed")
        known["base_0915_1"] = joined_base["base_0915_1"].to_numpy()
        known.insert(0, "origin", origin)
        all_parts.append(known)
        diagnostics.append({**detail, "elapsed_seconds": time.perf_counter() - start})
        print(f"Completed origin {origin}: {diagnostics[-1]['elapsed_seconds']:.1f}s", flush=True)
    predictions = pd.concat(all_parts, ignore_index=True)
    results, summary = evaluate(predictions)
    predictions.to_pickle(REPORT_DIR / f"{PREFIX}_predictions.pkl")
    (REPORT_DIR / f"{PREFIX}_results.json").write_text(results.to_json(orient="records", indent=2, double_precision=15) + "\n")
    (REPORT_DIR / f"{PREFIX}_summary.json").write_text(summary.to_json(orient="records", indent=2, double_precision=15) + "\n")
    metadata = {"plan": plan, "diagnostics": diagnostics,
                "alignment_sha256": hashlib.sha256(ALIGNMENT.read_bytes()).hexdigest(),
                "horizon_sha256": hashlib.sha256(HORIZON.read_bytes()).hexdigest(),
                "qualified": bool(summary.qualified.any()),
                "test_predictions_created": False, "submission_created": False}
    (REPORT_DIR / f"{PREFIX}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
