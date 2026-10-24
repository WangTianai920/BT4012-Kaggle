"""Frozen Trial 39: fixed three-model log-odds blend of incumbent, Trial 20, and Trial 34."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.combined_challengers import load_features
from src.competition import DATA_DIR, PROJECT_ROOT, REPORT_DIR
from src.horizon_hgb_20260917 import BASE, historical_columns, model
from src.label_status_multiclass_20260929 import label_status, training_weights
from src.shrinkage_logodds_blend_20261002 import (
    VARIANT as GATE_VARIANT,
    evaluate as frozen_evaluate,
    summarize as frozen_summarize,
)

PREFIX = "dual_steps_logodds_20261002"
PLAN_PATH = REPORT_DIR / f"{PREFIX}_plan.json"
TRIAL20_META = REPORT_DIR / "shrinkage_status_20261001_metadata.json"
TRIAL34_META = REPORT_DIR / "intermediate_steps_status_20261002_metadata.json"
T20_MODEL = PROJECT_ROOT / "models/shrinkage_status_20261001/final_lr03_iter600_through_35.joblib"
T20_COMPONENTS = REPORT_DIR / "shrinkage_status_20261001_final_components.pkl"
T20_CANDIDATE = PROJECT_ROOT / "submissions/candidate_shrinkage_status_20261001_lr03_iter600.csv"
INCUMBENT_CANDIDATE = PROJECT_ROOT / "submissions/candidate_label_status_multiclass_20260929.csv"
ORIGINS = {14: 28, 21: 35, 17: 24, 24: 31, 28: 35}
CONFIRMATION = {17: 31, 18: 32, 20: 34}
VARIANT = "dual_steps_logodds_75_12_5_12_5"
CLIP = 1e-12
KEY = ["origin", "txId", "time_step", "label"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config():
    return dict(BASE, learning_rate=.04, max_iter=450)


def score_logodds(incumbent, trial20, trial34):
    arrays = [np.asarray(value, dtype=float) for value in (incumbent, trial20, trial34)]
    if (any(value.ndim != 1 for value in arrays) or len({value.shape for value in arrays}) != 1
            or any(not np.isfinite(value).all() or ((value < 0) | (value > 1)).any()
                   for value in arrays)):
        raise ValueError("Three component inputs must be aligned finite probabilities")
    return expit(.75 * logit(np.clip(arrays[0], CLIP, 1 - CLIP))
                 + .125 * logit(np.clip(arrays[1], CLIP, 1 - CLIP))
                 + .125 * logit(np.clip(arrays[2], CLIP, 1 - CLIP)))


def conditional_score(fitted, frame, columns):
    proba = np.asarray(fitted.predict_proba(frame[columns]), dtype=float)
    classes = np.asarray(fitted.classes_, dtype=int)
    if not np.array_equal(classes, [0, 1, 2]) or proba.shape != (len(frame), 3):
        raise ValueError("Unexpected status-class order or prediction shape")
    denom = proba[:, 0] + proba[:, 1]
    score = np.divide(proba[:, 1], denom, out=np.zeros(len(frame)), where=denom > 0)
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Invalid conditional fraud score")
    return score


def verify_inputs(plan):
    if plan.get("trial_number") != 39 or plan.get("variant") != VARIANT:
        raise ValueError("Frozen Trial 39 plan missing or mismatched")
    if plan.get("weights") != {"incumbent": .75, "trial20_lr03_iter600": .125,
                                "trial34_lr04_iter450": .125}:
        raise ValueError("Frozen three-model weights changed")
    if plan.get("clip_epsilon") != CLIP:
        raise ValueError("Frozen log-odds clipping value changed")
    for name, digest in plan["input_sha256"].items():
        if sha(PROJECT_ROOT / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    prior = json.loads(TRIAL20_META.read_text())
    if (prior.get("winner") != "lr03_iter600" or prior.get("final_iterations") != 600
            or not prior.get("original_gate_passed") or not prior.get("confirmation_run")
            or prior.get("kaggle_upload") is not False):
        raise ValueError("Trial 20 provenance or eligibility changed")
    t34 = json.loads(TRIAL34_META.read_text())
    if (not t34.get("original_gate_passed") or not t34.get("confirmation_run")
            or t34.get("promoted_variants") or t34.get("submission_created")
            or t34.get("kaggle_upload") is not False):
        raise ValueError("Trial 34 saved forecast provenance changed")
    t38 = json.loads((REPORT_DIR / "dual_steps_blend_20261002_metadata.json").read_text())
    if (not t38.get("original_gate_passed") or not t38.get("confirmation_run")
            or t38.get("submission_created") or t38.get("kaggle_upload") is not False):
        raise ValueError("Trial 38 prior ensemble record changed")
    t38_summary = pd.read_csv(REPORT_DIR / "dual_steps_blend_20261002_summary.csv")
    if bool(t38_summary.loc[t38_summary.stage.eq("confirmation"), "qualified"].iloc[0]):
        raise ValueError("Trial 38 confirmation result changed")
    if prior.get("final_model_sha256") != plan["input_sha256"][
            "models/shrinkage_status_20261001/final_lr03_iter600_through_35.joblib"]:
        raise ValueError("Trial 20 final model hash disagrees with its metadata")
    if prior.get("final_components_sha256") != plan["input_sha256"][
            "reports/shrinkage_status_20261001_final_components.pkl"]:
        raise ValueError("Trial 20 final components hash disagrees with its metadata")
    if prior.get("candidate_sha256") != plan["input_sha256"][
            "submissions/candidate_shrinkage_status_20261001_lr03_iter600.csv"]:
        raise ValueError("Trial 20 candidate hash disagrees with its metadata")


def load_stage(stage, origins):
    prior_path = REPORT_DIR / f"shrinkage_status_20261001_{stage}_predictions.pkl"
    later_path = REPORT_DIR / f"intermediate_steps_status_20261002_{stage}_predictions.pkl"
    control_name = ("label_status_multiclass_20260929_predictions.pkl" if stage == "original"
                    else "label_status_confirmation_20260929_predictions.pkl")
    prior, later = pd.read_pickle(prior_path), pd.read_pickle(later_path)
    control = pd.read_pickle(REPORT_DIR / control_name)
    need_prior = set(KEY + ["incumbent", "lr03_iter600"])
    need_later = set(KEY + ["incumbent", "lr04_iter450"])
    if (not need_prior.issubset(prior.columns) or not need_later.issubset(later.columns)
            or not set(KEY + ["fraud_given_labelled"]).issubset(control.columns)):
        raise ValueError(f"{stage} saved forecast schema changed")
    for name, frame in (("Trial20", prior), ("Trial34", later), ("incumbent", control)):
        if frame.duplicated(KEY).any():
            raise ValueError(f"{stage} {name} keys are not unique")
    if set(prior.origin.unique()) != set(origins) or set(later.origin.unique()) != set(origins):
        raise ValueError(f"{stage} forecast origins changed")

    merged = prior[KEY + ["incumbent", "lr03_iter600"]].merge(
        later[KEY + ["incumbent", "lr04_iter450"]], on=KEY, how="outer",
        validate="one_to_one", sort=False, suffixes=("_t20", "_t34"), indicator=True)
    if len(merged) != len(prior) or not merged["_merge"].eq("both").all():
        raise ValueError(f"{stage} Trial20 and Trial34 forecast keys differ")
    exact_later = merged[KEY].merge(later[KEY], on=KEY, how="left", validate="one_to_one", sort=False)
    if len(exact_later) != len(merged):
        raise ValueError(f"{stage} challenger forecasts are not aligned")
    for control_values in (merged.incumbent_t20, merged.incumbent_t34):
        if not np.array_equal(control_values.to_numpy(), merged.incumbent_t20.to_numpy()):
            raise ValueError(f"{stage} Trial20 and Trial34 incumbent controls differ")
    control_aligned = merged[KEY].merge(control[KEY + ["fraud_given_labelled"]],
                                        on=KEY, how="left", validate="one_to_one", sort=False)
    if (len(control_aligned) != len(merged) or control_aligned.fraud_given_labelled.isna().any()
            or not np.array_equal(control_aligned.fraud_given_labelled.to_numpy(),
                                  merged.incumbent_t20.to_numpy())):
        raise ValueError(f"{stage} saved incumbent control differs from exact reference")
    for origin, stop in origins.items():
        part = merged.loc[merged.origin.eq(origin)]
        if part.empty or not part.time_step.between(origin + 1, stop).all() or int(part.time_step.max()) != stop:
            raise ValueError(f"{stage} forecast horizon changed at {origin}")
    out = merged[KEY + ["incumbent_t20", "lr03_iter600", "lr04_iter450"]].copy()
    out = out.rename(columns={"incumbent_t20": "incumbent"})
    out[VARIANT] = score_logodds(out.incumbent, out.lr03_iter600, out.lr04_iter450)
    values = out[["incumbent", "lr03_iter600", "lr04_iter450", VARIANT]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{stage} aligned scores contain invalid probabilities")
    return out


def evaluate(predictions, stage):
    aliased = predictions.rename(columns={VARIANT: GATE_VARIANT})
    results = frozen_evaluate(aliased, stage)
    results.loc[results.variant.eq(GATE_VARIANT), "variant"] = VARIANT
    return results


def summarize(results, stage):
    aliased = results.copy()
    aliased.loc[aliased.variant.eq(VARIANT), "variant"] = GATE_VARIANT
    summary = frozen_summarize(aliased, stage)
    summary.loc[:, "variant"] = VARIANT
    return summary


def export_candidate(plan, train, test_features, sample, full_columns):
    expected = sample["index"].to_numpy()
    if len(expected) != 15329 or not np.array_equal(test_features["index"].to_numpy(), expected):
        raise ValueError("Engineered test feature order changed")
    incumbent_file = pd.read_csv(INCUMBENT_CANDIDATE)
    t20_file = pd.read_csv(T20_CANDIDATE)
    t20_components = pd.read_pickle(T20_COMPONENTS)
    candidate_path = PROJECT_ROOT / plan["candidate_output"]
    component_path = REPORT_DIR / f"{PREFIX}_final_components.pkl"
    model_path = PROJECT_ROOT / "models" / PREFIX / "final_lr04_iter450_through_35.joblib"
    if candidate_path.exists() or component_path.exists() or model_path.exists():
        raise ValueError("Trial39 final artifacts already exist; preserve them")
    if (not np.array_equal(incumbent_file["index"].to_numpy(), expected)
            or not np.array_equal(t20_file["index"].to_numpy(), expected)
            or not np.array_equal(t20_components["index"].to_numpy(), expected)):
        raise ValueError("Saved incumbent or Trial20 candidate order changed")
    if (not np.allclose(t20_components.incumbent.to_numpy(), incumbent_file.target.to_numpy(), rtol=0, atol=1e-15)
            or not np.allclose(t20_components.challenger.to_numpy(), t20_file.target.to_numpy(), rtol=0, atol=1e-15)):
        raise ValueError("Saved Trial20 final components differ from their submission files")

    prior_model = joblib.load(T20_MODEL)
    columns = historical_columns(train, full_columns, 35)
    if prior_model["columns"] != columns or prior_model["model"].n_iter_ != 600:
        raise ValueError("Saved Trial20 model configuration or features changed")
    replay_t20 = conditional_score(prior_model["model"], test_features, columns)
    if not np.allclose(replay_t20, t20_file.target.to_numpy(), rtol=0, atol=1e-12):
        raise ValueError("Replayed Trial20 model scores differ from its saved candidate")

    history = train.loc[train.time_step.le(35)]
    status = label_status(history.label)
    weights, factor = training_weights(history.time_step, status, 35)
    share = float(weights[status == 2].sum() / weights.sum())
    if not np.isclose(share, .25, atol=1e-12) or not np.array_equal(np.unique(status), [0, 1, 2]):
        raise ValueError("Final unknown-status weighting or classes changed")
    fitted = model(config())
    fitted.fit(history[columns], status, sample_weight=weights)
    if fitted.n_iter_ != 450:
        raise ValueError("Final Trial34 component did not fit all 450 iterations")
    t34_score = conditional_score(fitted, test_features, columns)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted, "columns": columns, "plan_sha256": sha(PLAN_PATH),
                 "config": config(), "classes": [0, 1, 2]}, model_path)
    restored = joblib.load(model_path)
    replay_t34 = conditional_score(restored["model"], test_features, columns)
    if restored["columns"] != columns or not np.allclose(replay_t34, t34_score, rtol=0, atol=1e-12):
        raise ValueError("Saved final Trial34 component failed score replay")

    incumbent = incumbent_file.target.to_numpy(dtype=float)
    t20_score = t20_file.target.to_numpy(dtype=float)
    blend = score_logodds(incumbent, t20_score, t34_score)
    if not np.isfinite(blend).all() or ((blend < 0) | (blend > 1)).any() or np.std(blend) <= 0:
        raise ValueError("Invalid final three-model candidate scores")
    pd.DataFrame({"index": expected, "incumbent": incumbent, "lr03_iter600": t20_score,
                  "lr04_iter450": t34_score, VARIANT: blend}).to_pickle(component_path)
    pd.DataFrame({"index": expected, "target": blend}).to_csv(candidate_path, index=False)
    exported = pd.read_csv(candidate_path)
    if (len(exported) != len(expected) or not np.array_equal(exported["index"].to_numpy(), expected)
            or not np.allclose(exported.target, blend, rtol=0, atol=1e-15)):
        raise ValueError("Candidate output order or values changed")
    return {"candidate_file": str(candidate_path.relative_to(PROJECT_ROOT)),
            "candidate_sha256": sha(candidate_path), "candidate_rows": len(exported),
            "candidate_std": float(exported.target.std()),
            "candidate_sample_sha256": sha(DATA_DIR / "sample_submission.csv"),
            "final_model_file": str(model_path.relative_to(PROJECT_ROOT)),
            "final_model_sha256": sha(model_path),
            "final_components_file": str(component_path.relative_to(PROJECT_ROOT)),
            "final_components_sha256": sha(component_path), "final_columns": columns,
            "final_unknown_weight_share": share, "final_unknown_weight_factor": float(factor),
            "final_history_max_step": int(history.time_step.max()), "kaggle_upload": False}


def main():
    plan = json.loads(PLAN_PATH.read_text())
    verify_inputs(plan)
    metadata = {"plan_sha256": sha(PLAN_PATH),
                "source_sha256": sha(PROJECT_ROOT / "src" / f"{PREFIX}.py"),
                "input_sha256": plan["input_sha256"], "original_gate_passed": False,
                "confirmation_run": False, "submission_created": False, "kaggle_upload": False}
    train, test_features, sample, groups = load_features()
    full_columns = groups["regularized_two_hop"]
    if len(full_columns) != 182:
        raise ValueError("182-column source feature set changed")
    original = evaluate(load_stage("original", ORIGINS), "original")
    original_summary = summarize(original, "original")
    original.to_csv(REPORT_DIR / f"{PREFIX}_original_results.csv", index=False)
    original_summary.to_csv(REPORT_DIR / f"{PREFIX}_original_summary.csv", index=False)
    metadata["original_gate_passed"] = bool(original_summary.qualified.iloc[0])
    print("ORIGINAL GATE\n" + original_summary.to_string(index=False), flush=True)
    frames, summaries = [original], [original_summary]
    if metadata["original_gate_passed"]:
        confirmation = evaluate(load_stage("confirmation", CONFIRMATION), "confirmation")
        confirmation_summary = summarize(confirmation, "confirmation")
        confirmation.to_csv(REPORT_DIR / f"{PREFIX}_confirmation_results.csv", index=False)
        confirmation_summary.to_csv(REPORT_DIR / f"{PREFIX}_confirmation_summary.csv", index=False)
        metadata["confirmation_run"] = True
        frames.append(confirmation)
        summaries.append(confirmation_summary)
        print("CONFIRMATION GATE\n" + confirmation_summary.to_string(index=False), flush=True)
        if bool(confirmation_summary.qualified.iloc[0]):
            metadata.update(export_candidate(plan, train, test_features, sample, full_columns))
            metadata["submission_created"] = True
    pd.concat(frames, ignore_index=True).to_csv(REPORT_DIR / f"{PREFIX}_results.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(REPORT_DIR / f"{PREFIX}_summary.csv", index=False)
    (REPORT_DIR / f"{PREFIX}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
