"""Frozen Trial 44: small fixed Trial 34 increment to the Trial 39 log-odds score."""
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
from src.competition import DATA_DIR, PROJECT_ROOT, REPORT_DIR
from src.dual_steps_logodds_20261002 import (
    CONFIRMATION, ORIGINS, VARIANT as TRIAL39_VARIANT,
    conditional_score, load_stage as load_trial39_stage, score_logodds,
)
from src.horizon_hgb_20260917 import historical_columns

PREFIX = "lr04_increment_logodds_20261002"
PLAN_PATH = REPORT_DIR / f"{PREFIX}_plan.json"
VARIANT = "trial39_logodds_95_lr04_iter450_5"
KEY = ["origin", "txId", "time_step", "label"]
CLIP = 1e-12
T20_MODEL = PROJECT_ROOT / "models/shrinkage_status_20261001/final_lr03_iter600_through_35.joblib"
T34_MODEL = PROJECT_ROOT / "models/dual_steps_logodds_20261002/final_lr04_iter450_through_35.joblib"
T39_COMPONENTS = REPORT_DIR / "dual_steps_logodds_20261002_final_components.pkl"
T39_CANDIDATE = PROJECT_ROOT / "submissions/candidate_dual_steps_logodds_20261002.csv"
T20_CANDIDATE = PROJECT_ROOT / "submissions/candidate_shrinkage_status_20261001_lr03_iter600.csv"
OLD_INCUMBENT = PROJECT_ROOT / "submissions/candidate_label_status_multiclass_20260929.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blend_score(control, challenger):
    a = np.asarray(control, dtype=float)
    b = np.asarray(challenger, dtype=float)
    if (a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all()
            or not np.isfinite(b).all() or ((a < 0) | (a > 1)).any()
            or ((b < 0) | (b > 1)).any()):
        raise ValueError("Trial 39 and Trial 34 must be aligned finite probabilities")
    return expit(.95 * logit(np.clip(a, CLIP, 1 - CLIP))
                 + .05 * logit(np.clip(b, CLIP, 1 - CLIP)))


def verify_inputs(plan):
    if plan.get("trial_number") != 44 or plan.get("variant") != VARIANT:
        raise ValueError("Frozen Trial 44 plan missing or mismatched")
    if (plan.get("weights") != {"trial39_control": .95, "trial34_lr04_iter450": .05}
            or plan.get("clip_epsilon") != CLIP or plan.get("new_model_fit") is not False):
        raise ValueError("Frozen Trial 44 blend contract changed")
    for name, digest in plan["input_sha256"].items():
        if sha(PROJECT_ROOT / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    metadata = json.loads((REPORT_DIR / "dual_steps_logodds_20261002_metadata.json").read_text())
    audit = json.loads((REPORT_DIR / "dual_steps_logodds_20261002_parent_verification.json").read_text())
    if (not metadata.get("original_gate_passed") or not metadata.get("confirmation_run")
            or not metadata.get("submission_created") or metadata.get("kaggle_upload") is not False
            or metadata.get("candidate_sha256") != sha(T39_CANDIDATE)
            or metadata.get("final_model_sha256") != sha(T34_MODEL)
            or metadata.get("final_components_sha256") != sha(T39_COMPONENTS)
            or not audit.get("exact_controls_verified")
            or not audit.get("saved_models_and_formula_reproduce_csv")
            or not audit.get("gates", {}).get("original", {}).get("pass")
            or not audit.get("gates", {}).get("confirmation", {}).get("pass")):
        raise ValueError("Trial 39 is not the saved verified eligible control")


def load_stage(stage, origins):
    trial39 = load_trial39_stage(stage, origins)
    t34 = pd.read_pickle(REPORT_DIR / f"intermediate_steps_status_20261002_{stage}_predictions.pkl")
    if (trial39.duplicated(KEY).any() or t34.duplicated(KEY).any()
            or set(trial39.origin.unique()) != set(origins)
            or set(t34.origin.unique()) != set(origins)):
        raise ValueError(f"{stage} saved forecast keys or origins changed")
    joined = trial39[KEY + ["incumbent", "lr04_iter450", TRIAL39_VARIANT]].merge(
        t34[KEY + ["incumbent", "lr04_iter450"]], on=KEY, how="outer",
        validate="one_to_one", sort=False, suffixes=("", "_saved"), indicator=True)
    if (len(joined) != len(trial39) or not joined._merge.eq("both").all()
            or not np.array_equal(joined[KEY].to_numpy(), trial39[KEY].to_numpy())
            or not np.array_equal(joined.incumbent.to_numpy(), joined.incumbent_saved.to_numpy())
            or not np.array_equal(joined.lr04_iter450.to_numpy(), joined.lr04_iter450_saved.to_numpy())):
        raise ValueError(f"{stage} Trial 39 and Trial 34 component forecasts differ")
    for origin, stop in origins.items():
        part = joined.loc[joined.origin.eq(origin)]
        if (part.empty or not part.time_step.between(origin + 1, stop).all()
                or int(part.time_step.max()) != stop):
            raise ValueError(f"{stage} forecast horizon changed at origin {origin}")
    out = joined[KEY + ["incumbent", "lr04_iter450", TRIAL39_VARIANT]].copy()
    out = out.rename(columns={"incumbent": "incumbent_20260929",
                              TRIAL39_VARIANT: "trial39_control"})
    out[VARIANT] = blend_score(out.trial39_control, out.lr04_iter450)
    return out


def evaluate(predictions, stage):
    rows = []
    for origin, frame in predictions.groupby("origin", sort=False):
        slices = {"first7": frame.loc[frame.time_step.le(origin + 7)]}
        if int(frame.time_step.max()) == int(origin) + 14:
            slices.update(full14=frame, second7=frame.loc[frame.time_step.gt(origin + 7)])
        for slice_name, part in slices.items():
            for variant in ("incumbent_20260929", "trial39_control", "lr04_iter450", VARIANT):
                rows.append({"stage": stage, "origin": int(origin), "slice": slice_name,
                             "variant": variant, "n": len(part), "positives": int(part.label.sum()),
                             "auc": float(roc_auc_score(part.label, part[variant])),
                             "average_precision": float(average_precision_score(part.label, part[variant]))})
    return pd.DataFrame(rows)


def summarize(results, stage):
    pivot = results.pivot(index=["origin", "slice"], columns="variant", values="auc")
    if stage == "original":
        groups = {"full14": [(14, "full14"), (21, "full14")],
                  "second7": [(14, "second7"), (21, "second7")],
                  "primary7": [(o, "first7") for o in (14, 21, 28)],
                  "rolling7": [(o, "first7") for o in (17, 24)]}
    else:
        groups = {"full14": [(o, "full14") for o in CONFIRMATION],
                  "second7": [(o, "second7") for o in CONFIRMATION],
                  "first7": [(o, "first7") for o in CONFIRMATION]}
    row = {"stage": stage, "variant": VARIANT, "control_variant": "trial39_control"}
    deltas = pivot[VARIANT] - pivot.trial39_control
    for name, keys in groups.items():
        row[f"{name}_mean_auc"] = float(pivot.loc[keys, VARIANT].mean())
        row[f"{name}_mean_delta"] = float(deltas.loc[keys].mean())
        row[f"{name}_control_auc"] = float(pivot.loc[keys, "trial39_control"].mean())
    row["worst_slice_delta"] = float(deltas.min())
    gates = ([row[f"{name}_mean_delta"] > 0 for name in groups] if stage == "original" else
             [row["full14_mean_delta"] > 0, row["second7_mean_delta"] > 0,
              row["first7_mean_delta"] >= 0])
    row["qualified"] = bool(all(gates) and row["worst_slice_delta"] >= -.001)
    return pd.DataFrame([row])


def export_candidate(plan, train, test_features, sample, full_columns):
    expected = sample["index"].to_numpy()
    if len(expected) != 15329 or not np.array_equal(test_features["index"].to_numpy(), expected):
        raise ValueError("Engineered test feature order changed")
    prior = pd.read_csv(OLD_INCUMBENT)
    t20 = pd.read_csv(T20_CANDIDATE)
    control = pd.read_csv(T39_CANDIDATE)
    components = pd.read_pickle(T39_COMPONENTS)
    for name, frame in (("old incumbent", prior), ("Trial 20", t20),
                        ("Trial 39", control), ("Trial 39 components", components)):
        if not np.array_equal(frame["index"].to_numpy(), expected):
            raise ValueError(f"{name} test order changed")
    columns = historical_columns(train, full_columns, 35)
    if len(columns) != 172:
        raise ValueError("Final historical feature count changed")
    t20_model, t34_model = joblib.load(T20_MODEL), joblib.load(T34_MODEL)
    if (t20_model["columns"] != columns or t34_model["columns"] != columns
            or t20_model["model"].n_iter_ != 600 or t34_model["model"].n_iter_ != 450):
        raise ValueError("Saved final component model settings changed")
    t20_score = conditional_score(t20_model["model"], test_features, columns)
    t34_score = conditional_score(t34_model["model"], test_features, columns)
    if (not np.allclose(t20_score, t20.target, rtol=0, atol=1e-12)
            or not np.allclose(components.incumbent, prior.target, rtol=0, atol=1e-15)
            or not np.allclose(components.lr03_iter600, t20_score, rtol=0, atol=1e-12)
            or not np.allclose(components.lr04_iter450, t34_score, rtol=0, atol=1e-12)):
        raise ValueError("Saved Trial 39 components failed final model replay")
    reconstructed = score_logodds(prior.target.to_numpy(dtype=float), t20_score, t34_score)
    if (not np.allclose(reconstructed, components[TRIAL39_VARIANT], rtol=0, atol=1e-12)
            or not np.allclose(reconstructed, control.target, rtol=0, atol=1e-12)):
        raise ValueError("Trial 39 final score formula or candidate changed")
    score = blend_score(control.target.to_numpy(dtype=float), t34_score)
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any() or np.std(score) <= 0:
        raise ValueError("Invalid Trial 44 final scores")
    candidate_path = PROJECT_ROOT / plan["candidate_output"]
    components_path = REPORT_DIR / f"{PREFIX}_final_components.pkl"
    if candidate_path.exists() or components_path.exists():
        raise ValueError("Trial 44 export already exists; preserve it")
    pd.DataFrame({"index": expected, "trial39_control": control.target,
                  "lr04_iter450": t34_score, VARIANT: score}).to_pickle(components_path)
    pd.DataFrame({"index": expected, "target": score}).to_csv(candidate_path, index=False)
    readback = pd.read_csv(candidate_path)
    if (len(readback) != len(sample) or not np.array_equal(readback["index"].to_numpy(), expected)
            or not np.allclose(readback.target, score, rtol=0, atol=1e-15)):
        raise ValueError("Trial 44 candidate readback changed")
    return {"candidate_file": plan["candidate_output"], "candidate_sha256": sha(candidate_path),
            "candidate_rows": len(readback), "candidate_std": float(readback.target.std()),
            "sample_sha256": sha(DATA_DIR / "sample_submission.csv"),
            "trial39_control_sha256": sha(T39_CANDIDATE),
            "trial20_model_sha256": sha(T20_MODEL), "trial34_model_sha256": sha(T34_MODEL),
            "final_components_file": str(components_path.relative_to(PROJECT_ROOT)),
            "final_components_sha256": sha(components_path), "final_feature_count": len(columns),
            "final_history_max_step": 35, "kaggle_upload": False}


def main():
    plan = json.loads(PLAN_PATH.read_text())
    verify_inputs(plan)
    metadata = {"plan_sha256": sha(PLAN_PATH),
                "source_sha256": sha(PROJECT_ROOT / "src" / f"{PREFIX}.py"),
                "input_sha256": plan["input_sha256"], "original_gate_passed": False,
                "confirmation_run": False, "submission_created": False,
                "new_model_fit": False, "kaggle_upload": False}
    original_predictions = load_stage("original", ORIGINS)
    original_predictions.to_pickle(REPORT_DIR / f"{PREFIX}_original_predictions.pkl")
    original = evaluate(original_predictions, "original")
    original_summary = summarize(original, "original")
    original.to_csv(REPORT_DIR / f"{PREFIX}_original_results.csv", index=False)
    original_summary.to_csv(REPORT_DIR / f"{PREFIX}_original_summary.csv", index=False)
    metadata["original_gate_passed"] = bool(original_summary.qualified.iloc[0])
    print("ORIGINAL GATE\n" + original_summary.to_string(index=False), flush=True)
    frames, summaries = [original], [original_summary]
    if metadata["original_gate_passed"]:
        confirmation_predictions = load_stage("confirmation", CONFIRMATION)
        confirmation_predictions.to_pickle(REPORT_DIR / f"{PREFIX}_confirmation_predictions.pkl")
        confirmation = evaluate(confirmation_predictions, "confirmation")
        confirmation_summary = summarize(confirmation, "confirmation")
        confirmation.to_csv(REPORT_DIR / f"{PREFIX}_confirmation_results.csv", index=False)
        confirmation_summary.to_csv(REPORT_DIR / f"{PREFIX}_confirmation_summary.csv", index=False)
        metadata["confirmation_run"] = True
        frames.append(confirmation)
        summaries.append(confirmation_summary)
        print("CONFIRMATION GATE\n" + confirmation_summary.to_string(index=False), flush=True)
        if bool(confirmation_summary.qualified.iloc[0]):
            train, test_features, sample, groups = load_features()
            if len(groups["regularized_two_hop"]) != 182:
                raise ValueError("182-column source feature set changed")
            metadata.update(export_candidate(plan, train, test_features, sample,
                                             groups["regularized_two_hop"]))
            metadata["submission_created"] = True
    pd.concat(frames, ignore_index=True).to_csv(REPORT_DIR / f"{PREFIX}_results.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(REPORT_DIR / f"{PREFIX}_summary.csv", index=False)
    (REPORT_DIR / f"{PREFIX}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
