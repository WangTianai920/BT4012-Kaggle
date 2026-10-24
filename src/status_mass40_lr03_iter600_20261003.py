"""Frozen Trial 53: 40% unknown-status mass with slower HGB learning."""
from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

from src.combined_challengers import load_features
from src.competition import DATA_DIR, PROJECT_ROOT, REPORT_DIR
from src.dual_steps_logodds_20261002 import conditional_score
from src.horizon_hgb_20260917 import BASE, historical_columns, model, weights as recency_weights
from src.label_status_multiclass_20260929 import label_status
from src.lr04_increment_logodds_20261002 import (
    CONFIRMATION, ORIGINS, VARIANT as TRIAL44_VARIANT,
    load_stage as load_trial44_stage,
)

PREFIX = "status_mass40_lr03_iter600_20261003"
PLAN_PATH = REPORT_DIR / f"{PREFIX}_plan.json"
MODEL_DIR = PROJECT_ROOT / "models" / PREFIX
CHALLENGER = "status_mass40_lr03_iter600"
BLEND = "blend_75_25_logodds"
KEY = ["origin", "txId", "time_step", "label"]
CLIP = 1e-12
T44_CANDIDATE = PROJECT_ROOT / "submissions/candidate_lr04_increment_logodds_20261002.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config():
    return dict(BASE, learning_rate=.03, max_iter=600)


UNKNOWN_SHARE = .40


def training_weights_40(steps, status, cutoff):
    """Keep every known-row recency weight and set unknown status to 40% mass."""
    base = recency_weights(steps, cutoff, BASE["half_life"])
    unknown = np.asarray(status) == 2
    known_mass = float(base[~unknown].sum())
    unknown_mass = float(base[unknown].sum())
    if known_mass <= 0 or unknown_mass <= 0:
        raise ValueError("Both known and unknown status histories are required")
    factor = known_mass * UNKNOWN_SHARE / ((1 - UNKNOWN_SHARE) * unknown_mass)
    result = base.copy()
    result[unknown] *= factor
    if (not np.array_equal(result[~unknown], base[~unknown])
            or not np.isclose(result[unknown].sum() / result.sum(), UNKNOWN_SHARE,
                              rtol=0, atol=1e-12)):
        raise ValueError("Frozen 40% status mass or known recency weights changed")
    return result, factor


def blend_score(control, challenger):
    a, b = np.asarray(control, dtype=float), np.asarray(challenger, dtype=float)
    if (a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all()
            or not np.isfinite(b).all() or ((a < 0) | (a > 1)).any()
            or ((b < 0) | (b > 1)).any()):
        raise ValueError("Trial44 control and challenger must be aligned probabilities")
    return expit(.75 * logit(np.clip(a, CLIP, 1 - CLIP))
                 + .25 * logit(np.clip(b, CLIP, 1 - CLIP)))


def verify_inputs(plan):
    if plan.get("trial_number") != 53 or plan.get("variant") != PREFIX:
        raise ValueError("Frozen Trial53 plan missing or mismatched")
    if (plan.get("configuration") != config()
            or plan.get("weights") != {"trial44_control": .75, CHALLENGER: .25}
            or plan.get("clip_epsilon") != CLIP):
        raise ValueError("Frozen Trial53 configuration or blend changed")
    if sha(PROJECT_ROOT / "src" / f"{PREFIX}.py") != plan.get("source_sha256"):
        raise ValueError("Frozen Trial53 source hash changed")
    for name, digest in plan["input_sha256"].items():
        if sha(PROJECT_ROOT / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    metadata = json.loads((REPORT_DIR / "lr04_increment_logodds_20261002_metadata.json").read_text())
    audit = json.loads((REPORT_DIR / "lr04_increment_logodds_20261002_parent_verification.json").read_text())
    submission = json.loads((REPORT_DIR / "lr04_increment_logodds_20261002_submission_result.json").read_text())
    if (not metadata.get("original_gate_passed") or not metadata.get("confirmation_run")
            or not metadata.get("submission_created") or metadata.get("kaggle_upload") is not False
            or metadata.get("candidate_sha256") != sha(T44_CANDIDATE)
            or submission.get("submission_name") != "1002-3"
            or submission.get("public_roc_auc") != .95238
            or not audit.get("exact_trial39_controls_verified")
            or not audit.get("formula_verified")
            or not audit.get("gates", {}).get("original", {}).get("pass")
            or not audit.get("gates", {}).get("confirmation", {}).get("pass")):
        raise ValueError("Trial44 is not the recorded, independently verified eligible control")


def fit_origin(train, full_columns, origin, stop, stage, control_frame):
    history = train.loc[train.time_step.le(origin)]
    future = train.loc[train.time_step.between(origin + 1, stop)]
    columns = historical_columns(train, full_columns, origin)
    status = label_status(history.label)
    weights, factor = training_weights_40(history.time_step, status, origin)
    share = float(weights[status == 2].sum() / weights.sum())
    if (len(columns) != 172 or int(history.time_step.max()) != origin
            or not np.array_equal(np.unique(status), [0, 1, 2])
            or not np.isclose(share, UNKNOWN_SHARE, rtol=0, atol=1e-12)):
        raise ValueError("Historical features, status classes, or unknown weight changed")
    fitted = model(config())
    fitted.fit(history[columns], status, sample_weight=weights)
    if fitted.n_iter_ != 600:
        raise ValueError("HGB fit did not reach the fixed 600 iterations")
    challenger = conditional_score(fitted, future, columns)
    model_path = MODEL_DIR / f"{stage}_{origin}_{stop}_model.joblib"
    if model_path.exists():
        raise ValueError("Historical Trial53 model already exists; preserve it")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted, "columns": columns, "config": config(),
                 "stage": stage, "origin": origin, "forecast_end": stop,
                 "unknown_weight_share": share, "unknown_weight_factor": float(factor),
                 "plan_sha256": sha(PLAN_PATH)}, model_path)
    restored = joblib.load(model_path)
    if (restored["columns"] != columns or restored["config"] != config()
            or restored["plan_sha256"] != sha(PLAN_PATH)
            or not np.allclose(conditional_score(restored["model"], future, columns),
                               challenger, rtol=0, atol=1e-12)):
        raise ValueError("Saved historical Trial53 model replay changed")
    known = future.loc[future.label.notna(), KEY[1:]].copy()
    known[CHALLENGER] = challenger[future.label.notna().to_numpy()]
    control = control_frame.loc[control_frame.origin.eq(origin),
                                KEY + ["incumbent_20260929", "trial44_control"]]
    if control.duplicated(KEY).any():
        raise ValueError("Trial44 forecast keys are not unique")
    aligned = known[KEY[1:]].merge(control, on=KEY[1:], how="left", validate="one_to_one",
                                   sort=False, indicator=True)
    if (len(aligned) != len(known) or not aligned._merge.eq("both").all()
            or not np.array_equal(aligned.txId.to_numpy(), known.txId.to_numpy())
            or not np.array_equal(aligned.time_step.to_numpy(), known.time_step.to_numpy())
            or not np.array_equal(aligned.label.to_numpy(), known.label.to_numpy())):
        raise ValueError("Trial44 keyed control population, order, periods, or labels changed")
    known["trial44_control"] = aligned.trial44_control.to_numpy(dtype=float)
    known["incumbent_20260929"] = aligned.incumbent_20260929.to_numpy(dtype=float)
    known[BLEND] = blend_score(known.trial44_control, known[CHALLENGER])
    known.insert(0, "origin", origin)
    detail = {"stage": stage, "origin": origin, "forecast_end": stop,
              "history_rows": len(history), "future_rows": len(future),
              "known_forecast_rows": len(known), "history_max_step": int(history.time_step.max()),
              "feature_count": len(columns), "known_licit": int((status == 0).sum()),
              "known_illicit": int((status == 1).sum()), "unlabelled": int((status == 2).sum()),
              "unknown_weight_share": share, "unknown_weight_factor": float(factor),
              "base_config": config(), "iteration_count": int(fitted.n_iter_),
              "control": "Trial44 saved candidate, exactly keyed to Trial44 reconstructed validation scores",
              "model_file": str(model_path.relative_to(PROJECT_ROOT)),
              "model_sha256": sha(model_path)}
    return known, detail


def run_stage(train, full_columns, origins, stage):
    trial44 = load_trial44_stage(stage, origins)
    required = set(KEY + ["incumbent_20260929", TRIAL44_VARIANT])
    if (not required.issubset(trial44.columns) or trial44.duplicated(KEY).any()
            or set(trial44.origin.unique()) != set(origins)):
        raise ValueError("Trial44 loader control schema, keys, or origins changed")
    for origin, stop in origins.items():
        control_rows = trial44.loc[trial44.origin.eq(origin)]
        if (control_rows.empty or not control_rows.time_step.between(origin + 1, stop).all()
                or int(control_rows.time_step.max()) != stop):
            raise ValueError(f"Trial44 control horizon changed at origin {origin}")
    # Trial44's frozen loader exposes its saved candidate under the variant name.
    # Normalize that column before any Trial53 model fit or per-origin merge.
    trial44 = trial44.rename(columns={TRIAL44_VARIANT: "trial44_control"})
    parts, diagnostics = [], []
    for origin, stop in origins.items():
        part, detail = fit_origin(train, full_columns, origin, stop, stage, trial44)
        expected = train.loc[train.time_step.between(origin + 1, stop) & train.label.notna(), "txId"]
        if not np.array_equal(part.txId.to_numpy(), expected.to_numpy()):
            raise ValueError("Known validation population or order changed")
        parts.append(part)
        diagnostics.append(detail)
        print(f"Completed {stage} {origin}->{stop}: 40% unknown mass, 600 iterations", flush=True)
    return pd.concat(parts, ignore_index=True), diagnostics


def evaluate(predictions, stage):
    rows = []
    for origin, frame in predictions.groupby("origin", sort=False):
        slices = {"first7": frame.loc[frame.time_step.le(origin + 7)]}
        if int(frame.time_step.max()) == int(origin) + 14:
            slices.update(full14=frame, second7=frame.loc[frame.time_step.gt(origin + 7)])
        for slice_name, part in slices.items():
            for variant in ("incumbent_20260929", "trial44_control", CHALLENGER, BLEND):
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
    rows = []
    for variant in (CHALLENGER, BLEND):
        delta = pivot[variant] - pivot.trial44_control
        row = {"stage": stage, "variant": variant, "control_variant": "trial44_control"}
        for name, keys in groups.items():
            row[f"{name}_mean_auc"] = float(pivot.loc[keys, variant].mean())
            row[f"{name}_mean_delta"] = float(delta.loc[keys].mean())
            row[f"{name}_control_auc"] = float(pivot.loc[keys, "trial44_control"].mean())
        row["worst_slice_delta"] = float(delta.min())
        gates = ([row[f"{name}_mean_delta"] > 0 for name in groups] if stage == "original" else
                 [row["full14_mean_delta"] > 0, row["second7_mean_delta"] > 0,
                  row["first7_mean_delta"] >= 0])
        row["qualified"] = bool(all(gates) and row["worst_slice_delta"] >= -.001)
        rows.append(row)
    return pd.DataFrame(rows)


def export_candidate(plan, train, test_features, sample, full_columns, winner):
    expected = sample["index"].to_numpy()
    control = pd.read_csv(T44_CANDIDATE)
    if (len(expected) != 15329 or not np.array_equal(test_features["index"].to_numpy(), expected)
            or not np.array_equal(control["index"].to_numpy(), expected)):
        raise ValueError("Trial44 candidate or sample order changed")
    history = train.loc[train.time_step.le(35)]
    columns = historical_columns(train, full_columns, 35)
    status = label_status(history.label)
    weights, factor = training_weights_40(history.time_step, status, 35)
    share = float(weights[status == 2].sum() / weights.sum())
    if (len(columns) != 172 or not np.isclose(share, UNKNOWN_SHARE, rtol=0, atol=1e-12)
            or not np.array_equal(np.unique(status), [0, 1, 2])):
        raise ValueError("Final historical features, statuses, or unknown weight changed")
    fitted = model(config())
    fitted.fit(history[columns], status, sample_weight=weights)
    if fitted.n_iter_ != 600:
        raise ValueError("Final model did not reach 600 iterations")
    challenger = conditional_score(fitted, test_features, columns)
    model_path = MODEL_DIR / "final_status_mass40_lr03_iter600_through_35.joblib"
    components_path = REPORT_DIR / f"{PREFIX}_final_components.pkl"
    candidate_path = PROJECT_ROOT / plan["candidate_output"]
    if model_path.exists() or components_path.exists() or candidate_path.exists():
        raise ValueError("Trial53 candidate or final artifacts already exist; preserve them")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted, "columns": columns, "plan_sha256": sha(PLAN_PATH),
                 "config": config(), "classes": [0, 1, 2]}, model_path)
    replay = conditional_score(joblib.load(model_path)["model"], test_features, columns)
    if not np.allclose(replay, challenger, rtol=0, atol=1e-12):
        raise ValueError("Saved final 40%-unknown-mass model replay changed")
    control_score = control.target.to_numpy(dtype=float)
    blended = blend_score(control_score, challenger)
    selected = challenger if winner == CHALLENGER else blended
    if not np.isfinite(selected).all() or ((selected < 0) | (selected > 1)).any() or np.std(selected) <= 0:
        raise ValueError("Invalid Trial53 candidate scores")
    pd.DataFrame({"index": expected, "trial44_control": control_score,
                  CHALLENGER: challenger, BLEND: blended}).to_pickle(components_path)
    pd.DataFrame({"index": expected, "target": selected}).to_csv(candidate_path, index=False)
    exported = pd.read_csv(candidate_path)
    if (len(exported) != len(expected) or not np.array_equal(exported["index"].to_numpy(), expected)
            or not np.allclose(exported.target, selected, rtol=0, atol=1e-15)):
        raise ValueError("Trial53 candidate order or values changed")
    return {"winner": winner, "candidate_file": str(candidate_path.relative_to(PROJECT_ROOT)),
            "candidate_sha256": sha(candidate_path), "candidate_rows": len(exported),
            "candidate_std": float(exported.target.std()), "control_sha256": sha(T44_CANDIDATE),
            "final_model_file": str(model_path.relative_to(PROJECT_ROOT)),
            "final_model_sha256": sha(model_path),
            "final_components_file": str(components_path.relative_to(PROJECT_ROOT)),
            "final_components_sha256": sha(components_path), "final_columns": columns,
            "final_unknown_weight_share": share, "final_unknown_weight_factor": float(factor),
            "final_history_max_step": int(history.time_step.max()), "kaggle_upload": False}


def main():
    plan = json.loads(PLAN_PATH.read_text())
    verify_inputs(plan)
    metadata = {"plan_sha256": sha(PLAN_PATH), "source_sha256": sha(PROJECT_ROOT / "src" / f"{PREFIX}.py"),
                "input_sha256": plan["input_sha256"], "original_gate_passed": False,
                "confirmation_run": False, "submission_created": False, "kaggle_upload": False,
                "final_model_fit_performed": False,
                "primary_control": "Trial44 submitted as 1002-3; user-reported score 0.95238"}
    train, test_features, sample, groups = load_features()
    full_columns = groups["regularized_two_hop"]
    if len(full_columns) != 182:
        raise ValueError("182-column source feature set changed")
    original, diagnostics = run_stage(train, full_columns, ORIGINS, "original")
    original.to_pickle(REPORT_DIR / f"{PREFIX}_original_predictions.pkl")
    original_results, original_summary = evaluate(original, "original"), None
    original_summary = summarize(original_results, "original")
    original_results.to_csv(REPORT_DIR / f"{PREFIX}_original_results.csv", index=False)
    original_summary.to_csv(REPORT_DIR / f"{PREFIX}_original_summary.csv", index=False)
    metadata["original_gate_passed"] = bool(original_summary.qualified.any())
    print("ORIGINAL GATE\n" + original_summary.to_string(index=False), flush=True)
    result_frames, summary_frames = [original_results], [original_summary]
    if metadata["original_gate_passed"]:
        confirmation, confirm_diagnostics = run_stage(train, full_columns, CONFIRMATION, "confirmation")
        confirmation.to_pickle(REPORT_DIR / f"{PREFIX}_confirmation_predictions.pkl")
        confirmation_results = evaluate(confirmation, "confirmation")
        confirmation_summary = summarize(confirmation_results, "confirmation")
        confirmation_results.to_csv(REPORT_DIR / f"{PREFIX}_confirmation_results.csv", index=False)
        confirmation_summary.to_csv(REPORT_DIR / f"{PREFIX}_confirmation_summary.csv", index=False)
        metadata["confirmation_run"] = True
        diagnostics += confirm_diagnostics
        result_frames.append(confirmation_results)
        summary_frames.append(confirmation_summary)
        print("CONFIRMATION GATE\n" + confirmation_summary.to_string(index=False), flush=True)
        winners = set(original_summary.loc[original_summary.qualified, "variant"]) & set(
            confirmation_summary.loc[confirmation_summary.qualified, "variant"])
        if winners:
            winner = confirmation_summary.loc[confirmation_summary.variant.isin(winners)].sort_values(
                ["full14_mean_auc", "second7_mean_auc"], ascending=False).iloc[0].variant
            metadata.update(export_candidate(plan, train, test_features, sample, full_columns, winner))
            metadata["submission_created"] = True
            metadata["winner"] = winner
            metadata["final_model_fit_performed"] = True
    metadata["fit_diagnostics"] = diagnostics
    pd.concat(result_frames, ignore_index=True).to_csv(REPORT_DIR / f"{PREFIX}_results.csv", index=False)
    pd.concat(summary_frames, ignore_index=True).to_csv(REPORT_DIR / f"{PREFIX}_summary.csv", index=False)
    (REPORT_DIR / f"{PREFIX}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
