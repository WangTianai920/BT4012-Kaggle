"""Prepared Trial81: hard-status mass40 HGB with a single 31-leaf change."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

from src.combined_challengers import load_features
from src.competition import PROJECT_ROOT, RANDOM_SEED, REPORT_DIR
from src.dual_steps_logodds_20261002 import conditional_score
from src.horizon_hgb_20260917 import BASE, historical_columns, model
from src.label_status_multiclass_20260929 import label_status
from src.lr04_increment_logodds_20261002 import CONFIRMATION, ORIGINS
from src.status_mass40_lr03_iter600_20261003 import training_weights_40

PREFIX = "status_mass40_leaf31_public55_20261004"
PLAN_PATH = REPORT_DIR / f"{PREFIX}_plan.json"
ROOT_AUTHORIZATION = REPORT_DIR / f"{PREFIX}_root_authorization.json"
MODEL_DIR = PROJECT_ROOT / "models" / PREFIX
CANDIDATE_PATH = PROJECT_ROOT / "submissions" / f"candidate_{PREFIX}.csv"
COMPONENTS_PATH = REPORT_DIR / f"{PREFIX}_final_components.pkl"
CHALLENGER = "status_mass40_leaf31_lr03_iter600"
BLEND = "blend_75_25_logodds"
TRIAL55_PREFIX = "status_mass40_logodds_blend_20261003"
TRIAL55_VARIANT = "blend_75_25_logodds"
TRIAL55_CANDIDATE = PROJECT_ROOT / "submissions" / "candidate_status_mass40_logodds_blend_20261003.csv"
TRIAL80_PREFIX = "weighted_pls4_fixed14_public55_20261004"
TRIAL80_REGISTRY_ID = f"S80_{TRIAL80_PREFIX}_independent_parent_audit"
TRIAL77_RESULT = REPORT_DIR / "full_status_smoothing03_trial74_20261003_submission_result.json"
TRIAL77_CANDIDATE = PROJECT_ROOT / "submissions" / "candidate_full_status_smoothing03_trial74_20261003.csv"
KEY = ["origin", "txId", "time_step", "label"]
PUBLIC55_CONTROL = "public55_control"
CLIP_EPSILON = 1e-12
UNKNOWN_SHARE = 0.40
HALF_LIFE = 20.0
FEATURE_COUNT = 172


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config() -> dict:
    return dict(BASE, learning_rate=0.03, max_iter=600, max_leaf_nodes=31,
                min_samples_leaf=40, l2_regularization=2.0, max_features=1.0,
                half_life=HALF_LIFE)


def frozen_configuration() -> dict:
    return {**config(), "random_seed": 42, "early_stopping": False}


def blend_score(control, challenger):
    a = np.asarray(control, dtype=float)
    b = np.asarray(challenger, dtype=float)
    if (a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all()
            or not np.isfinite(b).all() or ((a < 0) | (a > 1)).any()
            or ((b < 0) | (b > 1)).any()):
        raise ValueError("Trial55 control and Trial81 challenger must be aligned probabilities")
    return expit(0.75 * logit(np.clip(a, CLIP_EPSILON, 1 - CLIP_EPSILON))
                 + 0.25 * logit(np.clip(b, CLIP_EPSILON, 1 - CLIP_EPSILON)))


def verify_gate_logic() -> None:
    """Fit-free checks for the fixed original and confirmation gates."""
    original_good = {"full14_mean_delta": .01, "second7_mean_delta": .01,
                     "primary7_mean_delta": .01, "rolling7_mean_delta": .01,
                     "worst_slice_delta": -.001}
    confirmation_good = {"full14_mean_delta": .01, "second7_mean_delta": .01,
                         "first7_mean_delta": 0.0, "worst_slice_delta": -.001}
    if not stage_gate_pass("original", original_good) or not stage_gate_pass("confirmation", confirmation_good):
        raise AssertionError("A synthetic qualifying candidate failed a stage gate")
    for key in ("full14_mean_delta", "second7_mean_delta", "primary7_mean_delta", "rolling7_mean_delta"):
        bad = dict(original_good); bad[key] = 0.0
        if stage_gate_pass("original", bad):
            raise AssertionError(f"Original gate accepted nonpositive {key}")
    for key in ("full14_mean_delta", "second7_mean_delta"):
        bad = dict(confirmation_good); bad[key] = 0.0
        if stage_gate_pass("confirmation", bad):
            raise AssertionError(f"Confirmation gate accepted nonpositive {key}")
    for key, value in (("first7_mean_delta", -1e-12), ("worst_slice_delta", -.0010001)):
        bad = dict(confirmation_good); bad[key] = value
        if stage_gate_pass("confirmation", bad):
            raise AssertionError(f"Confirmation gate accepted invalid {key}")


def stage_gate_pass(stage: str, row: dict) -> bool:
    worst_ok = row.get("worst_slice_delta", float("-inf")) >= -0.001
    if stage == "original":
        return (all(row.get(f"{name}_mean_delta", float("-inf")) > 0
                    for name in ("full14", "second7", "primary7", "rolling7")) and worst_ok)
    if stage == "confirmation":
        return (row.get("full14_mean_delta", float("-inf")) > 0
                and row.get("second7_mean_delta", float("-inf")) > 0
                and row.get("first7_mean_delta", float("-inf")) >= 0 and worst_ok)
    raise ValueError(f"Unknown validation stage: {stage}")


def verify_synthetic_contracts() -> None:
    """Check model settings, hard-status mass math, and gates without project data."""
    cfg = config()
    expected = {"max_iter": 600, "learning_rate": .03, "max_leaf_nodes": 31,
                "min_samples_leaf": 40, "l2_regularization": 2.0, "half_life": 20.0}
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise AssertionError("Trial81 differs from the one-variable Trial53 comparison")
    if RANDOM_SEED != 42:
        raise AssertionError("Pinned random seed changed")
    toy_status = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    toy_steps = pd.Series([1, 1, 1, 2, 2, 2])
    weights, _ = training_weights_40(toy_steps, toy_status, 2)
    if not np.isclose(weights[toy_status == 2].sum() / weights.sum(), UNKNOWN_SHARE,
                      rtol=0, atol=1e-12):
        raise AssertionError("Synthetic 40% unknown-status mass check failed")
    if model(cfg).get_params()["max_leaf_nodes"] != 31:
        raise AssertionError("Model constructor did not preserve the single leaf-count change")
    verify_gate_logic()


def verify_trial80_negative_closeout() -> None:
    """Require Trial80 failure, independent audit, and registration before data use."""
    meta_path = REPORT_DIR / f"{TRIAL80_PREFIX}_metadata.json"
    audit_path = REPORT_DIR / f"{TRIAL80_PREFIX}_parent_verification.json"
    if not meta_path.exists() or not audit_path.exists():
        raise RuntimeError("Trial80 must be negatively closed out and independently audited first")
    meta = json.loads(meta_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (meta.get("trial_number") != 80 or meta.get("variant") != TRIAL80_PREFIX
            or meta.get("reference_map_passed") is not True
            or meta.get("original_gate_passed") is not False
            or meta.get("confirmation_run") is not False
            or meta.get("submission_created") is not False
            or meta.get("final_model_fit_performed") is not False
            or audit.get("trial_number") != 80 or audit.get("variant") != TRIAL80_PREFIX
            or audit.get("reference_map_passed") is not True
            or audit.get("frozen_hashes_verified") is not True
            or audit.get("all_cached_outer_models_replayed") is not True
            or audit.get("eligible_variants") != []):
        raise RuntimeError("Trial80 is not a verified negative closeout")
    gates = audit.get("gates", {}).get("original", {})
    if not gates or any(item.get("pass") is not False for item in gates.values()):
        raise RuntimeError("Trial80 original-gate failure was not independently established")

    csv.field_size_limit(10_000_000)
    with (PROJECT_ROOT / "experiment_log.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    identifiers = [row.get("experiment_id", "") for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Experiment log contains duplicate IDs at Trial81 preflight")
    if len(set(identifiers)) < int(json.loads(PLAN_PATH.read_text()).get("trial80_registry_unique_ids", 0)):
        raise RuntimeError("Experiment log registry fell below the Trial80 preflight count")
    registered = [row for row in rows if row.get("experiment_id") == TRIAL80_REGISTRY_ID]
    if len(registered) != 1:
        raise RuntimeError("Trial80 independent audit must be registered exactly once")
    row = registered[0]
    if (row.get("study") != TRIAL80_PREFIX
            or row.get("evaluation_stage") != "diagnostic"
            or row.get("source_file") != f"reports/{TRIAL80_PREFIX}_parent_verification.json"
            or row.get("source_sha256") != sha(audit_path)):
        raise RuntimeError("Trial80 audit registration does not match the verified closeout")


def verify_public55_and_trial77(plan: dict, authorization: dict) -> None:
    control_meta_path = REPORT_DIR / f"{TRIAL55_PREFIX}_metadata.json"
    control_audit_path = REPORT_DIR / f"{TRIAL55_PREFIX}_parent_verification.json"
    control_result_path = REPORT_DIR / f"{TRIAL55_PREFIX}_submission_result.json"
    if not all(path.exists() for path in (control_meta_path, control_audit_path, control_result_path)):
        raise RuntimeError("Saved public-best Trial55 control lineage is incomplete")
    control_meta = json.loads(control_meta_path.read_text())
    control_audit = json.loads(control_audit_path.read_text())
    control_result = json.loads(control_result_path.read_text())
    if (not control_meta.get("original_gate_passed")
            or control_meta.get("confirmation_run") is not True
            or control_audit.get("gates", {}).get("confirmation", {}).get(TRIAL55_VARIANT, {}).get("pass") is not True
            or control_meta.get("submission_created") is not True
            or control_meta.get("kaggle_upload") is not False
            or control_meta.get("candidate_sha256") != sha(TRIAL55_CANDIDATE)
            or control_result.get("submission_name") != "1003-2"
            or control_result.get("public_roc_auc") != 0.95255
            or control_result.get("private_roc_auc") is not None
            or control_result.get("candidate_sha256") != sha(TRIAL55_CANDIDATE)
            or control_audit.get("eligible_variants") != [TRIAL55_VARIANT]
            or control_audit.get("all_saved_outer_models_replayed") is not True
            or control_audit.get("frozen_hashes_verified") is not True):
        raise RuntimeError("Trial55 is not the independently audited public-best control")

    if not TRIAL77_RESULT.exists() or not TRIAL77_CANDIDATE.exists():
        raise RuntimeError("Canonical Trial77 public result or candidate is missing")
    result = json.loads(TRIAL77_RESULT.read_text())
    expected = plan.get("public_trial77_result", {})
    result_hash = sha(TRIAL77_RESULT)
    if (result_hash != expected.get("sha256")
            or result.get("submission_name") != "1004-1"
            or result.get("source") != "user_reported"
            or result.get("public_roc_auc") != 0.94903
            or result.get("private_roc_auc") is not None
            or result.get("incumbent_name") != "1003-2"
            or result.get("incumbent_public_roc_auc") != 0.95255
            or result.get("delta_vs_incumbent") != -0.00352
            or result.get("candidate_sha256") != sha(TRIAL77_CANDIDATE)
            or authorization.get("public_trial77_underperformed_acknowledged") is not True
            or authorization.get("public_trial77_result_sha256") != result_hash):
        raise RuntimeError("Pinned Trial77 public-transfer result or acknowledgement changed")


def verify_inputs(plan: dict) -> None:
    if plan.get("trial_number") != 81 or plan.get("variant") != PREFIX:
        raise ValueError("Frozen Trial81 plan identity mismatch")
    if plan.get("configuration") != frozen_configuration():
        raise ValueError("Trial81 configuration differs from its frozen plan")
    if sha(PROJECT_ROOT / "src" / f"{PREFIX}.py") != plan.get("source_sha256"):
        raise ValueError("Trial81 source changed after plan freeze")
    for name, digest in plan["input_sha256"].items():
        if sha(PROJECT_ROOT / name) != digest:
            raise ValueError(f"Frozen Trial81 input changed: {name}")
    if not ROOT_AUTHORIZATION.exists():
        raise RuntimeError("Matching root authorization is required before project-data loading or fitting")
    authorization = json.loads(ROOT_AUTHORIZATION.read_text())
    audit_path = REPORT_DIR / f"{TRIAL80_PREFIX}_parent_verification.json"
    required_registry_count = int(plan.get("trial80_registry_unique_ids", 0))
    if (authorization.get("authorized") is not True
            or authorization.get("source_sha256") != plan["source_sha256"]
            or authorization.get("plan_sha256") != sha(PLAN_PATH)
            or authorization.get("trial80_negative_audited_registered") is not True
            or authorization.get("trial80_parent_audit_sha256") != sha(audit_path)
            or authorization.get("trial80_registry_id") != TRIAL80_REGISTRY_ID
            or int(authorization.get("registry_unique_ids_preflight", 0)) < required_registry_count):
        raise RuntimeError("Root authorization does not match Trial81 or Trial80 closeout")
    verify_trial80_negative_closeout()
    verify_public55_and_trial77(plan, authorization)
    verify_synthetic_contracts()
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        raise RuntimeError("Trial81 model artifacts already exist; preserve them")
    if CANDIDATE_PATH.exists() or COMPONENTS_PATH.exists():
        raise RuntimeError("Trial81 final artifacts already exist; preserve them")


def load_trial55_stage(stage: str, origins: dict[int, int]) -> pd.DataFrame:
    frame = pd.read_pickle(REPORT_DIR / f"{TRIAL55_PREFIX}_{stage}_predictions.pkl")
    required = KEY + [TRIAL55_VARIANT]
    if (not set(required).issubset(frame.columns) or frame.duplicated(KEY).any()
            or set(frame.origin.unique()) != set(origins)):
        raise ValueError("Saved Trial55 forecast schema, keys, or origins changed")
    frame = frame[required].copy().rename(columns={TRIAL55_VARIANT: PUBLIC55_CONTROL})
    for origin, stop in origins.items():
        part = frame.loc[frame.origin.eq(origin)]
        if (part.empty or not part.time_step.between(origin + 1, stop).all()
                or int(part.time_step.min()) != origin + 1 or int(part.time_step.max()) != stop):
            raise ValueError(f"Trial55 control horizon changed at origin {origin}")
    return frame


def fit_origin(train: pd.DataFrame, full_columns: list[str], origin: int,
               stop: int, stage: str, control: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    history = train.loc[train.time_step.le(origin)]
    future = train.loc[train.time_step.between(origin + 1, stop)]
    columns = historical_columns(history, full_columns, origin)
    status = label_status(history.label)
    weights, factor = training_weights_40(history.time_step, status, origin)
    share = float(weights[status == 2].sum() / weights.sum())
    original_class_counts = np.bincount(status, minlength=3).astype(int)
    original_class_weight_mass = np.bincount(status, weights=weights, minlength=3).astype(float)
    raw_hgb_weights_mean = float(np.mean(weights))
    if (len(columns) != FEATURE_COUNT or int(history.time_step.max()) != origin
            or not np.array_equal(np.unique(status), [0, 1, 2])
            or not np.isclose(share, UNKNOWN_SHARE, rtol=0, atol=1e-12)):
        raise ValueError("Historical columns, three hard statuses, or mass-40 weighting changed")
    fitted = model(config())
    fitted.fit(history[columns], status, sample_weight=weights)
    if fitted.n_iter_ != 600:
        raise ValueError("HGB did not reach the fixed 600 iterations")
    challenger = conditional_score(fitted, future, columns)
    model_path = MODEL_DIR / f"{stage}_{origin}_{stop}_model.joblib"
    if model_path.exists():
        raise RuntimeError(f"Trial81 model already exists; preserve {model_path}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    source_hash = sha(PROJECT_ROOT / "src" / f"{PREFIX}.py")
    joblib.dump({"model": fitted, "base_columns": columns, "columns": columns,
                 "config": config(), "stage": stage, "origin": origin,
                 "forecast_end": stop, "raw_hgb_weights_mean": raw_hgb_weights_mean,
                 "original_class_counts": original_class_counts.tolist(),
                 "original_class_weight_mass": original_class_weight_mass.tolist(),
                 "unknown_weight_share": share, "unknown_weight_factor": float(factor),
                 "source_sha256": source_hash, "plan_sha256": sha(PLAN_PATH)}, model_path)
    restored = joblib.load(model_path)
    replay = conditional_score(restored["model"], future, columns)
    if (restored["columns"] != columns or restored["config"] != config()
            or restored["plan_sha256"] != sha(PLAN_PATH)
            or not np.allclose(replay, challenger, rtol=0, atol=1e-12)):
        raise ValueError("Saved Trial81 model replay changed")

    known = future.loc[future.label.notna(), KEY[1:]].copy()
    known[CHALLENGER] = challenger[future.label.notna().to_numpy()]
    base = control.loc[control.origin.eq(origin), KEY + [PUBLIC55_CONTROL]]
    if base.duplicated(KEY).any():
        raise ValueError("Trial55 control forecast keys are not unique")
    aligned = known[KEY[1:]].merge(base[KEY[1:]+[PUBLIC55_CONTROL]], on=KEY[1:],
                                    how="left", validate="one_to_one", sort=False, indicator=True)
    if (len(aligned) != len(known) or not aligned._merge.eq("both").all()
            or not np.array_equal(aligned.txId.to_numpy(), known.txId.to_numpy())
            or not np.array_equal(aligned.time_step.to_numpy(), known.time_step.to_numpy())
            or not np.array_equal(aligned.label.to_numpy(), known.label.to_numpy())):
        raise ValueError("Trial55 control population, order, periods, or labels changed")
    known[PUBLIC55_CONTROL] = aligned[PUBLIC55_CONTROL].to_numpy(dtype=float)
    known[BLEND] = blend_score(known[PUBLIC55_CONTROL], known[CHALLENGER])
    known.insert(0, "origin", origin)
    diagnostics = {
        "stage": stage, "origin": origin, "forecast_end": stop,
        "history_rows": len(history), "future_rows": len(future),
        "known_forecast_rows": len(known), "history_max_step": int(history.time_step.max()),
        "base_columns": columns, "columns": columns, "feature_count": len(columns),
        "known_licit": int((status == 0).sum()),
        "known_illicit": int((status == 1).sum()), "unlabelled": int((status == 2).sum()),
        "original_class_counts": original_class_counts.tolist(),
        "original_class_weight_mass": original_class_weight_mass.tolist(),
        "raw_hgb_weights_mean": raw_hgb_weights_mean,
        "unknown_weight_share": share, "unknown_weight_factor": float(factor),
        "base_config": config(), "iteration_count": int(fitted.n_iter_),
        "source_sha256": source_hash, "plan_sha256": sha(PLAN_PATH),
        "row_triplication": False, "label_smoothing_alpha": 0.0,
        "control": "exact keyed PUBLICBEST Trial55/1003-2 saved forecast",
        "model_file": str(model_path.relative_to(PROJECT_ROOT)),
        "model_sha256": sha(model_path),
    }
    return known, diagnostics


def run_stage(train, full_columns, origins, stage):
    control = load_trial55_stage(stage, origins)
    parts, diagnostics = [], []
    for origin, stop in origins.items():
        part, detail = fit_origin(train, full_columns, origin, stop, stage, control)
        expected = train.loc[train.time_step.between(origin + 1, stop) & train.label.notna(), "txId"]
        if not np.array_equal(part.txId.to_numpy(), expected.to_numpy()):
            raise ValueError("Known validation rows or order changed")
        parts.append(part)
        diagnostics.append(detail)
        print(f"Completed {stage} {origin}->{stop}: hard-status mass40, 31 leaves", flush=True)
    return pd.concat(parts, ignore_index=True), diagnostics


def evaluate(predictions, stage):
    rows = []
    for origin, frame in predictions.groupby("origin", sort=False):
        slices = {"first7": frame.loc[frame.time_step.le(origin + 7)]}
        if int(frame.time_step.max()) == int(origin) + 14:
            slices.update(full14=frame, second7=frame.loc[frame.time_step.gt(origin + 7)])
        for slice_name, part in slices.items():
            for variant in (PUBLIC55_CONTROL, CHALLENGER, BLEND):
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
        delta = pivot[variant] - pivot[PUBLIC55_CONTROL]
        row = {"stage": stage, "variant": variant, "control_variant": PUBLIC55_CONTROL}
        for name, keys in groups.items():
            row[f"{name}_mean_auc"] = float(pivot.loc[keys, variant].mean())
            row[f"{name}_mean_delta"] = float(delta.loc[keys].mean())
            row[f"{name}_control_auc"] = float(pivot.loc[keys, PUBLIC55_CONTROL].mean())
        row["worst_slice_delta"] = float(delta.min())
        row["qualified"] = stage_gate_pass(stage, row)
        rows.append(row)
    return pd.DataFrame(rows)


def export_candidate(plan, train, test_features, sample, full_columns, winner):
    expected = sample["index"].to_numpy()
    control = pd.read_csv(TRIAL55_CANDIDATE)
    if (len(expected) != 15329 or not np.array_equal(test_features["index"].to_numpy(), expected)
            or not np.array_equal(control["index"].to_numpy(), expected)):
        raise ValueError("PUBLICBEST Trial55 candidate or sample order changed")
    history = train.loc[train.time_step.le(35)]
    columns = historical_columns(history, full_columns, 35)
    status = label_status(history.label)
    weights, factor = training_weights_40(history.time_step, status, 35)
    share = float(weights[status == 2].sum() / weights.sum())
    original_class_counts = np.bincount(status, minlength=3).astype(int)
    original_class_weight_mass = np.bincount(status, weights=weights, minlength=3).astype(float)
    raw_hgb_weights_mean = float(np.mean(weights))
    if (len(columns) != FEATURE_COUNT or not np.isclose(share, UNKNOWN_SHARE, rtol=0, atol=1e-12)
            or not np.array_equal(np.unique(status), [0, 1, 2])):
        raise ValueError("Final history columns, hard statuses, or mass40 changed")
    fitted = model(config())
    fitted.fit(history[columns], status, sample_weight=weights)
    if fitted.n_iter_ != 600:
        raise ValueError("Final HGB did not reach 600 iterations")
    challenger = conditional_score(fitted, test_features, columns)
    model_path = MODEL_DIR / "final_status_mass40_leaf31_lr03_iter600_through_35.joblib"
    if model_path.exists() or CANDIDATE_PATH.exists() or COMPONENTS_PATH.exists():
        raise RuntimeError("Trial81 final artifacts already exist; preserve them")
    source_hash = sha(PROJECT_ROOT / "src" / f"{PREFIX}.py")
    joblib.dump({"model": fitted, "base_columns": columns, "columns": columns,
                 "config": config(), "classes": [0, 1, 2], "origin": 35,
                 "stage": "final", "forecast_end": "test",
                 "raw_hgb_weights_mean": raw_hgb_weights_mean,
                 "original_class_counts": original_class_counts.tolist(),
                 "original_class_weight_mass": original_class_weight_mass.tolist(),
                 "unknown_weight_share": share, "unknown_weight_factor": float(factor),
                 "source_sha256": source_hash, "plan_sha256": sha(PLAN_PATH)}, model_path)
    replay = conditional_score(joblib.load(model_path)["model"], test_features, columns)
    if not np.allclose(replay, challenger, rtol=0, atol=1e-12):
        raise ValueError("Saved final Trial81 model replay changed")
    control_score = control.target.to_numpy(dtype=float)
    blended = blend_score(control_score, challenger)
    selected = challenger if winner == CHALLENGER else blended
    if not np.isfinite(selected).all() or ((selected < 0) | (selected > 1)).any() or np.std(selected) <= 0:
        raise ValueError("Invalid Trial81 candidate scores")
    pd.DataFrame({"index": expected, PUBLIC55_CONTROL: control_score,
                  CHALLENGER: challenger, BLEND: blended}).to_pickle(COMPONENTS_PATH)
    pd.DataFrame({"index": expected, "target": selected}).to_csv(CANDIDATE_PATH, index=False)
    exported = pd.read_csv(CANDIDATE_PATH)
    if (len(exported) != len(expected) or not np.array_equal(exported["index"].to_numpy(), expected)
            or not np.allclose(exported.target, selected, rtol=0, atol=1e-15)):
        raise ValueError("Trial81 candidate order or values changed")
    final_detail = {
        "stage": "final", "origin": 35, "forecast_end": "test",
        "history_rows": len(history), "history_max_step": int(history.time_step.max()),
        "base_columns": columns, "columns": columns, "feature_count": len(columns),
        "original_class_counts": original_class_counts.tolist(),
        "original_class_weight_mass": original_class_weight_mass.tolist(),
        "raw_hgb_weights_mean": raw_hgb_weights_mean,
        "unknown_weight_share": share, "unknown_weight_factor": float(factor),
        "base_config": config(), "iteration_count": int(fitted.n_iter_),
        "source_sha256": source_hash, "plan_sha256": sha(PLAN_PATH),
        "model_file": str(model_path.relative_to(PROJECT_ROOT)),
        "model_sha256": sha(model_path),
    }
    return {"winner": winner, "candidate_file": str(CANDIDATE_PATH.relative_to(PROJECT_ROOT)),
            "candidate_sha256": sha(CANDIDATE_PATH), "candidate_rows": len(exported),
            "candidate_std": float(exported.target.std()), "final_fit_diagnostic": final_detail,
            "final_model_file": str(model_path.relative_to(PROJECT_ROOT)),
            "final_model_sha256": sha(model_path),
            "final_components_file": str(COMPONENTS_PATH.relative_to(PROJECT_ROOT)),
            "final_components_sha256": sha(COMPONENTS_PATH), "final_base_columns": columns,
            "final_columns": columns, "final_origin": 35, "final_stage": "final",
            "final_forecast_end": "test", "final_raw_hgb_weights_mean": raw_hgb_weights_mean,
            "final_original_class_counts": original_class_counts.tolist(),
            "final_original_class_weight_mass": original_class_weight_mass.tolist(),
            "final_unknown_weight_share": share, "final_unknown_weight_factor": float(factor),
            "final_history_max_step": int(history.time_step.max()), "kaggle_upload": False}


def main():
    plan = json.loads(PLAN_PATH.read_text())
    verify_inputs(plan)
    authorization = json.loads(ROOT_AUTHORIZATION.read_text())
    metadata = {"trial_number": 81, "variant": PREFIX,
                "source_sha256": sha(PROJECT_ROOT / "src" / f"{PREFIX}.py"),
                "plan_sha256": sha(PLAN_PATH), "input_sha256": plan["input_sha256"],
                "original_gate_passed": False, "confirmation_run": False,
                "confirmation_gate_passed": False, "same_variant_passed_both_stages": False,
                "qualified_variants_both_stages": [], "submission_created": False,
                "kaggle_upload": False, "final_model_fit_performed": False,
                "primary_control": "PUBLICBEST Trial55/1003-2, user-reported public ROC-AUC .95255",
                "trial80_closeout_sha256": sha(REPORT_DIR / f"{TRIAL80_PREFIX}_parent_verification.json")}
    train, test_features, sample, groups = load_features()
    full_columns = groups["regularized_two_hop"]
    if len(full_columns) != 182:
        raise ValueError("182-column source feature set changed")
    original, diagnostics = run_stage(train, full_columns, ORIGINS, "original")
    original.to_pickle(REPORT_DIR / f"{PREFIX}_original_predictions.pkl")
    original_results = evaluate(original, "original")
    original_summary = summarize(original_results, "original")
    original_results.to_csv(REPORT_DIR / f"{PREFIX}_original_results.csv", index=False)
    original_summary.to_csv(REPORT_DIR / f"{PREFIX}_original_summary.csv", index=False)
    metadata["original_gate_passed"] = bool(original_summary.qualified.any())
    result_frames, summary_frames = [original_results], [original_summary]
    if metadata["original_gate_passed"]:
        confirmation, confirm_diagnostics = run_stage(train, full_columns, CONFIRMATION, "confirmation")
        confirmation.to_pickle(REPORT_DIR / f"{PREFIX}_confirmation_predictions.pkl")
        confirmation_results = evaluate(confirmation, "confirmation")
        confirmation_summary = summarize(confirmation_results, "confirmation")
        confirmation_results.to_csv(REPORT_DIR / f"{PREFIX}_confirmation_results.csv", index=False)
        confirmation_summary.to_csv(REPORT_DIR / f"{PREFIX}_confirmation_summary.csv", index=False)
        diagnostics += confirm_diagnostics
        result_frames.append(confirmation_results)
        summary_frames.append(confirmation_summary)
        metadata["confirmation_run"] = True
        metadata["confirmation_gate_passed"] = bool(confirmation_summary.qualified.any())
        winners = (set(original_summary.loc[original_summary.qualified, "variant"])
                   & set(confirmation_summary.loc[confirmation_summary.qualified, "variant"]))
        metadata["qualified_variants_both_stages"] = sorted(winners)
        metadata["same_variant_passed_both_stages"] = bool(winners)
        if winners:
            winner = confirmation_summary.loc[confirmation_summary.variant.isin(winners)].sort_values(
                ["full14_mean_auc", "second7_mean_auc"], ascending=False).iloc[0].variant
            export_metadata = export_candidate(plan, train, test_features, sample, full_columns, winner)
            diagnostics.append(export_metadata.pop("final_fit_diagnostic"))
            metadata.update(export_metadata)
            metadata["submission_created"] = True
            metadata["final_model_fit_performed"] = True
    metadata["fit_diagnostics"] = diagnostics
    (REPORT_DIR / f"{PREFIX}_fit_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    pd.concat(result_frames, ignore_index=True).to_csv(REPORT_DIR / f"{PREFIX}_results.csv", index=False)
    pd.concat(summary_frames, ignore_index=True).to_csv(REPORT_DIR / f"{PREFIX}_summary.csv", index=False)
    (REPORT_DIR / f"{PREFIX}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: metadata[key] for key in
                      ("original_gate_passed", "confirmation_run", "submission_created", "candidate_file")},
                     indent=2))


if __name__ == "__main__":
    main()
