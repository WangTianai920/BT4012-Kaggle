"""Frozen Trial 35: fixed log-odds blend of incumbent and Trial 20 scores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

from src.competition import DATA_DIR, PROJECT_ROOT, REPORT_DIR

PREFIX = "shrinkage_logodds_blend_20261002"
PLAN_PATH = REPORT_DIR / f"{PREFIX}_plan.json"
TRIAL20 = REPORT_DIR / "shrinkage_status_20261001_metadata.json"
ORIGINS = {14: 28, 21: 35, 17: 24, 24: 31, 28: 35}
CONFIRMATION = {17: 31, 18: 32, 20: 34}
VARIANT = "shrinkage_logodds_75_25"
CHALLENGER = "lr03_iter600"
CLIP = 1e-12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_score(incumbent, challenger):
    a = np.asarray(incumbent, dtype=float)
    b = np.asarray(challenger, dtype=float)
    if (a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all()
            or not np.isfinite(b).all() or ((a < 0) | (a > 1)).any()
            or ((b < 0) | (b > 1)).any()):
        raise ValueError("Components must be aligned finite probabilities")
    return expit(.75 * logit(np.clip(a, CLIP, 1 - CLIP))
                 + .25 * logit(np.clip(b, CLIP, 1 - CLIP)))


def verify_inputs(plan):
    if plan.get("trial_number") != 35 or plan.get("variant") != VARIANT:
        raise ValueError("Frozen Trial 35 plan missing or mismatched")
    if plan.get("weights") != {"incumbent": .75, "trial20_lr03_iter600": .25}:
        raise ValueError("Frozen blend weights changed")
    if plan.get("clip_epsilon") != CLIP:
        raise ValueError("Frozen clip changed")
    for name, digest in plan["input_sha256"].items():
        if sha(PROJECT_ROOT / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    prior = json.loads(TRIAL20.read_text())
    if (prior.get("winner") != CHALLENGER or prior.get("final_iterations") != 600
            or not prior.get("original_gate_passed") or not prior.get("confirmation_run")
            or prior.get("kaggle_upload") is not False):
        raise ValueError("Trial 20 provenance changed")
    if (prior["final_components_sha256"] != plan["input_sha256"]["reports/shrinkage_status_20261001_final_components.pkl"]
            or prior["final_model_sha256"] != plan["input_sha256"]["models/shrinkage_status_20261001/final_lr03_iter600_through_35.joblib"]):
        raise ValueError("Trial 20 final components or model changed")


def load_stage(stage, origins):
    challenger = pd.read_pickle(REPORT_DIR / f"shrinkage_status_20261001_{stage}_predictions.pkl")
    incumbent_name = ("label_status_multiclass_20260929_predictions.pkl" if stage == "original"
                      else "label_status_confirmation_20260929_predictions.pkl")
    incumbent = pd.read_pickle(REPORT_DIR / incumbent_name)
    key = ["origin", "txId", "time_step", "label"]
    if (challenger.duplicated(key).any() or incumbent.duplicated(key).any()
            or CHALLENGER not in challenger or "incumbent" not in challenger
            or "fraud_given_labelled" not in incumbent):
        raise ValueError(f"{stage} saved forecast schema or keys changed")
    aligned = challenger[key].merge(incumbent[key + ["fraud_given_labelled"]],
                                    on=key, how="left", validate="one_to_one", sort=False)
    if (len(aligned) != len(challenger)
            or not np.array_equal(aligned[key].to_numpy(), challenger[key].to_numpy())
            or aligned.fraud_given_labelled.isna().any()
            or not np.array_equal(aligned.fraud_given_labelled.to_numpy(),
                                  challenger.incumbent.to_numpy())):
        raise ValueError(f"{stage} incumbent population, order, or exact control changed")
    expected_origins = set(origins)
    if set(challenger.origin.unique()) != expected_origins:
        raise ValueError(f"{stage} forecast origins changed")
    for origin, stop in origins.items():
        part = challenger.loc[challenger.origin.eq(origin)]
        if part.empty or not part.time_step.between(origin + 1, stop).all() or part.time_step.max() != stop:
            raise ValueError(f"{stage} forecast horizon changed at {origin}")
    out = challenger[key + ["incumbent", CHALLENGER]].copy()
    out[VARIANT] = fixed_score(out.incumbent, out[CHALLENGER])
    return out


def evaluate(predictions, stage):
    rows = []
    for origin, frame in predictions.groupby("origin", sort=False):
        slices = {"first7": frame.loc[frame.time_step.le(origin + 7)]}
        if int(frame.time_step.max()) == int(origin) + 14:
            slices.update(full14=frame, second7=frame.loc[frame.time_step.gt(origin + 7)])
        for name, part in slices.items():
            for variant in ("incumbent", VARIANT):
                rows.append({"stage": stage, "origin": int(origin), "slice": name,
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
    row = {"stage": stage, "variant": VARIANT}
    for name, keys in groups.items():
        delta = pivot.loc[keys, VARIANT] - pivot.loc[keys, "incumbent"]
        row[f"{name}_mean_auc"] = float(pivot.loc[keys, VARIANT].mean())
        row[f"{name}_mean_delta"] = float(delta.mean())
    row["worst_slice_delta"] = float((pivot[VARIANT] - pivot.incumbent).min())
    gates = ([row[f"{name}_mean_delta"] > 0 for name in groups] if stage == "original" else
             [row["full14_mean_delta"] > 0, row["second7_mean_delta"] > 0,
              row["first7_mean_delta"] >= 0])
    row["qualified"] = bool(all(gates) and row["worst_slice_delta"] >= -.001)
    return pd.DataFrame([row])


def export_candidate(plan):
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    incumbent = pd.read_csv(PROJECT_ROOT / "submissions/candidate_label_status_multiclass_20260929.csv")
    standalone = pd.read_csv(PROJECT_ROOT / "submissions/candidate_shrinkage_status_20261001_lr03_iter600.csv")
    components = pd.read_pickle(REPORT_DIR / "shrinkage_status_20261001_final_components.pkl")
    expected = sample["index"].to_numpy()
    if (len(expected) != 15329 or not np.array_equal(incumbent["index"].to_numpy(), expected)
            or not np.array_equal(standalone["index"].to_numpy(), expected)
            or not np.array_equal(components["index"].to_numpy(), expected)):
        raise ValueError("Saved final components or submissions differ from sample order")
    if (not np.allclose(components.incumbent, incumbent.target, rtol=0, atol=1e-15)
            or not np.allclose(components.challenger, standalone.target, rtol=0, atol=1e-15)):
        raise ValueError("Trial 20 final component scores changed")
    score = fixed_score(components.incumbent, components.challenger)
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any() or np.std(score) <= 0:
        raise ValueError("Invalid final blend scores")
    path = PROJECT_ROOT / plan["candidate_output"]
    if path.exists():
        raise ValueError("Trial 35 candidate already exists; preserve the prior export")
    pd.DataFrame({"index": expected, "target": score}).to_csv(path, index=False)
    readback = pd.read_csv(path)
    if (len(readback) != len(sample) or not np.array_equal(readback["index"].to_numpy(), expected)
            or not np.allclose(readback.target, score, rtol=0, atol=1e-15)):
        raise ValueError("Trial 35 export readback changed")
    return {"candidate_file": plan["candidate_output"], "candidate_sha256": sha(path),
            "candidate_rows": len(readback), "candidate_std": float(readback.target.std())}


def main():
    plan = json.loads(PLAN_PATH.read_text())
    verify_inputs(plan)
    metadata = {"plan_sha256": sha(PLAN_PATH),
                "source_sha256": sha(PROJECT_ROOT / "src/shrinkage_logodds_blend_20261002.py"),
                "input_sha256": plan["input_sha256"], "original_gate_passed": False,
                "confirmation_run": False, "submission_created": False, "kaggle_upload": False}
    original = evaluate(load_stage("original", ORIGINS), "original")
    original_summary = summarize(original, "original")
    original.to_csv(REPORT_DIR / f"{PREFIX}_original_results.csv", index=False)
    original_summary.to_csv(REPORT_DIR / f"{PREFIX}_original_summary.csv", index=False)
    metadata["original_gate_passed"] = bool(original_summary.qualified.iloc[0])
    print("ORIGINAL GATE\n" + original_summary.to_string(index=False), flush=True)
    result_frames, summary_frames = [original], [original_summary]
    if metadata["original_gate_passed"]:
        confirmation = evaluate(load_stage("confirmation", CONFIRMATION), "confirmation")
        confirmation_summary = summarize(confirmation, "confirmation")
        confirmation.to_csv(REPORT_DIR / f"{PREFIX}_confirmation_results.csv", index=False)
        confirmation_summary.to_csv(REPORT_DIR / f"{PREFIX}_confirmation_summary.csv", index=False)
        metadata["confirmation_run"] = True
        print("CONFIRMATION GATE\n" + confirmation_summary.to_string(index=False), flush=True)
        result_frames.append(confirmation)
        summary_frames.append(confirmation_summary)
        if bool(confirmation_summary.qualified.iloc[0]):
            metadata.update(export_candidate(plan))
            metadata["submission_created"] = True
    pd.concat(result_frames, ignore_index=True).to_csv(REPORT_DIR / f"{PREFIX}_results.csv", index=False)
    pd.concat(summary_frames, ignore_index=True).to_csv(REPORT_DIR / f"{PREFIX}_summary.csv", index=False)
    (REPORT_DIR / f"{PREFIX}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
