"""Independent no-fit Trial117 cached-score and material-gain audit."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.verify_block_cycle2_public115_20261008 import verify_all as replay_trial116

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/"reports"
P="block_cycle_standalone117_20261008"
PARENT="block_cycle2_public115_20261008"
SOURCE=ROOT/"src"/f"{P}.py"
PLAN=REPORTS/f"{P}_plan.json"
AUTH=REPORTS/f"{P}_root_authorization.json"
SUMMARY=REPORTS/f"{P}_summary.json"
AUDIT=REPORTS/f"{P}_audit.json"
LOG=ROOT/"experiment_log.csv"
PARENT_AUDIT=REPORTS/f"{PARENT}_parent_verification.json"
PARENT_MANIFEST=REPORTS/f"{PARENT}_submission_manifest.json"
PARENT_COMPONENTS=REPORTS/f"{PARENT}_final_components.pkl"
PARENT_CANDIDATE=ROOT/"submissions"/f"candidate_{PARENT}.csv"
CANDIDATE=ROOT/"submissions"/f"candidate_{P}.csv"
SOURCE_SHA="c71567071e3508a99232608c2ef0e6b2e10d082a934c63136db3f55522a91b87"
PLAN_SHA="ec78c2c04b942c554f307faa8b063675ba3fb0d70ec8531bb3ae599bf3dbd4d6"
PREFLIGHT=15737
KEY=["origin","txId","time_step","label"]
CONTROL115="trial115_control"
CONTROL116="blend_75_25_logodds"
STANDALONE="block_cycle2_hgb"
CONTROLS=(CONTROL115,CONTROL116)
ORIGINAL={14:28,21:35,17:24,24:31,28:35}
CONFIRMATION={17:31,18:32,20:34}


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prefix_sha(path:Path,count:int)->str:
    raw=path.read_bytes()
    with io.StringIO(raw.decode("utf-8-sig"),newline="") as stream:
        reader=csv.reader(stream)
        next(reader)
        for _ in range(count):next(reader)
        physical_lines=reader.line_num
    return hashlib.sha256(b"".join(raw.splitlines(keepends=True)[:physical_lines])).hexdigest()


def row_sha(row:dict)->str:
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def check_frozen():
    csv.field_size_limit(20_000_000)
    plan,auth,summary=[json.loads(path.read_text()) for path in (PLAN,AUTH,SUMMARY)]
    if (sha(SOURCE)!=SOURCE_SHA or sha(PLAN)!=PLAN_SHA
            or plan.get("trial_number")!=117 or plan.get("variant")!=P
            or plan.get("source_sha256")!=SOURCE_SHA
            or plan.get("controls")!=list(CONTROLS)
            or plan.get("challenger")!=STANDALONE
            or plan.get("material_gain")!={"full14_mean_auc":.0005,
                                           "second7_mean_auc":.001,
                                           "other_mean_auc":0.,
                                           "worst_individual_slice_auc":-.001}
            or auth.get("authorized") is not True
            or auth.get("source_sha256")!=SOURCE_SHA
            or auth.get("plan_sha256")!=PLAN_SHA
            or auth.get("registry_rows")!=PREFLIGHT
            or auth.get("root_trial117_standalone_material_gain_reviewed") is not True
            or auth.get("trial116_audit_and_prepared_1009_1_acknowledged") is not True
            or auth.get("no_new_fit_no_upload_no_test_labels") is not True
            or summary.get("trial_number")!=117 or summary.get("variant")!=P
            or summary.get("source_sha256")!=SOURCE_SHA
            or summary.get("plan_sha256")!=PLAN_SHA
            or summary.get("new_model_fits")!=0
            or summary.get("graph_reconstruction") is not False
            or summary.get("kaggle_upload") is not False
            or summary.get("independent_holdout_claim") is not False):
        raise AssertionError("Trial117 frozen source/authorization/no-fit contract changed")
    pins=plan.get("input_sha256",{})
    if (len(pins)!=959 or "experiment_log.csv" not in pins
            or plan.get("registry_preflight",{}).get("unique_rows")!=PREFLIGHT
            or len(plan.get("trial116_exact_rows",{}))!=106):
        raise AssertionError("Trial117 frozen pin/predecessor inventory changed")
    for rel,digest in pins.items():
        actual=prefix_sha(LOG,PREFLIGHT) if rel=="experiment_log.csv" else sha(ROOT/rel)
        if actual!=digest:raise AssertionError(f"Trial117 input changed: {rel}")
    with LOG.open(newline="",encoding="utf-8-sig") as stream:rows=list(csv.DictReader(stream))
    ids=[r.get("experiment_id","") for r in rows]
    if len(rows)<PREFLIGHT or not all(ids) or len(ids)!=len(set(ids)):
        raise AssertionError("Trial117 registry preflight lost unique identities")
    by_id={r["experiment_id"]:r for r in rows}
    for eid,digest in plan["trial116_exact_rows"].items():
        if eid not in by_id or row_sha(by_id[eid])!=digest:
            raise AssertionError(f"Trial116 exact registered row changed: {eid}")
    parent=json.loads(PARENT_AUDIT.read_text())
    prepared=json.loads(PARENT_MANIFEST.read_text())
    if (parent.get("trial_number")!=116
            or parent.get("eligible_variants")!=sorted([CONTROL116,STANDALONE])
            or parent.get("saved_three_status_model_replays")!=9
            or parent.get("trial115_control_model_replays")!=9
            or parent.get("public99_control_member_replays")!=27
            or parent.get("metric_rows_recomputed")!=54
            or prepared.get("submission_name")!="1009-1"
            or prepared.get("submission_status")!="prepared_for_manual_submission"
            or prepared.get("selected_variant")!=CONTROL116
            or prepared.get("public_roc_auc") is not None
            or prepared.get("kaggle_upload") is not False):
        raise AssertionError("Trial116 audited control and pending 1009-1 changed")
    return plan,summary


def stage_metrics(stage):
    origins=ORIGINAL if stage=="original" else CONFIRMATION
    raw=pd.read_pickle(REPORTS/f"{PARENT}_{stage}_predictions.pkl")
    needed=KEY+[CONTROL115,CONTROL116,STANDALONE]
    if (not set(needed).issubset(raw.columns) or raw[KEY].duplicated().any()
            or raw[needed].isna().any().any() or set(raw.origin)!=set(origins)
            or not raw.label.isin([0,1]).all()):
        raise AssertionError(f"Trial117 {stage} cache schema/labels changed")
    all_scores=raw[list(CONTROLS)+[STANDALONE]].to_numpy(float)
    if not np.isfinite(all_scores).all() or (all_scores<0).any() or (all_scores>1).any():
        raise AssertionError("Trial117 cached scores are not probabilities")
    records=[]
    for origin,stop in origins.items():
        part=raw.loc[raw.origin.eq(origin)]
        if (part.empty or part.time_step.min()!=origin+1 or part.time_step.max()!=stop
                or not part.time_step.between(origin+1,stop).all()):
            raise AssertionError(f"Trial117 {stage}/{origin} horizon changed")
        slices=[("first7",part.loc[part.time_step.le(origin+7)])]
        if stop==origin+14:
            slices += [("full14",part),("second7",part.loc[part.time_step.gt(origin+7)])]
        for name,subset in slices:
            if subset.empty or subset.label.nunique()!=2:
                raise AssertionError("Trial117 binary AUC slice invalid")
            for variant in (*CONTROLS,STANDALONE):
                records.append({"stage":stage,"origin":origin,"slice":name,"variant":variant,
                                "n":len(subset),"positives":int(subset.label.sum()),
                                "auc":float(roc_auc_score(subset.label,subset[variant])),
                                "average_precision":float(average_precision_score(subset.label,subset[variant]))})
    return pd.DataFrame(records)


def stage_gate(stage,metrics):
    pivot=metrics.pivot(index=["origin","slice"],columns="variant",values="auc")
    if stage=="original":
        groups={"full14":[(14,"full14"),(21,"full14")],
                "second7":[(14,"second7"),(21,"second7")],
                "primary7":[(o,"first7") for o in (14,21,28)],
                "rolling7":[(o,"first7") for o in (17,24)]}
    else:
        groups={"full14":[(o,"full14") for o in CONFIRMATION],
                "second7":[(o,"second7") for o in CONFIRMATION],
                "first7":[(o,"first7") for o in CONFIRMATION]}
    rows=[]
    for control in CONTROLS:
        row={"stage":stage,"challenger":STANDALONE,"control":control}
        for group,keys in groups.items():
            challenger=pivot.loc[keys,STANDALONE]
            reference=pivot.loc[keys,control]
            row[f"{group}_mean_auc"]=float(challenger.mean())
            row[f"{group}_mean_delta"]=float((challenger-reference).mean())
        row["worst_slice_delta"]=float((pivot[STANDALONE]-pivot[control]).min())
        others=[row[f"{g}_mean_delta"] for g in groups if g not in ("full14","second7")]
        other_pass=all(v>0 for v in others) if stage=="original" else all(v>=0 for v in others)
        row["qualified"]=bool(row["full14_mean_delta"]>=.0005 and
                              row["second7_mean_delta"]>=.001 and other_pass and
                              row["worst_slice_delta"]>=-.001)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_csv(path,expected):
    pd.testing.assert_frame_equal(pd.read_csv(path).reset_index(drop=True),
                                  expected.reset_index(drop=True),
                                  check_dtype=False,check_exact=False,rtol=0,atol=1e-12)


def verify_all():
    plan,summary=check_frozen()
    before=sha(PARENT_AUDIT)
    replay=replay_trial116()
    if sha(PARENT_AUDIT)!=before:
        raise AssertionError("Trial116 parent audit changed during no-fit replay")
    if (replay.get("saved_three_status_model_replays")!=9
            or replay.get("trial115_control_model_replays")!=9
            or replay.get("public99_control_member_replays")!=27
            or replay.get("metric_rows_recomputed")!=54):
        raise AssertionError("Trial116 parent model/control replay incomplete")
    original=stage_metrics("original")
    original_gate=stage_gate("original",original)
    compare_csv(REPORTS/f"{P}_original_results.csv",original)
    compare_csv(REPORTS/f"{P}_original_gates.csv",original_gate)
    original_pass=bool(original_gate.qualified.all())
    if summary.get("original_gate_passed") is not original_pass:
        raise AssertionError("Trial117 original material gate changed")
    metrics_count=len(original);gate_count=len(original_gate)
    final={"candidate_created":False}
    if original_pass:
        confirmation=stage_metrics("confirmation")
        confirmation_gate=stage_gate("confirmation",confirmation)
        compare_csv(REPORTS/f"{P}_confirmation_results.csv",confirmation)
        compare_csv(REPORTS/f"{P}_confirmation_gates.csv",confirmation_gate)
        confirm_pass=bool(confirmation_gate.qualified.all())
        if (summary.get("confirmation_run") is not True
                or summary.get("confirmation_gate_passed") is not confirm_pass
                or summary.get("same_variant_passed_both_stages") is not confirm_pass):
            raise AssertionError("Trial117 confirmation material gate changed")
        metrics_count+=len(confirmation);gate_count+=len(confirmation_gate)
        if confirm_pass:
            components=pd.read_pickle(PARENT_COMPONENTS)
            sample=pd.read_csv(ROOT/"data"/"sample_submission.csv")
            pending=pd.read_csv(PARENT_CANDIDATE)
            control=pd.read_csv(ROOT/"submissions"/"candidate_articulation_cut2_public99_20261008.csv")
            candidate=pd.read_csv(CANDIDATE)
            ids=sample["index"].to_numpy()
            expected=components[STANDALONE].to_numpy(float)
            if (summary.get("candidate_created") is not True
                    or summary.get("candidate_file")!=str(CANDIDATE.relative_to(ROOT))
                    or summary.get("candidate_sha256")!=sha(CANDIDATE)
                    or summary.get("candidate_rows")!=15329
                    or len(ids)!=15329
                    or sample.columns.tolist()!=["index","target"]
                    or candidate.columns.tolist()!=["index","target"]
                    or not np.array_equal(candidate["index"],ids)
                    or not np.array_equal(components["index"],ids)
                    or not np.array_equal(pending["index"],ids)
                    or not np.array_equal(control["index"],ids)
                    or not np.isfinite(expected).all()
                    or (expected<0).any() or (expected>1).any()
                    or np.max(np.abs(candidate.target.to_numpy()-expected))>1e-15
                    or np.max(np.abs(components[CONTROL116]-pending.target))>1e-15
                    or np.max(np.abs(components[CONTROL115]-control.target))>1e-15):
                raise AssertionError("Trial117 final standalone/sample/control ancestry changed")
            final={"candidate_created":True,"candidate_rows":15329,
                   "candidate_sha256":sha(CANDIDATE)}
        elif CANDIDATE.exists():
            raise AssertionError("Trial117 failed confirmation but wrote candidate")
    elif (summary.get("confirmation_run") is not False or
          summary.get("candidate_created") is not False or CANDIDATE.exists()):
        raise AssertionError("Trial117 failed original but ran confirmation/export")
    audit={"trial_number":117,"variant":P,"source_sha256":SOURCE_SHA,"plan_sha256":PLAN_SHA,
           "frozen_hashes_verified":True,"root_authorization_verified":True,
           "pinned_input_count":len(plan["input_sha256"]),
           "immutable_registry_unique_ids_preflight":PREFLIGHT,
           "trial116_exact_rows_verified":len(plan["trial116_exact_rows"]),
           "parent_audit_sha256":before,
           "parent_saved_three_status_model_replays":9,
           "parent_trial115_control_model_replays":9,
           "parent_public99_control_member_replays":27,
           "cached_metric_rows_recomputed":metrics_count,
           "material_gate_rows_recomputed":gate_count,
           "both_controls_both_stages_verified":bool(final["candidate_created"]),
           "final_artifact_checks":final,
           "new_model_fits":0,"graph_reconstruction":False,
           "independent_holdout_claim":False,"kaggle_upload":False}
    AUDIT.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    return audit


if __name__=="__main__":print(json.dumps(verify_all(),indent=2,sort_keys=True))
