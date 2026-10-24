"""Stratified historical bootstrap ensembles for the existing risk specialists."""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from src.combined_challengers import load_features
from src.competition import PROJECT_ROOT, REPORT_DIR, load_competition_data
from src.graph_experiments import FOLDS
from src.compact_neighbor_risk_20260916 import EXTRA
from src.refined_historical_neighbor_risk import _generate_block, aggregate_refined_neighbor_risk
from src.historical_neighbor_risk import _fit_first_stage
from src.targeted_hgb_challengers import _model, _variants, _weights

SEEDS = (103, 211, 307, 401, 503)

def bootstrap_positions(meta, seed):
    """Resample inside each historical time/class stratum, preserving its count."""
    if meta.label.isna().any() or not meta.label.isin([0, 1]).all():
        raise ValueError("Bootstrap requires known binary historical labels")
    positions = meta.reset_index(drop=True).groupby(["time_step", "label"], sort=True).indices
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in positions.values()])

def fit_bags(meta, valid, legacy, enhanced, end):
    members = {"legacy": [], "enhanced": []}
    resample_hashes = []
    for seed in SEEDS:
        idx = bootstrap_positions(meta, seed)
        resample_hashes.append(hashlib.sha256(idx.tobytes()).hexdigest())
        sampled = meta.iloc[idx]
        for kind, columns in [("legacy", legacy), ("enhanced", legacy + enhanced)]:
            model = _model()
            model.fit(sampled[columns], sampled.label.astype(int), sample_weight=_weights(sampled.time_step, end, 20.))
            members[kind].append(model.predict_proba(valid[columns])[:, 1])
    ensemble = {}
    for count in (3, 5):
        for kind in members:
            ensemble[f"{kind}_{count}"] = np.mean(members[kind][:count], axis=0)
        ensemble[f"dual_{count}"] = .8 * ensemble[f"legacy_{count}"] + .2 * ensemble[f"enhanced_{count}"]
    diagnostics = dict(resample_hashes=resample_hashes, distinct_bootstraps=len(set(resample_hashes)),
                       legacy_mean_prediction_std=float(np.std(members["legacy"], axis=0).mean()),
                       enhanced_mean_prediction_std=float(np.std(members["enhanced"], axis=0).mean()))
    return ensemble, diagnostics

def main():
    train, test, sample, groups = load_features()
    _, _, edges, _ = load_competition_data()
    labelled = train.loc[train.label.notna()].copy()
    columns, _ = _variants(groups)["regularized_drop_top10_drift"]
    previous = json.loads((REPORT_DIR / "compact_neighbor_risk_20260916_metadata.json").read_text())
    saved = pd.read_csv(REPORT_DIR / "refined_historical_neighbor_risk_oof_predictions.csv")
    results, designs, diagnostics = {}, {}, {}
    for fold in list(FOLDS) + EXTRA:
        name, end = fold["fold"], fold["train_end"]
        print("Preparing " + name, flush=True)
        meta, legacy, enhanced = _generate_block(labelled, train, edges, columns, 20., end-6, end)
        va = train.loc[train.time_step.between(fold["valid_start"], fold["valid_end"])]
        first = _fit_first_stage(labelled.loc[labelled.time_step.le(end)], columns, end, 20.)
        features, lc, ec = aggregate_refined_neighbor_risk(va, edges, first.predict_proba(va[columns])[:, 1])
        assert lc == legacy and ec == enhanced
        valid = labelled.loc[labelled.time_step.between(fold["valid_start"], fold["valid_end"]), ["txId","time_step","label"]].merge(features,on=["txId","time_step"],validate="one_to_one",sort=False)
        singles = {}
        for kind, cols in [("legacy",legacy),("enhanced",legacy+enhanced)]:
            model=_model()
            model.fit(meta[cols],meta.label.astype(int),sample_weight=_weights(meta.time_step,end,20.))
            singles[kind]=model.predict_proba(valid[cols])[:,1]
        base=valid.hist_self_risk.to_numpy(float)
        old=.9*base+.1*singles["legacy"]
        anchor=.975*old+.025*singles["enhanced"]
        if name.startswith("fold_"):
            expected=valid[["txId","time_step","label"]].merge(saved.loc[saved.fold.eq(name)],on=["txId","time_step","label"],validate="one_to_one",sort=False)
            assert np.allclose(anchor,.975*expected.incumbent_0915_4+.025*expected.enhanced_recent,rtol=0,atol=1e-10)
        ensemble,diagnostics[name]=fit_bags(meta,valid,legacy,enhanced,end)
        predictions={"0916-1":anchor,"0915-4":old}
        for kind,p in ensemble.items():
            for weight in (.025,.05,.10):
                predictions[f"add_{kind}_{weight:.3f}"]=(1-weight)*anchor+weight*p
        # Match the incumbent component weights exactly; change only bagging.
        for count in (3,5):
            predictions[f"replace_dual_{count}"]=.8775*base+.0975*ensemble[f"legacy_{count}"]+.025*ensemble[f"enhanced_{count}"]
        results[name]={}
        for variant,p in predictions.items():
            score=dict(auc=float(roc_auc_score(valid.label,p)),ap=float(average_precision_score(valid.label,p)))
            if valid.time_step.eq(24).any():
                mask=valid.time_step.eq(24)
                score["step24_auc"]=float(roc_auc_score(valid.loc[mask,"label"],p[mask]))
            results[name][variant]=score
        assert abs(results[name]["0916-1"]["auc"]-previous["results"][name]["0916-1"]["auc"])<1e-10
        designs[name]=dict(meta_seed_end=end-7,specialist_steps=[end-6,end],validation_steps=[fold["valid_start"],fold["valid_end"]])
        (REPORT_DIR/"bagged_neighbor_risk_20260916_checkpoint.json").write_text(json.dumps(results,indent=2))
        print("Completed "+name,flush=True)
    primary=[f["fold"] for f in FOLDS]; extra=[f["fold"] for f in EXTRA]
    summary=[]
    for variant in results[primary[0]]:
        d={f:results[f][variant]["auc"]-results[f]["0916-1"]["auc"] for f in results}
        weak=min(results[f][variant]["auc"] for f in primary)
        step=results["fold_1"][variant]["step24_auc"]-results["fold_1"]["0916-1"]["step24_auc"]
        passed=(variant not in {"0916-1","0915-4"} and np.mean([d[f] for f in primary])>0
                and weak>min(results[f]["0916-1"]["auc"] for f in primary) and d["fold_1"]>0 and step>0
                and min(d.values())>=-.001 and np.mean([d[f] for f in extra])>=0)
        summary.append(dict(variant=variant,mean_auc=float(np.mean([results[f][variant]["auc"] for f in primary])),
                            weakest_auc=float(weak),step24_delta=float(step),deltas=d,
                            rolling_mean_delta=float(np.mean([d[f] for f in extra])),passes=bool(passed)))
    eligible=[s for s in summary if s["passes"]]
    # Prefer a conservative additive candidate, lower blend weight, then five members.
    eligible.sort(key=lambda s:(not s["variant"].startswith("add_"),
                               float(s["variant"].rsplit("_",1)[1]) if s["variant"].startswith("add_") else 1,
                               "_5_" not in s["variant"],-s["mean_auc"]))
    metadata=dict(control="0916-1",results=results,summaries=summary,temporal_design=designs,
                  bootstrap_seeds=SEEDS,bootstrap_strata=["time_step","label"],bootstrap_diagnostics=diagnostics,
                  current_period_neighbor_labels_used=False,test_labels_used=False,untouched_holdout=False,
                  additional_windows_overlap_previous_validation=True,candidate_created=False,kaggle_upload_performed=False)
    if eligible:
        choice=eligible[0]
        meta,legacy,enhanced=_generate_block(labelled,train,edges,columns,20.,29,35)
        first=_fit_first_stage(labelled,columns,35,20.)
        tf,_,_=aggregate_refined_neighbor_risk(test,edges,first.predict_proba(test[columns])[:,1])
        ens,final_diag=fit_bags(meta,tf,legacy,enhanced,35)
        path=PROJECT_ROOT/"submissions/candidate_dual_neighbor_risk_20260916.csv"
        anchor=pd.read_csv(path); assert anchor["index"].equals(sample["index"])
        name=choice["variant"]
        if name.startswith("add_"):
            kind,weight=name[4:].rsplit("_",1)
            p=(1-float(weight))*anchor.target.to_numpy(float)+float(weight)*ens[kind]
        else:
            count=name.rsplit("_",1)[1]
            p=.8775*tf.hist_self_risk.to_numpy(float)+.0975*ens[f"legacy_{count}"]+.025*ens[f"enhanced_{count}"]
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and len(np.unique(p))>1
        payload=dict(filename="candidate_bagged_neighbor_risk_20260916.csv",selected_variant=name,
                     records=[dict(index=i,target=float(v)) for i,v in zip(sample["index"].tolist(),p)])
        (REPORT_DIR/"bagged_neighbor_risk_20260916_payload.json").write_text(json.dumps(payload,indent=2))
        metadata.update(selected=choice,candidate_payload_created=True,final_bootstrap_diagnostics=final_diag,
                        anchor_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (REPORT_DIR/"bagged_neighbor_risk_20260916_metadata.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print("Selected: "+str(metadata.get("selected","No candidate passed")),flush=True)

if __name__=="__main__":
    main()
