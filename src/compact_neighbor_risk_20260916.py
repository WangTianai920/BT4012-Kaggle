"""Compact feature-group ablations with fixed and additional temporal checks."""
from __future__ import annotations
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from src.combined_challengers import load_features
from src.competition import PROJECT_ROOT, REPORT_DIR, load_competition_data
from src.graph_experiments import FOLDS
from src.refined_historical_neighbor_risk import _generate_block, aggregate_refined_neighbor_risk
from src.historical_neighbor_risk import _fit_first_stage
from src.targeted_hgb_challengers import _model, _variants, _weights

WEIGHTS = (0.025, 0.05, 0.10)
EXTRA = [dict(fold="rolling_17", train_end=17, valid_start=18, valid_end=24),
         dict(fold="rolling_24", train_end=24, valid_start=25, valid_end=31)]

def feature_groups(legacy, enhanced):
    tails = [c for c in enhanced if c.endswith(("_q90", "_high90_share", "_high99_share"))]
    disagreement = [c for c in enhanced if c.endswith(("_above_self_share", "_abs_diff_mean", "_mean_minus_self"))]
    direction = [c for c in enhanced if c.startswith("ref_in_out_")]
    return {"legacy": legacy, "tails": legacy + tails, "disagreement": legacy + disagreement,
            "direction": legacy + direction, "tails_disagreement": legacy + tails + disagreement,
            "compact_all": legacy + tails + disagreement + direction}

def scores(frame, p):
    return dict(auc=float(roc_auc_score(frame.label, p)), ap=float(average_precision_score(frame.label, p)))

def main():
    train, test, sample, groups = load_features()
    _, _, edges, _ = load_competition_data()
    labelled = train.loc[train.label.notna()].copy()
    columns, half_life = _variants(groups)["regularized_drop_top10_drift"]
    saved = pd.read_csv(REPORT_DIR / "refined_historical_neighbor_risk_oof_predictions.csv")
    results, designs, schemas = {}, {}, {}
    for fold in list(FOLDS) + EXTRA:
        name, end = fold["fold"], fold["train_end"]
        print(f"Preparing {name}: history 1–{end}; validation {fold['valid_start']}–{fold['valid_end']}", flush=True)
        meta, legacy, enhanced = _generate_block(labelled, train, edges, columns, half_life, end - 6, end)
        valid_all = train.loc[train.time_step.between(fold["valid_start"], fold["valid_end"])]
        history = labelled.loc[labelled.time_step.le(end)]
        first = _fit_first_stage(history, columns, end, half_life)
        features, check_legacy, check_enhanced = aggregate_refined_neighbor_risk(valid_all, edges, first.predict_proba(valid_all[columns])[:, 1])
        assert legacy == check_legacy and enhanced == check_enhanced
        valid = labelled.loc[labelled.time_step.between(fold["valid_start"], fold["valid_end"]), ["txId", "time_step", "label"]].merge(features, on=["txId", "time_step"], validate="one_to_one", sort=False)
        specs = feature_groups(legacy, enhanced)
        specialists = {}
        for group, cols in {**specs, "enhanced": legacy + enhanced}.items():
            model = _model()
            model.fit(meta[cols], meta.label.astype(int), sample_weight=_weights(meta.time_step, end, half_life))
            specialists[group] = model.predict_proba(valid[cols])[:, 1]
            schemas[group] = cols
        base = valid["hist_self_risk"].to_numpy(float)
        old = .9 * base + .1 * specialists["legacy"]
        anchor = .975 * old + .025 * specialists["enhanced"]
        if name.startswith("fold_"):
            stored = valid[["txId", "time_step", "label"]].merge(saved.loc[saved.fold.eq(name)], on=["txId", "time_step", "label"], validate="one_to_one", sort=False)
            assert len(stored) == len(valid)
            for actual, expected in [(base, stored.base_0915_1), (specialists["legacy"], stored.legacy_recent), (specialists["enhanced"], stored.enhanced_recent)]:
                if not np.allclose(actual, expected, rtol=0, atol=1e-10):
                    raise ValueError(f"Control reproduction failed for {name}")
        predictions = {"0916-1": anchor, "0915-4": old}
        for group in specs:
            for weight in WEIGHTS:
                predictions[f"{group}_{weight:.3f}"] = (1 - weight) * anchor + weight * specialists[group]
        results[name] = {variant: scores(valid, p) for variant, p in predictions.items()}
        if valid.time_step.eq(24).any():
            mask = valid.time_step.eq(24)
            for variant, p in predictions.items():
                results[name][variant]["step24_auc"] = float(roc_auc_score(valid.loc[mask, "label"], p[mask]))
        designs[name] = dict(meta_seed_end=end - 7, specialist_steps=[end - 6, end], validation_steps=[fold["valid_start"], fold["valid_end"]])
        (REPORT_DIR / "compact_neighbor_risk_20260916_checkpoint.json").write_text(json.dumps(results, indent=2))
        print(f"Completed {name}", flush=True)
    primary = [f["fold"] for f in FOLDS]
    extra = [f["fold"] for f in EXTRA]
    summaries = []
    for variant in results[primary[0]]:
        mean = np.mean([results[f][variant]["auc"] for f in primary])
        weak = min(results[f][variant]["auc"] for f in primary)
        delta = {f: results[f][variant]["auc"] - results[f]["0916-1"]["auc"] for f in results}
        step_delta = results["fold_1"][variant]["step24_auc"] - results["fold_1"]["0916-1"]["step24_auc"]
        passes = (variant not in {"0916-1", "0915-4"} and np.mean([delta[f] for f in primary]) > 0
                  and weak > min(results[f]["0916-1"]["auc"] for f in primary)
                  and delta["fold_1"] > 0 and step_delta > 0
                  and all(delta[f] >= -.001 for f in results) and np.mean([delta[f] for f in extra]) >= 0)
        group = variant.rsplit("_", 1)[0]
        summaries.append(dict(variant=variant, mean_auc=float(mean), weakest_auc=float(weak), deltas=delta,
                              step24_delta=float(step_delta), rolling_mean_delta=float(np.mean([delta[f] for f in extra])),
                              feature_count=len(schemas[group]) if group in schemas else None, passes=bool(passes)))
    eligible = [s for s in summaries if s["passes"]]
    # Prefer a compact design and lower contribution, rather than maximum reused-fold AUC.
    eligible.sort(key=lambda s: (s["feature_count"], float(s["variant"].rsplit("_", 1)[1]), -s["mean_auc"]))
    metadata = dict(control="0916-1", comparison_control="0915-4", feature_groups=schemas,
                    temporal_design=designs, results=results, summaries=summaries,
                    extra_splits_overlap_previous_validation=True, untouched_holdout=False,
                    test_labels_used=False, current_period_neighbor_labels_used=False,
                    selection="Smallest passing feature set then lowest passing blend contribution",
                    candidate_created=False, kaggle_upload_performed=False)
    if eligible:
        chosen = eligible[0]
        group, weight = chosen["variant"].rsplit("_", 1)
        meta, _, _ = _generate_block(labelled, train, edges, columns, half_life, 29, 35)
        first = _fit_first_stage(labelled, columns, 35, half_life)
        features, legacy, enhanced = aggregate_refined_neighbor_risk(test, edges, first.predict_proba(test[columns])[:, 1])
        cols = feature_groups(legacy, enhanced)[group]
        specialist = _model()
        specialist.fit(meta[cols], meta.label.astype(int), sample_weight=_weights(meta.time_step, 35, half_life))
        anchor_path = PROJECT_ROOT / "submissions/candidate_dual_neighbor_risk_20260916.csv"
        anchor = pd.read_csv(anchor_path)
        assert anchor["index"].equals(sample["index"])
        p = (1 - float(weight)) * anchor.target.to_numpy(float) + float(weight) * specialist.predict_proba(features[cols])[:, 1]
        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError("Invalid candidate")
        payload = dict(selected_variant=chosen["variant"], filename="candidate_compact_neighbor_risk_20260916.csv",
                       records=[dict(index=i, target=float(v)) for i, v in zip(sample["index"].tolist(), p)])
        (REPORT_DIR / "compact_neighbor_risk_20260916_payload.json").write_text(json.dumps(payload, indent=2))
        metadata.update(selected=chosen, candidate_payload_created=True,
                        anchor_sha256=hashlib.sha256(anchor_path.read_bytes()).hexdigest())
    (REPORT_DIR / "compact_neighbor_risk_20260916_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(summaries, indent=2), flush=True)
    print("Selected: " + str(metadata.get("selected", "No candidate passed")), flush=True)

if __name__ == "__main__":
    main()
