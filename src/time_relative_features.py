from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.competition import DATA_DIR, REPORT_DIR, feature_columns


DERIVED_PATH = DATA_DIR / "derived_time_relative_ranks.pkl"
METADATA_PATH = REPORT_DIR / "time_relative_feature_metadata.json"
DRIFT_RANKING_PATH = REPORT_DIR / "time_relative_early_drift_ranking.csv"


def early_time_drift_ranking(train: pd.DataFrame) -> pd.DataFrame:
    """Rank columns without labels or any observation after time step 14."""
    base = feature_columns(train)
    early = train.loc[train["time_step"].between(1, 14), ["time_step", *base]]
    if early.empty or early["time_step"].nunique() != 14:
        raise ValueError("The fixed early selection window must contain time steps 1 through 14")
    step_medians = early.groupby("time_step", sort=True)[base].median()
    between_step_range = step_medians.max() - step_medians.min()
    early_scale = early[base].std(ddof=0).replace(0, np.nan)
    score = (between_step_range / early_scale).fillna(0.0)
    ranking = (
        pd.DataFrame({
            "feature": base,
            "early_median_range_over_early_std": score.reindex(base).to_numpy(dtype=float),
        })
        .sort_values(["early_median_range_over_early_std", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranking["early_window_only"] = True
    return ranking


def build_time_relative_ranks(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Percentile-rank each feature only against rows in its own time step.

    Train's unknown-label rows are included, as are test rows within their own
    steps. No label or row from another time step enters any rank value.
    """
    base = feature_columns(train)
    if feature_columns(test) != base:
        raise ValueError("Train and test feature columns differ")
    train_steps = set(train["time_step"].unique())
    test_steps = set(test["time_step"].unique())
    if train_steps.intersection(test_steps):
        raise ValueError("Train and test time steps overlap; separate-step policy needs review")
    combined = pd.concat(
        [train[["txId", "time_step", *base]], test[["txId", "time_step", *base]]],
        ignore_index=True,
    )
    if combined["txId"].duplicated().any():
        raise ValueError("Transaction IDs must be unique before ranking")

    ranks = combined.groupby("time_step", sort=False)[base].rank(method="average", pct=True)
    ranks = ranks.astype("float32")
    ranks.columns = [f"time_rank_{feature}" for feature in base]
    if not np.isfinite(ranks.to_numpy()).all():
        raise ValueError("Time-relative ranks contain non-finite values")
    if not ranks.ge(0).all().all() or not ranks.le(1).all().all():
        raise ValueError("Time-relative ranks must lie in [0, 1]")

    derived = pd.concat([combined[["txId"]], ranks], axis=1)
    ranking = early_time_drift_ranking(train)
    metadata = {
        "method": "within_time_step_percentile_rank_average_ties",
        "feature_values_use_labels": False,
        "feature_selection_uses_labels": False,
        "feature_selection_window": "all supplied training rows in time steps 1-14 only",
        "feature_selection_score": "range of step medians divided by overall early-window standard deviation",
        "rank_population": "all supplied tabular rows in each individual time step, including unknown-label rows",
        "cross_time_rows_used_in_each_rank": False,
        "test_rows_used_for_test_ranks": True,
        "train_time_steps": sorted(int(value) for value in train_steps),
        "test_time_steps": sorted(int(value) for value in test_steps),
        "feature_count": len(base),
        "top_20_early_drift_features": ranking.head(20)["feature"].tolist(),
        "top_40_early_drift_features": ranking.head(40)["feature"].tolist(),
        "derived_rows": len(derived),
    }
    REPORT_DIR.mkdir(exist_ok=True)
    ranking.to_csv(DRIFT_RANKING_PATH, index=False)
    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    derived.to_pickle(DERIVED_PATH)
    return derived, ranking, metadata


def merge_time_relative_ranks(
    train: pd.DataFrame,
    test: pd.DataFrame,
    derived: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if derived["txId"].duplicated().any():
        raise ValueError("Derived time-rank IDs must be unique")
    train_enriched = train.merge(derived, on="txId", how="left", validate="one_to_one", sort=False)
    test_enriched = test.merge(derived, on="txId", how="left", validate="one_to_one", sort=False)
    if not train_enriched["txId"].equals(train["txId"]):
        raise ValueError("Time-rank merge changed training order")
    if not test_enriched["txId"].equals(test["txId"]):
        raise ValueError("Time-rank merge changed test order")
    added = [column for column in derived.columns if column != "txId"]
    if train_enriched[added].isna().any().any() or test_enriched[added].isna().any().any():
        raise ValueError("Time-rank merge left unmatched transaction IDs")
    return train_enriched, test_enriched
