from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.competition import DATA_DIR, REPORT_DIR, feature_columns


DERIVED_GRAPH_PATH = DATA_DIR / "derived_graph_features.pkl"
GRAPH_METADATA_PATH = REPORT_DIR / "graph_feature_metadata.json"
SELECTED_BASE_FEATURES_PATH = REPORT_DIR / "graph_aggregate_base_features.csv"


def select_aggregate_base_features(train: pd.DataFrame, count: int = 12) -> pd.DataFrame:
    """Select a fixed set using only labelled time steps 1–14."""
    selection_data = train.loc[train["label"].notna() & train["time_step"].between(1, 14)]
    target = selection_data["label"].astype(int)
    rows = []
    for feature in feature_columns(train):
        values = selection_data[feature]
        if values.nunique(dropna=False) <= 1:
            directional_auc = 0.5
        else:
            directional_auc = float(roc_auc_score(target, values))
        rows.append(
            {
                "feature": feature,
                "early_window_auc": directional_auc,
                "early_window_auc_distance_from_half": abs(directional_auc - 0.5),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["early_window_auc_distance_from_half", "feature"], ascending=[False, True]
    )
    ranking["selected_for_predecessor_aggregation"] = False
    ranking.iloc[:count, ranking.columns.get_loc("selected_for_predecessor_aggregation")] = True
    REPORT_DIR.mkdir(exist_ok=True)
    ranking.to_csv(SELECTED_BASE_FEATURES_PATH, index=False)
    return ranking


def build_graph_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    edges: pd.DataFrame,
    selected_count: int = 12,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    """Build structural, coverage, and directed predecessor aggregate features."""
    ranking = select_aggregate_base_features(train, selected_count)
    selected_features = ranking.loc[
        ranking["selected_for_predecessor_aggregation"], "feature"
    ].tolist()

    known_nodes = pd.concat(
        [train[["txId", "time_step", *selected_features]], test[["txId", "time_step", *selected_features]]],
        ignore_index=True,
    )
    if known_nodes["txId"].duplicated().any():
        raise ValueError("Combined train and test transaction IDs are not unique")

    node_ids = pd.Index(pd.unique(pd.concat([edges["txId1"], edges["txId2"], known_nodes["txId"]])))
    graph = pd.DataFrame({"txId": node_ids})

    in_degree = edges.groupby("txId2").size().rename("in_degree")
    out_degree = edges.groupby("txId1").size().rename("out_degree")
    unique_predecessors = edges.groupby("txId2")["txId1"].nunique().rename("unique_predecessors")
    unique_successors = edges.groupby("txId1")["txId2"].nunique().rename("unique_successors")
    graph = graph.join(in_degree, on="txId").join(out_degree, on="txId")
    graph = graph.join(unique_predecessors, on="txId").join(unique_successors, on="txId")
    degree_columns = ["in_degree", "out_degree", "unique_predecessors", "unique_successors"]
    graph[degree_columns] = graph[degree_columns].fillna(0).astype("int32")
    graph["total_degree"] = graph["in_degree"] + graph["out_degree"]
    graph["is_source_transaction"] = graph["in_degree"].eq(0).astype("int8")
    graph["is_sink_transaction"] = graph["out_degree"].eq(0).astype("int8")
    graph["incoming_degree_share"] = graph["in_degree"] / graph["total_degree"].clip(lower=1)
    graph["log1p_in_degree"] = np.log1p(graph["in_degree"])
    graph["log1p_out_degree"] = np.log1p(graph["out_degree"])
    graph["log1p_total_degree"] = np.log1p(graph["total_degree"])

    source_attributes = known_nodes.rename(
        columns={"txId": "txId1", "time_step": "source_time", **{f: f"source_{f}" for f in selected_features}}
    )
    target_time = known_nodes[["txId", "time_step"]].rename(columns={"txId": "txId2", "time_step": "target_time"})
    annotated_edges = edges.merge(source_attributes, on="txId1", how="left", validate="many_to_one")
    annotated_edges = annotated_edges.merge(target_time, on="txId2", how="left", validate="many_to_one")
    known_source = annotated_edges["source_time"].notna()
    known_target = annotated_edges["target_time"].notna()
    causal_source = known_source & known_target & (annotated_edges["source_time"] <= annotated_edges["target_time"])
    earlier_source = known_source & known_target & (annotated_edges["source_time"] < annotated_edges["target_time"])
    same_time_source = known_source & known_target & (annotated_edges["source_time"] == annotated_edges["target_time"])
    future_source = known_source & known_target & (annotated_edges["source_time"] > annotated_edges["target_time"])

    coverage = pd.DataFrame(index=pd.Index(known_nodes["txId"], name="txId"))
    coverage = coverage.join(annotated_edges.loc[known_source].groupby("txId2").size().rename("known_predecessor_count"))
    coverage = coverage.join(annotated_edges.loc[causal_source].groupby("txId2").size().rename("causal_predecessor_count"))
    coverage = coverage.join(annotated_edges.loc[earlier_source].groupby("txId2").size().rename("earlier_predecessor_count"))
    coverage = coverage.join(annotated_edges.loc[same_time_source].groupby("txId2").size().rename("same_time_predecessor_count"))
    coverage = coverage.join(annotated_edges.loc[future_source].groupby("txId2").size().rename("future_time_predecessor_count"))
    count_columns = list(coverage.columns)
    coverage[count_columns] = coverage[count_columns].fillna(0).astype("int32")

    graph_for_known = graph.set_index("txId").loc[known_nodes["txId"]]
    incoming_denominator = graph_for_known["in_degree"].clip(lower=1)
    coverage["known_predecessor_ratio"] = coverage["known_predecessor_count"] / incoming_denominator
    coverage["causal_predecessor_ratio"] = coverage["causal_predecessor_count"] / incoming_denominator
    coverage["earlier_predecessor_ratio"] = coverage["earlier_predecessor_count"] / incoming_denominator
    coverage["same_time_predecessor_ratio"] = coverage["same_time_predecessor_count"] / incoming_denominator
    coverage["future_time_predecessor_ratio"] = coverage["future_time_predecessor_count"] / incoming_denominator

    causal_edges = annotated_edges.loc[causal_source, ["txId2", *[f"source_{f}" for f in selected_features]]]
    aggregate_source_columns = [f"source_{feature}" for feature in selected_features]
    predecessor_means = causal_edges.groupby("txId2")[aggregate_source_columns].mean()
    predecessor_stds = causal_edges.groupby("txId2")[aggregate_source_columns].std(ddof=0)
    predecessor_means.columns = [column.replace("source_", "pred_mean_") for column in predecessor_means.columns]
    predecessor_stds.columns = [column.replace("source_", "pred_std_") for column in predecessor_stds.columns]
    predecessor_aggregates = predecessor_means.join(predecessor_stds, how="outer")

    derived = known_nodes[["txId"]].copy().set_index("txId")
    derived = derived.join(graph.set_index("txId"), how="left")
    derived = derived.join(coverage, how="left")
    derived = derived.join(predecessor_aggregates, how="left").reset_index()
    if len(derived) != len(known_nodes) or derived["txId"].duplicated().any():
        raise ValueError("Graph feature merge changed the one-row-per-transaction grain")

    structural_columns = [
        "in_degree",
        "out_degree",
        "unique_predecessors",
        "unique_successors",
        "total_degree",
        "is_source_transaction",
        "is_sink_transaction",
        "incoming_degree_share",
        "log1p_in_degree",
        "log1p_out_degree",
        "log1p_total_degree",
    ]
    coverage_columns = list(coverage.columns)
    aggregate_columns = list(predecessor_aggregates.columns)
    feature_groups = {
        "structural": structural_columns,
        "coverage": coverage_columns,
        "predecessor_aggregates": aggregate_columns,
    }
    metadata = {
        "selection_window": "labelled time steps 1-14 only",
        "selected_base_features": selected_features,
        "known_model_transactions": int(len(known_nodes)),
        "all_graph_nodes": int(len(node_ids)),
        "extra_graph_nodes_without_tabular_rows": int(len(node_ids) - len(known_nodes)),
        "edge_rows": int(len(edges)),
        "edges_with_known_source": int(known_source.sum()),
        "edges_with_known_target": int(known_target.sum()),
        "edges_with_source_time_not_later_than_target": int(causal_source.sum()),
        "edges_with_earlier_source": int(earlier_source.sum()),
        "edges_with_same_time_source": int(same_time_source.sum()),
        "edges_with_future_time_source": int(future_source.sum()),
        "feature_groups": feature_groups,
        "graph_feature_values_use_labels": False,
        "aggregate_base_feature_selection_uses_labels": True,
        "aggregate_base_feature_selection_label_scope": "labelled time steps 1-14 only",
    }
    REPORT_DIR.mkdir(exist_ok=True)
    with GRAPH_METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    derived.to_pickle(DERIVED_GRAPH_PATH)
    return derived, feature_groups, metadata


def load_or_build_graph_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    edges: pd.DataFrame,
    rebuild: bool = False,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    if rebuild or not DERIVED_GRAPH_PATH.exists() or not GRAPH_METADATA_PATH.exists():
        return build_graph_features(train, test, edges)
    derived = pd.read_pickle(DERIVED_GRAPH_PATH)
    with GRAPH_METADATA_PATH.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return derived, metadata["feature_groups"], metadata
