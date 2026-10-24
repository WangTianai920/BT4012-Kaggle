from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from src.competition import DATA_DIR, REPORT_DIR


DERIVED_PATH = DATA_DIR / "derived_graph_structure_features.pkl"
METADATA_PATH = REPORT_DIR / "graph_structure_feature_metadata.json"
PAGERANK_DAMPING = 0.85
PAGERANK_MAX_ITER = 200
PAGERANK_TOLERANCE = 1e-10


def _pagerank(adjacency, degree: np.ndarray, reverse: bool) -> tuple[np.ndarray, int]:
    """PageRank of the directed graph, reversing edges only when requested."""
    node_count = adjacency.shape[0]
    rank = np.full(node_count, 1.0 / node_count, dtype=np.float64)
    nonzero = degree > 0
    for iteration in range(1, PAGERANK_MAX_ITER + 1):
        scaled = np.divide(rank, degree, out=np.zeros_like(rank), where=nonzero)
        incoming = adjacency @ scaled if reverse else adjacency.T @ scaled
        updated = (
            (1.0 - PAGERANK_DAMPING) / node_count
            + PAGERANK_DAMPING * (incoming + rank[~nonzero].sum() / node_count)
        )
        if np.abs(updated - rank).sum() < PAGERANK_TOLERANCE:
            return updated, iteration
        rank = updated
    raise ValueError("Graph PageRank did not converge within the fixed iteration limit")


def build_graph_structure_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    """Build label-free graph-only features; do not use any tabular values."""
    known_ids = pd.concat([train["txId"], test["txId"]], ignore_index=True)
    if known_ids.duplicated().any():
        raise ValueError("Train/test transaction IDs must be unique")
    if edges[["txId1", "txId2"]].isna().any().any():
        raise ValueError("Graph edges have missing transaction IDs")
    if edges.duplicated().any() or (edges["txId1"] == edges["txId2"]).any():
        raise ValueError("Duplicate or self-loop edges need a defined counting policy")

    node_ids = pd.Index(pd.unique(pd.concat(
        [edges["txId1"], edges["txId2"], known_ids], ignore_index=True
    )))
    source = node_ids.get_indexer(edges["txId1"])
    target = node_ids.get_indexer(edges["txId2"])
    if (source < 0).any() or (target < 0).any():
        raise ValueError("Edge endpoints were not indexed")
    node_count = len(node_ids)
    adjacency = coo_matrix(
        (np.ones(len(edges), dtype=np.int32), (source, target)),
        shape=(node_count, node_count),
    ).tocsr()
    in_degree = np.asarray(adjacency.sum(axis=0)).ravel().astype(np.float64)
    out_degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)
    total_degree = in_degree + out_degree

    incoming_paths = np.asarray(adjacency.T @ in_degree).ravel()
    outgoing_paths = np.asarray(adjacency @ out_degree).ravel()
    predecessor_degree_sum = np.asarray(adjacency.T @ total_degree).ravel()
    successor_degree_sum = np.asarray(adjacency @ total_degree).ravel()
    predecessor_degree_max = np.zeros(node_count, dtype=np.float64)
    successor_degree_max = np.zeros(node_count, dtype=np.float64)
    np.maximum.at(predecessor_degree_max, target, total_degree[source])
    np.maximum.at(successor_degree_max, source, total_degree[target])

    undirected = (adjacency + adjacency.T).tocsr()
    undirected.data[:] = 1
    undirected.eliminate_zeros()
    undirected_degree = np.diff(undirected.indptr).astype(np.float64)
    triangle_count = np.asarray(
        undirected.multiply(undirected @ undirected).sum(axis=1)
    ).ravel().astype(np.float64) / 2.0
    possible_neighbor_pairs = undirected_degree * (undirected_degree - 1) / 2.0

    forward_rank, forward_iterations = _pagerank(adjacency, out_degree, reverse=False)
    reverse_rank, reverse_iterations = _pagerank(adjacency, in_degree, reverse=True)
    forward_scaled = np.log1p(node_count * forward_rank)
    reverse_scaled = np.log1p(node_count * reverse_rank)

    all_features = pd.DataFrame({
        "txId": node_ids,
        "graph2_incoming_two_hop_paths": np.log1p(incoming_paths),
        "graph2_outgoing_two_hop_paths": np.log1p(outgoing_paths),
        "graph2_predecessor_mean_total_degree": np.divide(
            predecessor_degree_sum, in_degree,
            out=np.zeros(node_count, dtype=np.float64), where=in_degree > 0,
        ),
        "graph2_successor_mean_total_degree": np.divide(
            successor_degree_sum, out_degree,
            out=np.zeros(node_count, dtype=np.float64), where=out_degree > 0,
        ),
        "graph2_predecessor_max_total_degree": np.log1p(predecessor_degree_max),
        "graph2_successor_max_total_degree": np.log1p(successor_degree_max),
        "graph2_triangle_count": np.log1p(triangle_count),
        "graph2_local_clustering": np.divide(
            triangle_count, possible_neighbor_pairs,
            out=np.zeros(node_count, dtype=np.float64), where=possible_neighbor_pairs > 0,
        ),
        "graph2_forward_pagerank": forward_scaled,
        "graph2_reverse_pagerank": reverse_scaled,
        "graph2_pagerank_asymmetry": forward_scaled - reverse_scaled,
    })
    feature_groups = {
        "two_hop_neighborhood": [
            "graph2_incoming_two_hop_paths",
            "graph2_outgoing_two_hop_paths",
            "graph2_predecessor_mean_total_degree",
            "graph2_successor_mean_total_degree",
            "graph2_predecessor_max_total_degree",
            "graph2_successor_max_total_degree",
        ],
        "triangle_motifs": ["graph2_triangle_count", "graph2_local_clustering"],
        "directed_centrality": [
            "graph2_forward_pagerank",
            "graph2_reverse_pagerank",
            "graph2_pagerank_asymmetry",
        ],
    }
    selected = all_features.set_index("txId").loc[known_ids].reset_index()
    added = [column for columns in feature_groups.values() for column in columns]
    if not selected["txId"].equals(known_ids) or not np.isfinite(selected[added].to_numpy()).all():
        raise ValueError("Graph-only feature output has missing/non-finite values or changed row order")
    metadata = {
        "method": "label-free directed two-hop, triangle, and PageRank features",
        "inputs": ["train.txId", "test.txId", "txs_edgelist.csv"],
        "uses_labels": False,
        "uses_tabular_feature_values": False,
        "uses_full_supplied_graph": True,
        "all_graph_nodes": node_count,
        "graph_edges": len(edges),
        "known_transaction_rows": len(selected),
        "nodes_in_undirected_triangles": int((triangle_count > 0).sum()),
        "forward_pagerank_iterations": forward_iterations,
        "reverse_pagerank_iterations": reverse_iterations,
        "feature_groups": feature_groups,
    }
    REPORT_DIR.mkdir(exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    selected.to_pickle(DERIVED_PATH)
    return selected, feature_groups, metadata


def load_or_build_graph_structure_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    edges: pd.DataFrame,
    rebuild: bool = False,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    if rebuild or not DERIVED_PATH.exists() or not METADATA_PATH.exists():
        return build_graph_structure_features(train, test, edges)
    derived = pd.read_pickle(DERIVED_PATH)
    with METADATA_PATH.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_ids = pd.concat([train["txId"], test["txId"]], ignore_index=True)
    if (not derived["txId"].equals(expected_ids)
            or metadata["graph_edges"] != len(edges)):
        raise ValueError("Cached graph-only features do not match current inputs")
    return derived, metadata["feature_groups"], metadata
