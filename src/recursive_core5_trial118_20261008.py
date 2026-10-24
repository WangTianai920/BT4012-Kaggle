"""Trial 118 preparation: label-free recursive core features on supplied graph inputs.

This module does not load competition data or fit a model. The later frozen trial
runner must supply the exact Trial 112 all-node map and directed edge catalog.
"""
from __future__ import annotations

import hashlib
import heapq
import sys

import numpy as np
import pandas as pd


CORE5 = [
    "graph_core_number",
    "graph_core_log1p_peel_wave",
    "graph_core_higher_neighbor_fraction",
    "graph_core_same_neighbor_fraction",
    "graph_core_log1p_shell_size",
]


def _core_and_wave(adjacency: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Remove degree <= k vertices in synchronous waves, increasing k as needed.

    Each vertex is removed once. The shell k at removal equals its k-core
    number; wave zero is the first simultaneous removal at that k.
    """
    n = len(adjacency)
    degree = np.fromiter((len(a) for a in adjacency), dtype=np.int32, count=n)
    original_degree = degree.copy()
    heap = [(int(degree[v]), v) for v in range(n)]
    heapq.heapify(heap)
    active = np.ones(n, dtype=bool)
    core = np.full(n, -1, dtype=np.int32)
    wave = np.full(n, -1, dtype=np.int32)
    removed = 0
    k = 0
    layer = 0
    while removed < n:
        while heap and (not active[heap[0][1]] or heap[0][0] != degree[heap[0][1]]):
            heapq.heappop(heap)
        if not heap:
            raise ValueError("Core heap exhausted before assigning every node")
        if heap[0][0] > k:
            k = heap[0][0]
            layer = 0
        frontier = []
        while heap:
            d, v = heap[0]
            if not active[v] or d != degree[v]:
                heapq.heappop(heap)
            elif d <= k:
                heapq.heappop(heap)
                frontier.append(v)
            else:
                break
        if not frontier:
            raise ValueError("Empty core peel frontier")
        for v in frontier:
            active[v] = False
            core[v] = k
            wave[v] = layer
        removed += len(frontier)
        for v in frontier:
            for u in adjacency[v]:
                if active[u]:
                    degree[u] -= 1
                    if degree[u] < 0:
                        raise ValueError("Negative residual graph degree")
                    heapq.heappush(heap, (int(degree[u]), u))
        layer += 1
    if np.any(core < 0) or np.any(wave < 0) or np.any(core > original_degree):
        raise ValueError("Invalid core or peel-wave assignment")
    return core, wave


def build_recursive_core_features(
    prior_node_map: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    expected_nodes: int | None = None,
    expected_components: int | None = None,
    expected_max_component_nodes: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build five core features on every node of the official graph projection.

    Direction is omitted for this representation. Reciprocal directed edges
    become one undirected edge. No labels or tabular attributes are read.
    """
    required = {"txId", "weak_component", "period"}
    if not required.issubset(prior_node_map.columns):
        raise ValueError("All-node map schema changed")
    if edges.columns.tolist() != ["txId1", "txId2"]:
        raise ValueError("Directed edge catalog schema changed")
    ids = prior_node_map.txId.to_numpy(dtype=np.int64)
    n = len(ids)
    if (n == 0 or (expected_nodes is not None and n != expected_nodes)
            or not np.array_equal(ids, np.sort(np.unique(ids)))):
        raise ValueError("All-node IDs changed or are unsorted")
    component = prior_node_map.weak_component.to_numpy(dtype=np.int32)
    periods = prior_node_map.period.to_numpy(dtype=np.int32)
    component_count = int(component.max()) + 1
    sizes = np.bincount(component, minlength=component_count)
    if (component.min() != 0 or np.any(sizes == 0) or np.any(periods <= 0)
            or (expected_components is not None and component_count != expected_components)
            or (expected_max_component_nodes is not None
                and int(sizes.max()) != expected_max_component_nodes)):
        raise ValueError("Frozen weak-component population changed")
    if any(np.unique(periods[component == c]).size != 1 for c in range(component_count)):
        raise ValueError("Weak component spans multiple periods")
    if (edges[["txId1", "txId2"]].isna().any().any() or edges.duplicated().any()
            or (edges.txId1 == edges.txId2).any()):
        raise ValueError("Duplicate, missing or self-loop directed edge")
    index = pd.Index(ids)
    src = index.get_indexer(edges.txId1.to_numpy(dtype=np.int64))
    dst = index.get_indexer(edges.txId2.to_numpy(dtype=np.int64))
    if ((src < 0).any() or (dst < 0).any()
            or np.any(component[src] != component[dst])
            or np.any(periods[src] != periods[dst])):
        raise ValueError("Edge endpoint or period changed")
    directed_index = np.column_stack((src, dst)).astype(np.int64)
    undirected = np.unique(
        np.column_stack((np.minimum(src, dst), np.maximum(src, dst))).astype(np.int64),
        axis=0,
    )
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for a, b in undirected:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    for neighbors in adjacency:
        neighbors.sort()
    core, wave = _core_and_wave(adjacency)
    degree = np.fromiter((len(a) for a in adjacency), dtype=np.int32, count=n)
    higher = np.zeros(n, dtype=np.int32)
    same = np.zeros(n, dtype=np.int32)
    for v, neighbors in enumerate(adjacency):
        higher[v] = sum(core[u] > core[v] for u in neighbors)
        same[v] = sum(core[u] == core[v] for u in neighbors)
    shell_size = np.empty(n, dtype=np.int32)
    for c in range(component_count):
        members = np.flatnonzero(component == c)
        _, inverse, counts = np.unique(core[members], return_inverse=True, return_counts=True)
        shell_size[members] = counts[inverse]
    values = np.column_stack((
        core,
        np.log1p(wave),
        np.divide(higher, degree, out=np.zeros(n, dtype=float), where=degree > 0),
        np.divide(same, degree, out=np.zeros(n, dtype=float), where=degree > 0),
        np.log1p(shell_size),
    )).astype(np.float32)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Core features contain invalid values")
    features = pd.DataFrame(values, columns=CORE5)
    features.insert(0, "txId", ids)
    node_map = prior_node_map.copy()
    node_map["core_number"] = core
    node_map["core_peel_wave"] = wave
    node_map["core_shell_size"] = shell_size
    for name in CORE5:
        node_map[name] = features[name].to_numpy()
    metadata = {
        "graph_node_count": n,
        "weak_component_count": component_count,
        "max_component_nodes": int(sizes.max()),
        "directed_edges_input": len(directed_index),
        "undirected_edges_after_dedup": len(undirected),
        "deduplicated_edge_index_sha256": hashlib.sha256(directed_index.tobytes()).hexdigest(),
        "undirected_projection_index_sha256": hashlib.sha256(undirected.tobytes()).hexdigest(),
        "cross_period_edge_count": 0,
        "max_core_number": int(core.max()),
        "max_peel_wave": int(wave.max()),
        "zero_core_nodes": int((core == 0).sum()),
        "graph_only_features": CORE5,
    }
    return features, node_map, metadata


def verify_pure_toys() -> None:
    """Small synthetic reference checks; never reads competition artifacts."""
    import networkx as nx

    def check(graph: nx.Graph, periods: dict[int, int] | None = None) -> None:
        nodes = sorted(graph.nodes)
        components = sorted(nx.connected_components(graph), key=lambda c: min(c))
        comp_for = {v: i for i, group in enumerate(components) for v in group}
        period_for = periods or {v: comp_for[v] + 1 for v in nodes}
        node_map = pd.DataFrame({"txId": nodes,
                                 "weak_component": [comp_for[v] for v in nodes],
                                 "period": [period_for[v] for v in nodes]})
        pairs = sorted((min(a, b), max(a, b)) for a, b in graph.edges)
        edge_frame = pd.DataFrame(pairs, columns=["txId1", "txId2"], dtype=np.int64)
        features, mapped, meta = build_recursive_core_features(node_map, edge_frame)
        expected = nx.core_number(graph)
        assert features.txId.tolist() == nodes
        assert mapped.core_number.tolist() == [expected[v] for v in nodes]
        assert meta["undirected_edges_after_dedup"] == len(pairs)
        # Independent synchronous reference, restricted to each k-core shell.
        active = set(nodes)
        expected_wave = {}
        for k in range(max(expected.values(), default=0) + 1):
            layer = 0
            while True:
                frontier = {v for v in active if sum(u in active for u in graph.neighbors(v)) <= k}
                if not frontier:
                    break
                for v in frontier:
                    assert expected[v] == k
                    expected_wave[v] = layer
                active -= frontier
                layer += 1
        assert not active
        assert mapped.core_peel_wave.tolist() == [expected_wave[v] for v in nodes]
        for v in nodes:
            pos = nodes.index(v)
            deg = graph.degree[v]
            h = sum(expected[u] > expected[v] for u in graph.neighbors(v)) / deg if deg else 0
            s = sum(expected[u] == expected[v] for u in graph.neighbors(v)) / deg if deg else 0
            assert np.isclose(features.iloc[pos][CORE5[2]], h, atol=1e-7)
            assert np.isclose(features.iloc[pos][CORE5[3]], s, atol=1e-7)
        if len(edge_frame):
            shuffled = edge_frame.iloc[::-1].copy().reset_index(drop=True)
            again, _, _ = build_recursive_core_features(node_map, shuffled)
            pd.testing.assert_frame_equal(features, again)

    graphs = [nx.empty_graph(3), nx.path_graph(9), nx.cycle_graph(9),
              nx.complete_graph(7), nx.barbell_graph(5, 3), nx.lollipop_graph(6, 4),
              nx.gnp_random_graph(31, .12, seed=118)]
    for graph in graphs:
        check(graph)
    check(nx.path_graph(1201))
    crossed = pd.DataFrame({"txId": [0, 1], "weak_component": [0, 0], "period": [1, 2]})
    one_edge = pd.DataFrame({"txId1": [0], "txId2": [1]})
    try:
        build_recursive_core_features(crossed, one_edge)
    except ValueError:
        pass
    else:
        raise AssertionError("Cross-period component accepted")
    print("Trial 118 pure core toys passed")


if __name__ == "__main__":
    if sys.argv[1:] != ["--fit-free-checks"]:
        raise SystemExit("Only --fit-free-checks is available during preparation")
    verify_pure_toys()
