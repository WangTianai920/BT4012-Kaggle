"""Stage 1 - regenerate every derived (label-free graph / ranking) artefact from raw data.

The raw competition files (data/train.csv, test.csv, txs_edgelist.csv) are turned into:

  data/derived_graph_features.pkl            src.graph_features.build_graph_features
  data/derived_graph_structure_features.pkl  src.graph_structure_features.build_graph_structure_features
  data/derived_time_relative_ranks.pkl       src.time_relative_features.build_time_relative_ranks
      (the three above feed src.combined_challengers.load_features -> 182 tabular/structural columns)
  GRAPH5   (5 directed-DAG chain features)   src.directed_chain_position5_public55_20261005.build_chain_position_features
  node map + deduplicated edges (period-pure weak components, terminal coverage)
                                             src.terminal_coverage4_public99_20261008.build_terminal_coverage_features
  CUT2     (2 articulation-cut features)     src.articulation_cut2_public99_20261008.build_articulation_cut_features
  BLOCK2   (2 biconnected-block features)    src.block_cycle2_public115_20261008.build_block_cycle_features

Nothing in the project's existing data/ or reports/ folders is overwritten: the legacy builders
write to module-level path constants, which are redirected into ``<work>/`` before they are called.
Graph features never use labels; the only label-dependent selection (12 base features for the
predecessor aggregates) uses labelled time steps 1-14 only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def redirect_derived_paths(work: Path) -> None:
    """Point every legacy cache/metadata path at ``work`` so existing files stay untouched."""
    from src import combined_challengers, graph_features, graph_structure_features, time_relative_features
    (work / "data").mkdir(parents=True, exist_ok=True)
    (work / "reports").mkdir(parents=True, exist_ok=True)
    graph_features.DERIVED_GRAPH_PATH = work / "data/derived_graph_features.pkl"
    graph_features.GRAPH_METADATA_PATH = work / "reports/graph_feature_metadata.json"
    graph_features.SELECTED_BASE_FEATURES_PATH = work / "reports/graph_aggregate_base_features.csv"
    graph_features.REPORT_DIR = work / "reports"
    graph_structure_features.DERIVED_PATH = work / "data/derived_graph_structure_features.pkl"
    graph_structure_features.METADATA_PATH = work / "reports/graph_structure_feature_metadata.json"
    graph_structure_features.REPORT_DIR = work / "reports"
    time_relative_features.DERIVED_PATH = work / "data/derived_time_relative_ranks.pkl"
    time_relative_features.METADATA_PATH = work / "reports/time_relative_feature_metadata.json"
    time_relative_features.DRIFT_RANKING_PATH = work / "reports/time_relative_early_drift_ranking.csv"
    time_relative_features.REPORT_DIR = work / "reports"
    # load_features() reads the ranking csv from REPORT_DIR and the rank pickle from PROJECT_ROOT/data.
    combined_challengers.REPORT_DIR = work / "reports"
    combined_challengers.PROJECT_ROOT = work


def regenerate(work: Path, verbose: bool = True) -> dict:
    """Build all derived artefacts under ``work`` and return their paths."""
    from src.competition import load_competition_data
    from src import graph_features, graph_structure_features, time_relative_features
    from src.combined_challengers import load_features
    from src.directed_chain_position5_public55_20261005 import build_chain_position_features
    from src.feature_propagation16_public99_20261007 import attach_graph5
    from src.terminal_coverage4_public99_20261008 import build_terminal_coverage_features
    from src.articulation_cut2_public99_20261008 import build_articulation_cut_features
    from src.block_cycle2_public115_20261008 import build_block_cycle_features
    log = print if verbose else (lambda *a, **k: None)
    work = Path(work)
    redirect_derived_paths(work)
    train, test, edges, sample = load_competition_data()

    log("[regen] tabular-derived caches (graph, graph-structure, within-step ranks)")
    graph_features.build_graph_features(train, test, edges)
    graph_structure_features.build_graph_structure_features(train, test, edges)
    time_relative_features.build_time_relative_ranks(train, test)
    train, test, sample, groups = load_features()      # now reads the regenerated caches in ``work``

    log("[regen] GRAPH5 directed chain-position features (all 203,769 graph nodes)")
    all_ids = np.unique(np.concatenate([train.txId.to_numpy(), test.txId.to_numpy(),
                                        edges.txId1.to_numpy(), edges.txId2.to_numpy()]))
    graph5, _, _ = build_chain_position_features(edges, all_ids)
    graph5.to_pickle(work / "graph5.pkl")

    log("[regen] terminal-coverage node map (period-pure weak components) + deduplicated edges")
    train_g, test_g = attach_graph5(train, test, graph5)
    attributed = pd.concat([train_g[["txId", "time_step"]], test_g[["txId", "time_step"]]], ignore_index=True)
    _, node_map, dedup_edges, _ = build_terminal_coverage_features(
        graph5.txId.to_numpy(dtype=np.int64), edges, attributed,
        expected_components=49, expected_max_size=7880, expect_no_self=True)
    node_map.to_pickle(work / "node_map.pkl")
    dedup_edges.to_pickle(work / "deduplicated_edges.pkl")

    log("[regen] CUT2 articulation-cut features")
    cut2, _, _ = build_articulation_cut_features(node_map, dedup_edges, expected_components=49, expected_max_size=7880)
    cut2.to_pickle(work / "cut2.pkl")

    log("[regen] BLOCK2 biconnected-block features")
    block2, _, _ = build_block_cycle_features(node_map, dedup_edges, expected_components=49, expected_max_size=7880)
    block2.to_pickle(work / "block2.pkl")
    return dict(graph5=work / "graph5.pkl", node_map=work / "node_map.pkl",
                edges=work / "deduplicated_edges.pkl", cut2=work / "cut2.pkl", block2=work / "block2.pkl")


def compare_to_existing(work: Path) -> dict:
    """Return {artefact: max abs difference or 'identical'} vs the project's frozen pickles."""
    from src.block_cycle2_public115_20261008 import GRAPH5_PATH, CUT2_PATH, GRAPH_FEATURES_PATH
    pairs = {
        "derived_graph_features": (work / "data/derived_graph_features.pkl", ROOT / "data/derived_graph_features.pkl"),
        "derived_graph_structure_features": (work / "data/derived_graph_structure_features.pkl", ROOT / "data/derived_graph_structure_features.pkl"),
        "derived_time_relative_ranks": (work / "data/derived_time_relative_ranks.pkl", ROOT / "data/derived_time_relative_ranks.pkl"),
        "graph5": (work / "graph5.pkl", GRAPH5_PATH),
        "node_map": (work / "node_map.pkl", ROOT / "reports/terminal_coverage4_public99_20261008_graph_node_map.pkl"),
        "deduplicated_edges": (work / "deduplicated_edges.pkl", ROOT / "reports/terminal_coverage4_public99_20261008_deduplicated_edges.pkl"),
        "cut2": (work / "cut2.pkl", CUT2_PATH),
        "block2": (work / "block2.pkl", GRAPH_FEATURES_PATH),
    }
    out = {}
    for name, (new_p, old_p) in pairs.items():
        a, b = pd.read_pickle(new_p), pd.read_pickle(old_p)
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            out[name] = f"SHAPE/COLUMN MISMATCH {a.shape} vs {b.shape}"
            continue
        worst = 0.0
        for c in a.columns:
            x, y = a[c].to_numpy(), b[c].to_numpy()
            if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
                d = np.nanmax(np.abs(x.astype(float) - y.astype(float))) if len(x) else 0.0
                worst = max(worst, float(d))
            elif not np.array_equal(x, y):
                worst = float("inf")
        out[name] = "identical" if worst == 0 else worst
    return out


if __name__ == "__main__":
    w = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "final/work")
    regenerate(w)
    for k, v in compare_to_existing(w).items():
        print(f"{k:36s} {v}")
