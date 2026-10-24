"""Shared building blocks for the final 1008-5 pipeline (used by run_final.py and validate_temporal.py).

Model in one paragraph
----------------------
Rows are Bitcoin transactions in 49 time steps (train: 1-35 with ~77% unlabelled rows, test: 36-49).
For a forecast origin ``o`` we only ever use rows with time_step <= o for fitting, and score rows later
than o (strict temporal split; feature selection and recency weights are also cut at ``o``):

 1. Features (181 columns): 172 tabular/structural columns pruned using only time steps <= min(14, o)
    (``historical_columns``) + 5 directed-DAG chain features (GRAPH5) + 2 articulation-cut features
    (CUT2) + 2 biconnected-block features (BLOCK2).  The 9 graph features are label-free.
 2. Class imbalance / missing labels: a 3-status target (0 licit, 1 illicit, 2 unlabelled) so the unlabelled
    majority is modelled explicitly instead of being dropped; recency weights with half-life 20 time
    steps; unlabelled mass rescaled to exactly 40% of total weight (``training_weights_40``).
 3. Teacher: HistGradientBoosting (600 iterations, lr .03, 15 leaves, l2=2, max_features=.8, seed 42)
    on the 3 statuses.  Score = p_illicit / (p_licit + p_illicit).
 4. Graph residual: a directed gated-mean GraphSAGE network (width 32) built on the transaction graph
    restricted to the time steps of the partition (history graph = steps 1..o, forecast graph = steps
    o+1..end; no edge ever crosses partitions).  It outputs a tanh-bounded additive correction to the
    teacher's class log-probabilities, trained 60 epochs with AdamW (lr .001, wd .001), seed 42,
    25% attribute-row masking, weighted cross-entropy on the labelled+unlabelled statuses.
    Final score = softmax(teacher_logits + tanh(gnn))[1] / (p0 + p1).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.block_cycle2_public115_20261008 import (BLOCK2, CUT2, GRAPH5, attach_block2, attach_cut2,  # noqa: E402
                                                 attach_graph5, model_config)
from src.combined_challengers import load_features  # noqa: E402
from src.directed_block_gnn120_public117_20261008 import GRAPH9, config as gnn_config, partition  # noqa: E402
from src.horizon_hgb_20260917 import historical_columns  # noqa: E402
from src.label_status_multiclass_20260929 import label_status  # noqa: E402
from src.status_mass40_lr03_iter600_20261003 import training_weights_40  # noqa: E402

WORKER = ROOT / "src/directed_block_gnn120_worker_20261008.py"


def load_inputs(work: Path | None):
    """Return train/test (with the 9 graph columns attached), sample, 187 candidate columns, graph objects.

    ``work`` is the regenerate_features.py output directory, or None to use the project's existing pickles.
    """
    if work is not None:
        from final.regenerate_features import redirect_derived_paths
        redirect_derived_paths(Path(work))
        p = dict(graph5=work / "graph5.pkl", cut2=work / "cut2.pkl", block2=work / "block2.pkl",
                 node_map=work / "node_map.pkl", edges=work / "deduplicated_edges.pkl")
    else:
        from src.block_cycle2_public115_20261008 import CUT2_PATH, GRAPH5_PATH, GRAPH_FEATURES_PATH
        r = ROOT / "reports"
        p = dict(graph5=GRAPH5_PATH, cut2=CUT2_PATH, block2=GRAPH_FEATURES_PATH,
                 node_map=r / "terminal_coverage4_public99_20261008_graph_node_map.pkl",
                 edges=r / "terminal_coverage4_public99_20261008_deduplicated_edges.pkl")
    train, test, sample, groups = load_features()
    g5, c2, b2 = (pd.read_pickle(p[k]) for k in ("graph5", "cut2", "block2"))
    for g, attach in ((g5, attach_graph5), (c2, attach_cut2), (b2, attach_block2)):
        train, test = attach(train, test, g)
    full = list(groups["regularized_two_hop"]) + GRAPH5
    graph9 = g5.merge(c2, on="txId", validate="one_to_one").merge(b2, on="txId", validate="one_to_one")
    return dict(train=train, test=test, sample=sample, full=full, graph9=graph9,
                node_map=pd.read_pickle(p["node_map"]), edges=pd.read_pickle(p["edges"]))


def history_view(inp, origin):
    """History rows (time_step <= origin), 181 columns, 3-status labels and recency/40%-unknown weights."""
    h = inp["train"][inp["train"].time_step <= origin]
    cols = historical_columns(inp["train"], inp["full"], origin) + CUT2 + BLOCK2
    assert len(cols) == 181
    status = label_status(h.label)
    w, _ = training_weights_40(h.time_step, status, origin)
    assert np.isclose(w[status == 2].sum() / w.sum(), .4, atol=1e-12, rtol=0)
    return h, cols, status, w


def fit_teacher(h, cols, status, w):
    """Frozen-config 3-status HGB teacher (identical to block_cycle2_public115 ``fit_one``)."""
    t = HistGradientBoostingClassifier(**model_config())
    t.fit(h[cols], status, sample_weight=w)
    assert t.n_iter_ == 600 and np.array_equal(t.classes_, [0, 1, 2])
    return t


def run_window(inp, origin, end, future, teacher=None, workdir=Path("/tmp"), tag="win"):
    """Fit teacher (unless given) + GNN residual at ``origin`` and score ``future`` (steps origin+1..end).

    Returns (teacher_score, gnn_score, teacher) aligned to the rows of ``future``.
    """
    h, cols, status, w = history_view(inp, origin)
    assert future.time_step.min() > origin and not np.intersect1d(h.txId, future.txId).size
    teacher = teacher or fit_teacher(h, cols, status, w)
    hp = partition(h, range(1, origin + 1), inp["node_map"], inp["edges"], inp["graph9"], cols, status, w)
    fp = partition(future, range(origin + 1, end + 1), inp["node_map"], inp["edges"], inp["graph9"], cols)
    assert hp["periods"].max() <= origin and fp["periods"].min() > origin
    for part in (hp, fp):      # teacher class log-probabilities (zero for graph-only nodes)
        lg = np.log(np.clip(teacher.predict_proba(pd.DataFrame(part["raw"], columns=cols)), 1e-12, 1))
        lg[~part["is_attr"]] = 0.
        part["teacher_logits"] = lg.astype(np.float32)
    job = dict(history=hp, future=fp, columns=cols, graph_columns=GRAPH9, origin=origin, stop=end,
               stage=tag, configuration=gnn_config())
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    ip, op = workdir / f"{tag}_{origin}_{end}_in.joblib", workdir / f"{tag}_{origin}_{end}_out.joblib"
    joblib.dump(job, ip, compress=3)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run([sys.executable, str(WORKER), "fit", str(ip), str(op)], env=env, check=True, cwd=ROOT)
    res = joblib.load(op)
    pos = fp["attributed_positions"]
    p_t = teacher.predict_proba(future[cols])
    teacher_score = p_t[:, 1] / (p_t[:, 0] + p_t[:, 1])
    return teacher_score, res["score"][pos], teacher
