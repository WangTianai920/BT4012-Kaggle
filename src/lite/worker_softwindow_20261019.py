"""Probe: train only on recent history with a soft exponential fade. History rows with period < extra['min_period']
(default 22) get their weight multiplied by 0.5 ** ((mp - period) / hl) where hl = extra['soft_hl'] (default 3.0);
rows with period >= mp keep their weight unchanged. If hl <= 0, fall back to the hard cut (weight 0), reproducing
worker_recent exactly. Applies to BOTH a refitted HGB teacher (Trial117 config, rs42, cached) and the Trial120
GatedMean GNN (worker_gm_tunable defaults); unknown-status share renormalised to 40% among the kept rows."""
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import src.lite.worker_gm_tunable_20261014 as G

SEED = 42
_score = G._score
CACHE = Path(__file__).resolve().parents[2] / 'reports/lite_20261013/teacher_cache'
CFG = dict(l2_regularization=2.0, learning_rate=0.03, max_features=0.8, max_iter=600, max_leaf_nodes=15,
           min_samples_leaf=40, early_stopping=False, random_state=42)


def fit(job):
    ex = job.get('extra', {})
    # 'window' = keep the last N history periods (relative to this window's origin); else absolute 'min_period'
    mp = int(job['origin']) - int(ex['window']) + 1 if 'window' in ex else int(ex.get('min_period', 22))
    hl = float(ex.get('soft_hl', 3.0))
    h = dict(job['history']); lab = h['labels']
    w = h['weights'].astype(float).copy()
    if hl <= 0:
        w[h['periods'] < mp] = 0.0
    else:
        old_mask = h['periods'] < mp
        w[old_mask] *= 0.5 ** ((mp - h['periods'][old_mask]) / hl)
    unk = lab == 2; kn = (lab == 0) | (lab == 1); w[unk] *= 0.4 / 0.6 * w[kn].sum() / max(w[unk].sum(), 1e-12)
    path = CACHE / f"{job['stage']}_{job['origin']}_{job['stop']}_softwin{mp}_hl{hl}.joblib"
    if path.exists(): p = joblib.load(path)
    else:
        cols = job['columns']; a = h['is_attr']; tr = (lab[a] >= 0) & (w[a] > 0)
        Xh = pd.DataFrame(h['raw'][a], columns=cols)
        m = HistGradientBoostingClassifier(**CFG).fit(Xh[tr], lab[a][tr], sample_weight=w[a][tr])
        f = job['future']; p = {'h': m.predict_proba(Xh), 'f': m.predict_proba(pd.DataFrame(f['raw'][f['is_attr']], columns=cols))}
        CACHE.mkdir(parents=True, exist_ok=True); joblib.dump(p, path)
    j = dict(job); j['future'] = dict(job['future']); h['weights'] = w; j['history'] = h
    for k, part in (('h', h), ('f', j['future'])):
        t = part['teacher_logits'].copy(); t[part['is_attr']] = np.log(np.clip(p[k], 1e-12, 1)); part['teacher_logits'] = t.astype(np.float32)
    old = G.SEED; G.SEED = SEED
    try: return G.fit(j)
    finally: G.SEED = old
