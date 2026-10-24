"""Compute binary report metrics from labelled, out-of-time predictions.

Positive class: illicit=1. Pass probabilities for that class, never rounded labels.
A fixed default threshold of 0.50 is descriptive; this script never optimises it.
For overlapping temporal folds, pass --group-cols to avoid double-counting them.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, roc_auc_score,
                            average_precision_score)


def binary_metrics(y_true, probability, threshold=0.5):
    y = np.asarray(y_true)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or not len(y) or len(y) != len(p):
        raise ValueError('Labels and probabilities must be equal-length, nonempty 1D arrays.')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Known binary labels 0=licit and 1=illicit are required; exclude unknowns explicitly.')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite values in [0,1].')
    if not np.isscalar(threshold) or not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Threshold must be a finite number strictly between 0 and 1.')
    y = y.astype(int)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    both = len(np.unique(y)) == 2
    return {
        'n': int(len(y)), 'positives': int(y.sum()), 'prevalence': float(y.mean()),
        'threshold': float(threshold), 'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, p)) if both else None,
        'average_precision': float(average_precision_score(y, p)) if both else None,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'predicted_positives': tp + fp, 'zero_division_policy': '0',
        'precision_defined': bool(tp + fp), 'recall_defined': bool(tp + fn),
    }


def evaluate_frame(frame, probability_col='probability', label_col='label',
                   group_cols=(), threshold=0.5, id_col=None):
    columns = [probability_col, label_col, *group_cols]
    if id_col:
        columns.append(id_col)
    if not set(columns).issubset(frame):
        raise ValueError(f'Missing columns: {set(columns) - set(frame)}')
    if frame.empty:
        raise ValueError('No rows to evaluate.')
    if group_cols and frame[list(group_cols)].isna().any().any():
        raise ValueError('Grouping columns cannot be missing.')
    groups = frame.groupby(list(group_cols), sort=True) if group_cols else [((), frame)]
    rows = []
    for key, group in groups:
        if id_col and (group[id_col].isna().any() or group[id_col].duplicated().any()):
            raise ValueError('Missing/duplicate transaction IDs within an evaluation group.')
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group_cols, key)),
                     **binary_metrics(group[label_col].to_numpy(),
                                      group[probability_col].to_numpy(), threshold)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label-col', default='label')
    parser.add_argument('--probability-col', default='probability')
    parser.add_argument('--group-cols', nargs='*', default=[])
    parser.add_argument('--id-col')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--drop-unknown', action='store_true',
                        help='Explicitly exclude labels outside {0,1}, including missing values.')
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    excluded = 0
    if args.drop_unknown:
        known = frame[args.label_col].isin([0, 1]); excluded = int((~known).sum())
        frame = frame.loc[known].copy()
    result = evaluate_frame(frame, args.probability_col, args.label_col,
                            args.group_cols, args.threshold, args.id_col)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(json.dumps({'excluded_unknown_rows': excluded, 'output': str(args.output)}))

if __name__ == '__main__':
    main()
