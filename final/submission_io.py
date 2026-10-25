"""Shared relative-path and output-integrity contract for both selected submissions."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILES = ('train.csv', 'test.csv', 'txs_edgelist.csv', 'sample_submission.csv')
SELECTED_MODELS = ('1019-6', '0929-1')
EXPECTED_ROWS = 15329


def repository_path(value):
    """Resolve relative arguments against the repository, independently of shell cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def resolve_data_dir(value=None):
    """Explicit argument, then BT4012_DATA_DIR, then this repository's data/."""
    return repository_path(value if value is not None else os.environ.get('BT4012_DATA_DIR', 'data'))


def check_input_files(data_dir):
    data_dir = Path(data_dir)
    missing = [name for name in INPUT_FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            'Missing competition files: ' + ', '.join(missing) +
            '. Place all four competition CSVs in the repository data/ folder, '
            'or supply --data-dir / BT4012_DATA_DIR. Dataset files are not hosted on GitHub.')
    return {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in INPUT_FILES}


def validate_submission(frame, sample, expected_rows=None):
    """Reject corrupt predictions and any change to sample IDs or their order."""
    if list(frame.columns) != ['index', 'target'] or 'index' not in sample:
        raise ValueError('Expected exact output columns index,target and a sample index column.')
    if not len(sample) or len(frame) != len(sample) or (expected_rows is not None and len(frame) != expected_rows):
        raise ValueError('Prediction row count does not match the submission template.')
    if sample['index'].isna().any() or not sample['index'].is_unique:
        raise ValueError('Sample submission IDs must be present and unique.')
    if frame['index'].isna().any() or not frame['index'].is_unique:
        raise ValueError('Prediction IDs must be present and unique.')
    if not np.array_equal(frame['index'].to_numpy(), sample['index'].to_numpy()):
        raise ValueError('Prediction IDs must match the exact sample submission order.')
    if not pd.api.types.is_numeric_dtype(frame['target']) or pd.api.types.is_bool_dtype(frame['target']):
        raise ValueError('Targets must be numeric probabilities.')
    values = frame['target'].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Targets must be finite probabilities in [0,1].')
    if np.unique(values).size < 2:
        raise ValueError('Constant predictions are not accepted.')
    return {'checks_passed': True, 'rows': len(frame),
            'minimum': float(values.min()), 'maximum': float(values.max())}


def write_submissions(results, sample, output_dir, reference_dir=None, expected_rows=EXPECTED_ROWS):
    """Write and re-read each selected CSV; references are optional post-fit checks."""
    if not results or not set(results).issubset(SELECTED_MODELS):
        raise ValueError('Expected predictions for selected models 1019-6 and/or 0929-1.')
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    # Validate all selected outputs before writing any of them.
    for name, score in results.items():
        values = np.asarray(score)
        if values.ndim != 1:
            raise ValueError('Expected a one-dimensional probability vector.')
        frame = pd.DataFrame({'index': sample['index'].to_numpy(), 'target': values})
        validate_submission(frame, sample, expected_rows)
        frames[name] = frame
    summary = {}
    for name, frame in frames.items():
        path = output_dir / f'{name}.csv'
        frame.to_csv(path, index=False)
        reread = pd.read_csv(path)
        item = validate_submission(reread, sample, expected_rows)
        item['file'] = path.name
        item['output_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        if reference_dir is not None:
            matches = [p for p in Path(reference_dir).glob('*.csv') if ''.join(p.stem.split()) == name]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one optional reference CSV for {name}.')
            reference = pd.read_csv(matches[0])
            validate_submission(reference, sample, expected_rows)
            delta = float(np.max(np.abs(reference['target'] - reread['target'])))
            item.update(max_absolute_difference=delta, acceptance_tolerance=1e-5, accepted=delta <= 1e-5)
            if not item['accepted']:
                raise ValueError(f'{name} differs from its reference by {delta}, exceeding 1e-5.')
        summary[name] = item
    return summary
