# BT4012 Fraudulent Bitcoin Transaction Detection

Wang TianAi · A0288096L · notebook identifier e1303010

Repository: https://github.com/WangTianai920/BT4012-Kaggle

## Start here

Run `e1303010.ipynb` from the repository root. It contains data checks, preprocessing, graph feature engineering, training of the top submission 1019-6, prediction integrity checks and CSV export. It also reproduces the second selected model 0929-1. Both `src/` and `final/` are required supporting source code.

## Setup and reproduce

Use Python 3.10 with the pinned dependencies:

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab e1303010.ipynb
```

Download the competition files yourself and place `train.csv`, `test.csv`, `txs_edgelist.csv` and `sample_submission.csv` in a local `data/` folder. Raw competition data are not included in this repository. Alternatively, set `BT4012_DATA_DIR` to their location before starting the notebook. Open the notebook and run all cells in order.

The outputs are `outputs/1019-6.csv`, `outputs/0929-1.csv` and `outputs/reproduction_receipt.json`. Each CSV is checked for 15,329 rows, the exact template ID order, columns `index,target`, and finite, nonconstant probabilities within [0,1]. These checks do not evaluate hidden test labels. The notebook does not require archived predictions to train either final model. Training runs locally and takes several minutes.

Equivalent command-line route:

```sh
python run_models.py --data-dir data --model both
```

To reproduce only the top submission:

```sh
python run_models.py --data-dir data --model 1019-6
```

Optional `BT4012_REFERENCE_DIR` may point to original submission CSVs for a post-training reproduction comparison; it is not required to fit the models. A fresh numerical-library build may cause small differences. The original complete notebook run and original-submission comparison are recorded in `docs/notebook_verification.json`. The later aggregate classification-display cell was executed directly in IPython (a new kernel could not bind a sandboxed local port); the model-training cells were not retrained during packaging. A fresh online dependency installation was not tested during packaging.

## Historical evaluation

The notebook distinguishes recorded historical validation evidence from its final model refit. `evidence/classification_metrics_20261024.csv` contains accuracy, precision, recall, F1, ROC-AUC and confusion counts for reused historical windows. The notebook displays those saved aggregate results. The underlying historical forecast caches are not included, so running the final notebook does not regenerate that table from scratch.

`evaluate_binary_predictions.py` computes metrics for independently supplied labelled validation forecasts. Example:

```sh
python evaluate_binary_predictions.py --input validation_predictions.csv --output metrics.csv --group-cols model origin --id-col txId --threshold 0.5
```

That example input must contain `model,origin,txId,label,probability`, with known labels 0=licit and 1=illicit. Never use a final model's in-sample training predictions as out-of-time validation. Unknown labels cause an error unless `--drop-unknown` is explicitly supplied. The code does not tune the threshold or infer hidden test labels.

Public scores are author-reported: 1019-6=0.95967 and 0929-1=0.94997. Private scores are unknown. Reused temporal windows and repeated public feedback introduce selection bias.

## Files

- `e1303010.ipynb`: main reproducible final-model notebook.
- `requirements.txt`: pinned dependencies.
- `src/`, `final/`: required training and feature code.
- `run_models.py`: equivalent command-line entry point.
- `evaluate_binary_predictions.py`: historical classification metric helper.
- `evidence/`, `analysis/`, `docs/`: retained aggregate results, selection evidence, analysis code and verification records.
- `report/e1303010.pdf`: final report copied unchanged from the author's 24 October PDF; the filename alone was standardised to lowercase.
- `report/make_figures.py`, `report/figures/`: report figure code and exports.
- `AI_USAGE.md`: disclosure of AI assistance.

Raw datasets, generated submission predictions, fitted models, caches, local environments and credentials are excluded. No code automatically uploads to Kaggle or pushes to GitHub.
