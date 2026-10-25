# BT4012 Fraudulent Bitcoin Transaction Detection

Wang TianAi · A0288096L · notebook identifier **e1303010**

Repository: https://github.com/WangTianai920/BT4012-Kaggle

## Reproduce both selected final submissions

The main notebook, **`e1303010.ipynb`**, regenerates **both** predictions selected for the final private leaderboard:

| Selected submission | Approach | Generated CSV |
|---|---|---|
| **1019-6** | Three-status HGB teacher, soft temporal weighting and five-seed directed graph residual | `outputs/1019-6.csv` |
| **0929-1** | Binary HGB, historical neighbour-risk specialists and period score alignment | `outputs/0929-1.csv` |

Both routes below perform data preprocessing, graph feature engineering, final model fitting, **inference on every test transaction**, prediction validation and CSV export. No downloaded model weights, archived submission predictions, hidden test labels or files outside this repository and the four supplied competition inputs are required. All computation runs locally. The model settings, blend weights and random seeds are fixed.

## 1 Place the competition data in data/

Download the four files from the competition and arrange them relative to the repository root:

```text
BT4012-Kaggle/
├── e1303010.ipynb
├── requirements.txt
├── run_models.py
├── src/
├── final/
└── data/
    ├── train.csv
    ├── test.csv
    ├── txs_edgelist.csv
    └── sample_submission.csv
```

The visible `data/` folder contains placement instructions only. The dataset is intentionally **not uploaded to GitHub**. The default input location is this repository's **`data/`** directory. Do not change column names or sample-submission ID order. Training uses periods 1–35; test inference covers periods 36–49. Unknown training labels remain separate from known licit/illicit outcomes.

An alternative local folder is optional: set `BT4012_DATA_DIR` before starting the notebook, or pass `--data-dir` to the command-line runner. An explicit command-line argument overrides the environment variable. Relative paths are interpreted from the repository root, not the terminal's current directory.

## 2 Install dependencies

Use **Python 3.10** and run these commands from the repository root:

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. `requirements.txt` lists the dependencies; separate citations for libraries are not required. The pinned dependencies are retained to minimise numerical differences from the submitted recipes. Historical XGBoost experiments may additionally need a system OpenMP runtime; neither final submission loads XGBoost or requires that runtime.

## 3 Run the notebook or command-line entry point

### Notebook

```sh
python -m jupyterlab e1303010.ipynb
```

Select the Python environment created above, restart its kernel and **run all cells in order**. The notebook contains the complete recipes for both models and exports both CSVs. To execute the notebook without opening JupyterLab:

```sh
python -m jupyter nbconvert --execute --to notebook --inplace e1303010.ipynb --ExecutePreprocessor.timeout=1800
```

### Command line

The default command also regenerates both selected submissions:

```sh
python run_models.py
```

Equivalent explicit command:

```sh
python run_models.py --data-dir data --model both
```

For an alternative input directory:

```sh
python run_models.py --data-dir /path/to/competition-data --model both
```

Single-model runs remain available with `--model 1019-6` or `--model 0929-1`, but use **`both`** or the default for assessment. Leave `src/` and `final/` intact: the notebook imports these supporting modules. Training and feature regeneration take several minutes on a CPU; no model checkpoints or graph caches need to be supplied.

## 4 Check the generated submissions

A successful complete run produces:

```text
outputs/
├── 1019-6.csv
├── 0929-1.csv
└── reproduction_receipt.json
```

Each CSV contains exactly the columns **`index,target`**, with **15,329** rows in the template's exact ID order. `target` is the model's illicit-risk probability. The shared validator checks unique IDs, finite values in [0,1], nonconstant predictions and row count, then re-reads each saved file to confirm its integrity. The receipt records both models, input/output hashes, elapsed time and `complete: true` only after both requested exports pass.

These output checks do not measure accuracy or ROC-AUC on hidden test labels. Optional reference comparisons are available with `--reference-dir` on the command line or `BT4012_REFERENCE_DIR` in the notebook. They run **after** inference and are not needed to generate either CSV.

You can run the lightweight validation tests without training:

```sh
python -m unittest discover -s tests -v
```

The fresh-environment notebook run on 25 October reproduced both original submissions within tolerance: see [`docs/reproduction_verification_20261025.md`](docs/reproduction_verification_20261025.md) and its JSON record. Results can differ slightly across numerical-library builds; comparisons with the author's original predictions use an absolute tolerance of 1e-5.

## Report and historical evidence

- `report/e1303010.pdf` is the report and includes the repository link. The report's main text is limited to four pages; references are excluded from that limit under the instructor's clarification.
- `evidence/` and `analysis/` contain retained aggregate historical results and selection evidence. **Final model fitting and inference do not depend on these files.**
- `evaluate_binary_predictions.py` computes accuracy, precision, recall, F1, confusion counts, ROC-AUC and average precision for separately supplied labelled validation predictions. Example:

```sh
python evaluate_binary_predictions.py --input validation_predictions.csv --output metrics.csv --group-cols model origin --id-col txId --threshold 0.5
```

That example input contains `model,origin,txId,label,probability`, where label 0=licit and 1=illicit. It is not a required competition input. The saved historical classification table is a retrospective result, not a fresh validation run or a private-leaderboard score. Reused/overlapping validation windows, historical attribute-availability differences and public-feedback selection limit generalisation claims.

Public ROC-AUC values are author-reported: **1019-6=0.95967**, **0929-1=0.94997**. Private scores are unknown. See `AI_USAGE.md` for AI assistance disclosure.

Raw data, generated predictions, fitted states, caches, local environments and credentials are excluded by `.gitignore`. Nothing automatically uploads to Kaggle or pushes to GitHub.
