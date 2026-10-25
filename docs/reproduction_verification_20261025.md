# Final submission reproduction verification

On 25 October 2026, the updated `e1303010.ipynb` executed top-to-bottom in a newly created Python 3.10.11 environment on macOS arm64. All pinned packages installed from `requirements.txt`, and `pip check` reported no broken requirements. All eight code cells completed without errors.

The four raw competition CSVs were supplied through local `data/`. Feature and model artifacts were rebuilt locally; no archived predictions or model weights were supplied to the notebook. Both final submissions were fitted and inferred through all test rows before the outputs were compared with the author's original CSVs.

| Selected submission | Rows | Maximum absolute difference from original | Result |
|---|---:|---:|---|
|1019-6|15,329|0.0|Exact match, including CSV hash|
|0929-1|15,329|2.6756374893466273e-14|Pass, within 1e-5 tolerance|

The notebook took approximately 439 seconds. Both CSVs passed shared validation before writing and after read-back: exact `index,target` schema, unique IDs in template order, 15,329 rows, finite nonconstant probabilities in [0,1]. The output receipt marked both requested models complete. These checks verify output integrity and reproduction, not hidden-test ROC-AUC.

Eight lightweight tests passed, covering malformed targets, row/ID/order errors, default/override paths, separate output files, optional reference comparisons and imports with XGBoost deliberately blocked. A fresh-environment issue was repaired by making a historical experiment's XGBoost import lazy. Neither selected model uses XGBoost; its settings and all final-model recipes, blend weights, feature choices and seeds were preserved.

The detailed machine-readable record is `reproduction_verification_20261025.json`. Earlier 24 October records are retained as historical evidence. The notebook's inference route was executed end-to-end; the command-line route uses the same recipes and shared I/O helper, with its argument interface checked separately.

No competition records, generated predictions, fitted models or caches were added to Git. Dataset and generated output locations are excluded by `.gitignore`.
