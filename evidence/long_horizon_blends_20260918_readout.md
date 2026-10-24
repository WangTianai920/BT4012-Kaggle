# Direct LightGBM/XGBoost blends on fourteen-step forecasts — 18 September 2026

All six fixed blends were evaluated directly, without requiring the alternative model to beat HGB on its own. None improved full-fourteen-step or second-half mean AUC versus 0916-2. Keep existing controls; no final test fit, new submission or upload was made.

| Model or addition to 0916-2 | Full14 mean AUC | Second7 mean AUC | Primary7 mean AUC | Rolling7 mean AUC |
| --- | ---: | ---: | ---: | ---: |
| Unchanged 0916-2 | 0.950210 | 0.918494 | 0.986074 | 0.976823 |
| 2.5% LightGBM | 0.949537 | 0.917280 | 0.986012 | 0.976805 |
| 5% LightGBM | 0.948885 | 0.916112 | 0.985932 | 0.976789 |
| 10% LightGBM | 0.947629 | 0.913860 | 0.985765 | 0.976718 |
| 2.5% XGBoost | 0.948803 | 0.916010 | 0.985937 | 0.976582 |
| 5% XGBoost | 0.947577 | 0.913827 | 0.985769 | 0.976347 |
| 10% XGBoost | 0.945524 | 0.910190 | 0.985449 | 0.975996 |
| Standalone LightGBM | 0.914223 | 0.857619 | 0.978891 | 0.974155 |
| Standalone XGBoost | 0.910632 | 0.853194 | 0.977101 | 0.969545 |

The closest blend loses .000672 full14 mean and .001214 second7 mean AUC. Increasing either model's contribution made both metrics progressively worse. This result rejects these fixed configurations/weights, not every possible LightGBM/XGBoost model. It does not support a new candidate from this screen.

## Protocol

Used the previously tested LightGBM15-leaf specification and XGBoost depth5 specification on the existing172 features. Applied recency half-life20 with mean-one weight normalization. Selected tree counts only on the last four historical periods inside each training window, using at most1200 trees and patience60, then refitted on all available labelled history. Feature pruning for the inner fit used history through min(14,inner training cutoff); the outer refit used history through min(14,outer cutoff). This strengthens the inner feature boundary compared with reusing the final feature list in early tree-count selection. No outer validation labels were used for fitting or tree-count choice.

The same frozen origins14→28,21→35,17→24,24→31,28→35 were used. Predictions from each alternative were blended with unchanged0916-2 at2.5%,5%,10%; no additional weight optimization followed. Both alternative standalone scores and all blends are reported. Acceptance required positive full14/second7 mean gains against both0915-4 and0916-2, nonnegative primary/rolling-seven-step mean changes, and no slice loss exceeding.001. There was no standalone-model prerequisite. Both original controls were preserved.

There were ten historical tree-count selections and ten final outer fits. The selected checkpoints and complete historical AUC learning curves are saved in metadata. XGBoost probabilities were promoted to float64 before blending to avoid hidden float32 arithmetic differences. No neighbour-risk models were rebuilt; these are direct complete-prediction blends.

## Verification and records

Seven tests passed, including blend arithmetic with float32 inputs and the existing historical validation checks. Independent audit passed99 rank-based outer AUC checks,10 historical selection checks, all six blend formulas,38,804 forecast rows, source/cache alignment, unchanged controls, summary/gate calculations and25 protected submission/tracker files. LightGBM emitted an eval_set deprecation warning; fitting completed successfully.

Appended109 records to experiment_log.csv:99 outer evaluations and10 historical tree-selection records, with complete learning curves/source references. Prior4,016 records were preserved; total4,125. A repeat registration added zero duplicates. These counts are evaluations/selection records, not109 independent trained models.

All windows overlap or have been reused; local outcomes do not establish private-test performance. The best reported public score remains0916-2 at.94960, supplied by the user and not independently checked here.

Reproduce with `DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/xgboost/lib" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m src.long_horizon_blends_20260918`. Audit: `scripts/verify_long_horizon_blends_20260918.py`; register: `scripts/register_long_horizon_blends_20260918.py`.

The next item in the agreed proposed sequence is supervised TabM, evaluated standalone and in fixed conservative blends. It has not been run in this task.
