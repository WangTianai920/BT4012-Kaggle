# Period score-level alignment — 29 September 2026

## Outcome

One small, label-free score correction passed the predeclared local gate. The 0.10 setting
improved all four mean ROC-AUC summaries against the frozen 0916-2 control, while its worst
individual slice lost 0.000412, inside the 0.001 limit. A candidate CSV was prepared for review:
submissions/candidate_score_level_alignment_20260929.csv. The student reports submitting it
as 0929 - 1 with public ROC-AUC 0.94997, 0.00037 above the previous reported best, 0916-2.
The private score remains unreported, and the Kaggle page was not independently checked.

| Variant | Full 14-step mean | Second-seven mean | Primary-seven mean | Rolling-seven mean | Worst slice delta | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0916-2 | 0.950210 | 0.918494 | 0.986074 | 0.976823 | 0 | Control |
| 0.10 alignment | **0.950328** | **0.918676** | **0.986120** | **0.977134** | -0.000412 | Pass |
| 0.25 alignment | 0.950402 | 0.918790 | 0.986158 | 0.977541 | -0.001080 | Fail |
| 0.50 alignment | 0.950253 | 0.918449 | 0.986129 | 0.978131 | -0.002394 | Fail |

The 0.10 gains over 0916-2 are 0.000118, 0.000182, 0.000046, and 0.000311 respectively.
The gains are uneven. At origin 14, the second-seven AUC rises from 0.925407 to 0.926182.
At origin 21, the second-seven AUC falls from 0.911581 to 0.911169. The correction therefore
does not solve the known changing fraud signature within individual periods.

## Method

The frozen 0916-2 model was rerun at each historical forecast origin to score every row in
the forecast periods, including transactions with unknown labels. The recomputed scores for
the labelled validation rows reproduced the saved control exactly. Each forecast period's
median log-odds was calculated from all of its scores. A shared reference was the median of
those period medians, weighting periods equally. For a score with log-odds z in period t,
the corrected probability is sigmoid(z - alpha × (median_t - reference)). The setting
alpha = 0.10 was selected from a fixed grid of 0.10, 0.25, and 0.50.

This adjustment preserves transaction order within every period. It only changes comparisons
between periods, so it targets the cross-period part of ROC-AUC. It uses no unknown labels and
does not refit the fraud model. The actual-test candidate applies the same 0.10 rule to all
15,329 test rows in periods 36–49 using the existing 0916-2 submission as its anchor.

## Reliability and verification

These historical windows overlap and have been used repeatedly for model selection. The
gain is small enough that it may not transfer to the public or private leaderboard. More
aggressive settings gain on some means but breach the per-slice loss guardrail; this is not
evidence to tune a nearby value on the same windows. The test population's period score
distribution may also differ from the historical forecast populations.

Independent checks recomputed all 45 reported AUCs, confirmed exact sample column/order and
row count, bounded finite predictions, the candidate's formula and SHA-256, unchanged
within-period rankings, and unique experiment-log IDs. The idempotent registration rerun
added zero rows. The user-reported public best is now 0929 - 1 at 0.94997.
