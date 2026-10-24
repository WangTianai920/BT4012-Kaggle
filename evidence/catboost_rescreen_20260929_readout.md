# Controlled CatBoost re-screen — 29 September 2026

Status: complete locally; **no candidate qualified**. No final test fit,
submission CSV, Kaggle upload, or public score was produced. The current
user-reported best remains 0929 - 1 at public ROC-AUC 0.94997.

## What changed

The 12 September CatBoost screen used one depth-7, 800-tree configuration
on an older 176-feature set, with and without recency weighting. This trial
used the current 172 features and three predeclared configurations, each
read at 400 and 800 trees. The five historical inner windows selected a
configuration using only labels available by each outer training cutoff.
The resulting choice was the feature-subsampled configuration at four of
five origins and the shallow configuration at origin 24. It was fitted
once at each of the same five frozen forecast origins used in recent trials.

The CatBoost scores received the same fixed label-free period alignment
as 0929-1. Standalone CatBoost and fixed 2.5%/5% probability blends with
0929-1 were evaluated. The old 0916-2 and current 0929-1 predictions
were reproduced exactly by transaction ID. No validation labels entered
model fitting, score alignment, or configuration selection for their own
outer window.

## Frozen results

| Strategy | Mean full 14-step AUC | Mean second-seven AUC | Mean primary-seven AUC | Mean rolling-seven AUC |
| --- | ---: | ---: | ---: | ---: |
| 0929-1 control | 0.950328 | 0.918676 | 0.986120 | 0.977134 |
| Selected CatBoost, aligned | 0.885020 | 0.822519 | 0.965870 | 0.965085 |
| Fixed 2.5% CatBoost blend | 0.949615 | 0.917329 | 0.986159 | 0.976907 |
| Fixed 5% CatBoost blend | 0.948914 | 0.916098 | 0.986095 | 0.976685 |

The closest blend gained 0.000039 on the primary seven-step mean but lost
0.000713 on the full fourteen-step mean, 0.001346 on the difficult second
seven steps, and 0.000227 on the rolling mean. Its worst individual-slice
loss was 0.007587. It failed the predeclared decision rule.

The hard step-24 diagnostic remained unfavorable. CatBoost's within-step
AUC was 0.594223 when trained through 21, versus 0.825275 for 0929-1;
even the 2.5% blend reduced it to 0.823120. Training through 14 or 17
also yielded a substantial step-24 loss. Feature subsampling improved inner
selection results but did not supply the missing later-period fraud ranking.
This is a measured failure of these three CatBoost shapes, not a general
proof that every CatBoost configuration is inferior.

## Audit and conclusion

The trial generated 30 inner scores, 60 outer metric records, and five
configuration-selection diagnostics. All 95 records were added to
`experiment_log.csv`; a second registration added zero. Four independent
tests passed, recomputing every outer AUC, checking earlier-window
selection, frozen controls, blend arithmetic, chronology, and log
uniqueness. The forecasts cover 38,804 labelled historical rows. These
windows overlap and have been reused many times, so even a small local
advantage would have been exploratory.

**Decision:** do not submit the CatBoost version or continue small blend
adjustments in this family. Retain 0929-1 as the reported public best.
