# AI assistance disclosure

AI assistants (Claude Code, and NUS SoCLaaS school-hosted models for routine code generation) were used to implement and run
experiments, write and debug code, audit submission files, package the selected pipelines, generate figure code and draft the
report. All modelling decisions — validation design, which experiments to pursue, and the final pair of submissions
(1019-6 and 0929-1) — were made by Wang TianAi, who reviewed the code and results used in the report. Generated code was
tested before use, and several bugs found in review were fixed before any result was relied on.

The selected models use the competition data only. No pretrained tabular foundation models (e.g. TabPFN), external
pretrained weights or AutoML systems were used; the TabM experiment was trained from scratch on the competition data.
Public scores are author-reported; private results are unknown.

The submission notebook e1303010.ipynb was prepared with Codex assistance. NUS SoCLaaS x-test-2 generated a bounded submission-validation helper, which was reviewed before use. Model training executes locally, not on school API compute. The notebook includes both selected models.
