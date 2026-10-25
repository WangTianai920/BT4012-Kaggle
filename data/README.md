# Local competition data

Place these four CSV files directly in this folder before running the notebook or command-line runner:

- `train.csv`
- `test.csv`
- `txs_edgelist.csv`
- `sample_submission.csv`

For example, the training file should be at `data/train.csv`, relative to the repository root. Download these files from the competition; do not upload them to GitHub. Only this README is tracked. All other contents of this folder are excluded by `.gitignore`.

From the repository root, regenerate both selected submissions with:

```sh
python run_models.py --data-dir data --model both
```
