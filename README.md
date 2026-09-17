# IPL 2027 Match Lab

Live app: https://ipl-2027-prediction.onrender.com

A Streamlit match-outcome explorer trained on historical IPL results. User-selected teams, venue, toss winner and toss decision drive the model. Includes head-to-head history, input validation, and transparent evaluation.

## Data and limitations

Cricsheet IPL JSON snapshot, downloaded 2026-09-16: https://cricsheet.org/downloads/ipl_json.zip
Derived metadata in `data/matches.csv`: 1,234 decisive matches, 2008-04-18 through 2026-05-31. Nine matches without recorded decisive winners excluded. Franchise renames normalized; venue strings retained as provided. Source, transformations and archive checksum: `data/source.json`.

This is an experimental historical baseline, not a validated 2027 forecast. It predicts individual matches conditional on a decisive result, not the tournament champion. No fixtures, squad changes, injuries, weather or live-score API are used.

## Model and evaluation

Features now include last-five-match form, Elo strength (K=24), smoothed venue win rates, overall win rates and head-to-head history, alongside team, toss and batting-order features. Every date is featurized before processing any of its outcomes, so same-day and future results cannot leak into inputs. UI displays the history snapshot as of the data cutoff, not live form.

Compared original logistic regression, history logistic regression, random forest and gradient boosting using expanding-season validation for 2023, 2024 and 2025 (215 matches). Model weights are frozen during each validation season; historical statistics update after each completed date. Select lowest pooled log loss, with fixed model settings. Gradient boosting selected (log loss 0.698); a constant 50/50 probability has log loss 0.693, so validation does not establish predictive advantage over chance.

Retrospective 2026 evaluation: **50.7% accuracy (37/73)** versus **42.5% (31/73)** for the old model and **45.2%** for the historical win-rate baseline. Brier score **0.248**, log loss **0.689**. 2026 results were already inspected in the previous version: this is NOT a fresh untouched test and does not establish 2027 accuracy. No model selection or parameter tuning used the 2026 comparison in this iteration.

The final serving pipeline is refit on all 1,234 matches, with preprocessing and the final historical state bundled in `model.joblib`. Paired training examples and prediction symmetrization enforce team-order invariance. Probabilities are not independently calibrated. Full comparison and per-season results: `metrics.json`. Previous report: `previous_metrics.json`.

## Run, reproduce and test

Python 3.11 (scikit-learn 1.8.0):
```
pip install -r requirements.txt
python train.py
python -m unittest discover -s tests -v
streamlit run app.py
```

To update data, download the official IPL JSON archive, then run:
```
python prepare_data.py /path/to/ipl_json.zip
python train.py
```
Review the new source manifest, date range and metrics before publishing updated artifacts.

## Deployment

Render free Python service in Singapore. `render.yaml` records the desired configuration. Build: `pip install -r requirements.txt`. Start: `streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true`. Health endpoint: `/_stcore/health`. No API key or database needed. GitHub main auto-deploy enabled.

Original project attribution is retained in `ORIGINAL_README.md` and `ipl.ipynb`. The legacy `ipl_model.pkl` is retained for provenance and is not loaded by the application; its demo inputs and random tournament-winner code are not used. New training uses only bundled historical match metadata and `train.py`.
