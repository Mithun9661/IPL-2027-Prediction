# IPL 2027 Match Lab

Live app: https://ipl-2027-prediction.onrender.com

A Streamlit match-outcome explorer trained on historical IPL results. User-selected teams, venue, toss winner and toss decision drive the model. Includes head-to-head history, input validation, and transparent evaluation.

## Data and limitations

Cricsheet IPL JSON snapshot, downloaded 2026-09-16: https://cricsheet.org/downloads/ipl_json.zip
Derived metadata in `data/matches.csv`: 1,234 decisive matches, 2008-04-18 through 2026-05-31. Nine matches without recorded decisive winners excluded. Franchise renames normalized; venue strings retained as provided. Source, transformations and archive checksum: `data/source.json`.

This is an experimental historical baseline, not a validated 2027 forecast. It predicts individual matches conditional on a decisive result, not the tournament champion. No fixtures, squad changes, injuries, weather or live-score API are used.

## Model and evaluation

Signed categorical features + DictVectorizer + regularized logistic regression. Encoders and classifier are bundled in `model.joblib`. Team-order reversal gives complementary probabilities; winners are always among the selected teams. Five-year half-life recency weights. Fixed regularization, no holdout tuning.

Chronological evaluation: train on 1,161 matches before 2026; test on 73 matches in 2026. Accuracy **42.5%**, historical win-rate baseline **45.2%**, Brier score **0.256**, log loss **0.706**. The model does not beat the baseline; do not claim reliable predictive advantage. The final serving pipeline is refit on all 1,234 matches. Probabilities are not independently calibrated. Full report: `metrics.json`.

## Run, reproduce and test

Python 3.11:
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
