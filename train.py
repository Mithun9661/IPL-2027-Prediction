"""Hold out the last season, report honest metrics, then refit on all matches."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from predictor import features

ROOT = Path(__file__).parent

def train():
    df = pd.read_csv(ROOT / 'data/matches.csv')
    dates = pd.to_datetime(df.date)
    cutoff = int(df.season.max())
    mask = df.season < cutoff
    rows = [features(r.team1, r.team2, r.venue, r.toss_winner, r.toss_decision) for r in df.itertuples()]
    y = (df.winner == df.team1).astype(int).to_numpy()
    def fit(indices):
        # No intercept and signed features enforce P(A beats B) = 1 - P(B beats A).
        pipe = make_pipeline(DictVectorizer(), LogisticRegression(C=0.3, fit_intercept=False, max_iter=2000, random_state=42))
        age = (dates.iloc[indices].max() - dates.iloc[indices]).dt.days.to_numpy() / 365.25
        pipe.fit([rows[i] for i in indices], y[indices], logisticregression__sample_weight=2 ** (-age / 5))
        return pipe
    train_ids = np.flatnonzero(mask)
    test_ids = np.flatnonzero(~mask)
    assert dates.iloc[train_ids].max() < dates.iloc[test_ids].min()
    evaluation = fit(train_ids)
    probs = evaluation.predict_proba([rows[i] for i in test_ids])[:, 1]
    # Baseline: choose the team with better smoothed historical win rate (training only).
    history = df[mask]
    played = pd.concat([history.team1, history.team2]).value_counts()
    wins = history.winner.value_counts()
    rates = (wins.reindex(played.index, fill_value=0) + 1) / (played + 2)
    base = [int(rates.get(df.iloc[i].team1, .5) >= rates.get(df.iloc[i].team2, .5)) for i in test_ids]
    report = dict(model='Regularized logistic regression with signed team, toss, batting-order and venue features',
        train_matches=len(train_ids), test_matches=len(test_ids), test_season=cutoff,
        accuracy=float(accuracy_score(y[test_ids], probs >= .5)),
        historical_win_rate_baseline_accuracy=float(accuracy_score(y[test_ids], base)),
        brier_score=float(brier_score_loss(y[test_ids], probs)), log_loss=float(log_loss(y[test_ids], probs)),
        final_training_matches=len(df), first_date=df.date.min(), data_through=df.date.max(),
        sklearn_version=sklearn.__version__, data_sha256=hashlib.sha256((ROOT/'data/matches.csv').read_bytes()).hexdigest(),
        protocol='Last season held out, no parameter tuning on holdout; final serving model refit on all matches. Five-year half-life sample weights.',
        limitations='Historical baseline only; no player lineups, injuries, weather, live scores or 2027 results. Probabilities are not independently calibrated. Match winner conditional on a decisive result, not tournament champion.')
    bundle = dict(pipeline=fit(np.arange(len(df))), teams=sorted(set(df.team1) | set(df.team2)),
                  active_teams=sorted(set(df[~mask].team1) | set(df[~mask].team2)), venues=sorted(df.venue.unique()), report=report)
    joblib.dump(bundle, ROOT / 'model.joblib', compress=3)
    (ROOT / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    train()
