"""Select on 2023-2025 rolling validation; report 2026 retrospectively once."""
import hashlib,json
from pathlib import Path
import joblib,numpy as np,pandas as pd,sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,brier_score_loss,log_loss
from history import historical_rows

ROOT=Path(__file__).parent
NAMES=['Original logistic regression','History logistic regression','Random forest','Gradient boosting']

def reverse(row):return {k:-v for k,v in row.items()}

def prepare(rows,name):
    if name==NAMES[0]:
        return [{k:v for k,v in r.items() if k.startswith(('team:','batting_at:')) or k in ['toss_won','batting_first']} for r in rows]
    return rows

def fit(name,rows,y,dates):
    if name in NAMES[:2]:model=LogisticRegression(C=.3,fit_intercept=False,max_iter=2000,random_state=42)
    elif name==NAMES[2]:model=RandomForestClassifier(n_estimators=200,max_depth=5,min_samples_leaf=20,max_features=.8,random_state=42,n_jobs=1)
    else:model=GradientBoostingClassifier(n_estimators=100,learning_rate=.03,max_depth=2,min_samples_leaf=20,random_state=42)
    rows=prepare(rows,name)
    weights=2**(-((dates.max()-dates).dt.days.to_numpy()/365.25)/5)
    pipe=Pipeline([('encoder',DictVectorizer(sparse=False)),('model',model)])
    pipe.fit(rows+[reverse(r) for r in rows], np.r_[y,1-y], model__sample_weight=np.r_[weights/2,weights/2])
    return pipe

def probabilities(pipe,rows,name):
    rows=prepare(rows,name)
    return (pipe.predict_proba(rows)[:,1]+1-pipe.predict_proba([reverse(r) for r in rows])[:,1])/2

def metrics(y,p):
    return dict(matches=len(y),accuracy=float(accuracy_score(y,p>=.5)),brier_score=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p)))

def train():
    df=pd.read_csv(ROOT/'data/matches.csv').sort_values(['date','match_id']).reset_index(drop=True)
    rows,state=historical_rows(df); dates=pd.to_datetime(df.date)
    y=(df.winner==df.team1).astype(int).to_numpy()
    comparison=[]
    for name in NAMES:
        combined_y=[];combined_p=[];folds=[]
        for year in [2023,2024,2025]:
            tr=np.flatnonzero(df.season<year); va=np.flatnonzero(df.season==year)
            pipe=fit(name,[rows[i] for i in tr],y[tr],dates.iloc[tr])
            p=probabilities(pipe,[rows[i] for i in va],name)
            folds.append(dict(season=year,**metrics(y[va],p)))
            combined_y.extend(y[va]);combined_p.extend(p)
        comparison.append(dict(model=name,**metrics(np.array(combined_y),np.array(combined_p)),folds=folds))
    selected=min(comparison,key=lambda r:r['log_loss'])['model']
    tr=np.flatnonzero(df.season<2026);test=np.flatnonzero(df.season==2026)
    pipe=fit(selected,[rows[i] for i in tr],y[tr],dates.iloc[tr])
    p=probabilities(pipe,[rows[i] for i in test],selected)
    evaluation=metrics(y[test],p)
    old=probabilities(fit(NAMES[0],[rows[i] for i in tr],y[tr],dates.iloc[tr]),[rows[i] for i in test],NAMES[0])
    report=dict(model=selected,selection_metric='Lowest pooled 2023-2025 rolling-validation log loss',
        comparison=comparison,train_matches=len(tr),test_matches=len(test),test_season=2026,
        **{k:v for k,v in evaluation.items() if k!='matches'},
        original_model_2026=metrics(y[test],old),
        historical_win_rate_baseline_accuracy=json.loads((ROOT/'previous_metrics.json').read_text())['historical_win_rate_baseline_accuracy'],
        final_training_matches=len(df),first_date=df.date.min(),data_through=df.date.max(),
        sklearn_version=sklearn.__version__,data_sha256=hashlib.sha256((ROOT/'data/matches.csv').read_bytes()).hexdigest(),
        protocol='Models selected only by 2023-2025 rolling validation. Each fold trains on prior seasons. Features use results from earlier dates only; model weights remain frozen during validation, history updates after each date. Final model refit on all matches.',
        evaluation_caveat='2026 results were already inspected in the previous version: this is a retrospective comparison, not a fresh untouched test. Future-season validation is needed.',
        limitations='No player lineups, injuries, weather or live data. Historical features stop at data cutoff. Probabilities not independently calibrated. Match outcomes conditional on decisive result; not tournament winner.')
    bundle=dict(pipeline=fit(selected,rows,y,dates),model_name=selected,history=state,
        teams=sorted(set(df.team1)|set(df.team2)),active_teams=sorted(set(df.iloc[test].team1)|set(df.iloc[test].team2)),venues=sorted(df.venue.unique()),report=report)
    joblib.dump(bundle,ROOT/'model.joblib',compress=3)
    (ROOT/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':train()
