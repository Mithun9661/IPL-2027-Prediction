"""Causal features: all matches on a date are featurized before any update."""
from collections import defaultdict
from predictor import features

class History:
    def __init__(self):
        self.elo = defaultdict(lambda: 1500.)
        self.form = defaultdict(list)
        self.venue = defaultdict(lambda: [0, 0])
        self.h2h = defaultdict(lambda: [0, 0])
        self.overall = defaultdict(lambda: [0, 0])

    @staticmethod
    def rate(record):
        wins, games = record
        return (wins + 2) / (games + 4)

    def row(self, a, b, venue, toss_winner, decision):
        row = features(a, b, venue, toss_winner, decision)
        recent = lambda t: (sum(self.form[t][-5:]) + 1) / (len(self.form[t][-5:]) + 2)
        row.update(elo_diff=(self.elo[a]-self.elo[b])/400,
            form5_diff=recent(a)-recent(b),
            venue_rate_diff=self.rate(self.venue[a,venue])-self.rate(self.venue[b,venue]),
            overall_rate_diff=self.rate(self.overall[a])-self.rate(self.overall[b]),
            h2h_diff=self.rate(self.h2h[a,b])-self.rate(self.h2h[b,a]))
        return row

    def update(self, r):
        a,b=r.team1,r.team2
        win=int(r.winner==a)
        expected=1/(1+10**((self.elo[b]-self.elo[a])/400))
        delta=24*(win-expected)
        self.elo[a]+=delta; self.elo[b]-=delta
        for team, other, result in [(a,b,win),(b,a,1-win)]:
            self.form[team].append(result)
            for target,key in [(self.venue,(team,r.venue)),(self.h2h,(team,other)),(self.overall,team)]:
                target[key][0]+=result; target[key][1]+=1

    def snapshot(self):
        return {name: dict(getattr(self,name)) for name in ['elo','form','venue','h2h','overall']}

    @classmethod
    def restore(cls, snapshot):
        h=cls()
        for name,data in snapshot.items():getattr(h,name).update(data)
        return h

def historical_rows(df):
    history=History(); rows=[]
    for date,day in df.groupby('date',sort=True):
        for r in day.itertuples():rows.append(history.row(r.team1,r.team2,r.venue,r.toss_winner,r.toss_decision))
        for r in day.itertuples():history.update(r)
    return rows,history.snapshot()
