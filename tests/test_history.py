import sys,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from history import historical_rows

class HistoryTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([
            dict(date='2020-01-01',team1='A',team2='B',venue='V',toss_winner='A',toss_decision='bat',winner='A'),
            dict(date='2020-01-01',team1='A',team2='C',venue='V',toss_winner='A',toss_decision='field',winner='C'),
            dict(date='2020-01-02',team1='A',team2='B',venue='V',toss_winner='B',toss_decision='bat',winner='B'),
            dict(date='2020-01-03',team1='B',team2='C',venue='V',toss_winner='C',toss_decision='bat',winner='C')])

    def test_future_and_current_outcomes_do_not_leak(self):
        df=self.frame(); original,_=historical_rows(df)
        for i in range(len(df)):
            changed=df.copy();changed.loc[i,'winner']=df.loc[i,'team2'] if df.loc[i,'winner']==df.loc[i,'team1'] else df.loc[i,'team1']
            modified,_=historical_rows(changed)
            for j in range(len(df)):
                if df.loc[j,'date']<=df.loc[i,'date']:self.assertEqual(original[j],modified[j])
        short,_=historical_rows(df.iloc[:3]);self.assertEqual(original[:3],short)

    def test_same_day_has_no_history_and_later_features_update(self):
        rows,state=historical_rows(self.frame())
        self.assertEqual(rows[0]['elo_diff'],0)
        self.assertEqual(rows[1]['elo_diff'],0)
        self.assertNotEqual(rows[2]['elo_diff'],0)
        self.assertEqual(state['form']['A'],[1,0,0])

if __name__=='__main__':unittest.main()
