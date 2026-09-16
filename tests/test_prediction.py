import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predictor import load_bundle, predict
from streamlit.testing.v1 import AppTest

class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()
        cls.venue = cls.bundle['venues'][0]

    def test_all_team_pairs_are_valid_and_symmetric(self):
        b = self.bundle
        for a in b['active_teams']:
            for c in b['active_teams']:
                if a == c:
                    continue
                for decision in ['bat', 'field']:
                    first = predict(b, a, c, self.venue, a, decision)
                    reverse = predict(b, c, a, self.venue, a, decision)
                    self.assertAlmostEqual(sum(first.values()), 1)
                    self.assertAlmostEqual(first[a], reverse[a], places=10)
                    self.assertTrue(all(0 < p < 1 for p in first.values()))
                    self.assertEqual(set(first), {a, c})

    def test_invalid_inputs_rejected(self):
        a, b = self.bundle['active_teams'][:2]
        for args in [(a,a,self.venue,a,'bat'), (a,b,self.venue,'Invalid','bat'),
                     (a,b,self.venue,a,'invalid'), (a,b,'Unknown',a,'bat')]:
            with self.assertRaises(ValueError):
                predict(self.bundle, *args)

    def test_inputs_change_probabilities(self):
        a, b, c = self.bundle['active_teams'][:3]
        values = [predict(self.bundle,a,b,v,t,d)[a] for v in self.bundle['venues'][:8]
                  for t in (a,b) for d in ('bat','field')]
        self.assertGreater(np.ptp(values), .001)
        self.assertNotAlmostEqual(predict(self.bundle,a,b,self.venue,a,'bat')[a],
                                  predict(self.bundle,a,c,self.venue,a,'bat')[a])

    def test_streamlit_flow_and_stale_results(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertNotIn(app.selectbox(key='team1').value, app.selectbox(key='team2').options)
        app.button(key='predict').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Model leans toward' in s.value for s in app.subheader))
        app.selectbox(key='decision').select('field').run()
        self.assertFalse(any('Model leans toward' in s.value for s in app.subheader))
        app.button(key='predict').click().run()
        self.assertFalse(app.exception)
        app.selectbox(key='team1').select(app.selectbox(key='team2').value).run()
        self.assertFalse(app.exception)
        self.assertNotEqual(app.selectbox(key='team1').value, app.selectbox(key='team2').value)
        app.button(key='predict').click().run()
        self.assertFalse(app.exception)

if __name__ == '__main__':
    unittest.main()
