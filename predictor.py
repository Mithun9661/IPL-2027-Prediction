"""Team-order invariant, pre-match prediction with bundled preprocessing."""
from pathlib import Path
import joblib

ROOT = Path(__file__).parent

def features(team1, team2, venue, toss_winner, toss_decision):
    if team1 == team2:
        raise ValueError('Choose two different teams.')
    if toss_winner not in (team1, team2):
        raise ValueError('Toss winner must be one of the selected teams.')
    if toss_decision not in ('bat', 'field'):
        raise ValueError('Toss decision must be bat or field.')
    toss = 1.0 if toss_winner == team1 else -1.0
    batting = toss if toss_decision == 'bat' else -toss
    return {f'team:{team1}': 1.0, f'team:{team2}': -1.0,
            'toss_won': toss, 'batting_first': batting,
            f'batting_at:{venue}': batting}

def load_bundle():
    return joblib.load(ROOT / 'model.joblib')

def predict(bundle, team1, team2, venue, toss_winner, toss_decision):
    if team1 not in bundle['teams'] or team2 not in bundle['teams']:
        raise ValueError('Selected team is not supported by the trained model.')
    if venue not in bundle['venues']:
        raise ValueError('Select a venue from the historical data.')
    from history import History
    from train import probabilities
    row = History.restore(bundle['history']).row(team1, team2, venue, toss_winner, toss_decision)
    p = float(probabilities(bundle['pipeline'], [row], bundle['model_name'])[0])
    return {team1: p, team2: 1.0 - p}
