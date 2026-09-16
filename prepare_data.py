"""Extract match metadata from the official Cricsheet IPL JSON archive."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import pandas as pd

ALIASES = {'Delhi Daredevils': 'Delhi Capitals', 'Kings XI Punjab': 'Punjab Kings',
           'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
           'Rising Pune Supergiants': 'Rising Pune Supergiant'}

def prepare(archive, output):
    rows = []
    excluded = 0
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.endswith('.json'):
                continue
            info = json.loads(z.read(name))['info']
            winner = info['outcome'].get('winner') or info['outcome'].get('eliminator')
            teams = info['teams']
            if winner not in teams or info['toss']['winner'] not in teams:
                excluded += 1
                continue
            canon = lambda x: ALIASES.get(x, x)
            # Keep the provider's venue names; no speculative ground mergers.
            rows.append(dict(match_id=Path(name).stem, date=info['dates'][0],
                season=int(str(info['dates'][0])[:4]), team1=canon(teams[0]), team2=canon(teams[1]),
                venue=info['venue'], toss_winner=canon(info['toss']['winner']),
                toss_decision=info['toss']['decision'], winner=canon(winner)))
    df = pd.DataFrame(rows).sort_values(['date', 'match_id']).reset_index(drop=True)
    assert df.match_id.is_unique and (df.team1 != df.team2).all()
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / 'matches.csv', index=False)
    metadata = dict(source='Cricsheet', source_url='https://cricsheet.org/downloads/',
        archive_url='https://cricsheet.org/downloads/ipl_json.zip', downloaded_on='2026-09-16',
        archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        matches=len(df), excluded_no_decisive_winner=excluded,
        first_date=df.date.min(), last_date=df.date.max(), aliases=ALIASES,
        transformations='Match metadata only; franchise renames normalized. No-result games excluded. Super-over winners included when recorded.')
    (output / 'source.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('archive', type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'data')
    args = p.parse_args()
    prepare(args.archive, args.output)
