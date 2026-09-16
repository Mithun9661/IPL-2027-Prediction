from pathlib import Path
import pandas as pd
import streamlit as st
from predictor import load_bundle, predict

ROOT = Path(__file__).parent
st.set_page_config(page_title='IPL 2027 | Match Lab', page_icon='🏏', layout='wide')
st.markdown('''<style>
.stApp {background: #0b1220; color: #eef2ff;}
.block-container {max-width: 1100px; padding-top: 2rem;}
[data-testid="stMetric"] {background: #152238; padding: 18px; border-radius: 14px;}
.stButton > button {background: #b7ee61; color: #132016; font-weight: 700; border: 0;}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def resources():
    return load_bundle(), pd.read_csv(ROOT / 'data/matches.csv')

bundle, matches = resources()
report = bundle['report']
st.caption('MATCH LAB  /  IPL 2027')
st.title('Every matchup starts with a question.')
st.write('Explore match outcomes using historical team, venue and toss data.')
st.warning(f"Experimental model · {report['test_season']} holdout accuracy: {report['accuracy']:.1%}. "
           'This model did not outperform the historical win-rate baseline. Treat its estimates as exploratory.')
a, b, c = st.columns(3)
a.metric('Historical matches', f"{len(matches):,}")
b.metric('Data through', report['data_through'])
c.metric('Teams available', len(bundle['active_teams']))

prediction_tab, history_tab, model_tab = st.tabs(['Match prediction', 'Head-to-head', 'Model & data'])
with prediction_tab:
    st.subheader('Set up your match')
    left, right = st.columns(2)
    team1 = left.selectbox('Team 1', bundle['active_teams'], key='team1')
    options = [t for t in bundle['active_teams'] if t != team1]
    team2 = right.selectbox('Team 2', options, key='team2')
    venues = bundle['venues']
    venue = st.selectbox('Venue', venues, key='venue')
    l, r = st.columns(2)
    toss_winner = l.selectbox('Toss winner', [team1, team2], key='toss_winner')
    decision = r.selectbox('Toss decision', ['bat', 'field'], key='decision')
    inputs = (team1, team2, venue, toss_winner, decision)
    if st.button('Estimate match outcome', type='primary', key='predict'):
        st.session_state['prediction'] = (inputs, predict(bundle, *inputs))
    result = st.session_state.get('prediction')
    if result and result[0] == inputs:
        probabilities = result[1]
        winner = max(probabilities, key=probabilities.get)
        st.subheader(f'Model leans toward {winner}')
        one, two = st.columns(2)
        one.metric(team1, f'{probabilities[team1]:.1%}')
        two.metric(team2, f'{probabilities[team2]:.1%}')
        st.progress(probabilities[team1], text=f'{team1} share of the two-team estimate')
        st.caption('Estimated probability conditional on a decisive result. These probabilities have not been independently calibrated.')
        venue_count = int((matches.venue == venue).sum())
        st.caption(f'{venue_count} historical matches at this recorded venue. Venue names follow the source and may contain naming variants.')
        if venue_count < 20:
            st.info('Limited history at this venue: its venue effect is based on fewer than 20 matches.')
    elif result:
        st.info('Selections changed. Click Estimate match outcome for an updated result.')
    st.caption('IPL 2027 is the project target. No 2027 results, fixtures, squads or live data are included. This predicts individual matches, not the tournament champion.')

with history_tab:
    st.subheader(f'{team1} vs {team2}')
    head = matches[((matches.team1 == team1) & (matches.team2 == team2)) |
                   ((matches.team1 == team2) & (matches.team2 == team1))]
    x, y, z = st.columns(3)
    x.metric('Decisive matches', len(head))
    y.metric(f'{team1} wins', int((head.winner == team1).sum()))
    z.metric(f'{team2} wins', int((head.winner == team2).sum()))
    if not head.empty:
        st.dataframe(head[['date', 'venue', 'toss_winner', 'winner']].sort_values('date', ascending=False), hide_index=True, width='stretch')
    else:
        st.info('No completed head-to-head matches in this dataset.')

with model_tab:
    st.subheader('What was tested?')
    st.write(f"Trained on {report['train_matches']:,} matches before {report['test_season']}; evaluated on {report['test_matches']} matches from {report['test_season']}. The deployed model was then refit on all {len(matches):,} matches.")
    st.dataframe(pd.DataFrame({'Evaluation': ['Model accuracy', 'Historical win-rate baseline', 'Brier score (lower is better)', 'Log loss (lower is better)'],
        'Value': [f"{report['accuracy']:.1%}", f"{report['historical_win_rate_baseline_accuracy']:.1%}", f"{report['brier_score']:.3f}", f"{report['log_loss']:.3f}"]}), hide_index=True)
    st.write('Regularized logistic regression uses team identity, who won the toss, batting order and the batting-order effect at the selected venue. Recent matches receive more weight. Swapping the teams preserves the same probabilities for each team.')
    st.write('No-result matches are excluded; recorded super-over winners are included. Renamed franchises are combined. Holdout metrics belong to the pre-2026 model, not the refitted serving model.')
    st.write('Limitations: player availability, form at prediction time, pitch conditions, weather and live scores are not modeled. Historical performance does not establish 2027 accuracy.')
    st.markdown('Data: [Cricsheet](https://cricsheet.org/downloads/) · downloaded 16 September 2026. Derived match metadata is included with source attribution and archive checksum.')
    st.download_button('Download evaluation report', (ROOT/'metrics.json').read_text(), 'ipl-evaluation.json', 'application/json')
    st.download_button('Download match dataset', (ROOT/'data/matches.csv').read_bytes(), 'ipl-matches.csv', 'text/csv')

st.caption('Built by Mithun Kumar · Historical analysis, transparent evaluation.')
