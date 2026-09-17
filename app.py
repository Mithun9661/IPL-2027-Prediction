from pathlib import Path
import pandas as pd
import streamlit as st
from predictor import load_bundle, predict

ROOT = Path(__file__).parent
st.set_page_config(page_title='IPL 2027 | Match Lab', page_icon='🏏', layout='wide')
from ui import matchup, probability, pills
st.markdown((ROOT / 'theme.css').read_text(), unsafe_allow_html=True)

@st.cache_resource
def resources():
    return load_bundle(), pd.read_csv(ROOT / 'data/matches.csv')

bundle, matches = resources()
report = bundle['report']
st.markdown('<div class="masthead"><div class="brand">MATCH<span>LAB</span> / CRICKET</div><div class="season">IPL 2027 · MATCH EXPLORER</div></div><div class="hero"><div class="field" aria-hidden="true"></div><div class="eyebrow">THE MATCHUP. THE HISTORY. THE EDGE.</div><h1>Know the contest.<br>Explore the possibilities.</h1><p>Compare team form, revisit the rivalry and explore a data-driven match estimate.</p></div>', unsafe_allow_html=True)
st.caption(f"Experimental model · {report['test_season']} retrospective accuracy {report['accuracy']:.1%}, close to chance. Not a reliable 2027 forecast. No live scores or confirmed fixtures.")
a, b, c = st.columns(3)
a.metric('Historical matches', f"{len(matches):,}")
b.metric('Data through', report['data_through'])
c.metric('Teams available', len(bundle['active_teams']))

prediction_tab, history_tab, model_tab = st.tabs(['Match prediction', 'Head-to-head', 'Model & data'])
with prediction_tab:
    st.markdown('<div class="sectionlabel">01 / CHOOSE YOUR MATCHUP</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    team1 = left.selectbox('Team 1', bundle['active_teams'], key='team1')
    options = [t for t in bundle['active_teams'] if t != team1]
    team2 = right.selectbox('Team 2', options, key='team2')
    st.markdown(matchup(team1, team2, bundle['history']), unsafe_allow_html=True)
    st.caption(f"Form and ratings through {report['data_through']} · Abbreviation badges, not official team logos.")
    st.markdown('<div class="sectionlabel">02 / MATCH CONDITIONS</div>', unsafe_allow_html=True)
    venues = bundle['venues']
    venue = st.selectbox('Venue', venues, key='venue')
    l, r = st.columns(2)
    toss_winner = l.selectbox('Toss winner', [team1, team2], key='toss_winner')
    decision = r.selectbox('Toss decision', ['bat', 'field'], key='decision')
    st.markdown(pills(venue, toss_winner, decision), unsafe_allow_html=True)
    inputs = (team1, team2, venue, toss_winner, decision)
    if st.button('Estimate match outcome', type='primary', key='predict'):
        st.session_state['prediction'] = (inputs, predict(bundle, *inputs))
    result = st.session_state.get('prediction')
    if result and result[0] == inputs:
        probabilities = result[1]
        winner = max(probabilities, key=probabilities.get)
        st.subheader(f'Model leans toward {winner}')
        st.markdown(probability(team1, team2, probabilities[team1]), unsafe_allow_html=True)
        st.caption('Estimated probability conditional on a decisive result. These probabilities have not been independently calibrated.')
        venue_count = int((matches.venue == venue).sum())
        st.caption(f'{venue_count} historical matches at this recorded venue. Venue names follow the source and may contain naming variants.')
        if venue_count < 20:
            st.info('Limited history at this venue: its venue effect is based on fewer than 20 matches.')
    elif result:
        st.info('Selections changed. Click Estimate match outcome for an updated result.')
    else:
        st.markdown('<div class="empty">Your matchup is ready. Choose the conditions above, then estimate the outcome.</div>', unsafe_allow_html=True)
    with st.expander('Team form and strength at data cutoff'):
        state = bundle['history']
        st.caption(f"Historical snapshot through {report['data_through']}; not live form.")
        details = []
        for team in [team1, team2]:
            recent = state['form'].get(team, [])[-5:]
            record = state['venue'].get((team, venue), [0, 0])
            details.append({'Team': team, 'Last 5 (oldest → newest)': ' '.join('W' if x else 'L' for x in recent),
                            'Elo rating': round(state['elo'].get(team, 1500)), 'Venue wins / games': f'{record[0]} / {record[1]}'})
        st.dataframe(pd.DataFrame(details), hide_index=True)
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
    st.write(f"Selected model: {report['model']}. Selection used lowest log loss across rolling 2023–2025 validation, not 2026 results.")
    comparison = pd.DataFrame([{k:v for k,v in row.items() if k != 'folds'} for row in report['comparison']])
    st.dataframe(comparison, hide_index=True)
    st.caption('A constant 50/50 estimate has log loss 0.693. None of these candidates beat that reference on pooled 2023–2025 validation.')
    st.info(report['evaluation_caveat'])
    st.write('Features include last-five-match form, Elo team strength, venue win history, head-to-head record, team identity, toss and batting order. Only earlier dates contribute to historical features. Team swapping preserves the same probabilities.')
    st.write('No-result matches are excluded; recorded super-over winners are included. Renamed franchises are combined. Retrospective metrics belong to the pre-2026 model, not the refitted serving model.')
    st.write('Limitations: player availability, current form after the dataset cutoff, pitch conditions, weather and live scores are not modeled. Historical performance does not establish 2027 accuracy.')
    st.markdown('Data: [Cricsheet](https://cricsheet.org/downloads/) · downloaded 16 September 2026. Derived match metadata is included with source attribution and archive checksum.')
    st.download_button('Download evaluation report', (ROOT/'metrics.json').read_text(), 'ipl-evaluation.json', 'application/json')
    st.download_button('Download match dataset', (ROOT/'data/matches.csv').read_bytes(), 'ipl-matches.csv', 'text/csv')

st.caption('Built by Mithun Kumar · Historical analysis, transparent evaluation.')
