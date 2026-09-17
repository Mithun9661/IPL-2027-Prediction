from html import escape

TEAMS={
'Chennai Super Kings':('CSK','#f6ce38','#282008'),
'Mumbai Indians':('MI','#338aee','#ffffff'),
'Royal Challengers Bengaluru':('RCB','#ed4a55','#ffffff'),
'Kolkata Knight Riders':('KKR','#ad84e7','#1c1030'),
'Delhi Capitals':('DC','#469bff','#081b38'),
'Punjab Kings':('PBKS','#ff626d','#351014'),
'Rajasthan Royals':('RR','#f289cb','#3c1231'),
'Gujarat Titans':('GT','#86b3cf','#092131'),
'Lucknow Super Giants':('LSG','#55cfdf','#08282e'),
'Sunrisers Hyderabad':('SRH','#ff9656','#341b08')}

def team_card(team,state):
    abbr,color,ink=TEAMS.get(team,('IPL','#a0b6cd','#122034'))
    form=''.join(f'<i class="{ "won" if w else "lost" }">{"W" if w else "L"}</i>' for w in state['form'].get(team,[])[-5:])
    return f'<div class="teamcard" style="--team:{color};--ink:{ink}"><div class="teamtop"><div class="badge">{abbr}</div><div><div class="teamname">{escape(team)}</div><div class="teamsub">TEAM PROFILE</div></div></div><div class="cardbottom"><div><div class="statlabel">LAST FIVE · OLDEST FIRST</div><div class="form">{form}</div></div><div><div class="statlabel">ELO RATING</div><div class="rating">{round(state["elo"].get(team,1500))}</div></div></div></div>'

def matchup(a,b,state):
    return '<div class="matchup">'+team_card(a,state)+'<div class="versus">VS</div>'+team_card(b,state)+'</div>'

def probability(a,b,p):
    aa,ca,_=TEAMS[a];bb,cb,_=TEAMS[b]
    return f'<div class="prob"><div class="eyebrow">MODEL ESTIMATE</div><div class="probhead"><div><small>{aa}</small><strong>{p:.1%}</strong></div><div style="text-align:right"><small>{bb}</small><strong>{1-p:.1%}</strong></div></div><div class="probbar" role="img" aria-label="{escape(a)} {p:.1%}, {escape(b)} {1-p:.1%}"><div style="width:{p*100}%;background:{ca}"></div><div style="width:{(1-p)*100}%;background:{cb}"></div></div><div class="probfoot">Historical model estimate · A close split means little separation between teams.</div></div>'

def pills(venue,toss,decision):
    return '<div class="matchmeta">'+''.join('<span class="pill">'+escape(s)+'</span>' for s in [venue,TEAMS[toss][0]+' won toss · chose to '+decision])+'</div>'
