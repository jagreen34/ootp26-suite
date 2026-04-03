"""
OOTP 26 Lineup Construction Tool (v2 — single-file format)
"""
import numpy as np
import pandas as pd
from itertools import permutations
from scipy.optimize import linear_sum_assignment

FIELD_POSITIONS = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
OFF_FEATURES = ['CON', 'POW', 'EYE', 'GAP', 'BABIP', 'SPE', 'STE']

ZR_MODELS = {
    'C':  {'intercept': -7.32,  'coefs': {'C ABI': 0.0628, 'C FRM': 0.0196, 'C ARM': 0.0539}},
    '1B': {'intercept': -13.43, 'coefs': {'IF RNG': 0.2540, 'IF ERR': 0.0293, 'IF ARM': -0.0013}},
    '2B': {'intercept': -47.24, 'coefs': {'IF RNG': 0.8635, 'IF ARM': 0.0644}},
    '3B': {'intercept': -31.44, 'coefs': {'IF RNG': 0.3331, 'IF ARM': 0.2642}},
    'SS': {'intercept': -66.76, 'coefs': {'IF RNG': 0.9064, 'IF ARM': 0.3012}},
    'LF': {'intercept': -29.47, 'coefs': {'OF RNG': 0.6079, 'OF ARM': 0.0041}},
    'CF': {'intercept': -46.77, 'coefs': {'OF RNG': 0.8833}},
    'RF': {'intercept': -52.45, 'coefs': {'OF RNG': 0.9968, 'OF ARM': 0.0669}},
}
ZR_TO_WAR = {'C': 0.1333, '1B': 0.1182, '2B': 0.1227, '3B': 0.1248, 'SS': 0.1040, 'LF': 0.1626, 'CF': 0.1111, 'RF': 0.1215}
POS_FLOORS = {'C': {'C ABI': 45}, 'SS': {'IF RNG': 50}, 'CF': {'OF RNG': 50}}
POS_MULTIPLIERS = {'CF': 1.55, 'SS': 1.50, 'C': 1.30, '2B': 1.30, 'RF': 1.25, '3B': 1.20, 'LF': 1.05, '1B': 1.00}

LINEUP_MODELS = {
    1: {'intercept': -6.5641, 'coefs': {'CON': 0.0300, 'POW': 0.0948, 'EYE': 0.0123, 'GAP': 0.0302, 'BABIP': 0.0505, 'SPE': 0.0004, 'STE': 0.0116}},
    2: {'intercept': -7.1458, 'coefs': {'CON': 0.0394, 'POW': 0.0996, 'EYE': 0.0093, 'GAP': 0.0238, 'BABIP': 0.0515, 'SPE': 0.0038, 'STE': 0.0106}},
    3: {'intercept': -6.6353, 'coefs': {'CON': 0.0341, 'POW': 0.1022, 'EYE': -0.0002, 'GAP': 0.0211, 'BABIP': 0.0551, 'SPE': 0.0093, 'STE': 0.0038}},
    4: {'intercept': -6.4545, 'coefs': {'CON': 0.0385, 'POW': 0.1023, 'EYE': 0.0015, 'GAP': 0.0212, 'BABIP': 0.0504, 'SPE': 0.0064, 'STE': -0.0008}},
    5: {'intercept': -6.8664, 'coefs': {'CON': 0.0466, 'POW': 0.1028, 'EYE': -0.0023, 'GAP': 0.0331, 'BABIP': 0.0385, 'SPE': 0.0084, 'STE': -0.0011}},
    6: {'intercept': -7.1354, 'coefs': {'CON': 0.0380, 'POW': 0.0995, 'EYE': -0.0041, 'GAP': 0.0270, 'BABIP': 0.0635, 'SPE': 0.0024, 'STE': 0.0025}},
    7: {'intercept': -7.2877, 'coefs': {'CON': 0.0447, 'POW': 0.0990, 'EYE': -0.0014, 'GAP': 0.0170, 'BABIP': 0.0581, 'SPE': 0.0157, 'STE': -0.0062}},
    8: {'intercept': -5.2399, 'coefs': {'CON': 0.0208, 'POW': 0.1001, 'EYE': -0.0149, 'GAP': 0.0196, 'BABIP': 0.0544, 'SPE': 0.0159, 'STE': -0.0112}},
}
PA_WEIGHTS = {1: 670, 2: 642, 3: 613, 4: 581, 5: 546, 6: 509, 7: 462, 8: 404}

def _safe(v, d=0):
    try:
        if pd.isna(v): return d
        return float(v)
    except: return d

def predict_zr(row, position):
    m = ZR_MODELS[position]; zr = m['intercept']
    for feat, coef in m['coefs'].items(): zr += _safe(row.get(feat, 0)) * coef
    return zr

def predict_def_war(row, position): return predict_zr(row, position) * ZR_TO_WAR[position]

def is_eligible(row, position, min_pos_rating=40):
    val = row.get(position, '-')
    if val == '-' or pd.isna(val): return False
    try: return float(val) >= min_pos_rating
    except: return False

def meets_floor(row, position):
    if position not in POS_FLOORS: return True
    for rn, mv in POS_FLOORS[position].items():
        if _safe(row.get(rn, 0)) < mv: return False
    return True

def predict_offensive_war_avg(row):
    total = 0
    for lp in range(1, 9):
        m = LINEUP_MODELS[lp]; rate = m['intercept']
        for feat in OFF_FEATURES: rate += m['coefs'][feat] * _safe(row.get(feat, 50))
        total += rate * (PA_WEIGHTS[lp] / 600)
    return total / 8

def predict_lineup_war(row, lp):
    m = LINEUP_MODELS[lp]; rate = m['intercept']
    for feat in OFF_FEATURES: rate += m['coefs'][feat] * _safe(row.get(feat, 50))
    return rate * (PA_WEIGHTS[lp] / 600)

def build_value_matrix(roster_df):
    players, matrix = [], []
    for idx, row in roster_df.iterrows():
        off = predict_offensive_war_avg(row)
        vals = []
        for pos in FIELD_POSITIONS:
            if is_eligible(row, pos) and meets_floor(row, pos):
                vals.append(off + predict_def_war(row, pos))
            else: vals.append(-999)
        players.append({'idx': idx, 'name': row.get('Name',''), 'age': row.get('Age',''), 'primary_pos': row.get('POS',''), 'off_war': off, 'row': row})
        matrix.append(vals)
    return players, np.array(matrix)

def solve_assignment(players, value_matrix):
    n_p, n_pos = len(players), len(FIELD_POSITIONS)
    cost = np.full((n_pos, n_p), 9999.0)
    for pi in range(n_p):
        for posi in range(n_pos):
            v = value_matrix[pi][posi]
            if v > -900: cost[posi][pi] = -v
    pos_idx, p_idx = linear_sum_assignment(cost)
    assignments, total = [], 0
    for posi, pi in zip(pos_idx, p_idx):
        pos = FIELD_POSITIONS[posi]; p = players[pi]; v = value_matrix[pi][posi]
        assignments.append({'position': pos, 'player_idx': pi, 'name': p['name'], 'age': p['age'], 'off_war': p['off_war'], 'def_war': predict_def_war(p['row'], pos), 'total_war': v, 'row': p['row']})
        total += v
    assignments.sort(key=lambda x: {p:i for i,p in enumerate(FIELD_POSITIONS)}[x['position']])
    return assignments, total

def optimize_batting_order(assignments):
    n = len(assignments); best_war, best_order = -999, None
    for perm in permutations(range(n)):
        t = sum(predict_lineup_war(assignments[perm[i]]['row'], i+1) for i in range(n))
        if t > best_war: best_war, best_order = t, perm
    return best_order, best_war

def build_depth_chart(players, value_matrix, starter_indices):
    depth = {}
    for posi, pos in enumerate(FIELD_POSITIONS):
        eligible = [{'name': players[i]['name'], 'age': players[i]['age'], 'value': value_matrix[i][posi],
                     'def_war': predict_def_war(players[i]['row'], pos), 'off_war': players[i]['off_war'],
                     'is_starter': i in starter_indices} for i in range(len(players)) if value_matrix[i][posi] > -900]
        eligible.sort(key=lambda x: x['value'], reverse=True)
        depth[pos] = eligible
    return depth

def run_full_optimization(roster_df):
    pos_players = roster_df[~roster_df['POS'].isin(['SP', 'RP', 'CL'])].copy()
    players, matrix = build_value_matrix(pos_players)
    assignments, aw = solve_assignment(players, matrix)
    batting_order, bw = optimize_batting_order(assignments)
    depth = build_depth_chart(players, matrix, set(a['player_idx'] for a in assignments))
    return {'assignments': assignments, 'assignment_war': aw, 'batting_order': batting_order, 'batting_war': bw, 'depth_chart': depth, 'value_matrix': matrix, 'players': players}

def render_lineup_construction(uploaded_df=None):
    import streamlit as st
    st.header("🏟️ Lineup Construction")
    st.caption("Position assignment (Hungarian algorithm) + batting order optimization (40,320 permutations)")
    if uploaded_df is None:
        st.info("Load a roster in the sidebar to use this tool.")
        return
    df = uploaded_df
    team = st.selectbox("Team", sorted(df[df['TM'] != '-']['TM'].unique()), key='lc_team')
    team_df = df[df['TM'] == team]
    pos_players = team_df[~team_df['POS'].isin(['SP', 'RP', 'CL'])]
    st.success(f"{len(pos_players)} position players on {team}")
    if st.button("🔄 Optimize Lineup", type="primary", key="lc_opt"):
        with st.spinner("Solving..."): st.session_state['lc_results'] = run_full_optimization(team_df)
    if 'lc_results' not in st.session_state: return
    results = st.session_state['lc_results']
    a, bo = results['assignments'], results['batting_order']
    tab1, tab2, tab3 = st.tabs(["Starting Lineup", "Depth Chart", "Value Matrix"])
    with tab1:
        rows = []
        for li, ai in enumerate(bo):
            aa = a[ai]; lp = li + 1; ow = predict_lineup_war(aa['row'], lp)
            rows.append({'LP': lp, 'Pos': aa['position'], 'Player': aa['name'], 'Age': aa['age'], 'Off WAR': round(ow,2), 'Def WAR': round(aa['def_war'],2), 'Total': round(ow+aa['def_war'],2)})
        ldf = pd.DataFrame(rows); st.dataframe(ldf, use_container_width=True, hide_index=True)
        c1,c2,c3 = st.columns(3); c1.metric("Offensive",f"{ldf['Off WAR'].sum():.1f}"); c2.metric("Defensive",f"{ldf['Def WAR'].sum():.1f}"); c3.metric("Combined",f"{ldf['Total'].sum():.1f}")
    with tab2:
        for pos in FIELD_POSITIONS:
            pl = results['depth_chart'][pos]
            if not pl: continue
            st.markdown(f"**{pos}** ({POS_MULTIPLIERS[pos]:.2f}x)")
            st.dataframe(pd.DataFrame([{'Rank':i+1,'Player':('★ ' if p['is_starter'] else '')+p['name'],'Age':p['age'],'Total':round(p['value'],2)} for i,p in enumerate(pl[:5])]), use_container_width=True, hide_index=True)
    with tab3:
        pl, mx = results['players'], results['value_matrix']
        mr = [{'Player': pl[i]['name'], 'Age': pl[i]['age'], **{pos: round(mx[i][j],1) if mx[i][j]>-900 else None for j,pos in enumerate(FIELD_POSITIONS)}} for i in range(len(pl))]
        mdf = pd.DataFrame(mr); mdf['Best'] = mdf[FIELD_POSITIONS].max(axis=1); mdf = mdf.sort_values('Best', ascending=False).drop(columns='Best')
        st.dataframe(mdf, use_container_width=True, hide_index=True)
