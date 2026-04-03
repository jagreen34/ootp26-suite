"""
OOTP 26 Pitching Staff Construction (v2 — single-file, PIT_CON)
"""
import numpy as np
import pandas as pd

ROLE_MODELS = {
    'Closer':        {'intercept': -5.2667, 'STU': 0.0351, 'MOV': 0.0502, 'CON': 0.0428, 'STM': 0.0011},
    'Setup':         {'intercept': -4.3989, 'STU': 0.0275, 'MOV': 0.0474, 'CON': 0.0291, 'STM': -0.0035},
    'Middle Relief': {'intercept': -4.4954, 'STU': 0.0247, 'MOV': 0.0561, 'CON': 0.0223, 'STM': -0.0022},
    'Long Relief':   {'intercept': -4.6718, 'STU': 0.0278, 'MOV': 0.0597, 'CON': 0.0219, 'STM': -0.0044},
    'Mop-Up':        {'intercept': -4.5584, 'STU': 0.0261, 'MOV': 0.0586, 'CON': 0.0193, 'STM': -0.0040},
    'Starter':       {'intercept': -1.2415, 'STU': -0.0108, 'MOV': 0.0689, 'CON': -0.0276, 'STM': 0.0578},
}
RELIEF_ROLES = ['Closer', 'Setup', 'Middle Relief', 'Long Relief', 'Mop-Up']
STM_THRESHOLD = 42
ROLE_TEMPLATES = {
    3: ['Closer','Setup','Middle Relief'],
    4: ['Closer','Setup','Middle Relief','Long Relief'],
    5: ['Closer','Setup','Middle Relief','Middle Relief','Long Relief'],
    6: ['Closer','Setup','Middle Relief','Middle Relief','Long Relief','Long Relief'],
    7: ['Closer','Setup','Setup','Middle Relief','Middle Relief','Long Relief','Long Relief'],
}

def _safe(v, d=0):
    try:
        if pd.isna(v): return d
        return float(v)
    except: return d

def _pit_con(row):
    """Get pitcher control — uses PIT_CON if available, falls back to CON."""
    return _safe(row.get('PIT_CON', row.get('CON', 0)))

def predict_role_war(row, role):
    m = ROLE_MODELS[role]
    return m['intercept'] + _safe(row.get('STU',0))*m['STU'] + _safe(row.get('MOV',0))*m['MOV'] + _pit_con(row)*m['CON'] + _safe(row.get('STM',0))*m['STM']

def sp_f1(row):
    return -5.932 + _safe(row.get('MOV',0))*0.1091 + _safe(row.get('STM',0))*0.056 + _safe(row.get('STU',0))*0.0207 + _pit_con(row)*0.0053

def rp_f1(row):
    return -2.852 + _safe(row.get('MOV',0))*0.0509 + _safe(row.get('STU',0))*0.0217 + _pit_con(row)*-0.0029

def build_staff(roster_df, rotation_size=6, pos_players=14):
    pitchers = roster_df[roster_df['POS'].isin(['SP','RP','CL'])].copy()
    for idx, row in pitchers.iterrows():
        pitchers.loc[idx, 'SP_F1'] = round(sp_f1(row), 2)
        pitchers.loc[idx, 'RP_F1'] = round(rp_f1(row), 2)
        pitchers.loc[idx, 'Starter_WAR'] = round(predict_role_war(row, 'Starter'), 2)
        for role in RELIEF_ROLES:
            pitchers.loc[idx, f'{role}_WAR'] = round(predict_role_war(row, role), 2)
        br = max(RELIEF_ROLES, key=lambda r: predict_role_war(row, r))
        pitchers.loc[idx, 'Best_Relief'] = br
        pitchers.loc[idx, 'Best_Relief_WAR'] = round(predict_role_war(row, br), 2)

    sp_pool = pitchers[pitchers['STM'] >= STM_THRESHOLD].sort_values('SP_F1', ascending=False)
    bp_only = pitchers[pitchers['STM'] < STM_THRESHOLD]
    rotation = sp_pool.head(rotation_size)
    bp_pool = pd.concat([bp_only, sp_pool.iloc[rotation_size:]])
    bp_size = 25 - pos_players - rotation_size
    roles = ROLE_TEMPLATES.get(bp_size, ['Closer','Setup'] + ['Middle Relief']*max(0,bp_size-2))[:bp_size]
    while len(roles) < bp_size: roles.append('Mop-Up')

    assignments, used = [], set()
    for role in roles:
        for _, r in bp_pool.sort_values(f'{role}_WAR', ascending=False).iterrows():
            if r['Name'] not in used:
                used.add(r['Name'])
                assignments.append({'role': role, 'name': r['Name'], 'age': r.get('Age',''), 'stu': int(_safe(r['STU'])), 'mov': int(_safe(r['MOV'])),
                    'con': int(_pit_con(r)), 'stm': int(_safe(r['STM'])), 'war': round(predict_role_war(r, role),2), 'sp_capable': _safe(r['STM']) >= STM_THRESHOLD, 'row': r})
                break
    return {'rotation': rotation, 'bullpen': assignments, 'rotation_size': rotation_size, 'bp_size': bp_size, 'all_pitchers': pitchers}

def render_staff_construction(uploaded_df=None):
    import streamlit as st
    st.header("⚙️ Pitching Staff Construction")
    st.caption("6-man rotation + bullpen roles. Models from 253K study.")
    if uploaded_df is None:
        st.info("Load a roster in the sidebar."); return
    df = uploaded_df
    team = st.selectbox("Team", sorted(df[df['TM'] != '-']['TM'].unique()), key='ps_team')
    team_df = df[df['TM'] == team]
    c1, c2 = st.columns(2)
    with c1: rotation_size = st.select_slider("Rotation Size", [3,4,5,6], value=6, help="6-man universally optimal (+3.2 WAR)")
    with c2: pos_players = st.number_input("Position Players", value=14, min_value=10, max_value=16, help="14 optimal, 13 safe")
    bp_size = 25 - pos_players - rotation_size
    st.caption(f"{rotation_size} SP + {bp_size} RP + {pos_players} POS = 25")
    if st.button("🔄 Build Staff", type="primary", key="ps_build"):
        st.session_state['ps_results'] = build_staff(team_df, rotation_size, pos_players)
    if 'ps_results' not in st.session_state: return
    r = st.session_state['ps_results']
    t1, t2, t3 = st.tabs(["Staff Card", "All Pitchers", "Methodology"])
    with t1:
        st.subheader(f"Rotation ({r['rotation_size']}-Man)")
        rr = [{'#':i+1,'Name':row['Name'],'Age':int(row['Age']),'STU':int(_safe(row['STU'])),'MOV':int(_safe(row['MOV'])),'PIT_CON':int(_pit_con(row)),'STM':int(_safe(row['STM'])),'SP F1':row['SP_F1']} for i,(_, row) in enumerate(r['rotation'].iterrows())]
        st.dataframe(pd.DataFrame(rr), use_container_width=True, hide_index=True)
        st.subheader(f"Bullpen ({r['bp_size']} Arms)")
        br = [{'Role':a['role'],'Name':a['name'],'Age':a['age'],'STU':a['stu'],'MOV':a['mov'],'PIT_CON':a['con'],'STM':a['stm'],'WAR':a['war'],'SP?':'✓' if a['sp_capable'] else ''} for a in r['bullpen']]
        st.dataframe(pd.DataFrame(br), use_container_width=True, hide_index=True)
        st.metric("Total BP WAR", f"{sum(a['war'] for a in r['bullpen']):.2f}")
    with t2:
        p = r['all_pitchers']
        cols = [c for c in ['Name','POS','Age','SP_F1','Starter_WAR','Closer_WAR','Setup_WAR','Middle Relief_WAR','Long Relief_WAR','Best_Relief','Best_Relief_WAR'] if c in p.columns]
        st.dataframe(p[cols].sort_values('SP_F1', ascending=False), use_container_width=True, hide_index=True)
    with t3:
        st.write("**6-man universally optimal.** +3.2 WAR/rotation member. 37-GS hard cap. Extra rest doesn't help.")
        st.write("**Roster: 6 SP + 5 RP + 14 POS.** 7th reliever = 0.2 WAR at 4.3 ERA. 13th pos player = 1.2 WAR.")
        st.write("**STM < 42 → bullpen.** Closer: best STU arm. All other RP: sort by MOV. HLD useless.")
