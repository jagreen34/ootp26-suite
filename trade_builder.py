"""
OOTP 26 Trade Package Builder (v2 — single-file, TM filter, PIT_CON, win-focused)
"""
import numpy as np
import pandas as pd
from itertools import combinations

POS_MULTS = {'SS': 1.50, 'CF': 1.55, 'C': 1.30, '2B': 1.30, '3B': 1.20, 'RF': 1.25, 'LF': 1.05, '1B': 1.00, 'SP': 1.00, 'RP': 0.60, 'CL': 0.60}

def _safe(v, d=0):
    try:
        if pd.isna(v): return d
        return float(v)
    except: return d

def _pit_con(row):
    return _safe(row.get('PIT_CON', row.get('CON', 0)))

def age_factor(age, pos=''):
    if pos in ('SP','RP','CL'):
        if age <= 27: return 1.0
        elif age <= 30: return 1.0 - (age-27)*0.08
        elif age <= 33: return 0.76 - (age-30)*0.10
        else: return max(0.1, 0.46 - (age-33)*0.12)
    else:
        if age <= 25: return 1.0
        elif age <= 28: return 1.0 - (age-25)*0.03
        elif age <= 31: return 0.91 - (age-28)*0.08
        elif age <= 34: return 0.67 - (age-31)*0.10
        else: return max(0.1, 0.37 - (age-34)*0.10)

def estimate_years(age, exp=None):
    if exp and exp > 0: return max(1, min(4, 6 - exp))
    if age <= 20: return 4
    elif age <= 23: return 3.5
    elif age <= 25: return 3
    elif age <= 27: return 2
    elif age <= 29: return 1.5
    return 1

def bat_f1(row):
    return (-14.168 + _safe(row.get('POW',0))*0.1142 + _safe(row.get('BABIP',0))*0.0725 + _safe(row.get('EYE',0))*0.04
        + _safe(row.get('CON',0))*0.0379 + _safe(row.get("K's", row.get('Ks',0)))*0.0317 + _safe(row.get('GAP',0))*0.0291 + _safe(row.get('SPE',0))*0.0128)

def sp_f1(row):
    return -5.932 + _safe(row.get('MOV',0))*0.1091 + _safe(row.get('STM',0))*0.056 + _safe(row.get('STU',0))*0.0207 + _pit_con(row)*0.0053

def rp_f1(row):
    return -2.852 + _safe(row.get('MOV',0))*0.0509 + _safe(row.get('STU',0))*0.0217 + _pit_con(row)*-0.0029

def calc_trade_value(row):
    pos = str(row.get('POS','')); age = _safe(row.get('Age',25)); exp = _safe(row.get('EXP',0)) or None
    if pos == 'SP': f1, bl = sp_f1(row), 0.9
    elif pos in ('RP','CL'): f1, bl = rp_f1(row), -0.5
    else: f1, bl = bat_f1(row), 0.2
    mult = POS_MULTS.get(pos, 1.0); years = estimate_years(age, exp); af = age_factor(age, pos)
    tv = max(0, f1-bl) * years * mult * af
    bonus = (23-age)*0.5*mult if age <= 22 and f1 > 2.0 else 0
    return {'name': row.get('Name',''), 'pos': pos, 'age': int(age), 'f1': round(f1,2), 'years': round(years,1), 'mult': mult, 'age_factor': round(af,2), 'trade_value': round(tv+bonus,1), 'prospect_bonus': round(bonus,1)}

def evaluate_roster(df):
    return pd.DataFrame([calc_trade_value(row) for _, row in df.iterrows()]).sort_values('trade_value', ascending=False)

def find_packages(my_tv, target_value, max_players=3, max_pct=90):
    packages = []; players = my_tv.to_dict('records')
    low, high = target_value*0.50, target_value*(max_pct/100)
    for size in range(1, min(max_players+1, len(players)+1)):
        for combo in combinations(range(len(players)), size):
            total = sum(players[i]['trade_value'] for i in combo)
            if low <= total <= high:
                packages.append({'players': [players[i] for i in combo], 'total_value': round(total,1), 'target_value': round(target_value,1), 'you_save': round(target_value-total,1), 'pct_of_target': round(total/max(target_value,0.1)*100,1), 'n_players': size})
    packages.sort(key=lambda p: (-p['you_save'], p['n_players']))
    return packages[:30]

def render_trade_builder(uploaded_df=None):
    import streamlit as st
    st.header("🔄 Trade Package Builder")
    st.caption("Select your team and their team. Pick a target. Find packages where you win.")
    if uploaded_df is None:
        st.info("Load a roster in the sidebar."); return
    df = uploaded_df
    teams = sorted(df[df['TM'] != '-']['TM'].unique())
    c1, c2 = st.columns(2)
    with c1: my_team = st.selectbox("My Team", teams, index=teams.index('Philadelphia') if 'Philadelphia' in teams else 0, key='tb_my')
    with c2: their_team = st.selectbox("Their Team", [t for t in teams if t != my_team], key='tb_their')
    my_df = df[df['TM'] == my_team]; their_df = df[df['TM'] == their_team]
    my_tv = evaluate_roster(my_df); their_tv = evaluate_roster(their_df)
    t1, t2, t3 = st.tabs(["Build Trade", "My Roster Values", "Their Roster Values"])
    with t1:
        tradeable = their_tv[their_tv['trade_value'] > 0].reset_index(drop=True)
        if tradeable.empty: st.warning("No tradeable players."); return
        opts = [f"{r['name']} ({r['pos']}, {r['age']}, F1={r['f1']}, TV={r['trade_value']})" for _, r in tradeable.iterrows()]
        ti = st.selectbox("Target", range(len(opts)), format_func=lambda i: opts[i])
        target = tradeable.iloc[ti]; tv = target['trade_value']
        c1,c2,c3,c4 = st.columns(4); c1.metric("F1",target['f1']); c2.metric("Years",target['years']); c3.metric("Age Factor",target['age_factor']); c4.metric("Trade Value",tv)
        cc1, cc2 = st.columns(2)
        with cc1: max_p = st.slider("Max players in package", 1, 4, 3)
        with cc2: max_pct = st.slider("Max % you'll pay", 60, 100, 90, help="Lower = bigger win for you")
        exclude = st.multiselect("Untouchables", my_tv[my_tv['trade_value']>0]['name'].tolist(), key='tb_excl')
        available = my_tv[(my_tv['trade_value']>0) & (~my_tv['name'].isin(exclude))].reset_index(drop=True)
        if st.button("🔍 Find Winning Packages", type="primary"):
            pkgs = find_packages(available, tv, max_p, max_pct)
            if not pkgs: st.warning(f"No packages at ≤{max_pct}%. Raise the slider.")
            else:
                st.subheader(f"Packages for {target['name']} (TV={tv})")
                for i, pkg in enumerate(pkgs):
                    pct = pkg['pct_of_target']; sv = pkg['you_save']
                    tag = "🟢 STEAL" if pct<=70 else "🟢 Great" if pct<=80 else "🟡 Good" if pct<=90 else "🟠 Slight edge"
                    with st.expander(f"{tag} — {' + '.join(p['name'] for p in pkg['players'])} ({pkg['total_value']} TV / {pct:.0f}% — save {sv})"):
                        st.dataframe(pd.DataFrame([{'Name':p['name'],'Pos':p['pos'],'Age':p['age'],'F1':p['f1'],'Years':p['years'],'TV':p['trade_value']} for p in pkg['players']]), use_container_width=True, hide_index=True)
                        st.write(f"**You give {pkg['total_value']}** → Get {pkg['target_value']} → **Profit {sv} TV ({100-pct:.0f}% discount)**")
    with t2: st.dataframe(my_tv, use_container_width=True, hide_index=True)
    with t3: st.dataframe(their_tv, use_container_width=True, hide_index=True)
