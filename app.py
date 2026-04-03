import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

st.set_page_config(page_title="OOTP 26 Evaluation Suite", page_icon="⚾", layout="wide")

@st.cache_resource
def load_models():
    gb_bat = joblib.load('models/gb_bat_season.pkl')
    gb_sp = joblib.load('models/gb_sp_season.pkl')
    bat_features = joblib.load('models/bat_features.pkl')
    sp_features = joblib.load('models/sp_features.pkl')
    return gb_bat, gb_sp, bat_features, sp_features
gb_bat, gb_sp, bat_features, sp_features = load_models()

POSITIONS = ['C','1B','2B','3B','SS','LF','CF','RF']
POS_MULTS = {'SS':1.50,'CF':1.55,'C':1.30,'2B':1.30,'3B':1.20,'RF':1.25,'LF':1.05,'1B':1.00}
ZR_MODELS = {
    'C':{'intercept':-7.32,'coef':{'C ABI':0.0628,'C FRM':0.0196,'C ARM':0.0539}},
    '1B':{'intercept':-13.43,'coef':{'IF RNG':0.2540,'IF ERR':0.0293,'IF ARM':-0.0013}},
    '2B':{'intercept':-47.24,'coef':{'IF RNG':0.8635,'IF ARM':0.0644}},
    '3B':{'intercept':-31.44,'coef':{'IF RNG':0.3331,'IF ARM':0.2642}},
    'SS':{'intercept':-66.76,'coef':{'IF RNG':0.9064,'IF ARM':0.3012}},
    'LF':{'intercept':-29.47,'coef':{'OF RNG':0.6079,'OF ARM':0.0041}},
    'CF':{'intercept':-46.77,'coef':{'OF RNG':0.8833}},
    'RF':{'intercept':-52.45,'coef':{'OF RNG':0.9968,'OF ARM':0.0669}},
}
ZR_WAR = {'C':0.1333,'1B':0.1182,'2B':0.1227,'3B':0.1248,'SS':0.1040,'LF':0.1626,'CF':0.1111,'RF':0.1215}
FLOORS = {'C':('C ABI',45),'2B':('IF RNG',50),'SS':('IF RNG',55),'CF':('OF RNG',55)}

def safe(v, d=0):
    try:
        if pd.isna(v): return d
        return float(v)
    except: return d

def pit_con(row):
    return safe(row.get('PIT_CON', row.get('CON', 0)))

def calc_def_war(row, pos):
    m = ZR_MODELS[pos]; zr = m['intercept']
    for f, c in m['coef'].items(): zr += safe(row.get(f, 0)) * c
    return zr * ZR_WAR[pos]

def best_def_war(row): return max(calc_def_war(row, p) for p in POSITIONS)

def f1_at_pos(row, pos):
    if pos in FLOORS:
        r, m = FLOORS[pos]
        if safe(row.get(r, 0)) < m: return -99
    return off_f1(row) + calc_def_war(row, pos)

def off_f1(row):
    return (-14.168 + safe(row['POW'])*0.1142 + safe(row['BABIP'])*0.0725 + safe(row['EYE'])*0.04
        + safe(row['CON'])*0.0379 + safe(row.get('Ks',0))*0.0317 + safe(row['GAP'])*0.0291 + safe(row['SPE'])*0.0128)

def best_position(row):
    bf, bp = -99, '1B'
    for pos in POSITIONS:
        f = f1_at_pos(row, pos)
        if f > bf: bf, bp = f, pos
    return bp, bf

def sp_f1(row): return -5.932 + safe(row['MOV'])*0.1091 + safe(row['STM'])*0.056 + safe(row['STU'])*0.0207 + pit_con(row)*0.0053
def rp_f1(row): return -2.852 + safe(row['MOV'])*0.0509 + safe(row['STU'])*0.0217 + pit_con(row)*-0.0029

def bat_f2(row):
    oc = safe(row.get('POW P',0))*0.194 + safe(row.get('GAP P',0))*0.234 + safe(row.get('EYE P',0))*0.188 + safe(row.get('CON P',0))*0.198 + safe(row.get('HT P',0))*0.186
    we = {'H':1,'N':0,'L':-1}.get(str(row.get('WE','')),0)
    iv = {'H':1,'N':0,'L':-1}.get(str(row.get('INT','')),0)
    ad = {'H':1,'N':0,'L':-1}.get(str(row.get('AD','')),0)
    f2 = -29.678 + oc*0.735 + safe(row['Age'])*0.163 + we*1.065 + iv*0.036 + ad*0.767 + best_def_war(row)*4.007
    if str(row.get('Prone','')) == 'Fragile': f2 *= 0.60
    return f2

def f2_tier(v):
    if v >= 25: return 'S'
    elif v >= 15: return 'A'
    elif v >= 5: return 'B'
    return 'C'

def prep_data(df):
    """Master prep: rename dupes, coerce numerics."""
    renames = {}
    if 'CON.1' in df.columns: renames['CON.1'] = 'PIT_CON'
    if 'CON P.1' in df.columns: renames['CON P.1'] = 'PIT_CON_P'
    if 'WAR.1' in df.columns: renames['WAR.1'] = 'PIT_WAR'
    if renames: df = df.rename(columns=renames)
    for col in ['CON','BABIP','GAP','POW','EYE','WAR','Age','PA','SPE','STE','IF RNG','IF ERR','OF RNG','OF ERR',
                'OF ARM','IF ARM','C ABI','C FRM','C ARM','POW P','GAP P','EYE P','CON P','HT P',
                'STU','MOV','STM','IP','EXP','PIT_CON','PIT_CON_P','PIT_WAR','HLD','TDP','VELO']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    if "K's" in df.columns: df['Ks'] = pd.to_numeric(df["K's"], errors='coerce')
    if 'STE' not in df.columns: df['STE'] = 40
    return df

def evaluate_batters(df):
    p = df[~df['POS'].isin(['SP','RP','CL'])].copy()
    for pos in POSITIONS: p[f'F1_{pos}'] = p.apply(lambda r: round(f1_at_pos(r, pos),1), axis=1)
    p['Off_F1'] = p.apply(lambda r: round(off_f1(r),1), axis=1)
    bp = p.apply(lambda r: best_position(r), axis=1)
    p['Best_Pos'] = [x[0] for x in bp]; p['Best_F1'] = [round(x[1],1) for x in bp]
    p['Pos_Mult'] = p['Best_Pos'].map(POS_MULTS)
    p['Useful_Pos'] = p.apply(lambda r: sum(1 for pos in POSITIONS if f1_at_pos(r, pos)>1.0), axis=1)
    p['F2'] = p.apply(lambda r: round(bat_f2(r),1) if safe(r['Age'])<=26 else 0, axis=1)
    p['F2_Tier'] = p['F2'].apply(f2_tier)
    p['best_def'] = p.apply(best_def_war, axis=1)
    for a in ['WE','INT','AD']: p[f'{a.lower()}_val'] = p[a].map({'H':1,'N':0,'L':-1}).fillna(0) if a in p.columns else 0
    for pt in ['Fan Fav','Captain','Sparkplug','Humble','Unmotivated','Disruptive','Selfish']:
        p[f'is_{pt.replace(" ","_").lower()}'] = (p['Type']==pt).astype(int) if 'Type' in p.columns else 0
    p['is_fragile'] = (p['Prone']=='Fragile').astype(int) if 'Prone' in p.columns else 0
    try: p['GB_WAR'] = np.round(gb_bat.predict(p[bat_features].fillna(0)),1)
    except: p['GB_WAR'] = 0
    p['Flags'] = p.apply(lambda r: ', '.join(filter(None, [
        'WE:H' if str(r.get('WE',''))=='H' else '','INT:H' if str(r.get('INT',''))=='H' else '',
        f'FLEX({r["Useful_Pos"]})' if r['Useful_Pos']>=5 else '',
        'FAN FAV' if str(r.get('Type',''))=='Fan Fav' else '',
        'UNMOTIVATED' if str(r.get('Type',''))=='Unmotivated' else '',
        'FRAGILE' if str(r.get('Prone',''))=='Fragile' else ''])), axis=1)
    return p

def evaluate_pitchers(df):
    p = df[df['POS'].isin(['SP','RP','CL'])].copy()
    p['SP_F1'] = p.apply(lambda r: round(sp_f1(r),1), axis=1)
    p['RP_F1'] = p.apply(lambda r: round(rp_f1(r),1), axis=1)
    p['CON_Flag'] = p['PIT_CON'].apply(lambda x: 'RED' if safe(x)<40 else 'YELLOW' if safe(x)<45 else 'OK') if 'PIT_CON' in p.columns else 'OK'
    for a in ['WE','INT','AD']: p[f'{a.lower()}_val'] = p[a].map({'H':1,'N':0,'L':-1}).fillna(0) if a in p.columns else 0
    for pt in ['Fan Fav','Captain','Humble','Unmotivated']:
        p[f'is_{pt.replace(" ","_").lower()}'] = (p['Type']==pt).astype(int) if 'Type' in p.columns else 0
    p['is_fragile'] = (p['Prone']=='Fragile').astype(int) if 'Prone' in p.columns else 0
    try: p['GB_WAR'] = np.round(gb_sp.predict(p[sp_features].fillna(0)),1)
    except: p['GB_WAR'] = 0
    p['Flags'] = p.apply(lambda r: ', '.join(filter(None, [
        'WE:H' if str(r.get('WE',''))=='H' else '','INT:H' if str(r.get('INT',''))=='H' else '',
        f'CON:{int(pit_con(r))}' if pit_con(r)>=50 else '',
        'FAN FAV' if str(r.get('Type',''))=='Fan Fav' else '',
        'UNMOTIVATED' if str(r.get('Type',''))=='Unmotivated' else '',
        'FRAGILE' if str(r.get('Prone',''))=='Fragile' else ''])), axis=1)
    return p

# ============================================================
# UI
# ============================================================
st.title("⚾ OOTP 26 Evaluation Suite")
st.caption("Philadelphia Quakers — v4 System")

st.sidebar.header("📂 Load Roster")
st.sidebar.caption("One CSV — bat + pitch combined. Export from OOTP using your saved shortlist.")
f = st.sidebar.file_uploader("Roster CSV", type=['csv'], key='main')
if f:
    st.session_state['roster'] = prep_data(pd.read_csv(f, encoding='utf-8-sig', low_memory=False))
    r = st.session_state['roster']
    nb = len(r[~r['POS'].isin(['SP','RP','CL'])]); np_ = len(r) - nb
    st.sidebar.success(f"{len(r)} players ({nb} bat / {np_} pitch)")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Mode", ["🏠 Quick Eval","📋 Offseason Phase 1","🎯 Trade Targets","📝 Draft Board",
    "⚾ Lineup Optimizer","🏟️ Lineup Construction","⚙️ Pitching Staff","📊 Draft Value","🔄 Trade Builder","📖 Reference"])

def get_data(): return st.session_state.get('roster', None)

# ============================================================
if mode == "🏠 Quick Eval":
    st.header("Quick Evaluation")
    df = get_data()
    if df is None:
        uf = st.file_uploader("Or upload any CSV here", type=['csv'], key='qe')
        if uf: df = prep_data(pd.read_csv(uf, encoding='utf-8-sig', low_memory=False))
    if df is not None:
        team = st.selectbox("Team (or All)", ['All'] + sorted(df[df['TM']!='-']['TM'].unique()), key='qe_tm')
        tdf = df if team == 'All' else df[df['TM']==team]
        t1, t2 = st.tabs(["Batters","Pitchers"])
        with t1:
            br = evaluate_batters(tdf)
            cols = [c for c in ['Name','TM','POS','Age','GB_WAR','Best_F1','Off_F1','Best_Pos','Pos_Mult','Useful_Pos','F2','F2_Tier','WAR','PA','POW','CON','EYE','GAP','SPE','Flags'] if c in br.columns]
            st.dataframe(br[cols].sort_values('GB_WAR', ascending=False), use_container_width=True, height=500)
        with t2:
            pr = evaluate_pitchers(tdf)
            cols = [c for c in ['Name','TM','POS','Age','GB_WAR','SP_F1','RP_F1','MOV','STU','PIT_CON','STM','CON_Flag','PIT_WAR','IP','FIP-','Flags'] if c in pr.columns]
            st.dataframe(pr[cols].sort_values('GB_WAR', ascending=False), use_container_width=True, height=500)

elif mode == "📋 Offseason Phase 1":
    st.header("Offseason Phase 1")
    df = get_data()
    if df is None: st.info("Load roster in sidebar."); st.stop()
    team = st.selectbox("Your Team", sorted(df[df['TM']!='-']['TM'].unique()), index=list(sorted(df[df['TM']!='-']['TM'].unique())).index('Philadelphia') if 'Philadelphia' in df['TM'].values else 0, key='op_tm')
    division = st.multiselect("Division Rivals", sorted(df[df['TM']!='-']['TM'].unique()), default=['Pittsburgh','Cleveland','New York','Brooklyn','Cincinnati','Washington'], key='op_div')
    my = df[df['TM']==team]; league = df[(df['TM']!=team) & (df['TM']!='-') & (~df['TM'].isin(division))]
    tb = evaluate_batters(my); tp = evaluate_pitchers(my); lb = evaluate_batters(league); lp = evaluate_pitchers(league)
    t1,t2,t3,t4,t5 = st.tabs(["My Batters","My Pitchers","Prospects","Bat Targets","SP Targets"])
    with t1: st.dataframe(tb[['Name','POS','Age','GB_WAR','Best_F1','Off_F1','Best_Pos','Useful_Pos','WAR','Flags']].sort_values('GB_WAR',ascending=False), use_container_width=True)
    with t2: st.dataframe(tp[['Name','POS','Age','GB_WAR','SP_F1','RP_F1','MOV','STU','PIT_CON','STM','CON_Flag','PIT_WAR','IP','Flags']].sort_values('GB_WAR',ascending=False), use_container_width=True)
    with t3:
        pr = tb[tb['Age']<=26]
        st.dataframe(pr[['Name','POS','Age','F2','F2_Tier','GB_WAR','Best_F1','Best_Pos','Flags']].sort_values('F2',ascending=False), use_container_width=True)
    with t4:
        mf = st.slider("Min F1",1.0,6.0,2.0,0.5,key='op_bf1')
        st.dataframe(lb[lb['Best_F1']>=mf][['Name','TM','POS','Age','GB_WAR','Best_F1','Best_Pos','Useful_Pos','WAR','Flags']].sort_values('GB_WAR',ascending=False).head(50), use_container_width=True)
    with t5:
        mf = st.slider("Min SP F1",2.0,7.0,3.5,0.5,key='op_sf1')
        st.dataframe(lp[lp['SP_F1']>=mf][['Name','TM','POS','Age','GB_WAR','SP_F1','MOV','STU','PIT_CON','STM','CON_Flag','PIT_WAR','IP','Flags']].sort_values('GB_WAR',ascending=False).head(50), use_container_width=True)

elif mode == "🎯 Trade Targets":
    st.header("Trade Target Evaluation")
    df = get_data()
    if df is None:
        uf = st.file_uploader("Upload CSV", type=['csv']); 
        if uf: df = prep_data(pd.read_csv(uf, encoding='utf-8-sig', low_memory=False))
    if df is not None:
        team = st.selectbox("Team", sorted(df[df['TM']!='-']['TM'].unique()), key='tt_tm')
        tdf = df[df['TM']==team]
        t1,t2 = st.tabs(["Batters","Pitchers"])
        with t1: st.dataframe(evaluate_batters(tdf).sort_values('GB_WAR',ascending=False), use_container_width=True)
        with t2: st.dataframe(evaluate_pitchers(tdf).sort_values('GB_WAR',ascending=False), use_container_width=True)

elif mode == "📝 Draft Board":
    st.header("Draft Board")
    uf = st.file_uploader("Draft Pool CSV", type=['csv'], key='draft')
    if uf:
        df = prep_data(pd.read_csv(uf, encoding='utf-8-sig', low_memory=False))
        t1,t2 = st.tabs(["Batters","Pitchers"])
        with t1:
            r = evaluate_batters(df); r = r[r['Age']<=22]
            r['AVOID'] = r.apply(lambda x: 'YES' if str(x.get('Type','')) in ['Unmotivated','Disruptive','Humble'] else '', axis=1)
            st.dataframe(r[['Name','POS','Age','F2','F2_Tier','GB_WAR','Best_F1','Best_Pos','Pos_Mult','AVOID','Flags']].sort_values('F2',ascending=False), use_container_width=True, height=600)
        with t2:
            r = evaluate_pitchers(df); r = r[r['Age']<=22]
            r['AVOID'] = r.apply(lambda x: 'YES' if str(x.get('Type','')) in ['Unmotivated','Disruptive','Humble'] else '', axis=1)
            st.dataframe(r[['Name','POS','Age','GB_WAR','SP_F1','MOV','STU','PIT_CON','STM','CON_Flag','AVOID','Flags']].sort_values('GB_WAR',ascending=False), use_container_width=True, height=600)

elif mode == "⚾ Lineup Optimizer":
    st.header("Lineup Optimizer (F1-Based)")
    df = get_data()
    if df is None: st.info("Load roster."); st.stop()
    team = st.selectbox("Team", sorted(df[df['TM']!='-']['TM'].unique()), key='lo_tm')
    r = evaluate_batters(df[df['TM']==team])
    pc = [f'F1_{p}' for p in POSITIONS]
    d = r[['Name','POS','Age']+pc+['Best_F1','Best_Pos','GB_WAR','Off_F1']].copy()
    for c in pc: d[c] = d[c].apply(lambda x: '' if x<=-50 else x)
    st.dataframe(d.sort_values('Best_F1',ascending=False), use_container_width=True, height=500)

elif mode == "🏟️ Lineup Construction":
    import lineup_construction as lc
    lc.render_lineup_construction(get_data())

elif mode == "⚙️ Pitching Staff":
    import pitching_staff as ps
    ps.render_staff_construction(get_data())

elif mode == "📊 Draft Value":
    import draft_value as dv
    dv.render_draft_value()

elif mode == "🔄 Trade Builder":
    import trade_builder as tb
    tb.render_trade_builder(get_data())

elif mode == "📖 Reference":
    st.header("System Reference")
    t1,t2,t3,t4 = st.tabs(["Formulas","Rules","Aging","Findings"])
    with t1:
        st.subheader("Batter F1 (R²=0.447)")
        st.code("= -14.168 + POW×0.1142 + BABIP×0.0725 + EYE×0.04 + CON×0.0379 + Ks×0.0317 + GAP×0.0291 + SPE×0.0128")
        st.subheader("SP F1"); st.code("= -5.932 + MOV×0.1091 + STM×0.056 + STU×0.0207 + PIT_CON×0.0053")
        st.subheader("RP F1"); st.code("= -2.852 + MOV×0.0509 + STU×0.0217 + PIT_CON×-0.0029")
        st.subheader("F2 v4 (R²=0.183)"); st.code("off_pot = POW_P×0.194 + GAP_P×0.234 + EYE_P×0.188 + CON_P×0.198 + BABIP_P×0.186\nF2 = -29.678 + off_pot×0.735 + Age×0.163 + WE×1.065 + INT×0.036 + AD×0.767 + best_DEF_WAR×4.007")
        st.subheader("Positional Multipliers"); st.dataframe(pd.DataFrame({'Pos':list(POS_MULTS.keys()),'Mult':list(POS_MULTS.values())}), hide_index=True)
    with t2:
        st.write("**Never draft:** Unmotivated (-0.51), Disruptive (-0.49). **Avoid:** Humble (-0.41). **Fragile:** ×0.60.")
        st.write("**Pitcher CON:** <40 RED, 40-44 YELLOW, 45+ OK, 50+ quality.")
        st.write("**Roster:** 6 SP + 5 RP + 14 POS. 6-man strict order. No SP in relief. Never 7+ RP.")
        st.write("**Closer:** Best STU arm. All other RP: MOV. STM <42 = bullpen only. HLD useless.")
        st.write("**Sell** batters 28-29. Hold high-EYE. Hold high-CON SP. Fan Fav: +0.70 WAR.")
    with t3:
        st.write("**Batters:** Peak 24-25. POW stable. EYE never declines. SPE declines from 25.")
        st.write("**SP:** Peak 27-28. MOV declines from 28. PIT_CON never declines (Jarry Effect). STM declines from 20.")
    with t4:
        st.write("**POW** king (batters). **MOV** king (pitchers, 3-5× STU). COMP anti-predictive. CERA debunked.")
        st.write("**6-man rotation:** +3.2 WAR/member. 37-GS cap. Extra rest doesn't help.")
        st.write("**Roster:** 13th pos player = 1.2 WAR vs 7th RP = 0.2 WAR. 407 PA from pitcher spot.")
        st.write("**Draft:** Picks 1-5 = 57.6% star. Cliff at #10. R3→R4 steepest. Never draft RP above R4.")
        st.write("**vR/vL:** Dead end (r=0.99). Use overall ratings for everything.")
