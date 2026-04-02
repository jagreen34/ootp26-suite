import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

st.set_page_config(page_title="OOTP 26 Evaluation Suite", page_icon="⚾", layout="wide")

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    gb_bat = joblib.load('models/gb_bat_season.pkl')
    gb_sp = joblib.load('models/gb_sp_season.pkl')
    bat_features = joblib.load('models/bat_features.pkl')
    sp_features = joblib.load('models/sp_features.pkl')
    return gb_bat, gb_sp, bat_features, sp_features

gb_bat, gb_sp, bat_features, sp_features = load_models()

# ============================================================
# CONSTANTS
# ============================================================
ZR_MODELS = {
    'C':  {'intercept': -7.32,  'coef': {'C ABI': 0.0628, 'C FRM': 0.0196, 'C ARM': 0.0539}},
    '1B': {'intercept': -13.43, 'coef': {'IF RNG': 0.2540, 'IF ERR': 0.0293, 'IF ARM': -0.0013}},
    '2B': {'intercept': -47.24, 'coef': {'IF RNG': 0.8635, 'IF ARM': 0.0644}},
    '3B': {'intercept': -31.44, 'coef': {'IF RNG': 0.3331, 'IF ARM': 0.2642}},
    'SS': {'intercept': -66.76, 'coef': {'IF RNG': 0.9064, 'IF ARM': 0.3012}},
    'LF': {'intercept': -29.47, 'coef': {'OF RNG': 0.6079, 'OF ARM': 0.0041}},
    'CF': {'intercept': -46.77, 'coef': {'OF RNG': 0.8833}},
    'RF': {'intercept': -52.45, 'coef': {'OF RNG': 0.9968, 'OF ARM': 0.0669}},
}
ZR_WAR = {'C': 0.1333, '1B': 0.1182, '2B': 0.1227, '3B': 0.1248, 'SS': 0.1040, 'LF': 0.1626, 'CF': 0.1111, 'RF': 0.1215}
FLOORS = {'C': ('C ABI', 45), '2B': ('IF RNG', 50), 'SS': ('IF RNG', 55), 'CF': ('OF RNG', 55)}
POSITIONS = ['C','1B','2B','3B','SS','LF','CF','RF']
POS_MULTS = {'SS': 1.50, 'CF': 1.55, 'C': 1.30, '2B': 1.30, '3B': 1.20, 'RF': 1.25, 'LF': 1.05, '1B': 1.00}

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def safe(v, d=0):
    try:
        if pd.isna(v): return d
        return float(v)
    except: return d

def calc_def_war(row, pos):
    m = ZR_MODELS[pos]; zr = m['intercept']
    for f, c in m['coef'].items():
        zr += safe(row.get(f, 0)) * c
    return zr * ZR_WAR[pos]

def best_def_war(row):
    return max(calc_def_war(row, pos) for pos in POSITIONS)

def f1_at_pos(row, pos):
    if pos in FLOORS:
        r, m = FLOORS[pos]
        if safe(row.get(r, 0)) < m: return -99
    off = (-14.168 + safe(row['POW'])*0.1142 + safe(row['BABIP'])*0.0725 + safe(row['EYE'])*0.04
        + safe(row['CON'])*0.0379 + safe(row.get('Ks',0))*0.0317 + safe(row['GAP'])*0.0291 + safe(row['SPE'])*0.0128)
    return off + calc_def_war(row, pos)

def off_f1(row):
    return (-14.168 + safe(row['POW'])*0.1142 + safe(row['BABIP'])*0.0725 + safe(row['EYE'])*0.04
        + safe(row['CON'])*0.0379 + safe(row.get('Ks',0))*0.0317 + safe(row['GAP'])*0.0291 + safe(row['SPE'])*0.0128)

def best_position(row):
    bf1 = -99; bp = '1B'
    for pos in POSITIONS:
        f1 = f1_at_pos(row, pos)
        if f1 > bf1: bf1 = f1; bp = pos
    return bp, bf1

def sp_f1(row):
    return -5.932 + safe(row['MOV'])*0.1091 + safe(row['STM'])*0.056 + safe(row['STU'])*0.0207 + safe(row['CON'])*0.0053

def rp_f1(row):
    return -2.852 + safe(row['MOV'])*0.0509 + safe(row['STU'])*0.0217 + safe(row['CON'])*-0.0029

def bat_f2_v4(row):
    off_comp = safe(row.get('POW P',0))*0.194 + safe(row.get('GAP P',0))*0.234 + safe(row.get('EYE P',0))*0.188 + safe(row.get('CON P',0))*0.198 + safe(row.get('HT P',0))*0.186
    we = {'H':1,'N':0,'L':-1}.get(str(row.get('WE','')),0)
    int_v = {'H':1,'N':0,'L':-1}.get(str(row.get('INT','')),0)
    ad_v = {'H':1,'N':0,'L':-1}.get(str(row.get('AD','')),0)
    bdw = best_def_war(row)
    f2 = -29.678 + off_comp*0.735 + safe(row['Age'])*0.163 + we*1.065 + int_v*0.036 + ad_v*0.767 + bdw*4.007
    if str(row.get('Prone','')) == 'Fragile': f2 *= 0.60
    return f2

def f2_tier(f2):
    if f2 >= 25: return 'S'
    elif f2 >= 15: return 'A'
    elif f2 >= 5: return 'B'
    else: return 'C'

def con_flag(con):
    if con < 40: return 'RED'
    elif con < 45: return 'YELLOW'
    return 'OK'

def prep_bat_data(df):
    for col in ['CON','BABIP','GAP','POW','EYE','WAR','Age','PA','SPE','STE',
                'IF RNG','OF RNG','C ABI','IF ARM','IF ERR','OF ARM','C FRM','C ARM',
                'POW P','GAP P','EYE P','CON P','HT P']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Ks'] = pd.to_numeric(df.get("K's"), errors='coerce')
    if 'STE' not in df.columns: df['STE'] = 40
    if 'EXP' in df.columns: df['EXP'] = pd.to_numeric(df['EXP'], errors='coerce')
    if 'Draft' in df.columns: df['Draft'] = pd.to_numeric(df['Draft'], errors='coerce')
    return df

def prep_pitch_data(df):
    for col in ['STU','MOV','CON','STM','Age','WAR','IP','STU P','MOV P','CON P']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'EXP' in df.columns: df['EXP'] = pd.to_numeric(df['EXP'], errors='coerce')
    return df

def evaluate_batters(df):
    df = prep_bat_data(df)
    pos_players = df[~df['POS'].isin(['SP','RP','CL'])].copy()
    
    # F1 at all positions
    for pos in POSITIONS:
        pos_players[f'F1_{pos}'] = pos_players.apply(lambda r: round(f1_at_pos(r, pos), 1), axis=1)
    
    pos_players['Off_F1'] = pos_players.apply(lambda r: round(off_f1(r), 1), axis=1)
    bp_data = pos_players.apply(lambda r: best_position(r), axis=1)
    pos_players['Best_Pos'] = [x[0] for x in bp_data]
    pos_players['Best_F1'] = [round(x[1], 1) for x in bp_data]
    pos_players['Pos_Mult'] = pos_players['Best_Pos'].map(POS_MULTS)
    pos_players['Useful_Pos'] = pos_players.apply(lambda r: sum(1 for pos in POSITIONS if f1_at_pos(r, pos) > 1.0), axis=1)
    
    # F2 for prospects
    pos_players['F2'] = pos_players.apply(lambda r: round(bat_f2_v4(r), 1) if safe(r['Age']) <= 26 else 0, axis=1)
    pos_players['F2_Tier'] = pos_players['F2'].apply(f2_tier)
    
    # GB prediction
    pos_players['best_def'] = pos_players.apply(best_def_war, axis=1)
    pos_players['we_val'] = pos_players['WE'].map({'H':1,'N':0,'L':-1}).fillna(0) if 'WE' in pos_players.columns else 0
    pos_players['int_val'] = pos_players['INT'].map({'H':1,'N':0,'L':-1}).fillna(0) if 'INT' in pos_players.columns else 0
    pos_players['ad_val'] = pos_players['AD'].map({'H':1,'N':0,'L':-1}).fillna(0) if 'AD' in pos_players.columns else 0
    for ptype in ['Fan Fav','Captain','Sparkplug','Humble','Unmotivated','Disruptive','Selfish']:
        col_name = f'is_{ptype.replace(" ","_").lower()}'
        pos_players[col_name] = (pos_players['Type'] == ptype).astype(int) if 'Type' in pos_players.columns else 0
    pos_players['is_fragile'] = (pos_players['Prone'] == 'Fragile').astype(int) if 'Prone' in pos_players.columns else 0
    
    X = pos_players[bat_features].fillna(0)
    pos_players['GB_WAR'] = np.round(gb_bat.predict(X), 1)
    
    # Trade value
    pos_players['Trade_Val'] = pos_players.apply(lambda r: round(max(0, (r['Best_F1'] - 0.2) * r['Pos_Mult']), 1), axis=1)
    
    # Flags
    pos_players['Flags'] = pos_players.apply(lambda r: ', '.join(filter(None, [
        'WE:H' if str(r.get('WE','')) == 'H' else '',
        'INT:H' if str(r.get('INT','')) == 'H' else '',
        f'FLEX({r["Useful_Pos"]})' if r['Useful_Pos'] >= 5 else '',
        'FAN FAV' if str(r.get('Type','')) == 'Fan Fav' else '',
        'HUMBLE' if str(r.get('Type','')) == 'Humble' else '',
        'UNMOTIVATED' if str(r.get('Type','')) == 'Unmotivated' else '',
        'FRAGILE' if str(r.get('Prone','')) == 'Fragile' else '',
    ])), axis=1)
    
    return pos_players

def evaluate_pitchers(df):
    df = prep_pitch_data(df)
    pitchers = df[df['POS'].isin(['SP','RP','CL'])].copy()
    
    pitchers['SP_F1'] = pitchers.apply(lambda r: round(sp_f1(r), 1), axis=1)
    pitchers['RP_F1'] = pitchers.apply(lambda r: round(rp_f1(r), 1), axis=1)
    pitchers['CON_Flag'] = pitchers['CON'].apply(con_flag)
    
    # GB prediction
    pitchers['we_val'] = pitchers['WE'].map({'H':1,'N':0,'L':-1}).fillna(0) if 'WE' in pitchers.columns else 0
    pitchers['int_val'] = pitchers['INT'].map({'H':1,'N':0,'L':-1}).fillna(0) if 'INT' in pitchers.columns else 0
    pitchers['ad_val'] = pitchers['AD'].map({'H':1,'N':0,'L':-1}).fillna(0) if 'AD' in pitchers.columns else 0
    for ptype in ['Fan Fav','Captain','Humble','Unmotivated']:
        col_name = f'is_{ptype.replace(" ","_").lower()}'
        pitchers[col_name] = (pitchers['Type'] == ptype).astype(int) if 'Type' in pitchers.columns else 0
    pitchers['is_fragile'] = (pitchers['Prone'] == 'Fragile').astype(int) if 'Prone' in pitchers.columns else 0
    
    X = pitchers[sp_features].fillna(0)
    pitchers['GB_WAR'] = np.round(gb_sp.predict(X), 1)
    
    # Trade value for SP
    pitchers['Trade_Val'] = pitchers.apply(lambda r: round(max(0, r['SP_F1'] - 0.9), 1), axis=1)
    
    pitchers['Flags'] = pitchers.apply(lambda r: ', '.join(filter(None, [
        'WE:H' if str(r.get('WE','')) == 'H' else '',
        'INT:H' if str(r.get('INT','')) == 'H' else '',
        f'CON:{int(safe(r["CON"]))}' if safe(r['CON']) >= 50 else '',
        'FAN FAV' if str(r.get('Type','')) == 'Fan Fav' else '',
        'HUMBLE' if str(r.get('Type','')) == 'Humble' else '',
        'UNMOTIVATED' if str(r.get('Type','')) == 'Unmotivated' else '',
        'FRAGILE' if str(r.get('Prone','')) == 'Fragile' else '',
    ])), axis=1)
    
    return pitchers

# ============================================================
# UI
# ============================================================
st.title("⚾ OOTP 26 Evaluation Suite")
st.caption("Philadelphia Quakers — v4 System (GB + Linear F1 + F2)")

mode = st.sidebar.radio("Mode", [
    "🏠 Quick Eval",
    "📋 Offseason Phase 1",
    "🎯 Trade Targets",
    "📝 Draft Board",
    "⚾ Lineup Optimizer",
    "🏟️ Lineup Construction",
    "⚙️ Pitching Staff",
    "📖 Reference"
])

# ============================================================
# QUICK EVAL
# ============================================================
if mode == "🏠 Quick Eval":
    st.header("Quick Evaluation")
    st.write("Upload any CSV (team roster, league-wide, or custom player list) and get full analysis.")
    
    col1, col2 = st.columns(2)
    with col1:
        bat_file = st.file_uploader("Batter CSV", type=['csv'])
    with col2:
        pitch_file = st.file_uploader("Pitcher CSV", type=['csv'])
    
    if bat_file:
        df = pd.read_csv(bat_file, encoding='utf-8-sig')
        results = evaluate_batters(df)
        
        display_cols = ['Name','TM','POS','Age','GB_WAR','Best_F1','Off_F1','Best_Pos','Pos_Mult',
                       'Useful_Pos','F2','F2_Tier','WAR','PA','POW','CON','EYE','GAP','SPE','Flags']
        display_cols = [c for c in display_cols if c in results.columns]
        
        st.subheader(f"Batters ({len(results)} players)")
        st.dataframe(results[display_cols].sort_values('GB_WAR', ascending=False), use_container_width=True, height=500)
        
        # Download
        output = io.BytesIO()
        results[display_cols].to_excel(output, index=False)
        st.download_button("📥 Download Batter Results", output.getvalue(), "batter_eval.xlsx")
    
    if pitch_file:
        df = pd.read_csv(pitch_file, encoding='utf-8-sig')
        results = evaluate_pitchers(df)
        
        display_cols = ['Name','TM','POS','Age','GB_WAR','SP_F1','RP_F1','MOV','STU','CON','STM',
                       'CON_Flag','WAR','IP','Flags']
        display_cols = [c for c in display_cols if c in results.columns]
        
        st.subheader(f"Pitchers ({len(results)} players)")
        st.dataframe(results[display_cols].sort_values('GB_WAR', ascending=False), use_container_width=True, height=500)
        
        output = io.BytesIO()
        results[display_cols].to_excel(output, index=False)
        st.download_button("📥 Download Pitcher Results", output.getvalue(), "pitcher_eval.xlsx")

# ============================================================
# OFFSEASON PHASE 1
# ============================================================
elif mode == "📋 Offseason Phase 1":
    st.header("Offseason Phase 1: Post-Season Evaluation")
    st.write("Upload team + league CSVs for full offseason analysis.")
    
    col1, col2 = st.columns(2)
    with col1:
        team_bat = st.file_uploader("Team Batter CSV", type=['csv'], key='tb')
        team_pitch = st.file_uploader("Team Pitcher CSV", type=['csv'], key='tp')
    with col2:
        league_bat = st.file_uploader("League Batter CSV", type=['csv'], key='lb')
        league_pitch = st.file_uploader("League Pitcher CSV", type=['csv'], key='lp')
    
    division = st.multiselect("Your Division Rivals (exclude from targets)", 
        ['Pittsburgh','Cleveland','New York','Brooklyn','Cincinnati','Washington',
         'Chicago','Denver','Kansas City','San Diego','St. Louis','Los Angeles','Milwaukee',
         'San Francisco','Dallas','Houston','Seattle','Twin Cities','Baltimore','Montreal',
         'Toronto','Boston','Atlanta','Detroit'],
        default=['Pittsburgh','Cleveland','New York','Brooklyn','Cincinnati','Washington'])
    
    if team_bat and team_pitch:
        st.subheader("Your Roster")
        
        tb = evaluate_batters(pd.read_csv(team_bat, encoding='utf-8-sig'))
        tp = evaluate_pitchers(pd.read_csv(team_pitch, encoding='utf-8-sig'))
        
        tab1, tab2, tab3 = st.tabs(["Batters", "Pitchers", "Prospects"])
        
        with tab1:
            cols = ['Name','POS','Age','GB_WAR','Best_F1','Off_F1','Best_Pos','Pos_Mult','Useful_Pos','WAR','PA','Flags']
            cols = [c for c in cols if c in tb.columns]
            st.dataframe(tb[cols].sort_values('GB_WAR', ascending=False), use_container_width=True)
        
        with tab2:
            cols = ['Name','POS','Age','GB_WAR','SP_F1','RP_F1','MOV','STU','CON','STM','CON_Flag','WAR','IP','Flags']
            cols = [c for c in cols if c in tp.columns]
            st.dataframe(tp[cols].sort_values('GB_WAR', ascending=False), use_container_width=True)
        
        with tab3:
            prospects = tb[tb['Age'] <= 26].copy()
            cols = ['Name','POS','Age','F2','F2_Tier','GB_WAR','Best_F1','Best_Pos','Pos_Mult','Useful_Pos','Flags']
            cols = [c for c in cols if c in prospects.columns]
            st.dataframe(prospects[cols].sort_values('F2', ascending=False), use_container_width=True)
    
    if league_bat and league_pitch:
        st.subheader("Trade Targets (Non-Division)")
        
        lb_df = evaluate_batters(pd.read_csv(league_bat, encoding='utf-8-sig'))
        lp_df = evaluate_pitchers(pd.read_csv(league_pitch, encoding='utf-8-sig'))
        
        # Filter: non-division, non-Philadelphia
        lb_targets = lb_df[(~lb_df['TM'].isin(division)) & (lb_df['TM'] != 'Philadelphia') & (lb_df['TM'] != '-')]
        lp_targets = lp_df[(~lp_df['TM'].isin(division)) & (lp_df['TM'] != 'Philadelphia') & (lp_df['TM'] != '-')]
        
        tab4, tab5, tab6 = st.tabs(["SP Targets", "Catcher Targets", "Bat Targets"])
        
        with tab4:
            sp_min_f1 = st.slider("Min SP F1", 2.0, 7.0, 3.5, 0.5)
            sp_min_con = st.slider("Min CON", 30, 70, 40, 5)
            sp_max_age = st.slider("Max Age", 20, 35, 27)
            
            sp_t = lp_targets[(lp_targets['SP_F1'] >= sp_min_f1) & (lp_targets['CON'] >= sp_min_con) & (lp_targets['Age'] <= sp_max_age)]
            cols = ['Name','TM','POS','Age','GB_WAR','SP_F1','MOV','STU','CON','STM','CON_Flag','WAR','IP','Flags']
            cols = [c for c in cols if c in sp_t.columns]
            st.dataframe(sp_t[cols].sort_values('GB_WAR', ascending=False), use_container_width=True)
        
        with tab5:
            c_min_f1 = st.slider("Min Off F1", 0.5, 4.0, 1.5, 0.5)
            c_min_abi = st.slider("Min C ABI", 40, 80, 50, 5)
            c_max_age = st.slider("Max Catcher Age", 20, 35, 30)
            
            # Find catchers
            catchers = lb_targets[lb_targets.apply(lambda r: safe(r.get('C ABI', 0)) >= c_min_abi, axis=1)]
            catchers = catchers[(catchers['Off_F1'] >= c_min_f1) & (catchers['Age'] <= c_max_age)]
            cols = ['Name','TM','POS','Age','GB_WAR','Off_F1','Best_F1','Best_Pos','WAR','PA','Flags']
            cols = [c for c in cols if c in catchers.columns]
            st.dataframe(catchers[cols].sort_values('Off_F1', ascending=False), use_container_width=True)
        
        with tab6:
            b_min_f1 = st.slider("Min Batter F1", 1.0, 6.0, 2.0, 0.5)
            b_max_age = st.slider("Max Batter Age", 20, 35, 28)
            
            bt = lb_targets[(lb_targets['Best_F1'] >= b_min_f1) & (lb_targets['Age'] <= b_max_age)]
            cols = ['Name','TM','POS','Age','GB_WAR','Best_F1','Off_F1','Best_Pos','Pos_Mult','Useful_Pos','F2','WAR','PA','Flags']
            cols = [c for c in cols if c in bt.columns]
            st.dataframe(bt[cols].sort_values('GB_WAR', ascending=False).head(50), use_container_width=True)

# ============================================================
# TRADE TARGETS
# ============================================================
elif mode == "🎯 Trade Targets":
    st.header("Trade Target Evaluation")
    st.write("Upload a CSV of specific players to evaluate for a trade.")
    
    target_type = st.radio("Player Type", ["Batters", "Pitchers"])
    target_file = st.file_uploader("Target Players CSV", type=['csv'])
    
    if target_file:
        df = pd.read_csv(target_file, encoding='utf-8-sig')
        if target_type == "Batters":
            results = evaluate_batters(df)
            cols = ['Name','TM','POS','Age','GB_WAR','Best_F1','Off_F1','Best_Pos','Pos_Mult',
                   'Trade_Val','Useful_Pos','F2','F2_Tier','WAR','PA','Flags']
        else:
            results = evaluate_pitchers(df)
            cols = ['Name','TM','POS','Age','GB_WAR','SP_F1','RP_F1','MOV','STU','CON','STM',
                   'CON_Flag','Trade_Val','WAR','IP','Flags']
        
        cols = [c for c in cols if c in results.columns]
        st.dataframe(results[cols].sort_values('GB_WAR', ascending=False), use_container_width=True)
        
        output = io.BytesIO()
        results[cols].to_excel(output, index=False)
        st.download_button("📥 Download Results", output.getvalue(), "trade_targets.xlsx")

# ============================================================
# DRAFT BOARD
# ============================================================
elif mode == "📝 Draft Board":
    st.header("Draft Board")
    st.write("Upload the draft pool CSV. Ranks all prospects by F2 with GB career projection.")
    
    draft_file = st.file_uploader("Draft Pool CSV (Batters)", type=['csv'], key='draft_bat')
    draft_pitch = st.file_uploader("Draft Pool CSV (Pitchers)", type=['csv'], key='draft_pitch')
    
    if draft_file:
        df = pd.read_csv(draft_file, encoding='utf-8-sig')
        results = evaluate_batters(df)
        
        # Filter to draft-eligible age
        results = results[results['Age'] <= 22]
        
        # Flag personality issues
        results['AVOID'] = results.apply(lambda r: 'YES' if str(r.get('Type','')) in ['Unmotivated','Disruptive','Humble'] else '', axis=1)
        
        cols = ['Name','POS','Age','F2','F2_Tier','GB_WAR','Best_F1','Off_F1','Best_Pos','Pos_Mult',
               'Useful_Pos','AVOID','Flags']
        cols = [c for c in cols if c in results.columns]
        
        st.subheader("Batter Prospects")
        st.dataframe(results[cols].sort_values('F2', ascending=False), use_container_width=True, height=600)
    
    if draft_pitch:
        df = pd.read_csv(draft_pitch, encoding='utf-8-sig')
        results = evaluate_pitchers(df)
        results = results[results['Age'] <= 22]
        results['AVOID'] = results.apply(lambda r: 'YES' if str(r.get('Type','')) in ['Unmotivated','Disruptive','Humble'] else '', axis=1)
        
        cols = ['Name','POS','Age','GB_WAR','SP_F1','MOV','STU','CON','STM','CON_Flag','AVOID','Flags']
        cols = [c for c in cols if c in results.columns]
        
        st.subheader("Pitcher Prospects")
        st.dataframe(results[cols].sort_values('GB_WAR', ascending=False), use_container_width=True, height=600)

# ============================================================
# LINEUP OPTIMIZER (legacy F1-based)
# ============================================================
elif mode == "⚾ Lineup Optimizer":
    st.header("Lineup Optimizer")
    st.write("Upload your team batter CSV. Shows optimal position assignment.")
    
    lineup_file = st.file_uploader("Team Batter CSV", type=['csv'])
    
    if lineup_file:
        df = pd.read_csv(lineup_file, encoding='utf-8-sig')
        results = evaluate_batters(df)
        
        st.subheader("F1 at Every Position")
        pos_cols = [f'F1_{pos}' for pos in POSITIONS]
        display = results[['Name','POS','Age'] + pos_cols + ['Best_F1','Best_Pos','GB_WAR','Off_F1']].copy()
        
        # Replace -99 with blank for display
        for col in pos_cols:
            display[col] = display[col].apply(lambda x: '' if x <= -50 else x)
        
        st.dataframe(display.sort_values('Best_F1', ascending=False), use_container_width=True, height=500)
        
        st.subheader("Recommended Starters")
        st.write("Assign each position to maximize total F1:")
        
        starters = results[results['Age'] >= 20].nlargest(12, 'Best_F1')
        assigned = {}
        used = set()
        
        for pos in ['SS','CF','C','2B','3B','RF','LF','1B']:
            best_val = -99; best_name = ''
            for _, r in starters.iterrows():
                name = str(r['Name'])
                if name in used: continue
                f1 = f1_at_pos(r, pos)
                if f1 > best_val: best_val = f1; best_name = name
            if best_name:
                assigned[pos] = (best_name, round(best_val, 1))
                used.add(best_name)
        
        total = 0
        for pos in POSITIONS:
            if pos in assigned:
                name, f1 = assigned[pos]
                st.write(f"**{pos}:** {name} (F1 = {f1})")
                total += f1
        st.write(f"**Total Lineup F1: {total:.1f}**")

# ============================================================
# LINEUP CONSTRUCTION (new — Hungarian + batting order)
# ============================================================
elif mode == "🏟️ Lineup Construction":
    import lineup_construction as lc
    lc.render_lineup_construction()

# ============================================================
# PITCHING STAFF CONSTRUCTION
# ============================================================
elif mode == "⚙️ Pitching Staff":
    import pitching_staff as ps
    ps.render_staff_construction()

# ============================================================
# REFERENCE
# ============================================================
elif mode == "📖 Reference":
    st.header("System Reference")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Formulas", "Rules", "Aging Curves", "Study Findings"])
    
    with tab1:
        st.subheader("Batter Offensive F1 (Linear, R²=0.447)")
        st.code("= -14.168 + (POW×0.1142) + (BABIP×0.0725) + (EYE×0.04) + (CON×0.0379) + (Ks×0.0317) + (GAP×0.0291) + (SPE×0.0128)")
        
        st.subheader("SP F1 (Linear)")
        st.code("= -5.932 + (MOV×0.1091) + (STM×0.056) + (STU×0.0207) + (CON×0.0053)")
        
        st.subheader("RP F1 (Linear)")
        st.code("= -2.852 + (MOV×0.0509) + (STU×0.0217) + (CON×-0.0029)")
        
        st.subheader("Batter F2 v4 (Career Prediction, R²=0.183)")
        st.code("off_pot = (POW_P×0.194) + (GAP_P×0.234) + (EYE_P×0.188) + (CON_P×0.198) + (BABIP_P×0.186)")
        st.code("F2 = -29.678 + (off_pot×0.735) + (Age×0.163) + (WE×1.065) + (INT×0.036) + (AD×0.767) + (best_DEF_WAR×4.007)")
        
        st.subheader("GB Season Model (R²=0.69 bat, 0.75 pitch)")
        st.write("Gradient Boosting with 300 trees. Uses current ratings + age + personality + defense. Cannot be expressed as a formula — use this tool.")
        
        st.subheader("Trade Value")
        st.code("Bat TV = (F1 - 0.2) × Years × Pos_Mult\nSP TV = (F1 - 0.9) × Years\nRP TV = (F1 + 0.8) × Years")
        
        st.subheader("Positional Multipliers (253K validated)")
        mult_df = pd.DataFrame({'Position': ['SS','CF','C','2B','3B','RF','LF','1B'],
                                'Multiplier': [1.50, 1.55, 1.30, 1.30, 1.20, 1.25, 1.05, 1.00]})
        st.dataframe(mult_df, hide_index=True)
    
    with tab2:
        st.subheader("Filters (Never Draft / Avoid)")
        st.write("- **Unmotivated:** -0.51 WAR/season. Never draft.")
        st.write("- **Disruptive:** -0.49 WAR/season. Never draft.")
        st.write("- **Humble:** -0.41 WAR/season. Avoid.")
        st.write("- **Fragile:** ×0.60 career WAR. Discount, don't filter (commissioner removes in draft pool).")
        
        st.subheader("CON Thresholds (Pitchers)")
        st.write("- **CON < 40:** RED — walk machine. GB model shows non-linear cliff.")
        st.write("- **CON 40-44:** YELLOW — marginal. May develop if CON P 50+.")
        st.write("- **CON 45+:** OK — competent.")
        st.write("- **CON 50+:** Quality — target for trades.")
        
        st.subheader("Defensive Floors")
        st.write("- **SS:** IF RNG 60+ (competent), 55+ (barely playable)")
        st.write("- **CF:** OF RNG 60+ (competent), 55+ (barely playable)")
        st.write("- **2B:** IF RNG 55+ (competent), 50+ (barely playable)")
        st.write("- **C:** C ABI 50+ (competent), 45+ (barely playable)")
        
        st.subheader("F2 Tiers")
        st.write("- **S ≥ 25:** Franchise prospect (avg 29+ career WAR)")
        st.write("- **A ≥ 15:** Future starter (avg 17+ career WAR)")
        st.write("- **B ≥ 5:** Contributor (avg 8+ career WAR)")
        st.write("- **C < 5:** Longshot")
        
        st.subheader("Sell/Hold Rules")
        st.write("- Sell batters at age 28-29 (decline starts 29)")
        st.write("- Sell SP when MOV starts dropping (check year-over-year)")
        st.write("- Hold high-EYE batters longer (EYE never declines)")
        st.write("- Hold high-CON pitchers longer (CON holds to 32+)")
        st.write("- Fan Favorite: +0.70 WAR/season bonus — value these players")
    
    with tab3:
        st.subheader("Batter Aging")
        st.write("Peak WAR: age 24-25. Hold through 28. Decline from 29+.")
        st.write("EYE rises from 35 → 62 through age 40. Never declines.")
        st.write("POW barely moves (40.5 → 37.0 from 25 → 40).")
        st.write("SPE declines steadily from 25 onward.")
        
        st.subheader("SP Aging")
        st.write("Peak WAR: age 27-28.")
        st.write("MOV declines from age 28 (-0.36/yr, accelerating to -1.3/yr by 35).")
        st.write("STU declines from age 28 (-0.11/yr).")
        st.write("CON NEVER declines (Jarry Effect). Rises from 33.6 → 47.4 through age 40.")
        st.write("STM declines from age 20 (55.5 → 46.5).")
    
    with tab4:
        st.subheader("Key Study Findings (253K rows)")
        st.write("- **MOV is 3-5x more valuable than STU per rating point for SP**")
        st.write("- **COMP is anti-predictive** — Average COMP outperforms Great by +0.83 WAR")
        st.write("- **Fan Favorite is real:** +0.70 WAR/season after controlling for talent")
        st.write("- **Catcher defense does NOT help pitching** (CERA study debunked)")
        st.write("- **WE:H adds ~1.5 rating points of development** (tiebreaker, not dealmaker)")
        st.write("- **INT:H adds ~1.3 STU development points for pitchers**")
        st.write("- **SB only profitable at STE 65+** (not SPE)")
        st.write("- **Offense is a commodity (~90 players at 3.0+ F1)**. Defense at premium positions is scarce.")
        st.write("- **Positional scarcity per year at 3.0+ F1:** C=1.8, SS=10.4, CF=19.6, 1B=26.0, RF=41.5")
