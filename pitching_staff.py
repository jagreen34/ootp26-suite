"""
OOTP 26 Pitching Staff Construction
=====================================
Given a pitcher roster, assigns:
  - Rotation (configurable 3-6 man)
  - Bullpen roles (Closer, Setup, MR, LR, Mop-Up)
  - Staff card with predicted WAR per role

All models from 253K pitcher-season simulation study.
"""

import numpy as np
import pandas as pd

# ============================================================================
# MODEL COEFFICIENTS (from 253K study, full precision)
# ============================================================================

ROLE_MODELS = {
    'Closer':        {'intercept': -5.2667, 'STU': 0.0351, 'MOV': 0.0502, 'CON': 0.0428, 'STM': 0.0011, 'R2': 0.444, 'n': 4202},
    'Setup':         {'intercept': -4.3989, 'STU': 0.0275, 'MOV': 0.0474, 'CON': 0.0291, 'STM': -0.0035, 'R2': 0.371, 'n': 5127},
    'Middle Relief': {'intercept': -4.4954, 'STU': 0.0247, 'MOV': 0.0561, 'CON': 0.0223, 'STM': -0.0022, 'R2': 0.360, 'n': 9488},
    'Long Relief':   {'intercept': -4.6718, 'STU': 0.0278, 'MOV': 0.0597, 'CON': 0.0219, 'STM': -0.0044, 'R2': 0.323, 'n': 553},
    'Mop-Up':        {'intercept': -4.5584, 'STU': 0.0261, 'MOV': 0.0586, 'CON': 0.0193, 'STM': -0.0040, 'R2': 0.357, 'n': 3143},
    'Starter':       {'intercept': -1.2415, 'STU': -0.0108, 'MOV': 0.0689, 'CON': -0.0276, 'STM': 0.0578, 'R2': 0.146, 'n': 37558},
}

RELIEF_ROLES = ['Closer', 'Setup', 'Middle Relief', 'Long Relief', 'Mop-Up']
PITCH_FEATURES = ['STU', 'MOV', 'CON', 'STM']
STM_THRESHOLD = 42

# Bullpen role slots by bullpen size
ROLE_TEMPLATES = {
    3: ['Closer', 'Setup', 'Middle Relief'],
    4: ['Closer', 'Setup', 'Middle Relief', 'Long Relief'],
    5: ['Closer', 'Setup', 'Middle Relief', 'Middle Relief', 'Long Relief'],
    6: ['Closer', 'Setup', 'Middle Relief', 'Middle Relief', 'Long Relief', 'Long Relief'],
    7: ['Closer', 'Setup', 'Setup', 'Middle Relief', 'Middle Relief', 'Long Relief', 'Long Relief'],
    8: ['Closer', 'Setup', 'Setup', 'Middle Relief', 'Middle Relief', 'Long Relief', 'Long Relief', 'Mop-Up'],
    9: ['Closer', 'Setup', 'Setup', 'Middle Relief', 'Middle Relief', 'Long Relief', 'Long Relief', 'Mop-Up', 'Mop-Up'],
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def predict_role_war(row, role):
    """Predict WAR for a pitcher in a given role."""
    m = ROLE_MODELS[role]
    war = m['intercept']
    for feat in PITCH_FEATURES:
        val = pd.to_numeric(row.get(feat, 0), errors='coerce')
        if pd.isna(val):
            val = 0
        war += val * m[feat]
    return war


def sp_f1(row):
    """SP F1 linear model."""
    mov = pd.to_numeric(row.get('MOV', 0), errors='coerce') or 0
    stm = pd.to_numeric(row.get('STM', 0), errors='coerce') or 0
    stu = pd.to_numeric(row.get('STU', 0), errors='coerce') or 0
    con = pd.to_numeric(row.get('CON', 0), errors='coerce') or 0
    return -5.932 + mov * 0.1091 + stm * 0.056 + stu * 0.0207 + con * 0.0053


def rp_f1(row):
    """RP F1 linear model."""
    mov = pd.to_numeric(row.get('MOV', 0), errors='coerce') or 0
    stu = pd.to_numeric(row.get('STU', 0), errors='coerce') or 0
    con = pd.to_numeric(row.get('CON', 0), errors='coerce') or 0
    return -2.852 + mov * 0.0509 + stu * 0.0217 + con * -0.0029


def build_staff(roster_df, rotation_size=6, pos_players=13):
    """
    Build optimal pitching staff.
    
    Args:
        roster_df: Full roster DataFrame (pitchers will be filtered)
        rotation_size: Number of starters (3-6)
        pos_players: Number of position players on 25-man roster
    
    Returns:
        dict with rotation, bullpen assignments, and analysis
    """
    # Filter to pitchers
    pitchers = roster_df[roster_df['POS'].isin(['SP', 'RP', 'CL'])].copy()
    for col in PITCH_FEATURES + ['Age', 'WAR', 'IP', 'ERA']:
        if col in pitchers.columns:
            pitchers[col] = pd.to_numeric(pitchers[col], errors='coerce')
    
    # Calculate all projections
    for idx, row in pitchers.iterrows():
        pitchers.loc[idx, 'SP_F1'] = round(sp_f1(row), 2)
        pitchers.loc[idx, 'RP_F1'] = round(rp_f1(row), 2)
        pitchers.loc[idx, 'Starter_WAR'] = round(predict_role_war(row, 'Starter'), 2)
        for role in RELIEF_ROLES:
            pitchers.loc[idx, f'{role}_WAR'] = round(predict_role_war(row, role), 2)
        
        # Best relief role
        best_relief = max(RELIEF_ROLES, key=lambda r: predict_role_war(row, r))
        pitchers.loc[idx, 'Best_Relief'] = best_relief
        pitchers.loc[idx, 'Best_Relief_WAR'] = round(predict_role_war(row, best_relief), 2)
    
    # SP candidates: STM >= threshold
    sp_pool = pitchers[pitchers['STM'] >= STM_THRESHOLD].sort_values('SP_F1', ascending=False)
    bp_only = pitchers[pitchers['STM'] < STM_THRESHOLD].copy()
    
    # Select rotation
    rotation = sp_pool.head(rotation_size)
    remaining_sp = sp_pool.iloc[rotation_size:]
    
    # Bullpen pool: explicit relievers + SP overflow
    bp_pool = pd.concat([bp_only, remaining_sp])
    bp_size = 25 - pos_players - rotation_size
    
    # Get role template
    if bp_size in ROLE_TEMPLATES:
        roles_to_fill = ROLE_TEMPLATES[bp_size].copy()
    else:
        # Fallback: build a reasonable template
        roles_to_fill = ['Closer', 'Setup']
        remaining = bp_size - 2
        while remaining > 0:
            if remaining >= 2:
                roles_to_fill.extend(['Middle Relief', 'Long Relief'])
                remaining -= 2
            else:
                roles_to_fill.append('Mop-Up')
                remaining -= 1
    
    # Assign bullpen roles greedily
    assignments = []
    used = set()
    
    for role in roles_to_fill:
        sort_col = f'{role}_WAR'
        best = None
        for _, r in bp_pool.sort_values(sort_col, ascending=False).iterrows():
            if r['Name'] not in used:
                best = r
                used.add(r['Name'])
                break
        if best is not None:
            war = predict_role_war(best, role)
            assignments.append({
                'role': role,
                'name': best['Name'],
                'age': best.get('Age', ''),
                'stu': int(best['STU']),
                'mov': int(best['MOV']),
                'con': int(best['CON']),
                'stm': int(best['STM']),
                'war': round(war, 2),
                'sp_capable': best['STM'] >= STM_THRESHOLD,
                'row': best,
            })
    
    return {
        'rotation': rotation,
        'bullpen': assignments,
        'rotation_size': rotation_size,
        'bp_size': bp_size,
        'all_pitchers': pitchers,
        'sp_pool': sp_pool,
    }


# ============================================================================
# STREAMLIT UI
# ============================================================================

def render_staff_construction(uploaded_df=None):
    """Render the pitching staff construction tool in Streamlit."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not available.")
        return
    
    st.header("⚾ Pitching Staff Construction")
    st.caption(
        "Rotation + bullpen role assignment. "
        "Models: SP F1 (linear), role-specific WAR (253K study), STM threshold = 42."
    )
    
    # File upload or passed DataFrame
    if uploaded_df is not None:
        df = uploaded_df
    else:
        f = st.file_uploader("Upload Pitcher CSV (OOTP export)", type=['csv'], key='psc_upload')
        if not f:
            st.info("Upload a pitcher CSV exported from OOTP to get started.")
            return
        df = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        rotation_size = st.select_slider(
            "Rotation Size",
            options=[3, 4, 5, 6],
            value=6,
            help="6-man is universally optimal (+3.2 WAR vs 5-man). 3-man for playoff only."
        )
    with col2:
        pos_players = st.number_input("Position Players on 25-man", value=13, min_value=10, max_value=15)
    
    bp_size = 25 - pos_players - rotation_size
    st.caption(f"Configuration: {rotation_size} SP + {bp_size} RP + {pos_players} position players = 25")
    
    if st.button("🔄 Build Staff", type="primary", key="psc_build"):
        results = build_staff(df, rotation_size, pos_players)
        st.session_state['psc_results'] = results
    
    if 'psc_results' not in st.session_state:
        return
    
    results = st.session_state['psc_results']
    
    # Tabs
    tab_staff, tab_allpitch, tab_method = st.tabs(["Staff Card", "All Pitchers", "Methodology"])
    
    # ==================== STAFF CARD ====================
    with tab_staff:
        # Rotation
        st.subheader(f"Rotation ({results['rotation_size']}-Man, Strict Order)")
        rot_rows = []
        for i, (_, r) in enumerate(results['rotation'].iterrows(), 1):
            starter_war = predict_role_war(r, 'Starter')
            rot_rows.append({
                '#': i,
                'Name': r['Name'],
                'Age': int(r['Age']),
                'STU': int(r['STU']),
                'MOV': int(r['MOV']),
                'CON': int(r['CON']),
                'STM': int(r['STM']),
                'SP F1': r['SP_F1'],
                'Starter WAR': round(starter_war, 2),
            })
        rot_df = pd.DataFrame(rot_rows)
        st.dataframe(rot_df, use_container_width=True, hide_index=True)
        
        total_sp_f1 = rot_df['SP F1'].sum()
        total_starter_war = rot_df['Starter WAR'].sum()
        c1, c2 = st.columns(2)
        c1.metric("Total SP F1", f"{total_sp_f1:.1f}")
        c2.metric("Total Starter WAR", f"{total_starter_war:.1f}")
        
        # Bullpen
        st.subheader(f"Bullpen ({results['bp_size']} Arms)")
        bp_rows = []
        for a in results['bullpen']:
            bp_rows.append({
                'Role': a['role'],
                'Name': a['name'],
                'Age': a['age'],
                'STU': a['stu'],
                'MOV': a['mov'],
                'CON': a['con'],
                'STM': a['stm'],
                'Pred WAR': a['war'],
                'SP-Capable': '✓' if a['sp_capable'] else '',
            })
        bp_df = pd.DataFrame(bp_rows)
        st.dataframe(bp_df, use_container_width=True, hide_index=True)
        
        total_bp_war = bp_df['Pred WAR'].sum()
        st.metric("Total Bullpen WAR", f"{total_bp_war:.2f}")
        
        # Key decisions
        with st.expander("Key Decisions & Edge Cases"):
            pitchers = results['all_pitchers']
            for _, r in pitchers.iterrows():
                stm = r['STM']
                if 40 <= stm <= 50:
                    starter_war = predict_role_war(r, 'Starter')
                    best_rp = r['Best_Relief_WAR']
                    st.markdown(
                        f"**{r['Name']}** — STM {int(stm)} (threshold zone). "
                        f"Starter WAR: {starter_war:.2f}, Best relief ({r['Best_Relief']}): {best_rp:.2f}"
                    )
    
    # ==================== ALL PITCHERS ====================
    with tab_allpitch:
        st.subheader("All Pitchers — Full Projection Table")
        pitchers = results['all_pitchers']
        display_cols = ['Name', 'POS', 'Age', 'STU', 'MOV', 'CON', 'STM',
                       'SP_F1', 'Starter_WAR', 'Closer_WAR', 'Setup_WAR',
                       'Middle Relief_WAR', 'Long Relief_WAR', 'Best_Relief', 'Best_Relief_WAR']
        display_cols = [c for c in display_cols if c in pitchers.columns]
        st.dataframe(
            pitchers[display_cols].sort_values('SP_F1', ascending=False),
            use_container_width=True, hide_index=True
        )
    
    # ==================== METHODOLOGY ====================
    with tab_method:
        st.subheader("How It Works")
        
        st.markdown("**Step 1 — SP vs Bullpen (STM Threshold)**")
        st.markdown(
            "STM ≥ 42 → starter candidate. STM < 42 → bullpen only. "
            "Below 42, closing generates more WAR/IP than starting."
        )
        
        st.markdown("**Step 2 — Rotation Selection**")
        st.markdown(
            "Top N starters by SP F1 = -5.932 + (MOV × 0.1091) + (STM × 0.056) + (STU × 0.0207) + (CON × 0.0053). "
            "MOV and STM dominate starter value."
        )
        
        st.markdown("**Step 3 — Bullpen Role Assignment**")
        st.markdown(
            "For each role (Closer → Setup → MR → LR → Mop-Up), assign the pitcher "
            "with the highest predicted WAR for that role. Greedy assignment, highest-value roles first."
        )
        
        st.markdown("**Role-Specific Models (WAR = intercept + STU×β + MOV×β + CON×β + STM×β)**")
        model_rows = []
        for role, m in ROLE_MODELS.items():
            model_rows.append({
                'Role': role,
                'STU': f"{m['STU']:+.4f}",
                'MOV': f"{m['MOV']:+.4f}",
                'CON': f"{m['CON']:+.4f}",
                'STM': f"{m['STM']:+.4f}",
                'R²': m['R2'],
                'N': m['n'],
            })
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
        
        st.markdown("**Key Findings — Bullpen Roles**")
        st.markdown(
            "- MOV is #1 for all relief roles (+0.05–0.06 WAR per point)\n"
            "- STU is co-dominant with MOV for Closers only\n"
            "- CON matters for Closers (3rd ranked): 55+ CON closers avg 2.80 WAR vs 1.07 for CON 35-\n"
            "- STM is irrelevant within the bullpen (near-zero coefficient)\n"
            "- HLD rating is useless — do not use for role assignment\n"
            "- High-STU Low-CON: Close if STM < 42, Start if STM ≥ 42"
        )
        
        st.markdown("---")
        st.markdown("**Rotation Size Study (4,140 team-seasons)**")
        st.markdown(
            "**6-man rotation is universally optimal.** No STM threshold or quality tier where 5-man outperforms 6-man."
        )
        st.markdown(
            "- **+3.2 WAR per additional rotation member** after controlling for quality (p < 0.0001)\n"
            "- **37-start hard cap** in OOTP — no pitcher exceeds 37 GS regardless of STM or rest. "
            "4-man rotations are structurally broken (18 games with no real starter)\n"
            "- **Extra rest does NOT help** — same pitcher produces identical WAR/GS (0.103 vs 0.102) "
            "and IP/GS (6.53 vs 6.49) in 5-man vs 6-man. Pitch counts driven by STM, not rest days\n"
            "- **Top 5 starters unaffected** — identical GS totals in both formats. "
            "6th man absorbs spot-starter innings, not ace innings\n"
            "- **Bullpen NOT harmed** — RP WAR/ERA virtually unchanged between formats\n"
            "- **OOTP Settings:** 6-Man Rotation, Strict Order, Allow SP in Relief: No"
        )


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/philadelphia_quakers_organization_-_roster_claude_request_pitch.csv'
    
    df = pd.read_csv(csv_path, low_memory=False)
    for size in [6, 5, 3]:
        results = build_staff(df, rotation_size=size)
        print(f"\n{'=' * 60}")
        print(f"{size}-MAN ROTATION + {results['bp_size']} BULLPEN")
        print(f"{'=' * 60}")
        print("\nRotation:")
        for i, (_, r) in enumerate(results['rotation'].iterrows(), 1):
            print(f"  {i}. {r['Name']:<20} SP_F1={r['SP_F1']:.2f}")
        print("\nBullpen:")
        for a in results['bullpen']:
            print(f"  {a['role']:<16} {a['name']:<20} WAR={a['war']:+.2f}")
