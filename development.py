"""
OOTP 26 Development Slider Optimizer
=====================================
Calculates optimal development focus for each player based on:
- Gap between current and potential ratings
- F1 coefficient weights (which ratings matter most)
- Zero-sum constraint (pushing one slider up requires others down)

Output: recommended slider positions per player (0-100 scale).
"""

import pandas as pd
import numpy as np

# ============================================================================
# F1 WEIGHTS — how much each rating matters
# ============================================================================

BAT_WEIGHTS = {
    'POW':   0.1142,   # Power — king
    'BABIP': 0.0725,   # BABIP
    'EYE':   0.0400,   # Eye / discipline
    'CON':   0.0379,   # Contact
    'GAP':   0.0291,   # Gap power
    'SPE':   0.0128,   # Speed / Running
}

# Map dev slider names to rating columns (current / potential)
BAT_SLIDER_MAP = {
    'Power':    {'current': 'POW',   'potential': 'POW P'},
    'BABIP':    {'current': 'BABIP', 'potential': 'HT P'},      # HT P is BABIP potential in OOTP
    'Eye':      {'current': 'EYE',   'potential': 'EYE P'},
    'Avoid K\'s': {'current': "K's",  'potential': 'K P'},
    'Gap':      {'current': 'GAP',   'potential': 'GAP P'},
    'Running':  {'current': 'SPE',   'potential': 'SPE'},       # SPE doesn't have a separate potential in our file
    'Defense':  {'current': None,    'potential': None},         # Handled separately
}

BAT_WEIGHT_MAP = {
    'Power':    0.1142,
    'BABIP':    0.0725,
    'Eye':      0.0400,
    'Avoid K\'s': 0.0317,
    'Gap':      0.0291,
    'Running':  0.0128,
    'Defense':  0.0300,  # Approximate — defense matters but position-dependent
}

PITCH_WEIGHTS = {
    'MOV': 0.1091,   # Movement — king
    'STM': 0.0560,   # Stamina
    'STU': 0.0207,   # Stuff
    'CON': 0.0053,   # Control (PIT_CON)
}

PITCH_SLIDER_MAP = {
    'Movement': {'current': 'MOV', 'potential': 'MOV P'},
    'Control':  {'current': 'PIT_CON', 'potential': 'PIT_CON_P'},
    'Stamina':  {'current': 'STM', 'potential': 'STM'},   # No separate STM potential in file
}

PITCH_WEIGHT_MAP = {
    'Movement': 0.1091,
    'Control':  0.0053,
    'Stamina':  0.0560,
}

# Pitch type slider map
PITCH_TYPE_MAP = {
    'FB': 'FBP', 'CB': 'CBP', 'CH': 'CHP', 'SL': 'SLP',
    'SI': 'SIP', 'CT': 'CTP', 'FO': 'FOP', 'CC': 'CCP',
    'SC': 'SCP', 'KC': 'KCP', 'KN': 'KNP', 'SP': 'SPP',
}


def _safe(v, d=0):
    try:
        if pd.isna(v): return d
        f = float(v)
        return f if f > 0 else d
    except: return d


def _is_premium_position(pos):
    return pos in ('SS', 'CF', 'C', '2B')


def calc_batter_sliders(row):
    """Calculate optimal development slider allocation for a batter."""
    priorities = {}

    for slider_name, mapping in BAT_SLIDER_MAP.items():
        if slider_name == 'Defense':
            # Defense priority: higher for premium positions, based on position potential
            pos = str(row.get('POS', ''))
            if _is_premium_position(pos):
                priorities['Defense'] = BAT_WEIGHT_MAP['Defense'] * 20  # Fixed priority boost
            else:
                priorities['Defense'] = BAT_WEIGHT_MAP['Defense'] * 5
            continue

        if slider_name == 'Running':
            # Running: SPE doesn't have a separate potential, use gap from 80 as proxy
            current = _safe(row.get('SPE', 50))
            gap = max(0, 80 - current)  # Assume 80 is theoretical max useful
            priorities[slider_name] = BAT_WEIGHT_MAP[slider_name] * gap
            continue

        current_col = mapping['current']
        potential_col = mapping['potential']

        current = _safe(row.get(current_col, 50))
        potential = _safe(row.get(potential_col, 50))

        gap = max(0, potential - current)
        weight = BAT_WEIGHT_MAP[slider_name]
        priorities[slider_name] = weight * gap

    # Normalize to fixed budget (zero-sum constraint)
    # 7 sliders x 50 midpoint = 350 total budget
    BUDGET = 350
    MIN_SLIDER = 10
    MAX_SLIDER = 90

    total_priority = sum(priorities.values())
    if total_priority == 0:
        return {k: 50 for k in priorities}

    # Distribute budget proportionally
    sliders = {}
    for name, prio in priorities.items():
        sliders[name] = (prio / total_priority) * BUDGET

    # Clamp and redistribute
    for _ in range(10):
        excess = 0
        n_free = 0
        for name, val in sliders.items():
            if val > MAX_SLIDER:
                excess += val - MAX_SLIDER
                sliders[name] = MAX_SLIDER
            elif val < MIN_SLIDER:
                excess -= MIN_SLIDER - val
                sliders[name] = MIN_SLIDER
            else:
                n_free += 1
        if abs(excess) < 0.5 or n_free == 0:
            break
        per_free = excess / n_free
        for name, val in sliders.items():
            if MIN_SLIDER < val < MAX_SLIDER:
                sliders[name] = max(MIN_SLIDER, min(MAX_SLIDER, val + per_free))

    sliders = {k: int(round(v)) for k, v in sliders.items()}

    # Final nudge to hit exact budget
    diff = BUDGET - sum(sliders.values())
    if diff != 0:
        top = max(priorities, key=priorities.get)
        sliders[top] = max(MIN_SLIDER, min(MAX_SLIDER, sliders[top] + diff))

    return sliders


def calc_pitcher_sliders(row):
    """Calculate optimal development slider allocation for a pitcher."""
    # Main sliders: Movement, Control, Stamina
    priorities = {}

    for slider_name, mapping in PITCH_SLIDER_MAP.items():
        current_col = mapping['current']
        potential_col = mapping['potential']

        current = _safe(row.get(current_col, 50))

        if potential_col and potential_col in row.index:
            potential = _safe(row.get(potential_col, 50))
        else:
            potential = current  # No potential column = assume at ceiling

        gap = max(0, potential - current)
        weight = PITCH_WEIGHT_MAP[slider_name]
        priorities[slider_name] = weight * gap

    # Zero-sum budget: 3 sliders x 50 = 150
    BUDGET = 150
    MIN_SLIDER = 10
    MAX_SLIDER = 90

    total = sum(priorities.values())
    if total == 0:
        return {'main': {k: 50 for k in priorities}, 'pitches': {}}

    sliders = {}
    for name, prio in priorities.items():
        sliders[name] = (prio / total) * BUDGET

    for _ in range(10):
        excess = 0
        n_free = 0
        for name, val in sliders.items():
            if val > MAX_SLIDER:
                excess += val - MAX_SLIDER
                sliders[name] = MAX_SLIDER
            elif val < MIN_SLIDER:
                excess -= MIN_SLIDER - val
                sliders[name] = MIN_SLIDER
            else:
                n_free += 1
        if abs(excess) < 0.5 or n_free == 0:
            break
        per_free = excess / n_free
        for name, val in sliders.items():
            if MIN_SLIDER < val < MAX_SLIDER:
                sliders[name] = max(MIN_SLIDER, min(MAX_SLIDER, val + per_free))

    sliders = {k: int(round(v)) for k, v in sliders.items()}
    diff = BUDGET - sum(sliders.values())
    if diff != 0:
        top = max(priorities, key=priorities.get)
        sliders[top] = max(MIN_SLIDER, min(MAX_SLIDER, sliders[top] + diff))

    # Pitch type sliders — focus on best 2-3 pitches
    pitch_sliders = {}
    for pitch, pot_col in PITCH_TYPE_MAP.items():
        current = _safe(row.get(pitch, 0))
        potential = _safe(row.get(pot_col, 0))
        if current > 0 or potential > 20:  # Has this pitch
            gap = max(0, potential - current)
            pitch_sliders[pitch] = {'current': int(current), 'potential': int(potential),
                                    'gap': int(gap), 'priority': int(gap * potential / 80)}

    # Sort pitches by priority
    pitch_sliders = dict(sorted(pitch_sliders.items(), key=lambda x: x[1]['priority'], reverse=True))

    return {'main': sliders, 'pitches': pitch_sliders}


def generate_dev_plan(df):
    """Generate development plans for entire roster."""
    plans = []

    for idx, row in df.iterrows():
        pos = str(row.get('POS', ''))
        name = row.get('Name', '')
        age = int(_safe(row.get('Age', 25)))
        team = row.get('TM', '')

        if age > 28:
            # Veterans past development age — skip
            plans.append({
                'Name': name, 'POS': pos, 'Age': age, 'TM': team,
                'Plan': 'VETERAN — development minimal, use defaults',
                'Sliders': {}
            })
            continue

        if pos in ('SP', 'RP', 'CL'):
            result = calc_pitcher_sliders(row)
            main = result['main']
            pitches = result['pitches']

            # Format pitch recommendations
            pitch_recs = []
            for p, info in list(pitches.items())[:3]:  # Top 3 pitches
                pitch_recs.append(f"{p}: {info['current']}→{info['potential']} (gap {info['gap']})")

            top_focus = max(main, key=main.get) if main else 'Movement'

            plans.append({
                'Name': name, 'POS': pos, 'Age': age, 'TM': team,
                'Focus': top_focus,
                'Movement': main.get('Movement', 50),
                'Control': main.get('Control', 50),
                'Stamina': main.get('Stamina', 50),
                'Top Pitches': ' | '.join(pitch_recs),
                'Plan': f"Focus: {top_focus}"
            })
        else:
            sliders = calc_batter_sliders(row)

            top_focus = max(sliders, key=sliders.get) if sliders else 'Power'

            plans.append({
                'Name': name, 'POS': pos, 'Age': age, 'TM': team,
                'Focus': top_focus,
                'Power': sliders.get('Power', 50),
                'BABIP': sliders.get('BABIP', 50),
                'Eye': sliders.get('Eye', 50),
                "Avoid K's": sliders.get("Avoid K's", 50),
                'Gap': sliders.get('Gap', 50),
                'Running': sliders.get('Running', 50),
                'Defense': sliders.get('Defense', 50),
                'Plan': f"Focus: {top_focus}"
            })

    return pd.DataFrame(plans)


def render_development(uploaded_df=None):
    import streamlit as st

    st.header("🌱 Development Slider Optimizer")
    st.caption("Optimal training focus for each player based on F1 weights × rating gaps. Zero-sum aware.")

    if uploaded_df is None:
        st.info("Load a roster in the sidebar.")
        return

    df = uploaded_df
    team = st.selectbox("Team", sorted(df[df['TM'] != '-']['TM'].unique()), key='dev_team')
    team_df = df[df['TM'] == team]

    max_age = st.slider("Max age to show (veterans skip development)", 22, 35, 28, key='dev_age')

    if st.button("🔄 Generate Development Plans", type="primary", key='dev_gen'):
        plans = generate_dev_plan(team_df)
        st.session_state['dev_plans'] = plans

    if 'dev_plans' not in st.session_state:
        return

    plans = st.session_state['dev_plans']

    t1, t2, t3 = st.tabs(["Batter Development", "Pitcher Development", "Methodology"])

    with t1:
        bat_plans = plans[(~plans['POS'].isin(['SP', 'RP', 'CL'])) & (plans['Age'] <= max_age)]
        if bat_plans.empty:
            st.info("No batter prospects under the age threshold.")
        else:
            display_cols = ['Name', 'POS', 'Age', 'Focus', 'Power', 'BABIP', 'Eye', "Avoid K's", 'Gap', 'Running', 'Defense']
            display_cols = [c for c in display_cols if c in bat_plans.columns]
            st.subheader(f"Batter Development Plans ({len(bat_plans)} players)")
            st.dataframe(bat_plans[display_cols].sort_values('Age'), use_container_width=True, hide_index=True)

            st.markdown("**How to read:** Higher numbers = push that slider higher. The game enforces zero-sum, "
                       "so pushing Power to 80 means other sliders come down proportionally.")

    with t2:
        pitch_plans = plans[(plans['POS'].isin(['SP', 'RP', 'CL'])) & (plans['Age'] <= max_age)]
        if pitch_plans.empty:
            st.info("No pitcher prospects under the age threshold.")
        else:
            display_cols = ['Name', 'POS', 'Age', 'Focus', 'Movement', 'Control', 'Stamina', 'Top Pitches']
            display_cols = [c for c in display_cols if c in pitch_plans.columns]
            st.subheader(f"Pitcher Development Plans ({len(pitch_plans)} players)")
            st.dataframe(pitch_plans[display_cols].sort_values('Age'), use_container_width=True, hide_index=True)

    with t3:
        st.subheader("How It Works")
        st.markdown(
            "**For each rating:** Priority Score = F1 Weight × (Potential - Current)\n\n"
            "**The rating with the biggest weighted gap gets the most development focus.**\n\n"
            "For batters, the F1 weights are:\n"
            "- Power: 0.1142 (highest — always develop first)\n"
            "- BABIP: 0.0725\n"
            "- Eye: 0.0400\n"
            "- Avoid K's: 0.0317\n"
            "- Gap: 0.0291\n"
            "- Running: 0.0128 (lowest — only develop for speed specialists)\n"
            "- Defense: bonus for premium positions (SS, CF, C, 2B)\n\n"
            "For pitchers:\n"
            "- Movement: 0.1091 (highest — always develop first)\n"
            "- Stamina: 0.0560\n"
            "- Control: 0.0053 (lowest — only matters as a floor)\n\n"
            "**Pitch type sliders:** Focus on the 2-3 pitches with the biggest gap between current and potential. "
            "Don't waste development on a pitch with 25 potential — it won't develop.\n\n"
            "**Veterans (age 28+):** Development gains are minimal at this age. Use defaults.\n\n"
            "**Check 'Prevent AI from Adjusting Focus'** for every player you set manually."
        )
