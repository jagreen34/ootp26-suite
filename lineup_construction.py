"""
OOTP 26 Lineup Construction Tool
==================================
Given a roster, finds the optimal 8-man starting lineup:
  - WHO starts (position assignment maximizing total WAR)
  - WHERE they play (position-specific defensive value)
  - WHAT ORDER they bat (lineup slot models)
  - DEPTH CHART for backups

All models validated on 253K player-seasons.
"""

import numpy as np
import pandas as pd
from itertools import permutations
from scipy.optimize import linear_sum_assignment

# ============================================================================
# MODEL COEFFICIENTS
# ============================================================================

FIELD_POSITIONS = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
OFF_FEATURES = ['CON', 'POW', 'EYE', 'GAP', 'BABIP', 'SPE', 'STE']

# --- Position-specific ZR models (ratings → predicted Zone Rating) ---
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

# --- ZR → WAR conversion per position ---
ZR_TO_WAR = {
    'C': 0.1333, '1B': 0.1182, '2B': 0.1227, '3B': 0.1248,
    'SS': 0.1040, 'LF': 0.1626, 'CF': 0.1111, 'RF': 0.1215
}

# --- Position eligibility floors ---
POS_FLOORS = {
    'C': {'C ABI': 45},
    'SS': {'IF RNG': 50},
    'CF': {'OF RNG': 50},
}

# --- Lineup slot models (WAR/600PA ~ offensive ratings) ---
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

# --- Positional trade multipliers (for reference/display, not used in optimization) ---
POS_MULTIPLIERS = {
    'CF': 1.55, 'SS': 1.50, 'C': 1.30, '2B': 1.30,
    'RF': 1.25, '3B': 1.20, 'LF': 1.05, '1B': 1.00
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def predict_zr(player_row, position):
    """Predict Zone Rating for a player at a specific field position."""
    model = ZR_MODELS[position]
    zr = model['intercept']
    for feat, coef in model['coefs'].items():
        val = pd.to_numeric(player_row.get(feat, 0), errors='coerce')
        if pd.isna(val):
            val = 0
        zr += val * coef
    return zr


def predict_def_war(player_row, position):
    """Predict defensive WAR for a player at a specific field position."""
    zr = predict_zr(player_row, position)
    return zr * ZR_TO_WAR[position]


def is_eligible(player_row, position, min_pos_rating=40):
    """Check if player is eligible at a position (has a rating ≥ minimum)."""
    val = player_row.get(position, '-')
    if val == '-' or pd.isna(val):
        return False
    try:
        rating = float(val)
        if rating < min_pos_rating:
            return False
    except (ValueError, TypeError):
        return False
    return True


def meets_floor(player_row, position):
    """Check if player meets the minimum rating floor for premium positions."""
    if position not in POS_FLOORS:
        return True
    for rating_name, min_val in POS_FLOORS[position].items():
        val = pd.to_numeric(player_row.get(rating_name, 0), errors='coerce')
        if pd.isna(val) or val < min_val:
            return False
    return True


def predict_offensive_war_avg(player_row):
    """Predict offensive WAR averaged across all 8 lineup slots (season-level)."""
    total = 0
    for lp in range(1, 9):
        model = LINEUP_MODELS[lp]
        rate = model['intercept']
        for feat in OFF_FEATURES:
            val = pd.to_numeric(player_row.get(feat, 50), errors='coerce')
            if pd.isna(val):
                val = 50
            rate += model['coefs'][feat] * val
        season_war = rate * (PA_WEIGHTS[lp] / 600)
        total += season_war
    return total / 8


def predict_lineup_war(player_row, lineup_pos):
    """Predict offensive season WAR at a specific batting order position."""
    model = LINEUP_MODELS[lineup_pos]
    rate = model['intercept']
    for feat in OFF_FEATURES:
        val = pd.to_numeric(player_row.get(feat, 50), errors='coerce')
        if pd.isna(val):
            val = 50
        rate += model['coefs'][feat] * val
    return rate * (PA_WEIGHTS[lineup_pos] / 600)


def build_value_matrix(roster_df):
    """
    Build Player × Position value matrix.
    Each cell = Offensive WAR (avg across lineup slots) + Defensive WAR at that position.
    Ineligible or below-floor = -999.
    """
    players = []
    matrix = []
    
    for idx, row in roster_df.iterrows():
        off_war = predict_offensive_war_avg(row)
        pos_values = []
        for pos in FIELD_POSITIONS:
            if is_eligible(row, pos) and meets_floor(row, pos):
                def_war = predict_def_war(row, pos)
                total = off_war + def_war
                pos_values.append(total)
            else:
                pos_values.append(-999)
        
        players.append({
            'idx': idx,
            'name': row.get('Name', f'Player {idx}'),
            'age': row.get('Age', ''),
            'primary_pos': row.get('POS', ''),
            'off_war': off_war,
            'row': row,
        })
        matrix.append(pos_values)
    
    return players, np.array(matrix)


def solve_assignment(players, value_matrix):
    """
    Solve the optimal player→position assignment using the Hungarian algorithm.
    Maximizes total WAR across 8 field positions.
    """
    n_players = len(players)
    n_positions = len(FIELD_POSITIONS)
    
    # Hungarian algorithm minimizes cost, so negate for maximization.
    # Build cost matrix: rows = positions (8), cols = players (n).
    # We need to assign exactly 8 positions to 8 of n players.
    
    # Create cost matrix (positions × players)
    cost = np.full((n_positions, n_players), 9999.0)
    for p_idx in range(n_players):
        for pos_idx in range(n_positions):
            val = value_matrix[p_idx][pos_idx]
            if val > -900:  # eligible
                cost[pos_idx][p_idx] = -val  # negate for minimization
    
    # Solve
    pos_indices, player_indices = linear_sum_assignment(cost)
    
    # Build results
    assignments = []
    total_war = 0
    for pos_idx, p_idx in zip(pos_indices, player_indices):
        pos = FIELD_POSITIONS[pos_idx]
        player = players[p_idx]
        val = value_matrix[p_idx][pos_idx]
        def_war = predict_def_war(player['row'], pos)
        
        assignments.append({
            'position': pos,
            'player_idx': p_idx,
            'name': player['name'],
            'age': player['age'],
            'off_war': player['off_war'],
            'def_war': def_war,
            'total_war': val,
            'row': player['row'],
        })
        total_war += val
    
    # Sort by field position order
    pos_order = {p: i for i, p in enumerate(FIELD_POSITIONS)}
    assignments.sort(key=lambda x: pos_order[x['position']])
    
    return assignments, total_war


def optimize_batting_order(assignments):
    """
    Given 8 assigned starters, find the optimal batting order.
    Brute-force 8! = 40,320 permutations.
    """
    n = len(assignments)
    best_war = -999
    best_order = None
    
    for perm in permutations(range(n)):
        total = 0
        for lp_idx, a_idx in enumerate(perm):
            lp = lp_idx + 1
            total += predict_lineup_war(assignments[a_idx]['row'], lp)
        if total > best_war:
            best_war = total
            best_order = perm
    
    return best_order, best_war


def build_depth_chart(players, value_matrix, starters_indices):
    """
    For each position, rank all eligible players by their value at that position.
    """
    depth = {}
    for pos_idx, pos in enumerate(FIELD_POSITIONS):
        eligible = []
        for p_idx, player in enumerate(players):
            val = value_matrix[p_idx][pos_idx]
            if val > -900:
                eligible.append({
                    'name': player['name'],
                    'age': player['age'],
                    'value': val,
                    'def_war': predict_def_war(player['row'], pos),
                    'off_war': player['off_war'],
                    'is_starter': p_idx in starters_indices,
                })
        eligible.sort(key=lambda x: x['value'], reverse=True)
        depth[pos] = eligible
    return depth


def run_full_optimization(roster_df):
    """
    Full pipeline: value matrix → assignment → batting order → depth chart.
    """
    # Filter to position players
    pos_players = roster_df[~roster_df['POS'].isin(['SP', 'RP', 'CL'])].copy()
    
    # Step 1: Value matrix
    players, matrix = build_value_matrix(pos_players)
    
    # Step 2: Position assignment
    assignments, assignment_war = solve_assignment(players, matrix)
    
    # Step 3: Batting order
    batting_order, batting_war = optimize_batting_order(assignments)
    
    # Step 4: Depth chart
    starter_indices = set(a['player_idx'] for a in assignments)
    depth = build_depth_chart(players, matrix, starter_indices)
    
    return {
        'assignments': assignments,
        'assignment_war': assignment_war,
        'batting_order': batting_order,
        'batting_war': batting_war,
        'depth_chart': depth,
        'value_matrix': matrix,
        'players': players,
    }


# ============================================================================
# DISPLAY
# ============================================================================

def print_results(results):
    """Print full optimization results."""
    assignments = results['assignments']
    batting_order = results['batting_order']
    
    print("=" * 85)
    print("OPTIMAL STARTING LINEUP — PHILADELPHIA QUAKERS")
    print("=" * 85)
    
    # Build ordered lineup
    ordered = []
    for lp_idx, a_idx in enumerate(batting_order):
        a = assignments[a_idx]
        lp = lp_idx + 1
        lineup_war = predict_lineup_war(a['row'], lp)
        ordered.append({
            'lp': lp,
            'pos': a['position'],
            'name': a['name'],
            'age': a['age'],
            'off_war': lineup_war,
            'def_war': a['def_war'],
            'total_war': lineup_war + a['def_war'],
        })
    
    print(f"\n{'LP':<4} {'Pos':<5} {'Player':<22} {'Age':>4} {'Off WAR':>8} {'Def WAR':>8} {'Total':>8}")
    print("-" * 65)
    
    total_off = 0
    total_def = 0
    for o in ordered:
        print(f"{o['lp']:<4} {o['pos']:<5} {o['name']:<22} {o['age']:>4} {o['off_war']:>8.2f} {o['def_war']:>8.2f} {o['total_war']:>8.2f}")
        total_off += o['off_war']
        total_def += o['def_war']
    
    print("-" * 65)
    print(f"{'':4} {'':5} {'TOTAL':<22} {'':>4} {total_off:>8.2f} {total_def:>8.2f} {total_off + total_def:>8.2f}")
    
    # Key offensive ratings for each starter
    print(f"\n{'LP':<4} {'Pos':<5} {'Player':<22} {'CON':>4} {'POW':>4} {'EYE':>4} {'GAP':>4} {'BABIP':>5} {'SPE':>4} {'STE':>4}")
    print("-" * 65)
    for lp_idx, a_idx in enumerate(batting_order):
        a = assignments[a_idx]
        r = a['row']
        lp = lp_idx + 1
        print(f"{lp:<4} {a['position']:<5} {a['name']:<22} {r['CON']:>4} {r['POW']:>4} {r['EYE']:>4} {r['GAP']:>4} {r['BABIP']:>5} {r['SPE']:>4} {r['STE']:>4}")
    
    # Key defensive ratings for each starter
    print(f"\n{'Pos':<5} {'Player':<22} {'IF RNG':>6} {'IF ARM':>6} {'OF RNG':>6} {'OF ARM':>6} {'C ABI':>5} {'Pos Rtg':>7} {'ZR→WAR':>7}")
    print("-" * 72)
    for a in assignments:
        r = a['row']
        pos_rtg = r.get(a['position'], '-')
        zr = predict_zr(r, a['position'])
        print(f"{a['position']:<5} {a['name']:<22} {r['IF RNG']:>6} {r['IF ARM']:>6} {r['OF RNG']:>6} {r['OF ARM']:>6} {r['C ABI']:>5} {pos_rtg:>7} {a['def_war']:>+7.2f}")
    
    # Depth chart
    print(f"\n{'=' * 85}")
    print("DEPTH CHART")
    print("=" * 85)
    
    depth = results['depth_chart']
    for pos in FIELD_POSITIONS:
        players_at_pos = depth[pos]
        line = f"{pos:<4}"
        for i, p in enumerate(players_at_pos[:4]):
            marker = "★" if p['is_starter'] else " "
            line += f"  {i+1}.{marker}{p['name']:<18} ({p['value']:+.1f})"
        print(line)
    
    # Value matrix summary
    print(f"\n{'=' * 85}")
    print("VALUE MATRIX (top 5 options per position)")
    print("=" * 85)
    
    players = results['players']
    matrix = results['value_matrix']
    
    for pos_idx, pos in enumerate(FIELD_POSITIONS):
        vals = [(players[i]['name'], matrix[i][pos_idx]) for i in range(len(players)) if matrix[i][pos_idx] > -900]
        vals.sort(key=lambda x: x[1], reverse=True)
        top5 = vals[:5]
        line = f"{pos:<4} ({POS_MULTIPLIERS[pos]:.2f}x): "
        line += " | ".join(f"{n} {v:+.1f}" for n, v in top5)
        print(line)


# ============================================================================
# STREAMLIT UI
# ============================================================================

def render_lineup_construction(uploaded_df=None):
    """
    Render the full lineup construction tool as a Streamlit section.
    Call from your main app.py like:
        import lineup_construction as lc
        lc.render_lineup_construction()
    """
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not available. Use print_results() for CLI output.")
        return

    st.header("⚾ Lineup Construction")
    st.caption(
        "Position assignment + batting order optimization. "
        "Models: ZR defense (per position), lineup slot WAR (253K study), Hungarian algorithm assignment."
    )

    # File upload or passed DataFrame
    if uploaded_df is not None:
        df = uploaded_df
    else:
        f = st.file_uploader("Upload Team Batter CSV (OOTP export)", type=['csv'], key='lc_upload')
        if not f:
            st.info("Upload a batter CSV exported from OOTP to get started.")
            return
        df = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)

    # Filter pitchers
    pos_players = df[~df['POS'].isin(['SP', 'RP', 'CL'])].copy()
    st.success(f"Loaded {len(pos_players)} position players from {len(df)} total.")

    # Run optimization
    if st.button("🔄 Optimize Lineup", type="primary", key="lc_optimize"):
        with st.spinner("Building value matrix and solving assignment..."):
            results = run_full_optimization(df)

        st.session_state['lc_results'] = results

    if 'lc_results' not in st.session_state:
        return

    results = st.session_state['lc_results']
    assignments = results['assignments']
    batting_order = results['batting_order']

    # --- Tab layout ---
    tab_lineup, tab_depth, tab_matrix, tab_method = st.tabs([
        "Starting Lineup", "Depth Chart", "Value Matrix", "Methodology"
    ])

    # ==================== STARTING LINEUP ====================
    with tab_lineup:
        st.subheader("Optimal Starting Lineup")

        # Build ordered lineup rows
        rows = []
        for lp_idx, a_idx in enumerate(batting_order):
            a = assignments[a_idx]
            lp = lp_idx + 1
            off = predict_lineup_war(a['row'], lp)
            rows.append({
                'LP': lp,
                'Pos': a['position'],
                'Player': a['name'],
                'Age': a['age'],
                'Off WAR': round(off, 2),
                'Def WAR': round(a['def_war'], 2),
                'Total': round(off + a['def_war'], 2),
            })

        lineup_df = pd.DataFrame(rows)
        st.dataframe(lineup_df, use_container_width=True, hide_index=True)

        # Summary metrics
        total_off = lineup_df['Off WAR'].sum()
        total_def = lineup_df['Def WAR'].sum()
        total_all = lineup_df['Total'].sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Offensive WAR", f"{total_off:.1f}")
        c2.metric("Total Defensive WAR", f"{total_def:.1f}")
        c3.metric("Combined WAR", f"{total_all:.1f}")

        # Offensive ratings table
        with st.expander("Offensive Ratings"):
            off_rows = []
            for lp_idx, a_idx in enumerate(batting_order):
                a = assignments[a_idx]
                r = a['row']
                off_rows.append({
                    'LP': lp_idx + 1,
                    'Pos': a['position'],
                    'Player': a['name'],
                    'CON': r['CON'], 'POW': r['POW'], 'EYE': r['EYE'],
                    'GAP': r['GAP'], 'BABIP': r['BABIP'],
                    'SPE': r['SPE'], 'STE': r['STE'],
                })
            st.dataframe(pd.DataFrame(off_rows), use_container_width=True, hide_index=True)

        # Defensive ratings table
        with st.expander("Defensive Ratings"):
            def_rows = []
            for a in assignments:
                r = a['row']
                pos_rtg = r.get(a['position'], '-')
                def_rows.append({
                    'Pos': a['position'],
                    'Player': a['name'],
                    'Pos Rating': pos_rtg,
                    'IF RNG': r['IF RNG'], 'IF ARM': r['IF ARM'],
                    'OF RNG': r['OF RNG'], 'OF ARM': r['OF ARM'],
                    'C ABI': r['C ABI'],
                    'Def WAR': round(a['def_war'], 2),
                })
            st.dataframe(pd.DataFrame(def_rows), use_container_width=True, hide_index=True)

    # ==================== DEPTH CHART ====================
    with tab_depth:
        st.subheader("Depth Chart")
        depth = results['depth_chart']

        for pos in FIELD_POSITIONS:
            players_at = depth[pos]
            if not players_at:
                continue
            st.markdown(f"**{pos}** ({POS_MULTIPLIERS[pos]:.2f}x)")
            dep_rows = []
            for i, p in enumerate(players_at[:5]):
                dep_rows.append({
                    'Rank': i + 1,
                    'Player': ('★ ' if p['is_starter'] else '') + p['name'],
                    'Age': p['age'],
                    'Off WAR': round(p['off_war'], 2),
                    'Def WAR': round(p['def_war'], 2),
                    'Total': round(p['value'], 2),
                })
            st.dataframe(pd.DataFrame(dep_rows), use_container_width=True, hide_index=True)

    # ==================== VALUE MATRIX ====================
    with tab_matrix:
        st.subheader("Player × Position Value Matrix")
        st.caption("Total projected WAR (offense + defense) for each player at each eligible position. Blank = ineligible.")

        players = results['players']
        matrix = results['value_matrix']

        matrix_rows = []
        for p_idx, player in enumerate(players):
            row = {'Player': player['name'], 'Age': player['age'], 'Primary': player['primary_pos']}
            for pos_idx, pos in enumerate(FIELD_POSITIONS):
                val = matrix[p_idx][pos_idx]
                row[pos] = round(val, 1) if val > -900 else None
            matrix_rows.append(row)

        matrix_df = pd.DataFrame(matrix_rows)
        # Sort by max value across positions
        matrix_df['Best'] = matrix_df[FIELD_POSITIONS].max(axis=1)
        matrix_df = matrix_df.sort_values('Best', ascending=False).drop(columns='Best')

        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    # ==================== METHODOLOGY ====================
    with tab_method:
        st.subheader("How It Works")

        st.markdown("**Step 1 — Value Matrix**")
        st.markdown(
            "For each player × each position they're eligible for (OOTP position rating ≥ 40), "
            "calculate: Offensive WAR (average across 8 lineup slots) + Defensive WAR "
            "(predicted ZR from position-specific models × ZR→WAR conversion)."
        )

        st.markdown("**Step 2 — Position Assignment (Hungarian Algorithm)**")
        st.markdown(
            "Find the combination of 8 players assigned to 8 field positions that maximizes total team WAR. "
            "This is a constrained optimization — a player can only be assigned to positions they're eligible for."
        )

        st.markdown("**Step 3 — Batting Order (Brute Force)**")
        st.markdown(
            "Once the 8 starters are selected, optimize their batting order by evaluating all 40,320 permutations "
            "through the lineup-slot-specific WAR models."
        )

        st.markdown("**Step 4 — Depth Chart**")
        st.markdown(
            "Rank all remaining roster players by their value at each position."
        )

        st.markdown("---")
        st.markdown("**Position-Specific ZR Models**")
        zr_rows = []
        for pos in FIELD_POSITIONS:
            m = ZR_MODELS[pos]
            formula = f"ZR = {m['intercept']:.2f}"
            for feat, coef in m['coefs'].items():
                formula += f" + ({feat} × {coef:.4f})"
            zr_rows.append({'Position': pos, 'Formula': formula, 'ZR→WAR': ZR_TO_WAR[pos]})
        st.dataframe(pd.DataFrame(zr_rows), use_container_width=True, hide_index=True)

        st.markdown("**Key Findings (253K Study)**")
        st.markdown(
            "- Offensive F1 averages ~3.0 for starters — defense separates players, not offense\n"
            "- POW dominates every lineup slot (2–3× any other rating)\n"
            "- Lineup arrangement adds ~2% explanatory power beyond total talent\n"
            "- vR/vL splits add zero predictive value (r = 0.993–0.997 with overall)"
        )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/philadelphia_quakers_organization_-_roster_claude_request_bat.csv'

    roster = pd.read_csv(csv_path, low_memory=False)
    results = run_full_optimization(roster)
    print_results(results)
