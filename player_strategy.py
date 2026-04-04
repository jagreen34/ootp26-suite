"""
OOTP 26 Player Strategy Optimizer
===================================
Generates per-player strategy recommendations based on ratings.
Uses "Adjust to Team Strategy" mode — slider values are RELATIVE to team settings.
Positive = more aggressive than team, Negative = less aggressive than team, 0 = use team default.

Based on 253K study findings:
- STE 65+ = steal green light
- STE < 35 = steal lockdown
- SB only profitable at STE 65+
- POW-first hitters should never bunt
- High-SPE bench players are PH/PR weapons
- MOV-first pitchers should pitch to contact (slow hook)
- Closer needs Force Role + quick hook
"""

import pandas as pd
import numpy as np
import streamlit as st


def _safe(v, d=0):
    try:
        if pd.isna(v): return d
        return float(v)
    except:
        return d


def classify_steal(ste):
    """Classify steal aggressiveness based on STE rating."""
    if ste >= 65:
        return 'GREEN LIGHT', +3
    elif ste >= 50:
        return 'SELECTIVE', +1
    elif ste >= 35:
        return 'NEUTRAL', 0
    else:
        return 'RED LIGHT', -3


def get_batter_strategy(row):
    """Generate strategy recommendations for a batter."""
    ste = _safe(row.get('STE', 50))
    spe = _safe(row.get('SPE', 50))
    pow_r = _safe(row.get('POW', 50))
    eye = _safe(row.get('EYE', 50))
    pos = str(row.get('POS', ''))
    age = _safe(row.get('Age', 25))
    name = str(row.get('Name', ''))

    steal_label, steal_adj = classify_steal(ste)

    # Base-running: correlates with SPE and STE
    if spe >= 70 and ste >= 60:
        baserun_adj = +2
        baserun_label = 'Aggressive'
    elif spe >= 55:
        baserun_adj = +1
        baserun_label = 'Moderate'
    elif spe <= 30:
        baserun_adj = -2
        baserun_label = 'Conservative'
    else:
        baserun_adj = 0
        baserun_label = 'Team Default'

    # Hit & Run: useful for high-contact guys to avoid DP
    if spe <= 35 and pow_r >= 45:
        hr_adj = +1  # Slow power guys benefit from H&R to avoid DP
        hr_label = 'Slight increase (DP avoidance)'
    else:
        hr_adj = 0
        hr_label = 'Team Default'

    # Sac Bunt: NEVER for power hitters, team already at 0
    sac_adj = 0
    sac_label = 'Team Default (0 everywhere)'

    # Bunt for Hit: only for extreme speed
    if spe >= 75 and ste >= 60:
        bunt_hit_adj = +2
        bunt_hit_label = 'Occasional (speed weapon)'
    elif spe >= 65:
        bunt_hit_adj = +1
        bunt_hit_label = 'Rare'
    else:
        bunt_hit_adj = 0
        bunt_hit_label = 'Team Default'

    # Pinch Hit settings
    ph_never_lift = False
    ph_save_high_leverage = False
    ph_frequency = 'Team Default'
    ph_notes = ''

    # Starters should not be lifted for PH
    # Bench players should be available
    if pow_r >= 45 and age <= 32:
        ph_never_lift = True
        ph_notes = 'Core starter — never lift'
    elif spe >= 70 and pow_r < 40:
        ph_save_high_leverage = True
        ph_frequency = 'High'
        ph_notes = 'Speed weapon off bench — save for leverage spots'
    elif pow_r >= 55:
        ph_never_lift = True
        ph_notes = 'Power bat — keep in lineup'

    # Mode recommendation
    # Use Override only for extreme cases
    if ste >= 75 or ste <= 25:
        mode = 'Override'
        mode_reason = 'Extreme steal rating — override prevents team scenario from overriding'
    else:
        mode = 'Adjust'
        mode_reason = 'Normal range — adjust rides on team scenario variation'

    return {
        'Name': name,
        'POS': pos,
        'Age': int(age),
        'STE': int(ste),
        'SPE': int(spe),
        'POW': int(pow_r),
        'Mode': mode,
        'Steal': f"{steal_adj:+d} ({steal_label})",
        'Steal_val': steal_adj,
        'BaseRun': f"{baserun_adj:+d} ({baserun_label})",
        'BaseRun_val': baserun_adj,
        'H&R': f"{hr_adj:+d} ({hr_label})",
        'HR_val': hr_adj,
        'SacBunt': f"{sac_adj:+d} ({sac_label})",
        'BuntHit': f"{bunt_hit_adj:+d} ({bunt_hit_label})",
        'BuntHit_val': bunt_hit_adj,
        'PH_NeverLift': ph_never_lift,
        'PH_SaveLeverage': ph_save_high_leverage,
        'PH_Notes': ph_notes,
        'Mode_Reason': mode_reason,
    }


def get_pitcher_strategy(row):
    """Generate strategy recommendations for a pitcher."""
    mov = _safe(row.get('MOV', 50))
    stu = _safe(row.get('STU', 50))
    stm = _safe(row.get('STM', 50))
    age = _safe(row.get('Age', 25))
    pos = str(row.get('POS', ''))
    name = str(row.get('Name', ''))
    gf = str(row.get('G/F', 'NEU'))

    # Try PIT_CON first, fall back to CON
    con = _safe(row.get('PIT_CON', 0))
    if con == 0:
        con = _safe(row.get('CON', 50))

    # Hook timing
    if pos == 'SP':
        if age <= 23:
            hook_adj = -1  # Slightly quicker hook for young arms
            hook_label = 'Protect (age ≤ 23)'
            pitch_count = 95
        elif mov >= 55 and stm >= 50:
            hook_adj = +2  # Let him go deep
            hook_label = 'Slow hook (elite MOV + STM)'
            pitch_count = 0  # No limit
        elif stm >= 60:
            hook_adj = +1
            hook_label = 'Moderate (good STM)'
            pitch_count = 0
        else:
            hook_adj = 0
            hook_label = 'Team Default'
            pitch_count = 0
    else:  # RP/CL
        hook_adj = 0
        hook_label = 'Team Default'
        pitch_count = 0

    # Force Role
    if pos == 'CL' or (pos == 'RP' and stu >= 65):
        force_role = 'Closer'
        role_reason = f'STU {int(stu)} — elite stuff for late innings'
    elif pos == 'RP' and stu >= 55:
        force_role = 'Setup'
        role_reason = f'STU {int(stu)} — strong setup option'
    elif pos == 'RP' and mov >= 55:
        force_role = 'Middle Relief'
        role_reason = f'MOV {int(mov)} — ground ball specialist, eat innings'
    else:
        force_role = 'None'
        role_reason = 'Let game decide'

    # Pitch Around / IBB: GB pitchers should pitch to contact
    if gf in ('GB', 'EX GB'):
        pitch_around_adj = -1
        ibb_adj = -1
        contact_note = 'GB pitcher — pitch to contact, let defense work'
    elif gf in ('FB', 'EX FB'):
        pitch_around_adj = +1
        ibb_adj = 0
        contact_note = 'FB pitcher — more selective, avoid damage'
    else:
        pitch_around_adj = 0
        ibb_adj = 0
        contact_note = 'Neutral — use team default'

    # Hold Runners: GB pitchers benefit more (DP opportunities)
    if gf in ('GB', 'EX GB'):
        hold_adj = +1
        hold_note = 'GB staff + hold = DP opportunities'
    else:
        hold_adj = 0
        hold_note = 'Team Default'

    # Aggressive Tiredness Hook
    aggressive_tired = False
    if age >= 30 and pos == 'SP':
        aggressive_tired = True

    return {
        'Name': name,
        'POS': pos,
        'Age': int(age),
        'MOV': int(mov),
        'STU': int(stu),
        'CON': int(con),
        'STM': int(stm),
        'G/F': gf,
        'Hook': f"{hook_adj:+d} ({hook_label})",
        'Hook_val': hook_adj,
        'PitchCount': pitch_count if pitch_count > 0 else 'None',
        'ForceRole': force_role,
        'RoleReason': role_reason,
        'PitchAround': f"{pitch_around_adj:+d} ({contact_note})",
        'IBB': f"{ibb_adj:+d}",
        'HoldRunners': f"{hold_adj:+d} ({hold_note})",
        'AggressiveTired': aggressive_tired,
    }


def render_player_strategy(uploaded_df=None):
    st.header("🎯 Player Strategy Optimizer")
    st.caption("Per-player strategy adjustments based on ratings. Uses 'Adjust to Team' mode — "
               "values shown are RELATIVE to your team strategy settings.")

    if uploaded_df is None:
        st.info("Load a roster in the sidebar.")
        return

    df = uploaded_df
    team = st.selectbox("Team", sorted(df[df['TM'] != '-']['TM'].unique()), key='ps_team')
    team_df = df[df['TM'] == team]

    if st.button("🔄 Generate Player Strategies", type="primary", key='ps_gen'):
        batter_recs = []
        pitcher_recs = []

        for _, row in team_df.iterrows():
            pos = str(row.get('POS', ''))
            if pos in ('SP', 'RP', 'CL'):
                pitcher_recs.append(get_pitcher_strategy(row))
            else:
                batter_recs.append(get_batter_strategy(row))

        st.session_state['ps_batters'] = pd.DataFrame(batter_recs)
        st.session_state['ps_pitchers'] = pd.DataFrame(pitcher_recs)

    if 'ps_batters' not in st.session_state:
        return

    bat_df = st.session_state['ps_batters']
    pit_df = st.session_state['ps_pitchers']

    t1, t2, t3 = st.tabs(["Batter Strategy", "Pitcher Strategy", "Quick Reference"])

    with t1:
        st.subheader("Offensive Strategy Adjustments")
        st.markdown("**Mode:** Adjust to Team unless noted as Override. "
                     "Values are +/- from your team settings.")

        # Color code by steal classification
        st.markdown("#### Steal Classifications")

        green = bat_df[bat_df['Steal_val'] == 3].sort_values('STE', ascending=False)
        selective = bat_df[bat_df['Steal_val'] == 1].sort_values('STE', ascending=False)
        neutral = bat_df[bat_df['Steal_val'] == 0].sort_values('STE', ascending=False)
        red = bat_df[bat_df['Steal_val'] == -3].sort_values('STE', ascending=False)

        if not green.empty:
            st.markdown("🟢 **GREEN LIGHT (STE 65+) — Steal freely**")
            st.dataframe(green[['Name','POS','Age','STE','SPE','POW','Mode','Steal','BaseRun','BuntHit']],
                        use_container_width=True, hide_index=True)

        if not selective.empty:
            st.markdown("🟡 **SELECTIVE (STE 50-64) — Steal occasionally**")
            st.dataframe(selective[['Name','POS','Age','STE','SPE','POW','Mode','Steal','BaseRun','BuntHit']],
                        use_container_width=True, hide_index=True)

        if not neutral.empty:
            st.markdown("⚪ **NEUTRAL (STE 35-49) — Use team default**")
            st.dataframe(neutral[['Name','POS','Age','STE','SPE','POW','Mode','Steal','BaseRun']],
                        use_container_width=True, hide_index=True)

        if not red.empty:
            st.markdown("🔴 **RED LIGHT (STE < 35) — Never steal**")
            st.dataframe(red[['Name','POS','Age','STE','SPE','POW','Mode','Steal','BaseRun']],
                        use_container_width=True, hide_index=True)

        st.markdown("#### Pinch Hit / Bench Settings")
        ph_df = bat_df[bat_df['PH_Notes'] != ''][['Name','POS','Age','SPE','POW',
                                                    'PH_NeverLift','PH_SaveLeverage','PH_Notes']]
        if not ph_df.empty:
            st.dataframe(ph_df, use_container_width=True, hide_index=True)
        else:
            st.info("No special PH settings needed.")

    with t2:
        st.subheader("Pitching Strategy Adjustments")

        # Split by SP and RP
        sp_df = pit_df[pit_df['POS'] == 'SP'].sort_values('Age')
        rp_df = pit_df[pit_df['POS'].isin(['RP', 'CL'])].sort_values('STU', ascending=False)

        if not sp_df.empty:
            st.markdown("#### Starting Pitchers")
            st.dataframe(sp_df[['Name','Age','MOV','STU','CON','STM','G/F','Hook','PitchCount',
                                'PitchAround','HoldRunners','AggressiveTired']],
                        use_container_width=True, hide_index=True)

        if not rp_df.empty:
            st.markdown("#### Relief Pitchers")
            st.dataframe(rp_df[['Name','Age','MOV','STU','CON','STM','G/F','ForceRole',
                                'RoleReason','PitchAround','HoldRunners']],
                        use_container_width=True, hide_index=True)

    with t3:
        st.subheader("Quick Entry Reference")
        st.markdown("Copy these settings into OOTP player strategy screens.")

        st.markdown("#### Batters — Settings to Change")
        for _, r in bat_df.iterrows():
            changes = []
            if r['Steal_val'] != 0:
                changes.append(f"Stealing {r['Steal_val']:+d}")
            if r['BaseRun_val'] != 0:
                changes.append(f"Base-Running {r['BaseRun_val']:+d}")
            if r.get('HR_val', 0) != 0:
                changes.append(f"H&R {r['HR_val']:+d}")
            if r.get('BuntHit_val', 0) != 0:
                changes.append(f"Bunt for Hit {r['BuntHit_val']:+d}")
            if r['PH_NeverLift']:
                changes.append("☑ Never Lift for PH")
            if r['PH_SaveLeverage']:
                changes.append("☑ Save for High Leverage PH")

            if changes:
                mode_tag = "⚠️ OVERRIDE" if r['Mode'] == 'Override' else "Adjust"
                st.markdown(f"**{r['Name']}** ({r['POS']}, {r['Age']}) — {mode_tag}: {', '.join(changes)}")

        st.markdown("---")
        st.markdown("#### Pitchers — Settings to Change")
        for _, r in pit_df.iterrows():
            changes = []
            if r['Hook_val'] != 0:
                changes.append(f"Hook {r['Hook']}")
            if r['PitchCount'] != 'None':
                changes.append(f"Pitch Count: {r['PitchCount']}")
            if r['ForceRole'] != 'None':
                changes.append(f"Force Role: {r['ForceRole']}")
            if r['AggressiveTired']:
                changes.append("☑ Aggressive Tiredness Hook")

            if changes:
                st.markdown(f"**{r['Name']}** ({r['POS']}, {r['Age']}) — {', '.join(changes)}")

        st.markdown("---")
        st.markdown("#### Players with NO changes needed (use team defaults):")
        no_change_bat = bat_df[(bat_df['Steal_val'] == 0) & (bat_df['BaseRun_val'] == 0) &
                                (bat_df['BuntHit_val'] == 0) & (~bat_df['PH_NeverLift']) &
                                (~bat_df['PH_SaveLeverage'])]
        if not no_change_bat.empty:
            names = ', '.join(no_change_bat['Name'].tolist())
            st.markdown(f"Batters: {names}")

        no_change_pit = pit_df[(pit_df['Hook_val'] == 0) & (pit_df['ForceRole'] == 'None') &
                                (pit_df['PitchCount'] == 'None') & (~pit_df['AggressiveTired'])]
        if not no_change_pit.empty:
            names = ', '.join(no_change_pit['Name'].tolist())
            st.markdown(f"Pitchers: {names}")
