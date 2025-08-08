# app.py
import streamlit as st
from typing import List, Dict
import random

# ---- bring in your engine & data (must be in same folder) ----
from nba_draft_sim import (
    Player, Team, POSITIONS, PLAYER_DB,
    list_by_position, simulate_game
)

st.set_page_config(page_title="NBA Top-75 Draft & Game Simulator", layout="wide")
st.title("🏀 NBA Top-75 Draft & Game Simulator")

# ------------- compatibility for rerun across Streamlit versions -------------
def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # older versions
        st.experimental_rerun()

# ----------------------------- Helpers -----------------------------
def overall_score(p: Player) -> float:
    r = p.ratings
    return r['finishing'] + r['three'] + r['mid'] + r['playmaking'] + r['per_def'] + r['int_def'] + 0.5*r['rebounding']

def player_label(p: Player) -> str:
    return f"{p.name} • {p.pos_primary}  |  {p.ppg:.1f} PPG, 3P {int(p.three_pct*100)}%, TS {p.ts:.3f}"

def pool_for_position(pool: List[Player], pos: str) -> List[Player]:
    cands = [p for p in pool if (p.pos_primary == pos or pos in p.pos_secondary)]
    return sorted(cands, key=overall_score, reverse=True)

def render_bars(title: str, syn: Dict[str, float]):
    st.subheader(title)
    cols = st.columns(7)
    keys = [
        ("spacing", "Spacing"),
        ("rim_pressure", "Rim"),
        ("ball_move", "Ball Move"),
        ("int_def", "Int D"),
        ("per_def", "Per D"),
        ("rebounding", "Reb"),
        ("pace", "Pace"),
    ]
    for i, (k, label) in enumerate(keys):
        with cols[i]:
            val = syn[k]
            if k == "pace":
                pct = max(0, min(100, (val - 90) * (100 / (104 - 90))))  # normalize pace 90–104 to 0–100
            else:
                pct = max(0, min(100, val))  # synergy is already ~0–100
            st.markdown(f"**{label}**")
            st.progress(int(pct), text=f"{val:.1f}")

def team_box_rows(team: Team, result: Dict):
    rows = []
    for p in team.all_players():
        r = result['box'].get(p.name, {})
        rows.append({
            "Player": p.name,
            "PTS": r.get('pts',0),
            "FG": f"{r.get('fgm',0)}-{r.get('fga',0)}",
            "FT": f"{r.get('ftm',0)}-{r.get('fta',0)}",
            "AST": r.get('ast',0),
            "REB": r.get('orb',0)+r.get('drb',0),
            "ORB": r.get('orb',0),
            "DRB": r.get('drb',0),
        })
    return rows

def build_team(name: str, picks: Dict[str, Player]) -> Team:
    return Team(name=name, lineup=picks)

def _clear_pick_options_cache():
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith("opts_r"):
            del st.session_state[k]

# ----------------------------- State init -----------------------------
if "pool" not in st.session_state:
    st.session_state.pool = PLAYER_DB.copy()

if "team_a" not in st.session_state:
    st.session_state.team_a = {pos: None for pos in POSITIONS}
if "team_b" not in st.session_state:
    st.session_state.team_b = {pos: None for pos in POSITIONS}

if "draft_round" not in st.session_state:
    st.session_state.draft_round = 1  # 1..5
if "draft_index" not in st.session_state:
    st.session_state.draft_index = 0  # 0 or 1 within a round
if "history" not in st.session_state:
    st.session_state.history = []  # stack of (team, pos, player)

def reset_all():
    st.session_state.pool = PLAYER_DB.copy()
    st.session_state.team_a = {pos: None for pos in POSITIONS}
    st.session_state.team_b = {pos: None for pos in POSITIONS}
    st.session_state.draft_round = 1
    st.session_state.draft_index = 0
    st.session_state.history = []
    st.session_state.seed = ""
    _clear_pick_options_cache()

# ----------------------------- Sidebar controls -----------------------------
st.sidebar.header("Settings")
seed_in = st.sidebar.text_input("Random seed (optional)", value=st.session_state.get("seed",""))
try:
    seed_val = int(seed_in) if seed_in.strip() else None
except:
    seed_val = None
    st.sidebar.warning("Seed must be an integer.")
st.session_state.seed = seed_in

if st.sidebar.button("🔄 Reset Draft"):
    reset_all()
    _rerun()

# ----------------------------- Snake Draft Logic -----------------------------
def current_team_name() -> str:
    # Snake: Round 1 A->B, Round 2 B->A, etc.
    r = st.session_state.draft_round
    idx = st.session_state.draft_index  # 0 then 1 each round
    if r % 2 == 1:   # odd round: A then B
        return "Team A" if idx == 0 else "Team B"
    else:            # even round: B then A
        return "Team B" if idx == 0 else "Team A"

def team_dict(name: str) -> Dict[str, Player]:
    return st.session_state.team_a if name == "Team A" else st.session_state.team_b

def unfilled_positions(name: str) -> List[str]:
    t = team_dict(name)
    return [pos for pos in POSITIONS if t[pos] is None]

def make_pick(team_name: str, pos: str, player: Player):
    team_dict(team_name)[pos] = player
    st.session_state.pool = [p for p in st.session_state.pool if p.name != player.name]
    st.session_state.history.append((team_name, pos, player))

    # advance pointer
    if st.session_state.draft_index == 0:
        st.session_state.draft_index = 1
    else:
        st.session_state.draft_index = 0
        st.session_state.draft_round += 1

    _clear_pick_options_cache()

def undo_last():
    if not st.session_state.history:
        return
    team_name, pos, player = st.session_state.history.pop()
    # step pointer back
    if st.session_state.draft_index == 1:
        st.session_state.draft_index = 0
    else:
        st.session_state.draft_index = 1
        st.session_state.draft_round = max(1, st.session_state.draft_round - 1)
    # restore
    team_dict(team_name)[pos] = None
    st.session_state.pool.append(player)
    _clear_pick_options_cache()

# ----------------------------- Draft UI -----------------------------
col_left, col_right = st.columns([1,2], gap="large")

with col_left:
    st.subheader("🧢 Draft — On the Clock")
    if st.session_state.draft_round <= 5:
        team_name = current_team_name()
        st.markdown(f"**Round {st.session_state.draft_round}** · **{team_name}** is on the clock")

        positions_remaining = unfilled_positions(team_name)
        pos_choice = st.selectbox("Choose position", options=positions_remaining, key="draft_pos")

        # candidate pool filtered by position (FULL list, unsorted for variety)
        cands = [p for p in st.session_state.pool if (p.pos_primary == pos_choice or pos_choice in p.pos_secondary)]

        # create a stable 5-option list for the CURRENT pick (sample from ALL candidates)
        opt_key = f"opts_r{st.session_state.draft_round}_i{st.session_state.draft_index}_{team_name}_{pos_choice}"
        if opt_key not in st.session_state:
            k = min(5, len(cands))
            st.session_state[opt_key] = random.sample(cands, k) if k > 0 else []
        shown = st.session_state[opt_key]

        # Optional reshuffle button
        if st.button("🎲 Reshuffle options"):
            k = min(5, len(cands))
            st.session_state[opt_key] = random.sample(cands, k) if k > 0 else []
            _rerun()

        pick_choice = st.selectbox(
            "Pick player",
            options=shown if shown else cands,
            format_func=player_label,
            key="draft_player"
        )

        draft_cols = st.columns([1,1,1])
        with draft_cols[0]:
            if st.button("✅ Make Pick"):
                make_pick(team_name, pos_choice, pick_choice)
                _rerun()
        with draft_cols[1]:
            if st.button("🤖 Auto-Pick (Best)"):
                if cands:
                    best_from_shown = sorted(shown, key=overall_score, reverse=True)[0] if shown else cands[0]
                    make_pick(team_name, pos_choice, best_from_shown)
                    _rerun()
        with draft_cols[2]:
            if st.button("↩️ Undo Last Pick", disabled=len(st.session_state.history)==0):
                undo_last()
                _rerun()
    else:
        st.success("Draft complete! You can simulate the game below.")

    # Draft Board
    st.divider()
    st.subheader("📋 Draft Board")
    a_list = [f"{pos}: {st.session_state.team_a[pos].name if st.session_state.team_a[pos] else '—'}" for pos in POSITIONS]
    b_list = [f"{pos}: {st.session_state.team_b[pos].name if st.session_state.team_b[pos] else '—'}" for pos in POSITIONS]
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Team A**")
        st.write("\n".join(a_list))
    with b2:
        st.markdown("**Team B**")
        st.write("\n".join(b_list))

with col_right:
    st.subheader("Lineups")
    l1, l2 = st.columns(2)
    with l1:
        st.markdown("**Team A**")
        for pos in POSITIONS:
            p = st.session_state.team_a[pos]
            st.write(f"{pos}: {p.name if p else '—'}")
    with l2:
        st.markdown("**Team B**")
        for pos in POSITIONS:
            p = st.session_state.team_b[pos]
            st.write(f"{pos}: {p.name if p else '—'}")


# ----------------------------- Simulate -----------------------------
ready = all(st.session_state.team_a.values()) and all(st.session_state.team_b.values())

st.divider()
simulate = st.button("🚀 Simulate Game", disabled=not ready)
if simulate and ready:
    team_a = build_team("Team A", st.session_state.team_a)
    team_b = build_team("Team B", st.session_state.team_b)
    result = simulate_game(team_a, team_b, seed=seed_val)

    # Scoreline
    a, b = result["score"]
    st.markdown("## Final")
    st.markdown(f"**Team A {a} — Team B {b}**  •  Possessions: {result['possessions']}")

    # Synergy (clean bars)
    sa, sb = st.columns(2)
    with sa:
        render_bars("Team A — Synergy", result["syn"]["Team A"])
    with sb:
        render_bars("Team B — Synergy", result["syn"]["Team B"])

    # Box scores
    box1, box2 = st.columns(2)
    with box1:
        st.markdown("### Team A Box")
        st.dataframe(team_box_rows(team_a, result), use_container_width=True)
    with box2:
        st.markdown("### Team B Box")
        st.dataframe(team_box_rows(team_b, result), use_container_width=True)

    st.success("Game complete. Draft new squads or change the seed and run again!")
elif not ready:
    st.info("Finish the 5-round draft (snake) to enable simulation.")
