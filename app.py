import warnings

# Suppress Streamlit warnings globally
warnings.filterwarnings("ignore", category=UserWarning, message=".*ScriptRunContext.*", module="streamlit.runtime.scriptrunner")

import datetime
import random
import csv
from io import StringIO
from pickleball_scheduler_v2 import Player, Scheduler
import argparse
import json
import os

# --- Place robust_json_load at the very top so it is always defined before use ---
def robust_json_load(f):
    try:
        import json
        return json.load(f)
    except Exception as e:
        # Try to recover from extra data by loading only the first JSON object
        f.seek(0)
        raw = f.read()
        depth = 0
        for i, c in enumerate(raw):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        import json
                        return json.loads(raw[:i+1])
                    except Exception as e2:
                        raise Exception(f"Corrupt or invalid JSON in file: {e2}")
        raise Exception(f"Extra data found in file and could not recover: {e}")

# --- History Management Functions ---
HISTORY_FILE = "pickleball_history.json"
MAX_HISTORY = 52  # Number of past runs to retain (rolling window)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as file:
            return robust_json_load(file)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as file:
        json.dump(history, file)

def prune_history(history):
    return history[-MAX_HISTORY:]

def update_history(history, session_data):
    history.append(session_data)
    return prune_history(history)

# --- Stats Management Functions ---
STATS_FILE = "pickleball_stats.json"
MAX_STATS = 52  # Number of past runs to retain (rolling window)

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as file:
            return robust_json_load(file)
    return []

def save_stats(stats):
    with open(STATS_FILE, "w") as file:
        json.dump(stats, file)

def prune_stats(stats):
    return stats[-MAX_STATS:]

def update_stats(stats, stat_entry):
    stats.append(stat_entry)
    return prune_stats(stats)

# --- Global Configurations ---
# Check if running in terminal mode using argparse
parser = argparse.ArgumentParser(description="Pickleball Scheduler")
parser.add_argument('--date', type=str, help='Session date (YYYY-MM-DD)')
parser.add_argument('--start', type=str, help='Session start time (HH:MM)')
parser.add_argument('--games', type=int, help='Number of game slots')
parser.add_argument('--interval', type=int, help='Game interval in minutes')
parser.add_argument('--players', type=int, help='Number of players')
parser.add_argument("--mode", type=str, choices=["terminal", "streamlit"], help="Mode to run the application in (terminal or streamlit)", default="streamlit")
parser.add_argument("--duration", type=int, help="Session duration in hours", default=2)
args, unknown = parser.parse_known_args()

# Use --mode argument directly for mode detection
is_terminal_mode = args.mode == "terminal"

# Define a fallback for unique_emojis globally
unique_emojis = [
    "❤️",  # Red Heart
    "⭐",  # Star
    "⚽",  # Soccer Ball
    "🌳",  # Deciduous Tree
    "🚗",  # Car
    "🔑",  # Key
    "⏰",  # Alarm Clock
    "🎈",  # Balloon
    "🍕",  # Pizza Slice
    "💧",  # Droplet
    "⚡",  # High Voltage/Lightning
    "🌙",  # Crescent Moon
    "🐙",  # Octopus
    "🌵",  # Cactus
    "👻",  # Ghost
    "🚀",  # Rocket
    "👑",  # Crown
    "🎶",  # Musical Notes
    "💡",  # Light Bulb
    "☂️",  # Umbrella
    "⚓",  # Anchor
    "💎",  # Gem Stone
    "🧩",  # Puzzle Piece
    "🦋",  # Butterfly
    "🍦",  # Soft Ice Cream
    "🚲",  # Bicycle
    "📚",  # Books
    "🎯",  # Direct Hit/Bullseye
    "🗺️",  # World Map
    "⚙️",  # Gear
]

# Import Streamlit only if not in terminal mode
if not is_terminal_mode:
    import streamlit as st
    from streamlit.runtime.scriptrunner import ScriptRunContext

    st.set_page_config(page_title="Pickleball Scheduler Pro+ THIS IS PICKLEBALL V1", layout="wide")

    # Initialize session state variables if not present
    if 'players' not in st.session_state:
        st.session_state['players'] = []
    if 'constraints' not in st.session_state:
        st.session_state['constraints'] = []
    if 'subs' not in st.session_state:
        st.session_state['subs'] = []

    # Suppress Streamlit warnings globally
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*ScriptRunContext.*",
        module="streamlit.runtime.scriptrunner"
    )

    # Helper: assign unique emoji
    def assign_unique_emoji(existing_emojis):
        for emoji in unique_emojis:
            if emoji not in existing_emojis:
                return emoji
        return "🏓"  # Default emoji if all are used

    # Helper: get emoji and name
    def get_emoji_with_name(name):
        for p_data in st.session_state.players:
            if p_data['name'] == name:
                return f"{p_data['emoji']} {name}"
        return name

    # --- Sidebar: Session Setup ---
    st.sidebar.markdown("## Session Setup")
    st.sidebar.markdown("""Set up your session details below. Use the options to define the session date, start time, and other parameters.""")
    session_date = st.sidebar.date_input("Session Date", value=datetime.datetime.today().date(), help="Select the date for the session.", key="session_date_1")
    session_start = st.sidebar.time_input("Session Start Time (e.g., 2 PM, 14:00)", value=datetime.time(14, 0), help="Specify the start time for the session.", key="session_start_1")
    games_per_day = st.sidebar.number_input("Number of Game Slots", min_value=1, max_value=20, value=8, help="Enter the total number of game slots for the session.", key="games_per_day_1")
    game_length = st.sidebar.number_input("Game Interval (minutes)", min_value=10, max_value=60, value=15, help="Specify the duration of each game in minutes.", key="game_length_1")
    session_seed = st.sidebar.text_input("Random Seed (optional for consistent shuffle)", value="", help="Enter a random seed for reproducibility.", key="session_seed_1")

    # --- Dynamic courts logic
    num_players = len(st.session_state.players) if 'players' in st.session_state else 14
    # Dynamically determine allowed courts based on player count
    if num_players <= 1:
        auto_courts = 0
    elif num_players <= 4:
        auto_courts = 1
    elif num_players <= 11:
        auto_courts = 2
    elif num_players <= 15:
        auto_courts = 3
    else:
        auto_courts = 4
    allowed_courts = [i for i in range(0, 5)]  # 0, 1, 2, 3, 4

    court_override = st.sidebar.checkbox("Override number of courts?", value=False)
    if court_override:
        num_courts = st.sidebar.selectbox("Number of Courts", allowed_courts, index=allowed_courts.index(auto_courts))
    else:
        num_courts = auto_courts
        st.sidebar.markdown(f"**Number of Courts:** {num_courts} (auto)")
    # --- 1v1/2v2 logic ---
    # Add a toggle for 1v1 or 2v2 if at least 2 players and at least 1 court
    if num_players >= 2 and num_courts >= 1:
        if 'is_1v1' not in st.session_state:
            st.session_state['is_1v1'] = False
        st.session_state['is_1v1'] = st.sidebar.checkbox("Use 1v1 (singles) instead of 2v2 (doubles)?", value=st.session_state['is_1v1'])
    else:
        st.session_state['is_1v1'] = False

    # --- Save/Load Inputs Feature ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Save/Load Session Settings")
    if 'last_inputs_file' not in st.session_state:
        st.session_state['last_inputs_file'] = 'pickleball_inputs.json'
    inputs_file = st.sidebar.text_input("Inputs File", value=st.session_state['last_inputs_file'], key="inputs_file")
    save_inputs_clicked = st.sidebar.button("💾 Save Inputs", key="save_inputs_btn")
    load_inputs_clicked = st.sidebar.button("📂 Load Inputs", key="load_inputs_btn")
    inputs_error = None
    def get_inputs_dict():
        return {
            'session_date': str(session_date),
            'session_start': session_start.strftime("%H:%M"),
            'games_per_day': games_per_day,
            'game_length': game_length,
            'session_seed': session_seed,
            'num_courts': num_courts,
            'court_override': court_override,
            'players': st.session_state.players,
            'constraints': st.session_state.constraints,
            'subs': st.session_state.subs
        }
    if save_inputs_clicked:
        try:
            with open(inputs_file, 'w') as f:
                json.dump(get_inputs_dict(), f, default=str)
            st.session_state['last_inputs_file'] = inputs_file
            st.sidebar.success(f"Inputs saved to {inputs_file}")
        except Exception as e:
            inputs_error = f"Error saving inputs: {e}"
            st.sidebar.error(inputs_error)
    if load_inputs_clicked:
        try:
            with open(inputs_file, 'r') as f:
                data = robust_json_load(f)
            # Restore all fields
            st.session_state['players'] = data.get('players', [])
            st.session_state['constraints'] = data.get('constraints', [])
            st.session_state['subs'] = data.get('subs', [])
            # Restore sidebar fields
            session_date = datetime.datetime.strptime(data.get('session_date', str(datetime.datetime.today().date())), "%Y-%m-%d").date()
            session_start = datetime.datetime.strptime(data.get('session_start', "14:00"), "%H:%M").time()
            games_per_day = int(data.get('games_per_day', 8))
            game_length = int(data.get('game_length', 15))
            session_seed = data.get('session_seed', "")
            st.sidebar.success(f"Inputs loaded from {inputs_file}")
        except Exception as e:
            inputs_error = f"Error loading inputs: {e}"
            st.sidebar.error(inputs_error)

    # --- Generate Schedule Button ---
    generate_clicked = st.sidebar.button("🔶 Generate Schedule", use_container_width=True, key="generate_schedule_btn")

    # --- App Title ---
    st.markdown("""
    <h1 style='font-size:2.5rem; display:flex; align-items:center;'>
        <span style='font-size:2.5rem; margin-right:0.5em;'>🏓</span>Pickleball Scheduler Pro+
    </h1>
    """, unsafe_allow_html=True)

    # --- Custom CSS for styling ---
    st.markdown("""
    <style>
    body, .stApp {
        background-color: #181c20 !important;
        color: #f5f6fa !important;
    }
    .court-box, .court-box-green {
        background: #b2d8e6;
        border-radius: 10px;
        padding: 1em 1.5em;
        margin-bottom: 0.5em;
        display: flex;
        align-items: center;
        font-size: 1.35em !important;
        font-weight: 700 !important;
        color: #23272b !important;
        text-shadow: 1px 1px 4px #fff, 0 0 2px #23272b;
        letter-spacing: 0.02em;
    }
    .court-box-green {
        background: #a8e6a3;
    }
    .court-label {
        font-weight: bold;
        margin-bottom: 0.2em;
        color: #f5f6fa;
    }
    .slot-label {
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 0.5em;
        color: #f5f6fa;
    }
    .sitout-label {
        font-weight: bold;
    }
    .sitout-box {
        color: #ffb347;
        font-weight: 600;
        font-size: 1.05em;
        margin-bottom: 1em;
    }
    .vs-label {
        color: #444;
        font-weight: bold;
        margin: 0 1em;
        font-size: 1.1em;
    }
    hr {
        border: 1px solid #333;
        margin: 1em 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- More colorful, fun emoji set ---
    unique_emojis = [
        "❤️",  # Red Heart
        "⭐",  # Star
        "⚽",  # Soccer Ball
        "🌳",  # Deciduous Tree
        "🚗",  # Car
        "🔑",  # Key
        "⏰",  # Alarm Clock
        "🎈",  # Balloon
        "🍕",  # Pizza Slice
        "💧",  # Droplet
        "⚡",  # High Voltage/Lightning
        "🌙",  # Crescent Moon
        "🐙",  # Octopus
        "🌵",  # Cactus
        "👻",  # Ghost
        "🚀",  # Rocket
        "👑",  # Crown
        "🎶",  # Musical Notes
        "💡",  # Light Bulb
        "☂️",  # Umbrella
        "⚓",  # Anchor
        "💎",  # Gem Stone
        "🧩",  # Puzzle Piece
        "🦋",  # Butterfly
        "🍦",  # Soft Ice Cream
        "🚲",  # Bicycle
        "📚",  # Books
        "🎯",  # Direct Hit/Bullseye
        "🗺️",  # World Map
        "⚙️",  # Gear
    ]

    # --- Tabs ---
    tabs = st.tabs(["👥 Players", "⚙️ Constraints", "🔄 Substitutions", "📅 Schedule & Export"])

    # --- Tab 1: Players ---
    with tabs[0]:
        st.markdown("## 👥 Player Roster & Availability")
        col_add, col_clear = st.columns([2,2])
        with col_add:
            if st.button("➕ Add Player", key="add_player_button"):
                existing_emojis = [p['emoji'] for p in st.session_state.players]
                new_emoji = assign_unique_emoji(existing_emojis)
                st.session_state.players.append({
                    'name': f'Player {len(st.session_state.players)+1}',
                    'emoji': new_emoji,
                    'start': session_start,
                    'end': (datetime.datetime.combine(datetime.date.today(), session_start) + datetime.timedelta(hours=2)).time()
                })
        with col_clear:
            if st.button("🗑️ Clear All Players"):
                st.session_state['confirm_clear_players'] = True
            if st.session_state.get('confirm_clear_players', False):
                st.warning("Are you sure you want to remove all players?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Confirm Remove All Players", key="confirm_remove_players_btn"):
                        st.session_state.players = []
                        st.session_state['confirm_clear_players'] = False
                        st.rerun()
                with col2:
                    if st.button("Cancel", key="cancel_remove_players_btn"):
                        st.session_state['confirm_clear_players'] = False
        st.caption("Define session time in sidebar. Default availability is for the whole session.")
        # Update player availability display and input
        for i, p in enumerate(st.session_state.players):
            label_emoji = p['emoji'] if p['emoji'] else '🎾'
            with st.expander(f"{label_emoji} Player {i+1}: {p['name']}", expanded=True):
                cols = st.columns([3,1,2,2,1])
                with cols[0]:
                    p['name'] = st.text_input(f"Name {i+1}", value=p['name'], key=f"name_{i}", help="Enter the player's name.")
                with cols[1]:
                    p['emoji'] = st.text_input(f"Emoji {i+1}", value=p['emoji'], max_chars=2, key=f"emoji_{i}", help="Pick a fun emoji for this player.")
                with cols[2]:
                    p['start'] = st.time_input(f"Start {i+1} (AM/PM)", value=p['start'], key=f"start_{i}", help="Player's earliest available time.")
                with cols[3]:
                    p['end'] = st.time_input(f"End {i+1} (AM/PM)", value=p['end'], key=f"end_{i}", help="Player's latest available time.")
                with cols[4]:
                    if st.button("➖", key=f"remove_{i}"):
                        st.session_state.players.pop(i)
                        st.rerun()

    # --- Tab 2: Constraints ---
    with tabs[1]:
        st.markdown("## ⚙️ Player Constraints")
        st.info("Examples: 'Don't Play With (DNP)': Alice & Bob will never be on the same team. 'Don't Play Against (DNO)': Alice & Bob will never be on opposing teams. 'Avoid Each Other': Alice & Bob will be scheduled apart as much as possible.")
        player_names = [p['name'] for p in st.session_state.players]
        c1, c2, c3 = st.columns(3)
        with c1:
            constraint_type = st.selectbox("Constraint Type", ["Don't Play With (DNP)", "Don't Play Against (DNO)", "Avoid Each Other"], key="constraint_type", help="Choose the type of constraint.")
        with c2:
            p1 = st.selectbox("Player 1", player_names, key="c_p1", help="First player in the constraint.")
        with c3:
            p2 = st.selectbox("Player 2", [n for n in player_names if n != p1], key="c_p2", help="Second player in the constraint.")
        if st.button("Add Constraint"):
            st.session_state.constraints.append((constraint_type, p1, p2))
        st.markdown("**Current Constraints:**")
        for idx, (ctype, p1, p2) in enumerate(st.session_state.constraints):
            st.write(f"{ctype}: {p1} & {p2}", key=f"constraint_{idx}")
            if st.button(f"Remove Constraint {idx}"):
                st.session_state.constraints.pop(idx)
                st.rerun()

    # --- Tab 3: Substitutions ---
    with tabs[2]:
        st.markdown("## 🔄 Player Substitutions")
        st.info("Example: Substitute Bob in for Alice at round 3.")
        if len(player_names) >= 2:
            out_name = st.selectbox("Player Out", player_names, key="sub_out", help="Player to be substituted out.")
            in_name = st.selectbox("Player In", [n for n in player_names if n != out_name], key="sub_in", help="Player to be substituted in.")
            at_round = st.number_input("At round (1-based)", min_value=1, max_value=games_per_day, value=1, key="sub_round", help="Round number for the substitution.")
            if st.button("Add Substitution"):
                st.session_state.subs.append((out_name, in_name, at_round))
        st.markdown("**Current Substitutions:**")
        for idx, (out_name, in_name, at_round) in enumerate(st.session_state.subs):
            st.write(f"Round {at_round}: {out_name} → {in_name}")
            if st.button(f"Remove Sub {idx}"):
                st.session_state.subs.pop(idx)
                st.rerun()

    # --- Tab 4: Schedule & Export ---
    with tabs[3]:
        st.markdown("## 📅 Schedule Generation & Export")
        error = None
        # Only update history when a schedule is generated
        if 'history' not in st.session_state:
            st.session_state['history'] = load_history()
        if generate_clicked:
            try:
                # Ensure times are passed in AM/PM format to the scheduler
                players = [
                    Player(
                        p['name'],
                        datetime.datetime.strptime(p['start'].strftime("%I:%M %p"), "%I:%M %p").time(),
                        datetime.datetime.strptime(p['end'].strftime("%I:%M %p"), "%I:%M %p").time(),
                        emoji=p['emoji']
                    )
                    for p in st.session_state.players
                ]
                if session_seed:
                    random.seed(session_seed)
                scheduler = Scheduler(players, num_courts=num_courts, is_1v1=st.session_state.get('is_1v1', False))
                scheduler.games_per_day = games_per_day
                schedule = scheduler.generate_schedule(constraints=st.session_state.constraints, substitutions=st.session_state.subs)
                fairness = scheduler.assess_fairness(schedule)

                # --- Stats Management ---
                stats = load_stats()
                stats = update_stats(stats, fairness)
                save_stats(stats)
                # --- Update history only here ---
                session_data = {
                    "date": str(session_date),
                    "players": [p['name'] for p in st.session_state.players],
                    "sit_outs": fairness["sit_outs"],
                    "fairness_stats": {
                        "games_played_range": fairness["games_played_stats"]["range"],
                        "games_played_min": fairness["games_played_stats"]["min"],
                        "games_played_max": fairness["games_played_stats"]["max"],
                        "sit_outs_range": fairness["sit_outs_stats"]["range"],
                        "sit_outs_min": fairness["sit_outs_stats"]["min"],
                        "sit_outs_max": fairness["sit_outs_stats"]["max"],
                        "partners_range": fairness["partners_stats"]["range"],
                        "partners_min": fairness["partners_stats"]["min"],
                        "partners_max": fairness["partners_stats"]["max"],
                        "opponents_range": fairness["opponents_stats"]["range"],
                        "opponents_min": fairness["opponents_stats"]["min"],
                        "opponents_max": fairness["opponents_stats"]["max"],
                    }
                }
                st.session_state['history'] = update_history(st.session_state['history'], session_data)
                save_history(st.session_state['history'])

                # Display schedule and fairness assessment
                st.header("🗓️ Final Schedule")
                for rnd, round_info in enumerate(schedule, 1):
                    # Calculate the slot time for this round
                    slot_time = (datetime.datetime.combine(datetime.date.today(), session_start) + datetime.timedelta(minutes=game_length * (rnd - 1))).time()
                    st.markdown(f"<div class='slot-label'>Slot: {slot_time.strftime('%I:%M %p')}</div>", unsafe_allow_html=True)
                    for court_num, game in enumerate(round_info['games'], 1):
                        st.markdown(f"<div class='court-label'>Court {court_num}</div>", unsafe_allow_html=True)
                        # Use a single container for both teams and vs, with less spacing
                        st.markdown("""
                        <div style='display: flex; align-items: center; background: #23272b; border-radius: 12px; margin-bottom: 0.5em; padding: 0.5em 0.5em;'>
                            <div class='court-box' style='margin-bottom: 0; margin-right: 0.5em;'>
                                {team1}
                            </div>
                            <div class='vs-label' style='margin: 0 0.5em;'>vs</div>
                            <div class='court-box-green' style='margin-bottom: 0; margin-left: 0.5em;'>
                                {team2}
                            </div>
                        </div>
                        """.format(
                            team1='  '.join([get_emoji_with_name(p_name) for p_name in game.team1]),
                            team2='  '.join([get_emoji_with_name(p_name) for p_name in game.team2])
                        ), unsafe_allow_html=True)
                    if round_info['sit_out']:
                        sitout_str = ', '.join([get_emoji_with_name(p_name) for p_name in round_info['sit_out']])
                        st.markdown(f"<div class='sitout-label'>• Sit Out This Slot:</div> <span class='sitout-box'>{sitout_str}</span>", unsafe_allow_html=True)
                    st.markdown("<hr>", unsafe_allow_html=True)

                st.header("📊 Fairness Assessment")
                st.markdown("""
                This section shows how balanced the schedule is.
                - **Range**: The difference between the most and least. A smaller range is better.
                - **StdDev (Standard Deviation)**: How spread out the numbers are. Smaller is better.
                """)

                # Games Played
                with st.expander("Games Played Per Player", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Total Games for Each Player:**")
                        for player, count in fairness['games_played'].items():
                            st.write(f"{get_emoji_with_name(player)}: {count} games")
                    with col2:
                        stats = fairness['games_played_stats']
                        st.write("**Summary:**")
                        st.write(f"- Most games played by one player: {stats['max']}")
                        st.write(f"- Fewest games played by one player: {stats['min']}")
                        st.write(f"- Difference (Range): {stats['range']}")
                        st.write(f"- Spread (StdDev): {stats['stddev']:.2f}")

                # Sit-Outs
                with st.expander("Sit-Outs Per Player"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Total Sit-Outs for Each Player:**")
                        for player, count in fairness['sit_outs'].items():
                            st.write(f"{get_emoji_with_name(player)}: {count} sit-outs")
                    with col2:
                        stats = fairness['sit_outs_stats']
                        st.write("**Summary:**")
                        st.write(f"- Most sit-outs for one player: {stats['max']}")
                        st.write(f"- Fewest sit-outs for one player: {stats['min']}")
                        st.write(f"- Difference (Range): {stats['range']}")
                        st.write(f"- Spread (StdDev): {stats['stddev']:.2f}")

                # Partners Count
                with st.expander("Number of Unique Partners Per Player"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Unique Partners for Each Player:**")
                        for player, count in fairness['partners_count'].items():
                            st.write(f"{get_emoji_with_name(player)}: {count} partners")
                    with col2:
                        stats = fairness['partners_stats']
                        st.write("**Summary:**")
                        st.write(f"- Most unique partners: {stats['max']}")
                        st.write(f"- Fewest unique partners: {stats['min']}")
                        st.write(f"- Difference (Range): {stats['range']}")
                        st.write(f"- Spread (StdDev): {stats['stddev']:.2f}")
                
                # Opponents Count
                with st.expander("Number of Unique Opponents Per Player"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Unique Opponents for Each Player:**")
                        for player, count in fairness['opponents_count'].items():
                            st.write(f"{get_emoji_with_name(player)}: {count} opponents")
                    with col2:
                        stats = fairness['opponents_stats']
                        st.write("**Summary:**")
                        st.write(f"- Most unique opponents: {stats['max']}")
                        st.write(f"- Fewest unique opponents: {stats['min']}")
                        st.write(f"- Difference (Range): {stats['range']}")
                        st.write(f"- Spread (StdDev): {stats['stddev']:.2f}")

                # CSV download button
                output = StringIO()
                writer = csv.writer(output)
                writer.writerow(["Round", "Court", "Team 1", "Team 2", "Sit-out"])
                for rnd, round_info in enumerate(schedule, 1):
                    for game in round_info['games']:
                        t1 = ' '.join([p_name for p_name in game.team1])  # No emoji
                        t2 = ' '.join([p_name for p_name in game.team2])  # No emoji
                        writer.writerow([rnd, game.court, t1, t2, ''])
                    if round_info['sit_out']:
                        sitout_str = ', '.join([p_name for p_name in round_info['sit_out']])  # No emoji
                        writer.writerow([rnd, '', '', '', sitout_str])
                st.download_button("Download Schedule as CSV", output.getvalue(), file_name="pickleball_schedule.csv", mime="text/csv")

            except Exception as e:
                error = str(e)
                st.error(f"Error generating schedule: {error}")
        else:
            st.write("Click the 'Generate Schedule' button to create a schedule.")

        # --- Fairness History (summary, expandable) ---
        st.markdown("---")
        st.header("📝 Fairness History")
        history_error = None
        try:
            history = st.session_state.get('history', load_history())
        except Exception as e:
            history = []
            history_error = str(e)
            st.error(f"Could not load history: {history_error}")
            if st.button("Clear Corrupted History File"):
                save_history([])
                st.session_state['history'] = []
                st.success("History file cleared. Please try again.")
                st.rerun()
        if not history_error:
            if history:
                # --- Overall history stats summary ---
                all_stats = [h.get('fairness_stats', {}) for h in history if isinstance(h, dict) and 'fairness_stats' in h]
                def stat_summary(min_key, max_key):
                    min_vals = [s.get(min_key) for s in all_stats if s.get(min_key) is not None]
                    max_vals = [s.get(max_key) for s in all_stats if s.get(max_key) is not None]
                    if not min_vals or not max_vals:
                        return "N/A"
                    overall_min = min(min_vals)
                    overall_max = max(max_vals)
                    return f"min: {overall_min}, max: {overall_max}, range: {overall_max - overall_min}"
                st.markdown("**Overall History Stats (across all sessions):**")
                st.write(f"- Games Played: {stat_summary('games_played_min', 'games_played_max')}")
                st.write(f"- Sit-Outs: {stat_summary('sit_outs_min', 'sit_outs_max')}")
                st.write(f"- Partners: {stat_summary('partners_min', 'partners_max')}")
                st.write(f"- Opponents: {stat_summary('opponents_min', 'opponents_max')}")

                show_all = st.session_state.get('show_full_history', False)
                num_to_show = 3
                if not show_all and len(history) > num_to_show:
                    if st.button(f"Show All ({len(history)}) Sessions", key="show_all_history_btn"):
                        st.session_state['show_full_history'] = True
                        st.rerun()
                elif show_all and len(history) > num_to_show:
                    if st.button("Show Fewer", key="show_less_history_btn"):
                        st.session_state['show_full_history'] = False
                        st.rerun()
                entries = history if show_all else history[-num_to_show:]
                st.write(f"### Historical Fairness Data ({'All' if show_all else 'Recent'}) {len(entries)} of {len(history)} Sessions")
                valid_entries = 0
                for session in entries:
                    if not isinstance(session, dict):
                        st.warning(f"Corrupted history entry skipped. Type: {type(session)}, Value: {session}")
                        continue
                    valid_entries += 1
                    st.markdown(f"**Date:** {session.get('date', 'N/A')}")
                    fairness_stats = session.get('fairness_stats')
                    if fairness_stats:
                        st.write("- **Games Played Range:** {} (min: {}, max: {})".format(
                            fairness_stats.get('games_played_range', 'N/A'),
                            fairness_stats.get('games_played_min', 'N/A'),
                            fairness_stats.get('games_played_max', 'N/A')))
                        st.write("- **Sit-Outs Range:** {} (min: {}, max: {})".format(
                            fairness_stats.get('sit_outs_range', 'N/A'),
                            fairness_stats.get('sit_outs_min', 'N/A'),
                            fairness_stats.get('sit_outs_max', 'N/A')))
                        st.write("- **Partners Range:** {} (min: {}, max: {})".format(
                            fairness_stats.get('partners_range', 'N/A'),
                            fairness_stats.get('partners_min', 'N/A'),
                            fairness_stats.get('partners_max', 'N/A')))
                        st.write("- **Opponents Range:** {} (min: {}, max: {})".format(
                            fairness_stats.get('opponents_range', 'N/A'),
                            fairness_stats.get('opponents_min', 'N/A'),
                            fairness_stats.get('opponents_max', 'N/A')))
                    else:
                        st.write("- No fairness stats available for this session.")
                    st.markdown("---")
                if valid_entries == 0:
                    st.error("All history entries are invalid or corrupted. Please clear the history to restore functionality.")
                # --- Add always-visible Clear History button ---
                if history:
                    if not st.session_state.get('confirm_clear_history', False):
                        if st.button('🗑️ Clear History', key='clear_history_btn'):
                            st.session_state['confirm_clear_history'] = True
                            st.rerun()
                # --- Fix: Show confirmation/cancel immediately after first click, and don't create new history entries on rerun ---
                if st.session_state.get('confirm_clear_history', False):
                    st.warning("Are you sure you want to clear all history?")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Confirm Clear History", key="confirm_clear_history_button"):
                            save_history([])
                            st.session_state['history'] = []
                            st.session_state['confirm_clear_history'] = False
                            st.success("History cleared successfully!")
                            st.rerun()
                    with col2:
                        if st.button("Cancel", key="cancel_clear_history_button"):
                            st.session_state['confirm_clear_history'] = False
            else:
                st.info("No historical data available.")

        # --- Stats Corruption Handling ---
        stats_error = None
        try:
            _ = load_stats()
        except Exception as e:
            stats_error = str(e)
            st.error(f"Could not load stats: {stats_error}")
            if st.button("Clear Corrupted Stats File"):
                save_stats([])
                st.success("Stats file cleared. Please try again.")
                st.rerun()

    # --- Visual Tutorial Overlay ---
    # (Removed: visual tutorial overlay and related logic)
    # If you want to re-enable, restore the tutorial_steps, tutorial_nav_btns, and related Streamlit UI code here.
# --- Terminal Mode ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pickleball Scheduler in terminal mode.")
    parser.add_argument("--date", type=str, help="Session date in YYYY-MM-DD format")
    parser.add_argument("--start", type=str, help="Session start time in HH:MM format (24-hour)")
    parser.add_argument("--games", type=int, help="Number of game slots", default=8)
    parser.add_argument("--interval", type=int, help="Game interval in minutes", default=15)
    parser.add_argument("--seed", type=str, help="Random seed for consistent shuffle", default="")
    parser.add_argument("--players", type=int, help="Number of players", default=14)
    parser.add_argument("--mode", type=str, choices=["terminal", "streamlit"], help="Mode to run the application in (terminal or streamlit)", default="streamlit")
    parser.add_argument("--duration", type=int, help="Session duration in hours", default=2)

    args = parser.parse_args()

    # Set default values for terminal mode arguments
    if args.date is None:
        args.date = datetime.datetime.today().strftime("%Y-%m-%d")
    if args.start is None:
        args.start = "14:00"  # Default start time set to 2 PM (14:00)

    # Parse date and time inputs
    session_date = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
    session_start = datetime.datetime.strptime(args.start, "%H:%M").time()

    # Generate players dynamically
    players = [
        Player(
            name=f'Player {i+1}',
            available_from=session_start,
            available_to=(datetime.datetime.combine(session_date, session_start) + datetime.timedelta(hours=2)).time(),
            emoji=unique_emojis[i % len(unique_emojis)]
        )
        for i in range(args.players)
    ]

    # Determine number of courts
    if args.players <= 11:
        num_courts = 2
    elif args.players <= 15:
        num_courts = 3
    else:
        num_courts = 4

    # Extract fairness assessment and stats calculation into a reusable function
    def display_fairness_stats(fairness):
        print("\n📊 Fairness Assessment")
        print("This section shows how balanced the schedule is.")
        print("- Range: The difference between the most and least. A smaller range is better.")
        print("- StdDev (Standard Deviation): How spread out the numbers are. Smaller is better.\n")

        # Games Played
        print("**Games Played Per Player:**")
        for player, count in fairness['games_played'].items():
            print(f"{player}: {count} games")
        stats = fairness['games_played_stats']
        print("\n**Summary:**")
        print(f"- Most games played by one player: {stats['max']}")
        print(f"- Fewest games played by one player: {stats['min']}")
        print(f"- Difference (Range): {stats['range']}")
        print(f"- Spread (StdDev): {stats['stddev']:.2f}\n")

        # Sit-Outs
        print("**Sit-Outs Per Player:**")
        for player, count in fairness['sit_outs'].items():
            print(f"{player}: {count} sit-outs")
        stats = fairness['sit_outs_stats']
        print("\n**Summary:**")
        print(f"- Most sit-outs for one player: {stats['max']}")
        print(f"- Fewest sit-outs for one player: {stats['min']}")
        print(f"- Difference (Range): {stats['range']}")
        print(f"- Spread (StdDev): {stats['stddev']:.2f}\n")

        # Partners Count
        print("**Number of Unique Partners Per Player:**")
        for player, count in fairness['partners_count'].items():
            print(f"{player}: {count} partners")
        stats = fairness['partners_stats']
        print("\n**Summary:**")
        print(f"- Most unique partners: {stats['max']}")
        print(f"- Fewest unique partners: {stats['min']}")
        print(f"- Difference (Range): {stats['range']}")
        print(f"- Spread (StdDev): {stats['stddev']:.2f}\n")

        # Opponents Count
        print("**Number of Unique Opponents Per Player:**")
        for player, count in fairness['opponents_count'].items():
            print(f"{player}: {count} opponents")
        stats = fairness['opponents_stats']
        print("\n**Summary:**")
        print(f"- Most unique opponents: {stats['max']}")
        print(f"- Fewest unique opponents: {stats['min']}")
        print(f"- Difference (Range): {stats['range']}")
        print(f"- Spread (StdDev): {stats['stddev']:.2f}\n")

    # Initialize Scheduler and generate schedule
    try:
        if args.seed:
            random.seed(args.seed)
        scheduler = Scheduler(players, num_courts=num_courts, is_1v1=st.session_state.get('is_1v1', False))
        scheduler.games_per_day = args.games

        # Load historical data
        try:
            history = load_history()
        except Exception as e:
            print(f"Warning: Could not load history file: {e}\nHistory will be reset.")
            save_history([])
            history = []

        # Terminal mode does not use Streamlit session state
        schedule = scheduler.generate_schedule()
        fairness = scheduler.assess_fairness(schedule)

        # --- Stats Management ---
        try:
            stats = load_stats()
        except Exception as e:
            print(f"Warning: Could not load stats file: {e}\nStats will be reset.")
            save_stats([])
            stats = []
        stats = update_stats(stats, fairness)
        save_stats(stats)

        # Update history with the latest session
        session_data = {
            "date": args.date,
            "players": [p.name for p in players],
            "sit_outs": fairness["sit_outs"],
            # Add fairness_stats for history summary
            "fairness_stats": {
                "games_played_range": fairness["games_played_stats"]["range"],
                "games_played_min": fairness["games_played_stats"]["min"],
                "games_played_max": fairness["games_played_stats"]["max"],
                "sit_outs_range": fairness["sit_outs_stats"]["range"],
                "sit_outs_min": fairness["sit_outs_stats"]["min"],
                "sit_outs_max": fairness["sit_outs_stats"]["max"],
                "partners_range": fairness["partners_stats"]["range"],
                "partners_min": fairness["partners_stats"]["min"],
                "partners_max": fairness["partners_stats"]["max"],
                "opponents_range": fairness["opponents_stats"]["range"],
                "opponents_min": fairness["opponents_stats"]["min"],
                "opponents_max": fairness["opponents_stats"]["max"],
            }
        }
        history = update_history(history, session_data)
        save_history(history)

        # Display schedule in terminal
        print("\n🗓️ Final Schedule")
        for rnd, round_info in enumerate(schedule, 1):
            print(f"\nRound {rnd}")
            for game in round_info['games']:
                t1 = ' & '.join(game.team1)
                t2 = ' & '.join(game.team2)
                print(f"Court {game.court}: {t1} vs {t2}")
            if round_info['sit_out']:
                sitout_str = ', '.join(round_info['sit_out'])
                print(f"🛋️ Sit-out: {sitout_str}")

        # Display fairness stats
        display_fairness_stats(fairness)

    except Exception as e:
        print(f"Error generating schedule: {e}")
        print("If you see a JSON error, try deleting or clearing pickleball_history.json and pickleball_stats.json.")
