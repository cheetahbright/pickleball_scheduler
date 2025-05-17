#!/usr/bin/env python3
import streamlit as st
import datetime
import json
import logging
import random
import shutil
import os
from pathlib import Path
import re
from typing import List, Dict, Tuple, Set, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import csv
import math
import io
import uuid

# --- Global Constants ---
DEFAULT_HISTORY_FILE = Path('pickleball_schedule_history_v4.5.json')
DRAFT_FILE = Path('pickleball_schedule_draft.json')
MAX_RETRIES = 3000
MAX_HISTORY_WEEKS = 52

# --- Setup Logger ---
logger = logging.getLogger("PickleballSchedulerStreamlit")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)

# --- Math Helper & Player Emojis ---
try:
    from math import comb
except ImportError:
    from math import factorial
    def comb(n: int, k: int) -> int:
        if k < 0 or k > n: return 0
        if k == 0 or k == n: return 1
        if k > n // 2: k = n - k
        res = 1
        for i in range(k): res = res * (n - i) // (i + 1)
        return res

PLAYER_EMOJIS = ["😀", "😎", "🤩", "🥳", "🏓", "🎾", "🥇", "👍", "💪", "🎉", "✨", "🚀", "🌟", "🎯", "🏆", "🐱", "🐶", "🦊", "🐻", "🐼", "🐸", "🐢", "🐧", "🦉"]
_emoji_counter = 0
def get_next_emoji():
    global _emoji_counter
    emoji = PLAYER_EMOJIS[_emoji_counter % len(PLAYER_EMOJIS)]
    _emoji_counter += 1
    return emoji

# --- Data Classes ---
@dataclass(frozen=True, eq=True)
class Player:
    name: str
    availability_windows: List[Tuple[datetime.time, datetime.time]]
    def __str__(self): return self.name

# --- UI Class (Minimal, for _parse_time_flexible) ---
class UI:
    @staticmethod
    def _parse_time_flexible(time_str_orig: str) -> Optional[datetime.time]:
        if not time_str_orig or not time_str_orig.strip(): return None
        time_str_norm = time_str_orig.upper().replace(" ", "")
        extracted_am_pm = ""; time_str_base = time_str_norm
        if "AM" in time_str_norm: extracted_am_pm = "AM"; time_str_base = time_str_norm.replace("AM", "")
        elif "PM" in time_str_norm: extracted_am_pm = "PM"; time_str_base = time_str_norm.replace("PM", "")
        if extracted_am_pm:
            temp_h_str, temp_m_str = "", "00"; numeric_part = "".join(filter(str.isdigit, time_str_base))
            if not numeric_part: pass
            elif ':' in time_str_base:
                parts = time_str_base.split(':', 1); temp_h_str = parts[0]
                if len(parts) > 1 and parts[1]: cleaned_minute_part = "".join(filter(str.isdigit, parts[1])); temp_m_str = cleaned_minute_part if cleaned_minute_part else "00"
            elif len(numeric_part) == 1: temp_h_str = numeric_part
            elif len(numeric_part) == 2: temp_h_str = numeric_part
            elif len(numeric_part) == 3: temp_h_str = numeric_part[0]; temp_m_str = numeric_part[1:]
            elif len(numeric_part) == 4: temp_h_str = numeric_part[:2]; temp_m_str = numeric_part[2:]
            if temp_h_str:
                try:
                    val_h, val_m = int(temp_h_str), int(temp_m_str)
                    if not (1 <= val_h <= 12 and 0 <= val_m <= 59): raise ValueError()
                    return datetime.datetime.strptime(f"{str(val_h).zfill(2)}:{str(val_m).zfill(2)}{extracted_am_pm}", "%I:%M%p").time()
                except ValueError: pass
        temp_h_str, temp_m_str = "", "00"; numeric_part_24h = "".join(filter(str.isdigit, time_str_norm))
        if not numeric_part_24h and ':' not in time_str_norm: pass
        elif ':' in time_str_norm:
            parts = time_str_norm.split(':',1); temp_h_str = parts[0]
            if len(parts) > 1 and parts[1]: cleaned_minute_part = "".join(filter(str.isdigit, parts[1].rstrip("APM"))); temp_m_str = cleaned_minute_part if cleaned_minute_part else "00"
        elif len(numeric_part_24h) == 1: temp_h_str = numeric_part_24h
        elif len(numeric_part_24h) == 2: temp_h_str = numeric_part_24h
        elif len(numeric_part_24h) == 3: temp_h_str = numeric_part_24h[0]; temp_m_str = numeric_part_24h[1:]
        elif len(numeric_part_24h) == 4: temp_h_str = numeric_part_24h[:2]; temp_m_str = numeric_part_24h[2:]
        if temp_h_str:
            try:
                val_h, val_m = int(temp_h_str), int(temp_m_str)
                if not (0 <= val_h <= 23 and 0 <= val_m <= 59): raise ValueError()
                return datetime.datetime.strptime(f"{str(val_h).zfill(2)}:{str(val_m).zfill(2)}", "%H:%M").time()
            except ValueError: pass
        return None

# --- HistoryManager ---
class HistoryManager:
    def __init__(self, history_file: Path = DEFAULT_HISTORY_FILE):
        self.history_file = history_file.resolve()

    def load(self) -> Dict[str, Any]:
        default = {'weeks': [], 'incompatibles': [], 'dno_pairs': [], 'avoid_pairs': [] }
        if not self.history_file.exists():
            logger.info(f"History file '{self.history_file}' not found. Starting fresh.")
            return default
        try:
            with self.history_file.open('r') as f:
                history_data = json.load(f)
            for key in ['weeks', 'incompatibles', 'dno_pairs', 'avoid_pairs']:
                if not isinstance(history_data.get(key), list):
                    history_data[key] = []
            return history_data
        except json.JSONDecodeError:
            logger.exception("Failed to parse history file '%s'.", self.history_file)
            st.warning(f"Corrupted history file '{self.history_file}'. Using fresh data for this session.")
            return default
        except Exception as e:
            logger.error(f"An unexpected error occurred loading history: {e}")
            st.error(f"Error loading history file '{self.history_file}'. Using fresh data.")
            return default

    def save(self, history: Dict[str, Any]):
        pruned_weeks = history.get('weeks', [])[-MAX_HISTORY_WEEKS:]
        def validate_pair_list(pair_list_name: str, pair_list: List[Any]) -> List[List[str]]:
            valid_pairs = []
            for item in pair_list:
                if isinstance(item, (list, tuple)) and len(item) == 2 and \
                   isinstance(item[0], str) and isinstance(item[1], str):
                    valid_pairs.append(list(sorted(item)))
                else:
                    logger.warning(f"Skipping invalid item in '{pair_list_name}' during save: {item}")
            return valid_pairs

        pruned_history = {
            'weeks': pruned_weeks,
            'incompatibles': validate_pair_list('incompatibles', history.get('incompatibles', [])),
            'dno_pairs': validate_pair_list('dno_pairs', history.get('dno_pairs', [])),
            'avoid_pairs': validate_pair_list('avoid_pairs', history.get('avoid_pairs', []))
        }
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with self.history_file.open('w') as f:
                json.dump(pruned_history, f, indent=2)
            logger.info(f"History saved to {self.history_file}")
        except Exception as e:
            logger.error(f"An unexpected error occurred saving history: {e}")
            st.error(f"Failed to save history to '{self.history_file}': {e}")

# --- Scheduler Class ---
class Scheduler:
    def __init__(self, courts: int, rng: random.Random, incompatibles: List[Tuple[str, str]], dno_pairs: List[Tuple[str,str]], game_interval_minutes: int, matchmaking_strategy: str = "Standard Rotation", historical_win_rates: Optional[Dict[str, float]] = None):
        self.courts = courts; self.rng = rng; self.game_interval_minutes = game_interval_minutes
        self.incompatibles = [tuple(sorted(pair)) for pair in incompatibles]
        self.dno_pairs = [tuple(sorted(pair)) for pair in dno_pairs]
        self.matchmaking_strategy = matchmaking_strategy
        self.historical_win_rates = historical_win_rates if historical_win_rates else {}
    @staticmethod
    def determine_courts(num_players: int) -> int:
        if num_players <= 0: return 0
        if 8 <= num_players <= 11: return 2
        if 12 <= num_players <= 15: return 3
        if num_players >= 16: return 4
        return 1 if num_players >= 4 else 0
    @staticmethod
    def time_slots(start_time_obj: datetime.time, games: int = 8, interval_minutes: int = 15) -> List[datetime.time]:
        slots: List[datetime.time] = []; current_time_dt = datetime.datetime.combine(datetime.date.min, start_time_obj)
        for _ in range(games): slots.append(current_time_dt.time()); current_time_dt += datetime.timedelta(minutes=interval_minutes)
        return slots
    def is_player_available_for_game_slot(self, player: Player, slot_start_time: datetime.time) -> bool:
        try: game_start_dt = datetime.datetime.combine(datetime.date.min, slot_start_time); game_end_dt = game_start_dt + datetime.timedelta(minutes=self.game_interval_minutes); required_game_end_time = game_end_dt.time()
        except Exception: return False
        for p_start, p_end in player.availability_windows:
            if p_start <= slot_start_time and required_game_end_time <= p_end: return True
        return False
    def _get_active_player_names_for_slot(self, all_session_players: List[Player], substitutions_map: Dict[str, Tuple[str, str]], current_slot_time: datetime.time) -> List[str]:
        statically_available_player_names = set(); player_obj_map = {p.name: p for p in all_session_players}
        for p_obj in all_session_players:
            if self.is_player_available_for_game_slot(p_obj, current_slot_time): statically_available_player_names.add(p_obj.name)
        current_active_player_names = set(statically_available_player_names)
        for p_out_name, (p_in_name, swap_time_str) in substitutions_map.items():
            try: swap_time = datetime.datetime.strptime(swap_time_str, "%I:%M%p").time()
            except ValueError: continue
            if current_slot_time >= swap_time:
                if p_out_name in current_active_player_names: current_active_player_names.remove(p_out_name)
                if p_in_name: current_active_player_names.add(p_in_name)
            else:
                p_in_obj = player_obj_map.get(p_in_name); is_p_in_statically_available_now = False
                if p_in_obj: is_p_in_statically_available_now = self.is_player_available_for_game_slot(p_in_obj, current_slot_time)
                if p_in_name in current_active_player_names and not is_p_in_statically_available_now:
                    if not p_in_obj or (p_in_obj and not is_p_in_statically_available_now): current_active_player_names.remove(p_in_name)
        return list(current_active_player_names)
    def prepare_players_for_matches(self, active_player_names: List[str], games_played_this_session: Dict[str, int]) -> List[str]:
        if self.matchmaking_strategy == "Balance by Win Rate (Experimental)" and self.historical_win_rates:
            eligible_players = sorted(active_player_names, key=lambda p_name: (self.historical_win_rates.get(p_name, 0.5), games_played_this_session.get(p_name, 0)))
        else:
            eligible_players = sorted(active_player_names, key=lambda p_name: games_played_this_session.get(p_name, 0))
        self.rng.shuffle(eligible_players)
        num_players_for_courts = self.courts * 4; players_for_matches = eligible_players[:num_players_for_courts]
        if len(players_for_matches) < 4 and len(players_for_matches) > 0 : return []
        return players_for_matches if len(players_for_matches) >= 2 else []
    def make_matches(self, slot_players_for_match_pool: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        matches: List[Dict[str, Any]] = []; players_assigned_to_matches = set(); temp_player_pool = list(slot_players_for_match_pool)
        for court_idx in range(self.courts):
            if len(temp_player_pool) < 4: break
            self.rng.shuffle(temp_player_pool); current_foursome = temp_player_pool[:4]
            team1 = [current_foursome[0], current_foursome[1]]; team2 = [current_foursome[2], current_foursome[3]]
            matches.append({'court': court_idx + 1, 'team1': team1, 'team2': team2, 'winner': None})
            for p in current_foursome: players_assigned_to_matches.add(p); temp_player_pool.remove(p)
        return matches, list(players_assigned_to_matches)
    def enforce_constraints(self, matches: List[Dict[str, Any]], partners_this_session: Set[Tuple[str, str]], opponents_this_session: Set[Tuple[str, str]]) -> bool:
        temp_partners_for_check_this_slot = set(); temp_opponents_for_check_this_slot = set()
        for m in matches:
            if 'team1' not in m or 'team2' not in m or len(m['team1']) != 2 or len(m['team2']) != 2: return False
            team1_players = m['team1']; team2_players = m['team2']
            t1_pair = tuple(sorted(team1_players)); t2_pair = tuple(sorted(team2_players))
            for inc_pair in self.incompatibles:
                if set(inc_pair).issubset(set(t1_pair)) or set(inc_pair).issubset(set(t2_pair)): return False
            for dno_p in self.dno_pairs:
                for p1_team1 in team1_players:
                    for p2_team2 in team2_players:
                        if set(dno_p) == set((p1_team1, p2_team2)): return False
            if t1_pair in temp_partners_for_check_this_slot or t1_pair in partners_this_session: return False
            if t2_pair in temp_partners_for_check_this_slot or t2_pair in partners_this_session: return False
            for p1 in team1_players:
                for p2 in team2_players:
                    opponent_pair_candidate = tuple(sorted((p1, p2)))
                    if opponent_pair_candidate in temp_opponents_for_check_this_slot or opponent_pair_candidate in opponents_this_session: return False
            temp_partners_for_check_this_slot.add(t1_pair); temp_partners_for_check_this_slot.add(t2_pair)
            for p1 in team1_players:
                for p2 in team2_players: temp_opponents_for_check_this_slot.add(tuple(sorted((p1,p2))))
        return True
    def select_matches_for_slot(self, slot_player_names_available: List[str], partners_this_session: Set[Tuple[str, str]], opponents_this_session: Set[Tuple[str, str]], games_played_this_session: Dict[str, int], current_slot_time_obj: Optional[datetime.time] = None) -> Tuple[List[Dict[str, Any]], Set[str]]:
        players_for_match_pool = self.prepare_players_for_matches(slot_player_names_available, games_played_this_session)
        players_in_final_matches_set = set()
        if not players_for_match_pool or len(players_for_match_pool) < 4 :
            if slot_player_names_available: logger.info(f"Not enough for game: {', '.join(slot_player_names_available)}")
            return [], players_in_final_matches_set
        for attempt in range(MAX_RETRIES):
            candidate_matches, players_in_cand_matches = self.make_matches(list(players_for_match_pool))
            if self.enforce_constraints(candidate_matches, partners_this_session, opponents_this_session):
                players_in_final_matches_set.update(players_in_cand_matches)
                return list(candidate_matches), players_in_final_matches_set
        slot_time_str = current_slot_time_obj.strftime('%I:%M %p') if current_slot_time_obj else "the current slot"
        st.warning(f"Scheduler tried {MAX_RETRIES} times for slot {slot_time_str}. Using last attempt (may violate rules).")
        last_attempt_matches, players_in_last_attempt = self.make_matches(list(players_for_match_pool))
        players_in_final_matches_set.update(players_in_last_attempt)
        return list(last_attempt_matches), players_in_final_matches_set
    def update_session_history_and_stats(self, matches_in_slot: List[Dict[str, Any]], session_partners: Set[Tuple[str, str]], session_opponents: Set[Tuple[str, str]], session_games_played: Dict[str, int]):
        for m in matches_in_slot:
            if 'team1' in m and 'team2' in m:
                team1_players = m['team1']; team2_players = m['team2']
                for p_name in team1_players + team2_players: session_games_played[p_name] += 1
                session_partners.add(tuple(sorted(team1_players))); session_partners.add(tuple(sorted(team2_players)))
                for p1_name in team1_players:
                    for p2_name in team2_players: session_opponents.add(tuple(sorted((p1_name, p2_name))))
    def schedule_day(self, time_slots_today: List[datetime.time], all_session_players: List[Player], substitutions_map: Dict[str, Tuple[str, str]]) -> Tuple[List[Tuple[datetime.time, List[Dict[str, Any]]]], Set[Tuple[str, str]], Set[Tuple[str, str]], Dict[str, int], Dict[str, int], Dict[datetime.time, List[str]], Dict[datetime.time, Dict[str, Set[str]]]]:
        partners_this_session: Set[Tuple[str,str]] = set(); opponents_this_session: Set[Tuple[str,str]] = set()
        games_played_this_session: Dict[str, int] = defaultdict(int); sit_out_counts_this_session: Dict[str, int] = defaultdict(int)
        full_schedule_today: List[Tuple[datetime.time, List[Dict[str, Any]]]] = []
        active_players_for_csv: Dict[datetime.time, List[str]] = {}; slot_assignments_for_csv: Dict[datetime.time, Dict[str, Set[str]]] = {}
        for slot_time in time_slots_today:
            active_player_names = self._get_active_player_names_for_slot(all_session_players, substitutions_map, slot_time)
            active_players_for_csv[slot_time] = list(active_player_names); num_active = len(active_player_names)
            self.courts = Scheduler.determine_courts(num_active)
            slot_actual_matches: List[Dict[str, Any]] = []; players_who_played_this_slot: Set[str] = set()
            if not(num_active < 4 or self.courts == 0):
                slot_actual_matches, players_who_played_this_slot = self.select_matches_for_slot(active_player_names, partners_this_session, opponents_this_session, games_played_this_session, slot_time)
            self.update_session_history_and_stats(slot_actual_matches, partners_this_session, opponents_this_session, games_played_this_session)
            slot_sit_outs_final = [p_name for p_name in active_player_names if p_name not in players_who_played_this_slot]
            for p_name in slot_sit_outs_final: sit_out_counts_this_session[p_name] += 1
            slot_assignments_for_csv[slot_time] = {'playing': players_who_played_this_slot, 'sitting_out_slot': set(slot_sit_outs_final)}
            combined_slot_activities = list(slot_actual_matches)
            if slot_sit_outs_final: combined_slot_activities.append({'sit_out': slot_sit_outs_final})
            full_schedule_today.append((slot_time, combined_slot_activities))
        return full_schedule_today, partners_this_session, opponents_this_session, games_played_this_session, sit_out_counts_this_session, active_players_for_csv, slot_assignments_for_csv

# --- Streamlit UI Helper Functions ---
def parse_availability_string_st(avail_str: str, session_start_time: datetime.time, session_end_time: datetime.time) -> List[Tuple[datetime.time, datetime.time]]:
    if not avail_str.strip(): return [(session_start_time, session_end_time)]
    windows = []; parts = avail_str.split(';')
    for part_str in parts:
        part_str = part_str.strip()
        if not part_str: continue
        if '-' not in part_str: st.warning(f"Avail segment '{part_str}' missing '-'."); continue
        try:
            start_str, end_str = [s.strip() for s in part_str.split('-', 1)]; start_t = UI._parse_time_flexible(start_str); end_t = UI._parse_time_flexible(end_str)
            if start_t and end_t:
                if start_t >= end_t: st.warning(f"For '{part_str}', start must be before end."); continue
                windows.append((start_t, end_t))
            else:
                if not start_t: st.warning(f"Bad start time '{start_str}' in '{part_str}'.")
                if not end_t: st.warning(f"Bad end time '{end_str}' in '{part_str}'.")
        except Exception as e: st.warning(f"Error parsing avail segment '{part_str}': {e}.")
    if not windows and avail_str.strip(): return [(session_start_time, session_end_time)]
    return windows if windows else [(session_start_time, session_end_time)]

def generate_csv_string(schedule_data, active_players_per_slot, slot_assignments) -> str:
    output = io.StringIO(); writer = csv.writer(output)
    header = ["Slot Time", "Court", "Team 1 Player 1", "Team 1 Player 2", "Team 2 Player 1", "Team 2 Player 2", "Players Sitting Out Entire Slot", "Winner (if tracked)"]
    writer.writerow(header)
    if not schedule_data: return output.getvalue()
    for slot_time, matches_in_slot_activities in schedule_data:
        time_str = slot_time.strftime('%I:%M %p')
        current_slot_assignment = slot_assignments.get(slot_time, {}); slot_sit_outs_list = sorted(list(current_slot_assignment.get('sitting_out_slot', set())))
        slot_sit_outs_str = ", ".join(slot_sit_outs_list)
        if not matches_in_slot_activities or not any('court' in m for m in matches_in_slot_activities):
            active_for_slot = active_players_per_slot.get(slot_time, [])
            slot_sit_outs_str_to_write = slot_sit_outs_str if slot_sit_outs_str else (", ".join(sorted(active_for_slot)) if active_for_slot else "No active players")
            writer.writerow([time_str, "N/A (No Matches)", "", "", "", "", slot_sit_outs_str_to_write, ""])
            continue
        matches_written_for_slot = False
        for m_info in matches_in_slot_activities:
            if 'court' in m_info:
                winner_info = ""
                if m_info.get('winner') == 'team1': winner_info = f"{m_info['team1'][0]} & {m_info['team1'][1]}"
                elif m_info.get('winner') == 'team2': winner_info = f"{m_info['team2'][0]} & {m_info['team2'][1]}"
                writer.writerow([time_str, m_info['court'], m_info['team1'][0], m_info['team1'][1], m_info['team2'][0], m_info['team2'][1], slot_sit_outs_str, winner_info])
                matches_written_for_slot = True
        if not matches_written_for_slot and slot_sit_outs_list:
             writer.writerow([time_str, "N/A (No Matches)", "", "", "", "", slot_sit_outs_str, ""])
    return output.getvalue()

def display_schedule(session_schedule_data_ref, session_date):
    if not session_schedule_data_ref: st.warning("No schedule data."); return
    st.subheader(f"Pickleball Schedule for {session_date.strftime('%A, %B %d, %Y')}")
    for slot_idx, (slot_time, matches_in_slot_list) in enumerate(session_schedule_data_ref):
        st.markdown(f"--- \n**Slot: {slot_time.strftime('%I:%M %p')}**")
        if not matches_in_slot_list: st.info("No matches/activities."); continue
        match_played_this_slot = False
        for match_idx, m_info in enumerate(matches_in_slot_list):
            if 'court' in m_info:
                match_played_this_slot = True
                                # --- START: New code for visual court display and win tracking ---
                                # --- START: New code for visual court display and win tracking ---
                t1p1_name, t1p2_name = m_info['team1'][0], m_info['team1'][1]
                t2p1_name, t2p2_name = m_info['team2'][0], m_info['team2'][1]

                t1p1e = st.session_state.player_emojis.get(t1p1_name, '😀')
                t1p2e = st.session_state.player_emojis.get(t1p2_name, '😎')
                t2p1e = st.session_state.player_emojis.get(t2p1_name, '🤩')
                t2p2e = st.session_state.player_emojis.get(t2p2_name, '🥳')

                st.markdown(f"**Court {m_info['court']}**")

                # --- Define Court Colors ---
                court_bg_color_team1 = "#A0D2DB"  # Light Blue for Team 1 side
                court_bg_color_team2 = "#98FB98"  # Light Green for Team 2 side
                kitchen_color = "#E0E0E0"       # Light Grey for kitchen/net area accent
                net_color = "#333333"           # Dark Grey for net line
                player_text_color = "#1E1E1E"   # Dark text for good contrast

                court_vis_col1, court_vis_vs, court_vis_col2 = st.columns([0.45, 0.1, 0.45])
                with court_vis_col1:
                    st.markdown(
                        f"""
                        <div style="
                            border: 2px solid {net_color};
                            padding: 10px;
                            border-radius: 8px;
                            text-align: center;
                            background-color: {court_bg_color_team1};
                            min-height: 70px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                            position: relative; /* For kitchen line */
                        ">
                            <div style="
                                position: absolute;
                                top: 0;
                                left: 0; /* Team 1 kitchen on left */
                                width: 30%; /* Kitchen width */
                                height: 100%;
                                background-color: {kitchen_color};
                                border-right: 2px solid {net_color};
                                opacity: 0.5;
                                z-index: 0;
                            "></div>
                            <div style="position: relative; z-index: 1; color: {player_text_color}; font-weight: 500;">
                                {t1p1e} {t1p1_name}<br>
                                {t1p2e} {t1p2_name}
                            </div>
                        </div>
                        """, unsafe_allow_html=True
                    )
                with court_vis_vs:
                    st.markdown(
                        f"""
                        <div style="
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            height: 100%;
                            font-size: 1.5em;
                            font-weight: bold;
                            color: {player_text_color};
                            min-height: 70px;
                            border-left: 2px dashed {net_color}; /* Net representation */
                            border-right: 2px dashed {net_color}; /* Net representation */
                            margin: 0 -2px; /* Overlap columns slightly for continuous net */
                        ">
                        VS
                        </div>
                        """, unsafe_allow_html=True
                    )
                with court_vis_col2:
                    st.markdown(
                        f"""
                        <div style="
                            border: 2px solid {net_color};
                            padding: 10px;
                            border-radius: 8px;
                            text-align: center;
                            background-color: {court_bg_color_team2};
                            min-height: 70px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                            position: relative; /* For kitchen line */
                        ">
                            <div style="
                                position: absolute;
                                top: 0;
                                right: 0; /* Team 2 kitchen on right */
                                width: 30%; /* Kitchen width */
                                height: 100%;
                                background-color: {kitchen_color};
                                border-left: 2px solid {net_color};
                                opacity: 0.5;
                                z-index: 0;
                            "></div>
                            <div style="position: relative; z-index: 1; color: {player_text_color}; font-weight: 500;">
                                {t2p1e} {t2p1_name}<br>
                                {t2p2e} {t2p2_name}
                            </div>
                        </div>
                        """, unsafe_allow_html=True
                    )

                st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

                if st.session_state.track_wins:
                    winner_key = f"winner_s{slot_idx}_m{match_idx}_court{m_info['court']}"
                    options = [
                        "-- Select Winner --",
                        f"Team 1 ({t1p1_name} & {t1p2_name}) Wins",
                        f"Team 2 ({t2p1_name} & {t2p2_name}) Wins"
                    ]
                    current_winner_val = m_info.get('winner')
                    current_idx = 0
                    if current_winner_val == 'team1': current_idx = 1
                    elif current_winner_val == 'team2': current_idx = 2

                    selected_winner_text = st.selectbox(
                        "Set Winner:",
                        options,
                        index=current_idx,
                        key=winner_key,
                        label_visibility="collapsed"
                    )

                    new_winner_val = None
                    if selected_winner_text.startswith("Team 1"): new_winner_val = 'team1'
                    elif selected_winner_text.startswith("Team 2"): new_winner_val = 'team2'

                    if new_winner_val != current_winner_val:
                        m_info['winner'] = new_winner_val
                else:
                    if m_info.get('winner') == 'team1':
                        st.caption(f"   ↳ _Winner: {t1p1_name} & {t1p2_name}_")
                    elif m_info.get('winner') == 'team2':
                        st.caption(f"   ↳ _Winner: {t2p1_name} & {t2p2_name}_")

                st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
                # --- END: New code for visual court display and win tracking ---
        sit_outs = [p for m in matches_in_slot_list if 'sit_out' in m for p in m['sit_out']]
        if sit_outs:
            sit_out_display_strings = [f"{st.session_state.player_emojis.get(p,'')}{p}" for p in sorted(list(set(sit_outs)))]
            st.markdown(f"  - **Sit Out**: {', '.join(sit_out_display_strings)}")
        elif not match_played_this_slot and matches_in_slot_list: st.info("  No matches formed.")

def display_session_stats(stats: dict, all_players_objects: List[Player], substitutions_map: dict):
    if not stats or not all_players_objects: st.info("No session stats."); return
    st.subheader("Player Statistics (This Session)")
    partners_this_session = stats.get('partners_session', set()); opponents_this_session = stats.get('opponents_session', set())
    games_played_this_session = stats.get('games_played_session', defaultdict(int)); sit_out_counts_this_session = stats.get('sit_out_counts_session', defaultdict(int))
    player_partner_counts = defaultdict(Counter); player_opponent_counts = defaultdict(Counter)
    for p1,p2 in partners_this_session: player_partner_counts[p1][p2]+=1; player_partner_counts[p2][p1]+=1
    for p1,p2 in opponents_this_session: player_opponent_counts[p1][p2]+=1; player_opponent_counts[p2][p1]+=1
    player_names_participated = set(p.name for p in all_players_objects)
    for _po, (pi,_pt) in substitutions_map.items(): player_names_participated.add(pi)
    player_objects_dict = {p.name: p for p in all_players_objects}
    for player_name in sorted(list(player_names_participated)):
        player_emoji = st.session_state.player_emojis.get(player_name, '')
        with st.expander(f"{player_emoji} Stats for {player_name}"):
            player_obj = player_objects_dict.get(player_name)
            if player_obj:
                avail_window_strings = [f"{s.strftime('%I:%M%p')}-{e.strftime('%I:%M%p')}" for s, e in player_obj.availability_windows]
                st.markdown(f"  - **Avail**: {'; '.join(avail_window_strings)}")
            else:
                sub_in_time_str = "N/A";
                for p_out_check, (p_in_name_from_sub, time_str) in substitutions_map.items():
                    if p_in_name_from_sub == player_name: sub_in_time_str = f"Sub IN for {p_out_check} @ {time_str}"; break
                st.markdown(f"  - **Avail**: {sub_in_time_str}")
            st.markdown(f"  - **Games**: {games_played_this_session.get(player_name,0)}, **Sat Out**: {sit_out_counts_this_session.get(player_name,0)}")
            my_partners = player_partner_counts[player_name]
            if my_partners:
                partners_str = ", ".join([f"{st.session_state.player_emojis.get(p,'')}{p}(x{c})" for p,c in my_partners.most_common()])
                st.markdown(f"  - **Partners ({len(my_partners)} unique)**: {partners_str}")
                most_p = my_partners.most_common(1)[0]; least_ps_val = my_partners.most_common()[-1][1]
                least_ps = [p for p, c in my_partners.items() if c == least_ps_val]
                least_partner_display_strings = [f"{st.session_state.player_emojis.get(lp,'')}{lp}" for lp in least_ps]
                st.caption(f"    Most: {st.session_state.player_emojis.get(most_p[0],'')}{most_p[0]}(x{most_p[1]}). Least: {', '.join(least_partner_display_strings)}(x{least_ps_val})")
            my_opps = player_opponent_counts[player_name]
            if my_opps:
                opps_str = ", ".join([f"{st.session_state.player_emojis.get(p,'')}{p}(x{c})" for p,c in my_opps.most_common()])
                st.markdown(f"  - **Opponents ({len(my_opps)} unique)**: {opps_str}")
                most_o = my_opps.most_common(1)[0]; least_os_val = my_opps.most_common()[-1][1]
                least_os = [p for p,c in my_opps.items() if c == least_os_val]
                least_opponent_display_strings = [f"{st.session_state.player_emojis.get(lo,'')}{lo}" for lo in least_os]
                st.caption(f"    Most: {st.session_state.player_emojis.get(most_o[0],'')}{most_o[0]}(x{most_o[1]}). Least: {', '.join(least_opponent_display_strings)}(x{least_os_val})")
    st.subheader("Session Fairness Metrics")
    all_games_values = [games_played_this_session.get(name, 0) for name in player_names_participated]
    if all_games_values:
        avg_g=sum(all_games_values)/len(all_games_values); min_g=min(all_games_values); max_g=max(all_games_values); std_g_str="N/A"
        if len(all_games_values)>1: var_g=sum([(x-avg_g)**2 for x in all_games_values])/len(all_games_values); std_g=math.sqrt(var_g); std_g_str=f"{std_g:.2f}"
        st.markdown(f"- Games Dist: Avg={avg_g:.1f}(StdDev={std_g_str}), Min={min_g}, Max={max_g}")
    all_sit_out_values = [sit_out_counts_this_session.get(name, 0) for name in player_names_participated if games_played_this_session.get(name,0) > 0 or sit_out_counts_this_session.get(name,0) > 0]
    if all_sit_out_values:
        avg_s=sum(all_sit_out_values)/len(all_sit_out_values) if all_sit_out_values else 0; min_s=min(all_sit_out_values) if all_sit_out_values else 0; max_s=max(all_sit_out_values) if all_sit_out_values else 0
        st.markdown(f"- Slots Sat Out: Avg={avg_s:.1f}, Min={min_s}, Max={max_s}")
    st.info("**Why some players play more/less:**\n- Availability, Odd Player Counts, Substitutions, Constraints, Matchmaking Strategy (if active).")

# --- Draft & History Load Functions ---
def save_draft():
    draft_data = {
        'session_date': st.session_state.session_date.isoformat(), 'start_time_str': st.session_state.start_time_str,
        'num_games': st.session_state.num_games, 'game_interval': st.session_state.game_interval, 'seed_str': st.session_state.seed_str,
        'players_data': st.session_state.players_data, 'player_emojis': st.session_state.player_emojis,
        'dnp_pairs_list': st.session_state.dnp_pairs_list, 'dno_pairs_list': st.session_state.dno_pairs_list,
        'avoid_pairs_list': st.session_state.avoid_pairs_list, 'substitutions_list': st.session_state.substitutions_list,
        'track_wins': st.session_state.track_wins, 'matchmaking_strategy': st.session_state.matchmaking_strategy
    }
    try:
        with DRAFT_FILE.open('w') as f: json.dump(draft_data, f, indent=2)
        st.success("Draft saved!")
    except Exception as e: st.error(f"Draft save error: {e}")

def load_draft():
    if not DRAFT_FILE.exists(): st.warning("No draft file."); return
    try:
        with DRAFT_FILE.open('r') as f: draft_data = json.load(f)
        st.session_state.session_date = datetime.date.fromisoformat(draft_data.get('session_date', datetime.date.today().isoformat()))
        st.session_state.start_time_str = draft_data.get('start_time_str', "2:00 PM")
        st.session_state.num_games = draft_data.get('num_games', 8); st.session_state.game_interval = draft_data.get('game_interval', 15)
        st.session_state.seed_str = draft_data.get('seed_str', "")
        loaded_players = []
        for p_data in draft_data.get('players_data',[]):
            if 'id' not in p_data: p_data['id']=str(uuid.uuid4())
            if 'emoji' not in p_data: p_data['emoji']=get_next_emoji()
            loaded_players.append(p_data)
        st.session_state.players_data = loaded_players
        st.session_state.player_emojis = draft_data.get('player_emojis',{})
        st.session_state.dnp_pairs_list = [tuple(p) for p in draft_data.get('dnp_pairs_list',[])]
        st.session_state.dno_pairs_list = [tuple(p) for p in draft_data.get('dno_pairs_list',[])]
        st.session_state.avoid_pairs_list = [tuple(p) for p in draft_data.get('avoid_pairs_list',[])]
        st.session_state.substitutions_list = draft_data.get('substitutions_list',[])
        st.session_state.track_wins = draft_data.get('track_wins', False)
        st.session_state.matchmaking_strategy = draft_data.get('matchmaking_strategy', "Standard Rotation")
        st.success("Draft loaded!"); st.rerun()
    except Exception as e: st.error(f"Draft load error: {e}")

def load_history_week_inputs(week_data: Dict[str, Any]):
    try:
        st.session_state.session_date = datetime.date.fromisoformat(week_data.get('date', datetime.date.today().isoformat()))
        loaded_players = []
        for p_hist in week_data.get('players_original_roster', []):
            player_id = str(uuid.uuid4()); player_name = p_hist.get('name', 'Unknown')
            avail_str = "; ".join([f"{s}-{e}" for s,e in p_hist.get('availability_windows',[])])
            emoji = p_hist.get('emoji') or st.session_state.player_emojis.get(player_name) or get_next_emoji()
            loaded_players.append({'id':player_id, 'name':player_name, 'avail_str':avail_str, 'emoji':emoji})
            st.session_state.player_emojis[player_name] = emoji
        st.session_state.players_data = loaded_players
        st.session_state.dnp_pairs_list = [tuple(sorted(p)) for p in week_data.get('incompatibles_used', week_data.get('incompatibles',[]))] # Prioritize 'used'
        st.session_state.dno_pairs_list = [tuple(sorted(p)) for p in week_data.get('dno_pairs_used', week_data.get('dno_pairs',[]))]
        st.session_state.avoid_pairs_list = [] # Avoids are derived, not usually stored directly this way
        st.session_state.substitutions_list = []
        for p_out, (p_in, time_str) in week_data.get('substitutions', {}).items():
            st.session_state.substitutions_list.append({'out':p_out, 'in':p_in, 'time_str':time_str})
            if p_in not in st.session_state.player_emojis: st.session_state.player_emojis[p_in] = get_next_emoji()
        st.session_state.track_wins = week_data.get('tracked_wins_this_session', False)
        st.success(f"Inputs from session on {week_data.get('date')} loaded!"); st.rerun()
    except Exception as e: st.error(f"Failed to load history week: {e}")


# --- Main Streamlit App Structure ---
def initialize_session_state():
    if 'player_emojis' not in st.session_state: st.session_state.player_emojis = {}
    if 'players_data' not in st.session_state: st.session_state.players_data = []
    if 'session_date' not in st.session_state: st.session_state.session_date = datetime.date.today()
    if 'start_time_str' not in st.session_state: st.session_state.start_time_str = "2:00 PM"
    if 'num_games' not in st.session_state: st.session_state.num_games = 8
    if 'game_interval' not in st.session_state: st.session_state.game_interval = 15
    if 'seed_str' not in st.session_state: st.session_state.seed_str = ""
    if 'effective_session_start_time' not in st.session_state: st.session_state.effective_session_start_time = datetime.time(14,0)
    if 'effective_session_end_time' not in st.session_state: st.session_state.effective_session_end_time = datetime.time(16,0)
    if 'track_wins' not in st.session_state: st.session_state.track_wins = False
    if 'matchmaking_strategy' not in st.session_state: st.session_state.matchmaking_strategy = "Standard Rotation"
    if 'dnp_pairs_list' not in st.session_state: st.session_state.dnp_pairs_list = []
    if 'dno_pairs_list' not in st.session_state: st.session_state.dno_pairs_list = []
    if 'avoid_pairs_list' not in st.session_state: st.session_state.avoid_pairs_list = []
    if 'substitutions_list' not in st.session_state: st.session_state.substitutions_list = []
    if 'generated_schedule_data' not in st.session_state: st.session_state.generated_schedule_data = None
    if 'session_stats' not in st.session_state: st.session_state.session_stats = {}
    if 'processed_players_for_stats' not in st.session_state: st.session_state.processed_players_for_stats = []
    if 'processed_subs_for_stats' not in st.session_state: st.session_state.processed_subs_for_stats = {}
    if 'current_history_file_path_str' not in st.session_state: st.session_state.current_history_file_path_str = str(DEFAULT_HISTORY_FILE.resolve())
    if 'history_manager_instance' not in st.session_state:
        try: st.session_state.history_manager_instance = HistoryManager(Path(st.session_state.current_history_file_path_str))
        except Exception: st.session_state.current_history_file_path_str = str(DEFAULT_HISTORY_FILE.resolve()); st.session_state.history_manager_instance = HistoryManager(DEFAULT_HISTORY_FILE)
    if 'history_data' not in st.session_state: st.session_state.history_data = st.session_state.history_manager_instance.load()
    if 'selected_history_week_index' not in st.session_state: st.session_state.selected_history_week_index = None
    if 'draft_load_attempted' not in st.session_state: st.session_state.draft_load_attempted = False

def main_streamlit_app():
    st.set_page_config(page_title="Pickleball Scheduler Pro+", layout="wide", initial_sidebar_state="expanded")
    st.title("🏓 Pickleball Scheduler Pro+")
    initialize_session_state()
    hm = st.session_state.history_manager_instance

    with st.sidebar: # Sidebar UI
        st.header("⚙️ Session Setup")
        st.session_state.session_date = st.date_input("Session Date", st.session_state.session_date)
        st.session_state.start_time_str = st.text_input("Session Start Time (e.g., '2 PM', '14:00')", st.session_state.start_time_str)
        parsed_session_start_time = UI._parse_time_flexible(st.session_state.start_time_str)
        if not parsed_session_start_time: st.error("Invalid session start time format.")
        else: st.caption(f"Parsed as: {parsed_session_start_time.strftime('%I:%M %p')}")
        st.session_state.num_games = st.number_input("Number of Game Slots", min_value=1, value=st.session_state.num_games, step=1)
        st.session_state.game_interval = st.number_input("Game Interval (minutes)", min_value=5, value=st.session_state.game_interval, step=5)
        st.session_state.seed_str = st.text_input("Random Seed (optional)", st.session_state.seed_str)
        
        st.divider()
        st.subheader("🏆 Win Tracking & Matchmaking")
        st.session_state.track_wins = st.checkbox("Track Game Winners?", value=st.session_state.track_wins, key="track_wins_sidebar_cb")
        if st.session_state.track_wins:
            st.session_state.matchmaking_strategy = st.selectbox("Matchmaking Strategy:",
                ["Standard Rotation", "Balance by Win Rate (Experimental)"],
                index=["Standard Rotation", "Balance by Win Rate (Experimental)"].index(st.session_state.matchmaking_strategy),
                key="matchmaking_strategy_sidebar_select")
            st.caption("'Balance by Win Rate' is conceptual.")
        
        time_slots_for_session = []
        if parsed_session_start_time:
            time_slots_for_session = Scheduler.time_slots(parsed_session_start_time, st.session_state.num_games, st.session_state.game_interval)
            if time_slots_for_session:
                st.session_state.effective_session_start_time = time_slots_for_session[0]
                st.session_state.effective_session_end_time = (datetime.datetime.combine(datetime.date.min, time_slots_for_session[-1]) + datetime.timedelta(minutes=st.session_state.game_interval)).time()
                st.markdown(f"**Effective Session:** {st.session_state.effective_session_start_time.strftime('%I:%M%p')} - {st.session_state.effective_session_end_time.strftime('%I:%M%p')}")

        st.divider(); st.header("💾 Draft & History Files")
        col_ds, col_dl = st.columns(2)
        with col_ds:
            if st.button("Save Draft", use_container_width=True, key="save_draft_sidebar_btn_main"): save_draft()
        with col_dl:
            if st.button("Load Draft", use_container_width=True, key="load_draft_sidebar_btn_main"): load_draft()
        st.subheader("History File Location")
        new_history_path_input = st.text_input("Current History File:", value=st.session_state.current_history_file_path_str, key="history_file_path_widget_sidebar_main")
        if st.button("🔄 Use This History File Path", key="switch_history_file_button_sidebar_main"):
            new_path = Path(new_history_path_input.strip())
            if not new_path.name or not new_path.suffix.lower() == ".json": st.warning("Invalid .json file path.")
            else:
                try:
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    st.session_state.current_history_file_path_str = str(new_path.resolve())
                    st.session_state.history_manager_instance = HistoryManager(new_path)
                    st.session_state.history_data = st.session_state.history_manager_instance.load()
                    st.success(f"Using history: {new_path.name}"); st.rerun()
                except Exception as e: st.error(f"Error setting history path: {e}")
        
        st.divider()
        if st.button("✨ Generate Schedule", type="primary", use_container_width=True, disabled=(not parsed_session_start_time), key="generate_schedule_sidebar_btn_main"):
            if not parsed_session_start_time: st.error("Cannot generate: Invalid session start time."); st.stop()
            if not st.session_state.players_data: st.error("No players added."); st.stop()
            
            all_session_players_objects: List[Player] = []
            for p_data in st.session_state.players_data:
                name = p_data['name'].strip()
                if not name: continue
                avail_windows = parse_availability_string_st(p_data['avail_str'], st.session_state.effective_session_start_time, st.session_state.effective_session_end_time)
                all_session_players_objects.append(Player(name, avail_windows))
                if name not in st.session_state.player_emojis: st.session_state.player_emojis[name] = p_data.get('emoji', get_next_emoji())
            if not all_session_players_objects: st.error("No valid players."); st.stop()
            st.session_state.processed_players_for_stats = all_session_players_objects

            final_dnp = list(set(tuple(sorted(p)) for p in st.session_state.dnp_pairs_list + st.session_state.avoid_pairs_list))
            final_dno = list(set(tuple(sorted(p)) for p in st.session_state.dno_pairs_list + st.session_state.avoid_pairs_list))
            sub_map = {sub['out']:(sub['in'], sub['time_str']) for sub in st.session_state.substitutions_list}
            st.session_state.processed_subs_for_stats = sub_map
            
            historical_win_rates_data = None
            if st.session_state.track_wins and st.session_state.matchmaking_strategy == "Balance by Win Rate (Experimental)":
                # Calculate historical_win_rates_data from st.session_state.history_data
                # This is a complex step, for now it's a placeholder.
                # It would involve iterating through all weeks, all matches with winners,
                # and calculating games_played and games_won for each player.
                st.info("Conceptual win rate balancing active.")
                pass 

            rng = random.Random()
            if st.session_state.seed_str:
                try: seed_val = int(st.session_state.seed_str); rng.seed(seed_val)
                except ValueError: rng.seed(st.session_state.seed_str)

            scheduler = Scheduler(Scheduler.determine_courts(len(all_session_players_objects)), rng, final_dnp, final_dno, st.session_state.game_interval, st.session_state.matchmaking_strategy, historical_win_rates_data)
            with st.spinner("Generating..."):
                try:
                    gen_schedule, partners_sess, opp_sess, games_played_sess, sit_out_counts_sess, active_csv, slot_assign_csv = scheduler.schedule_day(time_slots_for_session, all_session_players_objects, sub_map)
                    for _s_t, s_matches_list in gen_schedule:
                        for m_dict in s_matches_list:
                            if 'court' in m_dict and 'winner' not in m_dict: m_dict['winner'] = None
                    st.session_state.generated_schedule_data = gen_schedule
                    st.session_state.session_stats = {
                        'partners_session': partners_sess, 'opponents_session': opp_sess, 'games_played_session': games_played_sess,
                        'sit_out_counts_session': sit_out_counts_sess, 'active_players_for_csv': active_csv, 'slot_assignments_for_csv': slot_assign_csv
                    }
                    st.success("Schedule Generated!")
                    week_entry = {
                         'date': str(st.session_state.session_date),
                         'players_original_roster': [{'name': p.name, 'availability_windows': [[w_start.strftime("%I:%M%p"), w_end.strftime("%I:%M%p")] for w_start, w_end in p.availability_windows], 'emoji': st.session_state.player_emojis.get(p.name, '')} for p in all_session_players_objects],
                         'substitutions': sub_map, 'partners': [list(p) for p in partners_sess], 'opponents': [list(o) for o in opp_sess],
                         'games_played': dict(games_played_sess), 'sit_out_counts': dict(sit_out_counts_sess),
                         'incompatibles_used': final_dnp, 'dno_pairs_used': final_dno,
                         'tracked_wins_this_session': st.session_state.track_wins,
                         'match_results': [{'slot_time_iso': slot_t.isoformat(), 'matches': [{'court': m.get('court'), 'team1': m.get('team1'), 'team2': m.get('team2'), 'winner': m.get('winner')} for m in m_list if 'court' in m]} for slot_t, m_list in gen_schedule] if st.session_state.track_wins else []
                    }
                    st.session_state.history_data['weeks'].append(week_entry)
                    st.session_state.history_data['incompatibles'] = final_dnp # Update global DNP based on what was used.
                    st.session_state.history_data['dno_pairs'] = final_dno # Update global DNO
                    # Avoid pairs are not stored globally as they are derived for use
                    st.session_state.history_manager_instance.save(st.session_state.history_data)
                    st.toast("Session saved to history.")
                except Exception as e: st.error(f"Error generating: {e}"); logger.exception("Gen error")

    tab_players, tab_constraints, tab_history, tab_schedule, tab_stats, tab_overall_stats = st.tabs(["👥 Players", "🚫 Constraints", "📜 History", "🗓️ Schedule", "📊 Session Stats", "📈 Overall Stats"])

    with tab_players: # ... (Player input as before)
        st.header("Player Roster & Availability")
        col_add, col_info = st.columns([1,3])
        with col_add:
            if st.button("➕ Add Player", use_container_width=True, key="add_player_main_button_tab"):
                new_id = str(uuid.uuid4()); new_player_name = f"Player {len(st.session_state.players_data) + 1}"
                new_emoji = get_next_emoji()
                st.session_state.players_data.append({'id': new_id, 'name': new_player_name, 'avail_str': '', 'emoji': new_emoji})
                st.session_state.player_emojis[new_player_name] = new_emoji
                st.rerun()
        with col_info: st.caption(f"Session: {st.session_state.effective_session_start_time.strftime('%I:%M%p')} - {st.session_state.effective_session_end_time.strftime('%I:%M%p')}.")
        if not st.session_state.players_data: st.info("No players added.")
        for i, player_entry in enumerate(st.session_state.players_data):
            player_id = player_entry['id']; player_emoji = player_entry.get('emoji', '')
            with st.expander(f"{player_emoji} Player {i+1}: {player_entry.get('name', '') or '[No Name Yet]'}", expanded=True):
                name_col, avail_col, remove_col = st.columns([0.3, 0.6, 0.1])
                with name_col: player_entry['name'] = st.text_input("Name", player_entry['name'], key=f"p_name_{player_id}", label_visibility="collapsed", placeholder="Player Name")
                with avail_col: player_entry['avail_str'] = st.text_input("Availability", player_entry['avail_str'], key=f"p_avail_{player_id}", label_visibility="collapsed", placeholder="e.g., 2PM-3PM")
                with remove_col:
                    if st.button("➖", key=f"remove_p_{player_id}", help="Remove Player"):
                        if player_entry.get('name') in st.session_state.player_emojis: del st.session_state.player_emojis[player_entry['name']]
                        st.session_state.players_data = [p for p in st.session_state.players_data if p['id'] != player_id]; st.rerun()
                if not player_entry['name'].strip(): st.caption(":warning: Name is empty.")

    with tab_constraints: # Updated Constraint Input
        player_names_for_selection = sorted([p['name'] for p in st.session_state.players_data if p['name'].strip()])
        if not player_names_for_selection: st.warning("Add players in the 'Players' tab first to set constraints."); st.stop()
        st.header("🚫 Constraint Management")
        constraint_type = st.radio("Select constraint type to add:", ("Don't Play With (DNP)", "Don't Play Against (DNO)", "Avoid Each Other (DNP & DNO)"), horizontal=True, key="constraint_type_selector_main")
        col_c1, col_c2 = st.columns(2)
        with col_c1: p1_constraint = st.selectbox("Player 1", [""] + player_names_for_selection, key="c_p1_main", index=0)
        with col_c2: p2_constraint_options = [""] + [p for p in player_names_for_selection if p != p1_constraint]; p2_constraint = st.selectbox("Player 2", p2_constraint_options, key="c_p2_main", index=0)
        if st.button("➕ Add Constraint Pair", disabled=(not p1_constraint or not p2_constraint), key="add_constraint_btn_main"):
            if p1_constraint and p2_constraint:
                pair = tuple(sorted((p1_constraint, p2_constraint)))
                if constraint_type == "Don't Play With (DNP)":
                    if pair not in st.session_state.dnp_pairs_list: st.session_state.dnp_pairs_list.append(pair); st.toast(f"Added DNP: {pair[0]}-{pair[1]}")
                    else: st.toast("DNP pair already exists.")
                elif constraint_type == "Don't Play Against (DNO)":
                    if pair not in st.session_state.dno_pairs_list: st.session_state.dno_pairs_list.append(pair); st.toast(f"Added DNO: {pair[0]}-{pair[1]}")
                    else: st.toast("DNO pair already exists.")
                elif constraint_type == "Avoid Each Other (DNP & DNO)":
                    if pair not in st.session_state.avoid_pairs_list: st.session_state.avoid_pairs_list.append(pair); st.toast(f"Added Avoid: {pair[0]}-{pair[1]}")
                    else: st.toast("Avoid pair already exists.")
                st.rerun()
        for c_list_name, c_list_title in [('dnp_pairs_list', "Current DNP Pairs:"), ('dno_pairs_list', "Current DNO Pairs:"), ('avoid_pairs_list', "Current 'Avoid Each Other' Pairs:")]:
            st.subheader(c_list_title)
            current_list = getattr(st.session_state, c_list_name)
            if not current_list: st.caption("None added.")
            for idx, pair_to_disp in enumerate(current_list):
                disp_col, rem_col = st.columns([0.9,0.1])
                with disp_col: st.markdown(f"- {st.session_state.player_emojis.get(pair_to_disp[0],'')}{pair_to_disp[0]} & {st.session_state.player_emojis.get(pair_to_disp[1],'')}{pair_to_disp[1]}")
                with rem_col:
                    if st.button("✖️", key=f"remove_{c_list_name}_{idx}_btn"): current_list.pop(idx); st.rerun()
        
        st.header("♻️ Substitutions") # Updated Player In
        col_sub_out, col_sub_in_select, col_sub_in_text, col_sub_time = st.columns([0.25, 0.25, 0.25, 0.25])
        with col_sub_out: selected_p_out_sub = st.selectbox("Player Out", [""] + player_names_for_selection, key="sub_p_out_main_select", index=0)
        player_names_plus_new = ["-- Type New Player Name --"] + player_names_for_selection
        with col_sub_in_select: selected_p_in_option = st.selectbox("Player In Option", player_names_plus_new, key="sub_p_in_select_main_select", index=0)
        p_in_sub_final_name = ""
        if selected_p_in_option == "-- Type New Player Name --":
            with col_sub_in_text: p_in_sub_final_name = st.text_input("New Player In Name:", key="sub_p_in_text_main_input").strip()
        else: p_in_sub_final_name = selected_p_in_option
        with col_sub_time: selected_time_sub_str = st.text_input("Time of Swap", key="sub_time_main_input", placeholder="e.g., 3 PM")
        if st.button("➕ Add Substitution", disabled=(not selected_p_out_sub or not p_in_sub_final_name or not selected_time_sub_str), key="add_sub_btn_main_add"):
            parsed_sub_time = UI._parse_time_flexible(selected_time_sub_str.strip())
            if not parsed_sub_time: st.warning("Invalid sub time.")
            elif selected_p_out_sub == p_in_sub_final_name: st.warning("Sub out/in same player.")
            else:
                st.session_state.substitutions_list.append({'out':selected_p_out_sub, 'in':p_in_sub_final_name, 'time_str':parsed_sub_time.strftime("%I:%M%p")})
                if p_in_sub_final_name not in st.session_state.player_emojis: st.session_state.player_emojis[p_in_sub_final_name] = get_next_emoji()
                st.rerun()
        st.subheader("Current Substitutions:")
        if not st.session_state.substitutions_list: st.caption("None added.")
        for idx, sub_item in enumerate(st.session_state.substitutions_list):
            disp_sub_col, rem_sub_col = st.columns([0.9, 0.1])
            with disp_sub_col: st.markdown(f"- {st.session_state.player_emojis.get(sub_item['out'],'')}{sub_item['out']} ➔ {st.session_state.player_emojis.get(sub_item['in'],'')}{sub_item['in']} @ {sub_item['time_str']}")
            with rem_sub_col:
                if st.button("✖️", key=f"remove_sub_main_btn_{idx}"): st.session_state.substitutions_list.pop(idx); st.rerun()

    with tab_history: # Updated
        st.header("📜 Session History & Management")
        st.subheader("⚠️ Clear Current History File")
        st.caption(f"Current file: {st.session_state.current_history_file_path_str}")
        confirm_clear = st.checkbox("Yes, I want to clear the content of the current history file.", key="confirm_clear_hist_check_main")
        if st.button("🗑️ Clear Current History", disabled=not confirm_clear, type="secondary", key="clear_history_button_main"):
            if confirm_clear:
                try:
                    st.session_state.history_data = {'weeks': [], 'incompatibles': [], 'dno_pairs': [], 'avoid_pairs': []}
                    st.session_state.history_manager_instance.save(st.session_state.history_data) 
                    st.success(f"History file content cleared."); st.rerun()
                except Exception as e: st.error(f"Error clearing history: {e}")
        st.divider()
        st.subheader("Load Constraints from Global History")
        hist = st.session_state.history_data
        if hist.get('incompatibles'):
            if st.button(f"Load {len(hist['incompatibles'])} Global DNP(s)", key="load_hist_global_dnp_main"):
                st.session_state.dnp_pairs_list = [tuple(p) for p in hist['incompatibles']]; st.rerun()
        if hist.get('dno_pairs'):
            if st.button(f"Load {len(hist['dno_pairs'])} Global DNO(s)", key="load_hist_global_dno_main"):
                st.session_state.dno_pairs_list = [tuple(p) for p in hist['dno_pairs']]; st.rerun()
        if hist.get('avoid_pairs'):
            if st.button(f"Load {len(hist['avoid_pairs'])} Global Avoid(s)", key="load_hist_global_avoid_main"):
                st.session_state.avoid_pairs_list = [tuple(p) for p in hist['avoid_pairs']]; st.rerun()

        st.subheader("View & Load Full Inputs from Past Sessions")
        if not hist['weeks']: st.info("No past sessions.")
        else:
            options = ["Select session..."] + [f"{idx}: {w.get('date', 'N/A')} ({len(w.get('players_original_roster',[]))} players)" for idx, w in enumerate(hist['weeks'])]
            selected_opt_text = st.selectbox("Select past session:", options, index=0, key="hist_week_sel_main")
            if selected_opt_text != "Select session...":
                idx_to_load = int(selected_opt_text.split(":")[0])
                if st.button(f"Load Inputs from Session {hist['weeks'][idx_to_load].get('date')}", key=f"load_hist_week_inputs_btn_{idx_to_load}"):
                    load_history_week_inputs(hist['weeks'][idx_to_load])
                with st.expander("View Raw Week Data"): st.json(hist['weeks'][idx_to_load])

    with tab_schedule: # Updated
        st.header("🗓️ Generated Schedule")
        if st.session_state.generated_schedule_data:
            display_schedule(st.session_state.generated_schedule_data, st.session_state.session_date)
            csv_str = generate_csv_string(st.session_state.generated_schedule_data, st.session_state.session_stats.get('active_players_for_csv',{}), st.session_state.session_stats.get('slot_assignments_for_csv',{}))
            st.download_button("📥 Download CSV", csv_str, f"schedule_{st.session_state.session_date.strftime('%Y%m%d')}.csv", "text/csv", key="download_csv_schedule_tab")
        else: st.info("Generate a schedule first.")

    with tab_stats: # Updated
        st.header("📊 Session Statistics")
        if st.session_state.session_stats and st.session_state.processed_players_for_stats:
            display_session_stats(st.session_state.session_stats, st.session_state.processed_players_for_stats, st.session_state.processed_subs_for_stats)
        else: st.info("Generate a schedule first.")

    with tab_overall_stats: # Updated
        st.header("📈 Overall Statistics Across All Sessions")
        if not st.session_state.history_data['weeks']: st.info("No history for overall stats.")
        else:
            overall_games = defaultdict(int); overall_sitouts = defaultdict(int)
            overall_partners = defaultdict(set); overall_opponents = defaultdict(set)
            overall_partner_counts = defaultdict(Counter); overall_opponent_counts = defaultdict(Counter)
            all_hist_players = set()
            for week in st.session_state.history_data['weeks']:
                for p_roster in week.get('players_original_roster',[]): all_hist_players.add(p_roster['name'])
                for p_out, (p_in, _t) in week.get('substitutions',{}).items(): all_hist_players.add(p_in); all_hist_players.add(p_out)
                for p_name, count in week.get('games_played',{}).items(): overall_games[p_name]+=count
                for p_name, count in week.get('sit_out_counts',{}).items(): overall_sitouts[p_name]+=count
                for p1,p2 in week.get('partners',[]):
                    p1_s,p2_s = tuple(sorted((p1,p2)))
                    if p1_s != p2_s: overall_partners[p1_s].add(p2_s); overall_partners[p2_s].add(p1_s); overall_partner_counts[p1_s][p2_s]+=1; overall_partner_counts[p2_s][p1_s]+=1
                for p1,p2 in week.get('opponents',[]):
                    p1_s,p2_s = tuple(sorted((p1,p2)))
                    if p1_s != p2_s: overall_opponents[p1_s].add(p2_s); overall_opponents[p2_s].add(p1_s); overall_opponent_counts[p1_s][p2_s]+=1; overall_opponent_counts[p2_s][p1_s]+=1
            if not all_hist_players: st.info("No player data in history."); st.stop()
            for p_name in sorted(list(all_hist_players)):
                player_emoji = st.session_state.player_emojis.get(p_name, get_next_emoji())
                st.session_state.player_emojis[p_name] = player_emoji
                with st.expander(f"{player_emoji} Overall Stats for {p_name}"):
                    st.markdown(f"- Total Games: {overall_games[p_name]}, Total Sitouts: {overall_sitouts[p_name]}")
                    if overall_partner_counts[p_name]:
                        most_p = overall_partner_counts[p_name].most_common(1)[0]; least_ps_val = overall_partner_counts[p_name].most_common()[-1][1]
                        least_ps = [p for p,c in overall_partner_counts[p_name].items() if c == least_ps_val]
                        least_partner_disp_strs = [f"{st.session_state.player_emojis.get(lp,'')}{lp}" for lp in least_ps]
                        st.markdown(f"  - Most Partnered: {st.session_state.player_emojis.get(most_p[0],'')}{most_p[0]} (x{most_p[1]})")
                        st.markdown(f"  - Least Partnered: {', '.join(least_partner_disp_strs)} (x{least_ps_val})")
                    if overall_opponent_counts[p_name]:
                        most_o = overall_opponent_counts[p_name].most_common(1)[0]; least_os_val = overall_opponent_counts[p_name].most_common()[-1][1]
                        least_os = [p for p,c in overall_opponent_counts[p_name].items() if c == least_os_val]
                        least_opponent_disp_strs = [f"{st.session_state.player_emojis.get(lo,'')}{lo}" for lo in least_os]
                        st.markdown(f"  - Most Opposed: {st.session_state.player_emojis.get(most_o[0],'')}{most_o[0]} (x{most_o[1]})")
                        st.markdown(f"  - Least Opposed: {', '.join(least_opponent_disp_strs)} (x{least_os_val})")
                            # --- START: New code for Overall Fairness Stats ---
            st.subheader("📈 Overall Play Balance & Fairness Metrics")

            all_overall_games_values = [count for count in overall_games.values() if count > 0] # Filter out players with 0 games
            if all_overall_games_values:
                avg_g_overall = sum(all_overall_games_values) / len(all_overall_games_values)
                min_g_overall = min(all_overall_games_values)
                max_g_overall = max(all_overall_games_values)
                std_g_overall_str = "N/A"
                if len(all_overall_games_values) > 1:
                    var_g_overall = sum([(x - avg_g_overall) ** 2 for x in all_overall_games_values]) / len(all_overall_games_values)
                    std_g_overall = math.sqrt(var_g_overall)
                    std_g_overall_str = f"{std_g_overall:.2f}"
                st.markdown(f"- **Games Played Distribution (Overall):**")
                st.markdown(f"  - Average: {avg_g_overall:.1f} games per player session attended")
                st.markdown(f"  - Min: {min_g_overall}, Max: {max_g_overall}")
                st.markdown(f"  - Standard Deviation: {std_g_overall_str}")
            else:
                st.markdown("- No overall game data to calculate fairness metrics.")

            
            # A bit complex to get names for sitouts, let's simplify to just values if overall_games ensures participation
            active_player_names_for_sitouts = [name for name, games in overall_games.items() if games > 0 or overall_sitouts.get(name, 0) > 0]
            all_overall_sitout_values_filtered = [overall_sitouts.get(name,0) for name in active_player_names_for_sitouts]

            if all_overall_sitout_values_filtered:
                avg_s_overall = sum(all_overall_sitout_values_filtered) / len(all_overall_sitout_values_filtered) if all_overall_sitout_values_filtered else 0
                min_s_overall = min(all_overall_sitout_values_filtered) if all_overall_sitout_values_filtered else 0
                max_s_overall = max(all_overall_sitout_values_filtered) if all_overall_sitout_values_filtered else 0
                st.markdown(f"- **Slots Sat Out Distribution (Overall):**")
                st.markdown(f"  - Average: {avg_s_overall:.1f} sit-outs per player session attended")
                st.markdown(f"  - Min: {min_s_overall}, Max: {max_s_overall}")
            else:
                st.markdown("- No overall sit-out data to calculate fairness metrics.")
            
            st.caption("**Note on Overall Fairness:** These metrics reflect play balance across all recorded sessions. Variations are expected due to differing attendance, session lengths, and player availability over time.")
            # --- END: New code for Overall Fairness Stats ---
            st.info("**Overall Play Balance:** Factors: Attendance, availability, group size fluctuations.")

if __name__ == '__main__':
    main_streamlit_app()