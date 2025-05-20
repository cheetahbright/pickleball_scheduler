import datetime
from typing import List, Dict, Set, Tuple, Optional
import json
import os
import random
from collections import defaultdict

class Player:
    def __init__(self, name: str, available_from: datetime.time, available_to: datetime.time, emoji: str = None):
        self.name = name
        self.available_from = available_from
        self.available_to = available_to
        self.emoji = emoji
        self.do_not_pair: Set[str] = set()

class Game:
    def __init__(self, court: int, team1: List[str], team2: List[str]):
        self.court = court
        self.team1 = team1
        self.team2 = team2

class Scheduler:
    def __init__(self, players: List[Player], num_courts: int, history: Optional[Dict]=None, start_time: str = "14:00", is_1v1: bool = False):
        self.players = players
        self.num_courts = num_courts
        self.history = history if history else {}
        self.games_per_day = 8
        # Use the session start time provided in args.start
        self.start_time = datetime.datetime.strptime(start_time, "%H:%M").time()
        self.end_time = datetime.time(16, 0)
        self.game_length = 15  # minutes
        self.is_1v1 = is_1v1
        # Remove persistent history for cross-session fairness
        # self.cross_session_history = self.load_history()
        self.cross_session_history = {'partners': {}, 'opponents': {}}  # No cross-session history

    def generate_schedule(self, constraints: Optional[List[Tuple[str, str, str]]] = None, substitutions: Optional[List[Tuple[str, str, int]]] = None):
        all_initial_player_names = [p.name for p in self.players]
        if not all_initial_player_names: # Handle case with no players initially
            return []
        num_initial_players = len(all_initial_player_names)
        schedule = []

        # --- 1v1/2v2 logic ---
        players_per_court = 2 if self.is_1v1 else 4

        if num_initial_players < players_per_court and not substitutions: # Not enough for a single game initially
            return []

        # Initialize counts for all players who were present at the start
        play_counts = {name: 0 for name in all_initial_player_names}
        sit_counts = {name: 0 for name in all_initial_player_names}
        
        session_partner_counts = defaultdict(lambda: defaultdict(int))
        session_opponent_counts = defaultdict(lambda: defaultdict(int))

        # Parse constraints once at the beginning
        parsed_dnp_pairs = set()
        parsed_dno_pairs = set()
        if constraints:
            for ctype, p1_name, p2_name in constraints:
                if p1_name not in all_initial_player_names or p2_name not in all_initial_player_names:
                    # Constraint involves a player not in the initial list, skip or warn
                    continue 
                pair = frozenset([p1_name, p2_name])
                if ctype == "Don't Play With (DNP)":
                    parsed_dnp_pairs.add(pair)
                elif ctype == "Don't Play Against (DNO)":
                    parsed_dno_pairs.add(pair)
                elif ctype == "Avoid Each Other":
                    parsed_dnp_pairs.add(pair)
                    parsed_dno_pairs.add(pair)

        current_player_pool = set(all_initial_player_names)

        for round_num in range(self.games_per_day):
            actual_round_num = round_num + 1 # 1-based for substitutions
            round_info = {'games': [], 'sit_out': []}

            # Apply substitutions for the current round
            if substitutions:
                for out_name, in_name, sub_at_round in substitutions: # Iterate with out_name, in_name, sub_at_round
                    if actual_round_num == sub_at_round:
                        # Handle player leaving
                        if out_name and out_name in current_player_pool:
                            current_player_pool.remove(out_name)
                        
                        # Handle player joining
                        # Ensure 'in_name' is a known player from the initial roster.
                        # Adding to a set is idempotent, so if already in, no change.
                        if in_name and in_name in all_initial_player_names: 
                            current_player_pool.add(in_name)

            # Calculate the start and end time for the current round
            round_start_time = (datetime.datetime.combine(datetime.date.today(), self.start_time) + datetime.timedelta(minutes=round_num * self.game_length)).time()
            round_end_time = (datetime.datetime.combine(datetime.date.today(), round_start_time) + datetime.timedelta(minutes=self.game_length)).time()

            # Filter players based on availability
            active_players_this_round = [p.name for p in self.players if p.name in current_player_pool and p.available_from <= round_start_time < p.available_to]
            num_active_players = len(active_players_this_round)

            # Calculate number of games possible this round
            num_possible_games_by_players = num_active_players // players_per_court
            num_games_this_round = min(num_possible_games_by_players, self.num_courts)

            if num_active_players < players_per_court:
                round_info['sit_out'] = list(active_players_this_round)
                for p_name in active_players_this_round:
                    if p_name in sit_counts: # Ensure player was in initial list
                         sit_counts[p_name] += 1
                schedule.append(round_info)
                continue

            if num_games_this_round == 0:
                round_info['sit_out'] = list(active_players_this_round)
                for p_name in active_players_this_round:
                    if p_name in sit_counts:
                        sit_counts[p_name] += 1
                schedule.append(round_info)
                continue

            players_to_play_this_round_count = num_games_this_round * players_per_court
            
            # Select players for this round from the active pool
            # Filter candidate_players to only those active and in original counts
            eligible_players_for_selection = [p for p in active_players_this_round if p in play_counts]
            if not eligible_players_for_selection or len(eligible_players_for_selection) < players_to_play_this_round_count :
                 # Not enough eligible players (e.g. all subbed out, or too few left for even one game)
                round_info['sit_out'] = list(active_players_this_round)
                for p_name in active_players_this_round:
                    if p_name in sit_counts:
                        sit_counts[p_name] +=1
                schedule.append(round_info)
                continue


            random.shuffle(eligible_players_for_selection) # Tie-breaking for sort stability
            eligible_players_for_selection.sort(key=lambda p_name: (play_counts[p_name], -sit_counts.get(p_name, 0)))

            playing_this_round = eligible_players_for_selection[:players_to_play_this_round_count]
            sitting_this_round = eligible_players_for_selection[players_to_play_this_round_count:]
            
            for p_name in playing_this_round:
                play_counts[p_name] += 1
            for p_name in sitting_this_round:
                sit_counts[p_name] += 1 # sit_counts.get for safety, though should exist
            
            round_info['sit_out'] = sitting_this_round

            players_for_game_assignment_this_round = list(playing_this_round)
            random.shuffle(players_for_game_assignment_this_round)

            for court_idx in range(num_games_this_round):
                if len(players_for_game_assignment_this_round) < players_per_court:
                    break 
                if self.is_1v1:
                    # 1v1 singles: each game is two players, each their own team
                    p1, p2 = players_for_game_assignment_this_round[:2]
                    # Constraints: treat as team1 = [p1], team2 = [p2]
                    constraint_violated_flag = False
                    if frozenset([p1, p2]) in parsed_dnp_pairs:
                        constraint_violated_flag = True
                    if frozenset([p1, p2]) in parsed_dno_pairs:
                        constraint_violated_flag = True
                    # For fairness, score by how often they've played each other
                    current_score = session_opponent_counts[p1][p2] if not constraint_violated_flag else float('inf')
                    # No alternative matchups for singles, so just assign
                    round_info['games'].append(Game(court_idx + 1, [p1], [p2]))
                    # Update opponent counts
                    session_opponent_counts[p1][p2] += 1
                    session_opponent_counts[p2][p1] += 1
                    players_for_game_assignment_this_round = players_for_game_assignment_this_round[2:]
                else:
                    # 2v2 doubles logic (existing)
                    game_foursome_candidates = players_for_game_assignment_this_round[:players_per_court]
                    if len(game_foursome_candidates) == players_per_court:
                        p = game_foursome_candidates
                        possible_matchups_configs = [
                            {'team1': (p[0], p[1]), 'team2': (p[2], p[3])},
                            {'team1': (p[0], p[2]), 'team2': (p[1], p[3])},
                            {'team1': (p[0], p[3]), 'team2': (p[1], p[2])}
                        ]
                        best_matchups_for_score = []
                        min_score = float('inf')
                        for matchup_config in possible_matchups_configs:
                            current_score = 0
                            constraint_violated_flag = False
                            # Check DNP constraints
                            if frozenset(matchup_config['team1']) in parsed_dnp_pairs or \
                               frozenset(matchup_config['team2']) in parsed_dnp_pairs:
                                constraint_violated_flag = True
                            # Check DNO constraints
                            if not constraint_violated_flag:
                                for p_t1 in matchup_config['team1']:
                                    for p_t2 in matchup_config['team2']:
                                        if frozenset([p_t1, p_t2]) in parsed_dno_pairs:
                                            constraint_violated_flag = True
                                            break
                                    if constraint_violated_flag:
                                        break
                            if constraint_violated_flag:
                                current_score = float('inf')
                            else:
                                # Score for partners
                                current_score += session_partner_counts[matchup_config['team1'][0]][matchup_config['team1'][1]]
                                current_score += session_partner_counts[matchup_config['team2'][0]][matchup_config['team2'][1]]
                                # Score for opponents
                                for p1_team1 in matchup_config['team1']:
                                    for p2_team2 in matchup_config['team2']:
                                        current_score += session_opponent_counts[p1_team1][p2_team2]
                            if current_score < min_score:
                                min_score = current_score
                                best_matchups_for_score = [matchup_config]
                            elif current_score == min_score and current_score != float('inf'):
                                best_matchups_for_score.append(matchup_config)
                        if not best_matchups_for_score or min_score == float('inf'):
                            chosen_matchup = random.choice(possible_matchups_configs)
                        else:
                            chosen_matchup = random.choice(best_matchups_for_score)
                        team1_assigned = list(chosen_matchup['team1'])
                        team2_assigned = list(chosen_matchup['team2'])
                        round_info['games'].append(Game(court_idx + 1, team1_assigned, team2_assigned))
                        session_partner_counts[team1_assigned[0]][team1_assigned[1]] += 1
                        session_partner_counts[team1_assigned[1]][team1_assigned[0]] += 1
                        session_partner_counts[team2_assigned[0]][team2_assigned[1]] += 1
                        session_partner_counts[team2_assigned[1]][team2_assigned[0]] += 1
                        for p1 in team1_assigned:
                            for p2 in team2_assigned:
                                session_opponent_counts[p1][p2] += 1
                                session_opponent_counts[p2][p1] += 1
                        players_for_game_assignment_this_round = players_for_game_assignment_this_round[players_per_court:]
            
            schedule.append(round_info)

        return schedule

    def add_do_not_pair(self, player1: str, player2: str):
        for p in self.players:
            if p.name == player1:
                p.do_not_pair.add(player2)
            if p.name == player2:
                p.do_not_pair.add(player1)

    def set_player_availability(self, player_name: str, from_time: datetime.time, to_time: datetime.time):
        for p in self.players:
            if p.name == player_name:
                p.available_from = from_time
                p.available_to = to_time

    def assess_fairness(self, schedule):
        """
        Assess fairness of the schedule:
        - Games played per player (min, max, stddev, range)
        - Sit-outs per player (min, max, stddev, range)
        - Partner and opponent variety (min, max, stddev, range)
        """
        from collections import defaultdict
        import math
        players = [p.name for p in self.players]
        play_counts = {p: 0 for p in players}
        sit_counts = {p: 0 for p in players}
        partners = {p: set() for p in players}
        opponents = {p: set() for p in players}
        rounds = self.games_per_day
        for rnd, round_info in enumerate(schedule):
            playing = set()
            for game in round_info['games']:
                for team in [game.team1, game.team2]:
                    for p in team:
                        play_counts[p] += 1
                        playing.add(p)
                for t1, t2 in [(game.team1, game.team2), (game.team2, game.team1)]:
                    for p in t1:
                        partners[p].update([x for x in t1 if x != p])
                        opponents[p].update(t2)
            for p in round_info['sit_out']:
                sit_counts[p] += 1
        def stats(d):
            vals = list(d.values())
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            stddev = math.sqrt(var)
            return {
                'min': min(vals),
                'max': max(vals),
                'stddev': stddev,
                'range': max(vals) - min(vals)
            }
        fairness = {
            'games_played': play_counts,
            'sit_outs': sit_counts,
            'games_played_stats': stats(play_counts),
            'sit_outs_stats': stats(sit_counts),
            'partners_count': {p: len(partners[p]) for p in players},
            'partners_stats': stats({p: len(partners[p]) for p in players}),
            'opponents_count': {p: len(opponents[p]) for p in players},
            'opponents_stats': stats({p: len(opponents[p]) for p in players}),
        }
        return fairness

    # Removed all persistent stats/history methods: save_run_stats, load_all_stats, print_overall_stats, load_history, save_history, update_history, print_history_summary
    # No overall/cross-session stats are calculated or printed; only current run stats are shown.

# Example usage (to be replaced with actual input/UI):
# players = [Player('Alice', datetime.time(14,0), datetime.time(16,0)), ...]
# scheduler = Scheduler(players, num_courts=3)
# scheduler.add_do_not_pair('Alice', 'Bob')
# schedule = scheduler.generate_schedule()
# print(schedule)

if __name__ == "__main__":
    # Configuration to match app.py default for better comparison
    num_simulated_players = 14 # Changed back to 14 as requested
    simulated_num_courts = 3   # Auto-calculated for 12-15 players in app.py
    simulated_games_per_day = 8

    unique_emojis_main = ["😀", "😎", "🤩", "🥳", "😇", "🤓", "🧐", "😅", "😂", "😜", "🤔", "😴", "🤠", "😺", "🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼", "🐨", "🐯"]
    players = [
        Player(f'Player{i+1}', datetime.time(14,0), datetime.time(16,0), emoji=unique_emojis_main[i % len(unique_emojis_main)]) for i in range(num_simulated_players)
    ]
    
    player_emoji_map = {p.name: p.emoji for p in players}

    def get_emoji_with_name_terminal(name):
        emoji = player_emoji_map.get(name, '')
        return f"{emoji} {name}" if emoji else name

    scheduler = Scheduler(players, num_courts=simulated_num_courts)
    scheduler.games_per_day = simulated_games_per_day # Ensure this matches
    schedule = scheduler.generate_schedule()
    print("Schedule:")
    for rnd, round_info in enumerate(schedule, 1):
        print(f"\\nRound {rnd}:")
        for game in round_info['games']:
            team1_str = ' & '.join([get_emoji_with_name_terminal(p_name) for p_name in game.team1])
            team2_str = ' & '.join([get_emoji_with_name_terminal(p_name) for p_name in game.team2])
            print(f"  Court {game.court}: {team1_str} vs {team2_str}")
        if round_info['sit_out']:
            sit_out_str = ', '.join([get_emoji_with_name_terminal(p_name) for p_name in round_info['sit_out']])
            print(f"  Sit-out this round: {sit_out_str}")
    fairness = scheduler.assess_fairness(schedule)
    print("\nFairness assessment:")
    print(f"Games played per player: {fairness['games_played']}")
    print(f"  Stats: min={fairness['games_played_stats']['min']}, max={fairness['games_played_stats']['max']}, stddev={fairness['games_played_stats']['stddev']:.2f}, range={fairness['games_played_stats']['range']}")
    print(f"Sit-outs per player: {fairness['sit_outs']}")
    print(f"  Stats: min={fairness['sit_outs_stats']['min']}, max={fairness['sit_outs_stats']['max']}, stddev={fairness['sit_outs_stats']['stddev']:.2f}, range={fairness['sit_outs_stats']['range']}")
    print(f"Partners count per player: {fairness['partners_count']}")
    print(f"  Stats: min={fairness['partners_stats']['min']}, max={fairness['partners_stats']['max']}, stddev={fairness['partners_stats']['stddev']:.2f}, range={fairness['partners_stats']['range']}")
    print(f"Opponents count per player: {fairness['opponents_count']}")
    print(f"  Stats: min={fairness['opponents_stats']['min']}, max={fairness['opponents_stats']['max']}, stddev={fairness['opponents_stats']['stddev']:.2f}, range={fairness['opponents_stats']['range']}")