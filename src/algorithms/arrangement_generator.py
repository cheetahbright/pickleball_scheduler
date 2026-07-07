"""Round-arrangement pool generation for the genetic scheduler.

Extracted from GeneticPickleballScheduler so the search space construction can
be tested in isolation from the GA loop. Behavior is intentionally identical:
the generator is deterministic (fixed seed) and every produced arrangement is
structurally valid and respects the hard do-not-pair / do-not-oppose maps.
"""

from __future__ import annotations

import builtins
import random
from typing import Callable, List, Mapping, Sequence, Tuple, cast

try:
    from src.algorithms.scheduler_metrics import round_signature
except ImportError:
    from algorithms.scheduler_metrics import round_signature

GameTuple = Tuple[str, str, str, str]
Arrangement = List[GameTuple]


def game_valid(
    game: GameTuple,
    do_not_pair_map: Mapping[str, set],
    do_not_oppose_map: Mapping[str, set],
) -> bool:
    """Check if a game satisfies structural and hard-constraint rules."""
    p1, p2, p3, p4 = game

    # CRITICAL: Check for duplicate players within the game
    if len({p1, p2, p3, p4}) != 4:
        return False  # Same player appears multiple times in the game

    # Check do-not-pair constraints
    if p2 in do_not_pair_map[p1] or p1 in do_not_pair_map[p2]:
        return False
    if p4 in do_not_pair_map[p3] or p3 in do_not_pair_map[p4]:
        return False

    # Check do-not-oppose constraints
    for a in (p1, p2):
        for b in (p3, p4):
            if b in do_not_oppose_map[a] or a in do_not_oppose_map[b]:
                return False

    return True


def generate_arrangements(
    player_names: Sequence[str],
    num_courts: int,
    *,
    do_not_pair_map: Mapping[str, set],
    do_not_oppose_map: Mapping[str, set],
    max_arrangements: int = 400,
    seed: int = 123,
    printer: Callable[[str], None] = builtins.print,
) -> List[Arrangement]:
    """Generate MANY diverse, valid round arrangements to enable Range 0."""
    rng = random.Random(seed)  # Keep this deterministic for arrangement generation

    arrangements: List[Arrangement] = []

    # Generate many times more attempts to get diversity
    for _attempt in range(max_arrangements * 10):
        # Create fresh shuffled player list for each attempt
        player_pool = list(player_names)
        rng.shuffle(player_pool)

        round_games: Arrangement = []
        used_in_round: set = set()  # Track all players used in this round
        court = 0

        # GUARANTEED UNIQUE: Select 4 unused players for each game
        player_idx = 0
        while court < num_courts and player_idx < len(player_pool):
            # Find 4 players that haven't been used in this round
            game_players: List[str] = []
            temp_idx = player_idx

            while len(game_players) < 4 and temp_idx < len(player_pool):
                candidate = player_pool[temp_idx]
                if candidate not in used_in_round and candidate not in game_players:
                    game_players.append(candidate)
                temp_idx += 1

            # If we couldn't find 4 unused players, try next starting position
            if len(game_players) < 4:
                player_idx += 1
                continue

            p1, p2, p3, p4 = game_players

            # CRITICAL: Double-check uniqueness before accepting
            if len({p1, p2, p3, p4}) == 4 and game_valid((p1, p2, p3, p4), do_not_pair_map, do_not_oppose_map):
                round_games.append((p1, p2, p3, p4))
                used_in_round.update([p1, p2, p3, p4])
                court += 1
                player_idx = temp_idx  # Move past used players
            else:
                player_idx += 1  # Try different starting position

        # Only accept rounds that use exactly the right number of players
        if len(round_games) == num_courts and len(used_in_round) == num_courts * 4:
            # FINAL VALIDATION: Ensure no player appears in multiple games
            all_round_players: set = set()
            has_duplicates = False
            for game in round_games:
                game_set = set(game)
                if game_set & all_round_players:
                    has_duplicates = True
                    break
                all_round_players.update(game_set)

            if not has_duplicates and len(all_round_players) == num_courts * 4:
                arrangements.append(round_games)

        if len(arrangements) >= max_arrangements:
            break

    # Generate even more variations by swapping players within games
    original_count = len(arrangements)
    for arr in list(arrangements[: original_count // 2]):
        for _ in range(3):  # Create 3 variations of each
            var: Arrangement = []
            used_in_var: set = set()

            for game in arr:
                g = list(game)
                # Simple safe swaps within teams only (no cross-team swaps)
                if rng.random() < 0.5:
                    g[0], g[1] = g[1], g[0]  # Swap within team 1
                if rng.random() < 0.5:
                    g[2], g[3] = g[3], g[2]  # Swap within team 2

                game_tuple = cast(GameTuple, tuple(g))
                # Ensure no duplicates in this variation
                game_players_set = set(game_tuple)
                valid_game = len(game_players_set) == 4 and not any(p in used_in_var for p in game_players_set)
                if valid_game:
                    var.append(game_tuple)
                    used_in_var.update(game_players_set)
                else:
                    # Keep original game if swap would create duplicates
                    var.append(game)
                    used_in_var.update(game)

            # Only add variation if it maintains round validity
            if len(var) == len(arr) and len(used_in_var) == num_courts * 4:
                # FINAL VALIDATION: Ensure no player appears in multiple games in variation
                all_var_players: set = set()
                has_var_duplicates = False
                for game in var:
                    game_set = set(game)
                    if game_set & all_var_players:
                        has_var_duplicates = True
                        break
                    all_var_players.update(game_set)

                if not has_var_duplicates and len(all_var_players) == num_courts * 4:
                    arrangements.append(var)

        if len(arrangements) >= max_arrangements:
            break

    printer(f"✅ Generated {len(arrangements)} round arrangements")

    # FINAL FILTER: Remove any arrangements with duplicates
    valid_arrangements: List[Arrangement] = []
    for arr in arrangements:
        is_valid = True
        # Check each game in the arrangement
        for game in arr:
            if len(set(game)) != 4:
                is_valid = False
                break
        # Check for duplicates across games in the arrangement
        all_players: set = set()
        for game in arr:
            game_set = set(game)
            if game_set & all_players:
                is_valid = False
                break
            all_players.update(game_set)
        if is_valid and len(all_players) == num_courts * 4:
            valid_arrangements.append(arr)

    canonical_pattern_count = len({round_signature(arr) for arr in valid_arrangements})
    printer(f"✅ Filtered to {len(valid_arrangements)} valid round arrangements")
    printer(f"ℹ️ Canonical round patterns available: {canonical_pattern_count}")
    return valid_arrangements[:max_arrangements]
