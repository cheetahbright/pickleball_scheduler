"""Round-arrangement pool generation for the genetic scheduler.

Extracted from GeneticPickleballScheduler so the search space construction can
be tested in isolation from the GA loop. The generator is deterministic (fixed
seed) and every produced arrangement is structurally valid and respects the
hard do-not-pair / do-not-oppose maps.

Rounds are found by a backtracking search, so a constraint that rules out one
game makes the search try other partners and opponents instead of giving up on
the whole round. The pool holds each distinct round (by round_signature) once;
when it has room left, it also holds other court orders of those rounds.
"""

from __future__ import annotations

import builtins
import itertools
import random
from typing import Callable, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple

try:
    from src.algorithms.scheduler_metrics import round_signature
except ImportError:
    from algorithms.scheduler_metrics import round_signature

GameTuple = Tuple[str, str, str, str]
Arrangement = List[GameTuple]
IndexGame = Tuple[int, int, int, int]

# Search steps (one partner or opponent tried) the pool search may take by
# default. Only heavily constrained rosters come anywhere near it.
DEFAULT_NODE_BUDGET = 200_000
# Steps one sampled round may take before that sample is abandoned.
_SAMPLE_NODE_LIMIT = 2_000
# Samples in a row that may find nothing new before sampling stops and the
# search switches to visiting every valid round in turn.
_SAMPLE_STALL_LIMIT = 50


class ArrangementSearch(NamedTuple):
    """A round-arrangement pool, and how the search behind it ended."""

    arrangements: List[Arrangement]
    # True when the search visited every valid round, so a pool with room to
    # spare holds all of them.
    exhaustive: bool
    # True when the node budget ran out while the pool still had room for
    # more distinct rounds: its count of them is then only a lower bound, and
    # an empty pool proves nothing. A pool that filled up never sets it.
    budget_reached: bool


class _OutOfNodes(Exception):
    """The search used up its node budget."""


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


class _RoundSearch:
    """Depth-first search over the valid rounds for one player order.

    Every valid round is reached exactly once: one undecided player (the
    anchor) either plays - with any partner and any opposing pair - or sits
    out, while sit-out places remain. Each partner and opponent is checked
    against the same rules as game_valid as soon as it is placed, so a
    ruled-out choice prunes everything under it. Playing comes before
    sitting out, and earlier players before later ones.

    The anchor is normally the first undecided player in the order, so with
    no constraints the first round found is simply the order's players in
    groups of four, on courts in that order. With fail_first it is instead
    the player ruled out alongside the most undecided players, so one who
    fits almost nowhere is placed while there is still room: a full visit
    then proves quickly that no valid round exists, instead of retrying
    everyone else's games for each player left stranded.
    """

    def __init__(
        self,
        order: Sequence[str],
        num_courts: int,
        do_not_pair_map: Mapping[str, set],
        do_not_oppose_map: Mapping[str, set],
        *,
        fail_first: bool = False,
    ) -> None:
        self.order = list(order)
        self.num_courts = num_courts
        self.fail_first = fail_first
        self.sit_outs = len(self.order) - 4 * num_courts
        index = {name: i for i, name in enumerate(self.order)}
        self.no_pair: List[Set[int]] = [set() for _ in self.order]
        self.no_oppose: List[Set[int]] = [set() for _ in self.order]
        for conflicts, constraint_map in ((self.no_pair, do_not_pair_map), (self.no_oppose, do_not_oppose_map)):
            for name, i in index.items():
                for other in constraint_map.get(name, ()):
                    j = index.get(other)
                    if j is not None and j != i:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
        self.nodes = 0

    def visit(self, node_limit: int, on_round: Callable[[List[IndexGame]], bool]) -> bool:
        """Call on_round with each valid round (games as order indices) until
        it returns True. Returns True if it did, False once every valid round
        has been visited; raises _OutOfNodes past node_limit search steps."""
        if self.sit_outs < 0:
            return False  # not enough players to fill every court
        n = len(self.order)
        num_courts = self.num_courts
        no_pair, no_oppose = self.no_pair, self.no_oppose
        # Each player's conflicts as a bitmask, so counting the undecided ones
        # is a single AND and popcount.
        conflicts = [sum(1 << j for j in no_pair[i] | no_oppose[i]) for i in range(n)] if self.fail_first else []
        constrained = [i for i in range(len(conflicts)) if conflicts[i]]
        used = [False] * n
        games: List[IndexGame] = []
        limit = self.nodes + node_limit

        def pick_anchor() -> int:
            # With fail_first, the undecided player ruled out alongside the
            # most undecided players; only a constrained one can beat the
            # first undecided player.
            anchor = used.index(False)
            if not constrained:
                return anchor
            undecided = sum(1 << j for j in range(n) if not used[j])
            most_blocked = (conflicts[anchor] & undecided).bit_count()
            for i in constrained:
                if not used[i]:
                    blocked = (conflicts[i] & undecided).bit_count()
                    if blocked > most_blocked or (blocked == most_blocked and i < anchor):
                        anchor, most_blocked = i, blocked
            return anchor

        def place(sit_outs_left: int) -> bool:
            if len(games) == num_courts:
                return on_round(games)
            a = pick_anchor()
            used[a] = True
            pair_a, oppose_a = no_pair[a], no_oppose[a]
            for b in range(n):
                if used[b] or b in pair_a:
                    continue
                self.nodes += 1
                if self.nodes > limit:
                    raise _OutOfNodes
                used[b] = True
                oppose_b = no_oppose[b]
                for c in range(n):
                    if used[c] or c in oppose_a or c in oppose_b:
                        continue
                    self.nodes += 1
                    used[c] = True
                    pair_c = no_pair[c]
                    for d in range(c + 1, n):
                        if used[d] or d in pair_c or d in oppose_a or d in oppose_b:
                            continue
                        self.nodes += 1
                        used[d] = True
                        games.append((a, b, c, d))
                        if place(sit_outs_left):
                            return True
                        games.pop()
                        used[d] = False
                    used[c] = False
                used[b] = False
            # Player a sits this round out (left marked used, i.e. decided).
            if sit_outs_left and place(sit_outs_left - 1):
                return True
            used[a] = False
            return False

        return place(self.sit_outs)

    def names(self, games: Sequence[IndexGame]) -> Arrangement:
        order = self.order
        return [(order[a], order[b], order[c], order[d]) for a, b, c, d in games]


def _other_court_orders(rounds: Sequence[Arrangement], limit: int) -> List[Arrangement]:
    """Up to limit other court orders of rounds, taking one more order of
    each round in turn: its rotations first (each game on every court), then
    its remaining orders."""

    def orders(round_games: Arrangement) -> Iterator[Arrangement]:
        rotations = [round_games[shift:] + round_games[:shift] for shift in range(1, len(round_games))]
        yield from rotations
        for order in itertools.permutations(round_games):
            reordered = list(order)
            if reordered != round_games and reordered not in rotations:
                yield reordered

    pending = [orders(round_games) for round_games in rounds]
    extra: List[Arrangement] = []
    while pending and len(extra) < limit:
        still_pending = []
        for remaining in pending:
            reordered = next(remaining, None)
            if reordered is not None:
                extra.append(reordered)
                still_pending.append(remaining)
                if len(extra) == limit:
                    break
        pending = still_pending
    return extra


def search_arrangements(
    player_names: Sequence[str],
    num_courts: int,
    *,
    do_not_pair_map: Mapping[str, set],
    do_not_oppose_map: Mapping[str, set],
    max_arrangements: int = 400,
    seed: int = 123,
    node_budget: int = DEFAULT_NODE_BUDGET,
    printer: Callable[[str], None] = builtins.print,
) -> ArrangementSearch:
    """Build a pool of up to max_arrangements distinct valid rounds.

    First samples rounds: each sample shuffles the players and backtracks to
    the first valid round in that order, which spreads the pool across the
    whole space (with nothing ruled out, it is plain first-fit sampling).
    When sampling stops finding new rounds before the pool is full, the
    search visits every valid round in turn (placing the hardest-to-place
    players first) and fills the pool with a random selection of the rounds
    sampling missed.
    Finishing that visit within node_budget proves the pool complete; when
    the budget runs out first, the pool's count of distinct rounds is only a
    lower bound (ArrangementSearch.budget_reached).
    Any room left after that holds other court orders of the rounds found,
    so a round the schedule repeats can put its games on other courts.
    A name listed twice counts once: a player is never booked twice a round.
    """
    players = list(dict.fromkeys(player_names))
    rng = random.Random(seed)  # Keep this deterministic for arrangement generation
    nodes_used = 0
    arrangements: List[Arrangement] = []
    seen: set = set()
    exhaustive = False

    def first_round(order: List[str], fail_first: bool) -> Optional[Arrangement]:
        nonlocal nodes_used, exhaustive
        sampler = _RoundSearch(order, num_courts, do_not_pair_map, do_not_oppose_map, fail_first=fail_first)
        found: List[Arrangement] = []

        def keep_first(games: List[IndexGame]) -> bool:
            found.append(sampler.names(games))
            return True

        try:
            # A visit that ends without finding a round has visited them all.
            exhaustive = not sampler.visit(min(_SAMPLE_NODE_LIMIT, node_budget - nodes_used), keep_first)
        except _OutOfNodes:
            pass
        nodes_used += sampler.nodes
        return found[0] if found else None

    misses = 0
    for _attempt in range(max_arrangements * 10):
        if len(arrangements) >= max_arrangements or misses >= _SAMPLE_STALL_LIMIT or nodes_used >= node_budget // 2:
            break
        # Create fresh shuffled player list for each attempt
        player_pool = list(players)
        rng.shuffle(player_pool)
        round_games = first_round(player_pool, fail_first=False)
        if round_games is None and not exhaustive:
            # Lost in the shuffled order; place the hardest-to-place players
            # first instead, then shuffle the courts their games land on.
            round_games = first_round(player_pool, fail_first=True)
            if round_games is not None:
                rng.shuffle(round_games)
        if exhaustive:
            break  # no valid round exists at all
        signature = round_signature(round_games) if round_games else None
        if round_games is None or signature in seen:
            misses += 1
            continue
        misses = 0
        seen.add(signature)
        arrangements.append(round_games)

    if len(arrangements) < max_arrangements and not exhaustive:
        room = max_arrangements - len(arrangements)
        order = list(players)
        rng.shuffle(order)
        search = _RoundSearch(order, num_courts, do_not_pair_map, do_not_oppose_map, fail_first=True)
        reservoir: List[Arrangement] = []
        missed_rounds = 0

        def sample_missed(games: List[IndexGame]) -> bool:
            # Reservoir sampling: a uniform selection of the rounds sampling
            # missed, however many the search turns up.
            nonlocal missed_rounds
            round_games = search.names(games)
            if round_signature(round_games) not in seen:
                missed_rounds += 1
                if len(reservoir) < room:
                    reservoir.append(round_games)
                else:
                    slot = rng.randrange(missed_rounds)
                    if slot < room:
                        reservoir[slot] = round_games
            return False

        try:
            search.visit(max(0, node_budget - nodes_used), sample_missed)
            exhaustive = True
        except _OutOfNodes:
            pass
        nodes_used += search.nodes
        for round_games in reservoir:
            # fail_first puts the hardest-to-place player's game first;
            # shuffle the courts so no one is stuck on court 1.
            rng.shuffle(round_games)
            arrangements.append(round_games)

    distinct_rounds = len(arrangements)
    budget_reached = not exhaustive and distinct_rounds < max_arrangements
    # round_signature ignores court order, so these add no distinct round.
    court_orders = _other_court_orders(arrangements, max_arrangements - distinct_rounds)
    arrangements.extend(court_orders)

    printer(
        f"✅ Found {distinct_rounds} distinct valid round arrangements"
        + (" (every valid round visited)" if exhaustive else "")
        + (f", plus {len(court_orders)} other court orders of them" if court_orders else "")
    )
    if budget_reached:
        printer("ℹ️ The round search ran out of budget before visiting every valid round - more may exist")
    return ArrangementSearch(arrangements, exhaustive, budget_reached)


def generate_arrangements(
    player_names: Sequence[str],
    num_courts: int,
    *,
    do_not_pair_map: Mapping[str, set],
    do_not_oppose_map: Mapping[str, set],
    max_arrangements: int = 400,
    seed: int = 123,
    node_budget: Optional[int] = None,
    printer: Callable[[str], None] = builtins.print,
) -> List[Arrangement]:
    """The pool search_arrangements builds, without its exhaustiveness flag."""
    return search_arrangements(
        player_names,
        num_courts,
        do_not_pair_map=do_not_pair_map,
        do_not_oppose_map=do_not_oppose_map,
        max_arrangements=max_arrangements,
        seed=seed,
        node_budget=DEFAULT_NODE_BUDGET if node_budget is None else node_budget,
        printer=printer,
    ).arrangements
