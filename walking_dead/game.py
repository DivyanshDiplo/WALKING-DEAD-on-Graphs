"""
Single deterministic game playout with exact cycle-detection termination.

Round structure (per the specification):
  Round 0 (Setup): zombies placed first, then survivor placed at the farthest
                   vertex from all zombies.
  Each subsequent round:
    1. Zombies move (active: must move; lazy: may stay).
    2. Check for capture: if any zombie is on the survivor's vertex -> Capture.
    3. Survivor moves (active: must move; lazy: may stay).
    4. Record state = (survivor_pos, tuple(sorted(zombie_positions))).
    5. If state already in visited_states -> Survivor Win (exact cycle proof).
    6. Add state to visited_states.  Continue.

Termination is guaranteed because:
  - The state space is finite (|V|^(k+1) states for k zombies on |V| nodes).
  - Play is fully deterministic (tie-breaking by lowest index everywhere).
  - Therefore the game either captures the survivor or enters a state cycle.
  - Cycle detection identifies the cycle at its first repeated state.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Literal

import networkx as nx

from .strategies import (
    _dist_from,
    move_zombies,
    place_survivor,
    place_zombies,
    survivor_step,
)

log = logging.getLogger(__name__)


Outcome = Literal["capture", "survivor_win"]


@dataclass(frozen=True, slots=True)
class GameResult:
    outcome: Outcome
    rounds: int  # number of complete rounds played before termination


def _build_dist_cache(
    G: nx.Graph, zombie_positions: list[int]
) -> dict[int, dict[int, int]]:
    """Pre-compute BFS distances from every zombie node."""
    return {z: _dist_from(G, z) for z in zombie_positions}


def play_game(
    G: nx.Graph,
    k: int,
    zombie_lazy: bool,
    survivor_lazy: bool,
    rng: random.Random,
) -> GameResult:
    """
    Run a single game on graph G.

    Parameters
    ----------
    G            : the outerplanar graph (must be connected).
    k            : number of zombies.
    zombie_lazy  : True = lazy zombies (may wait); False = active (must move).
    survivor_lazy: True = lazy survivor (may stay); False = active (must move).
    rng          : seeded RNG used only for zombie initial placement.

    Returns
    -------
    GameResult with outcome and number of rounds played.
    """
    # ------------------------------------------------------------------
    # Round 0: Placement
    # ------------------------------------------------------------------
    zombie_positions: list[int] = place_zombies(G, k, rng)
    survivor_pos: int = place_survivor(G, zombie_positions)

    log.debug(
        "--- game start: zombies=%s  survivor=%d  z_lazy=%s  s_lazy=%s ---",
        zombie_positions, survivor_pos, zombie_lazy, survivor_lazy,
    )

    # Immediate capture at placement (very dense zombie placement on tiny graphs)
    if survivor_pos in zombie_positions:
        log.debug("CAPTURE at placement (round 0)")
        return GameResult(outcome="capture", rounds=0)

    visited_states: set[tuple[int, tuple[int, ...]]] = set()

    round_num = 0
    while True:
        round_num += 1
        log.debug(
            "round %d begin: zombies=%s  survivor=%d",
            round_num, zombie_positions, survivor_pos,
        )

        # ------------------------------------------------------------------
        # Step 1: Zombies move
        # ------------------------------------------------------------------
        zombie_positions = move_zombies(G, zombie_positions, survivor_pos, zombie_lazy)
        log.debug("round %d after zombie move: zombies=%s", round_num, zombie_positions)

        # ------------------------------------------------------------------
        # Step 2: Capture check (after zombie move, before survivor moves)
        # ------------------------------------------------------------------
        if survivor_pos in zombie_positions:
            log.debug(
                "round %d CAPTURE: zombie reached survivor at node %d",
                round_num, survivor_pos,
            )
            return GameResult(outcome="capture", rounds=round_num)

        # ------------------------------------------------------------------
        # Step 3: Survivor moves
        # ------------------------------------------------------------------
        dist_cache = _build_dist_cache(G, zombie_positions)
        prev_survivor = survivor_pos
        survivor_pos = survivor_step(
            G, survivor_pos, zombie_positions, survivor_lazy, dist_cache
        )
        log.debug(
            "round %d after survivor move: %d -> %d",
            round_num, prev_survivor, survivor_pos,
        )

        # ------------------------------------------------------------------
        # Step 4 & 5: State recording and cycle detection
        # ------------------------------------------------------------------
        state = (survivor_pos, tuple(sorted(zombie_positions)))
        if state in visited_states:
            log.debug(
                "round %d CYCLE DETECTED: state %s already visited -> survivor wins",
                round_num, state,
            )
            return GameResult(outcome="survivor_win", rounds=round_num)
        visited_states.add(state)
        log.debug(
            "round %d state recorded  (visited_states size=%d)",
            round_num, len(visited_states),
        )
