"""
Monte Carlo runner: runs many deterministic game trials and aggregates results.

Each trial re-randomises the zombies' starting positions (via the RNG), which
is the sole source of stochasticity.  The graph itself is kept fixed across all
trials of a single simulation run so that results reflect placement variance
on a consistent topology.

Reported statistics
-------------------
trials          : total trials requested.
captures        : number of trials that ended in zombie capture.
survivor_wins   : number of trials where the survivor escaped (cycle detected).
capture_rate    : captures / trials  (fraction, 0–1).
avg_rounds      : mean rounds-to-capture over captured trials only.
                  None if no trial ended in capture.
min_rounds      : minimum capture rounds (None if no capture).
max_rounds      : maximum capture rounds (None if no capture).
std_rounds      : std-dev of capture rounds (None if fewer than 2 captures).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import networkx as nx

from .game import GameResult, play_game
from .graph_builder import build_outerplanar


@dataclass
class SimulationResult:
    trials: int
    captures: int
    survivor_wins: int
    capture_rate: float
    avg_rounds: float | None
    min_rounds: int | None
    max_rounds: int | None
    std_rounds: float | None

    def summary(self) -> str:
        lines = [
            f"  Trials          : {self.trials}",
            f"  Captures        : {self.captures}  ({self.capture_rate * 100:.1f}%)",
            f"  Survivor wins   : {self.survivor_wins}",
        ]
        if self.avg_rounds is not None:
            lines += [
                f"  Avg rounds      : {self.avg_rounds:.2f}",
                f"  Min rounds      : {self.min_rounds}",
                f"  Max rounds      : {self.max_rounds}",
                f"  Std-dev rounds  : {self.std_rounds:.2f}" if self.std_rounds is not None else "  Std-dev rounds  : N/A",
            ]
        else:
            lines.append("  Avg rounds      : N/A (no captures)")
        return "\n".join(lines)


def run(
    n: int,
    k: int,
    zombie_lazy: bool,
    survivor_lazy: bool,
    trials: int,
    seed: int | None = None,
    graph_seed: int | None = None,
) -> SimulationResult:
    """
    Run `trials` Monte Carlo games and return aggregated statistics.

    Parameters
    ----------
    n            : number of nodes in the outerplanar graph.
    k            : number of zombies.
    zombie_lazy  : True = lazy zombies; False = active zombies.
    survivor_lazy: True = lazy survivor; False = active survivor.
    trials       : number of Monte Carlo trials.
    seed         : RNG seed for placement randomness (None = non-deterministic).
    graph_seed   : separate seed for graph construction (None = non-deterministic).

    Returns
    -------
    SimulationResult
    """
    G: nx.Graph = build_outerplanar(n, seed=graph_seed)
    rng = random.Random(seed)

    capture_rounds: list[int] = []
    survivor_wins = 0

    for _ in range(trials):
        result: GameResult = play_game(G, k, zombie_lazy, survivor_lazy, rng)
        if result.outcome == "capture":
            capture_rounds.append(result.rounds)
        else:
            survivor_wins += 1

    captures = len(capture_rounds)
    capture_rate = captures / trials if trials > 0 else 0.0

    if captures == 0:
        return SimulationResult(
            trials=trials,
            captures=0,
            survivor_wins=survivor_wins,
            capture_rate=capture_rate,
            avg_rounds=None,
            min_rounds=None,
            max_rounds=None,
            std_rounds=None,
        )

    avg = sum(capture_rounds) / captures
    mn = min(capture_rounds)
    mx = max(capture_rounds)
    std: float | None = None
    if captures >= 2:
        variance = sum((r - avg) ** 2 for r in capture_rounds) / (captures - 1)
        std = math.sqrt(variance)

    return SimulationResult(
        trials=trials,
        captures=captures,
        survivor_wins=survivor_wins,
        capture_rate=capture_rate,
        avg_rounds=avg,
        min_rounds=mn,
        max_rounds=mx,
        std_rounds=std,
    )
