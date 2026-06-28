"""
Interactive CLI for the Zombie Survivor Monte Carlo simulator.

Prompts the user for:
  - Graph size n  (number of nodes, >= 3)
  - Number of zombies k  (>= 1, < n)
  - Zombie mode: active or lazy
  - Survivor mode: active or lazy
  - Number of Monte Carlo trials

Then runs all requested variation(s) and prints a formatted summary table.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

from .logger import configure
from .simulator import SimulationResult, run, run_paper
from .visualizer import save_graph, save_report


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _prompt_int(prompt: str, lo: int, hi: int | None = None) -> int:
    """Prompt until the user enters an integer in [lo, hi]."""
    while True:
        raw = input(prompt).strip()
        try:
            val = int(raw)
        except ValueError:
            print(f"  Please enter a whole number.")
            continue
        if val < lo:
            print(f"  Value must be >= {lo}.")
            continue
        if hi is not None and val > hi:
            print(f"  Value must be <= {hi}.")
            continue
        return val


def _prompt_bool(prompt: str) -> bool:
    """Prompt until the user enters y/n."""
    while True:
        raw = input(prompt + " [y/n]: ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter y or n.")


def _prompt_float(prompt: str, lo: float, hi: float) -> float:
    """Prompt until the user enters a float in [lo, hi]."""
    while True:
        raw = input(prompt).strip()
        try:
            val = float(raw)
        except ValueError:
            print("  Please enter a decimal number.")
            continue
        if val < lo or val > hi:
            print(f"  Value must be between {lo} and {hi}.")
            continue
        return val


# ---------------------------------------------------------------------------
# Zombie mode helpers
# ---------------------------------------------------------------------------

def _prompt_zombie_modes(k: int) -> list[str]:
    """
    Ask the user to define a zombie type for each of the k zombies.

    If all zombies are the same type the user only answers two questions
    (lazy? / strategic?).  For a mixed squad the user enters counts per type.

    Returns a flat list of k mode strings:
      "active" | "lazy_greedy" | "lazy_strategic"
    """
    same = _prompt_bool("  All zombies same type?  (y = uniform, n = mixed types)")

    if same:
        lazy = _prompt_bool("  Lazy zombies?   (y = lazy, n = active)")
        if not lazy:
            return ["active"] * k
        strategic = _prompt_bool(
            "  Strategic lazy? (y = paper's assignment strategy, n = greedy)"
        )
        return ["lazy_strategic" if strategic else "lazy_greedy"] * k

    # Mixed: ask how many of each type (greedy remainder = strategic)
    print(f"  Enter counts — must total {k}:")
    n_active = _prompt_int(
        f"    Active zombies            (0-{k}): ", lo=0, hi=k
    )
    remaining = k - n_active
    n_greedy = _prompt_int(
        f"    Lazy-greedy zombies       (0-{remaining}): ", lo=0, hi=remaining
    )
    n_strategic = remaining - n_greedy
    print(f"    Lazy-strategic zombies  : {n_strategic}")

    return (["active"] * n_active
            + ["lazy_greedy"] * n_greedy
            + ["lazy_strategic"] * n_strategic)


def _zombie_label(modes: list[str]) -> str:
    """
    Build a compact human-readable label from a list of zombie mode strings.

    Single-type squads  →  "active" / "lazy_greedy" / "lazy_strategic"
    Mixed squads        →  "2×active+1×lazy_greedy" etc.
    """
    from collections import Counter
    counts = Counter(modes)
    if len(counts) == 1:
        return next(iter(counts))
    parts = [
        f"{counts[m]}×{m}"
        for m in ["active", "lazy_greedy", "lazy_strategic"]
        if m in counts
    ]
    return "+".join(parts)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_DIVIDER = "=" * 58
_THIN    = "-" * 58


def _header(title: str) -> None:
    print()
    print(_DIVIDER)
    print(f"  {title}")
    print(_DIVIDER)


def _print_result(label: str, result: SimulationResult) -> None:
    print(f"\n  [ {label} ]")
    print(_THIN)
    print(result.summary())


# ---------------------------------------------------------------------------
# Interface 1 — heuristic simulation (original)
# ---------------------------------------------------------------------------

def _interface_one() -> None:
    """Original Monte Carlo simulator with configurable zombie strategies."""
    print(
        "\n  Interface 1: Heuristic Simulation"
        "\n  ----------------------------------"
        "\n  Runs k zombies with the selected strategy (active / lazy-greedy /"
        "\n  lazy-strategic) against the survivor on a random outerplanar graph."
        "\n  Laziness is a heuristic approximation of the paper's concept.\n"
    )

    # --- Graph parameters ---
    print("  Graph settings")
    print(_THIN)
    n = _prompt_int("  Graph size  n (number of nodes, >= 3): ", lo=3)
    k = _prompt_int(f"  Zombies     k (1 - {n - 1}): ", lo=1, hi=n - 1)
    chord_keep_prob = _prompt_float(
        "  Chord keep probability (0.0 < p <= 1.0, 1.0 = fully triangulated): ",
        lo=0.01, hi=1.0,
    )

    # --- Mode parameters ---
    print()
    print("  Variation settings")
    print(_THIN)
    zombie_modes  = _prompt_zombie_modes(k)
    survivor_lazy = _prompt_bool("  Lazy survivor?  (y = lazy, n = active)")

    # --- Simulation parameters ---
    print()
    print("  Simulation settings")
    print(_THIN)
    trials = _prompt_int("  Number of Monte Carlo trials (>= 1): ", lo=1)

    # --- First run ---
    zombie_label   = _zombie_label(zombie_modes)
    survivor_label = "lazy" if survivor_lazy else "active"
    label = f"Zombies={zombie_label}, Survivor={survivor_label}"

    print()
    print(_THIN)
    print(f"  Running {trials} trial(s) on an outerplanar graph with {n} nodes, "
          f"{k} zombie(s), chord_keep_prob={chord_keep_prob:.2f}...")
    print(_THIN)

    result = run(
        n=n,
        k=k,
        zombie_modes=zombie_modes,
        survivor_lazy=survivor_lazy,
        trials=trials,
        chord_keep_prob=chord_keep_prob,
    )

    _header("Results")
    _print_result(label, result)

    session_results: list[tuple[str, SimulationResult]] = [(label, result)]

    # Offer to run more variations on the same graph / trial count
    print()
    while _prompt_bool("\n  Run another variation with the same graph/trial settings?"):
        print()
        zombie_modes  = _prompt_zombie_modes(k)
        survivor_lazy = _prompt_bool("  Lazy survivor?  (y = lazy, n = active)")
        zombie_label   = _zombie_label(zombie_modes)
        survivor_label = "lazy" if survivor_lazy else "active"
        label = f"Zombies={zombie_label}, Survivor={survivor_label}"

        print(_THIN)
        print(f"  Running {trials} trial(s)...")
        result = run(
            n=n,
            k=k,
            zombie_modes=zombie_modes,
            survivor_lazy=survivor_lazy,
            trials=trials,
            chord_keep_prob=chord_keep_prob,
        )
        _print_result(label, result)
        session_results.append((label, result))

    _save_outputs(n=n, k=k, trials=trials, chord_keep_prob=chord_keep_prob,
                  session_results=session_results)


# ---------------------------------------------------------------------------
# Interface 2 — paper's algorithm (Theorem 6 / Corollary 1)
# ---------------------------------------------------------------------------

def _interface_two() -> None:
    """
    Runs the paper's exact 2-lazy-zombie strategy (Bose, De Carufel & Shermer
    2022, Theorem 6 / Corollary 1).

    Key result replicated:
      uL(G) = 2 for every connected outerplanar graph:
      exactly 2 lazy zombies suffice to always catch the survivor in < 2n rounds.

    The paper's algorithm uses a coordinated guard-advance strategy:
      • One zombie (guard) sits stationary at a chord endpoint, sealing one
        boundary of the survivor territory.
      • The other zombie (advance) walks the outer Hamiltonian circuit one step
        per round, shrinking the territory.
      • When the advance hits a chord endpoint, it either jumps ahead or
        triggers a role-swap, always shrinking the territory.
    """
    print(
        "\n  Interface 2: Paper's Algorithm  (Theorem 6 / Corollary 1)"
        "\n  -----------------------------------------------------------"
        "\n  Always uses exactly 2 lazy zombies implementing the coordinated"
        "\n  guard-advance strategy.  Expected capture rate: 100 %.\n"
    )

    # --- Graph parameters ---
    print("  Graph settings")
    print(_THIN)
    n = _prompt_int("  Graph size  n (number of nodes, >= 3): ", lo=3)
    chord_keep_prob = _prompt_float(
        "  Chord keep probability (0.0 < p <= 1.0, 1.0 = fully triangulated): ",
        lo=0.01, hi=1.0,
    )

    # --- Mode parameters ---
    print()
    print("  Variation settings")
    print(_THIN)
    survivor_lazy = _prompt_bool("  Lazy survivor?  (y = lazy, n = active)")

    # --- Simulation parameters ---
    print()
    print("  Simulation settings")
    print(_THIN)
    trials = _prompt_int("  Number of Monte Carlo trials (>= 1): ", lo=1)

    # --- Run ---
    survivor_label = "lazy" if survivor_lazy else "active"
    label = f"Zombies=paper_algorithm(k=2), Survivor={survivor_label}"

    print()
    print(_THIN)
    print(f"  Running {trials} trial(s) [paper algorithm] on an outerplanar graph "
          f"with {n} nodes, chord_keep_prob={chord_keep_prob:.2f}...")
    print(_THIN)

    result = run_paper(
        n=n,
        survivor_lazy=survivor_lazy,
        trials=trials,
        chord_keep_prob=chord_keep_prob,
    )

    _header("Results")
    _print_result(label, result)

    session_results: list[tuple[str, SimulationResult]] = [(label, result)]

    # Offer to compare with different survivor mode or chord probability
    print()
    while _prompt_bool("\n  Run another variation with the same graph/trial settings?"):
        print()
        survivor_lazy = _prompt_bool("  Lazy survivor?  (y = lazy, n = active)")
        survivor_label = "lazy" if survivor_lazy else "active"
        label = f"Zombies=paper_algorithm(k=2), Survivor={survivor_label}"

        print(_THIN)
        print(f"  Running {trials} trial(s) [paper algorithm]...")
        result = run_paper(
            n=n,
            survivor_lazy=survivor_lazy,
            trials=trials,
            chord_keep_prob=chord_keep_prob,
        )
        _print_result(label, result)
        session_results.append((label, result))

    _save_outputs(n=n, k=2, trials=trials, chord_keep_prob=chord_keep_prob,
                  session_results=session_results)


# ---------------------------------------------------------------------------
# Shared output helper
# ---------------------------------------------------------------------------

def _save_outputs(
    n: int,
    k: int,
    trials: int,
    chord_keep_prob: float,
    session_results: list[tuple[str, SimulationResult]],
) -> None:
    """Save the matplotlib report and graph image to the tmp folder."""
    print()
    print(_THIN)
    print("  Generating report...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_config = {"n": n, "k": k, "trials": trials, "chord_keep_prob": chord_keep_prob}
    report_path = save_report(
        run_config=run_config,
        results=session_results,
        timestamp=ts,
    )
    first_graph = session_results[0][1].graph
    graph_path = save_graph(
        G=first_graph,
        run_config=run_config,
        timestamp=ts,
    )
    print(f"  Report saved -> {report_path}")
    print(f"  Graph saved  -> {graph_path}")
    print(_THIN)


# ---------------------------------------------------------------------------
# Main CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    debug = "--debug" in sys.argv
    configure(debug=debug)
    if debug:
        logging.getLogger("walking_dead").debug(
            "debug logging enabled (walking_dead.*)"
        )

    _header("Zombie Survivor -- Monte Carlo Simulator")
    print(
        "\n  Game: deterministic Zombies-and-Survivor on a random outerplanar graph."
        "\n  Termination: capture OR exact cycle / stall detection (Survivor Win)."
        "\n  Randomness: zombie initial placement only.\n"
    )

    print("  Select interface:")
    print("    1  Heuristic simulation  — configurable k zombies (active / lazy-greedy /")
    print("                               lazy-strategic), any zombie count")
    print("    2  Paper's algorithm     — exactly 2 lazy zombies, Theorem 6 strategy")
    print("                               (Bose, De Carufel & Shermer 2022)")
    print()
    interface = _prompt_int("  Interface (1 or 2): ", lo=1, hi=2)

    if interface == 1:
        _interface_one()
    else:
        _interface_two()

    print()
    print(_DIVIDER)
    print("  Goodbye.")
    print(_DIVIDER)
    print()
