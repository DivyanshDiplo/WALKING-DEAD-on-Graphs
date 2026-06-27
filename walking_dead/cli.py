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
from .simulator import SimulationResult, run
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
        "\n  Termination: capture OR exact cycle detection (Survivor Win)."
        "\n  Randomness: zombie initial placement only.\n"
    )

    # --- Graph parameters ---
    print("  Graph settings")
    print(_THIN)
    n = _prompt_int("  Graph size  n (number of nodes, >= 3): ", lo=3)
    k = _prompt_int(f"  Zombies     k (1 – {n - 1}): ", lo=1, hi=n - 1)

    # --- Mode parameters ---
    print()
    print("  Variation settings")
    print(_THIN)
    zombie_lazy   = _prompt_bool("  Lazy zombies?   (y = lazy, n = active)")
    survivor_lazy = _prompt_bool("  Lazy survivor?  (y = lazy, n = active)")

    # --- Simulation parameters ---
    print()
    print("  Simulation settings")
    print(_THIN)
    trials = _prompt_int("  Number of Monte Carlo trials (>= 1): ", lo=1)

    # --- Run ---
    zombie_label   = "lazy"   if zombie_lazy   else "active"
    survivor_label = "lazy"   if survivor_lazy else "active"
    label = f"Zombies={zombie_label}, Survivor={survivor_label}"

    print()
    print(_THIN)
    print(f"  Running {trials} trial(s) on an outerplanar graph with {n} nodes "
          f"and {k} zombie(s)...")
    print(_THIN)

    result = run(
        n=n,
        k=k,
        zombie_lazy=zombie_lazy,
        survivor_lazy=survivor_lazy,
        trials=trials,
    )

    _header("Results")
    _print_result(label, result)

    # Collect all results for the final report
    session_results: list[tuple[str, SimulationResult]] = [(label, result)]

    # Offer to run another variation on the same configuration
    print()
    while _prompt_bool("\n  Run another variation with the same graph/trial settings?"):
        print()
        zombie_lazy   = _prompt_bool("  Lazy zombies?   (y = lazy, n = active)")
        survivor_lazy = _prompt_bool("  Lazy survivor?  (y = lazy, n = active)")
        zombie_label   = "lazy"   if zombie_lazy   else "active"
        survivor_label = "lazy"   if survivor_lazy else "active"
        label = f"Zombies={zombie_label}, Survivor={survivor_label}"

        print(_THIN)
        print(f"  Running {trials} trial(s)...")
        result = run(
            n=n,
            k=k,
            zombie_lazy=zombie_lazy,
            survivor_lazy=survivor_lazy,
            trials=trials,
        )
        _print_result(label, result)
        session_results.append((label, result))

    # Save matplotlib report + graph (shared timestamp so files pair up)
    print()
    print(_THIN)
    print("  Generating report...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = save_report(
        run_config={"n": n, "k": k, "trials": trials},
        results=session_results,
        timestamp=ts,
    )
    first_graph = session_results[0][1].graph
    graph_path = save_graph(
        G=first_graph,
        run_config={"n": n, "k": k, "trials": trials},
        timestamp=ts,
    )
    print(f"  Report saved -> {report_path}")
    print(f"  Graph saved  -> {graph_path}")
    print(_THIN)

    print()
    print(_DIVIDER)
    print("  Goodbye.")
    print(_DIVIDER)
    print()
