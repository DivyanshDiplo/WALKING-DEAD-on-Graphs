# Zombie Survivor -- Monte Carlo Simulator

A graph-theory simulation of the **Zombies-and-Survivor** pursuit game, played on random outerplanar graphs. Implements exact cycle-detection termination and supports two zombie variants (active / lazy) and two survivor variants (active / lazy).

## Rules summary

- **Round 0:** zombies are placed first (randomly at any vertex). The survivor is then placed on the outer Hamiltonian cycle at the vertex that maximises its minimum BFS distance from all zombies.
- **Each subsequent round:** zombies move first, then the survivor moves.
- **Active zombie:** must move every turn along a shortest path toward the survivor; cannot stay still.
- **Lazy zombie:** may stay still or move; moves only when a strictly closer neighbour (on a shortest path) exists.
- **Active survivor:** must move to an adjacent vertex every turn; cannot stay still.
- **Lazy survivor:** may stay still or move; always chooses whichever option maximises its minimum distance from the nearest zombie.
- **Full information:** both sides know all positions at all times.
- **Capture:** a zombie occupies the same vertex as the survivor after a zombie turn.
- **Survivor win:** the full board state (survivor position + all zombie positions) repeats exactly. On a finite graph with fully deterministic play this is a mathematical proof that the survivor escapes forever.

## How the graph is built

A **random maximal outerplanar graph** (triangulated polygon) is generated on `n` nodes:

1. Build the outer Hamiltonian cycle `0 – 1 – 2 – … – (n-1) – 0`.
2. Recursively triangulate the interior by picking random split vertices and adding non-crossing chords.

Every node lies on the outer face (outerplanar by construction) and the graph is always connected.

## Project structure

```
walking_dead/
  graph_builder.py   # random outerplanar graph via recursive polygon triangulation
  strategies.py      # placement and deterministic move strategies for zombies and survivor
  game.py            # single game playout with exact cycle-detection termination
  simulator.py       # Monte Carlo runner and statistics aggregation
  logger.py          # centralised logger configuration (walking_dead.* namespace)
  cli.py             # interactive CLI prompts and result display
  __main__.py        # entry point  (python -m walking_dead)
```

## Requirements

- Python >= 3.10
- [Poetry](https://python-poetry.org/) (for dependency management)

Dependencies declared in `pyproject.toml`:

```
networkx >= 3.0
numpy < 2.0
```

## Setup

### Using Poetry

```bash
poetry install
poetry run python -m walking_dead
```

### Using a plain virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install networkx "numpy<2.0"
python -m walking_dead
```

## Running the simulator

### Normal run

```bash
python -m walking_dead
```

### Debug run (per-round trace logged to stderr)

```bash
python -m walking_dead --debug
```

The `--debug` flag activates the `walking_dead.*` logger hierarchy at `DEBUG` level. Every round, placement decision, zombie step, survivor step, state recording, and cycle-detection event is printed to `stderr`.

### CLI prompts

| Prompt | Description |
|---|---|
| Graph size `n` | Number of nodes in the outerplanar graph (>= 3) |
| Number of zombies `k` | Between 1 and n-1 |
| Lazy zombies? | `y` = lazy (may wait), `n` = active (must move) |
| Lazy survivor? | `y` = lazy (may wait), `n` = active (must move) |
| Number of trials | Monte Carlo sample size (e.g. 500) |

After the first run you can immediately re-run a different variation on the same graph and trial count for side-by-side comparison.

## Example session

```
==========================================================
  Zombie Survivor -- Monte Carlo Simulator
==========================================================

  Graph size  n (number of nodes, >= 3): 20
  Zombies     k (1 - 19): 2
  Lazy zombies?   (y = lazy, n = active) [y/n]: n
  Lazy survivor?  (y = lazy, n = active) [y/n]: y
  Number of Monte Carlo trials (>= 1): 500

  [ Zombies=active, Survivor=lazy ]
----------------------------------------------------------
  Trials          : 500
  Captures        : 312  (62.4%)
  Survivor wins   : 188
  Avg rounds      : 8.43
  Min rounds      : 2
  Max rounds      : 31
  Std-dev rounds  : 4.17

  Run another variation with the same graph/trial settings? [y/n]: y

  Lazy zombies?   (y = lazy, n = active) [y/n]: y
  Lazy survivor?  (y = lazy, n = active) [y/n]: y

  [ Zombies=lazy, Survivor=lazy ]
----------------------------------------------------------
  Trials          : 500
  Captures        : 289  (57.8%)
  Survivor wins   : 211
  Avg rounds      : 9.11
  Min rounds      : 2
  Max rounds      : 38
  Std-dev rounds  : 5.02
```

## How Monte Carlo works here

- The outerplanar graph topology is **fixed** for all trials in a single run.
- Each trial **randomises only the zombies' starting positions** — the sole source of stochasticity.
- The survivor is placed **deterministically** on the outer cycle at the vertex with maximum minimum BFS distance from all zombies.
- All moves are **fully deterministic** (lowest node-index tie-breaking everywhere), so cycle detection is mathematically exact: a repeated board state is a proven survivor win, not a heuristic cutoff.
- The key output metric is **average rounds to capture** (over trials that ended in capture), alongside capture rate, min, max, and standard deviation.

## Survivor placement detail

The survivor is restricted to the **outer Hamiltonian cycle** of the outerplanar graph. For each cycle vertex not occupied by a zombie, the minimum BFS distance (in the full graph, including interior chords) to the nearest zombie is computed. The vertex with the highest such distance is chosen; ties go to the lowest node index.

This is implemented efficiently by running one BFS per zombie (`O(k * n)`) and looking up distances from the resulting dictionaries, rather than calling shortest-path individually for every candidate pair.
