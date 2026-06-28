# Zombie Survivor -- Monte Carlo Simulator

A graph-theory simulation of the **Zombies-and-Survivor** pursuit game, played on random outerplanar graphs. Implements exact cycle-detection termination and supports three zombie modes (active / lazy-greedy / lazy-strategic) and two survivor modes (active / lazy).

## Rules summary

- **Round 0:** zombies are placed first (randomly at any vertex). The survivor is then placed on the outer Hamiltonian cycle at the vertex that maximises its minimum BFS distance from all zombies.
- **Each subsequent round:** zombies move first, then the survivor moves.
- **Active zombie:** must move every turn along a shortest path toward the survivor; cannot stay still.
- **Lazy zombie (greedy):** may stay still or move; moves only when a strictly closer neighbour (on a shortest path to the survivor) exists.
- **Lazy zombie (strategic):** paper's assignment-based strategy — each zombie is assigned a spread-out strategic target vertex `v`; it moves to a neighbour `w` only when `w` is simultaneously closer to **both** its target `v` **and** the survivor. If no such dual-closer neighbour exists the zombie stays still. Once a zombie reaches its target it falls back to greedy-lazy behaviour.
- **Active survivor:** must move to an adjacent vertex every turn; cannot stay still.
- **Lazy survivor:** may stay still or move; always chooses whichever option maximises its minimum distance from the nearest zombie.
- **Full information:** both sides know all positions at all times.
- **Capture:** a zombie occupies the same vertex as the survivor after a zombie turn.
- **Survivor win:** the full board state (survivor position + all zombie positions) repeats exactly. On a finite graph with fully deterministic play this is a mathematical proof that the survivor escapes forever.

## Zombie strategies in detail

| Mode | CLI label | Movement rule |
|---|---|---|
| Active | `active` | Must take one step along a shortest path to the survivor every turn. |
| Lazy (greedy) | `lazy_greedy` | Moves only when a strictly closer-to-survivor neighbour exists; otherwise stays still. |
| Lazy (strategic) | `lazy_strategic` | Paper's assignment-based strategy. Each zombie is pre-assigned a spread-out target vertex. Moves to neighbour `w` only when `w` is closer to **both** the assigned target **and** the survivor simultaneously. Stays still otherwise. Falls back to greedy-lazy once at the target. |

**Target assignment for strategic lazy zombies** uses a greedy k-center algorithm: seed with the highest-degree node, then repeatedly pick the node that maximises the minimum BFS distance to already-chosen targets. This approximates the ear-decomposition-based assignment from the paper while remaining applicable to any outerplanar graph.

**Why the distinction matters:** a greedy lazy zombie always converges toward the survivor (if any nearer neighbour exists), so on sufficiently dense graphs with enough zombies the capture probability can still be high. A strategic lazy zombie guards a fixed location while only advancing when it can make simultaneous progress toward both its post and the survivor, which can result in lower capture rates but better theoretical coverage of the graph.

## How the graph is built

A **random outerplanar graph** is generated on `n` nodes in two steps:

1. Build the outer Hamiltonian cycle `0 – 1 – 2 – … – (n-1) – 0`.
2. Recursively triangulate the interior by picking random split vertices and adding non-crossing chords.

This produces a **maximal outerplanar graph** (every interior face is a triangle). The `chord_keep_prob` parameter (0 < p ≤ 1.0, prompted at runtime) then randomly removes interior chord edges with probability `1 - p`, producing a sparser connected outerplanar graph. Setting `chord_keep_prob = 1.0` keeps all chords (fully triangulated); lower values approach the bare outer cycle.

Every node always lies on the outer face (outerplanar by construction) and the graph is always connected because the outer Hamiltonian cycle is never thinned.

## Project structure

```
walking_dead/
  graph_builder.py   # random outerplanar graph via recursive polygon triangulation
  strategies.py      # placement and deterministic move strategies for zombies and survivor
  game.py            # single game playout with exact cycle-detection termination
  simulator.py       # Monte Carlo runner and statistics aggregation
  visualizer.py      # matplotlib report and graph PNG generation
  logger.py          # centralised logger configuration (walking_dead.* namespace)
  cli.py             # interactive CLI prompts and result display
  __main__.py        # entry point  (python -m walking_dead)
  tmp/               # auto-created; timestamped PNG outputs saved here
```

## Requirements

- Python >= 3.10
- [Poetry](https://python-poetry.org/) (for dependency management)

Dependencies declared in `pyproject.toml`:

```
networkx >= 3.0
numpy < 2.0
matplotlib >= 3.7
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

pip install networkx matplotlib "numpy<2.0"
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
| Chord keep probability | Fraction of interior chord edges to retain (0.01–1.0); `1.0` = fully triangulated |
| Lazy zombies? | `y` = lazy (may wait), `n` = active (must move) |
| Strategic lazy? | *(only shown when lazy zombies = y)* `y` = paper's assignment strategy, `n` = greedy lazy |
| Lazy survivor? | `y` = lazy (may wait), `n` = active (must move) |
| Number of trials | Monte Carlo sample size (e.g. 500) |

After the first run you can immediately re-run a different variation on the same graph and trial count for side-by-side comparison.

### Output files

After every session two timestamped PNG files are saved to `walking_dead/tmp/`:

| File | Contents |
|---|---|
| `report_<timestamp>.png` | Summary stats table, capture-rate bar chart, avg-rounds bar chart with std-dev error bars, rounds-distribution histogram |
| `graph_<timestamp>.png` | The outerplanar game graph drawn with circular layout; outer cycle edges in blue, interior chords as dashed grey lines |

Both files share the same timestamp so runs are easy to pair.

## Example session

```
==========================================================
  Zombie Survivor -- Monte Carlo Simulator
==========================================================

  Graph size  n (number of nodes, >= 3): 20
  Zombies     k (1 - 19): 2
  Chord keep probability (0.0 < p <= 1.0, 1.0 = fully triangulated): 0.6
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
  Strategic lazy? (y = paper's assignment strategy, n = greedy) [y/n]: n
  Lazy survivor?  (y = lazy, n = active) [y/n]: y

  [ Zombies=lazy_greedy, Survivor=lazy ]
----------------------------------------------------------
  Trials          : 500
  Captures        : 289  (57.8%)
  Survivor wins   : 211
  Avg rounds      : 9.11
  Min rounds      : 2
  Max rounds      : 38
  Std-dev rounds  : 5.02

  Run another variation with the same graph/trial settings? [y/n]: y

  Lazy zombies?   (y = lazy, n = active) [y/n]: y
  Strategic lazy? (y = paper's assignment strategy, n = greedy) [y/n]: y
  Lazy survivor?  (y = lazy, n = active) [y/n]: y

  [ Zombies=lazy_strategic, Survivor=lazy ]
----------------------------------------------------------
  Trials          : 500
  Captures        : 241  (48.2%)
  Survivor wins   : 259
  Avg rounds      : 12.07
  Min rounds      : 3
  Max rounds      : 47
  Std-dev rounds  : 6.88

----------------------------------------------------------
  Generating report...
  Report saved -> walking_dead/tmp/report_20260628_142300.png
  Graph saved  -> walking_dead/tmp/graph_20260628_142300.png
----------------------------------------------------------
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
