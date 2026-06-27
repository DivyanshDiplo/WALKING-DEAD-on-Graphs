# Zombie Survivor -- Monte Carlo Simulator

A graph-theory simulation of the **Zombies-and-Survivor** pursuit game, played on random outerplanar graphs. Implements exact cycle-detection termination and supports two zombie variants (active / lazy) and two survivor variants (active / lazy).

## Rules summary

- **Round 0:** zombies are placed first (randomly), then the survivor is placed at the vertex farthest from all zombies.
- **Each subsequent round:** zombies move first, then the survivor moves.
- **Active zombie:** must move every turn along a shortest path toward the survivor.
- **Lazy zombie:** may stay still or move; moves only when a strictly closer neighbour exists.
- **Active survivor:** must move to an adjacent vertex every turn.
- **Lazy survivor:** may stay still or move; chooses whichever option maximises distance from the nearest zombie.
- **Capture:** a zombie lands on the survivor's vertex.
- **Survivor win:** the board state (survivor position + all zombie positions) repeats exactly -- on a finite graph with deterministic play this proves the survivor escapes forever.

## Project structure

```
walking_dead/
  graph_builder.py   # random outerplanar graph via recursive polygon triangulation
  strategies.py      # deterministic placement and move logic for zombies and survivor
  game.py            # single game playout with cycle-detection termination
  simulator.py       # Monte Carlo runner and statistics aggregation
  cli.py             # interactive CLI prompts and result display
  __main__.py        # entry point
```

## Requirements

- Python >= 3.10
- [Poetry](https://python-poetry.org/) (for dependency management)

Dependencies are declared in `pyproject.toml`:

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

```bash
python -m walking_dead
```

The CLI will prompt you for:

| Prompt | Description |
|---|---|
| Graph size `n` | Number of nodes in the outerplanar graph (>= 3) |
| Number of zombies `k` | Between 1 and n-1 |
| Lazy zombies? | `y` = lazy, `n` = active |
| Lazy survivor? | `y` = lazy, `n` = active |
| Number of trials | Monte Carlo sample size (e.g. 500) |

After the first run you can immediately re-run a different variation on the same graph and trial count for direct comparison.

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
```

## How Monte Carlo works here

- The graph topology is fixed for all trials in a run.
- Each trial randomises the zombies' starting positions (the sole source of randomness).
- The survivor is always placed deterministically at the farthest vertex from the zombies.
- All moves are deterministic (lowest node-index tie-breaking), so cycle detection is mathematically exact -- a repeated state is a proven survivor win, not a heuristic cutoff.
- The key output metric is **average rounds to capture** (over trials that ended in capture).
