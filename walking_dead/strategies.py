"""
Deterministic placement and movement strategies for zombies and the survivor.

All tie-breaking is done by lowest node index, which guarantees fully
deterministic play given a fixed graph and fixed initial positions.
This is essential for cycle-detection termination to be mathematically exact.

Definitions
-----------
active zombie          : must take one step along a shortest path toward the survivor.
lazy zombie (greedy)   : may stay still OR take one step; stays still when no neighbour
                         is strictly closer to the survivor.
lazy zombie (strategic): paper's assignment-based strategy.  Each zombie is assigned a
                         target vertex v.  It moves to neighbour w only when w is
                         simultaneously closer to BOTH its assigned target v AND the
                         survivor.  If no such neighbour exists it stays still.  Once
                         a zombie reaches its assigned vertex it falls back to greedy
                         lazy behaviour.
active survivor        : must move to an adjacent vertex every turn.
lazy survivor          : may stay still OR move; chooses whichever maximises its
                         minimum distance to the nearest zombie.
"""

import logging
import random

import networkx as nx

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _dist_from(G: nx.Graph, source: int) -> dict[int, int]:
    """BFS distances from `source` to all other nodes."""
    return nx.single_source_shortest_path_length(G, source)


def _dist_to_survivor(G: nx.Graph, zombie_pos: int, survivor_pos: int) -> int:
    return nx.shortest_path_length(G, zombie_pos, survivor_pos)


def _min_dist_to_zombies(
    G: nx.Graph,
    node: int,
    zombie_positions: list[int],
    dist_cache: dict[int, dict[int, int]],
) -> int:
    """Minimum BFS distance from `node` to the nearest zombie."""
    return min(dist_cache[z][node] for z in zombie_positions)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def assign_targets(G: nx.Graph, k: int) -> list[int]:
    """
    Pick k spread-out strategic target vertices for the assignment-based lazy
    zombie strategy using a greedy k-center algorithm.

    Step 1: seed with the highest-degree node (most structurally important).
    Step 2: repeatedly add the node that maximises the minimum BFS distance to
            all already-chosen nodes.

    This approximates the ear-decomposition endpoint assignment described in the
    paper while remaining graph-agnostic.

    Returns a list of k node indices (one target per zombie).
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    if k >= n:
        return nodes[:k]

    # Pre-compute BFS distances from every node once — O(n(n+m))
    dist: dict[int, dict[int, int]] = {
        v: dict(nx.single_source_shortest_path_length(G, v)) for v in nodes
    }

    # Seed: highest-degree node
    first = max(nodes, key=lambda v: G.degree(v))
    chosen: list[int] = [first]

    while len(chosen) < k:
        best = max(
            (v for v in nodes if v not in chosen),
            key=lambda v: min(dist[v][c] for c in chosen),
        )
        chosen.append(best)

    log.debug("assign_targets: k=%d  targets=%s", k, chosen)
    return chosen


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def _outer_cycle_nodes(G: nx.Graph) -> list[int]:
    """
    Return the nodes of the outer Hamiltonian cycle of the outerplanar graph.

    In our graph_builder construction the outer cycle is always the ring
    0 - 1 - 2 - ... - (n-1) - 0, which spans every node.  We confirm this by
    checking that all sequential ring edges (i, i+1 mod n) exist, then return
    the ordered node list.

    If for any reason the sequential ring is incomplete (custom graph), we fall
    back to all nodes sorted by index so placement still works.
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    if n < 3:
        return nodes

    ring_intact = all(
        G.has_edge(nodes[i], nodes[(i + 1) % n]) for i in range(n)
    )
    if ring_intact:
        log.debug("_outer_cycle_nodes: ring 0-%d-0 confirmed (%d nodes)", n - 1, n)
        return nodes

    # Fallback: traverse starting from the lowest-index node, following ring edges
    log.debug(
        "_outer_cycle_nodes: sequential ring incomplete, falling back to all nodes"
    )
    return nodes


def place_zombies(G: nx.Graph, k: int, rng: random.Random) -> list[int]:
    """
    Place k zombies at random distinct vertices.  This is the sole source of
    randomness in the Monte Carlo simulation.
    """
    if k > len(G):
        raise ValueError(f"Cannot place {k} zombies on a graph with {len(G)} nodes.")
    positions = rng.sample(sorted(G.nodes()), k)
    log.debug("place_zombies: placed %d zombie(s) at %s", k, positions)
    return positions


def place_survivor(G: nx.Graph, zombie_positions: list[int]) -> int:
    """
    Place the survivor on the outer Hamiltonian cycle at the vertex that
    maximises its minimum BFS distance (in the full graph) to any zombie.
    Ties broken by lowest node index (deterministic).

    Only cycle vertices are considered as candidates so the survivor starts
    on the outer ring, which is the natural habitat in the outerplanar game.
    """
    cycle_nodes = _outer_cycle_nodes(G)
    zombie_set = set(zombie_positions)

    # Pre-compute full-graph BFS distances from every zombie once — O(k * n)
    # rather than calling shortest_path_length for every (cycle_node, zombie) pair.
    dist_from_zombie: dict[int, dict[int, int]] = {
        z: nx.single_source_shortest_path_length(G, z) for z in zombie_positions
    }

    best_node = -1
    best_dist = -1
    for v in cycle_nodes:
        if v in zombie_set:
            continue
        min_d = min(dist_from_zombie[z][v] for z in zombie_positions)
        log.debug(
            "  place_survivor candidate: node %d  min_dist_to_zombie=%d", v, min_d
        )
        if min_d > best_dist:
            best_dist = min_d
            best_node = v

    log.debug(
        "place_survivor: placed survivor at cycle node %d (min_dist_to_zombie=%d)",
        best_node, best_dist,
    )
    return best_node


# ---------------------------------------------------------------------------
# Zombie movement
# ---------------------------------------------------------------------------

def zombie_step(
    G: nx.Graph,
    pos: int,
    survivor_pos: int,
    lazy: bool,
    assigned_vertex: int | None = None,
) -> int:
    """
    Compute the next position for a single zombie.

    Active zombie  (lazy=False)
    ---------------------------
    Must move.  Picks the neighbour strictly closer to the survivor (lowest
    index tie-break).  Falls back to any neighbour if none are closer (edge
    case on disconnected graphs, which we never generate).

    Lazy greedy  (lazy=True, assigned_vertex=None)
    -----------------------------------------------
    Our default lazy mode.  Moves to the closest-to-survivor neighbour when
    one exists; otherwise stays still.

    Lazy strategic  (lazy=True, assigned_vertex=v)
    -----------------------------------------------
    Paper's assignment-based strategy.  Moves to neighbour w only when w is
    strictly closer to BOTH the assigned target vertex v AND the survivor.
    If no such dual-closer neighbour exists the zombie stays still.

    Special case: when the zombie is already at its assigned vertex (or the
    target equals the survivor's position), the dual condition can never be
    satisfied so the zombie falls back to greedy lazy behaviour — it guards
    its target while still capturing if the survivor comes adjacent.
    """
    if pos == survivor_pos:
        return pos  # already captured; caller handles this

    current_dist_survivor = nx.shortest_path_length(G, pos, survivor_pos)

    # Neighbours strictly closer to the survivor
    closer_survivor = sorted(
        n for n in G.neighbors(pos)
        if nx.shortest_path_length(G, n, survivor_pos) == current_dist_survivor - 1
    )

    if not lazy:
        # Active: must move toward survivor
        if closer_survivor:
            log.debug(
                "zombie_step [active]: %d -> %d  (dist %d -> %d)",
                pos, closer_survivor[0], current_dist_survivor, current_dist_survivor - 1,
            )
            return closer_survivor[0]
        # Fallback: move to closest neighbour (should not occur on connected graphs)
        neighbours = sorted(
            G.neighbors(pos),
            key=lambda n: nx.shortest_path_length(G, n, survivor_pos),
        )
        log.debug(
            "zombie_step [active, fallback]: %d -> %d  (no closer neighbour found)",
            pos, neighbours[0],
        )
        return neighbours[0]

    # --- Lazy from here ---

    if assigned_vertex is None:
        # Greedy lazy: move if closer to survivor
        if closer_survivor:
            log.debug(
                "zombie_step [lazy, greedy]: %d -> %d  (dist %d -> %d)",
                pos, closer_survivor[0], current_dist_survivor, current_dist_survivor - 1,
            )
            return closer_survivor[0]
        log.debug("zombie_step [lazy, greedy]: %d stays (no closer neighbour)", pos)
        return pos

    # Strategic lazy: dual condition
    # Fallback when already at assigned vertex or target == survivor
    if pos == assigned_vertex or assigned_vertex == survivor_pos:
        if closer_survivor:
            log.debug(
                "zombie_step [lazy, strategic, at_target]: %d -> %d  (greedy fallback)",
                pos, closer_survivor[0],
            )
            return closer_survivor[0]
        log.debug(
            "zombie_step [lazy, strategic, at_target]: %d stays  (target=%d)",
            pos, assigned_vertex,
        )
        return pos

    current_dist_target = nx.shortest_path_length(G, pos, assigned_vertex)

    # Neighbours closer to BOTH target AND survivor simultaneously
    dual_closer = sorted(
        n for n in G.neighbors(pos)
        if (nx.shortest_path_length(G, n, survivor_pos) == current_dist_survivor - 1
            and nx.shortest_path_length(G, n, assigned_vertex) == current_dist_target - 1)
    )

    if dual_closer:
        log.debug(
            "zombie_step [lazy, strategic]: %d -> %d  "
            "(closer to both target=%d and survivor=%d)",
            pos, dual_closer[0], assigned_vertex, survivor_pos,
        )
        return dual_closer[0]

    log.debug(
        "zombie_step [lazy, strategic]: %d stays  "
        "(no dual-closer neighbour, target=%d, dist_target=%d, dist_survivor=%d)",
        pos, assigned_vertex, current_dist_target, current_dist_survivor,
    )
    return pos


def move_zombies(
    G: nx.Graph,
    zombie_positions: list[int],
    survivor_pos: int,
    lazy: bool,
    assignments: list[int | None] | None = None,
) -> list[int]:
    """Return new positions for all zombies after one turn.

    assignments: per-zombie target vertex list (None entry = no assignment).
                 Defaults to all-None when omitted (greedy or active mode).
    """
    if assignments is None:
        assignments = [None] * len(zombie_positions)
    return [
        zombie_step(G, z, survivor_pos, lazy, assignments[i])
        for i, z in enumerate(zombie_positions)
    ]


# ---------------------------------------------------------------------------
# Survivor movement
# ---------------------------------------------------------------------------

def survivor_step(
    G: nx.Graph,
    pos: int,
    zombie_positions: list[int],
    lazy: bool,
    dist_cache: dict[int, dict[int, int]],
) -> int:
    """
    Compute the survivor's next position.

    Strategy: maximise minimum distance to the nearest zombie.
    Tie-break by lowest node index (deterministic).

    Active survivor: must move to an adjacent vertex (cannot stay).
    Lazy survivor  : may stay still or move; chooses the best option.

    The survivor will never voluntarily move onto a zombie's current vertex.
    """
    zombie_set = set(zombie_positions)

    def safety(node: int) -> int:
        if node in zombie_set:
            return -1  # never step onto a zombie
        return min(dist_cache[z][node] for z in zombie_positions)

    candidates: list[int] = sorted(G.neighbors(pos))
    if lazy:
        candidates = [pos] + candidates  # staying is a legal option

    best_node = -1
    best_safety = -2  # worse than any valid safety value
    for c in candidates:
        s = safety(c)
        if s > best_safety:
            best_safety = s
            best_node = c

    # If every move leads onto a zombie (completely surrounded), fall back to
    # staying still (only reachable if the graph is very small and heavily
    # populated with zombies).
    if best_node == -1:
        log.debug("survivor_step: %d stays (surrounded by zombies)", pos)
        return pos

    mode = "lazy" if lazy else "active"
    action = "stays" if best_node == pos else f"-> {best_node}"
    log.debug(
        "survivor_step [%s]: %d %s  (safety %d)",
        mode, pos, action, best_safety,
    )
    return best_node
