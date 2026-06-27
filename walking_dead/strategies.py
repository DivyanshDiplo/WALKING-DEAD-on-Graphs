"""
Deterministic placement and movement strategies for zombies and the survivor.

All tie-breaking is done by lowest node index, which guarantees fully
deterministic play given a fixed graph and fixed initial positions.
This is essential for cycle-detection termination to be mathematically exact.

Definitions
-----------
active zombie  : must take one step along a shortest path toward the survivor.
lazy zombie    : may stay still OR take one step; stays still if it is already
                 on the unique best position (i.e. no neighbour on a shortest
                 path to the survivor is strictly closer than it currently is).
active survivor: must move to an adjacent vertex every turn.
lazy survivor  : may stay still OR move; chooses whichever maximises its
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
) -> int:
    """
    Compute the next position for a single zombie.

    Active zombie
    -------------
    Must move.  Picks the neighbour that lies on a shortest path to the
    survivor (i.e. is strictly closer than the current position).  Tie-breaks
    by lowest node index.  If all neighbours are equidistant or farther (only
    possible on disconnected graphs, which we never generate), stays put as a
    safe fallback.

    Lazy zombie
    -----------
    Moves only if doing so strictly reduces distance to the survivor.
    If the zombie is already at distance 0 (on the survivor — capture already
    handled upstream) or if no neighbour is strictly closer, it stays still.
    """
    if pos == survivor_pos:
        return pos  # already captured; caller handles this

    current_dist = nx.shortest_path_length(G, pos, survivor_pos)

    # Neighbours that are strictly one step closer on a shortest path
    closer = sorted(
        [
            n for n in G.neighbors(pos)
            if nx.shortest_path_length(G, n, survivor_pos) == current_dist - 1
        ]
    )

    if lazy:
        # Lazy: only move if there is a strictly closer neighbour
        if closer:
            log.debug(
                "zombie_step [lazy]: %d -> %d  (dist %d -> %d)",
                pos, closer[0], current_dist, current_dist - 1,
            )
            return closer[0]   # deterministic: lowest index
        log.debug("zombie_step [lazy]: %d stays (no strictly closer neighbour)", pos)
        return pos             # stay still
    else:
        # Active: must move; pick closest neighbour; fall back to any neighbour
        if closer:
            log.debug(
                "zombie_step [active]: %d -> %d  (dist %d -> %d)",
                pos, closer[0], current_dist, current_dist - 1,
            )
            return closer[0]
        # Edge case: no neighbour is closer (should not happen on connected graphs
        # when current_dist > 0).  Move to the neighbour with the smallest dist.
        neighbours = sorted(G.neighbors(pos))
        neighbours.sort(key=lambda n: nx.shortest_path_length(G, n, survivor_pos))
        log.debug(
            "zombie_step [active, fallback]: %d -> %d  (no closer neighbour found)",
            pos, neighbours[0],
        )
        return neighbours[0]


def move_zombies(
    G: nx.Graph,
    zombie_positions: list[int],
    survivor_pos: int,
    lazy: bool,
) -> list[int]:
    """Return new positions for all zombies after one turn."""
    return [zombie_step(G, z, survivor_pos, lazy) for z in zombie_positions]


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
