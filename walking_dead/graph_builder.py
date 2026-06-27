"""
Random outerplanar graph generation via recursive polygon triangulation.

A maximal outerplanar graph (MOP) on n >= 3 nodes is constructed by:
  1. Forming the outer Hamiltonian cycle  0 – 1 – 2 – … – (n-1) – 0.
  2. Recursively triangulating the interior: given a chord between node i and
     node j (on the cycle), pick a random intermediate node k and add the two
     non-crossing chords (i,k) and (k,j), then recurse on both sub-polygons.

The result is always planar, connected, and every vertex lies on the outer face
(outerplanar by construction).

An optional `chord_keep_prob` (0 < p <= 1.0, default 1.0) lets callers drop
interior chord edges at random after triangulation, producing sparser graphs
while keeping the outer cycle intact so the graph stays connected.
"""

import logging
import random

import networkx as nx

log = logging.getLogger(__name__)


def _triangulate(G: nx.Graph, cycle: list[int], rng: random.Random) -> None:
    """
    Recursively add interior chords to triangulate a polygon.

    `cycle` is the ordered list of node indices that form the current polygon
    boundary (the first and last elements are already connected by an edge).
    A polygon of size 2 is already a single edge — nothing to do.
    A polygon of size 3 is a triangle — nothing to add.
    """
    if len(cycle) < 4:
        return

    # Candidate split indices: everything between cycle[0] and cycle[-1]
    candidates = cycle[1:-1]
    k = rng.choice(candidates)
    k_idx = cycle.index(k)
    log.debug("  triangulate: polygon=%s  split_vertex=%d", cycle, k)

    # Add the two chords from the outer endpoints to the split vertex
    if not G.has_edge(cycle[0], k):
        G.add_edge(cycle[0], k)
    if not G.has_edge(k, cycle[-1]):
        G.add_edge(k, cycle[-1])

    # Recurse on the two sub-polygons.
    # cycle[:k_idx+1] has cycle[0]..cycle[k_idx] with the chord (cycle[0], k) closing it.
    # cycle[k_idx:]   has cycle[k_idx]..cycle[-1] with the chord (k, cycle[-1]) closing it.
    _triangulate(G, cycle[: k_idx + 1], rng)
    _triangulate(G, cycle[k_idx:], rng)


def build_outerplanar(
    n: int,
    chord_keep_prob: float = 0.4,
    seed: int | None = None,
) -> nx.Graph:
    """
    Return a random outerplanar graph on n nodes (n >= 3).

    Parameters
    ----------
    n : int
        Number of vertices.  Must be >= 3.
    chord_keep_prob : float
        Fraction of interior chord edges to retain after triangulation.
        1.0 = maximal outerplanar graph; lower values give sparser graphs.
        The outer Hamiltonian cycle is always kept (guarantees connectivity).
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    nx.Graph
    """
    if n < 3:
        raise ValueError(f"n must be >= 3, got {n}")
    if not (0.0 < chord_keep_prob <= 1.0):
        raise ValueError("chord_keep_prob must be in (0, 1]")

    log.debug(
        "build_outerplanar: n=%d  chord_keep_prob=%.2f  seed=%s",
        n, chord_keep_prob, seed,
    )

    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Outer Hamiltonian cycle — the backbone that guarantees connectivity
    for i in range(n):
        G.add_edge(i, (i + 1) % n)
    log.debug("  outer cycle added: %d edges", n)

    if n >= 4:
        # Triangulate the interior
        cycle = list(range(n))
        _triangulate(G, cycle, rng)
        log.debug("  triangulation complete: %d edges total", G.number_of_edges())

        # Optionally thin out interior chords (keep outer cycle intact)
        if chord_keep_prob < 1.0:
            outer_edges = {(i, (i + 1) % n) for i in range(n)}
            outer_edges |= {((i + 1) % n, i) for i in range(n)}
            chord_edges = [
                e for e in list(G.edges())
                if e not in outer_edges and (e[1], e[0]) not in outer_edges
            ]
            removed = 0
            for e in chord_edges:
                if rng.random() > chord_keep_prob:
                    G.remove_edge(*e)
                    removed += 1
            log.debug(
                "  chord thinning: removed %d / %d chords (keep_prob=%.2f)",
                removed, len(chord_edges), chord_keep_prob,
            )

    log.debug(
        "build_outerplanar done: %d nodes, %d edges",
        G.number_of_nodes(), G.number_of_edges(),
    )
    return G
