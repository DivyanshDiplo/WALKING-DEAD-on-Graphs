"""
Paper's 2-lazy-zombie algorithm for connected outerplanar graphs.

Implements Theorem 6 / Corollary 1 of:
  Bose, De Carufel & Shermer (2022)
  "Pursuit-Evasion in Graphs: Zombies, Lazy Zombies and a Survivor"

The key theoretical result replicated here:
  uL(G) = 2  for every connected outerplanar graph G.
  Two lazy zombies, starting anywhere, always catch the survivor in < 2n rounds.

Algorithm overview
------------------
The outer Hamiltonian cycle 0-1-2-..-(n-1)-0 is the spine of the strategy.
Edges not on the cycle are called *chords*.

Phase 1 – APPROACH
  Both zombies navigate (via BFS) to the two endpoints of the *best guard chord*
  — the chord bi-bj whose territory arc (the arc between bi and bj containing the
  survivor) is as small as possible.  The guard zombie goes to bi, the advance
  zombie goes to bj.  Each zombie moves one BFS step per round while the survivor
  also moves.

  On a pure cycle (no chords) no navigation is needed: the zombies can start the
  circuit phase from wherever they are.

Phase 2 – GUARD / ADVANCE (Theorem 6)
  Guard (z1) sits at bi, stationary.  Advance (z2) walks the outer circuit toward
  bi, entering the survivor territory one vertex per round.

  At each position of z2 the algorithm checks for chords into the territory:
    • If no chord from z2 into territory → take one circuit step (shrink territory).
    • If chord bk-bℓ exists (bk inside territory, bℓ = z2's position) and the
      survivor is strictly between bk and bℓ → ROLE SWAP: z2 becomes the new
      stationary guard; old guard (z1) navigates to bk (Phase 3), then resumes.
    • Otherwise → continue the normal circuit step toward guard.

  The outerplanarity guarantee ensures no chord crosses the guarded chord bi-bj,
  so the survivor cannot escape from the territory.

Phase 3 – NAVIGATE (post-swap)
  The new advance zombie navigates to bk via BFS.  Once it arrives the invariant
  is restored and Phase 2 resumes.

Phase 4 – RESTART
  If the territory is ever empty yet the survivor has not been caught (can happen
  when the survivor escapes during the approach phase before the chord is set),
  the algorithm picks a new best chord for the survivor's current position and
  both zombies navigate there again.  This ensures eventual capture.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

import networkx as nx

from .game import GameResult
from .strategies import _dist_from, place_survivor, place_zombies, survivor_step

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph structure helpers
# ---------------------------------------------------------------------------

def _build_chord_adj(G: nx.Graph, n: int) -> dict[int, list[int]]:
    """Map each vertex to its chord-neighbours (non-cycle neighbours)."""
    cycle_edges = {frozenset([i, (i + 1) % n]) for i in range(n)}
    adj: dict[int, list[int]] = defaultdict(list)
    for u, v in G.edges():
        if frozenset([u, v]) not in cycle_edges:
            adj[u].append(v)
            adj[v].append(u)
    return dict(adj)


def _cw_arc(start: int, end: int, n: int) -> set[int]:
    """Strictly interior vertices of the clockwise arc from *start* to *end*."""
    verts: set[int] = set()
    v = (start + 1) % n
    for _ in range(n):
        if v == end:
            break
        verts.add(v)
        v = (v + 1) % n
    return verts


# ---------------------------------------------------------------------------
# Chord selection helpers
# ---------------------------------------------------------------------------

def _find_best_guard_chord(
    chord_adj: dict[int, list[int]],
    survivor_pos: int,
    n: int,
    G: nx.Graph | None = None,
) -> tuple[int, int, set[int]] | None:
    """
    Find the chord bi-bj that minimises |territory|, where territory is the
    arc between bi and bj that contains the survivor.

    If *G* is provided, also verifies Theorem 6's invariant condition #3:
    the circuit step from the advance vertex (bj) into the territory must
    reduce the BFS distance to the survivor.  Chords that violate this are
    deprioritised (used as fallback only).

    Returns (guard_vertex=bi, advance_vertex=bj, territory_set) or None if no
    chord contains the survivor in one of its arcs.
    """
    best_valid: tuple[int, int, set[int]] | None = None
    best_valid_size = n + 1
    best_fallback: tuple[int, int, set[int]] | None = None
    best_fallback_size = n + 1

    seen: set[tuple[int, int]] = set()

    for u in chord_adj:
        for v in chord_adj[u]:
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)

            arc_uv = _cw_arc(u, v, n)
            arc_vu = _cw_arc(v, u, n)

            for bi, bj, arc in ((u, v, arc_uv), (v, u, arc_vu)):
                if survivor_pos not in arc:
                    continue
                sz = len(arc)

                if G is not None:
                    # Check condition #3: circuit step from bj must reduce
                    # BFS distance to the survivor.
                    nxt = _circuit_step(bj, arc, n)
                    if nxt is not None:
                        d_bj = nx.shortest_path_length(G, bj, survivor_pos)
                        d_nxt = nx.shortest_path_length(G, nxt, survivor_pos)
                        if d_nxt < d_bj and sz < best_valid_size:
                            best_valid_size = sz
                            best_valid = (bi, bj, arc)
                            continue
                # Fallback: no G check, or condition #3 not satisfied
                if sz < best_fallback_size:
                    best_fallback_size = sz
                    best_fallback = (bi, bj, arc)

    return best_valid if best_valid is not None else best_fallback


# ---------------------------------------------------------------------------
# Circuit helpers
# ---------------------------------------------------------------------------

def _circuit_step(advance: int, territory: set[int], n: int) -> int | None:
    """
    Next outer-circuit step: return the cycle-neighbour of *advance* that lies
    inside the territory (or None if territory is empty / no such neighbour).
    """
    for nb in ((advance - 1) % n, (advance + 1) % n):
        if nb in territory:
            return nb
    return None


def _bfs_step(G: nx.Graph, src: int, dst: int) -> int:
    """One BFS step from *src* toward *dst*; returns *src* if already there."""
    if src == dst:
        return src
    try:
        return nx.shortest_path(G, src, dst)[1]
    except nx.NetworkXNoPath:
        return src


# ---------------------------------------------------------------------------
# Main game function
# ---------------------------------------------------------------------------

def play_paper_game(
    G: nx.Graph,
    survivor_lazy: bool,
    rng: random.Random,
    max_rounds: int | None = None,
) -> GameResult:
    """
    Play one game using the paper's 2-lazy-zombie algorithm (Theorem 6 /
    Corollary 1).

    Parameters
    ----------
    G             : connected outerplanar graph with outer Hamiltonian cycle 0..n-1.
    survivor_lazy : True = lazy survivor; False = active survivor.
    rng           : seeded RNG for initial zombie placement only.
    max_rounds    : safety cap (default max(5000, 20·n)).

    Returns
    -------
    GameResult — should be "capture" on all valid outerplanar graphs.
    """
    n = len(G)
    if max_rounds is None:
        max_rounds = max(5000, 20 * n)

    chord_adj = _build_chord_adj(G, n)

    # ------------------------------------------------------------------
    # Initial placement
    # ------------------------------------------------------------------
    z_init = place_zombies(G, 2, rng)
    survivor_pos = place_survivor(G, z_init)

    if survivor_pos in z_init:
        log.debug("paper_game: capture at placement")
        return GameResult("capture", 0)

    z: list[int] = list(z_init)

    # ------------------------------------------------------------------
    # Helper: set up a fresh guard-advance configuration.
    # Returns (guard, advance, territory, phase, nav_target, approach_targets)
    # If no chord seals the survivor, returns guard/advance from _assign_roles
    # with an "open" territory (not chord-bounded).
    # ------------------------------------------------------------------
    def _setup(surv: int, z0: int, z1: int):
        """
        Pick the best guard chord and assign zombie targets.
        Returns (phase, guard, advance, territory, nav_target, targets)
        where *targets* is (target_for_z0, target_for_z1) used in 'approach'.
        """
        if not chord_adj:
            # Pure cycle: assign directly
            g, a, terr = _assign_roles_simple(z0, z1, surv, n)
            return "circuit", g, a, terr, -1, None

        best = _find_best_guard_chord(chord_adj, surv, n, G)
        if best is None:
            # Survivor not in any chord arc (rare); fall back to direct assignment
            g, a, terr = _assign_roles_simple(z0, z1, surv, n)
            return "circuit", g, a, terr, -1, None

        bi, bj, terr = best
        # Assign zombies to bi/bj to minimise total travel
        d00 = (nx.shortest_path_length(G, z0, bi)
               + nx.shortest_path_length(G, z1, bj))
        d01 = (nx.shortest_path_length(G, z0, bj)
               + nx.shortest_path_length(G, z1, bi))
        if d00 <= d01:
            tgts = (bi, bj)   # z[0]→guard_target=bi, z[1]→advance_target=bj
            guard_out, advance_out = bi, bj
        else:
            tgts = (bj, bi)   # z[0]→advance_target=bj, z[1]→guard_target=bi
            guard_out, advance_out = bi, bj  # final positions once reached

        log.debug(
            "setup: chord (%d,%d)  bi=%d bj=%d  |territory|=%d  z=(%d,%d)",
            bi, bj, bi, bj, len(terr), z0, z1,
        )
        return "approach", guard_out, advance_out, terr, -1, tgts

    phase, guard, advance, territory, nav_target, approach_targets = _setup(
        survivor_pos, z[0], z[1]
    )

    visited: set[tuple] = set()

    for rnd in range(1, max_rounds + 1):

        # ==============================================================
        # ZOMBIE TURN
        # ==============================================================

        if phase == "approach":
            assert approach_targets is not None
            tgt0, tgt1 = approach_targets
            if z[0] != tgt0:
                z[0] = _bfs_step(G, z[0], tgt0)
            if z[1] != tgt1:
                z[1] = _bfs_step(G, z[1], tgt1)

            if z[0] == tgt0 and z[1] == tgt1:
                # Both at chord endpoints; establish guard/advance
                # guard_target was assigned to whoever reaches the guard side
                if tgt0 == guard:
                    guard, advance = z[0], z[1]
                else:
                    guard, advance = z[1], z[0]
                # Territory might have shifted if survivor moved; recompute
                # around the actual chord that is now guarded.
                territory = _recompute_territory(guard, advance, survivor_pos, n,
                                                 chord_adj)
                phase = "circuit"
                log.debug(
                    "rnd%d approach done: guard=%d advance=%d |territory|=%d",
                    rnd, guard, advance, len(territory),
                )

        elif phase == "navigate":
            old = advance
            advance = _bfs_step(G, advance, nav_target)
            territory.discard(advance)
            log.debug(
                "rnd%d navigate: %d -> %d (target=%d)", rnd, old, advance, nav_target
            )
            if advance == nav_target:
                phase = "circuit"
                log.debug("rnd%d reached bk=%d -> circuit", rnd, nav_target)

        elif phase == "circuit":
            chords_in_terr = [
                w for w in chord_adj.get(advance, []) if w in territory
            ]

            if chords_in_terr:
                # bk = chord endpoint in territory *closest clockwise to guard*
                bk = min(chords_in_terr, key=lambda w: (w - guard) % n)
                arc_bk_adv = _cw_arc(bk, advance, n)

                # Role swap only when survivor is STRICTLY between bk and advance.
                # (If survivor == bk, the advance will naturally reach and capture it.)
                if survivor_pos in arc_bk_adv:
                    log.debug(
                        "rnd%d ROLE SWAP: advance=%d bk=%d survivor=%d",
                        rnd, advance, bk, survivor_pos,
                    )
                    old_guard = guard
                    guard    = advance       # stays → new stationary guard
                    advance  = old_guard     # old guard navigates to bk
                    territory = set(arc_bk_adv)
                    nav_target = bk
                    phase = "navigate"
                else:
                    # Normal circuit step (also handles survivor==bk case)
                    nxt = _circuit_step(advance, territory, n)
                    if nxt is not None:
                        territory.discard(nxt)
                        advance = nxt
                        log.debug(
                            "rnd%d circuit (chord, no-swap): adv->%d |T|=%d",
                            rnd, advance, len(territory),
                        )
                    else:
                        _handle_empty_territory(rnd, G, z, guard, advance,
                                                survivor_pos, chord_adj, n)
                        # Restart; clear visited to avoid false stall detection
                        phase, guard, advance, territory, nav_target, approach_targets = \
                            _setup(survivor_pos, guard, advance)
                        z[0], z[1] = guard, advance
                        visited.clear()
            else:
                nxt = _circuit_step(advance, territory, n)
                if nxt is not None:
                    territory.discard(nxt)
                    advance = nxt
                    log.debug(
                        "rnd%d circuit: adv->%d |T|=%d", rnd, advance, len(territory)
                    )
                else:
                    log.debug("rnd%d circuit: territory empty, restarting", rnd)
                    phase, guard, advance, territory, nav_target, approach_targets = \
                        _setup(survivor_pos, guard, advance)
                    z[0], z[1] = guard, advance
                    visited.clear()

        # ------ Capture check ------
        z_all = z if phase == "approach" else [guard, advance]
        if survivor_pos in z_all:
            log.debug("rnd%d CAPTURE (zombie move)", rnd)
            return GameResult("capture", rnd)

        # ==============================================================
        # SURVIVOR TURN
        # ==============================================================
        dist_cache = {zv: _dist_from(G, zv) for zv in z_all}
        survivor_pos = survivor_step(G, survivor_pos, z_all, survivor_lazy, dist_cache)

        if survivor_pos in z_all:
            log.debug("rnd%d CAPTURE (survivor stepped onto zombie)", rnd)
            return GameResult("capture", rnd)

        # ------ Stall / cycle detection ------
        if phase == "approach":
            state: tuple = (survivor_pos, z[0], z[1], phase)
        else:
            state = (survivor_pos, guard, advance, phase, nav_target, len(territory))
        if state in visited:
            log.debug("rnd%d STALL -> survivor wins", rnd)
            return GameResult("survivor_win", rnd)
        visited.add(state)

    log.debug("max rounds %d -> survivor wins", max_rounds)
    return GameResult("survivor_win", max_rounds)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assign_roles_simple(
    z0: int, z1: int, survivor: int, n: int
) -> tuple[int, int, set[int]]:
    """
    Assign guard / advance using whichever cycle arc contains the survivor.
    Used when no suitable guard chord is available.
    """
    arc0 = _cw_arc(z0, z1, n)
    if survivor in arc0:
        return z0, z1, arc0
    arc1 = _cw_arc(z1, z0, n)
    if survivor in arc1:
        return z1, z0, arc1
    return z0, z1, arc0   # fallback (survivor at z0 or z1)


def _recompute_territory(
    guard: int,
    advance: int,
    survivor_pos: int,
    n: int,
    chord_adj: dict[int, list[int]],
) -> set[int]:
    """
    After both zombies arrive at their chord endpoints, recompute the territory
    as the cycle arc between guard and advance that contains the survivor.
    If survivor has left both arcs, return the smaller arc.
    """
    arc_ga = _cw_arc(guard, advance, n)
    arc_ag = _cw_arc(advance, guard, n)
    if survivor_pos in arc_ga:
        return arc_ga
    if survivor_pos in arc_ag:
        # Survivor is in the zombie-territory arc.  Swap guard/advance so the
        # territory still contains the survivor.
        return arc_ag
    # Survivor at guard or advance — territory is whichever arc is smaller
    return arc_ga if len(arc_ga) <= len(arc_ag) else arc_ag


def _handle_empty_territory(
    rnd: int,
    G: nx.Graph,
    z: list[int],
    guard: int,
    advance: int,
    survivor_pos: int,
    chord_adj: dict[int, list[int]],
    n: int,
) -> None:
    """Log a message when territory has been exhausted without capture."""
    log.debug(
        "rnd%d territory exhausted: guard=%d advance=%d survivor=%d -> restart",
        rnd, guard, advance, survivor_pos,
    )
