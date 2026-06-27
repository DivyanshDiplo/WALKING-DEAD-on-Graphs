"""
Matplotlib-based results visualizer for the Zombie Survivor Monte Carlo simulator.

Generates a single PNG report per session containing:
  1. Header bar — run configuration (graph size, zombies, trials).
  2. Summary stats table — one row per variation with all key metrics.
  3. Capture-rate bar chart — fraction of trials ending in zombie victory.
  4. Avg rounds bar chart — mean rounds-to-capture with std-dev error bars.
  5. Rounds distribution — overlapping histograms for each variation.

Files are saved to  walking_dead/tmp/  with a timestamp in the filename so
successive runs never overwrite each other.
"""

from __future__ import annotations

import logging
import pathlib
from datetime import datetime
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from .simulator import SimulationResult

log = logging.getLogger(__name__)

# Directory is  <package_root>/tmp/
_TMP_DIR = pathlib.Path(__file__).parent / "tmp"


# ---------------------------------------------------------------------------
# Colour palette (colourblind-friendly)
# ---------------------------------------------------------------------------

_PALETTE = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
]

_BG        = "#F8F8F8"
_GRID_COL  = "#DDDDDD"
_TEXT      = "#222222"
_HEADER_BG = "#2C3E50"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt(val: float | None, fmt: str = ".2f", fallback: str = "—") -> str:
    return f"{val:{fmt}}" if val is not None else fallback


def _bar_chart(
    ax: plt.Axes,
    labels: list[str],
    values: list[float],
    errors: list[float | None],
    title: str,
    ylabel: str,
    colours: list[str],
    y_max: float | None = None,
) -> None:
    """Draw a bar chart with optional error bars on the given Axes."""
    x = np.arange(len(labels))
    err = [e if e is not None else 0.0 for e in errors]
    has_err = any(e is not None for e in errors)

    bars = ax.bar(
        x, values, color=colours[:len(labels)],
        edgecolor="white", linewidth=0.8,
        yerr=err if has_err else None,
        capsize=5, error_kw={"elinewidth": 1.4, "ecolor": "#555555"},
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=_TEXT)
    ax.set_ylabel(ylabel, fontsize=9, color=_TEXT)
    ax.set_title(title, fontsize=10, fontweight="bold", color=_TEXT, pad=8)
    ax.set_facecolor(_BG)
    ax.yaxis.grid(True, color=_GRID_COL, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_COL)
    if y_max is not None:
        ax.set_ylim(0, y_max)

    # Value labels on top of bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (y_max or max(values, default=1)) * 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=8, color=_TEXT,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_report(
    run_config: dict,
    results: list[tuple[str, "SimulationResult"]],
    timestamp: str | None = None,
) -> pathlib.Path:
    """
    Generate and save a PNG report.

    Parameters
    ----------
    run_config : dict with keys  n, k, trials
    results    : list of (label, SimulationResult) in the order they were run
    timestamp  : optional shared timestamp string (YYYYMMDD_HHMMSS); generated
                 internally when not provided.

    Returns
    -------
    pathlib.Path — path of the saved PNG file
    """
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _TMP_DIR / f"report_{ts}.png"

    n_vars = len(results)
    colours = _PALETTE[:n_vars]
    short_labels = [_short_label(lbl) for lbl, _ in results]

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10), facecolor=_BG)
    fig.patch.set_facecolor(_BG)

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[0.18, 0.42, 0.40],
        hspace=0.52,
        wspace=0.35,
        top=0.93, bottom=0.07, left=0.07, right=0.97,
    )

    # ------------------------------------------------------------------
    # 1. Header
    # ------------------------------------------------------------------
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor(_HEADER_BG)
    ax_hdr.axis("off")
    ax_hdr.text(
        0.5, 0.65,
        "Zombie Survivor — Monte Carlo Simulation Report",
        ha="center", va="center",
        fontsize=15, fontweight="bold", color="white",
        transform=ax_hdr.transAxes,
    )
    config_str = (
        f"Graph size n={run_config['n']}   |   "
        f"Zombies k={run_config['k']}   |   "
        f"Trials={run_config['trials']}   |   "
        f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}  {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
    )
    ax_hdr.text(
        0.5, 0.22,
        config_str,
        ha="center", va="center",
        fontsize=9, color="#BBCCDD",
        transform=ax_hdr.transAxes,
    )

    # ------------------------------------------------------------------
    # 2. Stats table
    # ------------------------------------------------------------------
    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.axis("off")
    ax_tbl.set_title("Summary Statistics", fontsize=10, fontweight="bold",
                     color=_TEXT, pad=6, loc="left")

    col_headers = [
        "Variation",
        "Trials", "Captures", "Survivor Wins",
        "Capture Rate", "Avg Rounds", "Min", "Max", "Std-dev",
    ]
    rows = []
    for label, r in results:
        rows.append([
            label,
            str(r.trials),
            str(r.captures),
            str(r.survivor_wins),
            f"{r.capture_rate * 100:.1f}%",
            _fmt(r.avg_rounds),
            _fmt(r.min_rounds, fmt="d"),
            _fmt(r.max_rounds, fmt="d"),
            _fmt(r.std_rounds),
        ])

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.7)

    # Widen the Variation column relative to the others
    for i in range(len(rows) + 1):
        tbl[i, 0].set_width(0.22)
    for i in range(len(rows) + 1):
        for j in range(1, len(col_headers)):
            tbl[i, j].set_width(0.087)

    # Style header row
    for j in range(len(col_headers)):
        cell = tbl[0, j]
        cell.set_facecolor(_HEADER_BG)
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor(_GRID_COL)

    # Style data rows with alternating shades and colour-coded first column
    for i, (colour, _) in enumerate(zip(colours, results)):
        row_bg = "#FFFFFF" if i % 2 == 0 else "#F0F3F8"
        for j in range(len(col_headers)):
            cell = tbl[i + 1, j]
            cell.set_facecolor(row_bg)
            cell.set_edgecolor(_GRID_COL)
            cell.set_text_props(color=_TEXT)
        # Highlight the variation label cell with its series colour
        tbl[i + 1, 0].set_facecolor(colour + "33")   # 20% alpha
        tbl[i + 1, 0].set_text_props(color=colour, fontweight="bold")

    # ------------------------------------------------------------------
    # 3. Capture-rate bar chart
    # ------------------------------------------------------------------
    ax_cap = fig.add_subplot(gs[2, 0])
    cap_vals = [r.capture_rate * 100 for _, r in results]
    _bar_chart(
        ax_cap, short_labels, cap_vals,
        errors=[None] * n_vars,
        title="Capture Rate",
        ylabel="% of trials",
        colours=colours,
        y_max=110,
    )

    # ------------------------------------------------------------------
    # 4. Avg-rounds bar chart with std-dev error bars
    # ------------------------------------------------------------------
    ax_avg = fig.add_subplot(gs[2, 1])
    avg_vals = [r.avg_rounds if r.avg_rounds is not None else 0.0 for _, r in results]
    std_vals = [r.std_rounds for _, r in results]
    _bar_chart(
        ax_avg, short_labels, avg_vals,
        errors=std_vals,
        title="Avg Rounds to Capture  (± std-dev)",
        ylabel="rounds",
        colours=colours,
    )

    # ------------------------------------------------------------------
    # 5. Rounds distribution (histograms) — inset on avg chart if space,
    #    else skip silently when no variation has capture data.
    # ------------------------------------------------------------------
    has_data = any(len(r.capture_rounds) > 1 for _, r in results)
    if has_data:
        # Add a small inset axes inside ax_avg for the distribution
        ax_dist = ax_avg.inset_axes([0.0, -0.72, 1.0, 0.60])
        ax_dist.set_facecolor(_BG)
        ax_dist.yaxis.grid(True, color=_GRID_COL, linewidth=0.6, zorder=0)
        ax_dist.set_axisbelow(True)
        for spine in ax_dist.spines.values():
            spine.set_edgecolor(_GRID_COL)

        all_rounds = [r for _, res in results for r in res.capture_rounds]
        if all_rounds:
            bin_min = min(all_rounds)
            bin_max = max(all_rounds)
            bins = range(bin_min, bin_max + 2)

            for (label, r), colour in zip(results, colours):
                if r.capture_rounds:
                    ax_dist.hist(
                        r.capture_rounds, bins=bins,
                        alpha=0.55, color=colour, edgecolor="white",
                        linewidth=0.5, label=_short_label(label), zorder=3,
                    )

            ax_dist.set_title(
                "Rounds Distribution (captured trials)",
                fontsize=8.5, fontweight="bold", color=_TEXT, pad=4,
            )
            ax_dist.set_xlabel("rounds", fontsize=8, color=_TEXT)
            ax_dist.set_ylabel("frequency", fontsize=8, color=_TEXT)
            ax_dist.tick_params(colors=_TEXT, labelsize=7)
            ax_dist.legend(
                fontsize=7, framealpha=0.8,
                facecolor=_BG, edgecolor=_GRID_COL,
            )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)

    log.debug("visualizer: report saved to %s", out_path)
    return out_path


def _short_label(label: str) -> str:
    """Shorten 'Zombies=active, Survivor=lazy' -> 'Z:active / S:lazy'."""
    return (
        label
        .replace("Zombies=", "Z:")
        .replace("Survivor=", "S:")
        .replace(", ", " / ")
    )


# ---------------------------------------------------------------------------
# Graph visualizer
# ---------------------------------------------------------------------------

def save_graph(
    G: nx.Graph,
    run_config: dict,
    timestamp: str | None = None,
) -> pathlib.Path:
    """
    Draw the outerplanar game graph and save it as a PNG.

    Layout: circular (matches the outer Hamiltonian ring construction exactly).
    Outer cycle edges are drawn as solid coloured lines; interior chord edges
    are drawn as dashed grey lines, so the triangulation structure is clear.

    Parameters
    ----------
    G          : the outerplanar NetworkX graph.
    run_config : dict with keys  n, k, trials  (used for the title).
    timestamp  : optional shared timestamp string (YYYYMMDD_HHMMSS).

    Returns
    -------
    pathlib.Path — path of the saved PNG file.
    """
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _TMP_DIR / f"graph_{ts}.png"

    n = G.number_of_nodes()
    nodes = sorted(G.nodes())

    # Circular layout — aligns perfectly with the outer Hamiltonian ring
    pos = nx.circular_layout(G)

    # Classify edges: outer ring vs interior chords
    outer_set: set[tuple[int, int]] = set()
    for i in range(n):
        u, v = nodes[i], nodes[(i + 1) % n]
        outer_set.add((u, v))
        outer_set.add((v, u))

    outer_edges = [(u, v) for u, v in G.edges() if (u, v) in outer_set]
    chord_edges  = [(u, v) for u, v in G.edges() if (u, v) not in outer_set]

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=_BG)
    ax.set_facecolor(_BG)
    ax.axis("off")

    # Title
    ax.set_title(
        f"Outerplanar Game Graph\n"
        f"n={run_config['n']} nodes   |   "
        f"k={run_config['k']} zombie(s)   |   "
        f"{run_config['trials']} trial(s)   |   "
        f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}",
        fontsize=12, fontweight="bold", color=_TEXT, pad=14,
    )

    # Chord edges (drawn first so outer ring renders on top)
    if chord_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=chord_edges, ax=ax,
            edge_color="#AAAAAA", width=1.0,
            style="dashed", alpha=0.7,
        )

    # Outer cycle edges
    nx.draw_networkx_edges(
        G, pos, edgelist=outer_edges, ax=ax,
        edge_color=_PALETTE[0], width=2.4, alpha=0.9,
    )

    # Nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=_HEADER_BG, node_size=520, linewidths=1.5,
        edgecolors=_PALETTE[0],
    )

    # Labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_color="white", font_size=8, font_weight="bold",
    )

    # Legend
    outer_patch = mpatches.Patch(color=_PALETTE[0], label=f"Outer cycle ({len(outer_edges)} edges)")
    chord_patch  = mpatches.Patch(color="#AAAAAA",   label=f"Chords ({len(chord_edges)} edges)")
    ax.legend(
        handles=[outer_patch, chord_patch],
        loc="lower center", ncol=2,
        fontsize=9, framealpha=0.9,
        facecolor=_BG, edgecolor=_GRID_COL,
        bbox_to_anchor=(0.5, -0.04),
    )

    # Stats strip at bottom
    fig.text(
        0.5, 0.01,
        f"Nodes: {G.number_of_nodes()}   |   "
        f"Edges: {G.number_of_edges()}   |   "
        f"Outer cycle: {len(outer_edges)}   |   "
        f"Interior chords: {len(chord_edges)}",
        ha="center", fontsize=8, color="#666666",
    )

    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)

    log.debug("visualizer: graph saved to %s", out_path)
    return out_path
