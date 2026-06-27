"""
Centralized logger configuration for the walking_dead package.

All modules use `logging.getLogger(__name__)` which creates loggers under the
`walking_dead.*` hierarchy.  Configuring the root `walking_dead` logger here
propagates settings to every child logger automatically.

Usage
-----
Normal run (WARNING+ only):
    python -m walking_dead

Debug run (every DEBUG message visible):
    python -m walking_dead --debug

The `configure()` function is called once from `cli.main()` before any
simulation work begins.
"""

import logging
import sys

_ROOT = "walking_dead"
_FMT  = "[%(levelname)-8s] %(name)s: %(message)s"


def configure(debug: bool = False) -> None:
    """
    Set up a StreamHandler on the root walking_dead logger.

    Safe to call multiple times — subsequent calls are no-ops once a handler
    has been attached.

    Parameters
    ----------
    debug : bool
        True  -> set level to DEBUG (verbose per-round output).
        False -> set level to WARNING (silent during normal use).
    """
    root = logging.getLogger(_ROOT)
    if root.handlers:
        return  # already configured; avoid duplicate handlers

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT))
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if debug else logging.WARNING)
