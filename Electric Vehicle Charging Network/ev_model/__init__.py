"""ev_model package
Expose package submodules for convenient imports.

This file intentionally keeps package exports minimal and deterministic: it
imports the submodules so users can do e.g. ``from ev_model import graphs``.
Avoid executing this file directly; import the package instead.
"""

from . import config, utils, graphs, entities, routing, simulation, viz  # noqa: F401

__all__ = [
    "config",
    "utils",
    "graphs",
    "entities",
    "routing",
    "simulation",
    "viz",
]
