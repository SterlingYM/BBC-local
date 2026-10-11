"""Per-process cache of prepared bias-correction simulations, for running many fits (e.g. many population cuts or data
samples) against the same simulations. Reading and deriving the simulations dominates the cost of a fit; a cached
BiasCorSims is reused whenever the files and the configuration items it depends on are the same.

Each entry holds all simulated rows (about 1 GB for ~4 million rows), so the cache keeps at most `maxsize` entries.
With multiprocessing, each worker process has its own cache; give each worker one model to avoid duplicate loads.
"""
from collections import OrderedDict

from .core import BiasCorSims, sim_key

_CACHE = OrderedDict()


def get_sims(paths, cfg, maxsize=1, **kw):
    """Return a BiasCorSims for these files and configuration, loading it only if it is not already cached."""
    key = (tuple(str(p) for p in paths), sim_key(cfg), tuple(sorted(kw.items())))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    while len(_CACHE) >= maxsize:
        _CACHE.popitem(last=False)
    S = BiasCorSims(paths, cfg, **kw)
    _CACHE[key] = S
    return S


def clear():
    _CACHE.clear()
