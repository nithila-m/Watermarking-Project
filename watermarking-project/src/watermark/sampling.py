"""Choose which images a run uses: reproducible, balanced across texture types, and NESTED.

Nested means a larger n always contains the images of a smaller n (same seed), so
you can start Phase 5 with --n 150 and later run --n 300: the first 150 are reused
from the saved results instead of being recomputed.
"""
import zlib

import numpy as np


def _order(names, seed, tag=""):
    """A fixed shuffled order of `names` that does not depend on n."""
    rng = np.random.default_rng([seed, zlib.crc32(tag.encode())])
    return [names[i] for i in rng.permutation(len(names))]


def sample(names, n, seed=0):
    """n names chosen at random (sorted). n <= 0 or n >= len(names) returns all."""
    names = sorted(names)
    if n <= 0 or n >= len(names):
        return names
    return sorted(_order(names, seed)[:n])


def stratified(names, n, category_of, seed=0):
    """About n names with an equal share from each category (e.g. Smooth/Medium/Complex).

    category_of: dict-like name -> category. Names without a category form the
    group "unknown". If a category runs short, the rest is filled from the others
    so the total is still n.
    """
    names = sorted(names)
    if n <= 0 or n >= len(names):
        return names
    groups = {}
    for nm in names:
        groups.setdefault(str(category_of.get(nm, "unknown")), []).append(nm)
    ordered = {g: _order(m, seed, g) for g, m in groups.items()}
    per = n // len(groups)
    chosen = [nm for g in sorted(ordered) for nm in ordered[g][:per]]
    if len(chosen) < n:  # fill evenly from what is left, round-robin over categories
        leftovers = {g: ordered[g][per:] for g in sorted(ordered)}
        i = 0
        while len(chosen) < n and any(leftovers.values()):
            g = sorted(leftovers)[i % len(leftovers)]
            if leftovers[g]:
                chosen.append(leftovers[g].pop(0))
            i += 1
    return sorted(chosen)
