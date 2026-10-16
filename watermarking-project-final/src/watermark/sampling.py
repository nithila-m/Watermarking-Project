"""Choose which images a run uses: reproducible, balanced across texture types, and NESTED.

Nested means a larger n always contains the images of a smaller n (same seed), so
you can start Phase 5 with --n 150 and later run --n 300: the first 150 are reused
from the saved results instead of being recomputed.
"""
import zlib

import numpy as np


def _order(names, seed, tag=""):
    """Generate a fixed, pseudo-random permutation of `names` independent of subset size n.

    Seeds NumPy's default_rng with a composite seed tuple incorporating both the numeric
    seed and the CRC32 hash of the optional tag (such as category name). This guarantees
    that prefix slices [:n] are strictly nested as n scales upward.

    Args:
        names (list[str]): List of image identifiers to permute.
        seed (int): Base random seed for reproducibility.
        tag (str, optional): Stratification or domain label used to diversify RNG streams.
            Defaults to "".

    Returns:
        list[str]: Deterministically shuffled list containing all elements of `names`.
    """
    rng = np.random.default_rng([seed, zlib.crc32(tag.encode())])
    return [names[i] for i in rng.permutation(len(names))]


def sample(names, n, seed=0):
    """Select n names at random while preserving nested ordering across sample sizes.

    Sorting names prior to shuffling guarantees determinism regardless of input order.
    The resulting sampled subset is returned in sorted order for consistent iteration.

    Args:
        names (iterable[str]): Available pool of image stem identifiers.
        n (int): Target number of images to draw. If n <= 0 or n >= len(names),
            all names are returned sorted.
        seed (int, optional): Deterministic random seed. Defaults to 0.

    Returns:
        list[str]: Sorted list of n selected image names.
    """
    names = sorted(names)
    if n <= 0 or n >= len(names):
        return names
    # Extract the first n elements from the stable permutation
    return sorted(_order(names, seed)[:n])


def stratified(names, n, category_of, seed=0):
    """Draw about n names with balanced representation across texture categories.

    Partitions names into groups (e.g. 'Smooth', 'Medium', 'Complex', or 'unknown'),
    applies independent deterministic permutations to each group, and extracts an equal
    base quota (n // num_groups) from each category. Any remaining deficit is allocated
    via round-robin selection across available category pools.

    Args:
        names (iterable[str]): Candidate image identifiers.
        n (int): Desired total sample count. If n <= 0 or n >= len(names),
            returns all candidate names sorted.
        category_of (Mapping or pd.Series): Mapping from image name to its texture label.
            Images absent from the mapping default to 'unknown'.
        seed (int, optional): Deterministic random seed. Defaults to 0.

    Returns:
        list[str]: Sorted list of n stratified image names.
    """
    names = sorted(names)
    if n <= 0 or n >= len(names):
        return names

    # Partition candidate names into distinct category lists
    groups = {}
    for nm in names:
        groups.setdefault(str(category_of.get(nm, "unknown")), []).append(nm)

    # Permute each category group independently using group name as RNG salt tag
    ordered = {g: _order(m, seed, g) for g, m in groups.items()}
    per = n // len(groups)
    chosen = [nm for g in sorted(ordered) for nm in ordered[g][:per]]

    # Fill remaining quota up to n via round-robin traversal over non-empty categories
    if len(chosen) < n:  # fill evenly from what is left, round-robin over categories
        leftovers = {g: ordered[g][per:] for g in sorted(ordered)}
        i = 0
        while len(chosen) < n and any(leftovers.values()):
            g = sorted(leftovers)[i % len(leftovers)]
            if leftovers[g]:
                chosen.append(leftovers[g].pop(0))
            i += 1

    return sorted(chosen)
