"""The learned-map settings chosen in Phase 4 (and possibly adjusted in Phase 6).

Stored in results/p4_choice.json. Every script that builds an adaptive map reads
them through here, so p5, export_for_colab, p6 and p7 always agree.
"""
import hashlib
import json
from pathlib import Path

# Canonical path to the stored strength-mapping hyperparameters
CHOICE = Path("results/p4_choice.json")

# Default hyperparameter values for adaptive strength mapping:
#   alpha : sensitivity exponent for predicted noise disturbance
#   beta  : sensitivity exponent for predicted erasure survival (0.0 = disabled)
#   shrink: pull-to-mean shrinkage factor (1.0 = full adaptation, 0.0 = uniform)
#   s_min : lower relative strength floor to prevent starved blocks
MAP_KEYS = {"alpha": 1.0, "beta": 0.0, "shrink": 1.0, "s_min": 0.25}


def load_choice(path=CHOICE):
    """The whole choice record ({} if Phase 4 has not run yet).

    Args:
        path (Path or str): Path to the JSON configuration file. Defaults to CHOICE.

    Returns:
        dict: Parsed parameter dictionary from disk, or an empty dict if the file does not exist.
    """
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def map_params(path=CHOICE, **overrides):
    """learned_map keyword arguments: stored values, defaults for anything missing.

    overrides with value None are ignored, so argparse defaults of None pass straight through.

    Args:
        path (Path or str): Location of the persisted choice JSON. Defaults to CHOICE.
        **overrides: Optional runtime keyword overrides (e.g., from CLI flags).

    Returns:
        dict[str, float]: Complete set of verified float hyperparameters for learned_map.
    """
    choice = load_choice(path)
    # Populate configuration with stored choices, falling back to canonical defaults
    params = {k: float(choice.get(k, d)) for k, d in MAP_KEYS.items()}
    # Apply explicit keyword overrides if provided
    params.update({k: float(v) for k, v in overrides.items() if v is not None})
    return params


def file_md5(path):
    """Short fingerprint of a file (e.g. the model), to detect stale results.

    Computes a 12-character MD5 hex digest to log model provenance and
    prevent downstream evaluation scripts from running with stale artifacts.

    Args:
        path (Path or str): Filepath to inspect.

    Returns:
        str or None: First 12 characters of MD5 digest, or None if the file is missing.
    """
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.md5(p.read_bytes()).hexdigest()[:12]
