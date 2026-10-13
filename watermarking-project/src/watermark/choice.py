"""The learned-map settings chosen in Phase 4 (and possibly adjusted in Phase 6).

Stored in results/p4_choice.json. Every script that builds an adaptive map reads
them through here, so p5, export_for_colab, p6 and p7 always agree.
"""
import hashlib
import json
from pathlib import Path

CHOICE = Path("results/p4_choice.json")
MAP_KEYS = {"alpha": 1.0, "beta": 0.0, "shrink": 1.0, "s_min": 0.25}


def load_choice(path=CHOICE):
    """The whole choice record ({} if Phase 4 has not run yet)."""
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def map_params(path=CHOICE, **overrides):
    """learned_map keyword arguments: stored values, defaults for anything missing.

    overrides with value None are ignored, so argparse defaults of None pass straight through.
    """
    choice = load_choice(path)
    params = {k: float(choice.get(k, d)) for k, d in MAP_KEYS.items()}
    params.update({k: float(v) for k, v in overrides.items() if v is not None})
    return params


def file_md5(path):
    """Short fingerprint of a file (e.g. the model), to detect stale results."""
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.md5(p.read_bytes()).hexdigest()[:12]
