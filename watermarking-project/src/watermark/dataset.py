"""Where the images live: data/clean/{train,val,test}/<name>.png.

Every script goes through these helpers, so the folder layout is defined in one place.
"""
from pathlib import Path

from .config import DATA_DIR, SPLITS
from .core import load_image


def split_dir(split, root=DATA_DIR):
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return Path(root) / split


def split_names(split, root=DATA_DIR):
    """Sorted image names (file stems) in one split. Empty list if the folder is missing."""
    d = split_dir(split, root)
    return sorted(p.stem for p in d.glob("*.png")) if d.exists() else []


def image_path(split, name, root=DATA_DIR):
    return split_dir(split, root) / f"{name}.png"


def load(split, name, root=DATA_DIR):
    """BGR uint8 image (grayscale stays 2-D)."""
    return load_image(image_path(split, name, root))


def require(split, root=DATA_DIR):
    """Names in a split, or a clear error telling you how to create it."""
    names = split_names(split, root)
    if not names:
        raise SystemExit(f"no images in {split_dir(split, root)}. Run scripts/prepare_dataset.py first "
                         f"(--samples for a quick dev set, --src raw/val2017 for COCO).")
    return names
