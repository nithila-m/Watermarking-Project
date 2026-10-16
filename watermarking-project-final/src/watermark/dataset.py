"""Where the images live: data/clean/{train,val,test}/<name>.png.

Every script goes through these helpers, so the folder layout is defined in one place.
"""
from pathlib import Path

from .config import DATA_DIR, SPLITS
from .core import load_image


def split_dir(split, root=DATA_DIR):
    """Return the filesystem directory path for a given dataset split.

    Args:
        split (str): Name of the dataset split (e.g., 'train', 'val', 'test').
        root (str or Path, optional): Base directory where clean dataset splits reside.
            Defaults to DATA_DIR ('data/clean').

    Returns:
        Path: Path object pointing to the specific split folder.

    Raises:
        ValueError: If split is not one of the recognized SPLITS.
    """
    # Validate split membership against known dataset partitions
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return Path(root) / split


def split_names(split, root=DATA_DIR):
    """Sorted image names (file stems) in one split. Empty list if the folder is missing.

    Scans the partition directory for '.png' images and returns an alphabetically
    ordered list of image stems without file extensions.

    Args:
        split (str): Target dataset split name.
        root (str or Path, optional): Dataset root directory. Defaults to DATA_DIR.

    Returns:
        list[str]: Sorted image stem identifiers, or an empty list if directory is absent.
    """
    d = split_dir(split, root)
    # Collect and sort image basenames for deterministic execution across splits
    return sorted(p.stem for p in d.glob("*.png")) if d.exists() else []


def image_path(split, name, root=DATA_DIR):
    """Construct the full filesystem path to an image given its split and stem name.

    Args:
        split (str): Name of the dataset split ('train', 'val', 'test').
        name (str): Stem identifier of the image file (excluding '.png').
        root (str or Path, optional): Dataset root directory. Defaults to DATA_DIR.

    Returns:
        Path: Full file path to the corresponding PNG image.
    """
    return split_dir(split, root) / f"{name}.png"


def load(split, name, root=DATA_DIR):
    """BGR uint8 image (grayscale stays 2-D).

    Resolves the image location via image_path and reads the array using core.load_image.

    Args:
        split (str): Dataset partition name.
        name (str): Image stem identifier.
        root (str or Path, optional): Dataset root directory. Defaults to DATA_DIR.

    Returns:
        np.ndarray: Loaded image array in BGR (or 2-D grayscale) uint8 format.
    """
    return load_image(image_path(split, name, root))


def require(split, root=DATA_DIR):
    """Names in a split, or a clear error telling you how to create it.

    Ensures that the requested split folder exists and is non-empty before downstream
    pipeline components proceed with processing.

    Args:
        split (str): Target dataset split name to verify.
        root (str or Path, optional): Dataset root directory. Defaults to DATA_DIR.

    Returns:
        list[str]: Non-empty list of sorted image stem names.

    Raises:
        SystemExit: If the directory does not exist or contains no PNG images.
    """
    names = split_names(split, root)
    # Guard against missing data partitions with an actionable error message
    if not names:
        raise SystemExit(f"no images in {split_dir(split, root)}. Run scripts/prepare_dataset.py first "
                         f"(--samples for a quick dev set, --src raw/val2017 for COCO).")
    return names
