"""Real AI edits made outside this repo (Stable Diffusion on Colab).

Folder layout (made by scripts/export_for_colab.py and colab/sd_edits.py):

    results/colab_export/                  what we send to Colab
        manifest.csv                        one row per exported image (bits, key, PSNR ...)
        train/<name>.png                    uniform watermark, key 1234 -> Phase 3 training targets
        test/<config>/<name>.png            each ablation config, key 2024 -> Phase 5 evaluation

    data/external/                          what comes back (copy the Colab "edited" folder here)
        <edit>/train/<name>.png
        <edit>/test/<config>/<name>.png

The older flat layout data/external/<edit>/<name>.png (from p2 --export-for-b)
is still read as training data.

Edit folders whose name contains inpaint / erase / crop are ERASURE edits
(they replace whole regions); everything else (vae, regen_*, style ...) is a
NOISE-type edit that disturbs every block a little.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import core
from .attacks import attack
from .config import PAYLOAD_BITS, TARGET_PSNR
from .embed import embed_at_psnr
from .extract import block_margins
from .features import block_features
from .predictor import ERASE_ATTACKS, NOISE_ATTACKS

# Keywords designating localized spatial erasure distortions rather than global perturbations
ERASE_WORDS = ("inpaint", "erase", "crop")
DEFAULT_EXPORT = Path("results/colab_export")
DEFAULT_EXTERNAL = Path("data/external")


def is_erase(edit_name):
    """Determines whether an edit operation is categorized as a spatial erasure.

    Args:
        edit_name (str): Identifier or folder name of the edit condition.

    Returns:
        bool: True if any keyword in ERASE_WORDS is present, False otherwise.
    """
    return any(w in edit_name.lower() for w in ERASE_WORDS)


def edit_names(root):
    """Edit folders under root (sorted), ignoring helper folders like masks/.

    Args:
        root (Path or str): Path to the directory containing external edit folders.

    Returns:
        list[str]: Sorted directory names corresponding to valid edit conditions.
    """
    root = Path(root)
    if not root.exists():
        return []
    # Filter out non-edit helper folders
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in {"masks", "logs"})


def train_edit_path(root, edit, name):
    """Edited training image, new layout first, then the old flat layout. None if missing.

    Args:
        root (Path or str): Base directory containing external edits.
        edit (str): Name of the edit condition.
        name (str): Base filename/identifier of the training image.

    Returns:
        Path or None: Resolved path to the edited training image, or None if absent.
    """
    # Probe hierarchical path (<edit>/train/<name>.png) then legacy flat path (<edit>/<name>.png)
    for p in (Path(root) / edit / "train" / f"{name}.png", Path(root) / edit / f"{name}.png"):
        if p.exists():
            return p
    return None


def test_edit_path(root, edit, config, name):
    """Locates an edited test image for a specific ablation configuration.

    Args:
        root (Path or str): Base directory containing external edits.
        edit (str): Name of the edit condition.
        config (str): Experimental watermarking configuration name (e.g., 'adaptive+ecc').
        name (str): Base filename/identifier of the test image.

    Returns:
        Path or None: Resolved path to the edited test image, or None if absent.
    """
    p = Path(root) / edit / "test" / config / f"{name}.png"
    return p if p.exists() else None


def external_train_versions(root, name):
    """(noise_imgs, erase_imgs, edit_names_found) for one training image.

    Scans the external directory to collect all available edited variants of a training image,
    segregating them into noise-like and erasure-like image arrays.

    Args:
        root (Path or str or None): Base external directory.
        name (str): Image stem identifier.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], list[str]]:
            - noise: Loaded image arrays for noise-type edits.
            - erase: Loaded image arrays for erasure-type edits.
            - found: List of edit condition names discovered.
    """
    noise, erase, found = [], [], []
    if root is None:
        return noise, erase, found
    # Iterate through all available edit conditions and load corresponding files
    for edit in edit_names(root):
        p = train_edit_path(root, edit, name)
        if p is not None:
            (erase if is_erase(edit) else noise).append(core.load_image(p))
            found.append(edit)
    return noise, erase, found


# ---------------- manifest ----------------
def bits_to_str(bits):
    """Converts a binary sequence into a compact string of digits ('0' and '1').

    Args:
        bits (array-like): Binary array.

    Returns:
        str: String representation of the bit sequence.
    """
    return "".join(str(int(b)) for b in bits)


def str_to_bits(s):
    """Parses a string of '0' and '1' characters into a 1-D numpy array of int64.

    Args:
        s (str): String containing binary characters.

    Returns:
        np.ndarray: 1-D array of binary bits (int64).
    """
    return np.array([int(c) for c in str(s)], dtype=np.int64)


def read_manifest(export_dir=DEFAULT_EXPORT):
    """Loads the export manifest CSV tracking image payloads, keys, and split metadata.

    Args:
        export_dir (Path or str, optional): Directory containing manifest.csv.
            Defaults to DEFAULT_EXPORT.

    Returns:
        pd.DataFrame: Manifest records with image, payload, and message loaded as strings.

    Raises:
        SystemExit: If the manifest CSV does not exist.
    """
    path = Path(export_dir) / "manifest.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/export_for_colab.py first.")
    # Ensure ID and bitstrings are parsed as raw strings to avoid numeric stripping
    return pd.read_csv(path, dtype={"image": str, "payload": str, "message": str})


# ---------------- survival targets with real edits ----------------
def survival_rows(img, image_id, ext_noise=(), ext_erase=(), ext_weight=0.5,
                  key=1234, seed=0, target_psnr=TARGET_PSNR, erase_repeats=2):
    """Like predictor.build_survival_rows, but real edits get a fixed share of the target.

    Pooling 3 real edits with 15-18 synthetic ones would let them barely move the
    target. Instead each target is  (1 - w) * synthetic + w * real  (w = ext_weight)
    when real edits exist for this image, and pure synthetic otherwise.
    The separate parts are kept as extra columns, so you can check how well the
    cheap synthetic attacks stand in for real diffusion edits.

    Args:
        img (np.ndarray): Pristine host image array (512x512 BGR uint8).
        image_id (str): Identifier name of the image.
        ext_noise (tuple or list, optional): Pre-loaded image arrays for real noise-type edits.
        ext_erase (tuple or list, optional): Pre-loaded image arrays for real erasure edits.
        ext_weight (float, optional): Linear interpolation weight allocated to real edits. Defaults to 0.5.
        key (int, optional): Fixed training key for uniform embedding. Defaults to 1234.
        seed (int, optional): Random seed for synthetic attacks. Defaults to 0.
        target_psnr (float, optional): Embedding target PSNR in dB. Defaults to TARGET_PSNR (42.0).
        erase_repeats (int, optional): Seed variations per synthetic erasure attack. Defaults to 2.

    Returns:
        pd.DataFrame: Block-level feature table containing 13 features along with synthetic,
                      external, and blended target labels (log_sigma and keep).
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, PAYLOAD_BITS)
    n_blocks = int(np.prod(core.grid_shape(img.shape)))

    # Embed uniform watermark calibrated to target PSNR
    wm, _, _ = embed_at_psnr(img, bits, np.ones(n_blocks), key, target_psnr)
    sign = 2.0 * bits[core.bit_assignment(n_blocks, PAYLOAD_BITS, key)] - 1.0
    d0 = block_margins(wm)

    def disturbance(imgs):
        """Computes mean absolute margin shift |Δ - Δ₀| across candidate images."""
        return np.mean([np.abs(block_margins(a) - d0) for a in imgs], axis=0)

    def survival(imgs):
        """Computes fraction of images where the block's margin preserves the correct sign."""
        return np.mean([(sign * block_margins(a) > 0) for a in imgs], axis=0)

    # Generate synthetic noise and erasure attack suites
    syn_noise = [attack(wm, k, s, seed=seed) for k, s in NOISE_ATTACKS]
    syn_erase = [attack(wm, k, s, seed=seed + r) for k, s in ERASE_ATTACKS for r in range(erase_repeats)]
    sig_syn, keep_syn = disturbance(syn_noise), survival(syn_erase)

    # Extract 13 statistical block features from the original unwatermarked image
    df = block_features(img)
    df["log_sigma_syn"] = np.log(sig_syn + 1e-3)
    df["keep_syn"] = keep_syn
    sig, keep = sig_syn, keep_syn

    # Blend external real generative noise edits into disturbance target
    if len(ext_noise):
        sig_ext = disturbance(ext_noise)
        df["log_sigma_ext"] = np.log(sig_ext + 1e-3)
        sig = (1 - ext_weight) * sig_syn + ext_weight * sig_ext
    else:
        df["log_sigma_ext"] = np.nan

    # Blend external real generative inpainting edits into survival target
    if len(ext_erase):
        keep_ext = survival(ext_erase)
        df["keep_ext"] = keep_ext
        keep = (1 - ext_weight) * keep_syn + ext_weight * keep_ext
    else:
        df["keep_ext"] = np.nan

    # Final combined regression targets
    df["log_sigma"] = np.log(sig + 1e-3)
    df["keep"] = keep
    df["n_external"] = len(ext_noise) + len(ext_erase)
    df["image_id"] = image_id
    df["block"] = np.arange(n_blocks)
    return df
