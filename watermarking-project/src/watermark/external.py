"""Real AI edits made outside this repo (Stable Diffusion on Colab).

Folder layout (made by scripts/export_for_colab.py and colab/sd_edits.py):

    results/colab_export/                  what we send to Colab
        manifest.csv                       one row per exported image (bits, key, PSNR ...)
        train/<name>.png                   uniform watermark, key 1234 -> Phase 3 training targets
        test/<config>/<name>.png           each ablation config, key 2024 -> Phase 5 evaluation

    data/external/                         what comes back (copy the Colab "edited" folder here)
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

ERASE_WORDS = ("inpaint", "erase", "crop")
DEFAULT_EXPORT = Path("results/colab_export")
DEFAULT_EXTERNAL = Path("data/external")


def is_erase(edit_name):
    return any(w in edit_name.lower() for w in ERASE_WORDS)


def edit_names(root):
    """Edit folders under root (sorted), ignoring helper folders like masks/."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in {"masks", "logs"})


def train_edit_path(root, edit, name):
    """Edited training image, new layout first, then the old flat layout. None if missing."""
    for p in (Path(root) / edit / "train" / f"{name}.png", Path(root) / edit / f"{name}.png"):
        if p.exists():
            return p
    return None


def test_edit_path(root, edit, config, name):
    p = Path(root) / edit / "test" / config / f"{name}.png"
    return p if p.exists() else None


def external_train_versions(root, name):
    """(noise_imgs, erase_imgs, edit_names_found) for one training image."""
    noise, erase, found = [], [], []
    if root is None:
        return noise, erase, found
    for edit in edit_names(root):
        p = train_edit_path(root, edit, name)
        if p is not None:
            (erase if is_erase(edit) else noise).append(core.load_image(p))
            found.append(edit)
    return noise, erase, found


# ---------------- manifest ----------------
def bits_to_str(bits):
    return "".join(str(int(b)) for b in bits)


def str_to_bits(s):
    return np.array([int(c) for c in str(s)], dtype=np.int64)


def read_manifest(export_dir=DEFAULT_EXPORT):
    path = Path(export_dir) / "manifest.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/export_for_colab.py first.")
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
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, PAYLOAD_BITS)
    n_blocks = int(np.prod(core.grid_shape(img.shape)))
    wm, _, _ = embed_at_psnr(img, bits, np.ones(n_blocks), key, target_psnr)
    sign = 2.0 * bits[core.bit_assignment(n_blocks, PAYLOAD_BITS, key)] - 1.0
    d0 = block_margins(wm)

    def disturbance(imgs):
        return np.mean([np.abs(block_margins(a) - d0) for a in imgs], axis=0)

    def survival(imgs):
        return np.mean([(sign * block_margins(a) > 0) for a in imgs], axis=0)

    syn_noise = [attack(wm, k, s, seed=seed) for k, s in NOISE_ATTACKS]
    syn_erase = [attack(wm, k, s, seed=seed + r) for k, s in ERASE_ATTACKS for r in range(erase_repeats)]
    sig_syn, keep_syn = disturbance(syn_noise), survival(syn_erase)

    df = block_features(img)
    df["log_sigma_syn"] = np.log(sig_syn + 1e-3)
    df["keep_syn"] = keep_syn
    sig, keep = sig_syn, keep_syn
    if len(ext_noise):
        sig_ext = disturbance(ext_noise)
        df["log_sigma_ext"] = np.log(sig_ext + 1e-3)
        sig = (1 - ext_weight) * sig_syn + ext_weight * sig_ext
    else:
        df["log_sigma_ext"] = np.nan
    if len(ext_erase):
        keep_ext = survival(ext_erase)
        df["keep_ext"] = keep_ext
        keep = (1 - ext_weight) * keep_syn + ext_weight * keep_ext
    else:
        df["keep_ext"] = np.nan
    df["log_sigma"] = np.log(sig + 1e-3)
    df["keep"] = keep
    df["n_external"] = len(ext_noise) + len(ext_erase)
    df["image_id"] = image_id
    df["block"] = np.arange(n_blocks)
    return df
