"""Phase 6 helpers: rebuild one Phase 5 case exactly and look inside it.

A "case" is (image, attack, severity). reproduce() repeats what evaluate_image did
for one config (same payload rule, key, map settings and attack seed), then
records WHERE the watermark died: which blocks voted the wrong way and which of
the 64 bits came out wrong. case_job() does this for two configs side by side,
saves a picture, and returns numbers for the summary.
"""
import zlib
from pathlib import Path

import numpy as np

from . import dataset, ecc
from .config import BLOCK, MESSAGE_BITS, PAYLOAD_BITS
from .core import grid_shape
from .embed import embed, embed_at_psnr
from .evaluate import CONFIGS, apply_attack, make_map
from .extract import extract
from .jobs import _model

# Relative strength threshold: blocks below 75% of the image mean are considered starved
STARVED = 0.75   # a block with relative strength below this got less than its share


def reproduce(split, name, config, model_path, params, kind, severity, key=2024, seed=0):
    """Rebuilds a single test trial exactly, collecting granular block and bit error telemetry.

    Reruns the deterministic pipeline for one image, configuration, and attack condition:
      1. Generates identical payloads via seeded pseudo-random bitstreams.
      2. Computes the strength map and scales to target PSNR via binary search.
      3. Records blocks skipped due to host coefficient opposition.
      4. Distorts the image and extracts candidate bits with tanh soft-voting.
      5. Isolates erroneous spatial blocks, misdecoded codeword bits, and per-bit margin sums.

    Args:
        split (str): Dataset split ('train', 'val', 'test').
        name (str): Image stem identifier.
        config (str): Experimental configuration name (e.g., 'adaptive+ecc', 'uniform+ecc').
        model_path (str or Path): Filepath to the serialized survival regressor.
        params (dict): Hyperparameters for learned adaptive strength mapping.
        kind (str): Attack type identifier.
        severity (str): Attack severity tier ('light', 'moderate', 'severe').
        key (int, optional): Pseudo-random block scattering key. Defaults to 2024.
        seed (int, optional): Deterministic attack seed. Defaults to 0.

    Returns:
        dict: Detailed internal case state including host and attacked image arrays,
              relative strength map, skipped block mask, wrong block mask, wrong bit mask,
              normalized vote totals, and exact recovery success flag.
    """
    img = dataset.load(split, name)
    # Mirror evaluate_image payload generation using CRC32 checksum of image name
    rng = np.random.default_rng(zlib.crc32(name.encode()))     # same rule as evaluate_image
    raw_bits = rng.integers(0, 2, PAYLOAD_BITS)
    msg_bits = rng.integers(0, 2, MESSAGE_BITS)
    cfg = CONFIGS[config]

    # Assign payload and message sequences based on error-correction mode
    msg, payload = (msg_bits, ecc.encode(msg_bits)) if cfg["ecc"] else (raw_bits, raw_bits)
    kw = params if cfg["map"] == "learned" else {}
    base = make_map(img, cfg["map"], _model(model_path) if cfg["map"] == "learned" else None, **kw)

    # Calibrate to target PSNR and track skipped blocks
    wm, s, _ = embed_at_psnr(img, payload, base, key)
    _, skipped = embed(img, payload, s, key, return_skipped=True)

    # Apply identical distortion operation
    att = wm if kind == "none" else apply_attack(wm, kind, severity, seed)
    bits, det = extract(att, PAYLOAD_BITS, key, return_details=True)

    # Evaluate exact payload recovery
    if cfg["ecc"]:
        dec, ok = ecc.decode(bits)
        success = bool(ok and np.array_equal(dec, msg))
    else:
        success = bool(np.array_equal(bits, payload))

    # Compute expected sign per block: +1.0 for bit 1, -1.0 for bit 0
    sign = 2.0 * payload[det["assign"]] - 1.0
    return {
        "img": img,
        "att": att,
        "rel": s / s.mean(),
        "skipped": skipped,
        # A block voted wrongly if its signed margin opposes the intended bit sign
        "wrong_block": sign * det["margins"] <= 0,
        "wrong_bit": bits != payload,
        # Align vote totals: positive values indicate votes favoring the correct bit
        "totals": det["totals"] * (2.0 * payload - 1.0),   # >0 = vote for the right bit
        "success": success,
    }


def _stats(r):
    """Extracts summary error metrics and starved-block correlations from a reproduced case.

    Args:
        r (dict): Output dictionary from reproduce().

    Returns:
        dict: Aggregated failure diagnostics, including wrong block fractions,
              relative strength of failing blocks, and starved block proportions.
    """
    wb, rel = r["wrong_block"], r["rel"]
    starved = rel < STARVED
    return {
        "success": r["success"],
        "wrong_bits": int(r["wrong_bit"].sum()),
        "wrong_blocks": round(float(wb.mean()), 4),
        "skipped_blocks": round(float(r["skipped"].mean()), 4),
        # Mean relative strength among blocks that voted incorrectly
        "rel_strength_wrong": round(float(rel[wb].mean()), 3) if wb.any() else np.nan,
        # Share of incorrectly voting blocks that were starved (strength < 0.75x mean)
        "starved_share_of_wrong": round(float(starved[wb].mean()), 3) if wb.any() else np.nan,
        # Baseline share of all blocks in the image that were starved
        "starved_share_all": round(float(starved.mean()), 3),
    }


def _panel(res, labels, title, path):
    """Renders a comparative diagnostic figure showing strength maps and error locations.

    Generates a 3-column figure per configuration:
      1. Heatmap of relative embedding strength across the 2D block grid.
      2. Attacked image with red spatial overlays highlighting blocks that voted incorrectly.
      3. Bar chart of per-bit soft-vote totals (red bars indicate flipped bits).

    Args:
        res (list[dict]): List of reproduced case outputs.
        labels (list[str]): Names of the configurations (e.g., ['adaptive+ecc', 'uniform+ecc']).
        title (str): Global figure suptitle.
        path (Path or str): Output filepath for the saved PNG graphic.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cols = grid_shape(res[0]["img"].shape)
    vmax = max(r["rel"].max() for r in res)
    fig, ax = plt.subplots(len(res), 3, figsize=(13, 4.2 * len(res)), squeeze=False)

    for i, (r, lab) in enumerate(zip(res, labels)):
        # Column 1: Spatial heatmap of relative embedding strength
        im = ax[i, 0].imshow(r["rel"].reshape(rows, cols), cmap="magma", vmin=0, vmax=vmax)
        ax[i, 0].set_title(f"{lab}: relative strength")
        fig.colorbar(im, ax=ax[i, 0], fraction=0.046)

        # Column 2: Attacked image with red highlights on incorrectly voting blocks
        rgb = r["att"][..., ::-1] if r["att"].ndim == 3 else r["att"]
        ax[i, 1].imshow(rgb, cmap="gray")
        # Upsample block-level error mask (16x16 pixel spatial footprint per block)
        wrong = np.kron(r["wrong_block"].reshape(rows, cols), np.ones((2 * BLOCK, 2 * BLOCK)))
        overlay = np.zeros((*wrong.shape, 4))
        overlay[..., 0], overlay[..., 3] = 1.0, 0.45 * wrong  # Semi-transparent red highlight
        ax[i, 1].imshow(overlay)
        ax[i, 1].set_title(f"after edit; red = wrong vote ({r['wrong_block'].mean():.0%} of blocks)")

        # Column 3: Soft-vote margin totals per codeword bit
        colors = ["#c0392b" if w else "#7f8c8d" for w in r["wrong_bit"]]
        ax[i, 2].bar(range(len(r["totals"])), r["totals"], color=colors)
        ax[i, 2].axhline(0, c="k", lw=0.8)
        ax[i, 2].set_title(f"vote per bit (red = wrong: {int(r['wrong_bit'].sum())}) -> "
                           f"{'RECOVERED' if r['success'] else 'FAILED'}")
        ax[i, 2].set_xlabel("payload bit")

        # Disable tick axes on spatial image visualizers
        for a in ax[i, :2]:
            a.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def case_job(args):
    """Worker function executing paired case reconstruction and visualization export.

    Args:
        args (tuple): (name, kind, severity, focus, base, model_path, params, out_dir).

    Returns:
        dict: Tabular row mapping failure metrics across focus and baseline configurations.
    """
    name, kind, severity, focus, base, model_path, params, out_dir = args
    # Rebuild both configurations side by side under identical image and attack conditions
    res = [reproduce("test", name, c, model_path, params, kind, severity) for c in (focus, base)]

    # Render diagnostic 3-column comparative figure
    out = Path(out_dir) / f"{name}_{kind}_{severity}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    _panel(res, [focus, base], f"test/{name}  |  {kind} ({severity})", out)

    # Compile comparative statistical metrics
    row = {"image": name, "attack": kind, "severity": severity, "picture": out.name}
    for c, r in zip(("focus", "base"), res):
        row.update({f"{c}_{k}": v for k, v in _stats(r).items()})
    return row
