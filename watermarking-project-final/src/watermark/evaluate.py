"""Shared evaluation loop used by the Phase 2, 4 and 5 scripts.

A configuration = (strength map mode, ECC on/off). For each image:
  - ECC off: 64 random payload bits; success = all 64 bits correct.
  - ECC on : 32 random message bits -> Reed-Solomon -> 64 coded bits;
             success = the 32-bit message decodes exactly.
  - ber_ok : the lenient benchmark criterion, raw BER <= BER_SUCCESS_THRESHOLD.
Both write 64 bits into the image, so they cost the same; all maps are
calibrated to the same PSNR.
"""
import time
import zlib

import numpy as np

from . import ecc
from .attacks import all_attacks, attack
from .config import MESSAGE_BITS, PAYLOAD_BITS, TARGET_PSNR
from .embed import embed_at_psnr
from .extract import extract
from .features import block_features
from .metrics import ber, is_successful_extraction, nc, psnr, ssim
from .strength_map import heuristic_map, learned_map, uniform_map

# Canonical experimental configurations evaluated across benchmarks
CONFIGS = {
    "uniform":           {"map": "uniform",   "ecc": False},
    "uniform+ecc":       {"map": "uniform",   "ecc": True},
    "heuristic+ecc":     {"map": "heuristic", "ecc": True},
    "adaptive":          {"map": "learned",   "ecc": False},
    "adaptive+ecc":      {"map": "learned",   "ecc": True},
}

# extra attack beyond the suite: an AI edit followed by re-compression, as happens
# when an edited image is re-uploaded
COMBO = [("inpaint+jpeg", "moderate")]


def apply_attack(img, kind, severity, seed):
    """Executes a single attack or a sequential composite attack on an image.

    Args:
        img (np.ndarray): Watermarked input image array.
        kind (str): Distortion identifier (e.g., 'blur', 'inpaint+jpeg').
        severity (str): Distortion tier ('light', 'moderate', 'severe').
        seed (int): Deterministic random seed for attack perturbations.

    Returns:
        np.ndarray: Attacked image array.
    """
    # Composite pipeline: apply inpainting followed immediately by lossy JPEG compression
    if kind == "inpaint+jpeg":
        return attack(attack(img, "inpaint", severity, seed), "jpeg", severity, seed)
    return attack(img, kind, severity, seed)


def make_map(img, mode, model=None, feats=None, **kw):
    """Generates the relative per-block embedding strength map for a given mode.

    Args:
        img (np.ndarray): Host image array.
        mode (str): Allocation strategy identifier ('uniform', 'heuristic', 'learned').
        model (Any, optional): Pretrained random forest survival regressor instance.
        feats (np.ndarray, optional): Precomputed 13-feature array for all blocks.
        **kw: Additional keyword arguments passed to heuristic or learned map generators.

    Returns:
        np.ndarray: Relative per-block strength array of shape (n_blocks,).
    """
    # Baseline: uniform weighting across all blocks
    if mode == "uniform":
        return uniform_map(img)
    # Lazy extraction of 13 block features if not provided
    feats = block_features(img) if feats is None else feats
    # Heuristic weighting based on block spatial standard deviation
    if mode == "heuristic":
        return heuristic_map(img, feats, **kw)
    # Learned weighting driven by random forest edit-survival predictions
    return learned_map(img, model, feats, **kw)


def evaluate_image(img, name, configs, model=None, key=2024, attacks=None, seed=0,
                   target_psnr=TARGET_PSNR, map_kw=None):
    """Return a list of result rows (dicts) for one image.

    Conducts a paired, equal-PSNR evaluation of multiple watermarking configurations
    across clean conditions and an adversarial perturbation suite.

    Args:
        img (np.ndarray): Host image array (BGR or grayscale uint8).
        name (str): Identifier or file stem of the host image.
        configs (dict): Configurations to evaluate (subset or all of CONFIGS).
        model (Any, optional): Pretrained survival model for adaptive configurations.
        key (int, optional): Pseudo-random key for block scattering. Defaults to 2024.
        attacks (list[tuple[str, str]], optional): List of (attack, severity) tuples.
            Defaults to all_attacks() + COMBO.
        seed (int, optional): Random seed for stochastic distortions. Defaults to 0.
        target_psnr (float, optional): Exact PSNR ceiling in dB. Defaults to TARGET_PSNR (42.0).
        map_kw (dict, optional): Keyword arguments dictionary partitioned by map mode.

    Returns:
        list[dict]: Granular trial records containing fidelity, extraction,
            and error correction metrics.
    """
    attacks = attacks if attacks is not None else all_attacks() + COMBO
    # Seed generator deterministically using image name checksum for reproducible payloads
    rng = np.random.default_rng(zlib.crc32(name.encode()))  # stable across runs

    # Extract block features once per image if any non-uniform config is present
    feats = block_features(img) if any(c["map"] != "uniform" for c in configs.values()) else None

    # Generate shared random payloads ensuring paired statistical fairness
    raw_bits = rng.integers(0, 2, PAYLOAD_BITS)        # same payloads for every config (paired comparison)
    msg_bits = rng.integers(0, 2, MESSAGE_BITS)
    rows = []

    # Iterate through all target experimental configurations
    for cname, cfg in configs.items():
        # Encode message with Reed-Solomon if ECC is active; otherwise use raw bits
        if cfg["ecc"]:
            msg, payload = msg_bits, ecc.encode(msg_bits)
        else:
            msg = payload = raw_bits

        # Generate unscaled relative strength profile
        base = make_map(img, cfg["map"], model, feats, **(map_kw or {}).get(cfg["map"], {}))

        # Embed codeword and scale strength map via binary search to hit exact target PSNR
        t0 = time.perf_counter()
        wm, _, p = embed_at_psnr(img, payload, base, key, target_psnr)
        t_embed = time.perf_counter() - t0
        s = ssim(img, wm)

        # Evaluate extraction under pristine baseline and each adversarial condition
        for kind, sev in [("none", "none")] + list(attacks):
            att = wm if kind == "none" else apply_attack(wm, kind, sev, seed)
            got = extract(att, PAYLOAD_BITS, key)

            # Determine payload decoding success
            if cfg["ecc"]:
                # Success requires exact byte-for-byte recovery of the 32-bit message
                dec, ok = ecc.decode(got)
                success = bool(ok and np.array_equal(dec, msg))
            else:
                # Raw success requires all 64 codeword bits to match exactly
                success = bool(np.array_equal(got, payload))

            b = ber(payload, got)
            rows.append({
                "image": name, "config": cname,
                "attack": kind, "severity": sev,
                "psnr": round(p, 3), "ssim": round(s, 4),
                "attacked_psnr": round(psnr(img, att), 3),
                "ber": b, "bit_acc": 1 - b, "nc": nc(payload, got),
                "success": success, "ber_ok": is_successful_extraction(b),
                "embed_seconds": round(t_embed, 3),
            })
    return rows
