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
    if kind == "inpaint+jpeg":
        return attack(attack(img, "inpaint", severity, seed), "jpeg", severity, seed)
    return attack(img, kind, severity, seed)


def make_map(img, mode, model=None, feats=None, **kw):
    if mode == "uniform":
        return uniform_map(img)
    feats = block_features(img) if feats is None else feats
    if mode == "heuristic":
        return heuristic_map(img, feats, **kw)
    return learned_map(img, model, feats, **kw)


def evaluate_image(img, name, configs, model=None, key=2024, attacks=None, seed=0,
                   target_psnr=TARGET_PSNR, map_kw=None):
    """Return a list of result rows (dicts) for one image."""
    attacks = attacks if attacks is not None else all_attacks() + COMBO
    rng = np.random.default_rng(zlib.crc32(name.encode()))  # stable across runs
    feats = block_features(img) if any(c["map"] != "uniform" for c in configs.values()) else None
    raw_bits = rng.integers(0, 2, PAYLOAD_BITS)        # same payloads for every config (paired comparison)
    msg_bits = rng.integers(0, 2, MESSAGE_BITS)
    rows = []
    for cname, cfg in configs.items():
        if cfg["ecc"]:
            msg, payload = msg_bits, ecc.encode(msg_bits)
        else:
            msg = payload = raw_bits
        base = make_map(img, cfg["map"], model, feats, **(map_kw or {}).get(cfg["map"], {}))
        t0 = time.perf_counter()
        wm, _, p = embed_at_psnr(img, payload, base, key, target_psnr)
        t_embed = time.perf_counter() - t0
        s = ssim(img, wm)
        for kind, sev in [("none", "none")] + list(attacks):
            att = wm if kind == "none" else apply_attack(wm, kind, sev, seed)
            got = extract(att, PAYLOAD_BITS, key)
            if cfg["ecc"]:
                dec, ok = ecc.decode(got)
                success = bool(ok and np.array_equal(dec, msg))
            else:
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
