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

STARVED = 0.75   # a block with relative strength below this got less than its share


def reproduce(split, name, config, model_path, params, kind, severity, key=2024, seed=0):
    img = dataset.load(split, name)
    rng = np.random.default_rng(zlib.crc32(name.encode()))     # same rule as evaluate_image
    raw_bits = rng.integers(0, 2, PAYLOAD_BITS)
    msg_bits = rng.integers(0, 2, MESSAGE_BITS)
    cfg = CONFIGS[config]
    msg, payload = (msg_bits, ecc.encode(msg_bits)) if cfg["ecc"] else (raw_bits, raw_bits)
    kw = params if cfg["map"] == "learned" else {}
    base = make_map(img, cfg["map"], _model(model_path) if cfg["map"] == "learned" else None, **kw)
    wm, s, _ = embed_at_psnr(img, payload, base, key)
    _, skipped = embed(img, payload, s, key, return_skipped=True)
    att = wm if kind == "none" else apply_attack(wm, kind, severity, seed)
    bits, det = extract(att, PAYLOAD_BITS, key, return_details=True)
    if cfg["ecc"]:
        dec, ok = ecc.decode(bits)
        success = bool(ok and np.array_equal(dec, msg))
    else:
        success = bool(np.array_equal(bits, payload))
    sign = 2.0 * payload[det["assign"]] - 1.0
    return {"img": img, "att": att, "rel": s / s.mean(), "skipped": skipped,
            "wrong_block": sign * det["margins"] <= 0, "wrong_bit": bits != payload,
            "totals": det["totals"] * (2.0 * payload - 1.0),   # >0 = vote for the right bit
            "success": success}


def _stats(r):
    wb, rel = r["wrong_block"], r["rel"]
    starved = rel < STARVED
    return {"success": r["success"],
            "wrong_bits": int(r["wrong_bit"].sum()),
            "wrong_blocks": round(float(wb.mean()), 4),
            "skipped_blocks": round(float(r["skipped"].mean()), 4),
            "rel_strength_wrong": round(float(rel[wb].mean()), 3) if wb.any() else np.nan,
            "starved_share_of_wrong": round(float(starved[wb].mean()), 3) if wb.any() else np.nan,
            "starved_share_all": round(float(starved.mean()), 3)}


def _panel(res, labels, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, cols = grid_shape(res[0]["img"].shape)
    vmax = max(r["rel"].max() for r in res)
    fig, ax = plt.subplots(len(res), 3, figsize=(13, 4.2 * len(res)), squeeze=False)
    for i, (r, lab) in enumerate(zip(res, labels)):
        im = ax[i, 0].imshow(r["rel"].reshape(rows, cols), cmap="magma", vmin=0, vmax=vmax)
        ax[i, 0].set_title(f"{lab}: relative strength")
        fig.colorbar(im, ax=ax[i, 0], fraction=0.046)
        rgb = r["att"][..., ::-1] if r["att"].ndim == 3 else r["att"]
        ax[i, 1].imshow(rgb, cmap="gray")
        wrong = np.kron(r["wrong_block"].reshape(rows, cols), np.ones((2 * BLOCK, 2 * BLOCK)))
        overlay = np.zeros((*wrong.shape, 4))
        overlay[..., 0], overlay[..., 3] = 1.0, 0.45 * wrong
        ax[i, 1].imshow(overlay)
        ax[i, 1].set_title(f"after edit; red = wrong vote ({r['wrong_block'].mean():.0%} of blocks)")
        colors = ["#c0392b" if w else "#7f8c8d" for w in r["wrong_bit"]]
        ax[i, 2].bar(range(len(r["totals"])), r["totals"], color=colors)
        ax[i, 2].axhline(0, c="k", lw=0.8)
        ax[i, 2].set_title(f"vote per bit (red = wrong: {int(r['wrong_bit'].sum())}) -> "
                           f"{'RECOVERED' if r['success'] else 'FAILED'}")
        ax[i, 2].set_xlabel("payload bit")
        for a in ax[i, :2]:
            a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def case_job(args):
    """args = (name, kind, severity, focus, base, model_path, params, out_dir)."""
    name, kind, severity, focus, base, model_path, params, out_dir = args
    res = [reproduce("test", name, c, model_path, params, kind, severity) for c in (focus, base)]
    out = Path(out_dir) / f"{name}_{kind}_{severity}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    _panel(res, [focus, base], f"test/{name}  |  {kind} ({severity})", out)
    row = {"image": name, "attack": kind, "severity": severity, "picture": out.name}
    for c, r in zip(("focus", "base"), res):
        row.update({f"{c}_{k}": v for k, v in _stats(r).items()})
    return row
