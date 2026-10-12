"""One-image jobs for watermark.parallel.pmap.

They live in the package (not in the scripts) because Windows worker processes
can only call functions they can import. Each job takes ONE tuple argument and
returns plain data (DataFrame / list of dicts), never images, to keep the
traffic between processes small.
"""
import zlib
from pathlib import Path

import numpy as np

from . import dataset, ecc
from .config import MESSAGE_BITS, PAYLOAD_BITS, TARGET_PSNR
from .core import save_image
from .embed import embed_at_psnr
from .evaluate import CONFIGS, evaluate_image, make_map
from .external import bits_to_str, external_train_versions, survival_rows
from .metrics import psnr, ssim
from .predictor import export_for_attack, load_model

_MODELS = {}


def _model(path):
    """Load the survival model once per worker process, single-threaded."""
    if path is None:
        return None
    path = str(path)
    if path not in _MODELS:
        m = load_model(path)
        for rf in (m.sigma_rf, m.keep_rf):
            rf.n_jobs = 1          # the worker pool already uses every core
        _MODELS[path] = m
    return _MODELS[path]


def survival_job(args):
    """Phase 3. args = (name, external_root or None, ext_weight)."""
    name, ext_root, ext_weight = args
    noise, erase, found = external_train_versions(ext_root, name)
    df = survival_rows(dataset.load("train", name), name, noise, erase, ext_weight=ext_weight)
    df["external_edits"] = ",".join(found)
    return df


def eval_job(args):
    """Phases 4 and 5. args = (split, name, config_names_or_dict, model_path, map_kw).

    config_names_or_dict: list of names from evaluate.CONFIGS, or a dict
    {name: {"map": ..., "ecc": ...}} for extra variants (the Phase 4 sweep).
    """
    split, name, configs, model_path, map_kw = args
    if not isinstance(configs, dict):
        configs = {c: CONFIGS[c] for c in configs}
    rows = evaluate_image(dataset.load(split, name), name, configs, _model(model_path),
                          map_kw=map_kw)
    for r in rows:
        r["split"] = split
    return rows


def eval_variants_job(args):
    """Phase 4 sweep: several learned-map variants on one image.

    args = (split, name, base_configs, variants, model_path) where
    variants = {variant_name: {"alpha": a, "beta": b}}.
    """
    split, name, base_configs, variants, model_path = args
    img = dataset.load(split, name)
    model = _model(model_path)
    rows = evaluate_image(img, name, {c: CONFIGS[c] for c in base_configs}, model)
    for vname, kw in variants.items():
        rows += evaluate_image(img, name, {vname: {"map": "learned", "ecc": True}}, model,
                               map_kw={"learned": kw})
    return rows


def export_train_job(args):
    """Watermarked training image exactly as Phase 3 sees it. args = (name, out_dir)."""
    name, out_dir = args
    img = dataset.load("train", name)
    wm = export_for_attack(img)                         # key 1234, seed 0, uniform
    out = Path(out_dir) / "train" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)       # cv2.imwrite fails silently otherwise
    save_image(out, wm)
    bits = np.random.default_rng(0).integers(0, 2, PAYLOAD_BITS)
    return [{"split": "train", "config": "uniform", "image": name, "key": 1234,
             "payload": bits_to_str(bits), "message": "", "ecc": False,
             "psnr": round(psnr(img, wm), 3), "ssim": round(ssim(img, wm), 4)}]


def export_test_job(args):
    """One test image watermarked with every requested config.

    args = (name, config_names, model_path, map_kw, out_dir, key)
    Payloads follow the same rule as evaluate.evaluate_image, so p5 and
    p5_external use identical messages for the same image.
    """
    name, config_names, model_path, map_kw, out_dir, key = args
    img = dataset.load("test", name)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    raw_bits = rng.integers(0, 2, PAYLOAD_BITS)
    msg_bits = rng.integers(0, 2, MESSAGE_BITS)
    model = _model(model_path)
    rows = []
    for c in config_names:
        cfg = CONFIGS[c]
        msg, payload = (msg_bits, ecc.encode(msg_bits)) if cfg["ecc"] else (raw_bits, raw_bits)
        base = make_map(img, cfg["map"], model, **(map_kw or {}).get(cfg["map"], {}))
        wm, _, p = embed_at_psnr(img, payload, base, key, TARGET_PSNR)
        out = Path(out_dir) / "test" / c / f"{name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_image(out, wm)
        rows.append({"split": "test", "config": c, "image": name, "key": key,
                     "payload": bits_to_str(payload), "message": bits_to_str(msg) if cfg["ecc"] else "",
                     "ecc": cfg["ecc"], "psnr": round(p, 3), "ssim": round(ssim(img, wm), 4)})
    return rows
