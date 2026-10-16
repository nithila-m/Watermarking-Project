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

# Worker-local model cache mapping file paths to loaded predictor instances
_MODELS = {}


def _model(path):
    """Load the survival model once per worker process, single-threaded.

    Ensures that parallel worker processes do not re-read model artifacts from disk
    on every task invocation, and sets scikit-learn tree estimators to single-threaded
    execution (n_jobs=1) to prevent CPU thread contention across worker pools.

    Args:
        path (str or Path or None): Filepath to the serialized survival model.

    Returns:
        SurvivalPredictor or None: Loaded model instance, or None if path is None.
    """
    if path is None:
        return None
    path = str(path)
    if path not in _MODELS:
        m = load_model(path)
        # Prevent CPU oversubscription across parallel worker processes
        for rf in (m.sigma_rf, m.keep_rf):
            rf.n_jobs = 1          # the worker pool already uses every core
        _MODELS[path] = m
    return _MODELS[path]


def survival_job(args):
    """Phase 3 task: extract block features and empirical survival labels for one training image.

    Args:
        args (tuple): A 3-element tuple containing:
            - name (str): Stem identifier of the training image.
            - ext_root (str or Path or None): Root directory of external real AI edits.
            - ext_weight (float): Target blending weight allocated to real AI edits.

    Returns:
        pd.DataFrame: Block-level feature and target label table for the image.
    """
    name, ext_root, ext_weight = args
    # Collect external real edits (e.g. Stable Diffusion inpainting and regeneration)
    noise, erase, found = external_train_versions(ext_root, name)
    # Compute block features and empirical target labels (log_sigma and keep)
    df = survival_rows(dataset.load("train", name), name, noise, erase, ext_weight=ext_weight)
    df["external_edits"] = ",".join(found)
    return df


def eval_job(args):
    """Phases 4 and 5 task: evaluate watermarking configurations across attacks on one image.

    Args:
        args (tuple): A 5-element tuple containing:
            - split (str): Dataset partition ('train', 'val', 'test').
            - name (str): Image stem identifier.
            - configs (list[str] or dict): Configuration names or dictionary mapping.
            - model_path (str or Path or None): Path to the trained survival predictor.
            - map_kw (dict or None): Keyword arguments passed to strength map generators.

    Returns:
        list[dict]: List of evaluated trial records containing fidelity, BER, and recovery metrics.
    """
    split, name, configs, model_path, map_kw = args
    # Resolve configuration dictionary from canonical registry if passed as a name list
    if not isinstance(configs, dict):
        configs = {c: CONFIGS[c] for c in configs}
    rows = evaluate_image(dataset.load(split, name), name, configs, _model(model_path),
                          map_kw=map_kw)
    for r in rows:
        r["split"] = split
    return rows


def eval_variants_job(args):
    """Phase 4 sweep task: evaluate several learned-map variants on one image.

    Args:
        args (tuple): A 5-element tuple containing:
            - split (str): Dataset split ('val' or 'test').
            - name (str): Image stem identifier.
            - base_configs (list[str]): Baseline configuration names to compare against.
            - variants (dict[str, dict]): Map variant names to hyperparameter dicts.
            - model_path (str or Path): Path to the serialized survival model.

    Returns:
        list[dict]: Compiled evaluation results covering baselines and candidate variants.
    """
    split, name, base_configs, variants, model_path = args
    img = dataset.load(split, name)
    model = _model(model_path)
    # Evaluate baseline configurations (e.g. uniform+ecc)
    rows = evaluate_image(img, name, {c: CONFIGS[c] for c in base_configs}, model)
    # Evaluate each learned-map parameter variant under equal PSNR
    for vname, kw in variants.items():
        rows += evaluate_image(img, name, {vname: {"map": "learned", "ecc": True}}, model,
                               map_kw={"learned": kw})
    return rows


def export_train_job(args):
    """Watermarked training image exactly as Phase 3 sees it.

    Embeds a uniform watermark at key 1234, saves the watermarked image to disk,
    and returns metadata for the export manifest.

    Args:
        args (tuple): A 2-element tuple containing:
            - name (str): Image stem identifier.
            - out_dir (str or Path): Output directory for exported assets.

    Returns:
        list[dict]: Manifest record containing image name, key, payload bits, PSNR, and SSIM.
    """
    name, out_dir = args
    img = dataset.load("train", name)
    # Generate canonical uniform watermark used for training data collection
    wm = export_for_attack(img)                         # key 1234, seed 0, uniform
    out = Path(out_dir) / "train" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)       # cv2.imwrite fails silently otherwise
    save_image(out, wm)
    # Replicate deterministic training payload (seed 0)
    bits = np.random.default_rng(0).integers(0, 2, PAYLOAD_BITS)
    return [{"split": "train", "config": "uniform", "image": name, "key": 1234,
             "payload": bits_to_str(bits), "message": "", "ecc": False,
             "psnr": round(psnr(img, wm), 3), "ssim": round(ssim(img, wm), 4)}]


def export_test_job(args):
    """One test image watermarked with every requested config.

    Embeds watermarks across all requested configurations, calibrated strictly to
    TARGET_PSNR (42 dB), saves output PNGs, and returns manifest records.

    Args:
        args (tuple): A 6-element tuple containing:
            - name (str): Test image stem identifier.
            - config_names (list[str]): Names of configurations to export.
            - model_path (str or Path or None): Path to the survival model.
            - map_kw (dict or None): Parameter overrides for adaptive strength mapping.
            - out_dir (str or Path): Output root directory.
            - key (int): Permutation key for block scattering.

    Returns:
        list[dict]: List of manifest rows for all exported test configurations.
    """
    name, config_names, model_path, map_kw, out_dir, key = args
    img = dataset.load("test", name)
    # Mirror evaluate_image payload generation to ensure identical bits across test scripts
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    raw_bits = rng.integers(0, 2, PAYLOAD_BITS)
    msg_bits = rng.integers(0, 2, MESSAGE_BITS)
    model = _model(model_path)
    rows = []

    # Iterate through each requested experimental configuration
    for c in config_names:
        cfg = CONFIGS[c]
        # Encode with Reed-Solomon if ECC is active; otherwise use raw random bits
        msg, payload = (msg_bits, ecc.encode(msg_bits)) if cfg["ecc"] else (raw_bits, raw_bits)
        base = make_map(img, cfg["map"], model, **(map_kw or {}).get(cfg["map"], {}))
        # Embed and calibrate strength map via binary search to hit exact TARGET_PSNR
        wm, _, p = embed_at_psnr(img, payload, base, key, TARGET_PSNR)
        out = Path(out_dir) / "test" / c / f"{name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_image(out, wm)

        # Append manifest trial metadata
        rows.append({"split": "test", "config": c, "image": name, "key": key,
                     "payload": bits_to_str(payload), "message": bits_to_str(msg) if cfg["ecc"] else "",
                     "ecc": cfg["ecc"], "psnr": round(p, 3), "ssim": round(ssim(img, wm), 4)})
    return rows
