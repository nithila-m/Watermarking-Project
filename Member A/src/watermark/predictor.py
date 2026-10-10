"""AI-edit survival predictor.

For each block we learn two things from UNIFORM-watermarked images that were attacked:

  log_sigma : how much the attacks disturb this block's watermark coefficient
              difference (noise-like edits: JPEG, blur, resize, noise, and the
              VAE re-encoding inside diffusion edits). Blocks that get disturbed
              more need a larger margin to keep their bit.
  keep      : fraction of erasure-type edits (inpainting, crop) after which the
              block still carries the right bit.

strength_map.py turns these into a strength map: s_i ~ sigma_i ** alpha * keep_i ** beta
(alpha=1, beta=0 by default = "equal signal-to-noise" allocation).

Train/validation split is BY IMAGE so blocks from one image never leak across.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

from . import core
from .attacks import SEVERITIES, attack
from .config import PAYLOAD_BITS, TARGET_PSNR
from .embed import embed_at_psnr
from .extract import block_margins
from .features import FEATURE_NAMES, block_features

NOISE_ATTACKS = [(k, s) for k in ("jpeg", "blur", "resize", "noise") for s in SEVERITIES]
ERASE_ATTACKS = [(k, s) for k in ("inpaint", "crop") for s in SEVERITIES]


def build_survival_rows(img, image_id, key=1234, seed=0, target_psnr=TARGET_PSNR,
                        external_noise=None, external_erase=None, erase_repeats=2):
    """Feature rows + targets for one image.

    external_noise / external_erase: optional lists of attacked versions of the
    watermarked image produced elsewhere (e.g. B's Stable Diffusion edits). Use
    export_for_attack() to get the watermarked image B should edit.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, PAYLOAD_BITS)
    n_blocks = int(np.prod(core.grid_shape(img.shape)))
    wm, _, _ = embed_at_psnr(img, bits, np.ones(n_blocks), key, target_psnr)
    sign = 2.0 * bits[core.bit_assignment(n_blocks, PAYLOAD_BITS, key)] - 1.0
    d0 = block_margins(wm)

    noisy = [attack(wm, k, s, seed=seed) for k, s in NOISE_ATTACKS] + list(external_noise or [])
    erased = [attack(wm, k, s, seed=seed + r) for k, s in ERASE_ATTACKS
              for r in range(erase_repeats)] + list(external_erase or [])

    df = block_features(img)
    df["log_sigma"] = np.log(np.mean([np.abs(block_margins(a) - d0) for a in noisy], axis=0) + 1e-3)
    df["keep"] = np.mean([(sign * block_margins(a) > 0) for a in erased], axis=0)
    df["image_id"] = image_id
    df["block"] = np.arange(n_blocks)
    return df


def export_for_attack(img, key=1234, seed=0, target_psnr=TARGET_PSNR):
    """The exact watermarked image build_survival_rows() uses, to hand to B for real AI edits."""
    bits = np.random.default_rng(seed).integers(0, 2, PAYLOAD_BITS)
    n_blocks = int(np.prod(core.grid_shape(img.shape)))
    return embed_at_psnr(img, bits, np.ones(n_blocks), key, target_psnr)[0]


def _rf(seed, **kw):
    params = dict(n_estimators=200, max_depth=14, min_samples_leaf=20, n_jobs=-1, random_state=seed)
    params.update(kw)
    return RandomForestRegressor(**params)


def train(df, val_frac=0.2, seed=0, **rf_kw):
    """Fit both regressors. Returns (SurvivalModel, report dict)."""
    X, groups = df[FEATURE_NAMES].values, df["image_id"].values
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
                  .split(X, groups=groups))
    report = {"n_train_blocks": int(len(tr)), "n_val_blocks": int(len(va)),
              "val_images": sorted(set(groups[va]))}
    models = {}
    for target in ("log_sigma", "keep"):
        y = df[target].values
        rf = _rf(seed, **rf_kw).fit(X[tr], y[tr])
        pred = rf.predict(X[va])
        report[target] = {
            "val_r2": round(float(rf.score(X[va], y[va])), 4),
            "val_spearman": round(float(spearmanr(pred, y[va])[0]), 4),
            # the hand-made fallback uses block texture (std); how well does that rank?
            "val_spearman_texture_only": round(float(spearmanr(df["std"].values[va], y[va])[0]), 4),
            "importances": {k: round(float(v), 4) for k, v in
                            sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda kv: -kv[1])},
        }
        models[target] = _rf(seed, **rf_kw).fit(X, y)   # refit on all rows for use
    return SurvivalModel(models["log_sigma"], models["keep"]), report


class SurvivalModel:
    def __init__(self, sigma_rf, keep_rf):
        self.sigma_rf, self.keep_rf = sigma_rf, keep_rf

    def predict_sigma(self, feats):
        return np.exp(self.sigma_rf.predict(feats[FEATURE_NAMES].values))

    def predict_keep(self, feats):
        return np.clip(self.keep_rf.predict(feats[FEATURE_NAMES].values), 0.01, 1.0)


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
