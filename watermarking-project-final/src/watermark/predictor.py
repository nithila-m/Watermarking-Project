"""AI-edit survival predictor.

For each block we learn two things from UNIFORM-watermarked images that were attacked:

  log_sigma : how much the attacks disturb this block's watermark coefficient
              difference (noise-like edits: JPEG, blur, resize, noise, AI smoothing, and the
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
from .attacks import ERASE_TYPE, NOISE_TYPE, SEVERITIES, attack
from .config import PAYLOAD_BITS, TARGET_PSNR
from .embed import embed_at_psnr
from .extract import block_margins
from .features import FEATURE_NAMES, block_features

# Multi-severity attack sweeps grouped by failure mode
NOISE_ATTACKS = [(k, s) for k in NOISE_TYPE for s in SEVERITIES]   # jpeg, blur, resize, noise, smooth
ERASE_ATTACKS = [(k, s) for k in ERASE_TYPE for s in SEVERITIES]   # inpaint, inpaint_square, crop


def build_survival_rows(img, image_id, key=1234, seed=0, target_psnr=TARGET_PSNR,
                        external_noise=None, external_erase=None, erase_repeats=2):
    """Feature rows + targets for one image.

    external_noise / external_erase: optional lists of attacked versions of the
    watermarked image produced elsewhere (e.g. B's Stable Diffusion edits). Use
    export_for_attack() to get the watermarked image B should edit.

    Args:
        img (np.ndarray): Pristine host image (BGR or grayscale uint8).
        image_id (str): Identifier name or stem for the image.
        key (int, optional): Fixed training key for watermark embedding. Defaults to 1234.
        seed (int, optional): Random seed for reproducible payload and attacks. Defaults to 0.
        target_psnr (float, optional): Target PSNR ceiling in dB. Defaults to TARGET_PSNR (42.0).
        external_noise (list[np.ndarray], optional): Real noise-type AI edits (e.g., regeneration/VAE).
        external_erase (list[np.ndarray], optional): Real erasure-type AI edits (e.g., SD inpainting).
        erase_repeats (int, optional): Random seeds per erasure attack to sample spatial variability.
            Defaults to 2.

    Returns:
        pd.DataFrame: Table with 13 features, log_sigma and keep targets, image_id, and block index.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, PAYLOAD_BITS)
    n_blocks = int(np.prod(core.grid_shape(img.shape)))

    # Uniform watermark at fixed quality baseline
    wm, _, _ = embed_at_psnr(img, bits, np.ones(n_blocks), key, target_psnr)
    sign = 2.0 * bits[core.bit_assignment(n_blocks, PAYLOAD_BITS, key)] - 1.0
    d0 = block_margins(wm)

    # Distort with synthetic suites and concatenate external real edits if supplied
    noisy = [attack(wm, k, s, seed=seed) for k, s in NOISE_ATTACKS] + list(external_noise or [])
    erased = [attack(wm, k, s, seed=seed + r) for k, s in ERASE_ATTACKS
              for r in range(erase_repeats)] + list(external_erase or [])

    # Extract 13 features from unwatermarked host image
    df = block_features(img)

    # log_sigma: average shift in coefficient margin across noise-like edits
    df["log_sigma"] = np.log(np.mean([np.abs(block_margins(a) - d0) for a in noisy], axis=0) + 1e-3)

    # keep: empirical fraction of erasure-like edits where the block's sign remains correct
    df["keep"] = np.mean([(sign * block_margins(a) > 0) for a in erased], axis=0)
    df["image_id"] = image_id
    df["block"] = np.arange(n_blocks)
    return df


def export_for_attack(img, key=1234, seed=0, target_psnr=TARGET_PSNR):
    """The exact watermarked image build_survival_rows() uses, to hand to B for real AI edits.

    Args:
        img (np.ndarray): Input host image array.
        key (int, optional): Fixed training key. Defaults to 1234.
        seed (int, optional): Random seed matching build_survival_rows payload. Defaults to 0.
        target_psnr (float, optional): Embedding target PSNR in dB. Defaults to TARGET_PSNR (42.0).

    Returns:
        np.ndarray: Uniformly watermarked 512x512 image array.
    """
    bits = np.random.default_rng(seed).integers(0, 2, PAYLOAD_BITS)
    n_blocks = int(np.prod(core.grid_shape(img.shape)))
    return embed_at_psnr(img, bits, np.ones(n_blocks), key, target_psnr)[0]


def _rf(seed, **kw):
    """Constructs a RandomForestRegressor with project standard hyperparameters.

    Args:
        seed (int): Random state for reproducibility.
        **kw: Additional estimator keyword arguments.

    Returns:
        RandomForestRegressor: Configured regressor instance.
    """
    params = dict(n_estimators=200, max_depth=14, min_samples_leaf=20, n_jobs=-1, random_state=seed)
    params.update(kw)
    return RandomForestRegressor(**params)


def train(df, val_frac=0.2, seed=0, **rf_kw):
    """Fit both regressors. Returns (SurvivalModel, report dict).

    Splits the training data into train/val using GroupShuffleSplit grouped by
    image_id so that blocks from the same image never appear in both splits.

    Args:
        df (pd.DataFrame): Combined block features and targets from training images.
        val_frac (float, optional): Fraction of images reserved for validation. Defaults to 0.2.
        seed (int, optional): Random seed for split and model fitting. Defaults to 0.
        **rf_kw: Additional keyword overrides for RandomForestRegressor.

    Returns:
        tuple[SurvivalModel, dict]: Fitted production SurvivalModel and validation metrics report.
    """
    X, groups = df[FEATURE_NAMES].values, df["image_id"].values

    # Image-grouped validation split to prevent spatial feature leakage
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
                  .split(X, groups=groups))
    report = {"n_train_blocks": int(len(tr)), "n_val_blocks": int(len(va)),
              "val_images": sorted(set(groups[va]))}
    models = {}

    for target in ("log_sigma", "keep"):
        y = df[target].values

        # Validation fit to compute R^2, Spearman rank correlation, and feature importances
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

        # Refit model on all available training rows for deployment
        models[target] = _rf(seed, **rf_kw).fit(X, y)   # refit on all rows for use

    return SurvivalModel(models["log_sigma"], models["keep"]), report


class SurvivalModel:
    """Container for the dual random-forest regression models.

    sigma_rf: Predicts log_sigma (margin shift under noise-like edits).
    keep_rf : Predicts keep (survival probability under erasure edits).
    """

    def __init__(self, sigma_rf, keep_rf):
        """Initializes SurvivalModel with trained regressor instances.

        Args:
            sigma_rf (RandomForestRegressor): Model trained on log_sigma target.
            keep_rf (RandomForestRegressor): Model trained on keep target.
        """
        self.sigma_rf, self.keep_rf = sigma_rf, keep_rf

    def predict_sigma(self, feats):
        """Predicts expected absolute margin disturbance sigma_i = exp(log_sigma_hat).

        Args:
            feats (pd.DataFrame): Block features matching FEATURE_NAMES.

        Returns:
            np.ndarray: Predicted margin disturbances of shape (n_blocks,).
        """
        return np.exp(self.sigma_rf.predict(feats[FEATURE_NAMES].values))

    def predict_keep(self, feats):
        """Predicts survival probability under erasures, bounded to [0.01, 1.0].

        Args:
            feats (pd.DataFrame): Block features matching FEATURE_NAMES.

        Returns:
            np.ndarray: Predicted survival rates of shape (n_blocks,).
        """
        return np.clip(self.keep_rf.predict(feats[FEATURE_NAMES].values), 0.01, 1.0)


def save_model(model, path):
    """Serializes a SurvivalModel to disk using joblib.

    Args:
        model (SurvivalModel): Model instance to save.
        path (str or Path): Destination filepath (e.g., 'models/survival.joblib').
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """Loads a serialized SurvivalModel from disk.

    Args:
        path (str or Path): Filepath to the saved joblib archive.

    Returns:
        SurvivalModel: Deserialized model instance.
    """
    return joblib.load(path)
