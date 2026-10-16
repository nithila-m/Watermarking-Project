"""Per-block embedding strength.

    image -> features -> predicted disturbance / survival -> relative strength map
          -> embed_at_psnr() picks one global scale so every map hits the same PSNR.

Only the SHAPE of the map matters here. The idea ("equal SNR"): a block whose
coefficients will be pushed around a lot by edits needs a bigger margin; a block
that edits barely touch is wasting budget if it gets the same strength.
"""
import numpy as np

from . import core
from .features import block_features

# Canonical bounds for relative per-block strength prior to global PSNR calibration
S_MIN, S_MAX = 0.25, 4.0   # relative strength limits (before global scaling)


def normalize(rel, s_min=S_MIN, s_max=S_MAX):
    """Normalizes a relative strength vector to unit mean subject to box constraints.

    Iteratively clips the relative strength profile to [s_min, s_max] and re-scales
    by the arithmetic mean until numerical convergence (up to 5 fixed-point passes).
    This guarantees that the average relative strength remains exactly 1.0 while
    preventing both starved blocks and disproportionately excessive local distortion.

    Args:
        rel (array-like): Raw unnormalized relative strength values.
        s_min (float, optional): Lower bound relative to the mean. Defaults to S_MIN (0.25).
        s_max (float, optional): Upper bound relative to the mean. Defaults to S_MAX (4.0).

    Returns:
        np.ndarray: Bounded, unit-mean 1-D array of relative strengths (float64).
    """
    rel = np.asarray(rel, dtype=np.float64)
    rel = rel / rel.mean()
    for _ in range(5):                       # clip, renormalise, repeat until stable
        rel = np.clip(rel, s_min, s_max)
        rel = rel / rel.mean()
    return rel


def uniform_map(img):
    """Generates a uniform (flat) relative strength map across all blocks.

    Serves as the unweighted control baseline where every 8x8 DCT block in the LL
    band receives an identical perturbation budget.

    Args:
        img (np.ndarray): Host image array (BGR or grayscale uint8).

    Returns:
        np.ndarray: 1-D array of ones of length n_blocks.
    """
    rows, cols = core.grid_shape(img.shape)
    return np.ones(rows * cols)


def heuristic_map(img, feats=None, alpha=1.0):
    """Day-8 fallback, no learning: strength grows with local texture (std of the block).

    Allocates embedding strength purely based on spatial luminance standard deviation,
    leveraging the human visual system's texture-masking property without an ML model.

    Args:
        img (np.ndarray): Host image array.
        feats (pd.DataFrame, optional): Precomputed block feature table.
            If None, computed on the fly via block_features().
        alpha (float, optional): Power-law scaling exponent for standard deviation.
            Defaults to 1.0.

    Returns:
        np.ndarray: Normalized 1-D relative strength map of length n_blocks.
    """
    f = block_features(img) if feats is None else feats
    return normalize((f["std"].values + 2.0) ** alpha)


def learned_map(img, model, feats=None, alpha=1.0, beta=0.0, shrink=1.0, s_min=S_MIN):
    """Compute learned adaptive strength: s_i ~ predicted_sigma_i ** alpha * predicted_keep_i ** beta.

    Derives block-wise embedding strengths from machine learning predictions:
      - predicted_sigma: Expected margin disturbance under noise-like edits (JPEG, blur, VAE).
      - predicted_keep : Expected bit survival probability under spatial erasures (inpaint, crop).

    Two optional safety knobs (Phase 6 fixes). The defaults reproduce the original map exactly:
      shrink < 1 : pull the map toward uniform, rel = (1 - shrink) + shrink * rel
      s_min      : floor for any block's relative strength (default 0.25 of the mean)

    Args:
        img (np.ndarray): Host image array.
        model (SurvivalModel): Pretrained dual random-forest regressor model.
        feats (pd.DataFrame, optional): Precomputed block feature DataFrame.
            If None, computed from img via block_features().
        alpha (float, optional): Sensitivity exponent for predicted margin disturbance.
            Defaults to 1.0 (equal SNR allocation).
        beta (float, optional): Sensitivity exponent for predicted erasure survival.
            Defaults to 0.0 (disabled by default).
        shrink (float, optional): Shrinkage factor pulling relative deviations toward uniform (1.0).
            shrink=1.0 preserves full ML adaptation; shrink=0.0 yields uniform weighting.
        s_min (float, optional): Minimum relative strength floor before global scaling.
            Defaults to S_MIN (0.25).

    Returns:
        np.ndarray: Normalized 1-D relative strength map of length n_blocks.
    """
    f = block_features(img) if feats is None else feats
    # Scale strength proportionally to expected absolute margin disturbance
    rel = model.predict_sigma(f) ** alpha
    # Modulate strength by predicted survival probability under spatial erasures
    if beta:
        rel = rel * model.predict_keep(f) ** beta
    # Pull extreme variance back toward uniform baseline if shrinkage is active
    if shrink != 1.0:
        rel = rel / rel.mean()
        rel = (1.0 - shrink) + shrink * rel
    # Enforce unit mean and clamp within bounds [s_min, S_MAX]
    return normalize(rel, s_min=s_min)


def strength_map(img, mode="uniform", model=None, **kw):
    """Unified factory dispatching to the requested strength map allocation strategy.

    Args:
        img (np.ndarray): Host image array.
        mode (str, optional): Strategy name: 'uniform', 'heuristic', or 'learned'.
            Defaults to 'uniform'.
        model (SurvivalModel, optional): Pretrained survival model required if mode='learned'.
        **kw: Additional keyword arguments passed directly to heuristic_map or learned_map
            (e.g., alpha, beta, shrink, s_min).

    Returns:
        np.ndarray: Relative per-block strength array of shape (n_blocks,).

    Raises:
        ValueError: If mode='learned' but model is None, or if mode is unrecognized.
    """
    if mode == "uniform":
        return uniform_map(img)
    if mode == "heuristic":
        return heuristic_map(img, **kw)
    if mode == "learned":
        if model is None:
            raise ValueError("learned mode needs a model: predictor.load_model('models/survival.joblib')")
        return learned_map(img, model, **kw)
    raise ValueError(f"unknown mode {mode}")
