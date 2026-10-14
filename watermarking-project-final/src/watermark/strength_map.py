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

S_MIN, S_MAX = 0.25, 4.0   # relative strength limits (before global scaling)


def normalize(rel, s_min=S_MIN, s_max=S_MAX):
    rel = np.asarray(rel, dtype=np.float64)
    rel = rel / rel.mean()
    for _ in range(5):                       # clip, renormalise, repeat until stable
        rel = np.clip(rel, s_min, s_max)
        rel = rel / rel.mean()
    return rel


def uniform_map(img):
    rows, cols = core.grid_shape(img.shape)
    return np.ones(rows * cols)


def heuristic_map(img, feats=None, alpha=1.0):
    """Day-8 fallback, no learning: strength grows with local texture (std of the block)."""
    f = block_features(img) if feats is None else feats
    return normalize((f["std"].values + 2.0) ** alpha)


def learned_map(img, model, feats=None, alpha=1.0, beta=0.0, shrink=1.0, s_min=S_MIN):
    """s_i ~ predicted_sigma_i ** alpha * predicted_keep_i ** beta.

    Two optional safety knobs (Phase 6 fixes). The defaults reproduce the original map exactly.
      shrink < 1 : pull the map toward uniform, rel = (1 - shrink) + shrink * rel
      s_min      : floor for any block's relative strength (default 0.25 of the mean)
    """
    f = block_features(img) if feats is None else feats
    rel = model.predict_sigma(f) ** alpha
    if beta:
        rel = rel * model.predict_keep(f) ** beta
    if shrink != 1.0:
        rel = rel / rel.mean()
        rel = (1.0 - shrink) + shrink * rel
    return normalize(rel, s_min=s_min)


def strength_map(img, mode="uniform", model=None, **kw):
    """mode: uniform | heuristic | learned"""
    if mode == "uniform":
        return uniform_map(img)
    if mode == "heuristic":
        return heuristic_map(img, **kw)
    if mode == "learned":
        if model is None:
            raise ValueError("learned mode needs a model: predictor.load_model('models/survival.joblib')")
        return learned_map(img, model, **kw)
    raise ValueError(f"unknown mode {mode}")