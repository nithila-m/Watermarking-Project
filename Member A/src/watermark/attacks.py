"""Local attack suite so A can test without waiting for B.

B owns the official suite (including real Stable Diffusion inpainting run on
Colab). `inpaint` here uses OpenCV's Telea inpainting as a fast stand-in that
behaves similarly: it overwrites whole regions with content synthesised from
the surroundings.

All attacks keep the image size and alignment (crop = cut a region out and
fill it in place; resize = shrink then scale back). Geometric attacks that
shift the grid need a resynchronisation step and are out of scope for 2 weeks.
"""
import cv2
import numpy as np

SEVERITIES = ("light", "moderate", "severe")

PARAMS = {
    "jpeg":    {"light": 90,   "moderate": 70,   "severe": 50},    # quality
    "blur":    {"light": 0.6,  "moderate": 1.0,  "severe": 1.6},   # gaussian sigma
    "noise":   {"light": 3,    "moderate": 6,    "severe": 12},    # gaussian std (0-255)
    "resize":  {"light": 0.75, "moderate": 0.5,  "severe": 0.35},  # downscale factor
    "crop":    {"light": 0.10, "moderate": 0.25, "severe": 0.45},  # fraction of area removed
    "inpaint": {"light": 0.10, "moderate": 0.20, "severe": 0.35},  # fraction of area inpainted
}


def _jpeg(img, q, rng):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def _blur(img, sigma, rng):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def _noise(img, std, rng):
    return np.clip(img + rng.normal(0, std, img.shape), 0, 255).astype(np.uint8)


def _resize(img, f, rng):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _crop(img, frac, rng):
    """Keep a random window holding (1-frac) of the area; fill the rest with black."""
    h, w = img.shape[:2]
    keep = np.sqrt(1 - frac)
    kh, kw = int(h * keep), int(w * keep)
    y0, x0 = rng.integers(0, h - kh + 1), rng.integers(0, w - kw + 1)
    out = np.zeros_like(img)
    out[y0:y0 + kh, x0:x0 + kw] = img[y0:y0 + kh, x0:x0 + kw]
    return out


def edit_mask(img, frac, rng, salient=True):
    """Rectangles until about `frac` of the image is covered.

    salient=True centres the rectangles on salient points, because real AI edits
    usually remove or replace the subject, not random background. B's real masks
    (object masks fed to Stable Diffusion) should replace this for final numbers.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if salient:
        from .features import _saliency_map
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p = (_saliency_map(gray) ** 2).ravel().astype(np.float64)
        p /= p.sum()
    while mask.mean() / 255 < frac:
        rh, rw = rng.integers(h // 10, h // 4), rng.integers(w // 10, w // 4)
        if salient:
            cy, cx = divmod(int(rng.choice(h * w, p=p)), w)
            y0, x0 = int(np.clip(cy - rh // 2, 0, h - rh)), int(np.clip(cx - rw // 2, 0, w - rw))
        else:
            y0, x0 = rng.integers(0, h - rh), rng.integers(0, w - rw)
        mask[y0:y0 + rh, x0:x0 + rw] = 255
    return mask


def _inpaint(img, frac, rng):
    mask = edit_mask(img, frac, rng)
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


_FUNCS = {"jpeg": _jpeg, "blur": _blur, "noise": _noise,
          "resize": _resize, "crop": _crop, "inpaint": _inpaint}


def attack(img, kind, severity="moderate", seed=0):
    """Apply one attack. kind='none' returns the image unchanged."""
    if kind == "none":
        return img.copy()
    rng = np.random.default_rng(seed)
    return _FUNCS[kind](img, PARAMS[kind][severity], rng)


def all_attacks():
    """[(kind, severity), ...] for every attack in the suite."""
    return [(k, s) for k in PARAMS for s in SEVERITIES]
