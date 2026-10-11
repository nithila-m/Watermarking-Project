"""Attack suite: classical distortions + simulated AI edits, all on BGR uint8 arrays.

Every attack keeps the image size. Most also keep the block grid aligned
(crop = cut a region out and fill it in place; resize = shrink then scale back).

    classical : jpeg, blur, noise, resize, crop
    AI edits  : inpaint         saliency-targeted OpenCV Telea inpainting (stand-in
                                for diffusion inpainting, which usually replaces the subject)
                inpaint_square  one random square inpainted (background or subject)
                smooth          edge-preserving bilateral filter, simulating AI
                                "enhance"/beauty filters that flatten fine texture
    geometric : crop_resize     cut the borders, then scale back up. This shifts the
                                block grid, so the blind extractor needs a resync
                                step that is out of scope; it is NOT in the default
                                suite (all_attacks(include_geometric=True) adds it).

Real Stable Diffusion edits are produced on Colab and fed in as external images
(see scripts/p3_build_dataset.py --external).
"""
import cv2
import numpy as np

SEVERITIES = ("light", "moderate", "severe")

PARAMS = {
    # classical
    "jpeg":    {"light": 90,   "moderate": 70,   "severe": 50},    # quality
    "blur":    {"light": 0.6,  "moderate": 1.0,  "severe": 1.6},   # gaussian sigma
    "noise":   {"light": 3,    "moderate": 6,    "severe": 12},    # gaussian std (0-255)
    "resize":  {"light": 0.75, "moderate": 0.5,  "severe": 0.35},  # downscale factor
    "crop":    {"light": 0.10, "moderate": 0.25, "severe": 0.45},  # fraction of area removed
    # AI edits
    "inpaint":        {"light": 0.10, "moderate": 0.20, "severe": 0.35},  # fraction inpainted (salient)
    "inpaint_square": {"light": 0.05, "moderate": 0.15, "severe": 0.30},  # fraction inpainted (one square)
    "smooth":         {"light": (5, 50), "moderate": (9, 75), "severe": (13, 120)},  # bilateral (d, sigma)
    # geometric (not in the default suite)
    "crop_resize":    {"light": 0.05, "moderate": 0.10, "severe": 0.15},  # border fraction cut per side
}
GEOMETRIC = {"crop_resize"}
NOISE_TYPE = ("jpeg", "blur", "resize", "noise", "smooth")   # disturb every block a little
ERASE_TYPE = ("inpaint", "inpaint_square", "crop")           # wipe out whole regions


# ---------------- classical ----------------
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


def _crop_resize(img, frac, rng):
    """Cut `frac` of the width/height off every border, then scale back to full size."""
    h, w = img.shape[:2]
    dh, dw = int(h * frac), int(w * frac)
    return cv2.resize(img[dh:h - dh, dw:w - dw], (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------- AI edits ----------------
def edit_mask(img, frac, rng, salient=True):
    """Rectangles until about `frac` of the image is covered.

    salient=True centres the rectangles on salient points, because real AI edits
    usually remove or replace the subject, not random background. Real object
    masks fed to Stable Diffusion should replace this for final numbers.
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


def square_mask(img, frac, rng):
    """One random square covering `frac` of the image (0 = keep, 255 = inpaint)."""
    h, w = img.shape[:2]
    side = int(np.sqrt(h * w * frac))
    y0 = rng.integers(0, max(1, h - side))
    x0 = rng.integers(0, max(1, w - side))
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y0 + side, x0:x0 + side] = 255
    return mask


def _inpaint(img, frac, rng):
    return cv2.inpaint(img, edit_mask(img, frac, rng), 5, cv2.INPAINT_TELEA)


def _inpaint_square(img, frac, rng):
    return cv2.inpaint(img, square_mask(img, frac, rng), 5, cv2.INPAINT_TELEA)


def _smooth(img, params, rng):
    d, sigma = params
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigma, sigmaSpace=sigma)


_FUNCS = {"jpeg": _jpeg, "blur": _blur, "noise": _noise, "resize": _resize, "crop": _crop,
          "inpaint": _inpaint, "inpaint_square": _inpaint_square, "smooth": _smooth,
          "crop_resize": _crop_resize}


def attack(img, kind, severity="moderate", seed=0):
    """Apply one attack. kind='none' returns the image unchanged."""
    if kind == "none":
        return img.copy()
    rng = np.random.default_rng(seed)
    return _FUNCS[kind](img, PARAMS[kind][severity], rng)


def all_attacks(include_geometric=False):
    """[(kind, severity), ...] for every attack in the suite."""
    return [(k, s) for k in PARAMS for s in SEVERITIES
            if include_geometric or k not in GEOMETRIC]
