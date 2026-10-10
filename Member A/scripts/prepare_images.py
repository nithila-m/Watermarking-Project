"""Resize a folder of images to IMG_SIZE x IMG_SIZE PNGs and write a train/test split.

Real data (DIV2K / COCO subset from B):
    python scripts/prepare_images.py --src data\raw --dst data\processed
Dev set from scikit-image samples (no download needed, for Day 1 testing):
    python scripts/prepare_images.py --samples --dst data\processed
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from watermark.config import IMG_SIZE

SAMPLES = ["astronaut", "camera", "chelsea", "coffee", "rocket", "immunohistochemistry",
           "hubble_deep_field", "moon", "brick", "grass", "gravel", "retina", "horse", "colorwheel"]


def center_resize(img, size):
    """Center-crop to a square, then resize."""
    h, w = img.shape[:2]
    m = min(h, w)
    y0, x0 = (h - m) // 2, (w - m) // 2
    return cv2.resize(img[y0:y0 + m, x0:x0 + m], (size, size), interpolation=cv2.INTER_AREA)


def sample_images(n_crops):
    """Sample images plus random crops/flips of them, so there are enough for a split."""
    import skimage.data as d
    rng = np.random.default_rng(0)
    for name in SAMPLES:
        img = getattr(d, name)()
        if img.dtype != np.uint8:
            img = (255 * (img > 0)).astype(np.uint8) if img.dtype == bool else img.astype(np.uint8)
        if img.ndim == 3:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        yield name, center_resize(img, IMG_SIZE)
        h, w = img.shape[:2]
        for k in range(n_crops):
            m = int(min(h, w) * rng.uniform(0.5, 0.85))
            y0, x0 = rng.integers(0, h - m + 1), rng.integers(0, w - m + 1)
            crop = img[y0:y0 + m, x0:x0 + m]
            if rng.random() < 0.5:
                crop = crop[:, ::-1]
            yield f"{name}_c{k}", cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)


def folder_images(src):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for p in sorted(Path(src).rglob("*")):
        if p.suffix.lower() in exts:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None and min(img.shape[:2]) >= IMG_SIZE // 2:
                yield p.stem, center_resize(img, IMG_SIZE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="folder of raw images")
    ap.add_argument("--samples", action="store_true", help="use scikit-image sample images")
    ap.add_argument("--crops", type=int, default=4, help="extra crops per sample image")
    ap.add_argument("--dst", default="data/processed")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="max images (0 = all)")
    args = ap.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    gen = sample_images(args.crops) if args.samples else folder_images(args.src)
    names = []
    for name, img in gen:
        cv2.imwrite(str(dst / f"{name}.png"), img)
        names.append(name)
        if args.limit and len(names) >= args.limit:
            break

    # Split by source image so crops of one photo never land in both train and test.
    groups = sorted({n.split("_c")[0] for n in names})
    rng = np.random.default_rng(42)
    rng.shuffle(groups)
    n_test = max(1, int(len(groups) * args.test_frac))
    test_groups = set(groups[:n_test])
    split = {"train": [n for n in names if n.split("_c")[0] not in test_groups],
             "test": [n for n in names if n.split("_c")[0] in test_groups]}
    (dst / "split.json").write_text(json.dumps(split, indent=1))
    print(f"wrote {len(names)} images to {dst} (train {len(split['train'])}, test {len(split['test'])})")


if __name__ == "__main__":
    main()
