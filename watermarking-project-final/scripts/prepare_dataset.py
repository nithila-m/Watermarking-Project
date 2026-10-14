r"""Build data/clean/{train,val,test}/ : 512x512 PNGs, center-cropped.

Real data (COCO val2017, or any folder of images):
    python scripts\prepare_dataset.py --src raw\val2017
      keeps images with aspect ratio 0.60-1.67, sorted by file name,
      split 70 / 10 / 20 % into train / val / test, saved as 000001.png, 000002.png, ...

Dev set from scikit-image samples (no download needed, for Day-1 testing):
    python scripts\prepare_dataset.py --samples
      sample images plus random crops/flips; split by SOURCE image so crops of
      one photo never land in both train and test.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from watermark.config import DATA_DIR, IMG_SIZE, SPLITS

MIN_ASPECT, MAX_ASPECT = 0.60, 1.67
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SAMPLES = ["astronaut", "camera", "chelsea", "coffee", "rocket", "immunohistochemistry",
           "hubble_deep_field", "moon", "brick", "grass", "gravel", "retina", "horse", "colorwheel"]


def center_square(img: Image.Image, size=IMG_SIZE) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    left, top = (w - m) // 2, (h - m) // 2
    return img.crop((left, top, left + m, top + m)).resize((size, size), Image.Resampling.LANCZOS)


def prepare_folder(src, dst, fracs, limit):
    paths = sorted(p for p in Path(src).rglob("*") if p.suffix.lower() in EXTS)
    print(f"found {len(paths)} raw images, filtering by aspect ratio...")
    valid = []
    for p in tqdm(paths, desc="filter"):
        try:
            with Image.open(p) as im:
                w, h = im.size
            if MIN_ASPECT <= w / h <= MAX_ASPECT and min(w, h) >= IMG_SIZE // 2:
                valid.append(p)
        except Exception:
            continue
    if limit:
        valid = valid[:limit]
    n_train, n_val = int(len(valid) * fracs[0]), int(len(valid) * fracs[1])
    parts = {"train": valid[:n_train], "val": valid[n_train:n_train + n_val],
             "test": valid[n_train + n_val:]}
    for split, items in parts.items():
        for i, p in enumerate(tqdm(items, desc=f"save {split}")):
            try:
                with Image.open(p) as im:
                    center_square(im.convert("RGB")).save(dst / split / f"{i + 1:06d}.png")
            except Exception as e:
                print(f"skipping {p}: {e}")


def sample_images(n_crops):
    """scikit-image samples (BGR) plus random crops/flips of them."""
    import skimage.data as d
    rng = np.random.default_rng(0)
    for name in SAMPLES:
        img = getattr(d, name)()
        if img.dtype != np.uint8:
            img = (255 * (img > 0)).astype(np.uint8) if img.dtype == bool else img.astype(np.uint8)
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        m = min(h, w)
        sq = img[(h - m) // 2:(h - m) // 2 + m, (w - m) // 2:(w - m) // 2 + m]
        yield name, cv2.resize(sq, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        for k in range(n_crops):
            c = int(m * rng.uniform(0.5, 0.85))
            y0, x0 = rng.integers(0, h - c + 1), rng.integers(0, w - c + 1)
            crop = img[y0:y0 + c, x0:x0 + c]
            if rng.random() < 0.5:
                crop = crop[:, ::-1]
            yield f"{name}_c{k}", cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)


def prepare_samples(dst, n_crops, fracs):
    items = list(sample_images(n_crops))
    groups = sorted({n.split("_c")[0] for n, _ in items})
    rng = np.random.default_rng(42)
    rng.shuffle(groups)
    n_test = max(1, round(len(groups) * fracs[2]))
    n_val = round(len(groups) * fracs[1])
    test_g, val_g = set(groups[:n_test]), set(groups[n_test:n_test + n_val])
    for name, img in items:
        g = name.split("_c")[0]
        split = "test" if g in test_g else "val" if g in val_g else "train"
        cv2.imwrite(str(dst / split / f"{name}.png"), img)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--src", help="folder of raw images (e.g. raw/val2017)")
    src.add_argument("--samples", action="store_true", help="use scikit-image sample images")
    ap.add_argument("--dst", default=str(DATA_DIR))
    ap.add_argument("--fracs", type=float, nargs=3, default=(0.70, 0.10, 0.20),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--crops", type=int, default=4, help="extra crops per sample image (--samples)")
    ap.add_argument("--limit", type=int, default=0, help="max images (0 = all, --src only)")
    ap.add_argument("--clean", action="store_true", help="empty the split folders first")
    args = ap.parse_args()

    dst = Path(args.dst)
    for s in SPLITS:
        if args.clean and (dst / s).exists():
            shutil.rmtree(dst / s)
        (dst / s).mkdir(parents=True, exist_ok=True)

    if args.samples:
        prepare_samples(dst, args.crops, args.fracs)
    else:
        prepare_folder(args.src, dst, args.fracs, args.limit)

    for s in SPLITS:
        print(f"  {s:5s}: {len(list((dst / s).glob('*.png')))} images")
    print("next: python scripts\\analyze_textures.py")


if __name__ == "__main__":
    main()
