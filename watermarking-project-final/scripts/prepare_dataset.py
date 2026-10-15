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

# Filtering thresholds: exclude extreme panoramic or vertical aspect ratios
MIN_ASPECT, MAX_ASPECT = 0.60, 1.67
# Allowed image extensions for dataset discovery
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# Predefined sample images available in scikit-image for standalone development
SAMPLES = ["astronaut", "camera", "chelsea", "coffee", "rocket", "immunohistochemistry",
           "hubble_deep_field", "moon", "brick", "grass", "gravel", "retina", "horse", "colorwheel"]


def center_square(img: Image.Image, size=IMG_SIZE) -> Image.Image:
    """Center-crops a PIL Image to a square bounding box and resizes it.
    
    Args:
        img (Image.Image): Input image to crop.
        size (int): Final output edge dimension in pixels (default: IMG_SIZE = 512).
        
    Returns:
        Image.Image: Center-cropped and resized square image using Lanczos interpolation.
    """
    w, h = img.size
    m = min(w, h)
    # Determine top-left anchor to center the square crop
    left, top = (w - m) // 2, (h - m) // 2
    return img.crop((left, top, left + m, top + m)).resize((size, size), Image.Resampling.LANCZOS)


def prepare_folder(src, dst, fracs, limit):
    """Processes a local raw image folder, filters by dimensions, and partitions into splits.
    
    Args:
        src (str or Path): Source folder containing raw input images.
        dst (Path): Destination root folder where split subdirectories reside.
        fracs (tuple): Proportion split for (train, val, test).
        limit (int): Optional cap on total valid images to process (0 for unlimited).
    """
    # Recursively find and sort all image paths to guarantee deterministic ordering
    paths = sorted(p for p in Path(src).rglob("*") if p.suffix.lower() in EXTS)
    print(f"found {len(paths)} raw images, filtering by aspect ratio...")
    valid = []
    
    # Filter candidates by aspect ratio and minimum resolution to prevent upscaling blur
    for p in tqdm(paths, desc="filter"):
        try:
            with Image.open(p) as im:
                w, h = im.size
            if MIN_ASPECT <= w / h <= MAX_ASPECT and min(w, h) >= IMG_SIZE // 2:
                valid.append(p)
        except Exception:
            continue
            
    # Apply optional cap if specified
    if limit:
        valid = valid[:limit]
        
    # Calculate partition boundaries
    n_train, n_val = int(len(valid) * fracs[0]), int(len(valid) * fracs[1])
    parts = {"train": valid[:n_train], "val": valid[n_train:n_train + n_val],
             "test": valid[n_train + n_val:]}
             
    # Crop, format, and save as lossless numbered PNGs (000001.png, ...)
    for split, items in parts.items():
        for i, p in enumerate(tqdm(items, desc=f"save {split}")):
            try:
                with Image.open(p) as im:
                    center_square(im.convert("RGB")).save(dst / split / f"{i + 1:06d}.png")
            except Exception as e:
                print(f"skipping {p}: {e}")


def sample_images(n_crops):
    """Generates resized base scikit-image samples and randomized sub-crops.
    
    Args:
        n_crops (int): Number of randomized crops to generate per base sample.
        
    Yields:
        tuple[str, np.ndarray]: Image identifier and corresponding BGR image array.
    """
    import skimage.data as d
    rng = np.random.default_rng(0)
    for name in SAMPLES:
        # Load sample array dynamically from scikit-image data module
        img = getattr(d, name)()
        # Normalize dtype to uint8
        if img.dtype != np.uint8:
            img = (255 * (img > 0)).astype(np.uint8) if img.dtype == bool else img.astype(np.uint8)
        # Standardize color channels to OpenCV BGR representation
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        m = min(h, w)
        # Extract base center square crop
        sq = img[(h - m) // 2:(h - m) // 2 + m, (w - m) // 2:(w - m) // 2 + m]
        yield name, cv2.resize(sq, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        # Generate random sub-crops with optional horizontal flip for data augmentation
        for k in range(n_crops):
            c = int(m * rng.uniform(0.5, 0.85))
            y0, x0 = rng.integers(0, h - c + 1), rng.integers(0, w - c + 1)
            crop = img[y0:y0 + c, x0:x0 + c]
            if rng.random() < 0.5:
                crop = crop[:, ::-1]  # Horizontal flip
            yield f"{name}_c{k}", cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)


def prepare_samples(dst, n_crops, fracs):
    """Partitions scikit-image samples by source identity to prevent split leakage.
    
    Args:
        dst (Path): Destination root folder where split subdirectories reside.
        n_crops (int): Number of extra crops generated per base sample.
        fracs (tuple): Proportion split for (train, val, test).
    """
    items = list(sample_images(n_crops))
    # Group by original base image name to ensure all crops of an image stay in the same split
    groups = sorted({n.split("_c")[0] for n, _ in items})
    rng = np.random.default_rng(42)
    rng.shuffle(groups)
    
    # Calculate partition counts based on groups
    n_test = max(1, round(len(groups) * fracs[2]))
    n_val = round(len(groups) * fracs[1])
    test_g, val_g = set(groups[:n_test]), set(groups[n_test:n_test + n_val])
    
    # Write images to their designated split directories
    for name, img in items:
        g = name.split("_c")[0]
        split = "test" if g in test_g else "val" if g in val_g else "train"
        cv2.imwrite(str(dst / split / f"{name}.png"), img)


def main():
    """CLI entry point for preparing and partitioning the watermarking dataset."""
    ap = argparse.ArgumentParser(description="Prepare standardized 512x512 dataset splits.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--src", help="folder of raw images (e.g. raw/val2017)")
    src.add_argument("--samples", action="store_true", help="use scikit-image sample images")
    ap.add_argument("--dst", default=str(DATA_DIR), help="output destination directory")
    ap.add_argument("--fracs", type=float, nargs=3, default=(0.70, 0.10, 0.20),
                    metavar=("TRAIN", "VAL", "TEST"), help="train, val, test split proportions")
    ap.add_argument("--crops", type=int, default=4, help="extra crops per sample image (--samples)")
    ap.add_argument("--limit", type=int, default=0, help="max images (0 = all, --src only)")
    ap.add_argument("--clean", action="store_true", help="empty the split folders first")
    args = ap.parse_args()

    dst = Path(args.dst)
    # Ensure destination directories exist; optionally wipe prior contents
    for s in SPLITS:
        if args.clean and (dst / s).exists():
            shutil.rmtree(dst / s)
        (dst / s).mkdir(parents=True, exist_ok=True)

    # Route dataset generation based on CLI input mode
    if args.samples:
        prepare_samples(dst, args.crops, args.fracs)
    else:
        prepare_folder(args.src, dst, args.fracs, args.limit)

    # Output dataset summary report
    for s in SPLITS:
        print(f"  {s:5s}: {len(list((dst / s).glob('*.png')))} images")
    print("next: python scripts\\analyze_textures.py")


if __name__ == "__main__":
    main()
