import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

RAW_DIR = Path("raw/val2017")
CLEAN_DIR = Path("data/clean")

TARGET_SIZE = (512, 512)
MIN_ASPECT_RATIO = 0.60
MAX_ASPECT_RATIO = 1.67

def smart_crop_and_resize(img: Image.Image, target_size=(512, 512)) -> Image.Image:
    """Center crops image to square and resizes to 512x512."""
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize(target_size, Image.Resampling.LANCZOS)

def process_dataset():
    # Automatically create output folders if they do not exist
    for split in ["train", "val", "test"]:
        (CLEAN_DIR / split).mkdir(parents=True, exist_ok=True)

    all_raw_images = sorted(list(RAW_DIR.glob("*.jpg")))
    print(f"Found {len(all_raw_images)} raw images. Filtering...")

    valid_images = []
    for img_path in tqdm(all_raw_images, desc="Filtering"):
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                ratio = w / h
                if MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
                    valid_images.append(img_path)
        except Exception:
            continue

    total_valid = len(valid_images)
    print(f"Passed filter: {total_valid} images.")

    # Dynamic split based on available images: 70% train, 10% val, 20% test
    n_train = int(total_valid * 0.70)
    n_val = int(total_valid * 0.10)

    split_counts = {
        "train": valid_images[:n_train],
        "val": valid_images[n_train:n_train + n_val],
        "test": valid_images[n_train + n_val:]
    }

    for split_name, img_paths in split_counts.items():
        dest_folder = CLEAN_DIR / split_name
        for i, img_path in enumerate(tqdm(img_paths, desc=f"Saving {split_name}")):
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    processed = smart_crop_and_resize(img, TARGET_SIZE)
                    out_path = dest_folder / f"{i+1:06d}.png"
                    processed.save(out_path, format="PNG")
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    print("\nDataset ready!")
    for split in ["train", "val", "test"]:
        count = len(list((CLEAN_DIR / split).glob("*.png")))
        print(f" - {split}: {count} images")

if __name__ == "__main__":
    process_dataset()