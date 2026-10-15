"""Profile the TEST images by texture (Laplacian variance) into Smooth / Medium / Complex thirds.

    python scripts/analyze_textures.py

Writes data/clean/test_metadata.csv, used by p5_ablation.py and p6_benchmark.py
to report results per texture category.
"""
import pandas as pd
from tqdm import tqdm

from watermark import dataset
from watermark.config import TEXTURE_CSV
from watermark.textures import categorize, laplacian_variance


def main():
    """Main execution entry point to compute texture variance on test images and save metadata."""
    # Retrieve all valid base filenames from the frozen test split
    names = dataset.require("test")
    print(f"analyzing {len(names)} test images for texture complexity...")

    # Calculate spatial Laplacian variance (sigma^2 of second spatial derivatives) for each image
    rows = [{"filename": f"{n}.png", "texture_variance": laplacian_variance(dataset.load("test", n))}
            for n in tqdm(names, desc="profiling")]
    df = pd.DataFrame(rows)

    # Discretize continuous variance values into quantile-based thirds: Smooth, Medium, Complex
    df["texture_category"] = categorize(df["texture_variance"])

    # Ensure destination parent directory (data/clean/) exists before writing
    TEXTURE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TEXTURE_CSV, index=False)

    # Print output path and distribution count across categories
    print(f"saved {TEXTURE_CSV}\n{df['texture_category'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
