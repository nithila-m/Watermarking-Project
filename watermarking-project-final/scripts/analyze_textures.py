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
    """Profiles the test split images by spatial texture complexity.
    
    Computes the Laplacian variance for each test image, categorizes them
    into thirds (Smooth, Medium, Complex), and saves the metadata to CSV.
    """
    # Ensure test dataset split exists and retrieve all test image identifiers
    names = dataset.require("test")
    print(f"analyzing {len(names)} test images for texture complexity...")
    
    # Measure texture variance across all test images using the Laplacian operator
    rows = [{"filename": f"{n}.png", "texture_variance": laplacian_variance(dataset.load("test", n))}
            for n in tqdm(names, desc="profiling")]
    df = pd.DataFrame(rows)
    
    # Discretize continuous texture variance values into Smooth, Medium, and Complex tiers
    df["texture_category"] = categorize(df["texture_variance"])
    
    # Ensure destination directory exists before writing metadata CSV
    TEXTURE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TEXTURE_CSV, index=False)
    
    # Display output destination and category distribution summary
    print(f"saved {TEXTURE_CSV}\n{df['texture_category'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
