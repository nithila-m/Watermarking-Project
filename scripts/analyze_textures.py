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
    names = dataset.require("test")
    print(f"analyzing {len(names)} test images for texture complexity...")
    rows = [{"filename": f"{n}.png", "texture_variance": laplacian_variance(dataset.load("test", n))}
            for n in tqdm(names, desc="profiling")]
    df = pd.DataFrame(rows)
    df["texture_category"] = categorize(df["texture_variance"])
    TEXTURE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TEXTURE_CSV, index=False)
    print(f"saved {TEXTURE_CSV}\n{df['texture_category'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
