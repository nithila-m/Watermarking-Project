from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

TEST_DIR = Path("data/clean/test")
OUT_CSV = Path("data/clean/test_metadata.csv")

def compute_spatial_variance(img_path):
    """Computes Laplacian variance to measure edge and texture detail."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return float(laplacian.var())

def run():
    img_files = sorted(list(TEST_DIR.glob("*.png")))
    print(f"Analyzing {len(img_files)} test images for texture complexity...")
    
    records = []
    for p in tqdm(img_files, desc="Profiling"):
        var_score = compute_spatial_variance(p)
        records.append({
            "filename": p.name,
            "texture_variance": var_score
        })
        
    df = pd.DataFrame(records)
    
    # Split into 3 equal bins: Low (smooth), Medium, High (complex/textured)
    df["texture_category"] = pd.qcut(
        df["texture_variance"], 
        q=3, 
        labels=["Smooth", "Medium", "Complex"]
    )
    
    df.to_csv(OUT_CSV, index=False)
    print(f"\nMetadata saved to {OUT_CSV}")
    print("\nCategory distribution:")
    print(df["texture_category"].value_counts())

if __name__ == "__main__":
    run()