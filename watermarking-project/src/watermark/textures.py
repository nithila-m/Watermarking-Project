"""Image-level texture profiling, used to split results into Smooth / Medium / Complex images."""
import cv2
import numpy as np
import pandas as pd

CATEGORIES = ["Smooth", "Medium", "Complex"]


def laplacian_variance(img):
    """Variance of the Laplacian of the grayscale image: higher = more edges and texture."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def categorize(variances):
    """Three equal-size bins (tertiles) of texture variance."""
    return pd.qcut(np.asarray(variances), q=3, labels=CATEGORIES)


def load_texture_table(path):
    """CSV with columns filename, texture_variance, texture_category -> indexed by image name."""
    df = pd.read_csv(path)
    df["image"] = df["filename"].str.replace(r"\.png$", "", regex=True)
    return df.set_index("image")
