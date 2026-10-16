"""Image-level texture profiling, used to split results into Smooth / Medium / Complex images.

Computes global high-frequency edge energy via Laplacian variance across grayscale
imagery, establishing an objective criterion to partition host images into three
empirical complexity tiers (tertiles):
  - Smooth  : Flat regions, low edge density, minimal natural texture masking.
  - Medium  : Balanced structural details and standard visual entropy.
  - Complex : High frequency content, dense textures, strong masking potential.
"""
import cv2
import numpy as np
import pandas as pd

# Standard 3-tier texture complexity categories
CATEGORIES = ["Smooth", "Medium", "Complex"]


def laplacian_variance(img):
    """Variance of the Laplacian of the grayscale image: higher = more edges and texture.

    Converts BGR imagery to single-channel luminance/grayscale if required, applies
    a second-order 2D differential Laplacian kernel (CV_64F), and calculates the
    variance of the resulting response map.

    Args:
        img (np.ndarray): Input image array (BGR uint8 or 2-D grayscale uint8).

    Returns:
        float: Scalar variance of the Laplacian filter response.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def categorize(variances):
    """Three equal-size bins (tertiles) of texture variance.

    Applies quantile-based discretization (qcut) to partition a distribution of
    Laplacian variance values into three balanced categories: Smooth, Medium, Complex.

    Args:
        variances (array-like): Sequence or 1-D array of numerical variance values.

    Returns:
        pd.Series or pd.Categorical: Categorical labels corresponding to CATEGORIES.
    """
    return pd.qcut(np.asarray(variances), q=3, labels=CATEGORIES)


def load_texture_table(path):
    """CSV with columns filename, texture_variance, texture_category -> indexed by image name.

    Reads the precomputed texture classification index, strips '.png' extensions to
    extract clean image stems, and indexes the DataFrame by the image name.

    Args:
        path (str or Path): Filepath to the CSV texture registry (TEXTURE_CSV).

    Returns:
        pd.DataFrame: Table containing texture statistics indexed by image stem.
    """
    df = pd.read_csv(path)
    # Strip file extension to align with internal stem identifiers across splits
    df["image"] = df["filename"].str.replace(r"\.png$", "", regex=True)
    return df.set_index("image")
