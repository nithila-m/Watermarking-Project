"""Method-agnostic benchmark harness.

evaluate.py runs OUR configurations (uniform / heuristic / adaptive, with or
without ECC). This module instead takes any `embed_fn` / `extract_fn` pair, so
the same named attack conditions can score another method (or a mock) too.

    embed_fn(img, bits) -> watermarked img        (BGR uint8 arrays)
    extract_fn(img)      -> extracted bits (same length as bits)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .attacks import attack
from .config import BER_SUCCESS_THRESHOLD, PAYLOAD_BITS, TARGET_PSNR
from .core import grid_shape, load_image
from .embed import embed_at_psnr
from .extract import extract
from .metrics import ber, is_successful_extraction, psnr, ssim

# Named conditions reported in the benchmark table: (attack kind, severity).
# Maps human-readable experimental labels to internal attack dispatch tuples.
BENCHMARK_SUITE = {
    "Identity (No Attack)":        ("none", "none"),
    "JPEG (Q=70)":                 ("jpeg", "moderate"),
    "JPEG (Q=50)":                 ("jpeg", "severe"),
    "Gaussian Noise (std=12)":     ("noise", "severe"),
    "Gaussian Blur (sigma=1.6)":   ("blur", "severe"),
    "Crop 10% (in place)":         ("crop", "light"),
    "AI Inpaint Light (5%)":       ("inpaint_square", "light"),
    "AI Inpaint Moderate (15%)":   ("inpaint_square", "moderate"),
    "AI Inpaint Salient (20%)":    ("inpaint", "moderate"),
    "AI Neural Smoothing":         ("smooth", "moderate"),
}


def default_methods(key=2024, target_psnr=TARGET_PSNR):
    """(embed_fn, extract_fn) for our uniform watermark, calibrated to target_psnr.
    
    Args:
        key (int): Pseudo-random permutation seed for block assignment (default: 2024).
        target_psnr (float): Visual fidelity constraint in decibels (default: 42.0 dB).
        
    Returns:
        tuple[callable, callable]: Standardized (embed_fn, extract_fn) closures.
    """
    def embed_fn(img, bits):
        """Embeds payload uniformly across all 8x8 DCT blocks at target PSNR."""
        n_blocks = int(np.prod(grid_shape(img.shape)))
        # Pass a flat ones vector to apply uniform embedding weight across all blocks
        return embed_at_psnr(img, bits, np.ones(n_blocks), key, target_psnr)[0]

    def extract_fn(img):
        """Extracts bit sequence blindly from watermarked/attacked image."""
        return extract(img, PAYLOAD_BITS, key)

    return embed_fn, extract_fn


def run_benchmark(image_paths, embed_fn, extract_fn, payload_bits, suite=None,
                  threshold=BER_SUCCESS_THRESHOLD, seed=0, output_csv=None, raw_csv=None):
    """Run every condition on every image. Returns (summary_df, raw_df).
    
    Args:
        image_paths (iterable): File paths of test images to evaluate.
        embed_fn (callable): Embedding closure taking (img, bits) -> watermarked_img.
        extract_fn (callable): Extraction closure taking (img) -> extracted_bits.
        payload_bits (array-like): Ground truth binary sequence to embed.
        suite (dict, optional): Custom attack dictionary. Defaults to BENCHMARK_SUITE.
        threshold (float): Maximum Bit Error Rate to qualify as successful extraction.
        seed (int): Deterministic random seed for attack perturbations.
        output_csv (str or Path, optional): Output filepath for aggregated summary statistics.
        raw_csv (str or Path, optional): Output filepath for granular per-image records.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Aggregated summary table and raw trial records.
    """
    suite = suite or BENCHMARK_SUITE
    payload_bits = np.asarray(payload_bits)
    records = []

    # Iterate over every test image in the evaluation split
    for path in tqdm(list(image_paths), desc="benchmark"):
        clean = load_image(path)
        wm = embed_fn(clean, payload_bits)
        # Verify visual transparency metrics against host image
        p, s = psnr(clean, wm), ssim(clean, wm)

        # Subject watermarked image to every configured adversarial condition
        for name, (kind, sev) in suite.items():
            # Apply distortion (or bypass if identity condition)
            got = extract_fn(attack(wm, kind, sev, seed) if kind != "none" else wm)
            b = ber(payload_bits, got)
            records.append({"image": Path(path).stem, "condition": name, "attack": kind,
                            "severity": sev, "psnr": p, "ssim": s, "ber": b,
                            "success": is_successful_extraction(b, threshold)})

    # Compile raw trial observations into a structured DataFrame
    raw = pd.DataFrame(records)

    # Compute condition-level summary statistics preserving suite ordering
    summary = (raw.groupby("condition", sort=False)
                  .agg(psnr=("psnr", "mean"), ssim=("ssim", "mean"),
                       ber=("ber", "mean"), success_rate=("success", "mean"))
                  .reset_index())

    # Write results to disk if destination paths were supplied
    for path, df in ((output_csv, summary), (raw_csv, raw)):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)

    return summary, raw


def format_summary(summary):
    """Percent strings for BER and success rate, for printing or the report.
    
    Args:
        summary (pd.DataFrame): Aggregated benchmarking summary table.
        
    Returns:
        pd.DataFrame: Formatted DataFrame with percentages and rounded metrics.
    """
    out = summary.copy()
    # Convert proportions to percentage strings for reporting
    out["ber"] = (out["ber"] * 100).map("{:.2f}%".format)
    out["success_rate"] = (out["success_rate"] * 100).map("{:.2f}%".format)
    return out.round({"psnr": 2, "ssim": 4})
