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
# Standardized suite mapping descriptive condition titles to internal attack parameters.
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

    Provides baseline embedding and extraction closures where embedding strength is
    spread uniformly across all available DCT blocks.

    Args:
        key (int): Seed for deterministic block pseudo-random shuffling.
        target_psnr (float): Visual transparency threshold in dB.

    Returns:
        tuple[callable, callable]: A pair of (embed_fn, extract_fn) functions.
    """
    def embed_fn(img, bits):
        """Embeds bits uniformly across all 8x8 blocks at the targeted PSNR."""
        n_blocks = int(np.prod(grid_shape(img.shape)))
        # np.ones(n_blocks) applies equal weighting across all block positions
        return embed_at_psnr(img, bits, np.ones(n_blocks), key, target_psnr)[0]

    def extract_fn(img):
        """Blindly extracts payload bits from watermarked or attacked image."""
        return extract(img, PAYLOAD_BITS, key)

    return embed_fn, extract_fn


def run_benchmark(image_paths, embed_fn, extract_fn, payload_bits, suite=None,
                  threshold=BER_SUCCESS_THRESHOLD, seed=0, output_csv=None, raw_csv=None):
    """Run every condition on every image. Returns (summary_df, raw_df).

    Args:
        image_paths (iterable): Collection of file paths to clean source images.
        embed_fn (callable): Watermark embedding function taking (img, bits).
        extract_fn (callable): Extraction function taking an image array.
        payload_bits (array-like): Ground truth binary sequence to embed.
        suite (dict, optional): Attack dictionary mapping names to (kind, severity).
        threshold (float): Maximum bit error rate considered an extraction success.
        seed (int): Deterministic seed for reproducible attack executions.
        output_csv (Path or str, optional): File path to save the aggregate summary.
        raw_csv (Path or str, optional): File path to save per-image granular trial data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (summary_df, raw_df) containing evaluation metrics.
    """
    suite = suite or BENCHMARK_SUITE
    payload_bits = np.asarray(payload_bits)
    records = []

    # Iterate through all provided test images
    for path in tqdm(list(image_paths), desc="benchmark"):
        clean = load_image(path)
        wm = embed_fn(clean, payload_bits)
        # Evaluate visual imperceptibility of watermarked image against the clean host
        p, s = psnr(clean, wm), ssim(clean, wm)

        # Subject watermarked copy to each distortion condition in the benchmark suite
        for name, (kind, sev) in suite.items():
            got = extract_fn(attack(wm, kind, sev, seed) if kind != "none" else wm)
            b = ber(payload_bits, got)
            # Record individual experiment outcome
            records.append({"image": Path(path).stem, "condition": name, "attack": kind,
                            "severity": sev, "psnr": p, "ssim": s, "ber": b,
                            "success": is_successful_extraction(b, threshold)})

    # Construct complete tabular view of all trials
    raw = pd.DataFrame(records)

    # Aggregate performance across conditions (preserving original suite ordering)
    summary = (raw.groupby("condition", sort=False)
                  .agg(psnr=("psnr", "mean"), ssim=("ssim", "mean"),
                       ber=("ber", "mean"), success_rate=("success", "mean"))
                  .reset_index())

    # Write summary and raw tabular data to disk if destinations are provided
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
        pd.DataFrame: Formatted copy with percentage representations and rounded metrics.
    """
    out = summary.copy()
    # Format rates into human-readable percentage strings
    out["ber"] = (out["ber"] * 100).map("{:.2f}%".format)
    out["success_rate"] = (out["success_rate"] * 100).map("{:.2f}%".format)
    # Round fidelity metrics for clear reporting
    return out.round({"psnr": 2, "ssim": 4})
