"""Method-agnostic benchmark harness.

evaluate.py runs OUR configurations (uniform / heuristic / adaptive, with or
without ECC). This module instead takes any `embed_fn` / `extract_fn` pair, so
the same named attack conditions can score another method (or a mock) too.

    embed_fn(img, bits) -> watermarked img        (BGR uint8 arrays)
    extract_fn(img)     -> extracted bits (same length as bits)
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
    """(embed_fn, extract_fn) for our uniform watermark, calibrated to target_psnr."""
    def embed_fn(img, bits):
        n_blocks = int(np.prod(grid_shape(img.shape)))
        return embed_at_psnr(img, bits, np.ones(n_blocks), key, target_psnr)[0]

    def extract_fn(img):
        return extract(img, PAYLOAD_BITS, key)

    return embed_fn, extract_fn


def run_benchmark(image_paths, embed_fn, extract_fn, payload_bits, suite=None,
                  threshold=BER_SUCCESS_THRESHOLD, seed=0, output_csv=None, raw_csv=None):
    """Run every condition on every image. Returns (summary_df, raw_df)."""
    suite = suite or BENCHMARK_SUITE
    payload_bits = np.asarray(payload_bits)
    records = []
    for path in tqdm(list(image_paths), desc="benchmark"):
        clean = load_image(path)
        wm = embed_fn(clean, payload_bits)
        p, s = psnr(clean, wm), ssim(clean, wm)
        for name, (kind, sev) in suite.items():
            got = extract_fn(attack(wm, kind, sev, seed) if kind != "none" else wm)
            b = ber(payload_bits, got)
            records.append({"image": Path(path).stem, "condition": name, "attack": kind,
                            "severity": sev, "psnr": p, "ssim": s, "ber": b,
                            "success": is_successful_extraction(b, threshold)})
    raw = pd.DataFrame(records)
    summary = (raw.groupby("condition", sort=False)
                  .agg(psnr=("psnr", "mean"), ssim=("ssim", "mean"),
                       ber=("ber", "mean"), success_rate=("success", "mean"))
                  .reset_index())
    for path, df in ((output_csv, summary), (raw_csv, raw)):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
    return summary, raw


def format_summary(summary):
    """Percent strings for BER and success rate, for printing or the report."""
    out = summary.copy()
    out["ber"] = (out["ber"] * 100).map("{:.2f}%".format)
    out["success_rate"] = (out["success_rate"] * 100).map("{:.2f}%".format)
    return out.round({"psnr": 2, "ssim": 4})
