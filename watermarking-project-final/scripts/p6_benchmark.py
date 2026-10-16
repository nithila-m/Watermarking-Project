"""Phase 6: named benchmark table (PSNR, SSIM, BER, success rate) on the TEST split.

    python scripts/p6_benchmark.py --n 50
    python scripts/p6_benchmark.py --mock            # check the harness with a fake method

Uses the method-agnostic harness in watermark/benchmark.py with our uniform
watermark (success = raw BER <= 15%). For the uniform-vs-adaptive comparison use
p5_ablation.py. Outputs in results/:
  p6_summary.csv      one row per named condition
  p6_raw.csv          one row per image x condition
  p6_by_texture.csv   success rate per condition x texture category (if available)
"""
import argparse
from pathlib import Path

import numpy as np

from watermark import dataset
from watermark.benchmark import default_methods, format_summary, run_benchmark
from watermark.config import PAYLOAD_BITS, TEXTURE_CSV
from watermark.textures import load_texture_table

# Output root directory where benchmark tables and CSV logs will be stored
OUT = Path("results")


def mock_methods(payload, rng):
    """A fake watermark: adds light noise, 'extracts' the payload with occasional flips.

    Used for quick dry-runs and pipeline sanity checks without executing the full
    DWT-DCT transform or PSNR optimization loops.

    Args:
        payload (np.ndarray): Binary sequence used as the target watermark.
        rng (np.random.Generator): Seeded random number generator.

    Returns:
        tuple[callable, callable]: Lightweight mock (embed_fn, extract_fn) closures.
    """
    def embed_fn(img, bits):
        """Mock embedder: introduces minimal Gaussian noise while preserving uint8 range."""
        return np.clip(img + rng.normal(0, 1.0, img.shape), 0, 255).astype(np.uint8)

    def extract_fn(img):
        """Mock extractor: returns payload with sporadic random bit inversions."""
        got = payload.copy()
        if rng.random() > 0.6:
            i = rng.integers(len(got))
            got[i] = 1 - got[i]  # Invert a single bit at index i
        return got

    return embed_fn, extract_fn


def main():
    """Main execution function for running Phase 6 benchmarking on test images."""
    # Parse CLI flags for sample size limit and dry-run mock mode
    ap = argparse.ArgumentParser(description="Run Phase 6 benchmarking across attack conditions.")
    ap.add_argument("--n", type=int, default=0, help="number of test images (0 = all)")
    ap.add_argument("--mock", action="store_true", help="use a fake embed/extract to test the harness")
    args = ap.parse_args()

    # Load image file paths from the frozen test split
    names = dataset.require("test")
    names = names[: args.n] if args.n else names
    paths = [dataset.image_path("test", n) for n in names]

    # Initialize reproducible ground-truth binary payload
    rng = np.random.default_rng(0)
    payload = rng.integers(0, 2, PAYLOAD_BITS)

    # Choose between fast mock routines and actual DWT-DCT baseline methods
    embed_fn, extract_fn = mock_methods(payload, rng) if args.mock else default_methods()

    # Execute standardized attack suite and log extraction metrics
    tag = "p6_mock" if args.mock else "p6"
    summary, raw = run_benchmark(paths, embed_fn, extract_fn, payload,
                                 output_csv=OUT / f"{tag}_summary.csv", raw_csv=OUT / f"{tag}_raw.csv")

    # Output aggregate benchmark results to stdout
    print(format_summary(summary).to_string(index=False))

    # Stratify survival rates across Smooth, Medium, and Complex texture tiers if metadata exists
    if TEXTURE_CSV.exists():
        tex = load_texture_table(TEXTURE_CSV)["texture_category"]
        raw["texture"] = raw.image.map(tex)

        # Build pivot table: attack condition rows vs texture category columns
        by_tex = raw.pivot_table(index="condition", columns="texture", values="success", sort=False)
        # Maintain consistent column ordering from least to most textured
        by_tex = by_tex[[c for c in ("Smooth", "Medium", "Complex") if c in by_tex]]
        by_tex.round(4).to_csv(OUT / f"{tag}_by_texture.csv")

        print("\nSuccess rate by texture:")
        print(by_tex.round(3).to_string())

    print(f"\nsaved results/{tag}_*.csv")


if __name__ == "__main__":
    main()
