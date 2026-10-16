"""Phase 2: first end-to-end numbers. Uniform watermark -> every attack -> extract -> BER.

    python scripts/p2_end_to_end.py --n 50
    python scripts/p2_end_to_end.py --export-for-b      # also save watermarked PNGs for the Colab diffusion edits

Writes results/p2_baseline.csv and prints a table by attack and severity.
These numbers are the CONTROL row for the whole project.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from watermark import dataset
from watermark.core import save_image
from watermark.evaluate import CONFIGS, evaluate_image
from watermark.predictor import export_for_attack

# Output root directory for benchmark logs and exported images
OUT = Path("results")


def main():
    """Executes the Phase 2 baseline evaluation pipeline.

    Evaluates uniform baseline watermarking (both raw and ECC-protected) across
    all standard distortion conditions to establish project-wide control metrics.
    Optionally exports watermarked train-set images for external Colab GPU diffusion edits.
    """
    ap = argparse.ArgumentParser(description="Run Phase 2 end-to-end baseline evaluation.")
    ap.add_argument("--n", type=int, default=50, help="number of images")
    ap.add_argument("--export-for-b", action="store_true",
                    help="save the watermarked training images to results/for_b/ for real AI edits (Colab)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # Gather image identifiers from available splits, constrained by the limit flag --n
    pairs = ([("train", n) for n in dataset.split_names("train")] +
             [("test", n) for n in dataset.split_names("test")])[: args.n]
    if not pairs:
        dataset.require("train")

    # Isolate control configurations: uniform uncoded and uniform with Reed-Solomon ECC
    configs = {k: CONFIGS[k] for k in ("uniform", "uniform+ecc")}

    # Iterate through selected images and evaluate every attack/severity condition
    rows = []
    for split, n in tqdm(pairs, desc="images"):
        rows += evaluate_image(dataset.load(split, n), f"{split}/{n}", configs)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p2_baseline.csv", index=False)

    # Pivot into aggregate summary table indexed by attack and severity level
    table = df.pivot_table(index=["attack", "severity"], columns="config",
                           values=["bit_acc", "success"], aggfunc="mean").round(3)
    print(table.to_string())
    print(f"\nmean PSNR {df.psnr.mean():.2f} dB, SSIM {df.ssim.mean():.4f}")
    print(f"saved {OUT / 'p2_baseline.csv'}")

    # Optionally export watermarked train-split PNGs for Colab-based diffusion inpainting/regeneration
    if args.export_for_b:
        dst = OUT / "for_b"
        dst.mkdir(exist_ok=True)
        for n in tqdm(dataset.split_names("train"), desc="export"):
            # Uniformly embed watermark using standard training key and PSNR calibration
            save_image(dst / f"{n}.png", export_for_attack(dataset.load("train", n)))
        print(f"watermarked images for the diffusion edits are in {dst}. Return edited "
              f"versions with the SAME file names in data/external/<edit_name>/")


if __name__ == "__main__":
    main()
