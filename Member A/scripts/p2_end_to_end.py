"""Phase 2: first end-to-end numbers. Uniform watermark -> every attack -> extract -> BER.

    python scripts/p2_end_to_end.py --n 50
    python scripts/p2_end_to_end.py --export-for-b      # also save watermarked PNGs for B's AI edits

Writes results/p2_baseline.csv and prints a table by attack and severity.
These numbers are the CONTROL row for the whole project.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from watermark.core import load_image, save_image
from watermark.evaluate import CONFIGS, evaluate_image
from watermark.predictor import export_for_attack

DATA = Path("data/processed")
OUT = Path("results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of images")
    ap.add_argument("--export-for-b", action="store_true",
                    help="save the watermarked training images to results/for_b/ for real AI edits")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    split = json.loads((DATA / "split.json").read_text())
    names = (split["train"] + split["test"])[: args.n]
    configs = {k: CONFIGS[k] for k in ("uniform", "uniform+ecc")}

    rows = []
    for n in tqdm(names, desc="images"):
        rows += evaluate_image(load_image(DATA / f"{n}.png"), n, configs)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p2_baseline.csv", index=False)

    table = df.pivot_table(index=["attack", "severity"], columns="config",
                           values=["bit_acc", "success"], aggfunc="mean").round(3)
    print(table.to_string())
    print(f"\nmean PSNR {df.psnr.mean():.2f} dB, SSIM {df.ssim.mean():.4f}")
    print(f"saved {OUT / 'p2_baseline.csv'}")

    if args.export_for_b:
        dst = OUT / "for_b"
        dst.mkdir(exist_ok=True)
        for n in tqdm(split["train"], desc="export"):
            save_image(dst / f"{n}.png", export_for_attack(load_image(DATA / f"{n}.png")))
        print(f"watermarked images for B's diffusion edits are in {dst}. Ask B to return edited "
              f"versions with the SAME file names in data/external/<edit_name>/")


if __name__ == "__main__":
    main()
