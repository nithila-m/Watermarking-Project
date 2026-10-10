r"""Phase 3: build the per-block training table for the survival predictor.

    python scripts/p3_build_dataset.py
    python scripts/p3_build_dataset.py --external data\external     # include B's real AI edits

For every TRAIN image: features of each block + how much the attacks disturbed it
(log_sigma) + how often it survived erasure edits (keep). Test images are never used.

External edits: folders like data/external/sd_inpaint/<name>.png, made by B from
results/for_b/<name>.png. Folder names containing "inpaint", "erase" or "crop"
count as erasure edits; everything else (style transfer, VAE round trip,
re-upload) counts as noise-type edits.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from watermark.core import load_image
from watermark.predictor import build_survival_rows

DATA = Path("data/processed")


def external_versions(ext_root, name):
    noise, erase = [], []
    if ext_root is None:
        return noise, erase
    for folder in sorted(Path(ext_root).iterdir()):
        f = folder / f"{name}.png"
        if folder.is_dir() and f.exists():
            img = load_image(f)
            is_erase = any(w in folder.name.lower() for w in ("inpaint", "erase", "crop"))
            (erase if is_erase else noise).append(img)
    return noise, erase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", default=None, help="folder of B's edited images (optional)")
    ap.add_argument("--out", default="data/survival/blocks.csv.gz")
    args = ap.parse_args()

    names = json.loads((DATA / "split.json").read_text())["train"]
    frames = []
    for n in tqdm(names, desc="images"):
        noise, erase = external_versions(args.external, n)
        frames.append(build_survival_rows(load_image(DATA / f"{n}.png"), n,
                                          external_noise=noise, external_erase=erase))
    df = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} block rows from {len(names)} images -> {out}")
    print(df[["std", "grad_mean", "saliency", "log_sigma", "keep"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
