r"""Phase 3: build the per-block training table for the survival predictor.

    python scripts\p3_build_dataset.py --limit 600
    python scripts\p3_build_dataset.py --limit 600 --external data\external     # with real AI edits

For each sampled TRAIN image: 13 block features + how much the attacks disturbed
each block (log_sigma) + how often it survived erasure edits (keep).
Val and test images are never touched.

Real AI edits (from colab/sd_edits.py) are blended in: when an image has them,
target = (1 - w) * synthetic + w * real, with w = --ext-weight (default 0.5).
Training images that have real edits are always included in the sample.

Output: data/survival/blocks.csv.gz  (+ a printed check of how well the cheap
synthetic attacks agree with the real diffusion edits).
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from watermark import dataset
from watermark.external import edit_names, train_edit_path
from watermark.jobs import survival_job
from watermark.parallel import default_workers, pmap
from watermark.sampling import sample


def names_with_external(root, names):
    if root is None:
        return set()
    edits = edit_names(root)
    return {n for n in names if any(train_edit_path(root, e, n) for e in edits)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=600, help="train images to use (0 = all)")
    ap.add_argument("--external", default=None, help="folder of real AI edits, e.g. data\\external")
    ap.add_argument("--ext-weight", type=float, default=0.5, help="share of the target from real edits")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/survival/blocks.csv.gz")
    args = ap.parse_args()

    all_names = dataset.require("train")
    with_ext = names_with_external(args.external, all_names)
    if args.external and not with_ext:
        print(f"! no edited training images found in {args.external} (check names/layout with "
              f"scripts\\check_data.py --external {args.external})")
    rest = sample(sorted(set(all_names) - with_ext), max(0, args.limit - len(with_ext)), args.seed) \
        if args.limit else sorted(set(all_names) - with_ext)
    names = sorted(with_ext) + rest
    print(f"{len(names)} train images ({len(with_ext)} with real AI edits), {args.workers} workers")

    frames, failed = [], []
    jobs = [(n, args.external, args.ext_weight) for n in names]
    for (n, _, _), df, err in pmap(survival_job, jobs, args.workers, desc="survival"):
        if err:
            failed.append(n)
            print(f"\n! {n} failed:\n{err}")
        else:
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n{len(df)} block rows from {df.image_id.nunique()} images -> {out}"
          + (f"  ({len(failed)} failed: {failed[:5]})" if failed else ""))
    print(df[["std", "grad_mean", "saliency", "log_sigma", "keep"]].describe().round(3).to_string())

    ext = df[df.n_external > 0]
    if len(ext):
        print(f"\nReal vs synthetic attacks on the {ext.image_id.nunique()} images that have both "
              f"(Spearman correlation per block; ~1 = synthetic attacks are a good stand-in):")
        for syn, real in (("log_sigma_syn", "log_sigma_ext"), ("keep_syn", "keep_ext")):
            part = ext[[syn, real]].dropna()
            if len(part):
                print(f"  {syn:14s} vs {real:14s}: {spearmanr(part[syn], part[real])[0]:.3f}  "
                      f"(mean {part[syn].mean():.3f} synthetic, {part[real].mean():.3f} real)")
        print("  Low correlation means the real edits teach the model something new, "
              "which is the point of including them.")


if __name__ == "__main__":
    main()
