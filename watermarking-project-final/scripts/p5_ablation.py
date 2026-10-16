r"""Phase 5: the ablation on the held-out TEST split, with significance tests.

    python scripts\p5_ablation.py                     # 300 test images, texture-balanced
    python scripts\p5_ablation.py --n 0               # every test image (slow)
    python scripts\p5_ablation.py --fresh             # ignore earlier partial results

Map settings (alpha, beta, shrink, s_min) come from results/p4_choice.json, written by
Phase 4 and possibly updated by Phase 6 (override alpha/beta with --alpha/--beta).
Configs: uniform, uniform+ecc, heuristic+ecc, adaptive, adaptive+ecc, all at the same PSNR.

Resumable: rows are appended to results/p5_raw.csv as each image finishes, so if
the laptop sleeps, re-run the same command and it continues where it stopped.

Outputs in results/:
  p5_raw.csv                one row per image x config x attack
  p5_by_severity.csv        success by severity (headline table)
  p5_by_attack.csv          success by attack x severity
  p5_by_texture.csv         success by Smooth / Medium / Complex image
  p5_stats.csv              paired 95% CIs and McNemar p-values (overall, per severity, per texture)
  p5_success_vs_severity.png, p5_by_attack.png
"""
# This script runs Phase 5 of the experiment pipeline: it evaluates different
# watermarking configurations on a held-out test split, aggregates results,
# runs paired statistical comparisons, and produces summary tables and plots.
#
# Key uses:
# - Run locally (or on a server) to reproduce the ablation results described in
#   the project write-up. Default uses 300 texture-balanced test images.
# - It's resumable: partial results are appended to results/p5_raw.csv and the
#   script will continue where it left off unless --fresh is supplied.
# - The script depends on outputs from earlier phases (e.g. the learned model
#   saved by p4_train_predictor.py and chosen map params in p4_choice.json).

import argparse
import json
from pathlib import Path

import matplotlib
# Use the "Agg" backend so the script can run headless (no display), e.g. on a
# CI server or remote machine and still write PNG files.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from watermark import dataset
from watermark.choice import file_md5, map_params
from watermark.config import TARGET_PSNR, TEXTURE_CSV
from watermark.jobs import eval_job
from watermark.parallel import check_failures, default_workers, pmap
from watermark.sampling import stratified
from watermark.stats import compare
from watermark.textures import load_texture_table

OUT = Path("results")
RAW = OUT / "p5_raw.csv"
RUN = OUT / "p5_run.json"
ORDER = ["uniform", "uniform+ecc", "heuristic+ecc", "adaptive", "adaptive+ecc"]
PAIRS = [("adaptive+ecc", "uniform+ecc"), ("heuristic+ecc", "uniform+ecc"),
         ("adaptive", "uniform"), ("uniform+ecc", "uniform")]


def plots(att):
    by_sev = att.pivot_table(index="config", columns="severity", values="success").reindex(ORDER)
    sev_cols = [c for c in ("light", "moderate", "severe") if c in by_sev]
    ax = by_sev[sev_cols].T.plot(marker="o", figsize=(6, 4))
    ax.axhline(0.8, ls="--", c="gray", lw=1)
    ax.set_ylabel("extraction success rate")
    ax.set_title("Success vs edit severity (equal PSNR)")
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(OUT / "p5_success_vs_severity.png", dpi=130)
    plt.close()

    sub = att[att.config.isin(["uniform+ecc", "heuristic+ecc", "adaptive+ecc"])]
    sub.pivot_table(index="attack", columns="config", values="success").plot.bar(figsize=(9, 4))
    plt.ylabel("success rate (all severities)")
    plt.title("Uniform vs heuristic vs adaptive, with ECC")
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(OUT / "p5_by_attack.png", dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="test images (0 = all), balanced by texture")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--model", default="models/survival.joblib")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--fresh", action="store_true", help="discard results/p5_raw.csv and start over")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    # verify the learned model file exists before starting long runs
    if not Path(args.model).exists():
        raise SystemExit(f"{args.model} not found. Run scripts\\p4_train_predictor.py first.")

    # map_params converts optional alpha/beta overrides into a full param dict
    # used by the watermarking maps. These come from Phase 4 by default.
    params = map_params(alpha=args.alpha, beta=args.beta)

    # load texture categories (if available) and build a stratified sample of
    # test image names. Passing tex.to_dict() lets stratified balance images by
    # texture category when selecting args.n images.
    tex = load_texture_table(TEXTURE_CSV)["texture_category"] if TEXTURE_CSV.exists() else pd.Series(dtype=str)
    names = stratified(dataset.require("test"), args.n, tex.to_dict(), seed=0)

    # record the run configuration (params, PSNR target, model path and MD5,
    # and the list/order of configs). This is written to results/p5_run.json and
    # used to ensure resuming uses the same settings.
    run = {**params, "target_psnr": TARGET_PSNR, "model": args.model,
           "model_md5": file_md5(args.model), "configs": ORDER}

    # ---- resume ----
    done = set()
    # If RAW exists and we're not forcing fresh, check that the run settings
    # match the existing run metadata. If different, abort to avoid mixing
    # incompatible results; --fresh forces deletion instead.
    if RAW.exists() and not args.fresh:
        old = json.loads(RUN.read_text()) if RUN.exists() else {}
        if old != run:
            raise SystemExit(f"results/p5_raw.csv was made with different settings:\n  {old}\n"
                             f"Re-run with --fresh to start over with:\n  {run}")
        prev = pd.read_csv(RAW, dtype={"image": str}, on_bad_lines="skip")
        counts = prev.image.value_counts()
        # find images that have a complete set of rows (all configs/attacks) and
        # drop any image that might have been cut off mid-write
        complete = set(counts[counts == counts.max()].index)   # drop an image cut off mid-write
        if len(complete) < len(counts):
            # overwrite RAW with only the complete images to ensure consistency
            prev[prev.image.isin(complete)].to_csv(RAW, index=False)
        done = complete
    elif RAW.exists():
        # if user requested fresh, remove the old raw file
        RAW.unlink()
    RUN.write_text(json.dumps(run, indent=1))

    todo = [n for n in names if n not in done]
    print(f"{len(names)} test images, map {params}: {len(names) - len(todo)} already done, "
          f"{len(todo)} to run on {args.workers} workers")

    # prepare jobs for parallel evaluation: each job is (split, image_name, configs, model, kwargs)
    jobs = [("test", n, ORDER, args.model, {"learned": params}) for n in todo]
    failed = []
    # pmap runs eval_job in parallel and yields (job, rows, err). rows is a list
    # of dicts (one per config/attack) which we append to the CSV as each image
    # completes — this is what makes the run resumable.
    for (_, n, *_), rows, err in pmap(eval_job, jobs, args.workers, desc="test images"):
        if err:
            failed.append((n, err))
            if len(failed) <= 2:
                print(f"\n! {n} failed:\n{err}")
            continue
        # append the rows for this image to the RAW CSV; header is written only
        # if the file does not already exist.
        pd.DataFrame(rows).to_csv(RAW, mode="a", header=not RAW.exists(), index=False)
    if failed:
        print(f"\n{len(failed)} image(s) failed and were NOT saved. Finished images are kept: re-run the "
              f"same command (with --workers 2 if it was a memory error) to complete them.")
    check_failures(failed, len(todo), "test images")

    # ---- tables ----
    # read back the accumulated raw results and filter to the images we ran
    df = pd.read_csv(RAW, dtype={"image": str})
    df = df[df.image.isin(names)]
    # 'att' contains only attacked images (exclude the "none"/clean baseline rows)
    att = df[df.attack != "none"].copy()
    # map each image to its texture category, filling missing values with 'unknown'
    att["texture"] = att.image.map(tex).fillna("unknown") if len(tex) else "unknown"

    # compute average quality metrics on clean images and success rates under attack
    quality = df[df.attack == "none"].groupby("config")[ ["psnr", "ssim", "success"] ].mean().reindex(ORDER)
    by_sev = att.pivot_table(index="config", columns="severity", values="success").reindex(ORDER)
    by_sev["all attacks"] = att.groupby("config").success.mean()
    by_sev["bit_acc"] = att.groupby("config").bit_acc.mean()
    by_attack = att.pivot_table(index=["attack", "severity"], columns="config", values="success")[ORDER]
    by_tex = att.pivot_table(index="config", columns="texture", values="success").reindex(ORDER)
    by_tex = by_tex[[c for c in ("Smooth", "Medium", "Complex", "unknown") if c in by_tex]]

    # run paired statistical comparisons (e.g. McNemar) between configuration pairs
    # compare() returns diffs and confidence intervals per-image; we run it overall,
    # split by severity, and split by texture and then concatenate the results.
    stats = pd.concat([compare(att, PAIRS).assign(split_by="all"),
                       compare(att, PAIRS, by="severity").assign(split_by="severity"),
                       compare(att, PAIRS, by="texture").assign(split_by="texture")], ignore_index=True)

    # write summary tables and plots
    by_sev.round(4).to_csv(OUT / "p5_by_severity.csv")
    by_attack.round(4).to_csv(OUT / "p5_by_attack.csv")
    by_tex.round(4).to_csv(OUT / "p5_by_texture.csv")
    stats.round(5).to_csv(OUT / "p5_stats.csv", index=False)
    plots(att)

    # print human-readable summaries to stdout
    print(f"\n{df.image.nunique()} test images. Image quality and clean extraction:")
    print(quality.round(4).to_string())
    print("\nExtraction success rate by edit severity (headline table):")
    print(by_sev.round(3).to_string())
    print("\nBy image texture:")
    print(by_tex.round(3).to_string())
    print("\nBy attack:")
    print(by_attack.round(3).to_string())
    print("\nPaired comparisons (diff = success A - success B, 95% CI over images):")
    show = stats[stats.split_by != "texture"][["group", "A", "B", "diff", "lo", "hi", "p_value", "significant"]]
    print(show.round(4).to_string(index=False))
    print(f"\nsaved tables and plots in {OUT}/")


if __name__ == "__main__":
    main()
