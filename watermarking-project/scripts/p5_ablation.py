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
import argparse
import json
from pathlib import Path

import matplotlib
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
    if not Path(args.model).exists():
        raise SystemExit(f"{args.model} not found. Run scripts\\p4_train_predictor.py first.")

    params = map_params(alpha=args.alpha, beta=args.beta)
    tex = load_texture_table(TEXTURE_CSV)["texture_category"] if TEXTURE_CSV.exists() else pd.Series(dtype=str)
    names = stratified(dataset.require("test"), args.n, tex.to_dict(), seed=0)
    run = {**params, "target_psnr": TARGET_PSNR, "model": args.model,
           "model_md5": file_md5(args.model), "configs": ORDER}

    # ---- resume ----
    done = set()
    if RAW.exists() and not args.fresh:
        old = json.loads(RUN.read_text()) if RUN.exists() else {}
        if old != run:
            raise SystemExit(f"results/p5_raw.csv was made with different settings:\n  {old}\n"
                             f"Re-run with --fresh to start over with:\n  {run}")
        prev = pd.read_csv(RAW, dtype={"image": str}, on_bad_lines="skip")
        counts = prev.image.value_counts()
        complete = set(counts[counts == counts.max()].index)   # drop an image cut off mid-write
        if len(complete) < len(counts):
            prev[prev.image.isin(complete)].to_csv(RAW, index=False)
        done = complete
    elif RAW.exists():
        RAW.unlink()
    RUN.write_text(json.dumps(run, indent=1))

    todo = [n for n in names if n not in done]
    print(f"{len(names)} test images, map {params}: {len(names) - len(todo)} already done, "
          f"{len(todo)} to run on {args.workers} workers")
    jobs = [("test", n, ORDER, args.model, {"learned": params}) for n in todo]
    failed = []
    for (_, n, *_), rows, err in pmap(eval_job, jobs, args.workers, desc="test images"):
        if err:
            failed.append((n, err))
            if len(failed) <= 2:
                print(f"\n! {n} failed:\n{err}")
            continue
        pd.DataFrame(rows).to_csv(RAW, mode="a", header=not RAW.exists(), index=False)
    if failed:
        print(f"\n{len(failed)} image(s) failed and were NOT saved. Finished images are kept: re-run the "
              f"same command (with --workers 2 if it was a memory error) to complete them.")
    check_failures(failed, len(todo), "test images")

    # ---- tables ----
    df = pd.read_csv(RAW, dtype={"image": str})
    df = df[df.image.isin(names)]
    att = df[df.attack != "none"].copy()
    att["texture"] = att.image.map(tex).fillna("unknown") if len(tex) else "unknown"

    quality = df[df.attack == "none"].groupby("config")[["psnr", "ssim", "success"]].mean().reindex(ORDER)
    by_sev = att.pivot_table(index="config", columns="severity", values="success").reindex(ORDER)
    by_sev["all attacks"] = att.groupby("config").success.mean()
    by_sev["bit_acc"] = att.groupby("config").bit_acc.mean()
    by_attack = att.pivot_table(index=["attack", "severity"], columns="config", values="success")[ORDER]
    by_tex = att.pivot_table(index="config", columns="texture", values="success").reindex(ORDER)
    by_tex = by_tex[[c for c in ("Smooth", "Medium", "Complex", "unknown") if c in by_tex]]

    stats = pd.concat([compare(att, PAIRS).assign(split_by="all"),
                       compare(att, PAIRS, by="severity").assign(split_by="severity"),
                       compare(att, PAIRS, by="texture").assign(split_by="texture")], ignore_index=True)

    by_sev.round(4).to_csv(OUT / "p5_by_severity.csv")
    by_attack.round(4).to_csv(OUT / "p5_by_attack.csv")
    by_tex.round(4).to_csv(OUT / "p5_by_texture.csv")
    stats.round(5).to_csv(OUT / "p5_stats.csv", index=False)
    plots(att)

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
