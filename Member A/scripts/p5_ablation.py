"""Phase 5: the four-way ablation on the held-out TEST split.

    python scripts/p5_ablation.py
    python scripts/p5_ablation.py --alpha 1.0 --beta 0.0     # use what Phase 4 chose

Configs: uniform, uniform+ecc, heuristic+ecc, adaptive, adaptive+ecc
(all calibrated to the same PSNR). Outputs in results/:
  p5_raw.csv              one row per image x config x attack
  p5_by_severity.csv      success rate by severity (headline table)
  p5_by_attack.csv        success rate by attack x severity
  p5_success_vs_severity.png, p5_by_attack.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from watermark.core import load_image
from watermark.evaluate import CONFIGS, evaluate_image
from watermark.predictor import load_model

DATA = Path("data/processed")
OUT = Path("results")
ORDER = ["uniform", "uniform+ecc", "heuristic+ecc", "adaptive", "adaptive+ecc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--model", default="models/survival.joblib")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    model = load_model(args.model)
    names = json.loads((DATA / "split.json").read_text())["test"]
    rows = []
    for n in tqdm(names, desc="test images"):
        rows += evaluate_image(load_image(DATA / f"{n}.png"), n, CONFIGS, model,
                               map_kw={"learned": {"alpha": args.alpha, "beta": args.beta}})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p5_raw.csv", index=False)
    att = df[df.attack != "none"]

    quality = df[df.attack == "none"].groupby("config")[["psnr", "ssim", "success"]].mean()
    by_sev = att.pivot_table(index="config", columns="severity", values="success").reindex(ORDER)
    by_sev["all attacks"] = att.groupby("config").success.mean()
    by_sev["bit_acc"] = att.groupby("config").bit_acc.mean()
    by_attack = att.pivot_table(index=["attack", "severity"], columns="config",
                                values="success")[ORDER]
    by_sev.round(4).to_csv(OUT / "p5_by_severity.csv")
    by_attack.round(4).to_csv(OUT / "p5_by_attack.csv")

    print("Image quality and clean extraction:")
    print(quality.reindex(ORDER).round(4).to_string())
    print("\nExtraction success rate by edit severity (headline table):")
    print(by_sev.round(3).to_string())
    print("\nBy attack:")
    print(by_attack.round(3).to_string())

    sev_cols = [c for c in ("light", "moderate", "severe") if c in by_sev]
    ax = by_sev[sev_cols].T.plot(marker="o", figsize=(6, 4))
    ax.axhline(0.8, ls="--", c="gray", lw=1)
    ax.set_ylabel("extraction success rate")
    ax.set_title("Success vs edit severity (equal PSNR)")
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(OUT / "p5_success_vs_severity.png", dpi=130)
    plt.close()

    sub = att[att.config.isin(["uniform+ecc", "adaptive+ecc"])]
    sub.pivot_table(index="attack", columns="config", values="success").plot.bar(figsize=(8, 4))
    plt.ylabel("success rate (all severities)")
    plt.title("Uniform vs adaptive, with ECC")
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(OUT / "p5_by_attack.png", dpi=130)
    plt.close()
    print(f"\nsaved tables and plots in {OUT}/")


if __name__ == "__main__":
    main()
