"""Phase 4: train the survival predictor, check it, and make the Day-8 decision.

    python scripts/p4_train_predictor.py
    python scripts/p4_train_predictor.py --skip-decision     # train only

Steps
  1. Train on data/survival/blocks.csv.gz (validation split by image).
  2. Save models/survival.joblib and results/p4_report.json.
  3. Plot feature importances and a strength map for one image.
  4. Day-8 decision: compare uniform / heuristic / learned maps on the VALIDATION
     images (never the test split), plus a small alpha/beta sweep. Prints which
     map to use in Phase 5.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from watermark import dataset
from watermark.evaluate import evaluate_image
from watermark.features import block_features
from watermark.predictor import save_model, train
from watermark.strength_map import heuristic_map, learned_map

OUT = Path("results")
MODEL = Path("models/survival.joblib")


def plot_maps(img, model, path):
    f = block_features(img)
    maps = {"heuristic": heuristic_map(img, f), "learned (alpha=1)": learned_map(img, model, f)}
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img[..., ::-1] if img.ndim == 3 else img, cmap="gray")
    ax[0].set_title("image")
    for a, (k, m) in zip(ax[1:], maps.items()):
        im = a.imshow(m.reshape(32, 32), cmap="magma")
        a.set_title(f"strength map: {k}")
        fig.colorbar(im, ax=a, fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/survival/blocks.csv.gz")
    ap.add_argument("--skip-decision", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    df = pd.read_csv(args.data)
    model, report = train(df)
    save_model(model, MODEL)
    (OUT / "p4_report.json").write_text(json.dumps(report, indent=1))
    for t in ("log_sigma", "keep"):
        r = report[t]
        print(f"{t:9s}  R2 {r['val_r2']:.3f}  Spearman {r['val_spearman']:.3f} "
              f"(texture-only {r['val_spearman_texture_only']:.3f})  "
              f"top features: {list(r['importances'])[:4]}")

    imp = pd.Series(report["log_sigma"]["importances"]).sort_values()
    imp.plot.barh(figsize=(6, 4), title="Feature importance (disturbance model)")
    plt.tight_layout()
    plt.savefig(OUT / "p4_feature_importance.png", dpi=120)
    plt.close()
    first_val = report["val_images"][0]
    plot_maps(dataset.load("train", first_val), model, OUT / "p4_strength_maps.png")
    print(f"saved {MODEL}, {OUT / 'p4_report.json'}, plots in {OUT}")

    if args.skip_decision:
        return

    # ---- Day-8 decision on validation images ----
    configs = {"uniform": {"map": "uniform", "ecc": True},
               "heuristic": {"map": "heuristic", "ecc": True}}
    variants = {"learned_a1_b0": {"alpha": 1.0, "beta": 0.0},
                "learned_a0.5_b0": {"alpha": 0.5, "beta": 0.0},
                "learned_a1_b1": {"alpha": 1.0, "beta": 1.0}}
    rows = []
    for n in tqdm(report["val_images"], desc="decision"):
        img = dataset.load("train", n)
        rows += evaluate_image(img, n, configs, model)
        for vname, kw in variants.items():
            for r in evaluate_image(img, n, {vname: {"map": "learned", "ecc": True}}, model,
                                    map_kw={"learned": kw}):
                rows.append(r)
    res = pd.DataFrame(rows)
    res = res[res.attack != "none"]
    summary = res.groupby("config").agg(success=("success", "mean"), bit_acc=("bit_acc", "mean"),
                                        psnr=("psnr", "mean")).sort_values("success", ascending=False)
    by_sev = res.pivot_table(index="config", columns="severity", values="success").round(3)
    print("\nValidation results (all at the same PSNR):")
    print(summary.round(4).to_string())
    print(by_sev.to_string())
    res.to_csv(OUT / "p4_decision.csv", index=False)

    best = summary.index[0]
    margin = summary.success.iloc[0] - summary.success.get("uniform", 0)
    if margin < 0.01:
        print("\nDECISION: no clear winner on the validation images (difference < 1 point). "
              "Keep the default learned map (alpha=1, beta=0) and let the Phase 5 test set decide; "
              "if validation is this easy, add harder attacks or raise TARGET_PSNR.")
    elif best.startswith("learned"):
        print(f"\nDECISION: use the learned map ({best}) in Phase 5.")
    elif best == "heuristic":
        print("\nDECISION: learned map does not beat the heuristic -> ship heuristic, report learned as ablation.")
    else:
        print("\nDECISION: neither adaptive map beats uniform yet -> revisit features/targets before Phase 5.")


if __name__ == "__main__":
    main()
