r"""Phase 4: train the survival predictor, then make the Day-8 decision on the VAL split.

    python scripts\p4_train_predictor.py
    python scripts\p4_train_predictor.py --val-n 60 --workers 6
    python scripts\p4_train_predictor.py --skip-decision          # train only
    python scripts\p4_train_predictor.py --decision-only          # reuse the saved model, decide only

Steps
  1. Train both random forests on data/survival/blocks.csv.gz
     (internal check: 20% of the training IMAGES held out).
  2. Save models/survival.joblib, results/p4_report.json and two plots.
  3. Decision: uniform+ecc vs heuristic+ecc vs learned variants (alpha, beta) on
     --val-n images from data/clean/val (never train, never test), with a paired
     confidence interval against uniform+ecc.
  4. Write results/p4_choice.json. p5_ablation.py and export_for_colab.py read it,
     so the chosen alpha/beta is used everywhere without retyping it.
"""
import argparse
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from watermark import dataset
from watermark.features import block_features
from watermark.jobs import eval_variants_job
from watermark.parallel import check_failures, default_workers, pmap
from watermark.predictor import load_model, save_model, train
from watermark.sampling import sample
from watermark.stats import compare
from watermark.strength_map import heuristic_map, learned_map

# Shared output locations let training and decision-only runs use the same artifacts.
OUT = Path("results")
MODEL = Path("models/survival.joblib")
CHOICE = OUT / "p4_choice.json"
VARIANTS = {"learned_a0.5_b0": {"alpha": 0.5, "beta": 0.0},
            "learned_a1_b0":   {"alpha": 1.0, "beta": 0.0},
            "learned_a1.5_b0": {"alpha": 1.5, "beta": 0.0},
            "learned_a1_b0.5": {"alpha": 1.0, "beta": 0.5}}


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
    ap.add_argument("--val-n", type=int, default=60, help="val images for the decision (0 = all)")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--trees", type=int, default=100, help="trees per forest (memory grows with this)")
    ap.add_argument("--min-leaf", type=int, default=30, help="min samples per leaf (bigger = smaller model)")
    ap.add_argument("--skip-decision", action="store_true", help="train only")
    ap.add_argument("--decision-only", action="store_true", help="skip training, use the saved model")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # Use this mode to rerun validation with an existing model, without retraining.
    if args.decision_only:
        if not MODEL.exists():
            raise SystemExit(f"{MODEL} not found: run without --decision-only first.")
        decide(args)
        return

    # ---- 1-2. train ----
    df = pd.read_csv(args.data, dtype={"image_id": str})   # keep leading zeros: "000012", not 12
    n_ext = int((df.groupby("image_id").n_external.first() > 0).sum()) if "n_external" in df else 0
    print(f"training on {len(df)} blocks from {df.image_id.nunique()} images ({n_ext} with real AI edits), "
          f"{args.trees} trees per forest")
    model, report = train(df, n_estimators=args.trees, min_samples_leaf=args.min_leaf)
    del df
    gc.collect()                          # free the training table before starting worker processes
    save_model(model, MODEL)
    report.update({"n_ext_images": n_ext, "trees": args.trees, "min_leaf": args.min_leaf})
    for t in ("log_sigma", "keep"):
        r = report[t]
        print(f"  {t:9s} R2 {r['val_r2']:.3f}  Spearman {r['val_spearman']:.3f} "
              f"(texture alone {r['val_spearman_texture_only']:.3f})  top: {list(r['importances'])[:4]}")

    imp = pd.Series(report["log_sigma"]["importances"]).sort_values()
    imp.plot.barh(figsize=(6, 4), title="Feature importance (disturbance model)")
    plt.tight_layout()
    plt.savefig(OUT / "p4_feature_importance.png", dpi=120)
    plt.close()
    plot_maps(dataset.load("train", report["val_images"][0]), model, OUT / "p4_strength_maps.png")
    (OUT / "p4_report.json").write_text(json.dumps(report, indent=1))
    print(f"saved {MODEL} ({MODEL.stat().st_size / 1e6:.0f} MB), {OUT / 'p4_report.json'}, plots in {OUT}")
    del model
    gc.collect()
    if args.skip_decision:
        return
    decide(args)


def decide(args):
    """Steps 3-4: compare maps on the val split and write results/p4_choice.json."""
    load_model(MODEL)                     # fail here, clearly, if the file is broken
    names = sample(dataset.require("val"), args.val_n, seed=1)
    # All candidate maps use the same images so their results can be compared in pairs.
    jobs = [("val", n, ["uniform+ecc", "heuristic+ecc"], VARIANTS, str(MODEL)) for n in names]
    rows, failed = [], []
    for (_, n, *_), r, err in pmap(eval_variants_job, jobs, args.workers, desc="decision (val)"):
        if err:
            failed.append((n, err))
            if len(failed) <= 2:                       # show the first errors, not 50 copies
                print(f"\n! {n} failed:\n{err}")
        else:
            rows += r
    check_failures(failed, len(names), "val images")
    if failed:
        print(f"note: {len(failed)} val image(s) failed and are left out: {[n for n, _ in failed]}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "p4_decision.csv", index=False)
    att = res[res.attack != "none"]

    summary = (att.groupby("config").agg(success=("success", "mean"), bit_acc=("bit_acc", "mean"),
                                         psnr=("psnr", "mean"))
                  .sort_values(["success", "bit_acc"], ascending=False))
    by_sev = att.pivot_table(index="config", columns="severity", values="success")
    tests = compare(att, [(c, "uniform+ecc") for c in ["heuristic+ecc"] + list(VARIANTS)])
    n_ok = res.image.nunique()
    print(f"\nValidation ({n_ok} val images, all at the same PSNR):")
    print(summary.join(by_sev).round(4).to_string())
    print("\nPaired difference vs uniform+ecc (success rate, 95% CI over images):")
    print(tests[["A", "diff", "lo", "hi", "p_value", "significant"]].round(4).to_string(index=False))
    tests.round(5).to_csv(OUT / "p4_decision_stats.csv", index=False)

    # ---- 4. choose ----
    best_v = summary.loc[[v for v in VARIANTS if v in summary.index]].index[0]
    t = tests.set_index("A")
    best, heur = t.loc[best_v], t.loc["heuristic+ecc"]
    if best["significant"] and best["diff"] > 0:
        verdict = f"learned map {best_v} beats uniform (significant). Use it in Phase 5."
    elif heur["significant"] and heur["diff"] > max(0, best["diff"]):
        verdict = ("heuristic beats uniform but learned does not: ship the heuristic as the main "
                   "method and report the learned map as an ablation.")
    elif best["diff"] > 0:
        verdict = (f"{best_v} is ahead of uniform but not significantly on {n_ok} images. "
                   f"Use it in Phase 5; the larger test set decides (or re-run with --val-n 120).")
    else:
        verdict = ("no adaptive variant beats uniform on val. Before Phase 5: check that real AI "
                   "edits were included in Phase 3, and look at which attacks it loses on "
                   "(results/p4_decision.csv).")
    choice = {"variant": best_v, **VARIANTS[best_v], "verdict": verdict,
              "val_images": n_ok, "val_success": summary.success.round(4).to_dict()}
    # Downstream Phase 5 scripts read this file as the single source for alpha and beta.
    CHOICE.write_text(json.dumps(choice, indent=1))
    print(f"\nDECISION: {verdict}\nalpha={choice['alpha']} beta={choice['beta']} saved to {CHOICE}")


if __name__ == "__main__":
    main()
