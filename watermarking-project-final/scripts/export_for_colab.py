r"""Export watermarked images for real Stable Diffusion edits on Colab.

Two rounds (each takes a few minutes here, then 30-60 min on a Colab T4):

  Round 1, BEFORE Phase 3 (training data for the predictor):
      python scripts\export_for_colab.py --train 200 --test 0

  Round 2, AFTER Phase 4 (evaluation; needs models/survival.joblib + results/p4_choice.json):
      python scripts\export_for_colab.py --train 0 --test 100

Output: results/colab_export/ (+ results/colab_export.zip with --zip)
    manifest.csv                 bits, key and PSNR of every exported image (keep this!)
    train/<name>.png             uniform watermark exactly as Phase 3 uses it
    test/<config>/<name>.png     one copy per ablation config, same payload as p5
Upload the folder (or zip) to Google Drive and run colab/sd_edits.py on it.
Re-running a round adds to the manifest instead of overwriting the other round.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

from watermark import dataset
from watermark.choice import file_md5, map_params
from watermark.config import TEXTURE_CSV
from watermark.external import DEFAULT_EXPORT
from watermark.jobs import export_test_job, export_train_job
from watermark.parallel import check_failures, default_workers, pmap
from watermark.sampling import sample, stratified
from watermark.textures import load_texture_table

# Default experimental configurations evaluated on the test set
DEFAULT_TEST_CONFIGS = ["uniform+ecc", "heuristic+ecc", "adaptive+ecc"]


def main():
    """Main CLI entry point for exporting watermarked images and logging manifests."""
    # Set up command-line argument parser
    ap = argparse.ArgumentParser(description="Export watermarked images for Colab GPU Stable Diffusion attacks.")
    ap.add_argument("--train", type=int, default=200, help="train images to export (0 = skip)")
    ap.add_argument("--test", type=int, default=0, help="test images to export (0 = skip)")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_TEST_CONFIGS)
    ap.add_argument("--model", default="models/survival.joblib")
    ap.add_argument("--key", type=int, default=2024)
    ap.add_argument("--out", default=str(DEFAULT_EXPORT))
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip for uploading")
    args = ap.parse_args()

    # Create destination output directory
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    # ----------------- Round 1: Train split export (uniform watermarking) -----------------
    if args.train:
        # Sample training images deterministically
        names = sample(dataset.require("train"), args.train, seed=7)
        jobs = [(n, str(out)) for n in names]
        failed = []

        # Execute parallelized embedding jobs across available worker processes
        for (n, _), r, err in pmap(export_train_job, jobs, args.workers, desc="export train"):
            if err:
                failed.append((n, err))
                print(f"! {n}: {err}")
            else:
                rows += r
        # Assert failure tolerances for training batch exports
        check_failures(failed, len(jobs), "train exports")

    # ----------------- Round 2: Test split export (multi-config ablation) -----------------
    if args.test:
        # Check model dependency if evaluating learned adaptive schemes
        needs_model = any("adaptive" in c for c in args.configs)
        if needs_model and not Path(args.model).exists():
            raise SystemExit("adaptive configs need the trained model: run p4_train_predictor.py first "
                             "(or drop adaptive+ecc from --configs).")

        # Load optimized strength-mapping hyperparameters (alpha, beta, clipping)
        params = map_params()                    # results/p4_choice.json (Phase 4, maybe Phase 6)
        map_kw = {"learned": params}
        # Compute MD5 checksum of the model file for reproducible provenance
        md5 = file_md5(args.model) if needs_model else None

        # Stratify test images evenly across Smooth, Medium, and Complex texture categories
        tex = load_texture_table(TEXTURE_CSV)["texture_category"].to_dict() if TEXTURE_CSV.exists() else {}
        names = stratified(dataset.require("test"), args.test, tex, seed=0)

        # Build payload export tuples for test images across all specified configurations
        jobs = [(n, args.configs, args.model if needs_model else None, map_kw, str(out), args.key)
                for n in names]
        failed = []

        # Execute parallel export across test configurations
        for (n, *_), r, err in pmap(export_test_job, jobs, args.workers, desc="export test"):
            if err:
                failed.append((n, err))
                print(f"! {n}: {err}")
            else:
                # Merge per-job result with strength parameters and model fingerprint
                rows += [{**x, **params, "model_md5": md5} for x in r]
        check_failures(failed, len(jobs), "test exports")

    # Exit if neither train nor test partition was selected for export
    if not rows:
        raise SystemExit("nothing exported (use --train N and/or --test N)")

    # ----------------- Manifest generation and merge logic -----------------
    new = pd.DataFrame(rows)
    man = out / "manifest.csv"

    # Merge with existing manifest records so separate rounds do not overwrite each other
    if man.exists():   # keep the other round's rows, replace rows we just re-exported
        old = pd.read_csv(man, dtype={"image": str, "payload": str, "message": str})
        # Identify uniquely updated image keys (split, config, image_id)
        keys = set(zip(new.split, new.config, new.image))
        old = old[[k not in keys for k in zip(old.split, old.config, old.image)]]
        new = pd.concat([old, new], ignore_index=True)

    # Persist updated manifest ordered by dataset split, config, and image ID
    new.sort_values(["split", "config", "image"]).to_csv(man, index=False)

    # Display console summary of visual imperceptibility across exported splits
    summary = new.groupby(["split", "config"]).agg(images=("image", "count"), psnr=("psnr", "mean"),
                                                    ssim=("ssim", "mean"))
    print(summary.round(3).to_string())
    print(f"\nmanifest: {man}")

    # Optionally package output folder into an archive for Google Drive upload
    if args.zip:
        z = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"zip for upload: {z}")

    print("Next: upload to Google Drive and run colab/sd_edits.py (see the steps in the guide).")


if __name__ == "__main__":
    main()
