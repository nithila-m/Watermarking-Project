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

DEFAULT_TEST_CONFIGS = ["uniform+ecc", "heuristic+ecc", "adaptive+ecc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=200, help="train images to export (0 = skip)")
    ap.add_argument("--test", type=int, default=0, help="test images to export (0 = skip)")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_TEST_CONFIGS)
    ap.add_argument("--model", default="models/survival.joblib")
    ap.add_argument("--key", type=int, default=2024)
    ap.add_argument("--out", default=str(DEFAULT_EXPORT))
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip for uploading")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    if args.train:
        names = sample(dataset.require("train"), args.train, seed=7)
        jobs = [(n, str(out)) for n in names]
        failed = []
        for (n, _), r, err in pmap(export_train_job, jobs, args.workers, desc="export train"):
            if err:
                failed.append((n, err))
                print(f"! {n}: {err}")
            else:
                rows += r
        check_failures(failed, len(jobs), "train exports")

    if args.test:
        needs_model = any("adaptive" in c for c in args.configs)
        if needs_model and not Path(args.model).exists():
            raise SystemExit("adaptive configs need the trained model: run p4_train_predictor.py first "
                             "(or drop adaptive+ecc from --configs).")
        params = map_params()                    # results/p4_choice.json (Phase 4, maybe Phase 6)
        map_kw = {"learned": params}
        md5 = file_md5(args.model) if needs_model else None
        tex = load_texture_table(TEXTURE_CSV)["texture_category"].to_dict() if TEXTURE_CSV.exists() else {}
        names = stratified(dataset.require("test"), args.test, tex, seed=0)
        jobs = [(n, args.configs, args.model if needs_model else None, map_kw, str(out), args.key)
                for n in names]
        failed = []
        for (n, *_), r, err in pmap(export_test_job, jobs, args.workers, desc="export test"):
            if err:
                failed.append((n, err))
                print(f"! {n}: {err}")
            else:
                rows += [{**x, **params, "model_md5": md5} for x in r]
        check_failures(failed, len(jobs), "test exports")

    if not rows:
        raise SystemExit("nothing exported (use --train N and/or --test N)")
    new = pd.DataFrame(rows)
    man = out / "manifest.csv"
    if man.exists():   # keep the other round's rows, replace rows we just re-exported
        old = pd.read_csv(man, dtype={"image": str, "payload": str, "message": str})
        keys = set(zip(new.split, new.config, new.image))
        old = old[[k not in keys for k in zip(old.split, old.config, old.image)]]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(["split", "config", "image"]).to_csv(man, index=False)

    summary = new.groupby(["split", "config"]).agg(images=("image", "count"), psnr=("psnr", "mean"),
                                                    ssim=("ssim", "mean"))
    print(summary.round(3).to_string())
    print(f"\nmanifest: {man}")
    if args.zip:
        z = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"zip for upload: {z}")
    print("Next: upload to Google Drive and run colab/sd_edits.py (see the steps in the guide).")


if __name__ == "__main__":
    main()
