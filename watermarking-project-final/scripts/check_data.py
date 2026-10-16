r"""Preflight check for the watermarking dataset and optional AI-edited images.

    python scripts\check_data.py
    python scripts\check_data.py --external data\external      # also check Colab edits
    python scripts\check_data.py --no-timing                    # skip the speed test

Run this before the expensive pipeline phases to catch missing, mismatched, or
incorrectly sized inputs and estimate the expected processing time.

Checks
  * data/clean/{train,val,test} exist and every PNG is 512x512 RGB
  * data/clean/test_metadata.csv lists exactly the test images
  * real AI edits (if any): which edits, which layout, names that match the
    train/test images, wrong sizes
Then times one image on THIS machine and prints how long Phases 3-5 will take
with the recommended sample sizes, plus the commands to run.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from PIL import Image

from watermark import dataset
from watermark.config import IMG_SIZE, SPLITS, TEXTURE_CSV
from watermark.external import DEFAULT_EXTERNAL, edit_names, is_erase
from watermark.parallel import default_workers

RECOMMENDED = {"p3_train": 600, "p4_val": 60, "p5_test": 300}


def check_split(split):
    names = dataset.split_names(split)
    bad = []
    for n in names:
        # Read image metadata only; decoding pixel data is unnecessary for size/mode checks.
        with Image.open(dataset.image_path(split, n)) as im:      # header only, fast
            if im.size != (IMG_SIZE, IMG_SIZE) or im.mode not in ("RGB", "L"):
                bad.append(f"{n}: {im.size} {im.mode}")
    return names, bad


def check_external(root, train_names, test_names):
    edits = edit_names(root)
    if not edits:
        print(f"\n[external] nothing in {root} yet (fine: Phases 3-5 run on synthetic attacks; "
              f"add real edits later)")
        return
    print(f"\n[external] {len(edits)} edit folder(s) in {root}")
    rows = []
    for e in edits:
        d = Path(root) / e
        # Accept both flat exports and the newer train/ and test/<config>/ layouts.
        flat = {p.stem for p in d.glob("*.png")}
        tr = {p.stem for p in (d / "train").glob("*.png")}
        te = {(p.parent.name, p.stem) for p in (d / "test").glob("*/*.png")}
        sizes = {Image.open(p).size for p in list(d.rglob("*.png"))[:20]}
        rows.append({"edit": e, "type": "erasure" if is_erase(e) else "noise",
                     "train (new layout)": len(tr & set(train_names)),
                     "train (flat layout)": len(flat & set(train_names)),
                     "test images": len({n for _, n in te} & set(test_names)),
                     "test configs": ",".join(sorted({c for c, _ in te})) or "-",
                     "unmatched names": len((tr | flat) - set(train_names)),
                     "sizes": ",".join(f"{w}x{h}" for w, h in sorted(sizes))})
    print(pd.DataFrame(rows).to_string(index=False))
    if any(r["unmatched names"] for r in rows):
        print("  ! some edited files don't match any training image name. They must keep the "
              "exact names written by export_for_colab.py / p2 --export-for-b.")
    if any(r["sizes"] != f"{IMG_SIZE}x{IMG_SIZE}" for r in rows):
        print(f"  ! some edits are not {IMG_SIZE}x{IMG_SIZE}. The blind extractor needs the "
              f"original grid; re-run the Colab script with --size {IMG_SIZE}.")


def time_one_image():
    """Seconds for one Phase 3 image and one Phase 5 image (5 configs) on this machine."""
    from watermark.evaluate import CONFIGS, evaluate_image
    from watermark.external import survival_rows
    name = dataset.split_names("train")[0]
    img = dataset.load("train", name)
    t = time.perf_counter()
    survival_rows(img, name)
    t3 = time.perf_counter() - t
    t = time.perf_counter()
    evaluate_image(img, name, {"uniform+ecc": CONFIGS["uniform+ecc"]})
    t5 = (time.perf_counter() - t) * len(CONFIGS)
    return t3, t5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", default=str(DEFAULT_EXTERNAL))
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()

    ok = True
    names = {}
    for s in SPLITS:
        names[s], bad = check_split(s)
        status = "OK" if names[s] and not bad else "PROBLEM"
        ok &= status == "OK"
        print(f"[{s:5s}] {len(names[s]):5d} images  {status}" + (f"  e.g. {bad[:3]}" if bad else ""))
    if not names["train"]:
        raise SystemExit("no training images. Put B's data in data/clean/ "
                         "(or run scripts/prepare_dataset.py --src raw/val2017).")

    if TEXTURE_CSV.exists():
        tex = pd.read_csv(TEXTURE_CSV)
        # Ensure texture annotations correspond one-to-one with the test image names.
        listed = set(tex.filename.str.replace(r"\.png$", "", regex=True))
        missing, extra = set(names["test"]) - listed, listed - set(names["test"])
        print(f"[texture] {len(listed)} rows; categories "
              f"{tex.texture_category.value_counts().to_dict()}")
        if missing or extra:
            ok = False
            print(f"  ! {len(missing)} test images not in the table, {len(extra)} rows with no image. "
                  f"Run: python scripts\\analyze_textures.py")
    else:
        ok = False
        print(f"[texture] {TEXTURE_CSV} missing. Run: python scripts\\analyze_textures.py")

    check_external(args.external, names["train"], names["test"])

    w = default_workers()
    if not args.no_timing:
        t3, t5 = time_one_image()
        n3 = min(RECOMMENDED["p3_train"], len(names["train"]))
        n4 = min(RECOMMENDED["p4_val"], len(names["val"]))
        n5 = min(RECOMMENDED["p5_test"], len(names["test"]))
        # Discount ideal parallel scaling to avoid overly optimistic runtime estimates.
        speed = max(1.0, w * 0.75)          # parallel speed-up is never perfect
        print(f"\n[timing] one image: Phase 3 {t3:.1f} s, Phase 5 {t5:.1f} s (5 configs). "
              f"{w} worker processes.")
        print(f"  Phase 3, {n3} train images : ~{n3 * t3 / speed / 60:.0f} min")
        print(f"  Phase 4, {n4} val images   : ~{n4 * t5 * 1.2 / speed / 60:.0f} min")
        print(f"  Phase 5, {n5} test images  : ~{n5 * t5 / speed / 60:.0f} min")

    print("\n" + ("DATA OK" if ok else "FIX THE ITEMS MARKED ! ABOVE") + ". Next:")
    print(f"  python scripts\\p3_build_dataset.py --limit {RECOMMENDED['p3_train']}")


if __name__ == "__main__":
    main()
