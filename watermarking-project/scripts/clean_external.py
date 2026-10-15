r"""Tidy hand-made or Colab edits before Phase 3: keep only real, usable edits.

    python scripts\clean_external.py              # report only, nothing is moved
    python scripts\clean_external.py --apply      # move the bad files aside

For every data/external/<edit>/train/<name>.png (and the flat layout <edit>/<name>.png):
  wrong size   not 512x512 (cropped or resized by the editing tool): the watermark grid
               no longer lines up, so the file is unusable
  unedited     practically identical to the exported watermarked image in
               results/colab_export/train/: it was copied but never edited, and would
               teach the model that "AI erasure" leaves everything intact
  ok           an edit we can use; the report shows how much of the image it changed
Moved files go to data/external_rejected/<reason>/<edit>/<name>.png (nothing is deleted).
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from watermark.config import IMG_SIZE
from watermark.external import DEFAULT_EXPORT, DEFAULT_EXTERNAL, edit_names

# Intensity difference threshold: a pixel counts as edited if any channel shifts by > 12
CHANGED = 12
# Minimum area threshold: edits covering < 0.5% of total pixels are deemed unedited/noise
MIN_EDITED = 0.005


def edited_files(root):
    """Discovers external edited images under hierarchical or flat folder layouts.

    Args:
        root (str or Path): Root folder containing external edit categories.

    Yields:
        tuple[str, Path]: The edit condition name and corresponding image filepath.
    """
    for edit in edit_names(root):
        d = Path(root) / edit
        # Support both partitioned layout (<edit>/train/*.png) and flat layout (<edit>/*.png)
        for p in sorted((d / "train").glob("*.png")) + sorted(d.glob("*.png")):
            yield edit, p


def main():
    """CLI entry point for validating, filtering, and organizing external edited images."""
    ap = argparse.ArgumentParser(description="Audit and quarantine malformed external image edits.")
    ap.add_argument("--external", default=str(DEFAULT_EXTERNAL), help="path to external edit directory")
    ap.add_argument("--export", default=str(DEFAULT_EXPORT), help="path to unedited export references")
    ap.add_argument("--apply", action="store_true", help="move bad files to data/external_rejected/")
    args = ap.parse_args()
    rejected_root = Path(args.external).parent / "external_rejected"

    rows = []
    # Inspect every discovered external edit file
    for edit, p in edited_files(args.external):
        with Image.open(p) as im:
            size = im.size
            # Convert to signed 16-bit integer array to prevent underflow during subtraction
            img = np.asarray(im.convert("RGB"), dtype=np.int16)

        ref_path = Path(args.export) / "train" / p.name

        # Enforce spatial dimension alignment with the watermark block grid
        if size != (IMG_SIZE, IMG_SIZE):
            status, frac = "wrong size", np.nan
        # Allow edits if the original reference is missing from local export cache
        elif not ref_path.exists():
            status, frac = "ok (no export to compare)", np.nan
        else:
            # Measure delta against pre-attack exported watermarked image
            ref = np.asarray(Image.open(ref_path).convert("RGB"), dtype=np.int16)
            # Compute fraction of pixels where max channel difference exceeds CHANGED threshold
            frac = float((np.abs(img - ref).max(axis=-1) > CHANGED).mean())
            status = "ok" if frac >= MIN_EDITED else "unedited"

        rows.append((edit, p, size, status, frac))

    if not rows:
        raise SystemExit(f"no edited images found under {args.external}/<edit>/train/")

    # Print summary reports of rejected files
    for status in ("wrong size", "unedited"):
        bad = [r for r in rows if r[3] == status]
        if bad:
            print(f"{status}: {len(bad)} file(s)")
            for edit, p, size, _, _ in bad[:10]:
                print(f"   {edit}/{p.name}  {size[0]}x{size[1]}")
            if len(bad) > 10:
                print(f"   ... and {len(bad) - 10} more")

    # Aggregate usable candidates and compute statistics on altered image areas
    ok = [r for r in rows if r[3].startswith("ok")]
    fr = np.array([r[4] for r in ok if not np.isnan(r[4])])
    print(f"\nusable edits: {len(ok)} of {len(rows)}")
    if len(fr):
        print(f"area changed per edit: median {np.median(fr):.1%}, range {fr.min():.1%} to {fr.max():.1%} "
              f"(aim for roughly 10-35%)")

    # Quarantine invalid files into rejection folders if --apply flag is set
    to_move = [r for r in rows if not r[3].startswith("ok")]
    if to_move and args.apply:
        for edit, p, _, status, _ in to_move:
            dst = rejected_root / status.replace(" ", "_") / edit / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))
        print(f"\nmoved {len(to_move)} file(s) to {rejected_root}")
        print("Next: python scripts\\check_data.py --external data\\external")
    elif to_move:
        print("\nNothing moved yet. Run again with --apply to move these files aside.")


if __name__ == "__main__":
    main()
