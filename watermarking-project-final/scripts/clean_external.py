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

# Per-channel intensity threshold: delta > 12 flags an edited pixel
CHANGED = 12
# Area fraction floor: files with < 0.5% changed pixels are marked unedited
MIN_EDITED = 0.005


def edited_files(root):
    """Discovers external edited images under hierarchical or flat directory structures.

    Args:
        root (str or Path): Root folder holding external edit subfolders.

    Yields:
        tuple[str, Path]: The edit condition name and the absolute or relative file path.
    """
    for edit in edit_names(root):
        d = Path(root) / edit
        # Search both standard split hierarchy (train/*.png) and flat layout (*.png)
        for p in sorted((d / "train").glob("*.png")) + sorted(d.glob("*.png")):
            yield edit, p


def main():
    """Validates external edited images and optionally moves defective ones to quarantine."""
    # Configure command-line arguments for directories and modification flags
    ap = argparse.ArgumentParser(description="Audit and clean external edited images before training.")
    ap.add_argument("--external", default=str(DEFAULT_EXTERNAL), help="path to external edits folder")
    ap.add_argument("--export", default=str(DEFAULT_EXPORT), help="path to exported baseline watermarked images")
    ap.add_argument("--apply", action="store_true", help="move bad files to data/external_rejected/")
    args = ap.parse_args()
    rejected_root = Path(args.external).parent / "external_rejected"

    rows = []
    # Inspect every discovered image against quality criteria
    for edit, p in edited_files(args.external):
        with Image.open(p) as im:
            size = im.size
            # Convert to 16-bit signed int to prevent arithmetic underflow/overflow on subtraction
            img = np.asarray(im.convert("RGB"), dtype=np.int16)
        ref_path = Path(args.export) / "train" / p.name
        
        # Verify resolution preserves 8x8 block alignment
        if size != (IMG_SIZE, IMG_SIZE):
            status, frac = "wrong size", np.nan
        # If no exported reference exists to compare against, tentatively accept
        elif not ref_path.exists():
            status, frac = "ok (no export to compare)", np.nan
        else:
            # Measure deviation from the unedited exported watermarked baseline
            ref = np.asarray(Image.open(ref_path).convert("RGB"), dtype=np.int16)
            frac = float((np.abs(img - ref).max(axis=-1) > CHANGED).mean())
            status = "ok" if frac >= MIN_EDITED else "unedited"
        rows.append((edit, p, size, status, frac))

    if not rows:
        raise SystemExit(f"no edited images found under {args.external}/<edit>/train/")
        
    # Output breakdown of non-viable files (wrong dimensions or unedited copies)
    for status in ("wrong size", "unedited"):
        bad = [r for r in rows if r[3] == status]
        if bad:
            print(f"{status}: {len(bad)} file(s)")
            for edit, p, size, _, _ in bad[:10]:
                print(f"   {edit}/{p.name}  {size[0]}x{size[1]}")
            if len(bad) > 10:
                print(f"   ... and {len(bad) - 10} more")
                
    # Report statistics on retained usable files
    ok = [r for r in rows if r[3].startswith("ok")]
    fr = np.array([r[4] for r in ok if not np.isnan(r[4])])
    print(f"\nusable edits: {len(ok)} of {len(rows)}")
    if len(fr):
        print(f"area changed per edit: median {np.median(fr):.1%}, range {fr.min():.1%} to {fr.max():.1%} "
              f"(aim for roughly 10-35%)")

    # Quarantine invalid files into rejection folders if --apply was supplied
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
