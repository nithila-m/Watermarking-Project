r"""Phase 5b: evaluate the ablation configs under REAL Stable Diffusion edits.

    python scripts\p5_external.py
    python scripts\p5_external.py --external data\external --export results\colab_export

Reads results/colab_export/manifest.csv (bits and key of every exported test image)
and data/external/<edit>/test/<config>/<name>.png (the edited versions from Colab),
extracts the watermark and scores it exactly like p5_ablation.py.
Only extraction runs here, so this takes a minute or two.

Outputs in results/:
  p5x_raw.csv          one row per image x config x edit
  p5x_by_edit.csv      success rate per edit x config (the AI-edit headline table)
  p5x_stats.csv        paired 95% CIs vs uniform+ecc, per edit
  p5x_by_edit.png
"""
import argparse
from pathlib import Path

import matplotlib
# Use headless backend to allow plot generation without a display server
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from watermark import ecc
from watermark.config import PAYLOAD_BITS
from watermark.core import load_image
from watermark.external import (DEFAULT_EXPORT, DEFAULT_EXTERNAL, edit_names, is_erase,
                                read_manifest, str_to_bits, test_edit_path)
from watermark.extract import extract
from watermark.metrics import ber, is_successful_extraction, nc, psnr
from watermark.stats import compare

# Output directory for evaluation logs, tabular summaries, and charts
OUT = Path("results")


def score(img, row):
    """Extracts the watermark from an image and computes verification and quality metrics.

    Args:
        img (np.ndarray): The watermarked or edited image array.
        row (NamedTuple): Manifest row containing metadata (payload, key, ECC flag, message).

    Returns:
        dict: Evaluation metrics including BER, bit accuracy, normalized correlation,
              and strict/lenient success indicators.
    """
    # Parse ground truth binary payload from manifest string
    payload = str_to_bits(row.payload)
    # Blindly extract raw bit sequence using the assigned pseudo-random permutation key
    got = extract(img, PAYLOAD_BITS, int(row.key))

    # Evaluate extraction success depending on whether ECC is active
    if row.ecc:
        # Decode via Reed-Solomon codec and verify exact byte match against original message
        dec, ok = ecc.decode(got)
        success = bool(ok and np.array_equal(dec, str_to_bits(row.message)))
    else:
        # Direct raw comparison for uncoded watermarking configurations
        success = bool(np.array_equal(got, payload))

    # Compute Hamming distance-based Bit Error Rate
    b = ber(payload, got)
    return {"ber": b, "bit_acc": 1 - b, "nc": nc(payload, got), "success": success,
            "ber_ok": is_successful_extraction(b)}


def main():
    """Main execution function to score watermarks against external Stable Diffusion edits."""
    # Parse directory arguments for external edits and export metadata
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", default=str(DEFAULT_EXTERNAL))
    ap.add_argument("--export", default=str(DEFAULT_EXPORT))
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # Read manifest of exported images and filter down to the test split
    man = read_manifest(args.export)
    man = man[man.split == "test"]
    if man.empty:
        raise SystemExit("no test rows in the manifest. Run export_for_colab.py --train 0 --test 100.")

    # Retrieve all recognized edit condition names from the external directory
    edits = edit_names(args.external)
    rows, missing = [], 0

    # Iterate through manifest records to find and evaluate corresponding image files
    for row in man.itertuples():
        wm_path = Path(args.export) / "test" / row.config / f"{row.image}.png"
        wm = load_image(wm_path) if wm_path.exists() else None

        # Sanity check evaluation on the unedited watermarked baseline
        if wm is not None:   # sanity row: the exported image itself, before any edit
            rows.append({"image": row.image, "config": row.config, "edit": "none", "type": "none",
                         "edit_psnr": 100.0, **score(wm, row)})

        # Evaluate every external generative AI edit for the current test case
        for e in edits:
            p = test_edit_path(args.external, e, row.config, row.image)
            if p is None:
                missing += 1
                continue
            img = load_image(p)
            rows.append({"image": row.image, "config": row.config, "edit": e,
                         "type": "erasure" if is_erase(e) else "noise",
                         "edit_psnr": round(psnr(wm, img), 2) if wm is not None else np.nan,
                         **score(img, row)})

    if not rows:
        raise SystemExit(f"no edited test images found under {args.external}/<edit>/test/<config>/")

    # Save granular trial observations to disk
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p5x_raw.csv", index=False)
    if missing:
        print(f"note: {missing} (image, config, edit) combinations have no edited file yet "
              f"(Colab run unfinished?). Results use what exists.")

    # Maintain consistent presentation order across evaluated configurations
    configs = [c for c in ["uniform", "uniform+ecc", "heuristic+ecc", "adaptive", "adaptive+ecc"]
               if c in set(df.config)]

    # Compile condition-level summary table of success rates and visual quality
    by_edit = df.pivot_table(index="edit", columns="config", values="success")[configs]
    by_edit["edit PSNR (dB)"] = df.groupby("edit").edit_psnr.mean()
    by_edit["images"] = df.groupby("edit").image.nunique()
    by_edit.round(4).to_csv(OUT / "p5x_by_edit.csv")

    print("Success rate under real AI edits (strict: RS-decoded message exactly right):")
    print(by_edit.round(3).to_string())

    # Flag unedited test images that failed verification before any distortion occurred
    if (df[df.edit == "none"].success == False).any():  # noqa: E712
        print("! some exported images fail even before editing: check that --export points at "
              "the same folder that was uploaded to Colab")

    # Paired statistics need the "attack"/"severity" columns that stats.compare expects
    case = df[df.edit != "none"].assign(attack=lambda d: d.edit, severity="real")
    base = "uniform+ecc"
    pairs = [(c, base) for c in configs if c != base and c.endswith("ecc")]

    # Calculate paired bootstrap confidence intervals and McNemar significance tests
    if base in configs and pairs:
        stats = pd.concat([compare(case, pairs).assign(split_by="all"),
                           compare(case, pairs, by="edit").assign(split_by="edit")], ignore_index=True)
        stats.round(5).to_csv(OUT / "p5x_stats.csv", index=False)
        print("\nPaired difference vs uniform+ecc (95% CI over images):")
        print(stats[["group", "A", "diff", "lo", "hi", "p_value", "significant"]].round(4).to_string(index=False))

    # Generate comparative bar chart of success rates across real AI edits
    ax = by_edit[configs].drop(index="none", errors="ignore").plot.bar(figsize=(9, 4))
    ax.axhline(0.8, ls="--", c="gray", lw=1)
    ax.set_ylabel("extraction success rate")
    ax.set_title("Real Stable Diffusion edits (equal PSNR)")
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(OUT / "p5x_by_edit.png", dpi=130)
    plt.close()
    print("\nsaved results/p5x_*.csv and p5x_by_edit.png")


if __name__ == "__main__":
    main()
