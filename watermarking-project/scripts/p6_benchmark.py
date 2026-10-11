"""Phase 6: named benchmark table (PSNR, SSIM, BER, success rate) on the TEST split.

    python scripts/p6_benchmark.py --n 50
    python scripts/p6_benchmark.py --mock            # check the harness with a fake method

Uses the method-agnostic harness in watermark/benchmark.py with our uniform
watermark (success = raw BER <= 15%). For the uniform-vs-adaptive comparison use
p5_ablation.py. Outputs in results/:
  p6_summary.csv     one row per named condition
  p6_raw.csv         one row per image x condition
  p6_by_texture.csv  success rate per condition x texture category (if available)
"""
import argparse
from pathlib import Path

import numpy as np

from watermark import dataset
from watermark.benchmark import default_methods, format_summary, run_benchmark
from watermark.config import PAYLOAD_BITS, TEXTURE_CSV
from watermark.textures import load_texture_table

OUT = Path("results")


def mock_methods(payload, rng):
    """A fake watermark: adds light noise, 'extracts' the payload with occasional flips."""
    def embed_fn(img, bits):
        return np.clip(img + rng.normal(0, 1.0, img.shape), 0, 255).astype(np.uint8)

    def extract_fn(img):
        got = payload.copy()
        if rng.random() > 0.6:
            i = rng.integers(len(got))
            got[i] = 1 - got[i]
        return got

    return embed_fn, extract_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="number of test images (0 = all)")
    ap.add_argument("--mock", action="store_true", help="use a fake embed/extract to test the harness")
    args = ap.parse_args()

    names = dataset.require("test")
    names = names[: args.n] if args.n else names
    paths = [dataset.image_path("test", n) for n in names]
    rng = np.random.default_rng(0)
    payload = rng.integers(0, 2, PAYLOAD_BITS)
    embed_fn, extract_fn = mock_methods(payload, rng) if args.mock else default_methods()

    tag = "p6_mock" if args.mock else "p6"
    summary, raw = run_benchmark(paths, embed_fn, extract_fn, payload,
                                 output_csv=OUT / f"{tag}_summary.csv", raw_csv=OUT / f"{tag}_raw.csv")
    print(format_summary(summary).to_string(index=False))

    if TEXTURE_CSV.exists():
        tex = load_texture_table(TEXTURE_CSV)["texture_category"]
        raw["texture"] = raw.image.map(tex)
        by_tex = raw.pivot_table(index="condition", columns="texture", values="success", sort=False)
        by_tex = by_tex[[c for c in ("Smooth", "Medium", "Complex") if c in by_tex]]
        by_tex.round(4).to_csv(OUT / f"{tag}_by_texture.csv")
        print("\nSuccess rate by texture:")
        print(by_tex.round(3).to_string())
    print(f"\nsaved results/{tag}_*.csv")


if __name__ == "__main__":
    main()
