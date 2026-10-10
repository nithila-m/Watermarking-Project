"""Phase 1 check: the uniform embedder/extractor works before anything else is built.

    python scripts/p1_selftest.py

Pass criteria (printed at the end):
  * BER = 0 with no attack on every image
  * PSNR >= 40 dB after calibration
  * BER < 5% after JPEG quality 70
Also prints a strength sweep so you can see the quality/robustness trade-off.
"""
import json
from pathlib import Path

import numpy as np

from watermark import ecc
from watermark.attacks import attack
from watermark.config import PAYLOAD_BITS, TARGET_PSNR
from watermark.core import load_image
from watermark.embed import embed, embed_at_psnr
from watermark.extract import extract
from watermark.metrics import ber, psnr, ssim

DATA = Path("data/processed")
KEY = 2024


def main():
    names = json.loads((DATA / "split.json").read_text())["train"][:12]
    rng = np.random.default_rng(0)
    ok = True

    print("1) ECC round trip")
    msg = rng.integers(0, 2, 32)
    coded = ecc.encode(msg)
    corrupted = coded.copy()
    corrupted[[3, 20]] ^= 1                      # two bit errors in two different bytes
    dec, good = ecc.decode(corrupted)
    print(f"   coded length {len(coded)} bits, 2 errors corrected: {good and np.array_equal(dec, msg)}")

    print(f"\n2) Uniform embed at {TARGET_PSNR} dB -> extract")
    print(f"   {'image':28s} {'strength':>8s} {'PSNR':>6s} {'SSIM':>6s} {'BER clean':>9s} {'BER jpeg70':>10s}")
    for n in names:
        img = load_image(DATA / f"{n}.png")
        bits = rng.integers(0, 2, PAYLOAD_BITS)
        wm, s, p = embed_at_psnr(img, bits, np.ones(1024), KEY)
        b0 = ber(bits, extract(wm, PAYLOAD_BITS, KEY))
        bj = ber(bits, extract(attack(wm, "jpeg", "moderate"), PAYLOAD_BITS, KEY))
        flag = "" if (b0 == 0 and p >= TARGET_PSNR - 0.5 and bj < 0.05) else "   <-- check"
        ok &= flag == ""
        print(f"   {n:28s} {s[0]:8.1f} {p:6.2f} {ssim(img, wm):6.3f} {b0:9.3f} {bj:10.3f}{flag}")

    print("\n3) Strength sweep on the first image (fixed strength, no calibration)")
    img = load_image(DATA / f"{names[0]}.png")
    bits = rng.integers(0, 2, PAYLOAD_BITS)
    print(f"   {'strength':>8s} {'PSNR':>6s} {'BER jpeg50':>10s} {'BER blur1.6':>11s}")
    for s in [5, 10, 20, 40, 60, 80]:
        wm = embed(img, bits, s, KEY)
        print(f"   {s:8d} {psnr(img, wm):6.2f} "
              f"{ber(bits, extract(attack(wm, 'jpeg', 'severe'), PAYLOAD_BITS, KEY)):10.3f} "
              f"{ber(bits, extract(attack(wm, 'blur', 'severe'), PAYLOAD_BITS, KEY)):11.3f}")

    print("\nPHASE 1 PASSED" if ok else "\nSome images need attention (see '<-- check')")


if __name__ == "__main__":
    main()
