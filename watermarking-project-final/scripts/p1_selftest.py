"""Phase 1 check: the uniform embedder/extractor works before anything else is built.

    python scripts/p1_selftest.py

Pass criteria (printed at the end):
  * BER = 0 with no attack on every image
  * PSNR >= 40 dB after calibration
  * BER < 5% after JPEG quality 70
Also prints a strength sweep so you can see the quality/robustness trade-off.
"""
import numpy as np

from watermark import dataset, ecc
from watermark.attacks import attack
from watermark.config import PAYLOAD_BITS, TARGET_PSNR
from watermark.embed import embed, embed_at_psnr
from watermark.extract import extract
from watermark.metrics import ber, psnr, ssim

# Deterministic key used for pseudo-random block scattering across the 8x8 DCT grid
KEY = 2024


def main():
    """Validates baseline components: Reed-Solomon ECC, uniform embedding/extraction, and PSNR calibration."""
    # Load the first 12 training images for initial smoke testing
    names = dataset.require("train")[:12]
    rng = np.random.default_rng(0)
    ok = True

    # ---------------- 1. Reed-Solomon Error Correction Code Validation ----------------
    print("1) ECC round trip")
    # Generate random 32-bit (4-byte) message
    msg = rng.integers(0, 2, 32)
    # Encode with RS(8, 4) to yield a 64-bit codeword (4 message bytes + 4 parity bytes)
    coded = ecc.encode(msg)
    corrupted = coded.copy()
    # Flip bit indices 3 and 20 (lying in two separate symbol bytes to test the 2-byte correction capacity)
    corrupted[[3, 20]] ^= 1                      # two bit errors in two different bytes
    dec, good = ecc.decode(corrupted)
    print(f"   coded length {len(coded)} bits, 2 errors corrected: {good and np.array_equal(dec, msg)}")

    # ---------------- 2. Uniform Embedding & Extraction at Target PSNR ----------------
    print(f"\n2) Uniform embed at {TARGET_PSNR} dB -> extract")
    print(f"   {'image':28s} {'strength':>8s} {'PSNR':>6s} {'SSIM':>6s} {'BER clean':>9s} {'BER jpeg70':>10s}")
    for n in names:
        img = dataset.load("train", n)
        bits = rng.integers(0, 2, PAYLOAD_BITS)

        # Embed uniformly across all 1,024 blocks (relative weights = np.ones(1024)) calibrated to TARGET_PSNR
        wm, s, p = embed_at_psnr(img, bits, np.ones(1024), KEY)

        # Baseline extraction with no distortions applied
        b0 = ber(bits, extract(wm, PAYLOAD_BITS, KEY))
        # Extraction survival following moderate lossy JPEG compression (Q=70)
        bj = ber(bits, extract(attack(wm, "jpeg", "moderate"), PAYLOAD_BITS, KEY))

        # Flag failures if clean BER > 0, calibrated PSNR drops below threshold, or JPEG BER >= 5%
        flag = "" if (b0 == 0 and p >= TARGET_PSNR - 0.5 and bj < 0.05) else "   <-- check"
        ok &= flag == ""
        print(f"   {n:28s} {s[0]:8.1f} {p:6.2f} {ssim(img, wm):6.3f} {b0:9.3f} {bj:10.3f}{flag}")

    # ---------------- 3. Distortion Trade-off Sweep Across Static Strengths ----------------
    print("\n3) Strength sweep on the first image (fixed strength, no calibration)")
    img = dataset.load("train", names[0])
    bits = rng.integers(0, 2, PAYLOAD_BITS)
    print(f"   {'strength':>8s} {'PSNR':>6s} {'BER jpeg50':>10s} {'BER blur1.6':>11s}")
    # Sweep global embedding strength s directly without PSNR binary-search calibration
    for s in [5, 10, 20, 40, 60, 80]:
        wm = embed(img, bits, s, KEY)
        # Evaluate visual imperceptibility and robustness against severe JPEG (Q=50) and severe blur (sigma=1.6)
        print(f"   {s:8d} {psnr(img, wm):6.2f} "
              f"{ber(bits, extract(attack(wm, 'jpeg', 'severe'), PAYLOAD_BITS, KEY)):10.3f} "
              f"{ber(bits, extract(attack(wm, 'blur', 'severe'), PAYLOAD_BITS, KEY)):11.3f}")

    # ---------------- Final Test Status Assessment ----------------
    print("\nPHASE 1 PASSED" if ok else "\nSome images need attention (see '<-- check')")


if __name__ == "__main__":
    main()
