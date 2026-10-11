"""Shared constants. Agree these as a team and don't change them silently."""
from pathlib import Path

# ---------------- data layout ----------------
DATA_DIR = Path("data/clean")    # data/clean/{train,val,test}/<name>.png  (made by scripts/prepare_dataset.py)
SPLITS = ("train", "val", "test")
TEXTURE_CSV = DATA_DIR / "test_metadata.csv"   # texture category per test image (scripts/analyze_textures.py)

IMG_SIZE = 512          # all images are resized to IMG_SIZE x IMG_SIZE
WAVELET = "haar"        # one-level DWT on the Y channel; we embed in the LL band
BLOCK = 8               # 8x8 DCT blocks inside the LL band (= 16x16 pixels in the image)

# The two mid-frequency DCT coefficients compared in every block.
# bit 1  ->  C[A] - C[B] >= +s      bit 0  ->  C[A] - C[B] <= -s
COEF_A = (2, 3)
COEF_B = (3, 2)

PAYLOAD_BITS = 64       # bits written into the image (message + ECC parity)
MESSAGE_BITS = 32       # message bits when ECC is on (4 bytes)
ECC_PARITY_BYTES = 4    # Reed-Solomon parity bytes (corrects up to 2 byte errors)

TARGET_PSNR = 42.0      # every configuration is calibrated to this PSNR (fair comparison)
                        # 40 dB made the uniform baseline near-perfect on smooth images,
                        # leaving no room to show what adaptivity adds
DEFAULT_STRENGTH = 12.0 # starting uniform strength before PSNR calibration

# Cost-aware skipping: a block whose host coefficients disagree with its bit by more
# than SKIP_RATIO x its target margin is left alone (its vote is wrong, but the
# other copies of that bit outvote it). Without this, textured images (grass,
# gravel) can't reach the PSNR target at all. Applies to every configuration.
SKIP_RATIO = 3.0

# Lenient success criterion used by the benchmark (raw BER at or below this is
# treated as recoverable by an error-correcting code). The strict criterion in
# evaluate.py is "the RS-decoded message is exactly right".
BER_SUCCESS_THRESHOLD = 0.15
