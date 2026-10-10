import numpy as np
from PIL import Image
from evaluation.metrics import WatermarkMetrics

# 1. Test visual metrics on an identical image pair
img1 = Image.open("data/clean/test/000001.png").convert("RGB")
img2 = img1.copy()

psnr_val = WatermarkMetrics.compute_psnr(img1, img2)
ssim_val = WatermarkMetrics.compute_ssim(img1, img2)
print(f"Identical Image -> PSNR: {psnr_val} dB (inf expected), SSIM: {ssim_val:.4f} (1.0 expected)")

# 2. Test bit error rate (BER)
secret_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
extracted_bits = np.array([1, 0, 1, 0, 0, 0, 1, 0])  # 1 bit error out of 8 (12.5% BER)

ber = WatermarkMetrics.compute_ber(secret_bits, extracted_bits)
success = WatermarkMetrics.is_successful_extraction(ber, threshold=0.15)
print(f"Bits Test -> BER: {ber * 100:.2f}%, Recoverable: {success}")