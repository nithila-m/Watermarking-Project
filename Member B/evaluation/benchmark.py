from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

from evaluation.metrics import WatermarkMetrics
from attacks.classical import ClassicalAttacks
from attacks.ai_attacks import AIAttacks

def run_benchmark(
    test_image_paths: list,
    embed_fn,
    extract_fn,
    payload_bits: np.ndarray,
    output_csv_path: str = "results/summary_metrics.csv"
):
    Path("results").mkdir(exist_ok=True)

    attack_suite = {
        "Identity (No Attack)": lambda img: img,
        "JPEG (Q=70)": lambda img: ClassicalAttacks.jpeg_compression(img, quality=70),
        "JPEG (Q=50)": lambda img: ClassicalAttacks.jpeg_compression(img, quality=50),
        "Gaussian Noise (std=0.05)": lambda img: ClassicalAttacks.gaussian_noise(img, std=0.05),
        "Crop 10%": lambda img: ClassicalAttacks.random_crop_and_resize(img, crop_percent=0.10),
        "Gaussian Blur (k=5)": lambda img: ClassicalAttacks.gaussian_blur(img, kernel_size=5, sigma=1.5),
        "AI Inpaint Light (5%)": lambda img: AIAttacks.local_ai_rewrite(img, mask_ratio=0.05),
        "AI Inpaint Moderate (15%)": lambda img: AIAttacks.local_ai_rewrite(img, mask_ratio=0.15),
        "AI Neural Smoothing": lambda img: AIAttacks.ai_neural_filter_simulation(img),
    }

    records = []

    print(f"Running benchmark on {len(test_image_paths)} images across {len(attack_suite)} conditions...")

    for img_path in tqdm(test_image_paths, desc="Evaluating"):
        clean_img = Image.open(img_path).convert("RGB")
        
        # 1. Embed watermark
        watermarked_img = embed_fn(clean_img, payload_bits)

        # 2. Quality Metrics (Imperceptibility)
        psnr_val = WatermarkMetrics.compute_psnr(clean_img, watermarked_img)
        ssim_val = WatermarkMetrics.compute_ssim(clean_img, watermarked_img)

        # 3. Robustness under each attack
        for attack_name, attack_fn in attack_suite.items():
            distorted_img = attack_fn(watermarked_img)
            extracted_bits = extract_fn(distorted_img)
            
            ber_val = WatermarkMetrics.compute_ber(payload_bits, extracted_bits)
            is_success = WatermarkMetrics.is_successful_extraction(ber_val, threshold=0.15)

            records.append({
                "Image": Path(img_path).name,
                "Attack": attack_name,
                "PSNR": psnr_val,
                "SSIM": ssim_val,
                "BER": ber_val,
                "Extraction_Success": 1.0 if is_success else 0.0
            })

    raw_df = pd.DataFrame(records)
    
    summary_df = raw_df.groupby("Attack").agg({
        "PSNR": "mean",
        "SSIM": "mean",
        "BER": lambda x: f"{x.mean() * 100:.2f}%",
        "Extraction_Success": lambda x: f"{x.mean() * 100:.2f}%"
    }).reset_index()

    summary_df.to_csv(output_csv_path, index=False)
    return summary_df