from pathlib import Path
import numpy as np
from PIL import Image
from evaluation.benchmark import run_benchmark

print("Starting benchmark run...")

# 1. Define dummy 32-bit payload
payload = np.random.randint(0, 2, size=32)

# 2. Mock functions simulating Person A's pipeline
def mock_embed(img: Image.Image, bits: np.ndarray) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    arr += np.random.normal(0, 1.0, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def mock_extract(img: Image.Image) -> np.ndarray:
    extracted = payload.copy()
    if np.random.rand() > 0.6:
        idx = np.random.randint(0, len(payload))
        extracted[idx] = 1 - extracted[idx]
    return extracted

# 3. Take 5 sample images
sample_test_images = sorted(list(Path("data/clean/test").glob("*.png")))[:5]
print(f"Loaded {len(sample_test_images)} sample images.")

# 4. Run the benchmark
summary = run_benchmark(
    test_image_paths=sample_test_images,
    embed_fn=mock_embed,
    extract_fn=mock_extract,
    payload_bits=payload,
    output_csv_path="results/mock_summary.csv"
)

print("\n--- FINAL BENCHMARK SUMMARY ---")
print(summary.to_string(index=False))