from pathlib import Path
from PIL import Image
from attacks.classical import ClassicalAttacks

img_path = Path("data/clean/test/000001.png")
img = Image.open(img_path).convert("RGB")

print("Testing classical attack functions on 000001.png...")

# Test each attack
jpeg_img = ClassicalAttacks.jpeg_compression(img, quality=50)
noise_img = ClassicalAttacks.gaussian_noise(img, std=0.05)
crop_img = ClassicalAttacks.random_crop_and_resize(img, crop_percent=0.10)
blur_img = ClassicalAttacks.gaussian_blur(img, kernel_size=5, sigma=1.5)

print("All classical attacks executed successfully!")