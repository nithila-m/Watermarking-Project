from pathlib import Path
from PIL import Image
from attacks.ai_attacks import AIAttacks

test_img_path = Path("data/clean/test/000001.png")
img = Image.open(test_img_path).convert("RGB")

print("Testing AI attack functions...")

# 1. Test Light Inpainting (5% area)
light_inpaint = AIAttacks.local_ai_rewrite(img, mask_ratio=0.05)
light_inpaint.save("results/test_ai_inpaint_light.png")

# 2. Test Moderate Inpainting (15% area)
mod_inpaint = AIAttacks.local_ai_rewrite(img, mask_ratio=0.15)
mod_inpaint.save("results/test_ai_inpaint_moderate.png")

# 3. Test AI Neural Enhancement Filter
neural_img = AIAttacks.ai_neural_filter_simulation(img)
neural_img.save("results/test_ai_neural_filter.png")

print("All AI attacks executed! Samples saved in 'results/' directory.")