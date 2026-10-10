import numpy as np
import cv2
from PIL import Image, ImageDraw

class AIAttacks:
    """Simulates Generative AI edits, localized inpainting, and neural post-processing."""

    @staticmethod
    def generate_random_inpaint_mask(width: int = 512, height: int = 512, area_ratio: float = 0.15) -> np.ndarray:
        """
        Creates a binary mask (0 = keep, 255 = inpaint) representing an AI edit zone.
        area_ratio: fraction of the image area to rewrite (e.g., 0.15 = 15%).
        """
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Calculate bounding box dimensions targeting the requested area
        side = int(np.sqrt(width * height * area_ratio))
        x1 = np.random.randint(0, max(1, width - side))
        y1 = np.random.randint(0, max(1, height - side))
        draw.rectangle([x1, y1, x1 + side, y1 + side], fill=255)
        return np.array(mask)

    @staticmethod
    def cv2_telea_inpaint(img: Image.Image, mask: np.ndarray) -> Image.Image:
        """
        Fast baseline algorithmic inpainting (Telea algorithm).
        Reconstructs selected mask region using boundary continuity.
        """
        img_arr = np.array(img.convert("RGB"))
        inpainted = cv2.inpaint(img_arr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return Image.fromarray(inpainted)

    @staticmethod
    def ai_neural_filter_simulation(img: Image.Image) -> Image.Image:
        """
        Simulates AI beauty/neural enhancement filters that flatten fine textures
        while keeping sharp edges intact (Edge-preserving bilateral smoothing).
        """
        img_arr = np.array(img.convert("RGB"))
        # Bilateral filter flattens high-frequency watermark noise in flat regions
        enhanced = cv2.bilateralFilter(img_arr, d=9, sigmaColor=75, sigmaSpace=75)
        return Image.fromarray(enhanced)

    @staticmethod
    def local_ai_rewrite(img: Image.Image, mask_ratio: float = 0.15) -> Image.Image:
        """Complete helper to apply random inpainting of a specified severity."""
        mask = AIAttacks.generate_random_inpaint_mask(img.width, img.height, area_ratio=mask_ratio)
        return AIAttacks.cv2_telea_inpaint(img, mask)