import io
import cv2
import numpy as np
from PIL import Image

class ClassicalAttacks:
    """Standard image distortions for watermarking evaluation."""

    @staticmethod
    def jpeg_compression(img: Image.Image, quality: int = 50) -> Image.Image:
        """Simulates lossy JPEG compression."""
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    @staticmethod
    def gaussian_noise(img: Image.Image, std: float = 0.05) -> Image.Image:
        """Adds zero-mean Gaussian noise to the image."""
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, std, arr.shape)
        noisy = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((noisy * 255.0).astype(np.uint8))

    @staticmethod
    def random_crop_and_resize(img: Image.Image, crop_percent: float = 0.15) -> Image.Image:
        """Crops borders by crop_percent and resamples back to original dimensions."""
        w, h = img.size
        dw, dh = int(w * crop_percent), int(h * crop_percent)
        cropped = img.crop((dw, dh, w - dw, h - dh))
        return cropped.resize((w, h), Image.Resampling.BILINEAR)

    @staticmethod
    def gaussian_blur(img: Image.Image, kernel_size: int = 5, sigma: float = 1.5) -> Image.Image:
        """Applies spatial Gaussian blur."""
        arr = np.array(img)
        # kernel_size must be odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(arr, (kernel_size, kernel_size), sigma)
        return Image.fromarray(blurred)