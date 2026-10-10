import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

class WatermarkMetrics:
    """Core evaluation metrics for watermark imperceptibility and robustness."""

    @staticmethod
    def compute_psnr(img1: Image.Image, img2: Image.Image) -> float:
        """Computes Peak Signal-to-Noise Ratio between two PIL images."""
        arr1 = np.array(img1.convert("RGB"))
        arr2 = np.array(img2.convert("RGB"))
        return float(psnr_fn(arr1, arr2, data_range=255))

    @staticmethod
    def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
        """Computes Structural Similarity Index (SSIM) across RGB channels."""
        arr1 = np.array(img1.convert("RGB"))
        arr2 = np.array(img2.convert("RGB"))
        return float(ssim_fn(arr1, arr2, channel_axis=2, data_range=255))

    @staticmethod
    def compute_ber(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
        """
        Bit Error Rate: Proportion of mismatched bits.
        Range: 0.0 (perfect extraction) to 0.5 (random guess/destroyed).
        """
        original_bits = np.asarray(original_bits).flatten()
        extracted_bits = np.asarray(extracted_bits).flatten()
        assert len(original_bits) == len(extracted_bits), "Bit vectors must have equal length"
        errors = np.sum(original_bits != extracted_bits)
        return float(errors / len(original_bits))

    @staticmethod
    def is_successful_extraction(ber: float, threshold: float = 0.15) -> bool:
        """
        Returns True if BER is within the error-correcting threshold (e.g. BCH/Reed-Solomon).
        """
        return ber <= threshold