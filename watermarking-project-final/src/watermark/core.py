"""Transforms shared by embed and extract: colour space, DWT, block DCT, keyed bit assignment.

Provides foundational spatial and transform-domain primitives:
  - Color space conversions: YCrCb decomposition isolating the luminance (Y) carrier channel.
  - Discrete Wavelet Transform (DWT): Single-level 2D Haar decomposition to obtain the LL subband.
  - Block 2D Orthonormal DCT: Segmentation into 8x8 blocks and block-wise frequency transformation.
  - Keyed Permutation: Deterministic pseudo-random spreading of codeword bits across spatial blocks.
"""
import cv2
import numpy as np
import pywt
from scipy.fft import dctn, idctn

from .config import BLOCK, WAVELET


def load_image(path):
    """Read an image as BGR uint8 (grayscale images are returned 2-D).

    Args:
        path (str or Path): Filepath to the target image.

    Returns:
        np.ndarray: Loaded image array in BGR (3 channels) or grayscale (2-D) format.

    Raises:
        FileNotFoundError: If the image cannot be read or does not exist at the path.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    # Strip alpha channel if present, standardizing to 3-channel BGR
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def save_image(path, img):
    """Save an image array to disk using OpenCV.

    Args:
        path (str or Path): Destination filepath.
        img (np.ndarray): Image array to write (uint8).
    """
    cv2.imwrite(str(path), img)


def split_luma(img):
    """Return (Y as float64, YCrCb image or None). Watermark lives in Y only.

    Converts BGR imagery into YCrCb and extracts the luminance channel (Y)
    as high-precision float64 to prevent numerical truncation during transform steps.

    Args:
        img (np.ndarray): Input BGR or grayscale image array.

    Returns:
        tuple[np.ndarray, np.ndarray or None]: (Luminance Y channel float64, full YCrCb float64 array or None).
    """
    # Grayscale images lack chrominance channels; return image array directly
    if img.ndim == 2:
        return img.astype(np.float64), None
    # Convert color representation and isolate luminance plane
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    return ycc[..., 0].copy(), ycc


def merge_luma(y, ycc):
    """Recombine modified luminance channel with original chrominance channels.

    Clips values to valid 8-bit dynamic range [0, 255] and converts back to BGR.

    Args:
        y (np.ndarray): Modified luminance plane (float64).
        ycc (np.ndarray or None): Original YCrCb container array, or None if grayscale.

    Returns:
        np.ndarray: Reconstructed uint8 image in BGR (or 2-D grayscale) format.
    """
    # Clamp luminance values to legal uint8 boundaries
    y = np.clip(np.round(y), 0, 255)
    if ycc is None:
        return y.astype(np.uint8)
    # Splice modified luminance back into the multi-channel container
    out = ycc.copy()
    out[..., 0] = y
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


def dwt(y):
    """Compute single-level 2D Discrete Wavelet Transform on luminance plane.

    Args:
        y (np.ndarray): 2-D float64 luminance channel.

    Returns:
        tuple[np.ndarray, tuple]: (LL approximation subband, (LH, HL, HH) detail subbands).
    """
    ll, details = pywt.dwt2(y, WAVELET)
    return ll, details


def idwt(ll, details):
    """Compute single-level 2D Inverse Discrete Wavelet Transform.

    Args:
        ll (np.ndarray): Approximation subband array.
        details (tuple): Tuple containing detail coefficient arrays (LH, HL, HH).

    Returns:
        np.ndarray: Reconstructed spatial luminance plane.
    """
    return pywt.idwt2((ll, details), WAVELET)


def grid_shape(img_shape):
    """Number of block rows/cols for an image of this shape.

    Calculates the 2D grid dimensions of 8x8 DCT blocks formed across the LL subband.
    Because single-level DWT halves spatial dimensions, an image requires dimensions
    divisible by 2 * BLOCK (i.e. multiples of 16).

    Args:
        img_shape (tuple): Spatial dimensions (height, width) of the original image.

    Returns:
        tuple[int, int]: (n_block_rows, n_block_cols) across the LL subband.

    Raises:
        ValueError: If height or width is not evenly divisible by 2 * BLOCK.
    """
    h, w = img_shape[:2]
    if h % (2 * BLOCK) or w % (2 * BLOCK):
        raise ValueError(f"image size {h}x{w} must be a multiple of {2 * BLOCK}; run prepare_images first")
    return h // (2 * BLOCK), w // (2 * BLOCK)


def to_blocks(band):
    """(H, W) -> (n_blocks, 8, 8), row-major block order.

    Slices a 2-D subband matrix into non-overlapping 8x8 spatial blocks.

    Args:
        band (np.ndarray): 2-D matrix (e.g., LL subband of shape H x W).

    Returns:
        np.ndarray: 3-D tensor of shape (n_blocks, BLOCK, BLOCK).
    """
    h, w = band.shape
    # Reshape and transpose axes to extract blocks in row-major sequence
    return (band.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
                .swapaxes(1, 2).reshape(-1, BLOCK, BLOCK))


def from_blocks(blocks, shape):
    """(n_blocks, 8, 8) -> (H, W) 2-D matrix.

    Reassembles non-overlapping 8x8 blocks back into a continuous 2-D array.

    Args:
        blocks (np.ndarray): 3-D tensor of shape (n_blocks, BLOCK, BLOCK).
        shape (tuple[int, int]): Target output spatial shape (H, W).

    Returns:
        np.ndarray: Reassembled 2-D continuous matrix.
    """
    h, w = shape
    # Invert swapaxes and reshape tensor back to the continuous spatial domain
    return (blocks.reshape(h // BLOCK, w // BLOCK, BLOCK, BLOCK)
                  .swapaxes(1, 2).reshape(h, w))


def block_dct(blocks):
    """Compute 2D orthonormal Discrete Cosine Transform on each 8x8 block.

    Args:
        blocks (np.ndarray): 3-D array of shape (n_blocks, BLOCK, BLOCK).

    Returns:
        np.ndarray: 3-D array of orthonormal DCT frequency coefficients.
    """
    return dctn(blocks, axes=(1, 2), norm="ortho")


def block_idct(coefs):
    """Compute 2D orthonormal Inverse Discrete Cosine Transform on each 8x8 block.

    Args:
        coefs (np.ndarray): 3-D array of DCT frequency coefficients.

    Returns:
        np.ndarray: 3-D array of reconstructed spatial blocks.
    """
    return idctn(coefs, axes=(1, 2), norm="ortho")


def bit_assignment(n_blocks, n_bits, key):
    """Keyed pseudo-random map block -> payload bit index.

    Every bit is repeated ~n_blocks/n_bits times, scattered over the image so a
    local edit (inpainting, crop) only removes some copies of each bit.

    Args:
        n_blocks (int): Total number of blocks in the grid (e.g., 1,024).
        n_bits (int): Codeword length in bits (e.g., 64).
        key (int): Seed for pseudo-random number generator ensuring reproducible permutation.

    Returns:
        np.ndarray: 1-D array of length n_blocks, mapping each block index to its assigned bit index.
    """
    # Seed RNG to ensure identical scattering during both embedding and blind extraction
    rng = np.random.default_rng(key)
    # Permute block sequence to scatter bit assignments across distant spatial coordinates
    order = rng.permutation(n_blocks)
    assign = np.empty(n_blocks, dtype=np.int64)
    # Distribute bits cyclically across the permuted block indices
    assign[order] = np.arange(n_blocks) % n_bits
    return assign
