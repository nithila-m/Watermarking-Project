r"""Final Member C embed -> attack -> extract -> ECC demo.

CLI example:
    python scripts\demo.py --image data\clean\test\grass.png --text TEST --attack blur --severity moderate
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from watermark import ecc
from watermark.attacks import SEVERITIES, attack
from watermark.config import IMG_SIZE, MESSAGE_BITS, PAYLOAD_BITS, TARGET_PSNR
from watermark.core import load_image, save_image
from watermark.embed import embed_at_psnr
from watermark.extract import extract
from watermark.metrics import ber, psnr, ssim
from watermark.strength_map import uniform_map

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
ATTACK_CHOICES = ("none", "jpeg", "blur", "resize", "crop", "smooth")
DEFAULT_KEY = 2024
DEFAULT_SEED = 2026


def prepare_image(image):
    """Validate an image and center-crop/resize it to the pipeline's 512x512 input."""
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise ValueError("Image pixels must be 8-bit; upload a standard PNG, JPEG, BMP, or WebP image.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Unsupported image layout; provide a grayscale, RGB, or RGBA image.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("The uploaded image is empty.")

    height, width = image.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    square = image[top:top + side, left:left + side]
    interpolation = cv2.INTER_AREA if side >= IMG_SIZE else cv2.INTER_CUBIC
    return cv2.resize(square, (IMG_SIZE, IMG_SIZE), interpolation=interpolation)


def load_demo_image(path):
    """Load a supported image path and standardize it for the watermark pipeline."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format {path.suffix or '(no extension)'}. "
            f"Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    try:
        image = load_image(path)
    except FileNotFoundError as exc:
        raise ValueError(f"Could not read image: {path}") from exc
    return prepare_image(image)


def decode_uploaded_image(contents):
    """Decode uploaded image bytes, raising a user-readable error on invalid data."""
    if not contents:
        raise ValueError("Upload an image or choose a sample image.")
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("The uploaded file is not a supported or readable image.")
    return prepare_image(image)


def validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a non-empty watermark.")
    encoded = text.encode("utf-8")
    capacity = MESSAGE_BITS // 8
    if len(encoded) > capacity:
        raise ValueError(
            f"Watermark is too long: the production ECC payload supports at most "
            f"{capacity} UTF-8 bytes (received {len(encoded)})."
        )
    return encoded


def run_demo(image, text, attack_kind="none", severity="moderate",
             key=DEFAULT_KEY, seed=DEFAULT_SEED):
    """Run the current production embed, attack, extract, and ECC pipeline."""
    original = prepare_image(image)
    validate_text(text)
    if attack_kind not in ATTACK_CHOICES:
        raise ValueError(f"Unsupported attack {attack_kind!r}. Choose from {ATTACK_CHOICES}.")
    if attack_kind != "none" and severity not in SEVERITIES:
        raise ValueError(f"Unsupported severity {severity!r}. Choose from {SEVERITIES}.")

    message_bits = ecc.text_to_bits(text)
    payload = ecc.encode(message_bits)
    if len(payload) != PAYLOAD_BITS:
        raise RuntimeError(f"ECC produced {len(payload)} bits; expected {PAYLOAD_BITS}.")

    watermarked, _, embed_psnr = embed_at_psnr(
        original, payload, uniform_map(original), int(key), target_psnr=TARGET_PSNR
    )
    attacked = (
        watermarked.copy()
        if attack_kind == "none"
        else attack(watermarked, attack_kind, severity, seed=int(seed))
    )
    extracted = extract(attacked, PAYLOAD_BITS, int(key))
    decoded, ecc_success = ecc.decode(extracted)
    recovered_text = ecc.bits_to_text(decoded).rstrip(" ")
    exact_recovery = bool(ecc_success and np.array_equal(decoded, message_bits))
    error_rate = ber(payload, extracted)

    return {
        "original_image": original,
        "watermarked_image": watermarked,
        "attacked_image": attacked,
        "original_text": text,
        "recovered_text": recovered_text,
        "exact_recovery": exact_recovery,
        "ecc_success": bool(ecc_success),
        "bit_accuracy": 1.0 - error_rate,
        "ber": error_rate,
        "psnr": psnr(original, watermarked),
        "ssim": ssim(original, watermarked),
        "embed_psnr": embed_psnr,
        "attack": attack_kind,
        "severity": "none" if attack_kind == "none" else severity,
    }


def format_result(result):
    """Format pipeline metrics consistently for terminal use."""
    return "\n".join((
        f"Original watermark: {result['original_text']}",
        f"Recovered watermark: {result['recovered_text'] or '(empty)'}",
        f"Exact recovery: {'YES' if result['exact_recovery'] else 'NO'}",
        f"ECC success: {'YES' if result['ecc_success'] else 'NO'}",
        f"Bit accuracy: {result['bit_accuracy']:.2%}",
        f"BER: {result['ber']:.4f}",
        f"PSNR: {result['psnr']:.2f} dB",
        f"SSIM: {result['ssim']:.4f}",
        f"Attack: {result['attack']}",
        f"Severity: {result['severity']}",
    ))


def main():
    parser = argparse.ArgumentParser(description="Run the production Member C watermark demo.")
    parser.add_argument("--image", required=True, help="Input image path (PNG/JPEG/BMP/WebP)")
    parser.add_argument("--text", required=True, help="Watermark text (up to 4 UTF-8 bytes)")
    parser.add_argument("--attack", choices=ATTACK_CHOICES, default="none")
    parser.add_argument("--severity", choices=SEVERITIES, default="moderate")
    parser.add_argument("--key", type=int, default=DEFAULT_KEY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save-dir", help="Optionally save watermarked and attacked images")
    args = parser.parse_args()

    try:
        image = load_demo_image(args.image)
        result = run_demo(
            image, args.text, args.attack, args.severity, args.key, args.seed
        )
        print(format_result(result))
        if args.save_dir:
            destination = Path(args.save_dir)
            destination.mkdir(parents=True, exist_ok=True)
            save_image(destination / "watermarked.png", result["watermarked_image"])
            save_image(destination / "attacked.png", result["attacked_image"])
            print(f"Saved watermarked and attacked images to {destination}")
    except (ValueError, OSError, cv2.error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
