r"""Evaluate confidence-based Reed-Solomon erasures on real-image attacks.

Run with: python -m scripts.p10_erasure_recovery
"""
from pathlib import Path

import numpy as np
import pandas as pd
from reedsolo import ReedSolomonError, RSCodec

from scripts.p8_image_recovery import ATTACK_CASES, KEY, SEED, run_experiment
from scripts.p9_symbol_error_analysis import N_IMAGES, select_test_images
from watermark import ecc
from watermark.config import ECC_PARITY_BYTES, MESSAGE_BITS, PAYLOAD_BITS

# Sweep thresholds used to convert continuous extraction confidence into erasure flags
ERASURE_THRESHOLDS = (0.0, 0.25, 0.50, 0.75, 1.0)
OUTPUT_CSV = Path("results") / "p10_erasure_recovery.csv"
# Total Reed-Solomon symbol count (8 bytes for a 64-bit codeword)
RS_SYMBOLS = PAYLOAD_BITS // 8
# Reed-Solomon codec initialized with 4 parity symbols (corrects up to 2 errors or 4 erasures)
_rs = RSCodec(ECC_PARITY_BYTES)


def normalize_bit_confidence(vote_totals):
    """Scale absolute per-bit vote totals to [0, 1] within one extraction.

    Args:
        vote_totals (array-like): Raw soft-vote accumulation values per bit.

    Returns:
        np.ndarray: Normalized confidence array bounded by [0.0, 1.0].

    Raises:
        ValueError: If vote_totals length differs from PAYLOAD_BITS.
    """
    magnitudes = np.abs(np.asarray(vote_totals, dtype=np.float64).ravel())
    if len(magnitudes) != PAYLOAD_BITS:
        raise ValueError(f"expected {PAYLOAD_BITS} vote totals, got {len(magnitudes)}")
    maximum = float(magnitudes.max(initial=0.0))
    if maximum == 0:
        return np.zeros_like(magnitudes)
    return magnitudes / maximum


def byte_confidence(bit_confidence):
    """Use the least confident bit as the confidence of each packed RS byte.

    Because any single bit error corrupts the entire 8-bit Reed-Solomon symbol,
    the byte-level confidence is bottlenecked by the minimum bit confidence.

    Args:
        bit_confidence (array-like): Normalized confidence values per bit.

    Returns:
        np.ndarray: Confidence scores for each of the 8 Reed-Solomon byte symbols.

    Raises:
        ValueError: If bit_confidence length differs from PAYLOAD_BITS.
    """
    confidence = np.asarray(bit_confidence, dtype=np.float64).ravel()
    if len(confidence) != PAYLOAD_BITS:
        raise ValueError(f"expected {PAYLOAD_BITS} bit confidences, got {len(confidence)}")
    # Reshape 64 bits into 8 groups of 8 bits and take the minimum across each byte
    return confidence.reshape(RS_SYMBOLS, 8).min(axis=1)


def erasure_indices(confidence, threshold):
    """Return byte positions whose normalized confidence is below threshold.

    Args:
        confidence (array-like): Array of byte confidence scores.
        threshold (float): Confidence cutoff in [0.0, 1.0].

    Returns:
        list[int]: Byte index positions flagged as erasures.

    Raises:
        ValueError: If threshold is out of bounds or confidence shape is invalid.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("erasure threshold must be between 0 and 1")
    scores = np.asarray(confidence, dtype=np.float64).ravel()
    if len(scores) != RS_SYMBOLS:
        raise ValueError(f"expected {RS_SYMBOLS} byte confidences, got {len(scores)}")
    return np.flatnonzero(scores < threshold).astype(int).tolist()


def decode_with_erasures(coded_bits, positions):
    """Decode with reedsolo's supported erase_pos byte-position argument.

    Args:
        coded_bits (array-like): Received 64-bit candidate codeword.
        positions (iterable[int]): Identified byte indices marked as erased.

    Returns:
        tuple[np.ndarray, bool]: (Decoded message bit sequence, success flag).

    Raises:
        ValueError: If symbols or erasure positions are malformed.
    """
    data = ecc.bits_to_bytes(coded_bits)
    if len(data) != RS_SYMBOLS:
        raise ValueError(f"expected {RS_SYMBOLS} Reed-Solomon symbols, got {len(data)}")
    positions = list(positions)
    # Ensure positions are unique and fall within valid symbol indices [0, 7]
    if len(set(positions)) != len(positions) or any(
        not isinstance(position, (int, np.integer)) or not 0 <= position < RS_SYMBOLS
        for position in positions
    ):
        raise ValueError("erasure positions must be unique byte indices from 0 to 7")
    try:
        # Decode utilizing known erasure locations to expand correction capacity
        message, _, _ = _rs.decode(data, erase_pos=positions)
        return ecc.bytes_to_bits(message), True
    except ReedSolomonError:
        # Fall back to uncorrected payload prefix on decoding failure
        return np.asarray(coded_bits[:MESSAGE_BITS], dtype=np.int64), False


def _message_success(decoded, decoded_ok, expected):
    """Validates that decoding succeeded and matches the ground-truth bit sequence."""
    return bool(decoded_ok and np.array_equal(decoded, expected))


def analyze_image(image, image_name, attack_cases=ATTACK_CASES, key=KEY, seed=SEED):
    """Run P8 once and measure normal, confidence, and oracle decoding per threshold.

    Compares standard hard decoding against:
      1. Confidence-based erasure decoding (flagging low-confidence bytes).
      2. Oracle erasure decoding (erasing true error locations as a theoretical ceiling).

    Args:
        image (np.ndarray): Preprocessed 512x512 BGR uint8 image.
        image_name (str): Identifier name of the image.
        attack_cases (list): Attack condition pairs to evaluate.
        key (int): Pseudo-random permutation key.
        seed (int): Deterministic distortion seed.

    Returns:
        list[dict]: Evaluation trial records spanning all erasure thresholds.
    """
    # Execute image recovery run to obtain extracted bits and per-bit soft-vote totals
    base_rows = run_experiment(
        image,
        image_name,
        attack_cases=attack_cases,
        key=key,
        seed=seed,
        include_bits=True,
    )
    expected_message = ecc.text_to_bits("TEST")
    rows = []
    for base in base_rows:
        coded_bits = np.asarray(base["coded_bits"], dtype=np.int64)
        extracted_bits = np.asarray(base["extracted_bits"], dtype=np.int64)
        vote_totals = base["vote_totals"]
        original_bytes = ecc.bits_to_bytes(coded_bits)
        extracted_bytes = ecc.bits_to_bytes(extracted_bits)

        # Identify ground-truth byte-level errors for oracle baseline calculation
        actual_errors = [
            index for index, (original, extracted) in enumerate(
                zip(original_bytes, extracted_bytes)
            )
            if original != extracted
        ]
        # Derive per-bit and per-byte confidence metrics from soft-voting totals
        bit_confidence = normalize_bit_confidence(vote_totals)
        byte_scores = byte_confidence(bit_confidence)

        # Baseline: Standard Reed-Solomon decoding without erasure hints
        normal_message, normal_ok = ecc.decode(extracted_bits)
        normal_success = _message_success(normal_message, normal_ok, expected_message)

        # Evaluate confidence-based erasure decoding across candidate thresholds
        for threshold in ERASURE_THRESHOLDS:
            predicted = erasure_indices(byte_scores, threshold)
            erased_message, erased_ok = decode_with_erasures(extracted_bits, predicted)
            oracle_message, oracle_ok = decode_with_erasures(extracted_bits, actual_errors)
            rows.append({
                "image": image_name,
                "attack": base["attack"],
                "severity": base["severity"],
                "threshold": threshold,
                "bit_accuracy": base["coded_bit_accuracy"],
                "actual_erroneous_rs_bytes": len(actual_errors),
                "predicted_erasure_bytes": len(predicted),
                "predicted_erasure_indices": ",".join(map(str, predicted)),
                "oracle_erasure_indices": ",".join(map(str, actual_errors)),
                "normal_ecc_success": normal_success,
                "erasure_ecc_success": _message_success(
                    erased_message, erased_ok, expected_message
                ),
                "oracle_erasure_success": _message_success(
                    oracle_message, oracle_ok, expected_message
                ),
                "byte_confidences": ",".join(f"{score:.8f}" for score in byte_scores),
            })
    return rows


def write_results(rows, output_csv=OUTPUT_CSV):
    """Serializes erasure recovery trial records to CSV format.

    Args:
        rows (list[dict]): Collected erasure decoding trial records.
        output_csv (Path or str): Output file destination.

    Returns:
        Path: Output path of the saved CSV.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


def main():
    """CLI execution entry point for Phase 10 confidence-based erasure evaluation."""
    # Retrieve clean test image cohort
    images = select_test_images()
    if len(images) != N_IMAGES:
        raise ValueError(f"P10 needs {N_IMAGES} clean test images (the P8/P9 set); found {len(images)}")

    # Execute erasure evaluation across all images and threshold values
    rows = []
    for name, image in images:
        rows.extend(analyze_image(image, name))
    output = write_results(rows)

    frame = pd.DataFrame(rows)
    # Deduplicate baseline trials to compute total non-erased success count
    normal_successes = int(
        frame.drop_duplicates(["image", "attack", "severity"])["normal_ecc_success"].sum()
    )
    # Deduplicate oracle trials to compute theoretical erasure recovery ceiling
    oracle_successes = int(
        frame.drop_duplicates(["image", "attack", "severity"])["oracle_erasure_success"].sum()
    )
    # Compute aggregate successes per confidence threshold
    threshold_summary = frame.groupby("threshold")["erasure_ecc_success"].sum()
    best_threshold = float(threshold_summary.idxmax())

    print(f"Images: {', '.join(name for name, _ in images)} | key={KEY} | seed={SEED}")
    print(f"Exact TEST recovery counts (out of {frame.drop_duplicates(['image', 'attack', 'severity']).shape[0]} image/attack cases):")
    print(f"  Normal ECC: {normal_successes}")
    print(f"  Oracle erasures: {oracle_successes}")
    print("  Confidence erasures by threshold:")
    for threshold, successes in threshold_summary.items():
        print(f"    {threshold:.2f}: {int(successes)}")
    print(f"Best confidence threshold: {best_threshold:.2f}")

    # Inspect outcomes on severe and moderate low-pass and cropping attacks
    focus = frame[
        frame["attack"].isin(("blur", "resize", "smooth", "crop"))
        & frame["severity"].isin(("moderate", "severe"))
        & (frame["threshold"] == best_threshold)
    ]
    summary = focus.groupby(["attack", "severity"], sort=False).agg(
        images=("image", "count"),
        mean_bit_accuracy=("bit_accuracy", "mean"),
        mean_actual_errors=("actual_erroneous_rs_bytes", "mean"),
        mean_predicted_erasures=("predicted_erasure_bytes", "mean"),
        normal_successes=("normal_ecc_success", "sum"),
        erasure_successes=("erasure_ecc_success", "sum"),
        oracle_successes=("oracle_erasure_success", "sum"),
    ).reset_index()
    print("\nSelected cases at best threshold:")
    print(summary.to_string(index=False, formatters={
        "mean_bit_accuracy": "{:.1%}".format,
        "mean_actual_errors": "{:.2f}".format,
        "mean_predicted_erasures": "{:.2f}".format,
    }))
    print(f"\nSaved {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
