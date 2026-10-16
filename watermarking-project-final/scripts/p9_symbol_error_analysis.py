r"""Analyze Reed-Solomon byte-symbol errors in real-image attack recoveries.

Run with: python -m scripts.p9_symbol_error_analysis
"""
import json
import re
from pathlib import Path

import pandas as pd

from watermark import dataset, ecc
from watermark.config import ECC_PARITY_BYTES, PAYLOAD_BITS
from scripts.p8_image_recovery import ATTACK_CASES, KEY, SEED, run_experiment

# Cohort size for byte-level symbol error profiling
N_IMAGES = 3
OUTPUT_CSV = Path("results") / "p9_symbol_error_analysis.csv"
# 64 codeword bits packed into 8 Reed-Solomon bytes (symbols)
RS_SYMBOLS = PAYLOAD_BITS // 8
# Reed-Solomon with 4 parity bytes can correct at most floor(4 / 2) = 2 erroneous byte symbols
CORRECTABLE_SYMBOL_ERRORS = ECC_PARITY_BYTES // 2


def symbol_error_profile(coded_bits, extracted_bits):
    """Compare payloads in the same MSB-first byte packing used by ecc.py.

    Analyzes how bit-level extraction errors cluster into byte-level Reed-Solomon symbols.
    Calculates bit errors per symbol, flags symbols with non-zero errors, and checks
    if the total corrupted byte count falls within the 2-symbol correction capacity.

    Args:
        coded_bits (array-like): Ground truth 64-bit embedded codeword.
        extracted_bits (array-like): Extracted 64-bit candidate codeword.

    Returns:
        dict: Detailed metrics including bit error counts, symbol error counts,
              error distribution across symbols, and capacity compliance flags.

    Raises:
        ValueError: If either payload does not unpack to exactly RS_SYMBOLS (8 bytes).
    """
    expected = ecc.bits_to_bytes(coded_bits)
    actual = ecc.bits_to_bytes(extracted_bits)
    if len(expected) != RS_SYMBOLS or len(actual) != RS_SYMBOLS:
        raise ValueError(f"expected {RS_SYMBOLS} Reed-Solomon symbols in each payload")

    # Count bit flips per byte symbol using bitwise XOR and popcount
    bit_errors_by_symbol = [
        (expected[index] ^ actual[index]).bit_count()
        for index in range(RS_SYMBOLS)
    ]
    # Isolate only corrupted symbols: {byte_index: bit_flip_count}
    erroneous = {
        index: count for index, count in enumerate(bit_errors_by_symbol) if count
    }
    # Histogram of bit errors per corrupted symbol
    distribution = {}
    for count in erroneous.values():
        distribution[count] = distribution.get(count, 0) + 1

    return {
        "total_bit_errors": sum(bit_errors_by_symbol),
        "erroneous_symbol_count": len(erroneous),
        "erroneous_symbol_indices": ",".join(map(str, erroneous)),
        "bit_errors_per_erroneous_symbol": ";".join(
            f"{index}:{count}" for index, count in erroneous.items()
        ),
        "max_bit_errors_in_symbol": max(bit_errors_by_symbol, default=0),
        "symbol_error_distribution": json.dumps(distribution, sort_keys=True),
        # Concentrated: errors fall into <= 2 symbols (correctable regardless of bit error count)
        # Spread: errors disperse into > 2 symbols (defeats hard algebraic RS decoding)
        "error_pattern": (
            "concentrated" if len(erroneous) <= CORRECTABLE_SYMBOL_ERRORS else "spread"
        ),
        "within_rs_symbol_capacity": len(erroneous) <= CORRECTABLE_SYMBOL_ERRORS,
        "original_rs_bytes_hex": expected.hex(),
        "extracted_rs_bytes_hex": actual.hex(),
    }


def select_test_images(max_images=N_IMAGES):
    """Select sorted, clean base images (exclude generated *_cN attack variants).

    Args:
        max_images (int): Maximum number of images to return. Defaults to N_IMAGES (3).

    Returns:
        list[tuple[str, np.ndarray]]: List of (image_name, image_array) tuples.

    Raises:
        ValueError: If the test split lacks clean base PNG images.
    """
    names = [
        name for name in dataset.split_names("test")
        if not re.search(r"_c\d+$", name)
    ]
    if not names:
        dataset.require("test")
        raise ValueError("test split contains no un-attacked base PNG images")
    return [(name, dataset.load("test", name)) for name in names[:max_images]]


def analyze_image(image, image_name, attack_cases=ATTACK_CASES, key=KEY, seed=SEED):
    """Reuse P8's real-image runner, adding byte-level error measurements.

    Args:
        image (np.ndarray): 512x512 BGR uint8 image.
        image_name (str): Identifier name of the image.
        attack_cases (list): Attack condition pairs.
        key (int): Permutation key.
        seed (int): Attack distortion seed.

    Returns:
        list[dict]: Collected trial records enriched with symbol error profiles.
    """
    # Execute full embedding and attack sweep, capturing raw bit arrays
    p8_rows = run_experiment(
        image, image_name, attack_cases=attack_cases, key=key, seed=seed,
        include_bits=True,
    )
    rows = []
    for p8_row in p8_rows:
        coded_bits = p8_row.pop("coded_bits")
        extracted_bits = p8_row.pop("extracted_bits")
        p8_row.pop("vote_totals", None)
        row = dict(p8_row)
        # Compute byte-level error distribution metrics and append to record
        row.update(symbol_error_profile(coded_bits, extracted_bits))
        rows.append(row)
    return rows


def summarize(rows):
    """Aggregate image-level results by attack and severity.

    Args:
        rows (list[dict]): Granular trial records.

    Returns:
        pd.DataFrame: Grouped summary showing mean accuracy, symbol error counts,
                      and decoding success totals per condition.
    """
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["attack", "severity"], sort=False)
        .agg(
            images=("image", "count"),
            mean_bit_accuracy=("coded_bit_accuracy", "mean"),
            mean_erroneous_symbols=("erroneous_symbol_count", "mean"),
            max_bit_errors_in_symbol=("max_bit_errors_in_symbol", "max"),
            within_symbol_capacity_cases=("within_rs_symbol_capacity", "sum"),
            ecc_successes=("ecc_decode_ok", "sum"),
            exact_message_successes=("exact_message_recovered", "sum"),
        )
        .reset_index()
    )


def write_results(rows, output_csv=OUTPUT_CSV):
    """Serializes symbol error records to CSV.

    Args:
        rows (list[dict]): Trial records.
        output_csv (Path or str): Output file path.

    Returns:
        Path: Output path of the written CSV.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


def main():
    """CLI execution entry point for Phase 9 symbol error analysis."""
    # Load first 3 un-attacked test images
    images = select_test_images()
    rows = []
    # Profile byte-level error distribution across images and attack conditions
    for name, image in images:
        rows.extend(analyze_image(image, name))

    output = write_results(rows)
    summary = summarize(rows)

    # Print summary table to console
    print(f"Images: {', '.join(name for name, _ in images)} | key={KEY} | attack seed={SEED}")
    print(
        summary.to_string(
            index=False,
            formatters={
                "mean_bit_accuracy": "{:.1%}".format,
                "mean_erroneous_symbols": "{:.2f}".format,
            },
        )
    )
    print(f"\nSaved {len(rows)} image/attack rows to {output}")


if __name__ == "__main__":
    main()
