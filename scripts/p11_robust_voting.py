r"""Compare deterministic aggregation rules for redundant block observations.

Run with: python -m scripts.p11_robust_voting
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.p8_image_recovery import (
    ATTACK_CASES,
    KEY,
    SEED,
    run_experiment as run_p8_experiment,
)
from scripts.p9_symbol_error_analysis import select_test_images
from watermark import ecc
from watermark.config import PAYLOAD_BITS

METHODS = (
    "current_soft",
    "median_margin",
    "trimmed_mean",
    "sign_majority",
    "confidence_weighted_sign",
)
OUTPUT_CSV = Path("results") / "p11_robust_voting.csv"


def aggregate_votes(margins, assignment, n_bits=PAYLOAD_BITS):
    """Return coded bits from five rules applied to shared block observations.

    Sign-majority treats zero margins as abstentions and resolves ties to bit 0.
    Confidence-weighted sign uses sum(sign(margin) * abs(margin)); ties decode 0.
    The current soft baseline preserves extract()'s median-scale tanh formula and
    strict-positive bit decision exactly.
    """
    margins = np.asarray(margins, dtype=np.float64).ravel()
    assignment = np.asarray(assignment, dtype=np.int64).ravel()
    if len(margins) != len(assignment):
        raise ValueError("margins and assignment must have equal lengths")
    if np.any(assignment < 0) or np.any(assignment >= n_bits):
        raise ValueError("assignment contains a bit index outside the payload")
    groups = [margins[assignment == bit_index] for bit_index in range(n_bits)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("every coded bit must have at least one block observation")

    scale = np.median(np.abs(margins)) + 1e-9
    current_votes = np.tanh(margins / scale)
    current_totals = np.bincount(
        assignment, weights=current_votes, minlength=n_bits
    )
    median_values = np.array([np.median(group) for group in groups])
    trimmed_values = []
    sign_totals = []
    weighted_sign_totals = []
    for group in groups:
        if len(group) >= 3:
            ordered = np.sort(group)
            trimmed = ordered[1:-1]
        else:
            trimmed = group
        trimmed_values.append(float(np.mean(trimmed)))
        signs = np.sign(group)
        sign_totals.append(float(np.sum(signs)))
        weighted_sign_totals.append(float(np.sum(signs * np.abs(group))))

    return {
        "current_soft": (current_totals > 0).astype(np.int64),
        "median_margin": (median_values > 0).astype(np.int64),
        "trimmed_mean": (np.asarray(trimmed_values) > 0).astype(np.int64),
        "sign_majority": (np.asarray(sign_totals) > 0).astype(np.int64),
        "confidence_weighted_sign": (
            np.asarray(weighted_sign_totals) > 0
        ).astype(np.int64),
    }


def evaluate_observation(image_name, attack, severity, original_bits, margins, assignment):
    """Score every vote aggregation using the same original block margins."""
    original_bits = np.asarray(original_bits, dtype=np.int64)
    methods = aggregate_votes(margins, assignment, len(original_bits))
    rows = []
    original_bytes = ecc.bits_to_bytes(original_bits)
    for method in METHODS:
        bits = methods[method]
        decoded, ecc_ok = ecc.decode(bits)
        recovered_text = ecc.bits_to_text(decoded)
        recovered_bytes = ecc.bits_to_bytes(bits)
        rows.append({
            "image": image_name,
            "attack": attack,
            "severity": severity,
            "voting_method": method,
            "coded_bit_accuracy": float(np.mean(bits == original_bits)),
            "coded_bit_errors": int(np.count_nonzero(bits != original_bits)),
            "rs_byte_errors": sum(
                actual != expected
                for actual, expected in zip(recovered_bytes, original_bytes)
            ),
            "ecc_success": bool(ecc_ok),
            "exact_text_recovery": bool(ecc_ok and recovered_text == "TEST"),
        })
    return rows


def run_experiment(images, attack_cases=ATTACK_CASES, key=KEY, seed=SEED):
    """Run P8 once per image; compare methods on each case's shared margins."""
    rows = []
    for image_name, image in images:
        cases = run_experiment_for_image(image, image_name, attack_cases, key, seed)
        for case in cases:
            rows.extend(evaluate_observation(
                image_name,
                case["attack"],
                case["severity"],
                case["coded_bits"],
                case["block_margins"],
                case["block_assignment"],
            ))
    return rows


def run_experiment_for_image(image, image_name, attack_cases, key, seed):
    """Obtain one P8 embed/attack run and retain observations for all methods."""
    return run_p8_experiment(
        image, image_name, attack_cases=attack_cases, key=key, seed=seed,
        include_bits=True, include_observations=True,
    )


def write_results(rows, output_csv=OUTPUT_CSV):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


def summarize(rows):
    frame = pd.DataFrame(rows)
    return frame.groupby("voting_method", sort=False).agg(
        cases=("image", "count"),
        mean_bit_accuracy=("coded_bit_accuracy", "mean"),
        coded_bit_errors=("coded_bit_errors", "sum"),
        ecc_successes=("ecc_success", "sum"),
        exact_text_recoveries=("exact_text_recovery", "sum"),
    ).reset_index()


def main():
    images = select_test_images()
    expected_images = ["grass", "gravel", "hubble_deep_field"]
    if [name for name, _ in images] != expected_images:
        raise ValueError(f"P11 requires the clean test images {expected_images}")

    rows = run_experiment(images)
    output = write_results(rows)
    summary = summarize(rows)
    print(f"Images: {', '.join(expected_images)} | key={KEY} | seed={SEED}")
    print(summary.to_string(index=False, formatters={
        "mean_bit_accuracy": "{:.2%}".format,
    }))

    focus = pd.DataFrame(rows)
    focus = focus[
        focus["attack"].isin(("blur", "resize", "smooth", "crop"))
        & focus["severity"].isin(("moderate", "severe"))
    ]
    focus_summary = focus.groupby(
        ["attack", "severity", "voting_method"], sort=False
    ).agg(
        mean_accuracy=("coded_bit_accuracy", "mean"),
        ecc_success=("ecc_success", "sum"),
        exact_text=("exact_text_recovery", "sum"),
    ).reset_index()
    print("\nSelected robustness cases:")
    print(focus_summary.to_string(index=False, formatters={
        "mean_accuracy": "{:.2%}".format,
    }))
    print(f"\nSaved {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
