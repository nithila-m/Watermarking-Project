"""Lightweight schema checks for the final Member C reporting package.

Validates the data extraction and reporting contracts used in Member C's
final consolidated evaluation pipeline:
  - Canonical record schema and column alignment across disparate phase outputs.
  - Correct initialization of optional/unrecorded fields as None.
  - Proper filtering and ratio aggregation for ECC configurations from Phase 2 logs.
"""
import pandas as pd

from scripts.final_member_c_evaluation import (
    FINAL_COLUMNS,
    make_record,
    p2_ecc_recovery,
)


def test_final_record_schema_and_unavailable_metrics_remain_empty():
    """Verify that make_record yields a dictionary matching FINAL_COLUMNS.

    Ensures that fields not populated during intermediate pipeline phases
    default cleanly to None without missing keys or altering column order.
    """
    record = make_record(
        "test source",
        image="sample",
        attack="none",
        method="current",
        bit_accuracy=1.0,
    )

    # Verify complete and ordered key match with canonical schema
    assert tuple(record) == FINAL_COLUMNS
    assert record["bit_accuracy"] == 1.0

    # Ensure unpopulated metrics are set to None
    assert record["BER"] is None
    assert record["PSNR"] is None
    assert record["SSIM"] is None
    assert record["RS_byte_errors"] is None
    assert record["ECC_success"] is None
    assert record["exact_recovery"] is None


def test_p2_ecc_recovery_denominator_filters_to_ecc_configuration():
    """Verify that p2_ecc_recovery isolates only ECC-enabled runs.

    Ensures non-ECC baseline rows (e.g., 'uniform') are excluded from
    both the numerator (successes) and denominator (total cases).
    """
    frame = pd.DataFrame({
        "config": ["uniform", "uniform+ecc", "uniform+ecc"],
        "success": [True, True, False],
    })

    # Out of 2 'uniform+ecc' entries, 1 succeeded -> (1, 2)
    assert p2_ecc_recovery(frame) == (1, 2)
