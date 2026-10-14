"""Lightweight schema checks for the final Member C reporting package."""
import pandas as pd

from scripts.final_member_c_evaluation import (
    FINAL_COLUMNS,
    make_record,
    p2_ecc_recovery,
)


def test_final_record_schema_and_unavailable_metrics_remain_empty():
    record = make_record(
        "test source",
        image="sample",
        attack="none",
        method="current",
        bit_accuracy=1.0,
    )

    assert tuple(record) == FINAL_COLUMNS
    assert record["bit_accuracy"] == 1.0
    assert record["BER"] is None
    assert record["PSNR"] is None
    assert record["SSIM"] is None
    assert record["RS_byte_errors"] is None
    assert record["ECC_success"] is None
    assert record["exact_recovery"] is None


def test_p2_ecc_recovery_denominator_filters_to_ecc_configuration():
    frame = pd.DataFrame({
        "config": ["uniform", "uniform+ecc", "uniform+ecc"],
        "success": [True, True, False],
    })

    assert p2_ecc_recovery(frame) == (1, 2)
