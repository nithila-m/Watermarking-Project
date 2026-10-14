"""Member C's redundancy + ECC partial-recovery experiment checks."""
from scripts.p7_partial_recovery import run_experiment


def test_partial_recovery_experiment_is_deterministic_and_uses_all_levels():
    first = run_experiment(seed=2026)
    second = run_experiment(seed=2026)

    assert first == second
    assert [result["requested_percent"] for result in first] == [0, 10, 25, 40, 50, 60, 75]
    assert all(result["observations_per_bit"] == 16 for result in first)
    assert first[0]["coded_accuracy"] == 1.0
    assert first[0]["ecc_ok"] and first[0]["reconstructed"]
    assert all(
        result["reconstructed"]
        for result in first[1:4]
    )


def test_partial_recovery_redundancy_majority_limits():
    results = {result["requested_percent"]: result for result in run_experiment()}

    # Up to 40% requested corruption rounds to at most 6 of 16 copies per bit.
    assert all(results[level]["coded_accuracy"] == 1.0 for level in (10, 25, 40))
    # 60% rounds to 10 of 16 copies, so every majority is reversed.
    assert results[60]["coded_accuracy"] == 0.0
    # At 75%, 12 of 16 observations are corrupted, reversing every voted bit.
    assert results[75]["coded_accuracy"] == 0.0
    assert results[50]["coded_accuracy"] == 0.453125
    assert all(not results[level]["reconstructed"] for level in (50, 60, 75))
