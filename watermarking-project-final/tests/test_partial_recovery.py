"""Member C's redundancy + ECC partial-recovery experiment checks.

Validates the Phase 7 partial-payload recovery simulation:
  - Verifies determinism across identical random seeds.
  - Ensures all designated synthetic corruption levels (0% to 75%) are exercised.
  - Confirms uniform block repetition factor (16 block observations per coded bit).
  - Asserts exact message recovery under sub-majority corruption thresholds.
  - Validates majority-vote boundary limits and failure behaviors under severe corruption.
"""
from scripts.p7_partial_recovery import run_experiment


def test_partial_recovery_experiment_is_deterministic_and_uses_all_levels():
    """Verify determinism, sweep coverage, and recovery under low-to-moderate corruption.

    Confirms that:
      1. Repeated runs with seed=2026 produce bit-for-bit identical results.
      2. The full sweep of requested corruption tiers [0, 10, 25, 40, 50, 60, 75] is present.
      3. Each of the 64 coded bits has exactly 16 independent block observations (1024 / 64).
      4. Pristine conditions (0% corruption) achieve 100% coded accuracy and exact message recovery.
      5. Corruption levels up to 40% (levels 10%, 25%, 40%) maintain exact reconstruction.
    """
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
    """Verify repetition coding majority vote thresholds and breakdown behavior.

    Analyzes soft-voting extraction behavior under synthetic margin flips:
      - 10%, 25%, and 40% corruption flip at most 6 of 16 copies per bit;
        the majority (>= 10 of 16) remains intact, yielding 100% coded accuracy.
      - 50% corruption flips exactly 8 of 16 copies; tie-breaking rules and
        tanh soft weighting produce near-chance accuracy (~45.3%).
      - 60% and 75% corruption flip 10 and 12 of 16 copies respectively;
        the majority is strictly inverted for every bit, driving coded accuracy to 0%.
      - Corruption at or above 50% exceeds Reed-Solomon correction capacity, preventing recovery.
    """
    results = {result["requested_percent"]: result for result in run_experiment()}

    # Up to 40% requested corruption rounds to at most 6 of 16 copies per bit.
    assert all(results[level]["coded_accuracy"] == 1.0 for level in (10, 25, 40))
    # 60% rounds to 10 of 16 copies, so every majority is reversed.
    assert results[60]["coded_accuracy"] == 0.0
    # At 75%, 12 of 16 observations are corrupted, reversing every voted bit.
    assert results[75]["coded_accuracy"] == 0.0
    assert results[50]["coded_accuracy"] == 0.453125
    assert all(not results[level]["reconstructed"] for level in (50, 60, 75))
