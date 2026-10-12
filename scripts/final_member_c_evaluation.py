r"""Consolidate existing Member C experiments into report-ready artifacts.

Run with: python -m scripts.final_member_c_evaluation
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.demo import run_demo
from scripts.p7_partial_recovery import run_experiment as run_p7_experiment
from scripts.p9_symbol_error_analysis import select_test_images
from watermark.config import (
    BLOCK,
    ECC_PARITY_BYTES,
    IMG_SIZE,
    PAYLOAD_BITS,
)

RESULTS = Path("results")
FINAL_CSV = RESULTS / "final_member_c_results.csv"
SUMMARY_CSV = RESULTS / "final_member_c_summary.csv"
TABLES_MD = RESULTS / "final_member_c_tables.md"
FINDINGS_MD = RESULTS / "final_member_c_findings.md"
REQUIRED_IMAGES = ("grass", "gravel", "hubble_deep_field")
P10_THRESHOLDS = (0.0, 0.25, 0.50, 0.75, 1.0)

FINAL_COLUMNS = (
    "experiment", "image", "attack", "severity", "method/configuration",
    "bit_accuracy", "BER", "PSNR", "SSIM", "RS_byte_errors",
    "ECC_success", "exact_recovery",
)


def make_record(experiment, image="", attack="", severity="", method="",
                bit_accuracy=None, ber=None, psnr=None, ssim=None,
                rs_byte_errors=None, ecc_success=None, exact_recovery=None):
    """Create a consistent row, retaining unavailable measurements as blanks."""
    return {
        "experiment": experiment,
        "image": image,
        "attack": attack,
        "severity": severity,
        "method/configuration": method,
        "bit_accuracy": bit_accuracy,
        "BER": ber,
        "PSNR": psnr,
        "SSIM": ssim,
        "RS_byte_errors": rs_byte_errors,
        "ECC_success": ecc_success,
        "exact_recovery": exact_recovery,
    }


def _read(name):
    path = RESULTS / name
    if not path.is_file():
        raise FileNotFoundError(f"Required existing evaluation result is missing: {path}")
    return pd.read_csv(path)


def _boolean(value):
    if pd.isna(value):
        return None
    return bool(value)


def _p2_rows(frame):
    rows = []
    for source in frame.to_dict("records"):
        has_ecc = source["config"] == "uniform+ecc"
        rows.append(make_record(
            "P2 baseline/end-to-end",
            image=source["image"],
            attack=source["attack"],
            severity=source["severity"],
            method=source["config"],
            bit_accuracy=source["bit_acc"],
            ber=source["ber"],
            psnr=source["psnr"],
            ssim=source["ssim"],
            ecc_success=_boolean(source["success"]) if has_ecc else None,
            exact_recovery=_boolean(source["success"]),
        ))
    return rows


def _p7_rows(source_rows=None):
    rows = []
    source_rows = source_rows if source_rows is not None else run_p7_experiment(seed=2026)
    for source in source_rows:
        rows.append(make_record(
            "P7 partial recovery",
            image="synthetic repeated observations",
            attack="controlled observation corruption",
            severity=f"{source['requested_percent']}% requested ({source['actual_percent']:.1f}% actual)",
            method="current soft vote + Reed-Solomon",
            bit_accuracy=source["coded_accuracy"],
            ber=1.0 - source["coded_accuracy"],
            ecc_success=source["ecc_ok"],
            exact_recovery=source["reconstructed"],
        ))
    return rows


def _p8_p9_rows(p8, p9):
    symbol_counts = {
        (row["image"], row["attack"], row["severity"]): row["erroneous_symbol_count"]
        for row in p9.to_dict("records")
    }
    rows = []
    for source in p8.to_dict("records"):
        rows.append(make_record(
            "P8 real-image pipeline",
            image=source["image"],
            attack=source["attack"],
            severity=source["severity"],
            method="uniform embed + production soft extract + RS",
            bit_accuracy=source["coded_bit_accuracy"],
            ber=source["ber"],
            psnr=source["embed_psnr"],
            rs_byte_errors=symbol_counts.get(
                (source["image"], source["attack"], source["severity"])
            ),
            ecc_success=_boolean(source["ecc_decode_ok"]),
            exact_recovery=_boolean(source["exact_message_recovered"]),
        ))
    for source in p9.to_dict("records"):
        rows.append(make_record(
            "P9 RS symbol analysis",
            image=source["image"],
            attack=source["attack"],
            severity=source["severity"],
            method="production extraction; byte error analysis",
            bit_accuracy=source["coded_bit_accuracy"],
            ber=source["ber"],
            psnr=source["embed_psnr"],
            rs_byte_errors=source["erroneous_symbol_count"],
            ecc_success=_boolean(source["ecc_decode_ok"]),
            exact_recovery=_boolean(source["exact_message_recovered"]),
        ))
    return rows


def _p10_rows(frame):
    rows = []
    for source in frame.to_dict("records"):
        common = {
            "image": source["image"],
            "attack": source["attack"],
            "severity": source["severity"],
            "bit_accuracy": source["bit_accuracy"],
            "ber": 1.0 - source["bit_accuracy"],
            "rs_byte_errors": source["actual_erroneous_rs_bytes"],
        }
        rows.append(make_record(
            "P10 confidence erasure analysis",
            method=f"confidence erasure threshold={source['threshold']:.2f}",
            ecc_success=_boolean(source["erasure_ecc_success"]),
            exact_recovery=_boolean(source["erasure_ecc_success"]),
            **common,
        ))
        if source["threshold"] == P10_THRESHOLDS[0]:
            rows.append(make_record(
                "P10 confidence erasure analysis",
                method="normal ECC baseline",
                ecc_success=_boolean(source["normal_ecc_success"]),
                exact_recovery=_boolean(source["normal_ecc_success"]),
                **common,
            ))
            rows.append(make_record(
                "P10 confidence erasure analysis",
                method="oracle erasure (analysis-only)",
                ecc_success=_boolean(source["oracle_erasure_success"]),
                exact_recovery=_boolean(source["oracle_erasure_success"]),
                **common,
            ))
    return rows


def _p11_rows(frame):
    return [
        make_record(
            "P11 robust voting",
            image=source["image"],
            attack=source["attack"],
            severity=source["severity"],
            method=source["voting_method"],
            bit_accuracy=source["coded_bit_accuracy"],
            ber=1.0 - source["coded_bit_accuracy"],
            rs_byte_errors=source["rs_byte_errors"],
            ecc_success=_boolean(source["ecc_success"]),
            exact_recovery=_boolean(source["exact_text_recovery"]),
        )
        for source in frame.to_dict("records")
    ]


def _demo_rows():
    images = dict(select_test_images())
    missing = [name for name in REQUIRED_IMAGES if name not in images]
    if missing:
        raise ValueError(f"Missing required P8/P9 test images: {missing}")
    rows = []
    for attack, severity in (("none", "none"), ("blur", "moderate")):
        result = run_demo(
            images["grass"], "TEST", attack, severity, key=2024, seed=2026
        )
        rows.append({"image": "grass", **result})
    return rows


def _success_rate(series):
    return f"{int(series.sum())}/{len(series)}"


def p2_ecc_recovery(frame):
    """Return exact uniform+ECC recoveries and the number of evaluated rows."""
    ecc_rows = frame[frame.config == "uniform+ecc"]
    return int(ecc_rows.success.sum()), len(ecc_rows)


def _table(headers, rows):
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(output)


def build_tables(p2, p8, p9, p10, p11, demo):
    clean = p2[(p2.config == "uniform+ecc") & (p2.attack == "none")]
    clean_row = [
        "P2 uniform+ECC (50 images)",
        f"{clean.psnr.mean():.2f} dB",
        f"{clean.ssim.mean():.4f}",
        f"{clean.bit_acc.mean():.2%}",
        f"{clean.ber.mean():.4f}",
        _success_rate(clean.success),
    ]
    dclean = demo[demo.attack == "none"].iloc[0]
    clean_row.append(
        f"P12 grass demo: {dclean.bit_accuracy:.2%}, exact={str(bool(dclean.exact_recovery)).upper()}"
    )

    attacks = p2[
        (p2.config == "uniform+ecc")
        & p2.attack.isin(("jpeg", "blur", "resize", "crop", "smooth"))
    ]
    attack_rows = []
    for (kind, severity), group in attacks.groupby(["attack", "severity"], sort=False):
        attack_rows.append((
            kind, severity, len(group), f"{group.bit_acc.mean():.2%}",
            f"{group.ber.mean():.4f}", _success_rate(group.success),
        ))

    symbol_rows = []
    focus_p9 = p9[
        p9.attack.isin(("blur", "resize", "smooth", "crop"))
        & p9.severity.isin(("moderate", "severe"))
    ]
    for (kind, severity), group in focus_p9.groupby(["attack", "severity"], sort=False):
        within = int(group.within_rs_symbol_capacity.sum())
        symbol_rows.append((
            kind, severity, f"{group.coded_bit_accuracy.mean():.2%}",
            f"{group.erroneous_symbol_count.mean():.2f}",
            f"{group.max_bit_errors_in_symbol.max():.0f}",
            f"{_success_rate(group.ecc_decode_ok)}",
            f"{within}/{len(group)} within ≤2-symbol correction bound",
        ))

    p10_unique = p10.drop_duplicates(["image", "attack", "severity", "threshold"])
    threshold_rows = []
    for threshold, group in p10_unique.groupby("threshold", sort=True):
        threshold_rows.append((
            f"{threshold:.2f}",
            _success_rate(group.erasure_ecc_success),
            f"{group.predicted_erasure_bytes.mean():.2f}",
        ))
    base_p10 = p10[p10.threshold == min(P10_THRESHOLDS)].drop_duplicates(
        ["image", "attack", "severity"]
    )
    normal_count = int(base_p10.normal_ecc_success.sum())
    oracle_count = int(base_p10.oracle_erasure_success.sum())

    voting_rows = []
    for method, group in p11.groupby("voting_method", sort=False):
        voting_rows.append((
            method, f"{group.coded_bit_accuracy.mean():.2%}",
            int(group.coded_bit_errors.sum()),
            _success_rate(group.exact_text_recovery),
        ))

    demo_rows = []
    for source in demo.to_dict("records"):
        demo_rows.append((
            source["attack"], source["severity"], source["recovered_text"],
            "YES" if source["ecc_success"] else "NO",
            "YES" if source["exact_recovery"] else "NO",
            f"{source['bit_accuracy']:.2%}", f"{source['ber']:.4f}",
            f"{source['psnr']:.2f} dB", f"{source['ssim']:.4f}",
        ))

    return "\n\n".join((
        "# Final Member C Evaluation Tables",
        "## TABLE 1 — Final clean-image performance",
        _table(("Source / cohort", "PSNR", "SSIM", "Bit accuracy", "BER", "ECC/exact recovery", "Additional demo result"), [clean_row]),
        "## TABLE 2 — Attack robustness (P2 uniform+ECC, 50 images)",
        _table(("Attack", "Severity", "Images", "Mean bit accuracy", "Mean BER", "Exact recovery"), attack_rows),
        "## TABLE 3 — Reed-Solomon symbol-error behavior (P9, three images)",
        _table(("Attack", "Severity", "Mean bit accuracy", "Mean erroneous RS bytes", "Max bit errors/byte", "ECC success", "Cases ≤2 erroneous bytes"), symbol_rows),
        "P9 observed that all 57 cases with at most two erroneous RS bytes decoded, and all cases with more than two failed.",
        "## TABLE 4 — Confidence-based erasure experiment (P10, 57 cases)",
        _table(("Confidence threshold", "Exact recovery", "Mean predicted erasures"), threshold_rows),
        f"Normal ECC: {normal_count}/57. Oracle erasure: {oracle_count}/57 (analysis-only; not production).",
        "## TABLE 5 — Voting-method comparison (P11, 57 cases)",
        _table(("Voting method", "Mean bit accuracy", "Coded-bit errors", "Exact recovery"), voting_rows),
        "Sign majority reached 49/57 versus 45/57 for the current production soft vote; this was not sufficiently consistent to replace production extraction.",
        "## TABLE 6 — Final demo verification (P12, grass image)",
        _table(("Attack", "Severity", "Recovered text", "ECC", "Exact", "Bit accuracy", "BER", "PSNR", "SSIM"), demo_rows),
    ))


def build_summary(p2, p7, p8, p9, p10, p11, demo):
    rows = []
    clean = p2[(p2.config == "uniform+ecc") & (p2.attack == "none")]
    p2_recovered, p2_cases = p2_ecc_recovery(p2)
    rows.extend([
        {"section": "A. Baseline/end-to-end", "finding": "P2 clean uniform+ECC, 50 images",
         "metric": "PSNR / SSIM / bit accuracy / exact recovery",
         "value": f"{clean.psnr.mean():.2f} dB / {clean.ssim.mean():.4f} / {clean.bit_acc.mean():.2%} / {int(clean.success.sum())}/{len(clean)}",
         "notes": "Existing P2 CSV; values are cohort means and exact recovery count."},
        {"section": "A. Baseline/end-to-end", "finding": "P2 attack outcomes",
         "metric": "uniform+ECC exact recovery",
         "value": f"{p2_recovered}/{p2_cases} ({p2_recovered / p2_cases:.2%})",
         "notes": "All uniform+ECC rows across attacks and severities in the existing P2 baseline CSV."},
    ])
    p2_ecc = p2[p2.config == "uniform+ecc"]
    p2_smooth_severe = p2_ecc[
        (p2_ecc.attack == "smooth") & (p2_ecc.severity == "severe")
    ]
    p2_jpeg = p2_ecc[p2_ecc.attack == "jpeg"]
    rows.extend([
        {"section": "A. Baseline/end-to-end", "finding": "P2 strongest listed attack group: JPEG",
         "metric": "exact recovery across all JPEG severities",
         "value": f"{int(p2_jpeg.success.sum())}/{len(p2_jpeg)}",
         "notes": "50-image P2 uniform+ECC results across light, moderate, and severe JPEG."},
        {"section": "A. Baseline/end-to-end", "finding": "P2 weakest listed attack: severe smooth",
         "metric": "mean coded-bit accuracy / exact recovery",
         "value": f"{p2_smooth_severe.bit_acc.mean():.2%} / {int(p2_smooth_severe.success.sum())}/{len(p2_smooth_severe)}",
         "notes": "50-image P2 uniform+ECC result; severe smoothing was the major weakness."},
    ])
    for source in p7.to_dict("records"):
        rows.append({
            "section": "B. Partial recovery",
            "finding": f"Controlled repeated-observation corruption {source['requested_percent']}% (actual {source['actual_percent']:.1f}%)",
            "metric": "bit accuracy / exact reconstruction",
            "value": f"{source['coded_accuracy']:.2%} / {'YES' if source['reconstructed'] else 'NO'}",
            "notes": "Deterministic P7 synthetic-margin test; not an image attack.",
        })

    within = p9[p9.within_rs_symbol_capacity]
    beyond = p9[~p9.within_rs_symbol_capacity]
    rows.append({
        "section": "C. RS symbol error analysis",
        "finding": "Observed decoder behavior by erroneous byte count",
        "metric": "ECC successes within ≤2 bytes / >2 bytes",
        "value": f"{int(within.ecc_decode_ok.sum())}/{len(within)} / {int(beyond.ecc_decode_ok.sum())}/{len(beyond)}",
        "notes": "P9 measured all ≤2-byte cases successful and all >2-byte cases failed in its 57-case set.",
    })
    rows.append({
        "section": "C. RS symbol error analysis",
        "finding": "Code dimensions",
        "metric": "coded bits / RS symbols / parity bytes / observations per bit",
        "value": f"{PAYLOAD_BITS} / {PAYLOAD_BITS // 8} / {ECC_PARITY_BYTES} / {((IMG_SIZE // (2 * BLOCK)) ** 2) // PAYLOAD_BITS}",
        "notes": "Derived from existing configuration and block-grid dimensions.",
    })

    p10_unique = p10.drop_duplicates(["image", "attack", "severity", "threshold"])
    threshold_counts = p10_unique.groupby("threshold").erasure_ecc_success.sum()
    best = int(threshold_counts.max())
    best_thresholds = [float(t) for t, count in threshold_counts.items() if count == best]
    p10_baseline = p10[p10.threshold == min(P10_THRESHOLDS)].drop_duplicates(
        ["image", "attack", "severity"]
    )
    rows.extend([
        {"section": "D. Confidence erasure analysis", "finding": "Normal ECC",
         "metric": "exact recoveries", "value": f"{int(p10_baseline.normal_ecc_success.sum())}/57",
         "notes": "Repeated across threshold rows in source; deduplicated by image/attack."},
        {"section": "D. Confidence erasure analysis", "finding": "Best tested confidence threshold",
         "metric": "threshold / exact recoveries", "value": f"{', '.join(f'{x:.2f}' for x in best_thresholds)} / {best}/57",
         "notes": "Threshold 0 predicts no erasures and equals normal decoding; no tested threshold improved recovery."},
        {"section": "D. Confidence erasure analysis", "finding": "Oracle erasure upper bound",
         "metric": "exact recoveries", "value": f"{int(p10_baseline.oracle_erasure_success.sum())}/57",
         "notes": "Analysis-only, using known erroneous RS byte positions; not part of production."},
    ])

    vote_summary = p11.groupby("voting_method", sort=False).agg(
        accuracy=("coded_bit_accuracy", "mean"),
        exact=("exact_text_recovery", "sum"),
        total=("exact_text_recovery", "count"),
    )
    baseline_exact = int(vote_summary.loc["current_soft", "exact"])
    majority_exact = int(vote_summary.loc["sign_majority", "exact"])
    rows.append({
        "section": "E. Robust voting analysis",
        "finding": "Sign-majority versus current soft vote",
        "metric": "exact recoveries / mean bit accuracy",
        "value": f"{majority_exact}/57 ({vote_summary.loc['sign_majority', 'accuracy']:.2%}) vs {baseline_exact}/57 ({vote_summary.loc['current_soft', 'accuracy']:.2%})",
        "notes": "Modest improvement; not consistent across all images/attacks and does not solve severe smoothing.",
    })
    for method, summary in vote_summary.iterrows():
        rows.append({
            "section": "E. Robust voting analysis",
            "finding": method,
            "metric": "exact recovery / mean bit accuracy",
            "value": f"{int(summary['exact'])}/{int(summary['total'])} / {summary['accuracy']:.2%}",
            "notes": "P11 existing 57-case voting comparison.",
        })

    for source in demo.to_dict("records"):
        rows.append({
            "section": "F. Final demo verification",
            "finding": f"grass / {source['attack']} / {source['severity']}",
            "metric": "recovered text / ECC / exact / bit accuracy / BER / PSNR / SSIM",
            "value": f"{source['recovered_text']} / {'YES' if source['ecc_success'] else 'NO'} / {'YES' if source['exact_recovery'] else 'NO'} / {source['bit_accuracy']:.2%} / {source['ber']:.4f} / {source['psnr']:.2f} / {source['ssim']:.4f}",
            "notes": "Measured with P12 run_demo using production methods and fixed key/seed.",
        })
    return pd.DataFrame(rows)


def build_findings(summary, p9, p10, p11):
    summary_text = "\n".join(
        f"- **{row.finding}:** {row.value}. {row.notes}"
        for row in summary.itertuples(index=False)
        if row.section in ("A. Baseline/end-to-end", "B. Partial recovery")
    )
    within = p9[p9.within_rs_symbol_capacity]
    beyond = p9[~p9.within_rs_symbol_capacity]
    baseline_p10 = p10[p10.threshold == min(P10_THRESHOLDS)].drop_duplicates(
        ["image", "attack", "severity"]
    )
    vote = p11.groupby("voting_method").exact_text_recovery.sum()
    return f"""# Final Member C Evaluation Findings

## 1. System Implementation

The evaluated production path maps a short text message to bits, applies the existing Reed-Solomon encoder, embeds the 64-bit codeword using the DWT-DCT block watermark, applies an optional attack, extracts using the existing soft-voting extractor, and decodes the result with the existing Reed-Solomon implementation. The 512x512 image setup uses 1024 blocks and 64 coded bits.

## 2. Redundancy and Reed-Solomon ECC

The payload is redundantly distributed across image blocks. For the 512x512 setup, the current configuration provides approximately 16 observations per coded bit. Reed-Solomon uses four parity bytes over an eight-byte codeword (64 coded bits), with a two-symbol error-correction bound.

## 3. Extraction and Reconstruction

Production extraction aggregates block DCT margins per coded bit using the current median-scaled tanh soft vote. Reed-Solomon operates on byte symbols rather than individual bits. Therefore, a small number of bit errors can still cause ECC failure when those errors are distributed across more RS byte symbols than the decoder can correct.

## 4. Experimental Methodology

P2 uses its existing 50-image baseline CSV. P8 contains its existing grass-image run; P9, P10, and P11 use the fixed-key setup on grass, gravel, and hubble_deep_field with the existing attacks and fixed attack seed. P7 is a deterministic synthetic repeated-observation test, not an image attack. P12 verification rows are generated by two fixed-seed production-pipeline demo runs on grass. No P2 or P8-P11 attack evaluations were rerun to produce this package.

Unavailable values are left empty rather than inferred: the P7 synthetic result does not report RS byte errors or image-quality metrics; the P8/P9/P10/P11 source files do not contain SSIM; and P12's demo result does not expose raw codeword bytes for symbol counting.

## 5. Results

{summary_text}

P2 uniform+ECC clean-image and severity-wise attack values are in `final_member_c_tables.md` and the consolidated CSV. They are measured outcomes, not guarantees for other image distributions.

## 6. Failure Analysis

In P9's evaluated 57 image/attack cases, **{len(within)}/{len(within)}** cases with at most two erroneous RS bytes decoded successfully, while **{int(beyond.ecc_decode_ok.sum())}/{len(beyond)}** cases with more than two erroneous bytes decoded successfully. Thus, in this sample the observed split was exactly successful for ≤2 erroneous bytes and failed for >2. High bit accuracy alone was not sufficient when errors spread across too many symbols. Severe smoothing was the strongest failure condition in P8-P11.

## 7. Alternative Recovery Experiments

P10 tested the existing confidence totals as erasure indicators. The tested confidence thresholds did not improve recovery: the best confidence result matched normal ECC at 45/57. Oracle erasures reached 53/57, but rely on knowing the true erroneous byte positions and are analysis-only.

P11 compared the current soft vote, median margin, trimmed mean, sign majority, and confidence-weighted sign on identical margins. Sign majority reached 49/57 versus current soft voting at 45/57. The gain was modest and not sufficiently consistent across images and attacks; severe smoothing remained unresolved.

## 8. Final Design Decision

The current production extraction and ECC configuration are retained. No experimental recovery or voting method was promoted to production: confidence erasures did not improve results, and sign majority's 49/57 result was not consistent enough to replace current extraction.

## 9. Limitations

- P8 covers one clean test image; P9-P11 cover three clean test images under one fixed key and attack seed.
- Severe smoothing remains a major weakness, including cases with extensive coded-bit and byte errors.
- Recovery can fail when bit errors are distributed across more than two RS symbols.
- The tested confidence thresholds did not reliably localize erroneous bytes.
- Voting alternatives did not provide a sufficiently consistent improvement to justify replacing production extraction.
- The controlled P7 corruption experiment and single-run demo do not establish general attack robustness.

## 10. Conclusion

The final package consolidates the measured Member C evidence while retaining the production pipeline unchanged. The experiments identify symbol-error distribution and severe smoothing as important limits; neither the confidence-erasure study nor the alternative-voting comparison met the evidence threshold for a production change.
"""


def build_package():
    RESULTS.mkdir(exist_ok=True)
    p2 = _read("p2_baseline.csv")
    p8 = _read("p8_image_recovery.csv")
    p9 = _read("p9_symbol_error_analysis.csv")
    p10 = _read("p10_erasure_recovery.csv")
    p11 = _read("p11_robust_voting.csv")

    if [name for name, _ in select_test_images()] != list(REQUIRED_IMAGES):
        raise ValueError(f"Expected clean images in this order: {REQUIRED_IMAGES}")
    if len(p8) != 19 or len(p9) != 57 or len(p10) != 285 or len(p11) != 285:
        raise ValueError("P8-P11 result CSV row counts differ from the completed experiment set.")

    p7_results = run_p7_experiment(seed=2026)
    p7_result = pd.DataFrame(p7_results)
    demo_results = pd.DataFrame(_demo_rows())

    rows = (
        _p2_rows(p2)
        + _p7_rows(p7_results)
        + _p8_p9_rows(p8, p9)
        + _p10_rows(p10)
        + _p11_rows(p11)
        + [
            make_record(
                "P12 final demo verification",
                image=source["image"],
                attack=source["attack"],
                severity=source["severity"],
                method="production CLI pipeline, TEST",
                bit_accuracy=source["bit_accuracy"],
                ber=source["ber"],
                psnr=source["psnr"],
                ssim=source["ssim"],
                ecc_success=source["ecc_success"],
                exact_recovery=source["exact_recovery"],
            )
            for source in demo_results.to_dict("records")
        ]
    )
    final = pd.DataFrame(rows, columns=FINAL_COLUMNS)
    summary = build_summary(p2, p7_result, p8, p9, p10, p11, demo_results)
    final.to_csv(FINAL_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    TABLES_MD.write_text(
        build_tables(p2, p8, p9, p10, p11, demo_results) + "\n",
        encoding="utf-8",
    )
    FINDINGS_MD.write_text(
        build_findings(summary, p9, p10, p11),
        encoding="utf-8",
    )
    return final, summary


def main():
    final, summary = build_package()
    print(f"Final detailed rows: {len(final)} -> {FINAL_CSV}")
    print(f"Summary findings: {len(summary)} -> {SUMMARY_CSV}")
    print(f"Report tables: {TABLES_MD}")
    print(f"Findings report: {FINDINGS_MD}")


if __name__ == "__main__":
    main()
