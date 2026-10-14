# Member C Final Report

## 1. Member C Contribution Overview

Member C's contribution focused on the Reed-Solomon error-correction layer, redundant payload distribution, extraction and reconstruction, integration with the existing watermark pipeline, testing, experiments, and the final user-facing demonstration. The work evaluated the current production implementation and documented where it succeeds and fails.

The project also contains adaptive/AI components developed by other team members. This report does not attribute those components to Member C and does not claim that Member C developed the adaptive model.

## 2. System Architecture

The Member C evaluation follows this data path:

```text
Input image
    ↓
Watermark text → 32 message bits
    ↓
Reed-Solomon encoding → 64 coded bits
    ↓
Keyed redundant block assignment + DWT-DCT embedding
    ↓
Watermarked image
    ↓
Optional image attack
    ↓
Blind DWT-DCT margin extraction
    ↓
Redundant soft voting → recovered coded bits
    ↓
Reed-Solomon decoding
    ↓
Recovered watermark text
    ↓
Bit accuracy, BER, PSNR, SSIM, and recovery status
```

Encoding adds byte-level redundancy before embedding. Keyed assignment distributes each coded bit among many image blocks. The extractor needs the attacked image and the embedding key, not the original host image. Image-quality metrics compare the original and watermarked images; payload metrics compare the extracted and encoded coded bits.

## 3. End-to-End Watermarking Pipeline

The final demo uses the existing production components: `text_to_bits` and `encode` from the Reed-Solomon utilities, `uniform_map` for the demo strength map, `embed_at_psnr` for embedding, the existing `attack` function when selected, `extract` for blind recovery, and `decode` for ECC reconstruction. Existing PSNR, SSIM, and BER helpers are reused.

The UI and CLI accept an image, watermark text, attack, and severity. The demo standardizes input images to 512×512, embeds the encoded payload, optionally attacks the watermarked image, extracts coded bits, decodes them, and reports recovered text and metrics. The evaluated demo uses the production soft voter; it does not use P10 erasure decoding or P11 experimental voters.

## 4. Redundancy and Reed-Solomon Error Correction

The configured message is 32 bits (four bytes). The Reed-Solomon code appends four parity bytes, producing an eight-byte, 64-bit codeword. Under the current code parameters, the decoder can correct up to two erroneous byte symbols when those positions are treated as errors.

For a 512×512 image, one-level Haar DWT produces a 256×256 LL band. Dividing it into 8×8 DCT blocks yields 1,024 blocks. The keyed assignment maps these blocks across the 64 coded-bit positions, giving approximately 16 block observations per coded bit. The assignment is pseudo-randomly scattered using the shared key so spatially local changes, such as a crop, need not remove every copy of any one coded bit.

Reed-Solomon operates on bytes, not independently on each bit. A single wrong bit can make its containing byte erroneous. Consequently, even high bit accuracy can still lead to more than two erroneous byte symbols if the few bit errors occur in many different bytes.

## 5. Watermark Extraction and Reconstruction

The production extractor computes a signed DCT coefficient-pair margin for every block. Its sign is the block's bit vote. In soft mode, it scales margins by the median absolute margin, applies `tanh`, and sums the weighted votes for each assigned coded bit. A positive total yields one; otherwise the bit is zero.

The resulting 64 bits are passed to the existing Reed-Solomon decoder. On successful decoding, the recovered 32 message bits are converted back to text. Exact recovery requires both successful ECC decoding and equality with the original message bits. The extraction is blind because it does not require the original image during recovery.

## 6. Experimental Methodology

The final evaluation package consolidates stored results rather than rerunning expensive attack sweeps:

- **P2:** existing 50-image baseline cohort; the results below refer to its uniform+ECC configuration.
- **P7:** deterministic synthetic corruption of repeated observations; this is not an image attack.
- **P8:** one real image (grass), with 19 attack/severity cases.
- **P9, P10, P11:** grass, gravel, and hubble_deep_field under the same fixed-key evaluation setup; 57 attack cases per baseline method, with experiment-specific rows for P10 and P11.
- **P12:** fixed-seed demo verification on grass, with one clean run and one moderate-blur run.

The cohorts have different image counts and experiment-specific conditions. Their results should not be combined as though they were one identical evaluation set. P8-P11 source files do not provide SSIM, and some experiment records do not contain image-quality or symbol-error metrics. Missing values remain unreported rather than inferred.

## 7. Baseline and Clean-Image Results

In the P2 uniform+ECC clean cohort of 50 images, the measured mean PSNR was **42.01 dB**, mean SSIM was **0.9537**, and mean coded-bit accuracy was **100%**. Reed-Solomon recovered the exact message in **50/50** cases.

The separate P12 clean grass demo recovered `TEST` exactly, with ECC success, **98.44%** coded-bit accuracy, **0.0156** BER, **42.00 dB** PSNR, and **0.9970** SSIM. The difference in coded-bit accuracy is not contradictory: P12 is a single run in a different experiment, while P2 reports a 50-image cohort mean/result.

## 8. Robustness Against Image Attacks

The P2 uniform+ECC cohort recorded **150/150** exact recoveries across light, moderate, and severe JPEG cases. In that cohort, blur and resize each had 50/50 exact recoveries at each listed severity, while crop had 50/50 exact recoveries despite mean coded-bit accuracy of 99.97%. Smoothing was the clear exception: severe smoothing had **37.12%** mean coded-bit accuracy and only **6/50** exact recoveries.

Other experiments demonstrate why attack outcomes are cohort-specific. P9's three-image, 57-case analysis found exact recovery in only 1/3 severe-blur cases, 1/3 severe-resize cases, 2/3 moderate-blur cases, and 0/3 severe-smoothing cases. In the separate P12 grass demo, moderate blur produced `TAST` rather than `TEST`; ECC and exact recovery both failed despite **95.31%** coded-bit accuracy. These measurements do not support a universal robustness claim.

## 9. Partial-Watermark Recovery

P7 tested the current redundant observations and soft-voting logic with deterministic controlled corruption. Exact recovery succeeded at requested corruption levels of 0%, 10%, 25%, and 40%; because each bit has a finite number of copies, the actual observed corruption was 0%, 12.5%, 25%, and 37.5%, respectively. At 50% actual corruption, coded-bit accuracy fell to 45.31% and reconstruction failed. At actual 62.5% and 75% corruption, bit accuracy was 0% and reconstruction failed.

This synthetic result demonstrates the voting layer under controlled conditions only. It is not evidence that arbitrary image attacks with the same nominal percentages will behave identically.

## 10. Reed-Solomon Symbol-Level Error Analysis

P9 measured erroneous RS byte symbols alongside bit errors for 57 real-image attack cases. In that tested set, all **45/45** cases with at most two erroneous RS bytes decoded successfully, while all **12/12** cases with more than two erroneous bytes failed.

This is an observed result for the evaluated configuration, not a guarantee for all possible error patterns or inputs. The symbol-level view explains why a high bit-accuracy percentage is not sufficient on its own: errors in different bytes can exceed the two-symbol correction capacity, even when the total number of wrong bits is small. Severe smoothing was particularly damaging; it averaged 5.67 erroneous bytes and yielded 0/3 exact recoveries in P9.

## 11. Confidence-Based Erasure Decoding Experiment

P10 tested whether existing extraction confidence totals could identify unreliable RS bytes and improve decoding. Normal ECC recovered **45/57** messages. The best tested confidence-based result was also **45/57**, at threshold 0.00, which predicted no erasures and therefore matched normal decoding. Other tested thresholds performed worse.

Oracle erasure decoding reached **53/57**, but it was given the actual erroneous byte positions for analysis. That information is unavailable to the production system, so the oracle result is analysis-only and was not adopted.

## 12. Robust Voting Experiment

P11 compared five voting rules over the same extracted block margins. Current production soft voting reached **45/57** exact recoveries. Median-margin voting reached 46/57; trimmed mean 18/57; sign majority 49/57; and confidence-weighted sign 17/57.

Sign majority's improvement over the baseline was modest and not sufficiently consistent across the tested images and attacks; it did not resolve severe smoothing. Therefore, the comparison did not justify replacing the production extractor.

## 13. Final Demonstration

The final Streamlit interface and CLI demonstrate the production `embed → optional attack → extract → ECC → reconstruct` path. The UI displays the original, watermarked, and attacked images, the recovered message, exact/ECC status, coded-bit accuracy, BER, PSNR, and SSIM.

Measured P12 examples on grass:

| Case | Recovered | Exact | ECC | Bit accuracy | BER | PSNR | SSIM |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Clean | `TEST` | Yes | Yes | 98.44% | 0.0156 | 42.00 dB | 0.9970 |
| Moderate blur | `TAST` | No | No | 95.31% | 0.0469 | 42.00 dB | 0.9970 |

These are two fixed-seed demo runs, not an aggregate robustness benchmark.

## 14. Failure Analysis

Redundant block observations provide resilience when individual blocks are disturbed or lost. Under many evaluated conditions, the votes still recovered the coded payload and ECC reconstructed the message. More severe distortion can flip multiple coded bits; when those errors are distributed across more than two of the eight RS bytes, the configured decoder may fail. P9's observed 45/45 versus 0/12 split supports this failure analysis for the tested cases.

Severe smoothing is the clearest weakness in the P2 and P9-P11 evidence. P10 confidence values did not reliably localize bytes that were actually wrong. P11 sign majority improved aggregate exact recovery in its test set, but did not do so consistently enough to warrant production adoption.

## 15. Final Design Decision

The current production extraction and ECC configuration were retained. Baseline performance was strong for many tested attacks, and redundancy plus Reed-Solomon provided meaningful recovery. P10 did not improve recovery; P11 produced a modest, inconsistent gain; and neither experiment justified additional production complexity. Severe smoothing remains a known limitation.

## 16. Limitations

- P8-P11 used a limited set of three clean images; P8 alone used one image.
- Severe smoothing remains a major weakness in the measured results.
- Bit errors spread across more than two RS symbols can cause ECC failure.
- The tested extraction confidence was not sufficiently calibrated for reliable erasure localization.
- Alternative voting did not provide sufficiently consistent improvement to replace production extraction.
- P7, P12, P2, and P8-P11 use different cohorts and conditions; their results are not one common test set.

## 17. Conclusion

Member C delivered and evaluated a practical end-to-end watermark pipeline with keyed redundant payload distribution, blind block-margin extraction, Reed-Solomon error correction, recovery metrics, and a user-facing Streamlit/CLI demonstration. The evaluation included baseline and attack results, controlled partial-recovery testing, symbol-level failure analysis, and experimental erasure and voting comparisons.

The measured evidence shows useful recovery across many tested conditions, while also identifying severe smoothing and symbol-spread errors as concrete limits. The production design was retained because the experimental alternatives did not provide a sufficiently consistent improvement. These conclusions describe the tested configurations and cohorts, not universal robustness.
