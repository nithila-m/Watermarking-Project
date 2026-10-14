# Member C Final Presentation Content

## Slide 1 — Member C Contribution

- Digital Watermarking Project
- Member C: systems integration, Reed-Solomon ECC, redundancy, extraction/reconstruction evaluation, testing, and final demo
- P2 and P7-P13 evidence consolidated
- Adaptive/AI model work belongs to other team members

## Slide 2 — System Architecture

```text
Text → ECC encode → keyed redundant embedding → Watermarked image
     → optional attack → blind extraction → redundant soft voting
     → Reed-Solomon decode → recovered text + metrics
```

- Original host image is not required for extraction.
- Shared key is required to reproduce the block assignment.

## Slide 3 — Redundancy + ECC

- Message payload: 32 bits / 4 bytes
- Reed-Solomon codeword: 64 bits / 8 bytes, including 4 parity bytes
- Configured correction capacity: up to 2 erroneous byte symbols
- 512×512 setup: 1,024 DWT-DCT blocks, about 16 observations per coded bit
- Redundancy handles noisy/missing block observations; ECC repairs residual symbol errors

## Slide 4 — End-to-End Pipeline

1. Convert short text to 32 message bits and RS-encode.
2. Create strength map and keyed block assignment.
3. Embed 64 coded bits using the production DWT-DCT embed API.
4. Optionally apply an existing attack.
5. Extract block margins, vote, and RS-decode.
6. Report recovered text, bit accuracy, BER, PSNR, SSIM, ECC and exact status.

Production demo APIs include `embed_at_psnr`, `extract`, and `ecc.encode/decode`.

## Slide 5 — Experimental Setup

- P2 baseline: 50-image uniform+ECC cohort
- P7: deterministic synthetic repeated-observation corruption
- P8: grass image, 19 attack/severity cases
- P9-P11: grass, gravel, hubble_deep_field; 57 cases per comparison baseline
- Attacks: JPEG, blur, resize, crop, smoothing, at available severities
- Metrics: coded-bit accuracy, BER, RS byte errors, ECC/exact recovery; PSNR/SSIM where recorded
- Results from distinct cohorts are reported separately

## Slide 6 — Clean / Baseline Results

- P2 clean uniform+ECC (50 images)
- Mean PSNR: **42.01 dB**
- Mean SSIM: **0.9537**
- Mean coded-bit accuracy: **100%**
- Exact message recovery: **50/50**
- Separate P12 clean grass demo: `TEST`, exact/ECC YES, 98.44% accuracy

## Slide 7 — Attack Robustness

- P2 JPEG: **150/150** exact recoveries across three severities
- P2 blur and resize: 50/50 at each reported severity
- P2 crop: 50/50 despite 99.97% mean bit accuracy
- P2 severe smoothing: **37.12%** mean bit accuracy; **6/50** exact recovery
- P9/P12 contain failures for other image/attack cohorts
- Severe smoothing is the clearest measured weakness

## Slide 8 — RS Symbol-Level Analysis

- Reed-Solomon corrects byte symbols, not a fixed percentage of individual bits.
- P9: **45/45** cases with ≤2 erroneous RS bytes decoded.
- P9: **0/12** cases with >2 erroneous bytes decoded.
- These are observed results for this test set/configuration, not a universal guarantee.
- Few bit errors can still affect many bytes and exceed correction capacity.

## Slide 9 — Recovery Experiments

| Method | Exact recovery |
| --- | ---: |
| Current soft vote (P11) | 45/57 |
| Sign majority (P11) | 49/57 |
| Normal ECC (P10) | 45/57 |
| Best confidence erasure (P10) | 45/57 |
| Oracle erasure (analysis-only) | 53/57 |

- Sign majority was modestly better but not sufficiently consistent.
- Oracle uses known error locations and is not deployable production decoding.

## Slide 10 — Final Demo

- Streamlit UI: `streamlit run app\demo_app.py`
- CLI: `python scripts\demo.py --image data\clean\test\grass.png --text TEST --attack blur --severity moderate`
- Clean grass: TEST, exact/ECC YES; 98.44% accuracy; BER 0.0156
- Moderate blur grass: TAST, exact/ECC NO; 95.31% accuracy; BER 0.0469
- Both examples: PSNR 42.00 dB; SSIM 0.9970
- Demo uses production extraction and ECC

## Slide 11 — Limitations + Design Decision

- Severe smoothing remains a major weakness.
- Errors spread across more than two RS bytes can defeat decoding.
- Confidence did not reliably identify erroneous bytes.
- Alternative voters did not show sufficiently consistent gains.
- **Decision:** retain current production extraction and ECC; no experimental method promoted.

## Slide 12 — Conclusion / Future Work

- Delivered integrated redundant embedding, blind extraction, RS recovery, evaluation, and demo.
- Used measured experiments to characterize both strengths and failure modes.
- No claim of perfect or universal robustness.
- Possible future work: evaluate confidence calibration and resilience to smoothing on broader, controlled image cohorts.
- Future work is proposed, not implemented in this evaluation.
