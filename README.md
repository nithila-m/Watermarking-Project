# Adaptive Digital Watermarking with AI-Based Image Analysis

Blind DWT-DCT image watermark whose per-block embedding strength is set by a learned
"AI-edit survival predictor", evaluated against classical distortions and simulated AI
edits (inpainting, neural smoothing). Every configuration is calibrated to the same PSNR,
so uniform vs. adaptive is a fair comparison.

## Setup (once, Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'saliency'))"   # must print True
python -m pytest
```
If `cv2.saliency` is missing, plain `opencv-python` is installed next to the contrib build:
`pip uninstall -y opencv-python opencv-contrib-python` then `pip install opencv-contrib-python`.

## Pipeline
| Step | Command | Output |
|---|---|---|
| 0 data | `python scripts\prepare_dataset.py --src raw\val2017` (COCO) or `--samples` (dev) | `data\clean\{train,val,test}\*.png` |
| 0 textures | `python scripts\analyze_textures.py` | `data\clean\test_metadata.csv` |
| 1 self-test | `python scripts\p1_selftest.py` | prints `PHASE 1 PASSED` |
| 2 baseline | `python scripts\p2_end_to_end.py --n 50 --export-for-b` | `results\p2_baseline.csv`, `results\for_b\` |
| 3 survival data | `python scripts\p3_build_dataset.py [--external data\external]` | `data\survival\blocks.csv.gz` |
| 4 predictor | `python scripts\p4_train_predictor.py` | `models\survival.joblib`, `results\p4_*` |
| 5 ablation | `python scripts\p5_ablation.py --alpha 1.0 --beta 0.0` | `results\p5_*.csv/png` (incl. per-texture table) |
| 6 benchmark | `python scripts\p6_benchmark.py` (`--mock` to test the harness) | `results\p6_*.csv` |
| 7 partial recovery | `python scripts\p7_partial_recovery.py` | deterministic redundancy + ECC recovery report |
| 8 image recovery | `python scripts\p8_image_recovery.py` | real-image attack recovery report and `results\p8_image_recovery.csv` |
| 9 RS symbol analysis | `python -m scripts.p9_symbol_error_analysis` | deterministic multi-image byte-symbol analysis and `results\p9_symbol_error_analysis.csv` |
| 10 erasure study | `python -m scripts.p10_erasure_recovery` | confidence-aware RS erasure experiment and `results\p10_erasure_recovery.csv` |
| 11 robust voting | `python -m scripts.p11_robust_voting` | experimental margin-voting comparison and `results\p11_robust_voting.csv` |
| 13 final evaluation | `python -m scripts.final_member_c_evaluation` | consolidated CSVs and report tables/findings in `results\` |
| final demo | `streamlit run app\demo_app.py` | interactive Member C embed/attack/recovery demo |
| demo CLI | `python scripts\demo.py --image data\clean\test\grass.png --text TEST` | prints embed/extract/ECC result |
| adaptive bench | `streamlit run app\streamlit_app.py` | existing adaptive watermark test bench |
| final evaluation | `python -m scripts.final_member_c_evaluation` | consolidated Member C CSVs and report-ready tables/findings |

P10 is an experimental confidence-aware erasure study based on existing extraction
vote totals; it does not change production decoding. Its oracle-erasure results use
known original symbols for analysis only and are **not part of the production system**.

P11 is an experimental comparison of aggregation rules over the same extracted
block margins; production extraction is unchanged. Alternatives are: median
margin (median signed margin, positive means 1); trimmed mean (drop one minimum
and maximum margin when at least three observations are available, then threshold
the mean at zero); sign majority (sum margin signs, ties decode to 0); and
confidence-weighted sign (sum `sign(margin) * abs(margin)`, positive means 1,
ties decode to 0). The baseline remains the existing median-scaled tanh soft vote.

## Final Demo

The final user-facing demo uses the **production Member C pipeline**: text-to-bits,
existing Reed-Solomon encoding, a uniform strength map, target-PSNR embedding,
optional existing image attack, production extraction, and existing Reed-Solomon
decoding. It does not use P10 erasures or P11 experimental voting.

Launch the upload/sample-image interface from the repository root:

```powershell
streamlit run app\demo_app.py
```

Choose an uploaded image or a clean test image, enter a watermark such as `TEST`,
select an optional attack and severity, then click **Embed & Test**. Inputs are
center-cropped and resized to 512x512 for the watermark pipeline. The UI displays
the original, watermarked, and attacked images; recovered text; exact recovery;
ECC status; bit accuracy; BER; PSNR; and SSIM.

Run the CLI from the repository root:

```powershell
python scripts\demo.py --image data\clean\test\grass.png --text TEST --attack blur --severity moderate
```

Use `--attack none` for a clean round trip. Watermarks are limited to four UTF-8
bytes by the existing 32-bit message payload. Bit accuracy and BER compare the
extracted 64-bit coded payload with the encoded payload; PSNR and SSIM compare
the original input with the watermarked image. These metrics describe a single run
and do not imply robustness beyond the measured P8/P9/P10/P11 experiments.

The final evaluation package reuses the saved P2 and P8-P11 CSV files. It generates
only the missing lightweight P7 synthetic-recovery rows and P12 fixed-seed demo
verification rows. Outputs are `results\final_member_c_results.csv`,
`results\final_member_c_summary.csv`, `results\final_member_c_tables.md`, and
`results\final_member_c_findings.md`.

P13 consolidates the existing Member C results into `results\final_member_c_results.csv`,
`results\final_member_c_summary.csv`, `results\final_member_c_tables.md`, and
`results\final_member_c_findings.md`. It reuses saved P2 and P8-P11 result files;
only missing lightweight P7 synthetic rows and P12 demo-verification rows are
generated when building the package.

Real Stable Diffusion edits: `p2 --export-for-b` writes the watermarked training images to
`results\for_b\`; edit them on Colab and put the results, with the same file names, in
`data\external\<edit_name>\` (folder names containing inpaint/erase/crop count as erasure edits).

## File structure
```text
watermarking-project/
├── README.md
├── requirements.txt          # single dependency list for the whole project
├── pyproject.toml            # installs src/watermark as the `watermark` package; pytest config
├── .gitignore
├── src/watermark/
│   ├── __init__.py           # exports embed, embed_at_psnr, extract
│   ├── config.py             # shared constants: data paths, block size, coefficient pair, payload, PSNR target
│   ├── dataset.py            # data/clean/{train,val,test} helpers used by every script
│   ├── core.py               # image I/O, YCrCb, Haar DWT, 8x8 block DCT, keyed bit-to-block assignment
│   ├── embed.py              # coefficient-pair embedding with per-block strength; PSNR calibration
│   ├── extract.py            # blind extraction with soft (tanh-weighted) voting
│   ├── ecc.py                # Reed-Solomon: 32 message bits -> 64 coded bits; text <-> bits
│   ├── attacks.py            # classical + AI-edit attack suite, 3 severities each
│   ├── features.py           # 13 per-block features (texture, gradients, saliency, DCT, position)
│   ├── predictor.py          # survival dataset builder + two random forests (log_sigma, keep)
│   ├── strength_map.py       # uniform / heuristic / learned strength maps
│   ├── evaluate.py           # shared evaluation loop and the 5 ablation configurations
│   ├── benchmark.py          # method-agnostic benchmark harness with named attack conditions
│   ├── textures.py           # Laplacian-variance texture profiling (Smooth / Medium / Complex)
│   └── metrics.py            # PSNR, SSIM, BER, NC, BER-threshold success
├── scripts/
│   ├── prepare_dataset.py    # COCO (aspect filter, 70/10/20 split) or scikit-image dev set
│   ├── analyze_textures.py   # writes data/clean/test_metadata.csv
│   ├── p1_selftest.py … p6_benchmark.py, p7_partial_recovery.py … p11_robust_voting.py, demo.py
├── app/
│   ├── demo_app.py           # final Member C upload/sample -> embed -> attack -> recovery demo
│   └── streamlit_app.py      # existing adaptive watermark test bench
├── tests/                    # pytest: metrics, attacks, end-to-end, recovery experiments, final demo
│   ├── conftest.py
│   ├── test_metrics.py
│   ├── test_attacks.py
│   └── test_pipeline.py
└── data/clean/
    └── test_metadata.csv     # texture category of each COCO test image (regenerate after re-splitting)
```

## Attack suite (`watermark.attacks`)
| Kind | light / moderate / severe | Type |
|---|---|---|
| jpeg | quality 90 / 70 / 50 | classical |
| blur | Gaussian sigma 0.6 / 1.0 / 1.6 | classical |
| noise | Gaussian std 3 / 6 / 12 (0-255) | classical |
| resize | downscale 0.75 / 0.5 / 0.35 and back | classical |
| crop | 10 / 25 / 45 % of area blacked out, in place | classical |
| inpaint | 10 / 20 / 35 % inpainted, centred on salient regions | AI edit |
| inpaint_square | one random square, 5 / 15 / 30 % | AI edit |
| smooth | bilateral filter (AI "enhance" filter) | AI edit |
| crop_resize | cut 5 / 10 / 15 % borders and rescale | geometric, not in the default suite (shifts the block grid) |

## Two success criteria
* **Strict** (`success` in evaluate.py / p5): ECC off, all 64 bits right; ECC on, the 32-bit message decodes exactly.
* **Lenient** (`ber_ok` in evaluate.py, `success` in benchmark.py / p6): raw BER ≤ 15 %.
Report which one a number uses.
