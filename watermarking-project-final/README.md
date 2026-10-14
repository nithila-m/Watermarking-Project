# Adaptive Digital Watermarking with AI-Based Image Analysis

Blind DWT-DCT image watermark whose per-block embedding strength is set by a learned
"AI-edit survival predictor", evaluated against classical distortions and simulated and real
AI edits (inpainting, neural smoothing, hand/Colab edits). Every configuration is calibrated
to the same PSNR (`TARGET_PSNR = 42` dB in `src/watermark/config.py`), so uniform vs.
adaptive is a fair comparison.

Team: **A** watermark core + predictor (Nithila) · **B** data + evaluation (Madhumithra) ·
**C** recovery experiments, demo, final report (Jithendar).

## Setup (once, Windows PowerShell, from this folder)
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

**Always run from this folder (the repo root).** Member C's P9-P11 and the final evaluation
import `scripts.*`, so run them with `python -m scripts.<name>` (no `.py`).

## Pipeline
| Step | Owner | Command | Output |
|---|---|---|---|
| 0 data | B | `python scripts\prepare_dataset.py --src raw\val2017` (COCO) or `--samples` (dev) | `data\clean\{train,val,test}\*.png` |
| 0 textures | B | `python scripts\analyze_textures.py` | `data\clean\test_metadata.csv` |
| 0 check | A | `python scripts\check_data.py [--external data\external]` | data sanity + time estimates |
| 1 self-test | A | `python scripts\p1_selftest.py` | prints `PHASE 1 PASSED` |
| 2 baseline | A | `python scripts\p2_end_to_end.py --n 50 --export-for-b` | `results\p2_baseline.csv`, `results\for_b\` |
| 2b AI edits | A/B | `python scripts\export_for_colab.py --train 200 --zip`, `colab\sd_edits.py` or hand edits, then `python scripts\clean_external.py --apply` | `data\external\<edit>\train\*.png` |
| 3 survival data | A | `python scripts\p3_build_dataset.py --external data\external` | `data\survival\blocks.csv.gz` |
| 4 predictor | A | `python scripts\p4_train_predictor.py` | `models\survival.joblib`, `results\p4_choice.json` |
| 5 ablation | A | `python scripts\p5_ablation.py` (resumable) | `results\p5_*.csv/png`, stats + per-texture |
| 5b real edits | A | `python scripts\p5_external.py` | ablation on real AI edits |
| 6 benchmark | B | `python scripts\p6_benchmark.py` (`--mock` to test the harness) | `results\p6_*.csv` (lenient BER criterion) |
| 6 failures | A | `python scripts\p6_failure_analysis.py [--try-fixes 40]` | `results\p6_fail*`, case pictures |
| 7 verify | A | `python scripts\p7_verify.py` | checks + `results\final_numbers.md` |
| 7 partial recovery | C | `python -m scripts.p7_partial_recovery` | synthetic redundancy + ECC recovery |
| 8 image recovery | C | `python -m scripts.p8_image_recovery` | `results\p8_image_recovery.csv` (first test image) |
| 9 RS symbols | C | `python -m scripts.p9_symbol_error_analysis` | `results\p9_symbol_error_analysis.csv` (3 test images) |
| 10 erasures | C | `python -m scripts.p10_erasure_recovery` | `results\p10_erasure_recovery.csv` |
| 11 robust voting | C | `python -m scripts.p11_robust_voting` | `results\p11_robust_voting.csv` |
| 13 final package | C | `python -m scripts.final_member_c_evaluation` | `results\final_member_c_*.csv/md` |
| demo CLI | C | `python -m scripts.demo --image <png> --text TEST --attack blur --severity moderate` | one embed/attack/decode run |
| demo app | C | `streamlit run app\demo_app.py` | final user-facing demo (uniform map) |
| test bench | A | `streamlit run app\streamlit_app.py` | uniform / heuristic / learned comparison |

Name overlaps: `p6_benchmark` (B, method-agnostic harness) and `p6_failure_analysis` (A, why
adaptive fails) are different steps; so are `p7_verify` (A, final checks) and
`p7_partial_recovery` (C, synthetic recovery). P8-P11 need `results\p2_baseline.csv` only for
the final package, so run P2 first.

Settings: `results\p4_choice.json` (alpha/beta/shrink, written by P4, read by P5-P7 through
`watermark.choice`) is the single source of the learned-map settings. P5 and P7 record the
model's md5, so a retrained model is detected.

P10 is an experimental confidence-aware erasure study based on existing extraction vote
totals; it does not change production decoding. Its oracle-erasure results use known original
symbols for analysis only. P11 compares aggregation rules (median, trimmed mean, sign majority,
confidence-weighted sign) over the same block margins; production keeps the tanh soft vote.
P8-P11 use the first 1 / 3 test images (sorted names), so they run on whatever data is in
`data\clean\test` (scikit-image samples or COCO).

The final demo uses the production pipeline: text-to-bits, Reed-Solomon, uniform strength map,
target-PSNR embedding, optional attack, extraction, RS decoding. Watermarks are limited to four
UTF-8 bytes (32-bit message).

## File structure
```text
watermarking-project/
├── README.md, requirements.txt, pyproject.toml, .gitignore
├── src/watermark/
│   ├── config.py        # shared constants (block size, payload, TARGET_PSNR = 42)
│   ├── dataset.py       # data/clean/{train,val,test} helpers
│   ├── core.py          # image I/O, YCrCb, Haar DWT, 8x8 DCT, keyed bit-to-block assignment
│   ├── embed.py         # coefficient-pair embedding, PSNR calibration
│   ├── extract.py       # blind extraction, soft (tanh) voting
│   ├── ecc.py           # Reed-Solomon 32 -> 64 bits; text <-> bits
│   ├── attacks.py       # classical + AI-edit attacks, 3 severities
│   ├── features.py      # 13 per-block features
│   ├── predictor.py     # survival dataset + two random forests
│   ├── strength_map.py  # uniform / heuristic / learned maps (alpha, beta, shrink, s_min)
│   ├── evaluate.py      # evaluation loop, 5 ablation configs
│   ├── benchmark.py     # B's benchmark harness
│   ├── textures.py      # Smooth / Medium / Complex profiling
│   ├── metrics.py       # PSNR, SSIM, BER, NC
│   ├── parallel.py      # multi-core runs (Windows spawn, <=4 workers, failure guard)
│   ├── jobs.py          # per-image worker jobs
│   ├── external.py      # real AI edits (Colab / hand) layout and survival rows
│   ├── sampling.py      # nested, texture-stratified samples
│   ├── stats.py         # paired bootstrap CI + McNemar
│   ├── choice.py        # reads results/p4_choice.json, model md5
│   └── failure.py       # Phase 6 failure reproduction
├── scripts/             # see the pipeline table
├── colab/sd_edits.py    # Stable Diffusion inpaint / regenerate / VAE edits on Colab
├── app/
│   ├── demo_app.py      # C: final demo
│   └── streamlit_app.py # A: adaptive test bench
├── tests/               # pytest for all members' code
├── report_assets/       # A's final numbers and plots; member_c/ = C's report + slides text
└── data/clean/test_metadata.csv   # texture category of each COCO test image
```
Not in git (make them with the scripts, or copy them from a teammate): `data/clean/*/`,
`data/external/`, `data/survival/`, `models/`, `results/`.

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
* **Strict** (`success` in evaluate.py / p5 / P8-P11 "exact recovery"): ECC off, all 64 bits right; ECC on, the 32-bit message decodes exactly.
* **Lenient** (`ber_ok` in evaluate.py, `success` in benchmark.py / p6_benchmark): raw BER ≤ 15 %.
Report which one a number uses.
