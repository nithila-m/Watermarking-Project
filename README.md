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
python -m pytest                                                            # 37 tests, ~5 s
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
| any time | `streamlit run app\streamlit_app.py` | interactive test bench |

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
│   ├── p1_selftest.py … p6_benchmark.py
├── app/
│   └── streamlit_app.py      # embed -> edit -> verify test bench
├── tests/                    # pytest: metrics, every attack, end-to-end round trip, benchmark harness
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
