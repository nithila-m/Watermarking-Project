# Adaptive watermarking: Member A code (Phases 1-5)

All commands are for Windows PowerShell, run from the project folder with the venv active.

## Setup (once)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'saliency'))"   # must print True
```

## Phase by phase
| Phase | Command | Output |
|---|---|---|
| 1 data | `python scripts\prepare_images.py --samples` (dev) or `--src data\raw` (real) | `data\processed\*.png`, `split.json` |
| 1 check | `python scripts\p1_selftest.py` | prints `PHASE 1 PASSED` |
| 2 baseline | `python scripts\p2_end_to_end.py --n 50 --export-for-b` | `results\p2_baseline.csv`, `results\for_b\` |
| 3 dataset | `python scripts\p3_build_dataset.py [--external data\external]` | `data\survival\blocks.csv.gz` |
| 4 predictor | `python scripts\p4_train_predictor.py` | `models\survival.joblib`, `results\p4_*` |
| 5 ablation | `python scripts\p5_ablation.py --alpha 1.0 --beta 0.0` | `results\p5_*.csv`, `results\p5_*.png` |
| any time | `streamlit run app\streamlit_app.py` | test bench in the browser |

## Code map (`src\watermark\`)
- `config.py` shared constants (block size, coefficient pair, payload, target PSNR, skip ratio)
- `core.py` YCrCb, Haar DWT, 8x8 block DCT, keyed bit-to-block assignment
- `embed.py` pair-comparison embedding with a per-block strength map; `embed_at_psnr` calibrates to a PSNR
- `extract.py` blind extraction with soft (tanh-weighted) voting
- `ecc.py` Reed-Solomon (32 message bits -> 64 coded bits); C owns the final version
- `attacks.py` local attack suite incl. saliency-targeted OpenCV inpainting as a stand-in for diffusion edits
- `features.py` 13 per-block features
- `predictor.py` survival dataset + two random forests (disturbance `log_sigma`, erasure `keep`)
- `strength_map.py` uniform / heuristic / learned maps
- `evaluate.py` shared evaluation loop and the ablation configurations
- `metrics.py` PSNR, SSIM, BER, NC
