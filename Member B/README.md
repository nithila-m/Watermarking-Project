# Member B: Data Engineering & Evaluation Lead

## Role & Module Overview
This directory contains the dataset preparation pipelines, texture-profiling scripts, distortion/attack test suites, and automated evaluation harnesses for the adaptive digital watermarking system.

---

## Deliverables & File Structure

```text
Member B/
├── attacks/
│   ├── classical.py         # Standard distortions (JPEG, Gaussian blur/noise, cropping)
│   └── ai_attacks.py        # Generative perturbations (AI inpainting, neural smoothing)
├── data/
│   └── clean/
│       └── test_metadata.csv # Texture complexity metrics for evaluation images
├── evaluation/
│   ├── metrics.py           # Core evaluation metrics (PSNR, SSIM, BER, extraction rate)
│   └── benchmark.py         # Automated multi-condition test runner
├── prepare_dataset.py       # COCO val2017 preprocessing & split generation
├── analyze_textures.py      # Laplacian variance profiling (Smooth, Medium, Complex)
├── test_attacks.py          # Unit verification for classical distortion suite
├── test_ai_attacks.py       # Unit verification for generative manipulation suite
├── test_metrics.py          # Metric computation verification tests
├── test_benchmark.py        # Mock end-to-end benchmark pipeline test
└── requirements.txt         # Module dependencies
