# Final numbers (generated 2026-10-13 by scripts/p7_verify.py)

Checks: 11 passed or warned, 0 failed (details in results/p7_checks.csv).

## Setup

- Test images: 300 (Complex 100, Smooth 100, Medium 100), every config calibrated to 42.0 dB PSNR
- Learned map settings: {'alpha': 0.5, 'beta': 0.0, 'shrink': 1.0, 's_min': 0.25}; model fingerprint 097030026d4e
- Success = the 32-bit message decodes exactly (ECC configs) / all 64 bits right (no ECC)

## Table 1. Image quality and clean extraction  (source: results/p5_raw.csv)

| config | PSNR (dB) | SSIM | no-edit success |
|---|---|---|---|
| uniform | 42.01 | 0.9781 | 99.7% |
| uniform+ecc | 42.01 | 0.9781 | 100.0% |
| heuristic+ecc | 42.02 | 0.9885 | 100.0% |
| adaptive | 42.01 | 0.9799 | 99.7% |
| adaptive+ecc | 42.01 | 0.9799 | 100.0% |

## Table 2. Success rate by edit severity  (source: results/p5_by_severity.csv)

| config | light | moderate | severe | all attacks | bit accuracy |
|---|---|---|---|---|---|
| uniform | 99.7% | 99.0% | 85.0% | 94.8% | 98.7% |
| uniform+ecc | 100.0% | 99.9% | 89.4% | 96.6% | 98.7% |
| heuristic+ecc | 100.0% | 99.1% | 88.9% | 96.1% | 99.2% |
| adaptive | 99.8% | 98.9% | 84.8% | 94.7% | 98.6% |
| adaptive+ecc | 100.0% | 99.9% | 89.2% | 96.5% | 98.7% |

## Table 3. Success rate by image texture  (source: results/p5_by_texture.csv)

| config | Smooth | Medium | Complex |
|---|---|---|---|
| uniform | 95.9% | 95.3% | 93.1% |
| uniform+ecc | 96.0% | 96.5% | 97.2% |
| heuristic+ecc | 96.1% | 96.2% | 96.1% |
| adaptive | 95.8% | 95.2% | 92.9% |
| adaptive+ecc | 96.0% | 96.6% | 96.9% |

## Significance  (source: results/p5_stats.csv)

- adaptive+ecc vs uniform+ecc: -0.1 points [95% CI -0.2, +0.0], McNemar p = 0.109 (not significant; 300 images, 7500 cases)
- heuristic+ecc vs uniform+ecc: -0.4 points [95% CI -0.9, -0.1], McNemar p = 0.000628 (significant; 300 images, 7500 cases)
- adaptive vs uniform: -0.1 points [95% CI -0.3, +0.1], McNemar p = 0.358 (not significant; 300 images, 7500 cases)
- uniform+ecc vs uniform: +1.8 points [95% CI +1.2, +2.4], McNemar p = 1.59e-36 (significant; 300 images, 7500 cases)
- severity light: adaptive+ecc vs uniform+ecc: +0.0 points [95% CI +0.0, +0.0], McNemar p = 1 (not significant; 300 images, 2400 cases)
- severity moderate: adaptive+ecc vs uniform+ecc: -0.0 points [95% CI -0.1, +0.0], McNemar p = 1 (not significant; 300 images, 2700 cases)
- severity severe: adaptive+ecc vs uniform+ecc: -0.2 points [95% CI -0.5, +0.0], McNemar p = 0.18 (not significant; 300 images, 2400 cases)

## 80% target (adaptive+ecc)

- Met for: light edits (100.0%), moderate edits (99.9%), severe edits (89.2%)
- Not met for: none
