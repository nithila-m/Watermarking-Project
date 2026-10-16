"""Adaptive DWT-DCT watermarking with an AI-edit survival predictor.

This package exposes the core blind transform-domain embedding and extraction routines
for the project:
  - embed: Low-level DWT-DCT coefficient-pair embedder using a fixed global strength scalar.
  - embed_at_psnr: Binary-search calibrated embedder guaranteeing exact target PSNR (e.g., 42 dB).
  - extract: Blind extractor recovering embedded bitstreams via median-scaled tanh soft voting.
"""
from .embed import embed, embed_at_psnr
from .extract import extract

# Public API definitions exported at package level
__all__ = ["embed", "embed_at_psnr", "extract"]
