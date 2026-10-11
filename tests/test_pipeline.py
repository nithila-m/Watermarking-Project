"""End-to-end: embed -> attack -> extract -> ECC decode, and the benchmark harness."""
import cv2
import numpy as np

from watermark import ecc
from watermark.attacks import attack
from watermark.benchmark import BENCHMARK_SUITE, default_methods, run_benchmark
from watermark.config import MESSAGE_BITS, PAYLOAD_BITS, TARGET_PSNR
from watermark.embed import embed_at_psnr
from watermark.extract import extract
from watermark.metrics import ber

KEY = 2024


def test_ecc_corrects_two_byte_errors(rng):
    msg = rng.integers(0, 2, MESSAGE_BITS)
    coded = ecc.encode(msg)
    coded[[3, 20]] ^= 1
    dec, ok = ecc.decode(coded)
    assert ok and np.array_equal(dec, msg)


def test_clean_and_jpeg_round_trip(image, rng):
    bits = rng.integers(0, 2, PAYLOAD_BITS)
    wm, _, p = embed_at_psnr(image, bits, np.ones(1024), KEY)
    assert p >= TARGET_PSNR - 0.5
    assert ber(bits, extract(wm, PAYLOAD_BITS, KEY)) == 0
    assert ber(bits, extract(attack(wm, "jpeg", "moderate"), PAYLOAD_BITS, KEY)) < 0.05


def test_wrong_key_is_near_random(image, rng):
    bits = rng.integers(0, 2, PAYLOAD_BITS)
    wm, _, _ = embed_at_psnr(image, bits, np.ones(1024), KEY)
    assert ber(bits, extract(wm, PAYLOAD_BITS, KEY + 1)) > 0.25


def test_benchmark_harness(image, tmp_path, rng):
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), image)
    payload = rng.integers(0, 2, PAYLOAD_BITS)
    embed_fn, extract_fn = default_methods()
    summary, raw = run_benchmark([path], embed_fn, extract_fn, payload,
                                 output_csv=tmp_path / "summary.csv")
    assert len(raw) == len(BENCHMARK_SUITE)
    ident = summary.set_index("condition").loc["Identity (No Attack)"]
    assert ident.ber == 0 and ident.success_rate == 1.0
    assert (tmp_path / "summary.csv").exists()
