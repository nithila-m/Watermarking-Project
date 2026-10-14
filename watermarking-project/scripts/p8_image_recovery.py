r"""Evaluate Member C's ECC + repeated-block recovery on attacked test images.

Run with: python scripts\p8_image_recovery.py
"""
import re
from pathlib import Path

import pandas as pd

from watermark import dataset, ecc
from watermark.attacks import attack
from watermark.config import PAYLOAD_BITS, TARGET_PSNR
from watermark.extract import extract
from watermark.metrics import ber
from watermark.strength_map import uniform_map
from watermark.embed import embed_at_psnr

KEY = 2024
SEED = 2026
MESSAGE_TEXT = "TEST"
ATTACK_CASES = [("none", "none")] + [
    (kind, severity)
    for kind in ("blur", "jpeg", "resize", "crop", "inpaint", "smooth")
    for severity in ("light", "moderate", "severe")
]
DEFAULT_CSV = Path("results") / "p8_image_recovery.csv"


def select_test_image():
    """Choose the first sorted, un-attacked test image for repeatable evaluation."""
    names = [name for name in dataset.split_names("test")
             if not re.search(r"_c\d+$", name)]
    if not names:
        dataset.require("test")
        raise ValueError("test split contains no un-attacked base PNG images")
    name = names[0]
    return name, dataset.load("test", name)


def run_experiment(image, image_name, attack_cases=ATTACK_CASES, key=KEY, seed=SEED,
                   include_bits=False, include_observations=False):
    """Embed once, then measure extraction and ECC recovery for each attack case."""
    message = ecc.text_to_bits(MESSAGE_TEXT)
    payload = ecc.encode(message)
    if len(payload) != PAYLOAD_BITS:
        raise ValueError(f"ECC produced {len(payload)} bits; expected {PAYLOAD_BITS}")

    watermarked, _, embed_psnr = embed_at_psnr(
        image, payload, uniform_map(image), key, target_psnr=TARGET_PSNR
    )
    rows = []
    for kind, severity in attack_cases:
        attacked = watermarked.copy() if kind == "none" else attack(
            watermarked, kind, severity, seed=seed
        )
        extracted, details = extract(
            attacked, PAYLOAD_BITS, key, mode="soft", return_details=True
        )
        decoded, ecc_ok = ecc.decode(extracted)
        error_rate = ber(payload, extracted)
        totals = details["totals"]
        row = {
            "image": image_name,
            "attack": kind,
            "severity": severity,
            "coded_bit_accuracy": 1.0 - error_rate,
            "ber": error_rate,
            "ecc_decode_ok": bool(ecc_ok),
            "exact_message_recovered": bool(
                ecc_ok and (decoded == message).all()
            ),
            "decoded_text": ecc.bits_to_text(decoded),
            "mean_vote_strength": float(abs(totals).mean() / max(
                1, len(details["assign"]) // PAYLOAD_BITS
            )),
            "tied_votes": int((totals == 0).sum()),
            "embed_psnr": float(embed_psnr),
        }
        if include_bits:
            row["coded_bits"] = payload.tolist()
            row["extracted_bits"] = extracted.tolist()
            row["vote_totals"] = totals.tolist()
        if include_observations:
            row["block_margins"] = details["margins"].tolist()
            row["block_assignment"] = details["assign"].tolist()
        rows.append(row)
    return rows


def write_results(rows, output_csv=DEFAULT_CSV):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


def print_summary(rows):
    print("Attack / severity       Coded accuracy   BER     ECC decode   Exact TEST")
    for row in rows:
        label = f"{row['attack']} / {row['severity']}"
        print(
            f"{label:<23} {row['coded_bit_accuracy']:>6.1%}"
            f"         {row['ber']:.3f}"
            f"       {str(row['ecc_decode_ok']):>5}"
            f"        {str(row['exact_message_recovered']):>5}"
        )


def main():
    image_name, image = select_test_image()
    rows = run_experiment(image, image_name)
    output_csv = write_results(rows)
    print(f"Image: {image_name}.png | key={KEY} | attack seed={SEED} | PSNR target={TARGET_PSNR} dB")
    print_summary(rows)
    print(f"\nSaved {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
