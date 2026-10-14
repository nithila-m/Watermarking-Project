"""Final user-facing Member C demo.

Launch with: streamlit run app/demo_app.py
"""
import re
import sys
from pathlib import Path

import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]      # repo root, so `scripts.demo` imports
if str(ROOT) not in sys.path:                    # however streamlit was started
    sys.path.insert(0, str(ROOT))

from scripts.demo import (
    ATTACK_CHOICES,
    decode_uploaded_image,
    load_demo_image,
    run_demo,
)
from watermark import dataset
from watermark.attacks import SEVERITIES

st.set_page_config(page_title="Digital Watermark Demo", layout="wide")
st.title("Digital Watermarking — Final Demo")
st.caption("Production Member C pipeline: embed → attack → extract → Reed-Solomon recovery.")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"]
    )
    sample_names = [
        name for name in dataset.split_names("test")
        if not re.search(r"_c\d+$", name)
    ]
    selected_sample = st.selectbox(
        "Or select a test image",
        ["(choose an image)"] + sample_names,
    )
    watermark_text = st.text_input("Watermark (up to 4 UTF-8 bytes)", value="TEST")
    attack_kind = st.selectbox(
        "Attack",
        ATTACK_CHOICES,
        format_func=lambda name: "None" if name == "none" else name.upper(),
    )
    severity = st.selectbox(
        "Severity",
        SEVERITIES,
        disabled=(attack_kind == "none"),
    )
    run_button = st.button("Embed & Test", type="primary")

if run_button:
    try:
        if uploaded is not None:
            image = decode_uploaded_image(uploaded.getvalue())
        elif selected_sample != "(choose an image)":
            image = load_demo_image(dataset.image_path("test", selected_sample))
        else:
            raise ValueError("Upload an image or choose a test image.")

        result = run_demo(image, watermark_text, attack_kind, severity)
        st.subheader("Images")
        original_col, watermarked_col, attacked_col = st.columns(3)
        original_col.image(result["original_image"], channels="BGR", caption="Original image")
        watermarked_col.image(
            result["watermarked_image"], channels="BGR", caption="Watermarked image"
        )
        attacked_col.image(
            result["attacked_image"],
            channels="BGR",
            caption=f"Attacked image — {result['attack']} / {result['severity']}",
        )

        st.subheader("Recovery")
        status_col, recovered_col, ecc_col, attack_col = st.columns(4)
        status_col.metric("Exact recovery", "YES" if result["exact_recovery"] else "NO")
        recovered_col.metric("Recovered watermark", result["recovered_text"] or "(empty)")
        ecc_col.metric("ECC decoding", "SUCCESS" if result["ecc_success"] else "FAILED")
        attack_col.metric("Attack", f"{result['attack']} / {result['severity']}")

        metric_cols = st.columns(4)
        metric_cols[0].metric("Bit accuracy", f"{result['bit_accuracy']:.2%}")
        metric_cols[1].metric("BER", f"{result['ber']:.4f}")
        metric_cols[2].metric("PSNR", f"{result['psnr']:.2f} dB")
        metric_cols[3].metric("SSIM", f"{result['ssim']:.4f}")
        st.caption(f"Original watermark: {result['original_text']}")
    except (ValueError, OSError, cv2.error) as exc:
        st.error(str(exc))
