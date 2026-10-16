"""Interactive Streamlit test bench for experimenting with watermark robustness.

Users can upload or select an image, choose a watermark and embedding strategy,
apply an attack, and inspect recovery and image-quality diagnostics. The public
demo is built separately.

Launch with: streamlit run app/streamlit_app.py
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from watermark import dataset, ecc
from watermark.attacks import PARAMS, SEVERITIES
from watermark.config import PAYLOAD_BITS, SPLITS, TARGET_PSNR
from watermark.core import grid_shape
from watermark.embed import embed, embed_at_psnr
from watermark.evaluate import apply_attack, make_map
from watermark.extract import extract
from watermark.metrics import ber, psnr, ssim
from watermark.predictor import load_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean"
MODEL_PATH = ROOT / "models" / "survival.joblib"

st.set_page_config(page_title="Adaptive Watermark Bench", layout="wide")
st.title("Adaptive watermark test bench")


@st.cache_resource
def get_model():
    # Load the optional learned-map model once and reuse it across Streamlit reruns.
    return load_model(MODEL_PATH) if MODEL_PATH.exists() else None


def to_rgb(img):
    # OpenCV reads color images as BGR, while Matplotlib expects RGB.
    return img[..., ::-1] if img.ndim == 3 else img


def prepare(img):
    """Center-crop to square and resize to 512 so it matches the pipeline."""
    h, w = img.shape[:2]
    m = min(h, w)
    img = img[(h - m) // 2:(h - m) // 2 + m, (w - m) // 2:(w - m) // 2 + m]
    return cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("1. Image")
    up = st.file_uploader("Upload", type=["png", "jpg", "jpeg", "bmp", "webp"])
    samples = [f"{s}/{n}" for s in ("test",) + tuple(x for x in SPLITS if x != "test")
               for n in dataset.split_names(s, DATA)]
    pick = st.selectbox("...or a dataset image", samples) if samples else None

    st.header("2. Watermark")
    message = st.text_input("Owner ID (4 characters)", "NITH", max_chars=4)
    key = st.number_input("Secret key", value=2024, step=1)
    model = get_model()
    modes = ["uniform", "heuristic"] + (["learned"] if model else [])
    mode = st.radio("Strength map", modes, index=len(modes) - 1, horizontal=True)
    if not model:
        st.caption("Train the model (Phase 4) to enable the learned map.")
    alpha = st.slider("alpha (how strongly the map adapts)", 0.0, 2.0, 1.0, 0.25) if mode != "uniform" else 1.0
    target = st.slider("Target PSNR (dB)", 36.0, 48.0, float(TARGET_PSNR), 0.5)

    st.header("3. Edit")
    kinds = ["none"] + list(PARAMS) + ["inpaint+jpeg"]
    kind = st.selectbox("Attack", kinds, index=kinds.index("inpaint"))
    sev = st.select_slider("Severity", SEVERITIES, value="moderate") if kind != "none" else "none"
    seed = st.number_input("Attack seed", value=0, step=1)

if up is not None:
    img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
elif pick:
    img = cv2.imread(str(DATA / f"{pick}.png"), cv2.IMREAD_COLOR)
else:
    st.info("Upload an image or run scripts/prepare_dataset.py first.")
    st.stop()
# Match uploaded and dataset images to the pipeline's fixed square input size.
img = prepare(img)

# ---------------- embed ----------------
msg_bits = ecc.text_to_bits(message)
payload = ecc.encode(msg_bits)
kw = {"alpha": alpha} if mode != "uniform" else {}
base = make_map(img, mode, model, **kw)
# Adjust embedding strength to meet the selected visual-quality target.
wm, s_map, p = embed_at_psnr(img, payload, base, int(key), target)
_, skipped = embed(img, payload, s_map, int(key), return_skipped=True)

# ---------------- attack + verify ----------------
att = wm if kind == "none" else apply_attack(wm, kind, sev, int(seed))
got = extract(att, PAYLOAD_BITS, int(key))
# ECC decodes the extracted payload and verifies it against the original watermark.
dec, ok = ecc.decode(got)
recovered = ecc.bits_to_text(dec)
verified = ok and np.array_equal(dec, msg_bits)
wrong_key = extract(att, PAYLOAD_BITS, int(key) + 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("PSNR", f"{p:.2f} dB")
c2.metric("SSIM", f"{ssim(img, wm):.4f}")
c3.metric("Bit accuracy (after edit)", f"{1 - ber(payload, got):.1%}")
c4.metric("Ownership", "VERIFIED" if verified else "FAILED", f"recovered '{recovered}'")

col = st.columns(3)
col[0].image(to_rgb(img), caption="Original", width="stretch")
col[1].image(to_rgb(wm), caption=f"Watermarked ({mode})", width="stretch")
col[2].image(to_rgb(att), caption=f"After edit: {kind} {sev}  (PSNR vs original {psnr(img, att):.1f} dB)",
             width="stretch")

rows, cols = grid_shape(img.shape)
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
im0 = ax[0].imshow(s_map.reshape(rows, cols), cmap="magma")
ax[0].set_title("Embedding strength per block")
fig.colorbar(im0, ax=ax[0], fraction=0.046)
diff = np.abs(wm.astype(float) - img.astype(float)).mean(axis=-1) if img.ndim == 3 else np.abs(wm.astype(float) - img)
ax[1].imshow(diff * 10, cmap="gray", vmin=0, vmax=255)
ax[1].set_title("|watermarked - original| x10")
ax[2].imshow(skipped.reshape(rows, cols), cmap="gray_r")
ax[2].set_title(f"Skipped blocks ({skipped.mean():.1%})")
for a in ax:
    a.axis("off")
fig.tight_layout()
st.pyplot(fig)

with st.expander("Details"):
    st.write(f"Wrong-key bit accuracy (should be ~50%): {1 - ber(payload, wrong_key):.1%}")
    st.write(f"Raw bit errors before ECC: {int(ber(payload, got) * PAYLOAD_BITS)} / {PAYLOAD_BITS}")
    st.write(f"Strength range: {s_map.min():.1f} to {s_map.max():.1f} (mean {s_map.mean():.1f})")
