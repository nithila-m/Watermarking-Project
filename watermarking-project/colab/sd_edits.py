r"""Real AI edits with Stable Diffusion, for Google Colab (GPU). Standalone: does NOT need the watermark package.

Edits written for every exported image:
    inpaint_light / _moderate / _severe   SD inpainting of 10 / 20 / 35 % of the image, pasted back
                                          into the original like a photo editor does (ERASURE edits)
    regen_0.15 / regen_0.30               diffusion "regeneration": noise the whole image a little and
                                          let SD redraw it (NOISE edits; the classic watermark-removal attack)
    vae                                   encode + decode through the SD autoencoder only (NOISE edit;
                                          every diffusion edit does this to the whole image)

Fair comparison: the mask, prompt and random seed depend only on the image name and
the edit, never on the watermark config, so uniform and adaptive copies of the
same test image get the same edit.

Colab, in cells (Runtime > Change runtime type > T4 GPU):
    from google.colab import drive; drive.mount('/content/drive')
    !pip -q install diffusers transformers accelerate
    !python /content/drive/MyDrive/wm/sd_edits.py \
        --src /content/drive/MyDrive/wm/colab_export --dst /content/drive/MyDrive/wm/edited --limit 3
    # check a few outputs, then run again without --limit. Re-running skips finished files,
    # so a Colab disconnect only costs the image that was in progress.

Output: <dst>/<edit>/train/<name>.png and <dst>/<edit>/test/<config>/<name>.png
        <dst>/masks/<split>/<name>_<edit>.png  (the inpainting masks, for the report)
"""
import argparse
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

EDITS = {
    "inpaint_light":    {"kind": "inpaint", "frac": 0.10},
    "inpaint_moderate": {"kind": "inpaint", "frac": 0.20},
    "inpaint_severe":   {"kind": "inpaint", "frac": 0.35},
    "regen_0.15":       {"kind": "regen", "strength": 0.15},
    "regen_0.30":       {"kind": "regen", "strength": 0.30},
    "vae":              {"kind": "vae"},
}

# Tried in order until one downloads. Hugging Face model names move sometimes:
# if all fail, search huggingface.co for "stable-diffusion inpainting" and pass --model.
INPAINT_MODELS = [
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
    "stabilityai/stable-diffusion-2-inpainting",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
]

PROMPTS = ["a potted plant", "a wooden bench", "a parked bicycle", "a small dog", "a stack of books",
           "a street lamp", "a flower bed", "a red umbrella", "an empty background", "a cardboard box",
           "a coffee cup on a table", "green grass"]
REGEN_PROMPT = "a high quality photograph"


def seed_for(split, name, edit):
    """Same seed for every config copy of an image, different per image and edit."""
    return zlib.crc32(f"{split}|{name}|{edit}".encode())


def make_mask(size, frac, rng):
    """1-3 rectangles until about `frac` of the image is covered. 255 = repaint."""
    w, h = size
    mask = np.zeros((h, w), np.uint8)
    while mask.mean() / 255 < frac:
        rh = int(rng.integers(h // 6, h // 2))
        rw = int(rng.integers(w // 6, w // 2))
        y0, x0 = int(rng.integers(0, h - rh)), int(rng.integers(0, w - rw))
        mask[y0:y0 + rh, x0:x0 + rw] = 255
    return mask


def load_pipeline(model_ids, device, dtype):
    from diffusers import AutoPipelineForInpainting
    errors = []
    for mid in model_ids:
        for kw in ({"safety_checker": None, "requires_safety_checker": False}, {}):
            try:
                pipe = AutoPipelineForInpainting.from_pretrained(mid, torch_dtype=dtype, **kw)
                print(f"loaded {mid}")
                return pipe.to(device)
            except Exception as e:  # noqa: BLE001  try the next option
                errors.append(f"{mid} {kw or ''}: {type(e).__name__}: {str(e)[:150]}")
    raise SystemExit("could not load any inpainting model:\n  " + "\n  ".join(errors))


def _to_size(img, size):
    return img if img.size == size else img.resize(size, Image.Resampling.LANCZOS)


def inpaint(pipe, img, mask, prompt, gen, steps, guidance):
    out = pipe(prompt=prompt, image=img, mask_image=Image.fromarray(mask), height=img.height,
               width=img.width, num_inference_steps=steps, guidance_scale=guidance,
               generator=gen).images[0]
    out = _to_size(out, img.size)
    # paste back with a soft edge: pixels outside the mask stay exactly as they were
    soft = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(3))
    return Image.composite(out, img, soft)


def regen(pipe, img, strength, gen, steps):
    # the pipeline only runs int(steps * strength) denoising steps; make sure that is >= 3
    steps = max(steps, int(np.ceil(3 / strength)))
    full = Image.new("L", img.size, 255)
    out = pipe(prompt=REGEN_PROMPT, image=img, mask_image=full, height=img.height, width=img.width,
               strength=strength, num_inference_steps=steps, guidance_scale=5.0,
               generator=gen).images[0]
    return _to_size(out, img.size)


def vae_roundtrip(vae, img, device):
    import torch
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
    with torch.no_grad():
        x = x.to(device=device, dtype=vae.dtype)
        y = vae.decode(vae.encode(x).latent_dist.mode()).sample
    y = ((y[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(y)


def find_images(src, only):
    """[(split, rel_path)] for src/train/*.png and src/test/<config>/*.png."""
    items = []
    if only in (None, "train"):
        items += [("train", p.relative_to(src)) for p in sorted((src / "train").glob("*.png"))]
    if only in (None, "test"):
        items += [("test", p.relative_to(src)) for p in sorted((src / "test").glob("*/*.png"))]
    return items


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the colab_export folder (has train/ and/or test/)")
    ap.add_argument("--dst", required=True, help="where edited images go")
    ap.add_argument("--edits", nargs="+", default=list(EDITS), choices=list(EDITS))
    ap.add_argument("--only", choices=["train", "test"], default=None)
    ap.add_argument("--model", default=None, help="inpainting model id (default: try the list)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (quick trial)")
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    items = find_images(src, args.only)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"no PNGs under {src}/train or {src}/test/<config>/")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cpu":
        print("WARNING: no GPU. This will be extremely slow. Colab: Runtime > Change runtime type > T4 GPU")
    pipe = load_pipeline([args.model] if args.model else INPAINT_MODELS, device, dtype)
    pipe.set_progress_bar_config(disable=True)
    vae32 = None
    if "vae" in args.edits:   # fp16 autoencoders can return NaNs; a float32 copy is safe
        import copy
        vae32 = copy.deepcopy(pipe.vae).to(device=device, dtype=torch.float32)

    todo = [(s, rel, e) for s, rel in items for e in args.edits if not (dst / e / rel).exists()]
    print(f"{len(items)} images x {len(args.edits)} edits: {len(todo)} files to make on {device}")
    t0, done = time.time(), 0
    cache = {}
    for split, rel, edit in todo:
        name = rel.stem
        if rel not in cache:
            cache = {rel: Image.open(src / rel).convert("RGB")}
        img = cache[rel]
        cfg = EDITS[edit]
        seed = seed_for(split, name, edit)
        gen = torch.Generator(device=device).manual_seed(seed)
        if cfg["kind"] == "inpaint":
            rng = np.random.default_rng(seed)
            mask = make_mask(img.size, cfg["frac"], rng)
            prompt = PROMPTS[int(rng.integers(len(PROMPTS)))]
            out = inpaint(pipe, img, mask, prompt, gen, args.steps, args.guidance)
            mpath = dst / "masks" / split / f"{name}_{edit}.png"
            if not mpath.exists():
                mpath.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(mpath)
        elif cfg["kind"] == "regen":
            out = regen(pipe, img, cfg["strength"], gen, args.steps)
        else:
            out = vae_roundtrip(vae32, img, device)
        path = dst / edit / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        out.save(path)
        done += 1
        if done % 25 == 0 or done == len(todo):
            rate = (time.time() - t0) / done
            print(f"  {done}/{len(todo)}  {rate:.1f} s/file  ~{rate * (len(todo) - done) / 60:.0f} min left",
                  flush=True)
    print(f"done. Download {dst} and put its contents in data/external/ of the repo.")


if __name__ == "__main__":
    main()
