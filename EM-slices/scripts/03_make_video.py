"""Turn the downloaded PNG slices into looping MP4 animations (forward then back, so the loop is seamless).

Edit the settings below, then run:  python 03_make_video.py
Needs imageio and imageio-ffmpeg (ffmpeg is bundled with imageio-ffmpeg).
"""
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_images"
OUT = ROOT / "video"

# name of a folder in raw_images, frames per second, output width in pixels (height keeps the aspect, rounded to even)
VIDEOS = [
    {"folder": "whole_brain_sweep_512nm", "fps": 30, "width": 1446, "scale_bar_um": 100},
    {"folder": "neuropil_closeup_8nm", "fps": 24, "width": 1080, "scale_bar_um": 2},
]
CREDIT = "MaleCNS v1.0 EM | FlyEM, HHMI Janelia + Cambridge + MRC LMB + Google Research | CC BY 4.0"
LOOP_FORWARD_AND_BACK = True
CRF = 28   # lower is higher quality and a bigger file (EM texture is noisy, so files get big fast)


def label_frame(img, text, scale_bar_px, bar_label):
    img = img.convert("RGBA")
    font = ImageFont.load_default(size=max(12, img.width // 70))
    pad = max(8, img.width // 120)
    strip = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shade = ImageDraw.Draw(strip)
    shade.rectangle([0, 0, img.width, font.size + pad * 2], fill=(0, 0, 0, 170))
    shade.rectangle([0, img.height - font.size * 2 - pad * 3 - 6, img.width, img.height], fill=(0, 0, 0, 170))
    img = Image.alpha_composite(img, strip)
    draw = ImageDraw.Draw(img)
    draw.text((pad, pad), text, fill=(255, 235, 120), font=font)
    draw.text((pad, img.height - pad - font.size), CREDIT, fill=(230, 230, 230), font=font)
    y = img.height - pad * 2 - font.size * 2
    draw.rectangle([pad, y, pad + scale_bar_px, y + 4], fill=(255, 255, 255))
    draw.text((pad + scale_bar_px + pad, y - font.size // 2 + 2), bar_label, fill=(255, 255, 255), font=font)
    return img.convert("RGB")


OUT.mkdir(exist_ok=True)
for spec in VIDEOS:
    folder = RAW / spec["folder"]
    manifest = json.loads((folder / "manifest.json").read_text())
    files = sorted(folder.glob("*.png"))
    nm_per_px = manifest["voxel_size_nm"][0]
    first = Image.open(files[0])
    width = spec["width"]
    height = round(first.height * width / first.width / 2) * 2
    width = width // 2 * 2
    shrink = width / first.width
    bar_px = round(spec["scale_bar_um"] * 1000 / nm_per_px * shrink)
    frames = []
    for i, path in enumerate(files):
        img = Image.open(path).resize((width, height), Image.LANCZOS)
        z_um = (manifest["z_range_nm"][0] + i * nm_per_px) / 1000
        text = f"z = {z_um:.1f} um   slice {i + 1}/{len(files)}"
        frames.append(np.asarray(label_frame(img, text, bar_px, f"{spec['scale_bar_um']} um")))
    sequence = frames + frames[-2:0:-1] if LOOP_FORWARD_AND_BACK else frames
    target = OUT / f"{spec['folder']}.mp4"
    imageio.mimsave(target, sequence, fps=spec["fps"], codec="libx264", macro_block_size=2,
                    output_params=["-crf", str(CRF), "-movflags", "+faststart"])
    print(f"wrote {target}  {len(sequence)} frames  {target.stat().st_size / 1e6:.1f} MB")
