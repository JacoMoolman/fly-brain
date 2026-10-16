"""Scout the MaleCNS EM volume at very low resolution to find where the tissue is.

Edit the settings below, then run:  python 01_scout_brain.py
Writes a contact sheet PNG of coarse slices through the whole volume, so you can pick
a region for 02_download_slices.py. Downloads only a few MB.
"""
from pathlib import Path

import numpy as np
from cloudvolume import CloudVolume
from PIL import Image, ImageDraw

SOURCE = "precomputed://gs://flyem-male-cns/em/em-clahe-jpeg"
MIP = 8                      # 8 = 2048 nm voxels (volume is 368 x 306 x 526 there)
Z_VALUES = list(range(40, 526, 45))   # coarse z positions to sample at that mip
OUTPUT = Path(__file__).resolve().parent.parent / "scout_overview.png"

vol = CloudVolume(SOURCE, use_https=True, mip=MIP, progress=False, fill_missing=True)
size = vol.bounds.size3()
print("volume at this mip (x, y, z):", size, "resolution nm:", vol.resolution)

tiles = []
for z in Z_VALUES:
    plane = np.asarray(vol[:, :, z, 0]).squeeze()
    tissue = float((plane > 5).mean())
    print(f"z={z}: mean {plane.mean():.1f}  std {plane.std():.1f}  non-empty fraction {tissue:.2f}")
    # cloud-volume arrays are (x, y); transpose so the image is x across, y down
    img = Image.fromarray(plane.T.astype(np.uint8)).convert("RGB")
    ImageDraw.Draw(img).text((4, 4), f"z={z}", fill=(255, 255, 0))
    tiles.append(img)

cols = 4
rows = (len(tiles) + cols - 1) // cols
w, h = tiles[0].size
sheet = Image.new("RGB", (cols * w, rows * h))
for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % cols) * w, (i // cols) * h))
sheet.save(OUTPUT)
print("wrote", OUTPUT, sheet.size)

# Bounding box of the brain (the head-end of the volume), found from a coarse block of slices.
BRAIN_Z_RANGE = (30, 200)    # coarse z range that holds the brain in the contact sheet
block = np.asarray(vol[:, :, BRAIN_Z_RANGE[0]:BRAIN_Z_RANGE[1], 0]).squeeze() > 5
scale = 2 ** MIP             # coarse voxel -> 8 nm voxel
for axis, name, offset in ((0, "x", 0), (1, "y", 0), (2, "z", BRAIN_Z_RANGE[0])):
    other = tuple(a for a in range(3) if a != axis)
    profile = block.mean(axis=other)
    keep = np.flatnonzero(profile > 0.02)
    lo, hi = int(keep[0]) + offset, int(keep[-1]) + 1 + offset
    print(f"brain {name}: coarse {lo}-{hi}  ->  8 nm voxels {lo * scale}-{hi * scale}")
