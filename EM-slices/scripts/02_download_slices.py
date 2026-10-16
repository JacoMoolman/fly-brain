"""Download consecutive grayscale EM slices from the MaleCNS volume as PNGs.

Edit the settings below, then run:  python 02_download_slices.py
Run once with DRY_RUN = True to see the estimated download size, then set it to False.

Data: MaleCNS v1.0 (FlyEM / HHMI Janelia, Cambridge, MRC LMB, Google Research), CC BY 4.0.
Volume: precomputed://gs://flyem-male-cns/em/em-clahe-jpeg  (contrast-enhanced, JPEG-encoded,
8 nm isotropic at mip 0, axes x, y, z, chunk size 64 x 64 x 64 voxels).
"""
import json
import time
from math import ceil
from pathlib import Path

import numpy as np
import requests
from cloudvolume import CloudVolume
from PIL import Image

DRY_RUN = False
MAX_DOWNLOAD_MB = 1500

SOURCE = "precomputed://gs://flyem-male-cns/em/em-clahe-jpeg"
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "raw_images"
SAMPLE_CHUNKS_PER_SIDE = 6   # size estimate fetches a 6 x 6 x 1 chunk block per job

# Coordinates are in voxels of the chosen mip (mip 0 = 8 nm, each mip doubles the voxel size).
JOBS = [
    {   # Whole-brain sweep through the head: every 512 nm section from the front to the back of the brain.
        "name": "whole_brain_sweep_512nm",
        "mip": 6,
        "x": (24, 1470), "y": (16, 868), "z": (220, 740),
        "note": "Brain cross-sections (x-y planes) stepping through z, 512 nm per pixel and per slice.",
    },
    {   # Native resolution close-up: membranes, mitochondria, vesicles. 12.3 x 12.3 um, 1.2 um deep.
        "name": "neuropil_closeup_8nm",
        "mip": 0,
        "x": (47000, 48536), "y": (25000, 26536), "z": (33280, 33430),
        "note": "Native 8 nm resolution, 150 consecutive slices of central-brain neuropil.",
    },
]


class DownloadLimitExceeded(Exception):
    pass


# The volume is stored in neuroglancer shards, so individual chunk sizes cannot be listed.
# Every HTTP response is counted instead: a small sample block is fetched to extrapolate
# the size, and the real download aborts if it passes MAX_DOWNLOAD_MB.
bytes_received = [0]
_original_send = requests.Session.send


def _counting_send(self, request, **kwargs):
    response = _original_send(self, request, **kwargs)
    bytes_received[0] += int(response.headers.get("Content-Length", 0))
    if not DRY_RUN and bytes_received[0] > MAX_DOWNLOAD_MB * 1e6 * 1.2:
        raise DownloadLimitExceeded(f"received more than {MAX_DOWNLOAD_MB} MB")
    return response


requests.Session.send = _counting_send


def chunk_count(vol, job):
    (x0, x1), (y0, y1), (z0, z1) = job["x"], job["y"], job["z"]
    size = vol.scale["chunk_sizes"][0]
    return (ceil(x1 / size[0]) - x0 // size[0]) * (ceil(y1 / size[1]) - y0 // size[1]) * \
           (ceil(z1 / size[2]) - z0 // size[2])


def estimate_mb(vol, job):
    n = chunk_count(vol, job)
    size = vol.scale["chunk_sizes"][0]
    side = SAMPLE_CHUNKS_PER_SIDE
    # Sample around the middle of the region, where tissue is, not at its empty edge.
    mid = [(job[a][0] + job[a][1]) // 2 for a in "xyz"]
    x0 = max(0, mid[0] // size[0] * size[0] - side // 2 * size[0])
    y0 = max(0, mid[1] // size[1] * size[1] - side // 2 * size[1])
    z0 = mid[2] // size[2] * size[2]
    before = bytes_received[0]
    vol[x0:x0 + side * size[0], y0:y0 + side * size[1], z0:z0 + size[2], 0]
    sampled = bytes_received[0] - before
    per_chunk = sampled / (side * side)
    return n, per_chunk, n * per_chunk / 1e6


def download(vol, job):
    (x0, x1), (y0, y1), (z0, z1) = job["x"], job["y"], job["z"]
    out = OUTPUT_ROOT / job["name"]
    out.mkdir(parents=True, exist_ok=True)
    depth = vol.scale["chunk_sizes"][0][2]
    count = 0
    for batch_start in range(z0 // depth * depth, z1, depth):
        lo, hi = max(batch_start, z0), min(batch_start + depth, z1)
        block = np.asarray(vol[x0:x1, y0:y1, lo:hi, 0])[:, :, :, 0]
        for i in range(hi - lo):
            plane = block[:, :, i].T
            Image.fromarray(plane, mode="L").save(out / f"{count:04d}_z{lo + i}.png", optimize=True)
            count += 1
        print(f"  {job['name']}: {count}/{z1 - z0} slices written", flush=True)
    res = vol.resolution.tolist()
    manifest = {
        "source": SOURCE, "license": "CC BY 4.0", "mip": job["mip"],
        "voxel_size_nm": res, "x_range_voxels": [x0, x1], "y_range_voxels": [y0, y1],
        "z_range_voxels": [z0, z1], "slices": count, "note": job["note"],
        "x_range_nm": [x0 * res[0], x1 * res[0]], "y_range_nm": [y0 * res[1], y1 * res[1]],
        "z_range_nm": [z0 * res[2], z1 * res[2]],
        "image_axes": "PNG columns = x, rows = y; one PNG per z slice, file index increases with z",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


total_mb = 0.0
volumes = []
for job in JOBS:
    vol = CloudVolume(SOURCE, use_https=True, mip=job["mip"], progress=False, fill_missing=True)
    volumes.append(vol)
    n, per_chunk, mb = estimate_mb(vol, job)
    total_mb += mb
    print(f"{job['name']}: {n} chunks, about {per_chunk / 1000:.1f} KB per chunk -> about {mb:.0f} MB")
print(f"Estimated total download: about {total_mb:.0f} MB (limit {MAX_DOWNLOAD_MB} MB)")

if DRY_RUN:
    print("DRY_RUN is True, nothing downloaded.")
elif total_mb > MAX_DOWNLOAD_MB:
    print("Estimate is over MAX_DOWNLOAD_MB, nothing downloaded.")
else:
    started = time.time()
    for job, vol in zip(JOBS, volumes):
        print("downloading", job["name"])
        download(vol, job)
    print(f"done in {time.time() - started:.0f} s, files in {OUTPUT_ROOT}")
