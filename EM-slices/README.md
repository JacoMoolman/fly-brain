# MaleCNS electron microscope slices

Real grayscale electron microscope (EM) images of the male *Drosophila melanogaster* central
nervous system, from the volume that was used to reconstruct the MaleCNS connectome.

## Attribution (required, CC BY 4.0)

Data: **MaleCNS v1.0**, a collaboration between FlyEM (HHMI Janelia Research Campus), the
University of Cambridge (Department of Zoology), the MRC Laboratory of Molecular Biology and
Google Research. Licensed under Creative Commons Attribution 4.0 (CC BY 4.0).

- Project and download page: https://male-cns.janelia.org/download/
- Media gallery: https://male-cns.janelia.org/media/
- Paper: "Sexual dimorphism in the complete Drosophila male central nervous system connectome",
  Cell (2026), https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6
- Interactive viewer: https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json

Suggested credit line for the blog: "EM data: MaleCNS v1.0, FlyEM / HHMI Janelia, University of
Cambridge, MRC LMB and Google Research, CC BY 4.0. Slices cropped, resampled and animated."

## Source

- Volume: `precomputed://gs://flyem-male-cns/em/em-clahe-jpeg` (contrast-enhanced with CLAHE,
  JPEG-encoded, uint8). Full size 94088 x 78317 x 134576 voxels, 8 nm isotropic at full resolution.
- Higher mips are the same data downsampled by 2 per level (mip 6 = 512 nm voxels).
- The uncontrasted, aligned volume is at `gs://flyem_cns_z0720_07m_dvidcoords_n5` (N5, 8 nm).
- Read with the `cloud-volume` Python package from the official download page. The data are stored
  in sharded precomputed format, so individual chunks are not separate files.

## What is here

| Folder | Mip / pixel size | Region (voxels at that mip) | Region (nm) | Slices |
|---|---|---|---|---|
| `raw_images/whole_brain_sweep_512nm` | 6 / 512 nm | x 24-1470, y 16-868, z 220-740 | x 12288-752640, y 8192-444416, z 112640-378880 | 520, 512 nm apart |
| `raw_images/neuropil_closeup_8nm` | 0 / 8 nm | x 47000-48536, y 25000-26536, z 33280-33430 | x 376000-388288, y 200000-212288, z 266240-267440 | 150, 8 nm apart |

- Each PNG is one z slice (an x-y plane). Columns are x, rows are y. File names are
  `<index>_z<z voxel>.png`; the index increases with z.
- The sweep covers the whole brain (and the connected head tissue) from front to back, so you can
  see the brain appear, widen to full size around slice 250 and shrink away again.
- The close-up is a 12.3 um square of central-brain neuropil at the original 8 nm resolution:
  cell membranes, mitochondria, synaptic vesicles. 150 slices span 1.2 um of depth.
- Each folder has a `manifest.json` with the exact source, ranges and units.
- `scout_overview.png` is a coarse overview of the whole volume used to choose the regions. The
  brain lies at roughly z = 110-380 um; the ventral nerve cord (the "spinal cord" of the body)
  lies further along z and is not in these sets.

## Videos (`video/`)

Looping MP4s (play forward then backward so they loop seamlessly), with a slice label, a scale bar
and the credit burned in. Made by `scripts/03_make_video.py`.

- `whole_brain_sweep_512nm.mp4`: 1446 x 852, 30 fps, about 35 s, 59 MB.
- `neuropil_closeup_8nm.mp4`: 1080 x 1080, 24 fps, about 12 s, 24 MB.

## Scripts

Settings are variables at the top of each file, no command-line arguments.

1. `scripts/01_scout_brain.py`: coarse scan of the whole volume, finds where the tissue is.
2. `scripts/02_download_slices.py`: estimates the download size, then downloads and writes PNGs.
3. `scripts/03_make_video.py`: builds the MP4s from the PNGs.

Python packages used: `cloud-volume`, `numpy`, `pillow`, `imageio`, `imageio-ffmpeg`.
