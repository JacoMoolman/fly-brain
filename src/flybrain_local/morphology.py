"""Cached, measured MaleCNS neurites in the same coordinates as the soma view."""
from concurrent.futures import ThreadPoolExecutor
import threading
import urllib.error
import urllib.request

import numpy as np
import pyarrow.feather as feather

SOURCE = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-precomputed/"


class Morphology:
    def __init__(self, paths, brain):
        self.brain = brain
        self.cache = paths.dataset / "skeletons"
        self.cache.mkdir(exist_ok=True)
        ann = feather.read_table(paths.dataset / "flat-connectome/annotations.feather",
                                 columns=["bodyId", "somaLocation"]).to_pandas()
        ann = ann[ann.bodyId.isin(brain.ids)]
        raw = [np.asarray(v, dtype=np.float64) for v in ann.somaLocation
               if isinstance(v, (list, np.ndarray)) and len(v) == 3]
        raw = np.stack(raw)
        self.center = raw.mean(axis=0)
        self.scale = float(np.percentile(np.abs(raw - self.center), 98)) or 1.0
        # Round-robin annotation groups, shuffled reproducibly within each group.
        # Small overviews include both rare structures and large optic-lobe groups.
        rng = np.random.default_rng(42)
        labels = np.where(brain.neuron_class != "", brain.neuron_class, brain.superclass)
        groups = [rng.permutation(np.flatnonzero(labels == label)) for label in np.unique(labels)]
        self.order = np.asarray([int(group[k]) for k in range(max(map(len, groups)))
                                 for group in groups if k < len(group)], dtype=np.int32)
        self.pool = ThreadPoolExecutor(max_workers=12, thread_name_prefix="morphology")
        self.lock = threading.Lock()

    def _load(self, index, detail):
        body_id = int(self.brain.ids[index])
        path = self.cache / f"{body_id}.bin"
        try:
            if path.exists():
                data = path.read_bytes()
            else:
                with urllib.request.urlopen(SOURCE + str(body_id), timeout=20) as response:
                    data = response.read()
                # Atomic publication so concurrent browser requests never see a partial file.
                with self.lock:
                    temp = path.with_suffix(".tmp")
                    temp.write_bytes(data)
                    temp.replace(path)
            n, e = np.frombuffer(data, dtype="<u4", count=2)
            points = np.frombuffer(data, dtype="<f4", count=int(n) * 3, offset=8).reshape(-1, 3)
            edges = np.frombuffer(data, dtype="<u4", count=int(e) * 2,
                                  offset=8 + int(n) * 12).reshape(-1, 2)
            if not np.isfinite(points).all() or (edges.size and int(edges.max()) >= n):
                raise ValueError("Invalid skeleton coordinates or edge indices")
            # Preserve actual recorded segments, never draw invented soma-to-soma chords.
            # Overview density is sampled; individual-neuron requests retain every edge.
            if detail and len(edges) > detail:
                edges = edges[np.linspace(0, len(edges) - 1, detail, dtype=int)]
            xyz = ((points[edges].astype(np.float64) / 8.0 - self.center) / self.scale)
            rows = np.empty((len(edges), 7), dtype="<f4")
            rows[:, :6] = xyz.reshape(-1, 6)
            rows[:, 6] = index
            return rows, None
        except (OSError, ValueError) as exc:
            return np.empty((0, 7), dtype="<f4"), {"id": str(body_id), "error": str(exc)}

    def batch(self, offset=0, count=32, detail=1024, body_id=None):
        if body_id is not None:
            matches = np.flatnonzero(self.brain.ids == body_id)
            if not len(matches):
                raise ValueError("Neuron ID is not in the retained MaleCNS graph")
            indices = matches
            detail = 0
        else:
            indices = self.order[offset:offset + count]
        results = list(self.pool.map(lambda i: self._load(int(i), detail), indices))
        rows = np.concatenate([r[0] for r in results]) if results else np.empty((0, 7), dtype="<f4")
        failures = [r[1] for r in results if r[1] is not None]
        return rows.tobytes(), len(indices), failures
