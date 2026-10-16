"""Verify the live morphology endpoint against cached published geometry."""
import json
import urllib.request
import urllib.error
import numpy as np
import pyarrow.feather as feather
from flybrain_local.paths import ProjectPaths
paths = ProjectPaths.discover()
base = 'http://127.0.0.1:8765/api/morphology'
with urllib.request.urlopen(base + '?body_id=12781') as response:
    assert response.headers['X-Neurons-Missing'] == '0'
    rows = np.frombuffer(response.read(), dtype='<f4').reshape(-1, 7)
source = (paths.dataset / 'skeletons/12781.bin').read_bytes()
n, e = np.frombuffer(source, dtype='<u4', count=2)
points = np.frombuffer(source, dtype='<f4', count=int(n)*3, offset=8).reshape(-1, 3)
edges = np.frombuffer(source, dtype='<u4', count=int(e)*2, offset=8+int(n)*12).reshape(-1, 2)
graph = np.load(paths.prepared / 'brain_graph.npz')
ann = feather.read_table(paths.dataset / 'flat-connectome/annotations.feather', columns=['bodyId', 'somaLocation']).to_pandas()
ann = ann[ann.bodyId.isin(graph['ids'])]
raw = np.stack([np.asarray(v, dtype=np.float64) for v in ann.somaLocation if isinstance(v, (list, np.ndarray)) and len(v) == 3])
center = raw.mean(axis=0)
scale = float(np.percentile(np.abs(raw-center), 98))
expected = ((points[edges].astype(np.float64)/8-center)/scale).reshape(-1,6)
assert len(rows) == int(e)
assert np.isfinite(rows).all()
assert np.allclose(rows[:,:6], expected, atol=1e-6)
assert np.all(rows[:,6] == int(np.flatnonzero(graph['ids'] == 12781)[0]))
with urllib.request.urlopen(base + '?count=4&detail=32') as response:
    overview = np.frombuffer(response.read(), dtype='<f4').reshape(-1,7)
    assert len(overview) <= 128
    assert np.isfinite(overview).all()
try:
    urllib.request.urlopen(base + '?body_id=0')
    raise AssertionError('Unknown ID accepted')
except urllib.error.HTTPError as error:
    assert error.code == 400
print(json.dumps({'full_neuron_edges': int(e), 'coordinates': 'match published skeleton', 'overview_and_invalid_id': 'passed'}))
