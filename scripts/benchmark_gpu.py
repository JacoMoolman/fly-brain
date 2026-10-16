import time
import numpy as np
from flybrain_local.brain import MaleCNSBrain
from flybrain_local.paths import ProjectPaths
import warp as wp
b=MaleCNSBrain(ProjectPaths.discover().prepared/'brain_graph.npz',backend='cuda')
lum=np.full(len(b.retina),.35)
for ms in [3,15,30]:
    b.step(lum,ms)
    times=[]
    for _ in range(10):
        t=time.perf_counter(); b.step(lum,ms); times.append(time.perf_counter()-t)
    print('duration_ms',ms,'mean_wall_ms',np.mean(times)*1000,flush=True)
g=b._gpu
samples=[]
for _ in range(30):
    a=time.perf_counter(); g.drive.assign(b.drive); c=time.perf_counter()
    wp.capture_launch(g.graphs[30]); d=time.perf_counter()
    counts=g.counts.numpy(); e=time.perf_counter()
    active=g.active_count.numpy(); f=time.perf_counter()
    samples.append([c-a,d-c,e-d,f-e])
print('drive/launch/counts/active ms',np.mean(samples,axis=0)*1000,flush=True)
