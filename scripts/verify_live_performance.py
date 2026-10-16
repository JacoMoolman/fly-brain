import base64,json,time,urllib.request
import numpy as np
from pathlib import Path
samples=[]
for _ in range(30):
    s=json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/state'))
    if 'brain' not in s:
        time.sleep(.2); continue
    assert s['brain']['backend'].startswith('CUDA'), s
    data=np.frombuffer(base64.b64decode(s['brain']['activity_binary']), dtype='<u4').reshape(-1,2)
    assert data[:,1].sum() == s['brain']['frame_spikes']
    assert (data[:,0] < s['neurons']).all()
    samples.append([s['brain']['step_wall_ms'],s['loop']['body_wall_ms'],s['loop']['wall_ms'],s['loop']['realtime_factor']])
    time.sleep(.2)
median=np.median(samples,axis=0)
result={'backend':s['brain']['backend'],'median_brain_ms':float(median[0]),'median_body_ms':float(median[1]),'median_loop_ms':float(median[2]),'median_realtime_factor':float(median[3]),'activity_neurons_in_last_frame':len(data),'binary_activity':'All spike counts match frame total'}
path=Path(r'J:\models\fruit fly\results\verification\cuda-live-performance.json')
path.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
