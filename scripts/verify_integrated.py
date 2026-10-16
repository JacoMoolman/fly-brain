"""Regression checks for transforms, manual commands and paused reset."""
import json
import time
from pathlib import Path
import numpy as np
from flybrain_local.runtime import IntegratedRuntime


def wait_for(runtime, predicate, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = runtime.state()
        if state['status'] == 'error':
            raise RuntimeError(state['error'])
        if predicate(state):
            return state
        time.sleep(0.05)
    raise AssertionError('Timed out waiting for runtime state')


runtime = IntegratedRuntime()
runtime.start()
results = {}
try:
    state = wait_for(runtime, lambda s: s.get('loop', {}).get('iteration', 0) >= 30)
    names = [s['name'] for s in runtime.skeleton()['segments']]
    positions = np.array(state['body']['pose']['positions'])
    assert np.isfinite(positions).all()
    assert np.linalg.norm(positions[names.index('c_head')] - positions[names.index('l_eye')]) < 0.5
    assert np.linalg.norm(positions[names.index('c_head')] - positions[names.index('rh_tarsus5')]) > 0.5
    results['head_transform'] = 'Head near eyes, separate from hind foot'
    for stimulus in ('forward', 'left', 'right', 'loom', 'backward'):
        runtime.control(stimulus=stimulus)
        state = wait_for(runtime, lambda s: s.get('control', {}).get('stimulus') == stimulus, 15)
        assert np.isfinite([state['motor']['left'], state['motor']['right']]).all()
        results[stimulus] = [state['motor']['left'], state['motor']['right']]
    runtime.control(stimulus='none')
    assert runtime.state()['control']['stimulus'] is None
    runtime.control(paused=True)
    assert runtime.state()['status'] == 'paused'
    runtime.control(reset=True)
    state = wait_for(runtime, lambda s: s.get('loop', {}).get('iteration') == 0)
    assert state['status'] == 'paused'
    assert state['brain']['simulated_ms'] == 0
    assert state['control']['stimulus'] is None
    runtime.control(paused=False)
    wait_for(runtime, lambda s: s.get('loop', {}).get('iteration', 0) >= 2)
    results['pause_reset_resume'] = 'passed'
finally:
    runtime.stop()
output = Path(r'J:\models\fruit fly\results\verification')
output.mkdir(parents=True, exist_ok=True)
(output / 'integrated-verification.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
