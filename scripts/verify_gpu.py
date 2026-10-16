"""CUDA parity including delay wraparound, high spike counts and activation."""
from types import SimpleNamespace
import math
import numpy as np
from flybrain_local.brain import _advance
from flybrain_local.gpu_brain import CUDABrain

def fixture():
    n=5500
    ptr=np.arange(n+1,dtype=np.int64)*2
    post=np.column_stack(((np.arange(n)+1)%n, 4500+np.arange(n)%1000)).astype(np.int32).ravel()
    weight=np.tile(np.array([0.25,-0.125],np.float32),n)
    initial=np.arange(4500,dtype=np.int32)
    active=np.zeros(n,np.int32); active[:len(initial)]=initial
    flag=np.zeros(n,np.uint8); flag[initial]=1
    drive=np.zeros(n,np.float32); drive[initial]=34
    return SimpleNamespace(n=n,ptr=ptr,post=post,weight=weight,voltage=np.full(n,-52,np.float32),
      conductance=np.zeros(n,np.float32),refractory=np.zeros(n,np.int16),drive=drive,
      active=active,active_flag=flag,active_count=np.array([len(initial)],np.int32),
      counts=np.zeros(n,np.int32),queue=np.zeros((19,n),np.int32),queue_count=np.zeros(19,np.int32),cursor=0)

cpu=fixture(); gpu=fixture(); backend=CUDABrain(gpu)
for steps in [17,40,19,100,31,23]:
    cpu.counts.fill(0)
    cpu.cursor=_advance(cpu.ptr,cpu.post,cpu.weight,cpu.voltage,cpu.conductance,cpu.refractory,
      cpu.drive,cpu.queue,cpu.queue_count,cpu.cursor,steps,.1,cpu.counts,cpu.active,cpu.active_flag,cpu.active_count)
    gpu.cursor=backend.advance(gpu,steps); backend.sync_state(gpu)
    assert np.array_equal(cpu.counts,gpu.counts), (steps, np.sum(cpu.counts),np.sum(gpu.counts))
    assert np.array_equal(cpu.refractory,gpu.refractory)
    assert np.array_equal(cpu.active_count,gpu.active_count)
    assert np.allclose(cpu.voltage,gpu.voltage,atol=0.0002,rtol=0), np.max(np.abs(cpu.voltage-gpu.voltage))
    assert np.allclose(cpu.conductance,gpu.conductance,atol=0.0002,rtol=0)
    print('parity steps',steps,'spikes',int(cpu.counts.sum()),'active',int(cpu.active_count[0]),flush=True)
fresh=fixture(); backend.reset(fresh)
backend.sync_state(fresh)
assert np.all(fresh.voltage == -52) and np.all(fresh.refractory == 0)
print('CUDA timing, delayed spikes, activation, high-count queue and reset parity passed.')
