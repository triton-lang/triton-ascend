from pathlib import Path
import json,random,time,statistics
import torch
import native_guard
from access_guard import StorageView,conv_certificate_check

torch.set_num_threads(1)
rng=random.Random(42)
def call_args(t,D,N,W,BD,sx=None,sy=None):
 return tuple(t[k] for k in ['x','cache','weight','bias','y'])+(D,N,W,BD,(D+BD-1)//BD,N,sx or D,sy or D)
def metadata(t):
 return {k:StorageView(v.untyped_storage().data_ptr(),v.untyped_storage().nbytes(),v.data_ptr(),v.element_size(),str(v.device),str(v.dtype)) for k,v in t.items()}
count=0
for i in range(1000):
 D=rng.choice([1,3,17,127,257]);N=rng.choice([1,3,5]);W=rng.choice([2,4,8,16,32,64]);BD=rng.choice([8,32,128,256]);dt=rng.choice([torch.float16,torch.float32])
 t={k:torch.empty(n,dtype=dt) for k,n in dict(x=N*D,cache=N*D*W,weight=D*W,bias=D,y=N*D).items()}
 variant=i%5
 if variant==1:t['cache']=t['cache'][1:]
 if variant==2:t['weight']=t['cache']
 if variant==3:t['y']=t['x']
 if variant==4:
  storage=torch.empty(sum(v.numel() for v in t.values()),dtype=dt);offset=0
  for k,v in list(t.items()):t[k]=storage[offset:offset+v.numel()];offset+=v.numel()
 args=call_args(t,D,N,W,BD)
 expected=conv_certificate_check(metadata(t),D=D,N=N,W=W,BD=BD,grid=(args[-4],N),stride_x_n=D,stride_y_n=D).fast
 assert native_guard.check_cpu_test(*args)==expected,(i,variant,expected)
 assert native_guard.check(*args) is False # production path rejects CPU
 count+=1
# B16/D2048 f32 metadata microbenchmark: no kernel launches or tensor contents.
D,N,W,BD=2048,16,4,1024
t={k:torch.empty(n,dtype=torch.float32) for k,n in dict(x=N*D,cache=N*D*W,weight=D*W,bias=D,y=N*D).items()}
args=call_args(t,D,N,W,BD)
for _ in range(100):assert native_guard.check_cpu_test(*args)
rounds=[]
for _ in range(5):
 begin=time.perf_counter_ns()
 for i in range(10000):native_guard.check_cpu_test(*args)
 rounds.append((time.perf_counter_ns()-begin)/10000/1000)
result=dict(real_tensor_cases_passed=count,production_CPU_rejections=count,host_cpu_metadata_guard_us=rounds,median_host_cpu_metadata_guard_us=statistics.median(rounds),device_launches=0,scope='Native metadata guard on real CPU tensors; not an NPU end-to-end launch benchmark')
Path(__file__).with_name('native_result.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
