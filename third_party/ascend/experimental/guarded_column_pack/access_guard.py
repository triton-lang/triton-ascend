"""Host-side access-envelope versioning prototype; no device operations.

The affine footprint certificate is caller/compiler-supplied. It must be tied
 to the exact analyzed kernel; this module does not infer footprints from Python.
 Unknown objects/strides/devices/dtypes fail closed. No decisions are cached.
"""
from dataclasses import dataclass
from typing import Mapping

I32_MAX=(1<<31)-1
I64_MAX=(1<<63)-1
PTR_MAX=(1<<64)-1

@dataclass(frozen=True)
class StorageView:
    allocation_start:int
    allocation_bytes:int
    pointer:int
    itemsize:int
    device:str
    dtype:str

@dataclass(frozen=True)
class Footprint:
    name:str
    rows:int
    columns:int
    stride:int  # elements between rows
    offset:int=0
    writable:bool=False

@dataclass(frozen=True)
class Decision:
    fast:bool
    reason:str


def check_envelopes(views:Mapping[str,StorageView], accesses:tuple[Footprint,...])->Decision:
    """Conservative disjoint byte envelopes, including newly read bytes."""
    spans=[];devices=set()
    for a in accesses:
        v=views.get(a.name)
        if v is None:return Decision(False,'missing metadata: '+a.name)
        if min(a.rows,a.columns,a.stride,v.itemsize)<=0 or a.offset<0:
            return Decision(False,'unsupported extent/stride: '+a.name)
        if a.stride<a.columns:return Decision(False,'overlapping rows: '+a.name)
        if min(v.allocation_start,v.allocation_bytes,v.pointer)<0 or v.allocation_start==0:
            return Decision(False,'invalid allocation: '+a.name)
        allocation_end=v.allocation_start+v.allocation_bytes
        last=a.offset+(a.rows-1)*a.stride+a.columns
        if last>I64_MAX//v.itemsize or allocation_end>PTR_MAX:
            return Decision(False,'address arithmetic overflow: '+a.name)
        lo=v.pointer+a.offset*v.itemsize;hi=v.pointer+last*v.itemsize
        if not v.allocation_start<=v.pointer<allocation_end or not v.allocation_start<=lo<hi<=allocation_end:
            return Decision(False,'outside allocation: '+a.name)
        if v.pointer%v.itemsize:return Decision(False,'misaligned element: '+a.name)
        devices.add(v.device);spans.append((lo,hi,a))
    if len(devices)!=1:return Decision(False,'different devices')
    for i,(lo,hi,a) in enumerate(spans):
        for low,high,b in spans[i+1:]:
            if (a.writable or b.writable) and max(lo,low)<min(hi,high):
                return Decision(False,'overlap: '+a.name+'/'+b.name)
    return Decision(True,'bounded disjoint access envelopes')


def tensor_metadata(tensors):
    # Restrict to actual Torch tensors: arbitrary data_ptr objects do not expose
    # trustworthy storage bounds. This queries host metadata, not tensor values.
    import torch
    views={}
    try:
        for name,t in tensors.items():
            if type(t) is not torch.Tensor or t.layout!=torch.strided or t.device.type!='npu':
                return None
            if t.is_conj() or t.is_neg():return None
            storage=t.untyped_storage()
            views[name]=StorageView(storage.data_ptr(),storage.nbytes(),t.data_ptr(),t.element_size(),str(t.device),str(t.dtype))
    except (AttributeError,RuntimeError,TypeError):return None
    return views


def conv_certificate_check(views,*,D,N,W,BD,grid,stride_x_n,stride_y_n,stride_x_d=1,stride_y_d=1,chunk_offset=0):
    """Certificate for frozen original depthwise update affine accesses.

    Program (p,q) owns channel rows p*BD+[0,BD), masked by d<D, batch q.
    Full cache rows are also written in the original, with no synchronization.
    This certificate is not valid for arbitrary kernels with these parameters.
    """
    if min(D,N,W,BD)<=0 or W not in (2,4,8,16,32,64) or BD&(BD-1):
        return Decision(False,'unsupported dimensions')
    if grid!=((D+BD-1)//BD,N) or grid[0]*grid[1]>65535 or chunk_offset!=0:
        return Decision(False,'unproved program partition')
    if stride_x_d!=1 or stride_y_d!=1 or min(stride_x_n,stride_y_n)<D:
        return Decision(False,'unsupported x/y layout')
    # The actual channel-offset multiply is i32; masked lanes still compute it.
    if (grid[0]*BD-1)*W>I32_MAX or N*D*W>I64_MAX:
        return Decision(False,'index arithmetic overflow')
    if set(views)!={'x','cache','weight','bias','y'}:
        return Decision(False,'unsupported optional arguments')
    types={v.dtype for v in views.values()};sizes={v.itemsize for v in views.values()}
    if (types,sizes) not in [({'torch.float16'},{2}),({'torch.float32'},{4})]:
        return Decision(False,'unsupported ABI dtype')
    accesses=(Footprint('x',N,D,stride_x_n),Footprint('cache',N*D,W,W,writable=True),
              Footprint('weight',D,W,W),Footprint('bias',1,D,D),Footprint('y',N,D,stride_y_n,writable=True))
    return check_envelopes(views,accesses)


def guarded_call(fast,fallback,tensors,**launch):
    views=tensor_metadata(tensors)
    decision=Decision(False,'unknown tensor storage') if views is None else conv_certificate_check(views,**launch)
    return (fast if decision.fast else fallback)()
