"""Bounded interpretation of the candidate's verified optimized scalar entry.

DMA fields are checked against CPU compiler probes, not performance labels.
No physical GM addresses or tensor values are evaluated.
"""
import math
import re
from backend_cost_model import Unsupported, parse_module, split_top, type_and_value
from vv_mixed_model import MixedSchedule


def entry_abi(text, name):
    m = re.search(r'^define [^\n]*@' + re.escape(name) + r'\((.*)\) [^\n]*\{', text, re.M)
    if not m: raise Unsupported('optimized entry definition missing')
    return [type_and_value(v)[0] for v in split_top(m[1]) if v]


def validate_entry(raw, optimized, name):
    if entry_abi(raw, name) != entry_abi(optimized, name):
        raise Unsupported('optimized entry ABI changed')
    if parse_module(optimized)[name].vector:
        raise Unsupported('optimized entry is not a scalar AIV entry')


def decode_dma(name, args):
    if name == 'llvm.hivm.MOV.UB.TO.UB.v310':
        if len(args) != 3 or type(args[2]) is not int or not 0 <= args[2] < 2**64:
            raise Unsupported('UB DMA control word is not static unsigned i64')
        count = args[2] & 0xffff
        length = (args[2] >> 16) & 0xffff
        if not count or not length: raise Unsupported('zero UB DMA burst is not modeled')
        return 'ub_copy', count * length * 32
    if len(args) != 4 or any(type(v) is not int or v < 0 or v >= 2**64 for v in args[2:]):
        raise Unsupported('DMA control words are not static unsigned i64')
    first, second = args[2:]
    if name in ('llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV', 'llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f16.DV',
                'llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.bf16.DV', 'llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.u8.DV',
                'llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.s8.DV', 'llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.u32.DV',
                'llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV'):
        load = '.OUT.TO.UB.' in name
        mask = (0xfff << 4) | (0x1fffff << 25) | ((1 << 58) if load else 0)
        if first & ~mask: raise Unsupported('unsupported DMA padding/control flags')
        count = (first >> 4) & 0xfff
        length = (first >> 25) & 0x1fffff
        if not count or not length: raise Unsupported('zero DMA burst is not modeled')
        return ('load' if load else 'store'), count * length
    m = re.fullmatch(r'llvm.hivm.ND.DMA.OUT.TO.UB.b(8|16|32)', name)
    if m:
        # Compiler header __CCE__NDDMA_OUT_TO_UB: bit 56 selects padding
        # value mode. With all padding extents zero it does not add traffic.
        if first & 15 or second & ~((1 << 40) - 1 | (1 << 56)):
            raise Unsupported('unsupported NDDMA control flags')
        dims = [(first >> s) & 0xfffff for s in (4, 24, 44)] + [(second >> s) & 0xfffff for s in (0, 20)]
        if not all(dims): raise Unsupported('zero NDDMA extent is not modeled')
        return 'load', math.prod(dims) * int(m[1]) // 8
    raise Unsupported('unmodelled lowered DMA instruction: ' + name)


def lowered_intrinsic(schedule, name, args):
    if name == 'llvm.hivm.MOV.UB.TO.UB.v310':
        _, size = decode_dma(name, args)
        # Retain v4's explicit UB-copy throughput assumption. Compiler probes
        # establish the byte count, not measured throughput or pipe occupancy.
        schedule.vf(
            dict(cycles=math.ceil(size / 256),
                 counts={'ub_copy_calls': 1, 'ub_copy_bytes': size, 'cost_source:ub_copy_assumption': 1}))
        schedule.counts['lowered_dma:' + name] += 1
        return True
    if name.startswith(
        ('llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.', 'llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.', 'llvm.hivm.ND.DMA.OUT.TO.UB.')):
        if '.ND.DMA.' in name and getattr(schedule, 'nd_pad', None) != 0:
            raise Unsupported('NDDMA padding is unknown or nonzero')
        direction, size = decode_dma(name, args)
        schedule.dma(direction, size)
        schedule.counts['lowered_dma:' + name] += 1
        return True
    if name == 'llvm.hivm.SET.PAD.CNT.NDDMA':
        if len(args) != 1 or args[0] != 0: raise Unsupported('nonzero NDDMA padding')
        schedule.nd_pad = 0
        schedule.counts['lowered_dma_setup'] += 1
        return True
    if name in ('llvm.hivm.SET.MOV.PAD.VAL', 'llvm.hivm.SET.PAD.VAL.NDDMA') or re.fullmatch(
            r'llvm.hivm.SET.LOOP[0-4].STRIDE.NDDMA', name):
        if len(args) != 1: raise Unsupported('unknown DMA setup ABI')
        schedule.counts['lowered_dma_setup'] += 1
        return True
    if name.startswith(('llvm.lifetime.start.', 'llvm.lifetime.end.')):
        return True
    return False


class LoweredEntrySchedule(MixedSchedule):

    def extra_call(self, name, args):
        if lowered_intrinsic(self, name, args): return True
        return super().extra_call(name, args)
