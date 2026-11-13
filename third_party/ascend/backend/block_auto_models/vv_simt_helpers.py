"""Bounded work counts from the candidate's verified, linked SIMT bodies.

No kernel names or timing labels participate. Work-item coordinates and launch
dimensions are static. Tensor values stay unknown; only the two mask branches
of the supported gather are evaluated as conservative alternatives. Costs use
serialized warp issue with the existing per-instruction cost assumption, not
measured hardware throughput. Lane memory traffic is never divided by warp size.
"""
from collections import Counter
import itertools
import math
import re

from .backend_cost_model import ASSUMPTIONS, Analyzer, Unsupported, known, parse_module, split_top, type_and_value, typed_value

SUPPORTED = {'indirect_load_nonvolatile_2d_float_int64_t', 'cumsum_1d_float_dim0'}
LOWERED_SUPPORTED = SUPPORTED | {'indirect_load_nonvolatile_1d_bfloat16_t_int64_t'}
# RegBase/Vector/CumsumSimtSklansky.cpp uses CUMSUM_WARP=32 and threadIdx.x
# as the lane. Costing assumes one issued warp instruction per model cycle.
WARP_WIDTH = 32
INTRINSICS = {'llvm.hivm.get.TID.' + a
              for a in 'XYZ'} | {'llvm.hivm.get.BLOCK.DIM.' + a
                                 for a in 'XYZ'} | {'llvm.hivm.shfl.idx.i32', 'llvm.hivm.sync.workitems'}


class WorkItem:

    def __init__(self, coordinates, dimensions, mask=None):
        self.coordinates, self.dimensions, self.mask = coordinates, dimensions, mask
        self.counts = Counter()
        self.events = {}

    def record(self, rhs, location=None):
        if location is not None:
            self.events[location] = rhs
        op = rhs.split()[0]
        if op not in ('phi', 'ret', 'bitcast'):
            self.counts['instruction_steps'] += 1
        if rhs.startswith(('load ', 'store ')):
            match = re.match(r'(load|store) (float|double|half|bfloat|i\d+)(?=[ ,])', rhs)
            address = re.search(r'ptr addrspace\((\d+)\)', rhs)
            if not match or not address or address[1] not in ('1', '6'):
                raise Unsupported('unsupported SIMT memory access: ' + rhs)
            dtype = match[2]
            bits = {'float': 32, 'double': 64, 'half': 16, 'bfloat': 16}.get(dtype)
            bits = bits if bits is not None else int(dtype[1:])
            if bits <= 0 or bits > 64:
                raise Unsupported('unsupported SIMT memory width')
            space = 'gm' if address[1] == '1' else 'ub'
            self.counts[space + '_' + match[1] + 's'] += 1
            self.counts[space + '_' + match[1] + '_bytes'] += (bits + 7) // 8
        if '@llvm.hivm.shfl.idx.i32(' in rhs:
            self.counts['shuffles'] += 1
        if '@llvm.hivm.sync.workitems(' in rhs:
            self.counts['workitem_barriers'] += 1

    def handles(self, name):
        return name in INTRINSICS

    def intrinsic(self, name, args):
        if name.startswith('llvm.hivm.get.TID.'):
            return self.coordinates['XYZ'.index(name[-1])]
        if name.startswith('llvm.hivm.get.BLOCK.DIM.'):
            return self.dimensions['XYZ'.index(name[-1])]
        return None  # Shuffled tensor values are never evaluated.

    def load(self, rhs):
        if self.mask is not None and rhs.startswith('load i8, ptr addrspace(6)'):
            return self.mask
        return None


def decode_dimensions(info):
    # HIVMRegbaseIntrinsToLLVMIRTranslation.cpp::convertLaunchFunc:
    # blockX | (blockY << 16) | (blockZ << 32).
    if type(info) is not int or info < 0 or info >> 48:
        raise Unsupported('invalid SIMT launch configuration')
    dims = tuple((info >> shift) & 0xffff for shift in (0, 16, 32))
    if not all(dims) or math.prod(dims) > 4096:
        raise Unsupported('SIMT launch exceeds bounded analysis domain')
    return dims


def validate_abi(name, args):
    """Only cost static, bounded descriptors; never silently use zero extents."""
    if name == 'cumsum_1d_float_dim0':
        if len(args) != 16:
            raise Unsupported('unknown cumsum descriptor ABI')
        views = [(args[i + 2], args[i + 3], args[i + 4]) for i in (0, 5, 10)]
        for offset, size, stride in views:
            if any(type(v) is not int for v in (offset, size, stride)) or offset < 0 or size <= 0 or stride != 1:
                raise Unsupported('cumsum requires static contiguous nonempty views')
        if views[0][1] != views[1][1] or args[15] not in (0, 1):
            raise Unsupported('inconsistent cumsum extents or direction')
    else:
        if len(args) != 33:
            raise Unsupported('unknown 2D gather descriptor ABI')
        views = [args[i:i + 7] for i in (5, 12, 19, 26)]
        shapes = []
        for view in views:
            offset, rows, cols, stride0, stride1 = view[2:]
            if any(type(v) is not int for v in view[2:]) or min(rows, cols) <= 0 or offset < 0:
                raise Unsupported('2D gather requires static nonempty views')
            if stride1 != 1 or stride0 < cols:
                raise Unsupported('2D gather requires nonoverlapping row-major views')
            shapes.append((rows, cols))
        if len(set(shapes)) != 1:
            raise Unsupported('2D gather view extents differ')


class VerifiedSimtHelpers:
    """Caller must first verify executable equality with verified_vectors()."""

    def __init__(self, optimized_llvm, assumptions, parameters, lowered_abi=False):
        self.text, self.parameters = optimized_llvm, parameters
        self.assumptions = dict(ASSUMPTIONS, **assumptions)
        self.functions = parse_module(optimized_llvm)
        self.headers = {m[1]: m[0] for m in re.finditer(r'^define [^\n]*@([^ (]+)[^\n]*\{', optimized_llvm, re.M)}
        self.cache = {}
        self.lowered_abi = lowered_abi

    def profile(self, name, args):
        if name not in (LOWERED_SUPPORTED if self.lowered_abi else SUPPORTED):
            return None
        bound = None
        key = name
        if self.lowered_abi:
            if name not in self.functions or len(args) != len(self.functions[name].arguments):
                raise Unsupported('lowered SIMT helper ABI mismatch')
            signature = re.search(r'@' + re.escape(name) + r'\((.*)\) ', self.headers[name])[1]
            bound = []
            for arg, operand in zip(args, split_top(signature)):
                dtype, _ = type_and_value(operand)
                if dtype.startswith('ptr'): bound.append(None)
                elif type(arg) is int: bound.append(arg)
                else: raise Unsupported('dynamic lowered SIMT scalar argument')
            key = name + ':' + repr(bound)
        else:
            validate_abi(name, args)
        if key not in self.cache:
            self.cache[key] = self._trace(name, bound)
        return self.cache[key]

    def _trace(self, name, bound=None):
        fn = self.functions.get(name)
        if fn is None:
            raise Unsupported('optimized helper definition missing: ' + name)
        lines = [l for ls in fn.blocks.values() for l in ls]
        if any(l.startswith(('br ', 'switch ')) for l in lines):
            raise Unsupported('helper launch requires unresolved scalar control')
        infos = re.findall(r'@llvm\.hivm\.store\.vfsimt\.info\(i64 (\d+)\)', '\n'.join(lines))
        launches = [re.search(r'\bcall simt_entry void @([\w.$-]+)\((.*)\)$', l) for l in lines]
        launches = [m for m in launches if m]
        if len(infos) != 1 or len(launches) != 1:
            raise Unsupported('expected one statically configured SIMT launch')
        dims = decode_dimensions(int(infos[0]))
        target, operands = launches[0].groups()
        if target not in self.functions or 'simt_entry' not in self.headers[target]:
            raise Unsupported('linked SIMT function missing')
        allowed_calls = INTRINSICS | {'llvm.hivm.store.vfsimt.info', 'llvm.hivm.BARRIER', target}
        for line in lines:
            call = re.search(r'\bcall\b[^\n]*@([\w.$-]+)\(', line)
            if call and call[1] not in allowed_calls:
                raise Unsupported('unmodelled helper wrapper call: ' + call[1])
        if any('store ' in line or 'load ' in line for line in lines):
            raise Unsupported('helper launch wrapper contains unmodelled memory work')
        if bound is None:
            arguments = []
            for operand in split_top(operands):
                if not operand: continue
                dtype, _ = type_and_value(operand)
                arguments.append(None if dtype.
                                 startswith('ptr') else known(typed_value(operand, {}), 'dynamic SIMT scalar argument'))
        else:
            # Follow the actual optimized wrapper's casts/arithmetic. Its
            # scalar launch arguments may derive from the caller's shape.
            class Launch:
                arguments = None

                def record(self, rhs, location=None):
                    pass

                def handles(self, callee):
                    return callee in {target, 'llvm.hivm.store.vfsimt.info', 'llvm.hivm.BARRIER'}

                def intrinsic(self, callee, args):
                    if callee == target:
                        if self.arguments is not None: raise Unsupported('multiple SIMT launches')
                        self.arguments = args
                    return None

            launch = Launch()
            Analyzer(self.text, {}, self.assumptions, scalar_hooks=launch).trace(name, bound, 0)
            if launch.arguments is None: raise Unsupported('missing resolved SIMT launch')
            arguments = launch.arguments
            target_signature = re.search(r'@' + re.escape(target) + r'\((.*)\) ', self.headers[target])[1]
            for arg, operand in zip(arguments, split_top(target_signature)):
                if not type_and_value(operand)[0].startswith('ptr') and type(arg) is not int:
                    raise Unsupported('dynamic SIMT scalar argument')
        total, warps = Counter(), {}
        modes = (0, 1) if name.startswith('indirect_load_') else (None, )
        analyzer = Analyzer(self.text, {}, self.assumptions, max_steps=100000)
        for coord in itertools.product(*(range(n) for n in dims)):
            linear = coord[0] + dims[0] * (coord[1] + dims[1] * coord[2])
            events = warps.setdefault(linear // WARP_WIDTH, {})
            alternatives = []
            for mask in modes:
                hook = WorkItem(coord, dims, mask)
                analyzer.scalar_hooks = hook
                analyzer.trace(target, arguments, 0)
                alternatives.append(hook.counts)
                events.update(hook.events)
            upper = Counter()
            for counts in alternatives:
                upper |= counts
            total.update(upper)
        issued = Counter()
        for events in warps.values():
            hook = WorkItem((0, 0, 0), dims)
            for rhs in events.values():
                hook.record(rhs)
            issued.update(hook.counts)
        if total['gm_stores']:
            raise Unsupported('unexpected GM store in supported SIMT helper')
        p, a = self.parameters, self.assumptions
        steps = sum(not l.startswith('ret ') for l in lines)
        transactions = total['gm_loads'] * p['simt_transaction_bytes']
        startup = 2 * p['simt_transition_cycles'] + steps * a['scalar_instruction_cycles']
        cycles = startup + issued['instruction_steps'] * p['simt_instruction_cycles']
        cycles += issued['ub_loads'] * a['vector_load_latency'] + issued['ub_stores'] * a['vector_store_latency']
        cycles += (total['ub_load_bytes'] + total['ub_store_bytes']) / a['dma_bytes_per_cycle']
        cycles += issued['workitem_barriers'] * a['flag_cycles']
        cycles += transactions / a['dma_bytes_per_cycle']
        return dict(
            cycles=cycles, startup=startup, transaction_bytes_upper=transactions, dimensions=dims,
            work_items=math.prod(dims), counts=dict(total), warp_width=WARP_WIDTH, warps=len(warps),
            warp_counts=dict(issued), wrapper_steps=steps, simt_function=target,
            policy='verified linked body; serialized warp issue; divergent paths union; lane memory traffic upper bound'
        )
