"""CPU-only, bounded scalar tracing of frozen A5 LLVM IR for VV cost ranking.

This is a cost analysis, NOT execution of tensor computations or a correctness
proof. Vector/data-loaded values stay unknown. A data-dependent branch, DMA
extent, or unsupported instruction rejects analysis instead of guessing a path.
LLVM is before final machine scheduling: costs below are estimates, not cycles
measured on a device. No Torch/Triton runtime is imported.
"""
from collections import Counter
from dataclasses import dataclass, field
import math
import re

VERSION = "vv-backend-scalar-trace-v2"
ASSUMPTIONS = {
    "dma_bytes_per_cycle": 32.0,
    "global_bytes_per_cycle": 512.0,
    "dma_startup_cycles": 32.0,
    "vf_call_cycles": 8.0,
    "scalar_instruction_cycles": 1.0,
    "flag_cycles": 1.0,
    "vector_load_latency": 4.0,
    "vector_store_latency": 4.0,
    "issue_queue_length": 64,
}
# The transfer bandwidth/startup values are explicit modelling assumptions.
# Only AVE arithmetic throughput entries come from the target's cost table.


class Unsupported(ValueError):
    pass


UNKNOWN = None
SSA = r"%[\w.$-]+"


def split_top(text):
    """Comma split respecting LLVM aggregate types and constant expressions."""
    out, start, depth = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>" and (i == 0 or text[i - 1] != "-"):
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(text[start:i].strip())
            start = i + 1
    out.append(text[start:].strip())
    return out


def value(text, env):
    text = text.strip()
    pointer = re.fullmatch(r'inttoptr \(i64 (-?\d+) to ptr addrspace\(6\)\)', text)
    if pointer:
        return ('ub', int(pointer[1]))
    if re.fullmatch(SSA, text):
        return env.get(text)
    if text in ("true", "false"):
        return int(text == "true")
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", text):
        return float(text)
    if text.startswith(('{', '[')) and text.endswith(('}', ']')):
        return tuple(typed_value(v, env) for v in split_top(text[1:-1]))
    return UNKNOWN


def type_and_value(text):
    text = text.strip()
    if text.startswith(('{', '[', '<')):
        depth = 0
        for i, ch in enumerate(text):
            if ch in '{[<': depth += 1
            elif ch in '}]>': depth -= 1
            if depth == 0:
                return text[:i + 1], text[i + 1:].strip()
    m = re.match(r'(ptr(?: addrspace\(\d+\))?|i\d+|float|double|half|bfloat)\s+(.*)', text)
    if not m:
        raise Unsupported('unrecognized operand type: ' + text)
    return m.groups()


def empty_aggregate(dtype):
    if dtype.startswith('{'):
        return tuple(empty_aggregate(t) for t in split_top(dtype[1:-1]))
    if dtype.startswith('['):
        m = re.fullmatch(r'\[(\d+) x (.*)\]', dtype)
        return tuple(empty_aggregate(m.group(2)) for _ in range(int(m.group(1))))
    return UNKNOWN


def typed_value(text, env):
    dtype, payload = type_and_value(text)
    payload = re.sub(
        r'^(?:(?:nonnull|noundef|nocapture|noalias|readonly|writeonly)\s+|align \d+\s+|dereferenceable\(\d+\)\s+)*', '',
        payload)
    if dtype == 'ptr addrspace(6)' and payload == 'null':
        return ('ub', 0)
    if dtype.startswith('<'):
        return UNKNOWN
    if payload in ('undef', 'poison'):
        return empty_aggregate(dtype)
    return value(payload, env)


def insert_aggregate(aggregate, path, val):
    if not path:
        return val
    if not isinstance(aggregate, tuple):
        return UNKNOWN
    items = list(aggregate)
    items[path[0]] = insert_aggregate(items[path[0]], path[1:], val)
    return tuple(items)


def known(v, reason):
    if v is None:
        raise Unsupported(reason)
    return v


def integer(v, bits, signed=True):
    v = int(v) % (1 << bits)
    return v - (1 << bits) if signed and v >= (1 << (bits - 1)) else v


@dataclass
class Function:
    name: str
    arguments: list
    blocks: dict
    entry: str
    vector: bool


def parse_module(text):
    functions = {}
    lines = iter(text.splitlines())
    for line in lines:
        if not line.startswith("define "):
            continue
        match = re.search(r"@([\w.$-]+)\((.*)\) [^{]*\{", line)
        if not match:
            raise Unsupported("unrecognized function signature: " + line)
        name, signature = match.groups()
        args = [re.search(SSA + r"$", arg).group() for arg in split_top(signature) if arg]
        # LLVM's unnamed entry block consumes the next unnamed identifier.
        entry = str(max((int(a[1:]) for a in args if a[1:].isdigit()), default=-1) + 1)
        blocks = {entry: []}
        current = entry
        for body in lines:
            if body == "}":
                break
            body = re.sub(r", ![\w.]+ !\d+", "", body.split(";")[0]).strip()
            if not body:
                continue
            label = re.fullmatch(r"([\w.$-]+):", body)
            if label:
                if current == entry and len(blocks) == 1 and not blocks[entry]:
                    del blocks[entry]
                    entry = label.group(1)
                current = label.group(1)
                blocks[current] = []
            else:
                blocks[current].append(body)
        functions[name] = Function(name, args, blocks, entry, "ptc_simdvf" in line)
    return functions


def read_cost_table(text):
    return {(op, src, dst): tuple(map(float, (interval, latency, units)))
            for op, src, dst, interval, latency, units in re.findall(
                r"AVE_COST_ENTRY\((\w+), (\w+), (\w+), (\d+), (\d+), (\d+)\)", text)}


OP_MAP = {
    "vadd": "VFAddOp",
    "vadds": "VFAddsOp",
    "vsub": "VFSubOp",
    "vmul": "VFMulOp",
    "vmuls": "VFMulsOp",
    "vdiv": "VFDivOp",
    "vsqrt": "VFSqrtOp",
    "vexp": "VFExpOp",
    "vexpdif": "VFExpdifOp",
    "vln": "VFLnOp",
    "vmax": "VFMaxOp",
    "vmaxs": "VFMaxsOp",
    "vmin": "VFMinOp",
    "vmins": "VFMinsOp",
    "vabs": "VFAbsOp",
    "vneg": "VFNegOp",
    "vand": "VFAndOp",
    "vor": "VFOrOp",
    "vxor": "VFXorOp",
    "vnot": "VFNotOp",
    "vsel": "VFSelectOp",
    "vintlv": "VFInterleaveOp",
    "vdintlv": "VFDeInterleaveOp",
}


def instruction_cost(name, table):
    op = name.removeprefix("llvm.hivm.").split(".")[0]
    types = re.findall(r"\.v\d+(f\d+|i\d+|bf16)", name)
    dst = types[0].upper() if types else "Unknown"
    src = types[1].upper() if len(types) > 1 else dst
    key = (OP_MAP.get(op), src, dst)
    if key in table:
        return (*table[key], "target_table")
    # Match AVECostModel's fallback categories. Reductions, predicates,
    # broadcasts etc. still lack complete target timing entries.
    if op in ("vdiv", "vsqrt", "vrsqrt"):
        return (4., 14., 2., "fallback")
    if op == "vexp":
        return (4., 13., 2., "fallback")
    if op == "vln":
        return (4., 15., 2., "fallback")
    return (1., 4., 2., "fallback")


@dataclass
class VectorSchedule:
    table: dict
    assumptions: dict
    slots: dict = field(default_factory=lambda: {"exec": [0., 0.], "load": [0., 0.], "store": [0.]})
    ready: dict = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)
    pending: list = field(default_factory=list)
    issue: float = 0.
    finish: float = 0.

    def add(self, name, rhs, dest):
        op = name.removeprefix("llvm.hivm.").split(".")[0]
        if op.startswith(("vld", "vlds")):
            pipe, interval, latency, units, source = "load", 1., self.assumptions[
                "vector_load_latency"], 2, "load_assumption"
        elif op.startswith(("vst", "vsts")):
            pipe, interval, latency, units, source = "store", 1., self.assumptions[
                "vector_store_latency"], 1, "store_assumption"
        else:
            interval, latency, units, source = instruction_cost(name, self.table)
            pipe = "exec"
        self.counts[name] += 1
        self.counts["cost_source:" + source] += 1
        slots = self.slots[pipe]
        slot = min(range(int(units)), key=slots.__getitem__)
        dependency = max((self.ready.get(x, 0.) for x in re.findall(SSA, rhs)), default=0.)
        queue = sorted(t for t in self.pending if t > self.issue)
        queue_ready = queue[0] if len(queue) >= self.assumptions["issue_queue_length"] else 0.
        start = max(self.issue, slots[slot], dependency, queue_ready)
        slots[slot] = start + interval
        self.issue = start + 0.5  # two issue slots; approximation before ISA scheduling
        self.finish = max(self.finish, start + latency)
        self.pending = [t for t in queue if t > start] + [start + latency]
        if dest:
            self.ready[dest] = start + latency


@dataclass
class KernelSchedule:
    assumptions: dict
    pipes: dict = field(default_factory=lambda: {0: 0., 1: 0., 4: 0., 5: 0.})
    flags: dict = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)
    scalar: float = 0.
    dma_read_bytes: int = 0
    dma_write_bytes: int = 0

    def scalar_step(self):
        self.scalar = max(self.scalar, self.pipes[0]) + self.assumptions["scalar_instruction_cycles"]
        self.pipes[0] = self.scalar
        self.counts["scalar_steps"] += 1

    def vf(self, trace):
        self.pipes[1] = max(self.pipes[1], self.scalar) + trace["cycles"] + self.assumptions["vf_call_cycles"]
        self.counts.update(trace["counts"])
        self.counts["vf_calls"] += 1

    def dma(self, direction, size):
        pipe = 4 if direction == "load" else 5
        self.pipes[pipe] = max(
            self.pipes[pipe],
            self.scalar) + self.assumptions["dma_startup_cycles"] + size / self.assumptions["dma_bytes_per_cycle"]
        if pipe == 4:
            self.dma_read_bytes += size
        else:
            self.dma_write_bytes += size
        self.counts[direction + "_dma_calls"] += 1

    def flag(self, name, args):
        src, dst, ident = [int(known(x, "unknown flag identity")) for x in args]
        if src not in self.pipes or dst not in self.pipes:
            raise Unsupported(f"unmodelled synchronization pipes {src}->{dst}")
        key = (src, dst, ident)
        self.counts["flags"] += 1
        if ".SET." in name:
            self.flags[key] = max(self.scalar, self.pipes[src])
        else:
            if key not in self.flags:
                raise Unsupported(f"wait without statically observed set {key}")
            self.pipes[dst] = max(self.scalar, self.pipes[dst], self.flags.pop(key)) + self.assumptions["flag_cycles"]
            if dst == 0:
                self.scalar = self.pipes[0]

    def result(self):
        return {
            "cycles": max(self.scalar, *self.pipes.values()), "pipe_cycles": self.pipes, "dma_read_bytes":
            self.dma_read_bytes, "dma_write_bytes": self.dma_write_bytes, "counts": dict(self.counts)
        }

    def barrier(self, args):
        self.scalar = max(self.scalar, *self.pipes.values())


class Analyzer:

    def __init__(self, text, table, assumptions=None, max_steps=1000000, kernel_factory=None, subblock_id=None,
                 vector_factory=None, scalar_hooks=None, lowered_entry=False):
        self.functions = parse_module(text)
        self.table = table
        self.assumptions = dict(ASSUMPTIONS, **(assumptions or {}))
        self.max_steps = max_steps
        self.vf_cache = {}
        self.kernel_factory = kernel_factory or KernelSchedule
        self.vector_factory = vector_factory or VectorSchedule
        self.subblock_id = subblock_id
        self.scalar_hooks = scalar_hooks
        self.lowered_entry = lowered_entry
        self.call_depth = 0

    def trace(self, name, arguments, block_id, initial_memory=None, kernel_override=None):
        fn = self.functions[name]
        if len(arguments) != len(fn.arguments):
            raise Unsupported(f"ABI mismatch {name}: {len(arguments)} != {len(fn.arguments)}")
        key = (name, tuple(arguments), tuple(sorted((initial_memory or {}).items())))
        if fn.vector and key in self.vf_cache:
            return self.vf_cache[key]
        env, memory = dict(zip(fn.arguments, arguments)), dict(initial_memory or {})
        vector = self.vector_factory(self.table, self.assumptions) if fn.vector else None
        kernel = None if vector else kernel_override if kernel_override is not None else self.kernel_factory(
            self.assumptions)
        block, previous, steps, visits = fn.entry, None, 0, Counter()
        while True:
            visits[block] += 1
            lines = fn.blocks[block]
            # Phi nodes are parallel assignments using predecessor values.
            phi_values, phi_ready = {}, {}
            for line in lines:
                if " = phi " not in line:
                    continue
                dest, rhs = line.split(" = ", 1)
                incoming = re.findall(r"\[\s*(.*?),\s*%([\w.$-]+)\s*\]", rhs)
                item = next((v for v, label in incoming if label == previous), None)
                if item is None:
                    raise Unsupported(f"missing predecessor {name}:{block} from {previous}")
                phi_values[dest] = value(item, env)
                if vector:
                    phi_ready[dest] = vector.ready.get(item, 0.)
            env.update(phi_values)
            if vector:
                vector.ready.update(phi_ready)
            next_block = None
            for instruction_index, line in enumerate(lines):
                steps += 1
                if steps > self.max_steps:
                    raise Unsupported(f"scalar tracing limit {self.max_steps} in {name}")
                dest, rhs = line.split(" = ", 1) if " = " in line else (None, line)
                if self.scalar_hooks is not None:
                    self.scalar_hooks.record(rhs, (block, visits[block], instruction_index))
                if rhs.startswith("phi "):
                    continue
                if rhs.startswith("ret "):
                    effects = [(p, memory.get(p)) for p in (initial_memory or {})] if self.lowered_entry else []
                    if kernel_override is not None:
                        if rhs != 'ret void': raise Unsupported('nonvoid lowered helper return')
                        return {'stack_effects': effects}
                    if vector and hasattr(vector, 'finalize'):
                        vector.finalize()
                    result = {"cycles": vector.finish, "counts": dict(vector.counts)} if vector else kernel.result()
                    result.update(scalar_trace_steps=steps, block_visits=dict(visits))
                    if self.lowered_entry: result['stack_effects'] = effects
                    if vector:
                        self.vf_cache[key] = result
                    return result
                if rhs.startswith("br "):
                    if vector and hasattr(vector, 'control'):
                        vector.control(rhs, None)
                    labels = re.findall(r"label %([\w.$-]+)", rhs)
                    if len(labels) == 1:
                        next_block = labels[0]
                    else:
                        condition = known(typed_value(split_top(rhs[3:])[0], env),
                                          f"data-dependent branch: {name} {rhs}")
                        next_block = labels[0 if condition else 1]
                    break
                if kernel:
                    kernel.scalar_step()
                deps_ready = (vector.dependencies(rhs) if vector and hasattr(vector, 'dependencies') else max(
                    (vector.ready.get(x, 0.) for x in re.findall(SSA, rhs)), default=0.) if vector else 0.)
                result = UNKNOWN
                extra_vector_call = False
                call = re.search(r"\bcall\b.*?@([\w.$-]+)\((.*)\)(?: #[0-9]+)?$", rhs)
                if call:
                    callee, argtext = call.groups()
                    args = [typed_value(v, env) for v in split_top(argtext) if v]
                    if self.scalar_hooks is not None and self.scalar_hooks.handles(callee):
                        result = self.scalar_hooks.intrinsic(callee, args)
                    elif callee in self.functions and self.functions[callee].vector:
                        if vector:
                            raise Unsupported("nested vector functions")
                        inherited = {p: memory[p]
                                     for p in args
                                     if isinstance(p, tuple) and p in memory} if self.lowered_entry else None
                        work = self.trace(callee, args, block_id, initial_memory=inherited)
                        for p, v in work.get('stack_effects', []):
                            memory[p] = v
                        kernel.vf(work)
                    elif callee.startswith(("load_gm_to_ubuf_", "store_ubuf_to_gm_")):
                        if self.lowered_entry and callee in self.functions:
                            if self.call_depth >= 8: raise Unsupported('lowered helper recursion limit')
                            self.call_depth += 1
                            inherited = {p: memory[p] for p in args if isinstance(p, tuple) and p in memory}
                            try:
                                work = self.trace(callee, args, block_id, initial_memory=inherited,
                                                  kernel_override=kernel)
                            finally:
                                self.call_depth -= 1
                            for p, v in work.get('stack_effects', []):
                                memory[p] = v
                            continue
                        m = re.fullmatch(
                            r"(load_gm_to_ubuf|store_ubuf_to_gm)_(\d+)d_(float|half|bfloat16_t|int\d+_t|uint\d+_t)",
                            callee)
                        if not m or vector:
                            raise Unsupported("unmodelled DMA helper " + callee)
                        kind, rank, dtype = m.groups()
                        rank = int(rank)
                        sizes = [int(known(v, "unknown DMA extent " + callee)) for v in args[3:3 + rank]]
                        if any(v < 0 for v in sizes):
                            raise Unsupported("negative DMA extent")
                        bits = {"float": 32, "half": 16, "bfloat16_t": 16}.get(dtype)
                        if bits is None:
                            bits = int(re.search(r"\d+", dtype).group())
                        kernel.dma("load" if kind.startswith("load") else "store", math.prod(sizes) * bits // 8)
                    elif callee.startswith("copy_ubuf_to_ubuf_") and kernel:
                        if self.lowered_entry:
                            if callee not in self.functions:
                                raise Unsupported('lowered UB copy definition missing')
                            if self.call_depth >= 8: raise Unsupported('lowered helper recursion limit')
                            self.call_depth += 1
                            inherited = {p: memory[p] for p in args if isinstance(p, tuple) and p in memory}
                            try:
                                work = self.trace(callee, args, block_id, initial_memory=inherited,
                                                  kernel_override=kernel)
                            finally:
                                self.call_depth -= 1
                            for p, v in work.get('stack_effects', []):
                                memory[p] = v
                            continue
                        m = re.fullmatch(r"copy_ubuf_to_ubuf_(\d+)d_(float|half)", callee)
                        if not m:
                            raise Unsupported("unmodelled UB copy " + callee)
                        rank, dtype = m.groups()
                        size = math.prod(int(known(v, "unknown UB copy extent"))
                                         for v in args[3:3 + int(rank)]) * (4 if dtype == 'float' else 2)
                        # Linked meta-op body is absent from this IR. Model a
                        # 256-byte vector load/store pair per chunk explicitly.
                        kernel.vf({
                            'cycles': math.ceil(size / 256), 'counts':
                            {'ub_copy_calls': 1, 'ub_copy_bytes': size, 'cost_source:ub_copy_assumption': 1}
                        })
                    elif callee.startswith(("llvm.smin.", "llvm.smax.", "llvm.umin.", "llvm.umax.")):
                        if all(v is not None for v in args):
                            result = (min if "min" in callee else max)(args)
                    elif callee == "llvm.hivm.GET.BLOCK.IDX":
                        result = block_id
                    elif self.lowered_entry and callee == "llvm.hivm.GET.SYS.VA.BASE":
                        result = UNKNOWN  # never guess a physical device address
                    elif callee == "llvm.hivm.GET.SUBBLOCKID" and self.subblock_id is not None:
                        result = self.subblock_id
                    elif callee == "llvm.hivm.GET.CTRL":
                        result = UNKNOWN if self.lowered_entry else 0
                    elif callee.startswith("llvm.hivm.SBITSET"):
                        if all(v is not None for v in args):
                            result = (args[0] | (1 << args[1])) if "SBITSET1" in callee else (args[0] & ~(1 << args[1]))
                    elif callee == "llvm.hivm.SET.CTRL":
                        pass
                    elif ".FLAG." in callee and kernel:
                        kernel.flag(callee, args)
                    elif callee == "llvm.hivm.BARRIER" and kernel:
                        kernel.barrier(args)
                    elif callee.startswith("llvm.hivm.") and vector:
                        vector.add(callee, rhs, dest)
                    elif vector and hasattr(vector, 'extra_call') and vector.extra_call(callee, args, rhs, dest):
                        extra_vector_call = True
                    elif kernel and hasattr(kernel, 'extra_call') and kernel.extra_call(callee, args):
                        pass
                    else:
                        raise Unsupported("unmodelled call: " + callee)
                elif rhs.startswith("freeze "):
                    result = typed_value(rhs[len("freeze "):], env)
                elif rhs.startswith("alloca "):
                    result = ("stack", name, self.call_depth, dest) if self.lowered_entry else ("stack", dest)
                elif rhs.startswith("store "):
                    parts = split_top(rhs[6:])
                    ptr = typed_value(parts[1], env)
                    if isinstance(ptr, tuple) and ptr[0] == "stack":
                        memory[ptr] = typed_value(parts[0], env)
                    elif kernel and 'ptr addrspace(6)' in parts[1]:
                        # Scalar stores into UB are costed but never evaluated
                        # numerically. Their loaded payload remains unknown.
                        kernel.counts['scalar_ub_stores'] += 1
                    else:
                        raise Unsupported("unmodelled scalar write: " + rhs)
                elif rhs.startswith("load "):
                    ptr = typed_value(split_top(rhs[5:])[1], env)
                    result = memory.get(ptr) if isinstance(ptr, tuple) else UNKNOWN
                    if kernel and 'ptr addrspace(6)' in rhs:
                        kernel.counts['scalar_ub_loads'] += 1
                    if self.scalar_hooks is not None:
                        result = self.scalar_hooks.load(rhs)
                elif rhs.startswith("select "):
                    parts = split_top(rhs[7:])
                    cond = typed_value(parts[0], env)
                    if cond is not None:
                        result = typed_value(parts[1 if cond else 2], env)
                elif rhs.startswith('extractvalue '):
                    parts = split_top(rhs[len('extractvalue '):])
                    result = typed_value(parts[0], env)
                    for index in parts[1:]:
                        result = result[int(index)] if isinstance(result, tuple) else UNKNOWN
                elif rhs.startswith('insertvalue '):
                    parts = split_top(rhs[len('insertvalue '):])
                    result = insert_aggregate(typed_value(parts[0], env), list(map(int, parts[2:])),
                                              typed_value(parts[1], env))
                elif rhs.startswith("icmp "):
                    m = re.fullmatch(r"icmp (\w+) i(\d+) (.*)", rhs)
                    if not m:
                        raise Unsupported("unmodelled compare " + rhs)
                    pred, width, operands = m.groups()
                    a, b = [value(v, env) for v in split_top(operands)]
                    if a is not None and b is not None:
                        width = int(width)
                        signed = not pred.startswith("u")
                        a, b = integer(a, width, signed), integer(b, width, signed)
                        result = int({"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a
                                      >= b}[pred.lstrip("su")])
                elif re.match(r"(sext|zext|trunc|sitofp|uitofp|fptosi|fptoui|fpext|fptrunc|bitcast) ", rhs):
                    m = re.match(r"(\w+) (i\d+|float|double|half) (.*?) to (i\d+|float|double|half)$", rhs)
                    if m:
                        op, src, val, dst = m.groups()
                        result = value(val, env)
                        if result is not None and src.startswith("i"):
                            result = integer(result, int(src[1:]), op not in ("zext", "uitofp"))
                        if result is not None and dst.startswith("i"):
                            result = integer(result, int(dst[1:]))
                elif re.match(r"(add|sub|mul|sdiv|udiv|srem|urem|and|or|xor|shl|lshr|ashr) ", rhs):
                    m = re.fullmatch(r"(\w+) (?:(?:nuw|nsw|exact) )*(i\d+) (.*)", rhs)
                    if not m:
                        raise Unsupported("unmodelled arithmetic " + rhs)
                    op, dtype, operands = m.groups()
                    a, b = [value(v, env) for v in split_top(operands)]
                    if a is not None and b is not None:
                        bits = int(dtype[1:])
                        a, b = integer(a, bits, not op.startswith("u")), integer(b, bits, not op.startswith("u"))
                        if op == "add": result = a + b
                        elif op == "sub": result = a - b
                        elif op == "mul": result = a * b
                        elif op in ("sdiv", "udiv"): result = (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
                        elif op in ("srem", "urem"):
                            result = a - ((abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)) * b
                        elif op == "and":
                            result = a & b
                        elif op == "or":
                            result = a | b
                        elif op == "xor":
                            result = a ^ b
                        elif op == "shl":
                            result = a << b
                        elif op == "ashr":
                            result = a >> b
                        elif op == "lshr":
                            result = integer(a, bits, False) >> b
                        result = integer(result, bits)
                elif self.lowered_entry and rhs.startswith('ptrtoint '):
                    pointer = typed_value(rhs[len('ptrtoint '):].rsplit(' to ', 1)[0], env)
                    if isinstance(pointer, tuple) and pointer[0] == 'ub': result = pointer[1]
                elif self.lowered_entry and rhs.startswith('getelementptr '):
                    parts = split_top(re.sub(r'^getelementptr (?:inbounds )?', '', rhs))
                    pointer = typed_value(parts[1], env)
                    if isinstance(pointer, tuple) and pointer[0] == 'ub':
                        if len(parts) != 3: raise Unsupported('unsupported UB GEP layout')
                        # LLVM scalar i1 has byte-sized allocation stride; this
                        # is pointer arithmetic, not a packed vector UB estimate.
                        size = {'float': 4, 'half': 2, 'bfloat': 2, 'i1': 1, 'i8': 1, 'i16': 2, 'i32': 4, 'i64':
                                8}.get(parts[0])
                        index = typed_value(parts[2], env)
                        if size is None: raise Unsupported('unsupported UB GEP type')
                        if index is not None: result = ('ub', pointer[1] + index * size)
                elif rhs.startswith(("getelementptr ", "inttoptr ", "ptrtoint ", "fadd ", "fsub ", "fmul ", "fdiv ",
                                     "fcmp ", "extractelement ", "insertelement ", "shufflevector ")):
                    # Their tensor/pointer payload remains unknown. Any use in
                    # scalar control or transfer sizes will fail at that use.
                    pass
                else:
                    raise Unsupported("unmodelled instruction " + rhs)
                if dest:
                    env[dest] = result
                    if vector and not extra_vector_call and not (call and call.group(1).startswith("llvm.hivm.")):
                        vector.ready[dest] = deps_ready
                if vector and not call and not rhs.startswith(
                    ('alloca ', 'store ', 'load ')) and hasattr(vector, 'control'):
                    vector.control(rhs, dest)
            if next_block is None:
                raise Unsupported("missing terminator " + name + ":" + block)
            previous, block = block, next_block


def launch_arguments(ttir, contract, axis, factor, *, preserve_grid=False):
    signature = next(line for line in ttir.splitlines() if "tt.func public @" in line)
    arg_names = re.findall(r"%([\w.$-]+):", signature)
    grid = list(contract["grid"])
    if factor != 1 and not preserve_grid:
        grid[axis] = math.ceil(grid[axis] / factor)
    values = [UNKNOWN, UNKNOWN]  # standard TA synchronization and workspace pointers
    for name in arg_names:
        if name.startswith("__hacc_original_grid_"):
            values.append(contract["grid"]["xyz".index(name[-1])])
        elif name in contract["arguments"]:
            arg = contract["arguments"][name]
            values.append(arg["value"] if arg["kind"] == "scalar" else UNKNOWN)
        else:
            raise Unsupported("missing launch scalar/argument " + name)
    return values + grid, grid


def analyze_candidate(llvm, ttir, final_ir, metadata, contract, axis, factor, table, assumptions=None, *,
                      kernel_factory=None, include_traces=False, vector_factory=None, mixed_validator=None,
                      lowered_entry=False):
    m = re.search(r'"VECTOR_CORE_COUNT", (\d+) : i32', final_ir)
    if not m:
        raise Unsupported("CPU target vector core count missing")
    core_count = int(m.group(1))
    mode = metadata.get("parallel_mode")
    if metadata.get("mix_mode") != "aiv" or mode not in ("simd", "mix_simd_simt"):
        raise Unsupported("only pure AIV SIMD is implemented")
    if mode == "mix_simd_simt":
        if mixed_validator is None:
            raise Unsupported("AIV MIX requires a verified mixed-path adapter")
        mixed_validator(llvm, final_ir)
    for field in ("ptsm_cap_authorized", "row_coalescing_applied"):
        if metadata.get(field):
            raise Unsupported("additional grid transform " + field)
    if metadata.get("coalesce_factor", 1) != 1:
        raise Unsupported("additional coalesce transform")
    if factor not in (1, 2, 4, 6, 8):
        raise Unsupported("factor outside experiment")
    from .v38_runtime_policy import preserves_grid
    preserve_grid = preserves_grid(metadata, factor)
    args, grid = launch_arguments(ttir, contract, axis, factor, preserve_grid=preserve_grid)
    auto = metadata["auto_blockify_enabled"]
    tasks = math.prod(grid)
    blocks = min(tasks, core_count) if auto else tasks
    analyzer = Analyzer(llvm, table, assumptions, kernel_factory=kernel_factory, vector_factory=vector_factory,
                        lowered_entry=lowered_entry)
    traces = [analyzer.trace(metadata["name"], args, index) for index in range(blocks)]
    # AutoBlockify's actual per-core loop is traced above. In the nonpersistent
    # case, hardware block-to-core assignment isn't encoded: use a greedy
    # scheduling estimate, clearly distinguished from exact loop traversal.
    cores = [0.] * min(core_count, blocks)
    if auto:
        cores = [t["cycles"] for t in traces]
    else:
        for trace in traces:
            core = min(range(len(cores)), key=cores.__getitem__)
            cores[core] += trace["cycles"]
    dma_read = sum(t["dma_read_bytes"] for t in traces)
    dma_write = sum(t["dma_write_bytes"] for t in traces)
    counts = Counter()
    for trace in traces:
        counts.update(trace["counts"])
    global_memory_bound = (dma_read + dma_write) / analyzer.assumptions["global_bytes_per_cycle"]
    result = {
        "status": "STATIC_COST_ESTIMATED", "score": max(max(cores), global_memory_bound), "critical_core_cycles":
        max(cores), "global_memory_bound": global_memory_bound, "core_cycles": cores, "logical_grid": grid,
        "launched_blocks": blocks, "core_count": core_count, "auto_blockify": auto, "scheduling":
        "compiled grid-stride loop" if auto else "greedy hardware assignment estimate", "dma_read_bytes": dma_read,
        "dma_write_bytes": dma_write, "counts": dict(counts), "vf_trace_count": len(analyzer.vf_cache),
        "entry_block_visits": [t["block_visits"] for t in traces], "llvm_steps": sum(t["scalar_trace_steps"]
                                                                                     for t in traces), "assumptions":
        analyzer.assumptions, "device_executed": False, "numerical_correctness_verified": False
    }
    if include_traces:
        result["block_traces"] = traces
    return result


def choose(candidates):
    """Only static scores and current candidate resource proofs enter selection."""
    eligible = {
        k: v
        for k, v in candidates.items()
        if v.get("resource_checked") is True and v.get("status") == "STATIC_COST_ESTIMATED"
    }
    if not eligible:
        return None
    return min(eligible, key=lambda k: (eligible[k]["score"], 1 if k == "unmerged" else int(k.split("_f")[1]), k))
