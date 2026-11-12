"""CPU-only cost ranking for A5 Cube and Cube/Vector compiled candidates.

Bounded scalar tracing supplies the actual loops and matrix/transfer extents.
An event graph models local pipes and paired Cube/Vector flag dependencies.
This is a ranking prototype, not a cycle-accurate simulator, tensor evaluator,
resource proof, or device launcher. Resource admission is handled by the caller.
"""
from collections import Counter, defaultdict, deque
import math
import re

from backend_cost_model import ASSUMPTIONS as VECTOR_ASSUMPTIONS
from backend_cost_model import Analyzer, Unsupported, known, launch_arguments
from cv_batch_cost import batch_fixpipe, batch_mma, batch_nd2nz

VERSION = 'cv-backend-event-graph-v5-batch-helpers'
ASSUMPTIONS = dict(
    VECTOR_ASSUMPTIONS, **{
        # These are explicit uncalibrated relative costs, NOT hardware guarantees.
        'cube_flops_per_cycle': 2048.,
        'cube_startup_cycles': 16.,
        'l1_l0_bytes_per_cycle': 64.,
        'l1_l0_startup_cycles': 8.,
        'fixpipe_bytes_per_cycle': 64.,
        'fixpipe_startup_cycles': 16.,
        'cross_flag_cycles': 4.,
        'vector_cycle_scale': 1.,
    })
PIPE_NAMES = {0: 'scalar', 1: 'vector', 2: 'cube', 3: 'mte1', 4: 'mte2', 5: 'mte3', 10: 'fixpipe'}


def validated_assumptions(overrides=None):
    """One target-wide cost profile; no kernel or measured-time inputs."""
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict):
        raise Unsupported('cost parameters must be an object')
    unknown = overrides.keys() - ASSUMPTIONS.keys()
    if unknown:
        raise Unsupported('unknown cost parameters: ' + ', '.join(sorted(unknown)))
    result = dict(ASSUMPTIONS, **overrides)
    for key, value in result.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise Unsupported('cost parameter must be finite and positive: ' + key)
    if type(result['issue_queue_length']) is not int:
        raise Unsupported('issue_queue_length must be a positive integer')
    return result


def extent(values, label):
    sizes = [int(known(v, 'unknown extent: ' + label)) for v in values]
    if any(v < 0 for v in sizes):
        raise Unsupported('negative extent: ' + label)
    return math.prod(sizes)


def align(x, size):
    return math.ceil(x / size) * size


class EventGraph:
    """Dependency graph with FIFO flag matching and longest-path evaluation."""

    def __init__(self):
        self.nodes = []
        self.cross_sets = defaultdict(list)
        self.cross_waits = defaultdict(list)

    def add(self, role, pipe, label, duration, parents=()):
        ident = len(self.nodes)
        self.nodes.append({
            'role': role, 'pipe': pipe, 'label': label, 'duration': float(duration), 'parents':
            sorted(set(p for p in parents if p is not None))
        })
        return ident

    def finish(self, retain_events=False):
        for key in self.cross_sets.keys() | self.cross_waits.keys():
            sets, waits = self.cross_sets[key], self.cross_waits[key]
            if len(sets) != len(waits):
                raise Unsupported(f'unmatched cross-core flag {key}: {len(sets)} sets, {len(waits)} waits')
            for source, target in zip(sets, waits):
                self.nodes[target]['parents'].append(source)
                self.nodes[target]['parents'] = sorted(set(self.nodes[target]['parents']))
        degree = [len(n['parents']) for n in self.nodes]
        children = defaultdict(list)
        for target, node in enumerate(self.nodes):
            for parent in node['parents']:
                children[parent].append(target)
        queue = deque(i for i, d in enumerate(degree) if d == 0)
        ready, end, completed = [0.] * len(degree), [0.] * len(degree), 0
        while queue:
            source = queue.popleft()
            end[source] = ready[source] + self.nodes[source]['duration']
            completed += 1
            for child in children[source]:
                ready[child] = max(ready[child], end[source])
                degree[child] -= 1
                if degree[child] == 0:
                    queue.append(child)
        if completed != len(self.nodes):
            raise Unsupported('cyclic scheduling dependencies in CV cost graph')
        busy, finish = Counter(), Counter()
        for node, done in zip(self.nodes, end):
            name = node['role'] + ':' + PIPE_NAMES[node['pipe']]
            busy[name] += node['duration']
            finish[name] = max(finish[name], done)
        result = {
            'cycles': max(end, default=0.), 'pipe_work': dict(busy), 'pipe_finish': dict(finish), 'event_count':
            len(self.nodes), 'cross_flag_pairs': sum(map(len, self.cross_sets.values()))
        }
        if retain_events:
            result['events'] = [dict(n, start=r, end=e) for n, r, e in zip(self.nodes, ready, end)]
        return result


class CVSchedule:
    """A participant's ordered pipes within one Cube + two Vector core group."""

    def __init__(self, graph, role, assumptions):
        self.graph, self.role, self.assumptions = graph, role, assumptions
        self.pipes = dict.fromkeys(PIPE_NAMES)
        self.flags = defaultdict(deque)
        self.counts = Counter()
        self.scalar = None
        self.dma_read_bytes = self.dma_write_bytes = 0
        self.l0_pingpong = 0
        self.unit_mma, self.unit_fix = {}, {}

    def op(self, pipe, label, duration, dependencies=()):
        if pipe not in self.pipes:
            raise Unsupported('unknown CV pipe ' + str(pipe))
        ident = self.graph.add(self.role, pipe, label, duration, (self.pipes[pipe], self.scalar, *dependencies))
        self.pipes[pipe] = ident
        if pipe == 0:
            self.scalar = ident
        return ident

    def scalar_step(self):
        self.op(0, 'scalar_ir', self.assumptions['scalar_instruction_cycles'])
        self.counts['scalar_steps'] += 1

    def vf(self, trace):
        self.op(1, 'vector_function',
                trace['cycles'] * self.assumptions['vector_cycle_scale'] + self.assumptions['vf_call_cycles'])
        self.counts.update(trace['counts'])
        self.counts['vf_calls'] += 1

    def dma(self, direction, size):
        self.op(4 if direction == 'load' else 5, 'gm_' + direction,
                self.assumptions['dma_startup_cycles'] + size / self.assumptions['dma_bytes_per_cycle'])
        if direction == 'load':
            self.dma_read_bytes += size
        else:
            self.dma_write_bytes += size
        self.counts[direction + '_dma_calls'] += 1

    def flag(self, name, args):
        src, dst, ident = [int(known(x, 'unknown CV flag')) for x in args]
        key = (src, dst, ident)
        if '.SET.' in name:
            self.flags[key].append(self.op(src, f'set {key}', self.assumptions['flag_cycles']))
        else:
            if not self.flags[key]:
                raise Unsupported(f'local flag wait without set: {self.role} {key}')
            self.op(dst, f'wait {key}', self.assumptions['flag_cycles'], (self.flags[key].popleft(), ))
        self.counts['local_flags'] += 1

    def barrier(self, args):
        pipe = int(known(args[0], 'unknown barrier pipe'))
        if pipe == 6:  # PIPE_ALL
            self.op(0, 'barrier_all', 0., tuple(self.pipes.values()))
        else:
            self.op(pipe, 'barrier_pipe', 0.)
        self.counts['barriers'] += 1

    def cross_flag(self, name, args):
        pipe, ident = [int(known(x, 'unknown cross-core flag')) for x in args]
        is_set = '.SET.' in name
        # The A5 split-core lowering emits IDs 0..15 and 16..31 for the
        # two AIV subblocks on AIC; each AIV sees its local IDs 0..15.
        if self.role == 'aic':
            if not 0 <= ident < 32:
                raise Unsupported('cross-core flag outside split-core range')
            subblock, local = divmod(ident, 16)
            direction = 'c_to_v' if is_set else 'v_to_c'
        else:
            subblock, local = int(self.role[-1]), ident
            if not 0 <= local < 16:
                raise Unsupported('AIV cross-core flag outside local range')
            direction = 'v_to_c' if is_set else 'c_to_v'
        node = self.op(pipe, ('cross_set ' if is_set else 'cross_wait ') + str(ident),
                       self.assumptions['cross_flag_cycles'])
        (self.graph.cross_sets if is_set else self.graph.cross_waits)[direction, subblock, local].append(node)
        self.counts['cross_flags'] += 1

    def mma(self, name, args):
        if len(args) != 46 or self.role != 'aic':
            raise Unsupported('unsupported MMA helper ABI ' + name)
        m, k, n = [int(known(v, 'unknown MMA size')) for v in args[23:26]]
        if min(m, k, n) <= 0:
            raise Unsupported('nonpositive MMA size')
        # A5 RegBase/Cube/LocalMmad.cpp::L1Mmad slices K to fit each
        # 32 KiB L0A/L0B ping-pong half. Both slices use the fixed L0
        # layouts (A zN, B nZ), including transposed L1 operands.
        pm, pk, pn = align(m, 16), align(k, 8), align(n, 16)
        k_tile = min((32 * 1024 // 4 // pm) // 8 * 8, (32 * 1024 // 4 // pn) // 8 * 8) // 16 * 16
        if k_tile == 0:
            raise Unsupported('single-buffer MMA helper not implemented')
        wait_a, wait_b, release_a, release_b = [int(known(v, 'unknown MMA event')) for v in args[37:41]]
        if args[41:44] != [-1, -1, -1]:
            raise Unsupported('alternate MMA double-buffer protocol not implemented')
        unit, group = [int(known(v, 'unknown MMA unit flag')) for v in args[44:46]]
        if unit not in (0, 3):
            raise Unsupported('partial MMA unit-flag mode not implemented')
        if unit and (extent(args[30:32], 'L0C M') != m or extent((args[29], args[32]), 'L0C N') != n):
            raise Unsupported('partial L0C unit-flag update not implemented')
        for start in range(0, k, k_tile):
            actual_k = min(k_tile, k - start)
            padded_k = align(actual_k, 8)
            first, last = start == 0, start + actual_k == k
            l0_event = self.l0_pingpong
            self.l0_pingpong ^= 1
            self.flag('llvm.hivm.WAIT.FLAG.IMM', [2, 3, l0_event])
            for dim_size, wait, release in ((pm * padded_k * 4, wait_a, release_a), (padded_k * pn * 4, wait_b,
                                                                                     release_b)):
                if first and wait >= 0:
                    self.flag('llvm.hivm.WAIT.FLAG.IMM', [4, 3, wait])
                self.op(3, 'l1_to_l0',
                        self.assumptions['l1_l0_startup_cycles'] + dim_size / self.assumptions['l1_l0_bytes_per_cycle'])
                if last and release >= 0:
                    self.flag('llvm.hivm.SET.FLAG.IMM', [3, 4, release])
                self.counts['l1_to_l0_bytes'] += dim_size
            self.flag('llvm.hivm.SET.FLAG.IMM', [3, 2, 0])
            self.flag('llvm.hivm.WAIT.FLAG.IMM', [3, 2, 0])
            dependencies = [self.pipes[3]]
            if unit and group in self.unit_fix:
                dependencies.append(self.unit_fix[group])
            node = self.op(
                2, name, self.assumptions['cube_startup_cycles'] +
                2 * pm * padded_k * pn / self.assumptions['cube_flops_per_cycle'], dependencies)
            if unit and last:
                self.unit_mma[group] = node
            self.flag('llvm.hivm.SET.FLAG.IMM', [2, 3, l0_event])
            self.counts.update(mma_calls=1, useful_flops=2 * m * actual_k * n, padded_flops=2 * pm * padded_k * pn)
            self.counts['mma_shape:' + str((m, actual_k, n))] += 1

    def fixpipe(self, name, args):
        dual = name == 'fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf'
        if len(args) != (26 if dual else 25) or self.role != 'aic':
            raise Unsupported('unsupported Fixpipe helper ABI')
        if dual and args[25] != 1:
            raise Unsupported('unsupported dual Fixpipe split mode')
        unit, group = [int(known(v, 'unknown Fixpipe unit flag')) for v in args[23:25]]
        if unit not in (0, 3):
            raise Unsupported('partial Fixpipe unit-flag mode not implemented')
        size = extent(args[14:16], 'Fixpipe destination') * 4 * (2 if dual else 1)
        dependencies = []
        if unit:
            if group not in self.unit_mma:
                raise Unsupported('unit-flag Fixpipe without preceding MMA')
            dependencies.append(self.unit_mma[group])
        node = self.op(10, 'fixpipe_to_ub' if dual else 'fixpipe_to_gm',
                       self.assumptions['fixpipe_startup_cycles'] + size / self.assumptions['fixpipe_bytes_per_cycle'],
                       dependencies)
        if unit:
            self.unit_fix[group] = node
        if not dual:
            self.dma_write_bytes += size
        self.counts['fixpipe_calls'] += 1
        self.counts['fixpipe_bytes'] += size

    def extra_call(self, name, args):
        if name in ('llvm.hivm.SET.INTRA.BLOCKI.mode', 'llvm.hivm.WAIT.INTRA.BLOCKI.mode'):
            self.cross_flag(name, args)
        elif name == 'nd2nz_batch_float':
            batch_nd2nz(self, args)
        elif name in ('batch_mma_tile_float_to_float', 'batch_mma_tile_float_to_float_ta',
                      'batch_mma_tile_float_to_float_tb', 'batch_mma_tile_float_to_float_ta_tb'):
            batch_mma(self, name, args)
        elif name in ('fixpipe_nz2nd_float_to_float_5d_to_3d_gm', 'fixpipe_nz2nd_float_to_float_5d_to_3d_ubuf'):
            batch_fixpipe(self, name, args)
        elif name == 'nd2nz_float' and self.role == 'aic':
            if len(args) != 19:
                raise Unsupported('unsupported ND2NZ helper ABI')
            src, padded = extent(args[3:5], 'ND2NZ source') * 4, extent(args[10:14], 'NZ layout') * 4
            self.op(4, 'gm_to_l1_nd2nz',
                    self.assumptions['dma_startup_cycles'] + max(src, padded) / self.assumptions['dma_bytes_per_cycle'])
            self.dma_read_bytes += src
            self.counts.update(nd2nz_calls=1, nd2nz_source_bytes=src, nd2nz_padded_bytes=padded)
        elif name == 'set_l1_2d_float' and self.role == 'aic':
            if len(args) != 6:
                raise Unsupported('unsupported L1 fill helper ABI')
            size = extent(args[4:5], 'L1 fill') * 4
            self.op(4, 'fill_l1',
                    self.assumptions['dma_startup_cycles'] + size / self.assumptions['dma_bytes_per_cycle'])
            self.counts.update(l1_fill_calls=1, l1_fill_bytes=size)
        elif name in ('mma_tile_float_to_float', 'mma_tile_float_to_float_ta', 'mma_tile_float_to_float_tb'):
            self.mma(name, args)
        elif name in ('fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf', 'fixpipe_nz2nd_float_to_float_4d_to_2d_gm'):
            self.fixpipe(name, args)
        else:
            return False
        return True

    def result(self):
        if any(self.flags.values()):
            raise Unsupported('unconsumed local flags: ' + str([k for k, v in self.flags.items() if v]))
        return {
            'counts': dict(self.counts), 'dma_read_bytes': self.dma_read_bytes, 'dma_write_bytes': self.dma_write_bytes
        }


def analyze_candidate(llvm_modules, ttir, final_ir, metadata, contract, axis, factor, table, assumptions=None,
                      retain_events=False):
    assumptions = validated_assumptions(assumptions)
    if metadata.get('parallel_mode') != 'simd':
        raise Unsupported('CV cost model requires SIMD compiled IR')
    if factor not in (1, 2, 4, 6, 8):
        raise Unsupported('factor outside configured candidates')
    for field in ('ptsm_cap_authorized', 'row_coalescing_applied'):
        if metadata.get(field):
            raise Unsupported('additional grid transform ' + field)
    if metadata.get('coalesce_factor', 1) != 1:
        raise Unsupported('additional coalesce transform')
    core = re.search(r'"CUBE_CORE_COUNT", (\d+) : i32', final_ir)
    if not core:
        raise Unsupported('CPU target cube core count missing')
    core_count = int(core.group(1))
    roles = sorted(llvm_modules)
    if roles not in (['aic'], ['aic', 'aiv']):
        raise Unsupported('CV LLVM module roles must be aic or aic+aiv')
    if 'aiv' in roles:
        vectors = re.search(r'"VECTOR_CORE_COUNT", (\d+) : i32', final_ir)
        if not vectors or int(vectors.group(1)) != 2 * core_count:
            raise Unsupported('only one Cube with two Vector subblocks is modelled')
    args, grid = launch_arguments(ttir, contract, axis, factor)
    tasks, auto = math.prod(grid), metadata['auto_blockify_enabled']
    blocks = min(tasks, core_count) if auto else tasks
    if blocks <= 0 or blocks > 10000:
        raise Unsupported('launch size outside bounded CV analysis')
    counts, traces, steps = Counter(), [], 0
    read_bytes = write_bytes = 0
    for block in range(blocks):
        graph = EventGraph()
        group_counts = Counter()
        group_read_bytes = group_write_bytes = 0
        participants = [('aic', None)] + ([('aiv', 0), ('aiv', 1)] if 'aiv' in roles else [])
        visits = {}
        for role, subblock in participants:
            name = role if subblock is None else role + str(subblock)
            schedule = CVSchedule(graph, name, assumptions)
            analyzer = Analyzer(llvm_modules[role], table, assumptions, kernel_factory=lambda _, s=schedule: s,
                                subblock_id=subblock)
            entry = metadata['name'] + ('_mix_' + role if len(roles) > 1 else '')
            if entry not in analyzer.functions:
                raise Unsupported('missing CV entry: ' + entry)
            trace = analyzer.trace(entry, args, block)
            counts.update(trace['counts'])
            group_counts.update(trace['counts'])
            read_bytes += trace['dma_read_bytes']
            write_bytes += trace['dma_write_bytes']
            group_read_bytes += trace['dma_read_bytes']
            group_write_bytes += trace['dma_write_bytes']
            steps += trace['scalar_trace_steps']
            visits[name] = trace['block_visits']
        result = graph.finish(retain_events)
        result.update(block=block, entry_block_visits=visits, counts=dict(group_counts),
                      dma_read_bytes=group_read_bytes, dma_write_bytes=group_write_bytes)
        traces.append(result)
    cores = [0.] * min(core_count, blocks)
    for index, trace in enumerate(traces):
        slot = index if auto else min(range(len(cores)), key=cores.__getitem__)
        cores[slot] += trace['cycles']
    memory_bound = (read_bytes + write_bytes) / assumptions['global_bytes_per_cycle']
    return {
        'status': 'STATIC_COST_ESTIMATED', 'score': max(max(cores), memory_bound), 'model': VERSION, 'mode':
        'Cube+Vector' if len(roles) > 1 else 'Cube', 'critical_group_cycles': max(cores), 'global_memory_bound':
        memory_bound, 'group_cycles': cores, 'cube_core_count': core_count, 'logical_grid': grid, 'launched_groups':
        blocks, 'auto_blockify': auto, 'active_groups': min(blocks, core_count), 'scheduling':
        'compiled grid-stride loop' if auto else 'greedy group assignment estimate', 'dma_read_bytes': read_bytes,
        'dma_write_bytes': write_bytes, 'counts': dict(counts), 'llvm_steps': steps, 'group_traces': traces,
        'assumptions': assumptions, 'device_executed': False, 'numerical_correctness_verified': False
    }
