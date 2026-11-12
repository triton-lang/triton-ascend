"""Finite-window list scheduling of optimized SIMD instruction dependencies.

Dispatch and execution are separate. A waiting consumer does not stall later
independent work. Latencies, issue widths and queue capacity remain explicit
model assumptions; this is not a final machine scheduler.
"""
from collections import Counter
import heapq
import math
import re

import backend_cost_model as backend


class ReadyVectorSchedule:

    def __init__(self, table, assumptions, model):
        self.table, self.assumptions, self.model = table, assumptions, model
        self.ready = {}
        self.counts = Counter()
        self.nodes = []
        self.finish = 0.
        self.control_ref = frozenset()
        self.retirement = []
        self.scheduled = {}
        window = assumptions['issue_queue_length']
        if type(window) is not int or window <= 0:
            raise backend.Unsupported('invalid SIMD issue queue length')

    def dependencies(self, rhs):
        deps = set()
        for name in re.findall(backend.SSA, rhs):
            value = self.ready.get(name, ())
            if isinstance(value, (tuple, set, frozenset)):
                deps.update(value)
        return frozenset(deps)

    def node(self, rhs, dest, unit, pipe, interval, latency, slots):
        index = len(self.nodes)
        deps = set(self.dependencies(rhs)) | set(self.control_ref)
        # In-order retirement bounds the amount of unfinished work in flight.
        # Waiting for every instruction older than the window is conservative.
        window = self.assumptions['issue_queue_length']
        ordinal = len(self.retirement)
        if ordinal >= window:
            deps.add(self.retirement[ordinal - window])
        self.nodes.append(
            dict(deps=deps, unit=unit, pipe=pipe, interval=interval, latency=latency, slots=slots,
                 earliest=ordinal * .5))
        retired = len(self.nodes)
        self.nodes.append(dict(deps={index} | set(self.retirement[-1:]), unit=None, earliest=0.))
        self.retirement.append(retired)
        if dest:
            self.ready[dest] = frozenset((index, ))
        return index

    def add(self, name, rhs, dest):
        op = name.removeprefix('llvm.hivm.').split('.')[0]
        info = self.model.instruction(name)
        if op.startswith('vld'):
            unit = pipe = 'load'
            interval, latency, slots, source = 1., self.assumptions[
                'vector_load_latency'], 2, 'retained_load_assumption'
        elif op.startswith('vst'):
            unit = pipe = 'store'
            interval, latency, slots, source = 1., self.assumptions[
                'vector_store_latency'], 1, 'retained_store_assumption'
        elif info is not None:
            rate = info[1]
            unit, pipe = rate['unit'], rate.get('pipe') or rate['unit']
            interval, slots = rate['throughput_cycles'], rate['issue_slots']
            latency = backend.instruction_cost(name, self.table)[1]
            source = 'skill_throughput_retained_latency'
        else:
            interval, latency, slots, source = backend.instruction_cost(name, self.table)
            unit = pipe = 'vector_alu'
            source = 'retained_' + source
        self.node(rhs, dest, unit, pipe, interval, latency, int(slots))
        self.counts[name] += 1
        self.counts['cost_source:' + source] += 1

    def control(self, rhs, dest):
        # Cast/aggregate aliases do not introduce an executing instruction here.
        op = rhs.split()[0]
        if op in ('bitcast', 'extractvalue', 'insertvalue'):
            return
        index = self.node(rhs, dest, 'vf_control', 'vf_control', 1., 1., 1)
        self.counts['vf_control:' + op] += 1
        self.counts['cost_source:optimized_vf_control_assumption'] += 1
        if op == 'br':
            self.control_ref = frozenset((index, ))

    def finalize(self):
        nodes = self.nodes
        children = [[] for _ in nodes]
        remaining = [len(n['deps']) for n in nodes]
        earliest = [n['earliest'] for n in nodes]
        ready = []
        for i, n in enumerate(nodes):
            for dep in n['deps']:
                if not 0 <= dep < i:
                    raise backend.Unsupported('invalid dynamic SIMD dependency')
                children[dep].append(i)
            if not remaining[i]:
                heapq.heappush(ready, (earliest[i], i))
        units, pipes, completed = {}, {}, 0
        issue = 0.
        while ready:
            time, i = heapq.heappop(ready)
            n = nodes[i]
            if n['unit'] is None:
                completed += 1
                for child in children[i]:
                    earliest[child] = max(earliest[child], time)
                    remaining[child] -= 1
                    if remaining[child] == 0:
                        heapq.heappush(ready, (earliest[child], child))
                continue
            resource = pipes.setdefault((n['unit'], n['pipe']), [0., 0.])
            slot = min(range(n['slots']), key=resource.__getitem__)
            start = max(time, issue, units.get(n['unit'], 0.), resource[slot])
            if ready and start > ready[0][0]:
                heapq.heappush(ready, (start, i))
                continue
            resource[slot] = start + n['interval']
            units[n['unit']] = start + 1. / n['slots']
            issue = start + .5
            finish = start + n['latency']
            self.scheduled[i] = (start, finish)
            self.finish = max(self.finish, finish)
            completed += 1
            for child in children[i]:
                earliest[child] = max(earliest[child], finish)
                remaining[child] -= 1
                if remaining[child] == 0:
                    heapq.heappush(ready, (earliest[child], child))
        if completed != len(nodes) or not math.isfinite(self.finish):
            raise backend.Unsupported('SIMD dependency schedule did not finish')
