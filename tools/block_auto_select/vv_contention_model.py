"""CPU-only VV ranking with explicit DMA contention and block dispatch.

The scalar interpreter supplies a dependency graph, including the actual
AutoBlockify loop and flag dependencies. This module schedules those graphs
against shared command-issue and memory-bandwidth limits. It never consumes
kernel names, empirical timing labels, tensor values, or a device runtime.

Default rates are EXPERIMENTAL assumptions, not measured A5 cycle costs. Scores
rank candidates; they are not predictions in microseconds. Numerical correctness
and final-binary resource proofs remain separate gates.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import heapq
import math

from backend_cost_model import KernelSchedule, Unsupported
from backend_cost_model import analyze_candidate as trace_candidate
from backend_cost_model import choose as choose_static

VERSION = "vv-shared-dma-dispatch-v1"


@dataclass(frozen=True)
class ContentionProfile:
    # Preserve the existing 512-byte aggregate bandwidth assumption. The new
    # issuer interval represents four command slots per old 32-cycle startup
    # window. Dispatch uses that same startup scale. Neither is a hardware fact.
    global_bytes_per_cycle: float = 512.0
    dma_issue_interval: float = 8.0
    block_dispatch_interval: float = 32.0

    def __post_init__(self):
        for name, val in asdict(self).items():
            if isinstance(val, bool) or not math.isfinite(val) or val < 0:
                raise ValueError(f"invalid profile value: {name}")
        if self.global_bytes_per_cycle <= 0:
            raise ValueError("global bandwidth must be positive")


class DependencySchedule(KernelSchedule):
    """Keep legacy counts and additionally record the same local dependencies.

    References are (event index, delay). -1 is block entry. Scalar-only steps
    accumulate as delays, avoiding a node for every LLVM SSA bookkeeping op.
    """

    def __init__(self, assumptions):
        super().__init__(assumptions)
        self.events = []
        self.pipe_refs = {pipe: (-1, 0.0) for pipe in self.pipes}
        self.scalar_ref = (-1, 0.0)
        self.flag_refs = {}

    def event(self, kind, refs, duration=0.0, **fields):
        deps = {}
        for parent, delay in refs:
            deps[parent] = max(deps.get(parent, 0.0), delay)
        index = len(self.events)
        self.events.append(dict(kind=kind, dependencies=sorted(deps.items()), duration=duration, **fields))
        return (index, 0.0)

    def scalar_step(self):
        super().scalar_step()
        parent, delay = self.scalar_ref
        self.scalar_ref = (parent, delay + self.assumptions["scalar_instruction_cycles"])
        self.pipe_refs[0] = self.scalar_ref

    def vf(self, trace):
        super().vf(trace)
        self.pipe_refs[1] = self.event("vector", [self.scalar_ref, self.pipe_refs[1]],
                                       trace["cycles"] + self.assumptions["vf_call_cycles"])

    def dma(self, direction, size):
        if direction not in ("load", "store") or size < 0:
            raise Unsupported("invalid DMA direction or extent")
        super().dma(direction, size)
        pipe = 4 if direction == "load" else 5
        startup = self.assumptions["dma_startup_cycles"]
        self.pipe_refs[pipe] = self.event("dma", [self.scalar_ref, self.pipe_refs[pipe]],
                                          startup + size / self.assumptions["dma_bytes_per_cycle"], direction=direction,
                                          bytes=size, startup=startup)

    def flag(self, name, args):
        super().flag(name, args)  # retain unknown/unmatched-flag rejection
        src, dst, ident = map(int, args)
        key = (src, dst, ident)
        if ".SET." in name:
            self.flag_refs[key] = self.event("set", [self.scalar_ref, self.pipe_refs[src]])
        else:
            self.pipe_refs[dst] = self.event(
                "wait",
                [self.scalar_ref, self.pipe_refs[dst], self.flag_refs.pop(key)], self.assumptions["flag_cycles"])
            if dst == 0:
                self.scalar_ref = self.pipe_refs[0]

    def barrier(self, args):
        super().barrier(args)
        self.scalar_ref = self.event("barrier", [self.scalar_ref, *self.pipe_refs.values()])

    def result(self):
        result = super().result()
        terminal, _ = self.event("end", [self.scalar_ref, *self.pipe_refs.values()])
        result.update(events=self.events, terminal=terminal)
        return result


def schedule(traces, core_count, profile=None):
    """List schedule independent blocks; ready DMAs share a FIFO command path.

    Bandwidth service and local DMA latency may overlap, so completion takes
    their maximum rather than adding the same transfer cost twice. A consumer
    waits for completion through the recorded flag edges. When physical blocks
    exceed core count, the next block enters only after a core becomes free.
    """
    profile = profile or ContentionProfile()
    if core_count <= 0 or not traces:
        raise Unsupported("empty launch or invalid core count")
    prepared = []
    for trace in traces:
        events = trace["events"]
        if trace["terminal"] != len(events) - 1 or events[-1]["kind"] != "end":
            raise Unsupported("invalid terminal event")
        children = defaultdict(list)
        for idx, event in enumerate(events):
            if not math.isfinite(event["duration"]) or event["duration"] < 0:
                raise Unsupported("invalid event duration")
            if not event["dependencies"]:
                raise Unsupported("event without block-entry dependency")
            for parent, delay in event["dependencies"]:
                if parent < -1 or parent >= idx or not math.isfinite(delay) or delay < 0:
                    raise Unsupported("invalid dependency graph")
                children[parent].append((idx, delay))
        prepared.append((children, [len(e["dependencies"]) for e in events], [0.0] * len(events)))

    ready, core_for_block = [], {}
    completion = [None] * len(traces)
    releases = [None] * len(traces)
    next_block, dispatch_free, issue_free, bus_free = 0, 0.0, 0.0, 0.0
    totals = Counter()

    def release_children(block, parent, finish):
        children, remaining, earliest = prepared[block]
        for child, delay in children[parent]:
            earliest[child] = max(earliest[child], finish + delay)
            remaining[child] -= 1
            if remaining[child] == 0:
                heapq.heappush(ready, (earliest[child], block, child))

    def start_block(core, available):
        nonlocal next_block, dispatch_free
        if next_block == len(traces):
            return
        block = next_block
        next_block += 1
        start = max(available, dispatch_free)
        dispatch_free = start + profile.block_dispatch_interval
        releases[block] = start
        core_for_block[block] = core
        release_children(block, -1, start)

    for core in range(min(core_count, len(traces))):
        start_block(core, 0.0)
    while ready:
        earliest, block, idx = heapq.heappop(ready)
        event = traces[block]["events"][idx]
        finish = earliest + event["duration"]
        if event["kind"] == "dma":
            issue = max(earliest, issue_free)
            issue_free = issue + profile.dma_issue_interval
            payload_ready = issue + event["startup"]
            payload_start = max(payload_ready, bus_free)
            bus_free = payload_start + event["bytes"] / profile.global_bytes_per_cycle
            finish = max(issue + event["duration"], bus_free)
            totals["dma_issue_wait_cycles"] += issue - earliest
            totals["dma_bandwidth_wait_cycles"] += payload_start - payload_ready
            totals["dma_requests"] += 1
            totals["dma_bytes"] += event["bytes"]
        elif event["kind"] == "simt_gather" or (event["kind"] == "simt_helper" and event["bytes"] > 0):
            # SIMT loads share GM bandwidth but do not use the MTE command queue.
            payload_ready = earliest + event["startup"]
            payload_start = max(payload_ready, bus_free)
            bus_free = payload_start + event["bytes"] / profile.global_bytes_per_cycle
            finish = max(finish, bus_free)
            totals["simt_bandwidth_wait_cycles"] += payload_start - payload_ready
            totals["simt_gm_transaction_bytes_upper"] += event["bytes"]
        if event["kind"] == "end":
            completion[block] = finish
            start_block(core_for_block[block], finish)
        release_children(block, idx, finish)
    if any(value is None for value in completion):
        raise Unsupported("dependency graph did not complete")
    return dict(score=max(completion), block_completion=completion, block_release=releases, profile=asdict(profile),
                model=VERSION, cost_units="model units; not measured cycles", dma_service_end=bus_free, **dict(totals))


def analyze_candidate(llvm, ttir, final_ir, metadata, contract, axis, factor, table, *, profile=None, assumptions=None,
                      keep_traces=False):
    profile = profile or ContentionProfile()
    result = trace_candidate(llvm, ttir, final_ir, metadata, contract, axis, factor, table, assumptions,
                             kernel_factory=DependencySchedule, include_traces=True)
    traces = result.pop("block_traces")
    result["legacy_score"] = result["score"]
    result.update(schedule(traces, result["core_count"], profile))
    result["event_count"] = sum(len(trace["events"]) for trace in traces)
    if keep_traces:
        result["block_traces"] = traces
    return result


def choose(candidates):
    # Fail closed even for malformed scores. The runner additionally rechecks
    # final IR, compiler and binary hashes; a small score never bypasses UB.
    eligible = {
        key: value
        for key, value in candidates.items()
        if isinstance(value.get("score"), (int, float)) and math.isfinite(value["score"]) and value["score"] >= 0
    }
    return choose_static(eligible)
