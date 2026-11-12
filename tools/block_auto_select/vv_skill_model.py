"""A5 VV instruction-throughput model with SSA dependencies and shared DMA.

The vendored skill engine/data are unchanged. Instruction rates come from that
engine; dependency latencies and noncovered native instructions retain the
existing, explicitly reported assumptions. No kernel timings are inputs.
"""
from collections import Counter
import configparser
from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import re

import backend_cost_model as backend
from vv_contention_model import ContentionProfile, DependencySchedule, schedule

VERSION = "vv-skill-lowered-entry-v6-trial"
DATA = Path(__file__).with_name("vv_skill_data")
API = {
    "vadd": "Add",
    "vadds": "Adds",
    "vsub": "Sub",
    "vmul": "Mul",
    "vmuls": "Muls",
    "vdiv": "Div",
    "vsqrt": "Sqrt",
    "vexp": "Exp",
    "vexpdif": "ExpSub",
    "vln": "Ln",
    "vmax": "Max",
    "vmaxs": "Maxs",
    "vmin": "Min",
    "vmins": "Mins",
    "vabs": "Abs",
    "vneg": "Neg",
    "vand": "And",
    "vor": "Or",
    "vxor": "Xor",
    "vnot": "Not",
    "vsel": "Select",
    "vcmp": "Compare",
    "vcadd": "ReduceSum",
    "vcmax": "ReduceMax",
    "vbr": "Duplicate",
    "vdups": "Duplicate",
}


class InstructionModel:

    def __init__(self, data=DATA):
        self.data = Path(data)
        spec = importlib.util.spec_from_file_location("a5_vv_roofline_engine", self.data / "engine.py")
        self.engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.engine)
        self.db = self.engine.load_json(self.data / "vector_instruction_types.json")
        self.hardware = self.engine.load_json(self.data / "hardware-local.json")

    @lru_cache(maxsize=1024)
    def instruction(self, name):
        op = name.removeprefix("llvm.hivm.").split(".")[0]
        if name == "llvm.hivm.vcvtff.bf162f32.x":
            inst = dict(op="Cast", src_dtype="bf16", dst_dtype="fp32", elements=64)
        else:
            if op not in API:
                return None
            types = re.findall(r"\.v(\d+)(bf16|f\d+|i\d+|s\d+)", name)
            # Predicate result widths are not arithmetic operand dtypes.
            types = [item for item in types if item[1] != "i1"]
            if not types:
                return None
            _, dtype = types[0]
            dtype = "fp" + dtype[1:] if dtype.startswith("f") else dtype
            dtype = "int" + dtype[1:] if dtype.startswith(("i", "s")) else dtype
            inst = dict(op=API[op], dtype=dtype)
            if op == "vbr":
                inst["mode"] = "tensor"
        try:
            entry, model_name = self.engine.resolve_instruction_entry(inst, self.db)
            self.engine.validate_instruction_dtypes(inst, entry)
            model = self.engine.resolve_execution_model(inst, entry, model_name, self.db)
            width = self.engine.instruction_dtype_width(inst, model, self.db)
        except self.engine.EvalError as exc:
            raise backend.Unsupported(f"skill instruction {name}: {exc}") from exc
        slots, interval = model["issue_slots"], model["throughput_cycles"]
        if slots not in (1, 2) or not math.isfinite(interval) or interval <= 0 or width <= 0:
            raise backend.Unsupported("invalid skill instruction rate: " + name)
        # A traced native instruction is one SIMD instruction, even when the
        # high-level tensor is larger or the last instruction has masked lanes.
        inst["elements"] = model["bytes_per_instruction"] / width
        return inst, model

    def vector_factory(self, table, assumptions):
        return SkillVectorSchedule(table, assumptions, self)


class SkillVectorSchedule(backend.VectorSchedule):

    def __init__(self, table, assumptions, model):
        super().__init__(table, assumptions)
        self.model = model
        self.unit_issue = {}

    def add(self, name, rhs, dest):
        op = name.removeprefix("llvm.hivm.").split(".")[0]
        info = self.model.instruction(name)
        if op.startswith("vld"):
            unit, pipe, interval, latency, slots, source = ("load", "load", 1., self.assumptions["vector_load_latency"],
                                                            2, "retained_load_assumption")
        elif op.startswith("vst"):
            unit, pipe, interval, latency, slots, source = ("store", "store", 1.,
                                                            self.assumptions["vector_store_latency"], 1,
                                                            "retained_store_assumption")
        elif info is not None:
            _, model = info
            unit, pipe = model["unit"], model.get("pipe") or model["unit"]
            interval, slots = float(model["throughput_cycles"]), int(model["issue_slots"])
            _, latency, _, _ = backend.instruction_cost(name, self.table)
            source = "skill_throughput_retained_latency"
        else:
            interval, latency, slots, old_source = backend.instruction_cost(name, self.table)
            slots = int(slots)
            unit = pipe = "vector_alu"
            source = "retained_" + old_source
        self.counts[name] += 1
        self.counts["cost_source:" + source] += 1
        key = (unit, pipe)
        resource = self.slots.setdefault(key, [0.] * slots)
        if len(resource) != slots:
            raise backend.Unsupported("inconsistent skill subpipe issue slots")
        slot = min(range(slots), key=resource.__getitem__)
        dependency = max((self.ready.get(x, 0.) for x in re.findall(backend.SSA, rhs)), default=0.)
        queue = sorted(t for t in self.pending if t > self.issue)
        queue_ready = queue[0] if len(queue) >= self.assumptions["issue_queue_length"] else 0.
        start = max(self.issue, self.unit_issue.get(unit, 0.), resource[slot], dependency, queue_ready)
        resource[slot] = start + interval
        # Different subpipes share their unit's issue port. Long transcendental
        # occupancy must not serialize independent one-cycle ALU operations.
        self.unit_issue[unit] = start + 1. / slots
        self.issue = start + .5  # retain the explicit overall issue approximation
        self.finish = max(self.finish, start + latency)
        self.pending = [t for t in queue if t > start] + [start + latency]
        if dest:
            self.ready[dest] = start + latency


def ini_rates(path):
    cfg = configparser.ConfigParser(strict=False)
    cfg.read(path)
    frequency = float(cfg["VectorCoreSpec"]["vec_freq"]) * 1e6
    cores = int(cfg["SoCInfo"]["vector_core_cnt"])
    bandwidth = float(cfg["VectorCoreMemoryRates"]["ddr_rate"]) * 1e9
    if cores <= 0 or any(not math.isfinite(v) or v <= 0 for v in (frequency, bandwidth)):
        raise ValueError("invalid INI frequency/core count/bandwidth")
    local = bandwidth / frequency
    return dict(frequency_hz=frequency, core_count=cores, local_bytes_per_cycle=local,
                global_bytes_per_cycle=local * cores,
                provenance="INI ddr_rate interpreted as GB/s/core; not local bandwidth calibration")


def roofline_diagnostic(model, result, traces):
    """Unmodified engine: partial-work ideal bound, never the selection score."""
    mapped, omitted = {}, {}
    for name, count in result["counts"].items():
        if not name.startswith("llvm."):
            continue
        info = model.instruction(name)
        if info is None:
            omitted[name] = count
        else:
            mapped[name] = dict(info[0], count=count)
    if not mapped:
        return dict(status="UNSUPPORTED", reason="no mapped arithmetic instructions")
    case = dict(case_id="compiled-candidate", hardware_variant="ascend950pr_9579_local_ini",
                gm_bandwidth_model="vector_core_conservative", active_vector_cores=result["core_count"],
                memory=dict(gm_read_bytes=result["dma_read_bytes"],
                            gm_write_bytes=result["dma_write_bytes"]), vector=dict(instructions=list(mapped.values())),
                target_efficiency=1., overlap_model="max_pipe", fill_drain=False)
    evaluated = model.engine.evaluate_case(case, model.hardware, model.db)
    durations = []
    for trace in traces:
        instructions = [
            dict(model.instruction(n)[0], count=c)
            for n, c in trace["counts"].items()
            if n.startswith("llvm.") and model.instruction(n) is not None
        ]
        local = dict(case, active_vector_cores=1, memory={}, vector=dict(instructions=instructions))
        durations.append(
            model.engine.evaluate_case(local, model.hardware, model.db)["vector_time_us"] if instructions else 0.)
    cores = [0.] * min(result["core_count"], len(durations))
    if result["auto_blockify"]:
        critical = max(durations)
    else:
        for duration in durations:
            cores[min(range(len(cores)), key=cores.__getitem__)] += duration
        critical = max(cores)
    return dict(status="PARTIAL_WORK_IDEAL_BOUND", microseconds=max(evaluated["mte2_time_us"],
                                                                    evaluated["mte3_time_us"], critical),
                omitted_instructions=omitted, includes_scalar=False, includes_dependency_latency=False,
                includes_dma_startup=False, includes_register_access=False, used_for_selection=False)


def analyze_candidate(llvm, ttir, final_ir, metadata, contract, axis, factor, table, *, model=None, bandwidth="ini",
                      diagnostic=True, library_evidence=None, mixed_evidence=None, optimized_evidence=None,
                      lowered_entry=False):
    if factor not in (1, 2, 4):
        raise backend.Unsupported("only unmerged/2/4 candidates are supported")
    if metadata.get("target", {}).get("arch") != "Ascend950PR_9579":
        raise backend.Unsupported("skill hardware profile requires Ascend950PR_9579")
    model = model or InstructionModel()
    rates = ini_rates(model.data / "Ascend950PR_9579.ini")
    if bandwidth not in ("ini", "legacy"):
        raise ValueError("unknown bandwidth model")
    assumptions = {"dma_bytes_per_cycle": rates["local_bytes_per_cycle"]} if bandwidth == "ini" else {}
    profile = ContentionProfile(
        **({"global_bytes_per_cycle": rates["global_bytes_per_cycle"]} if bandwidth == "ini" else {}))
    library = None
    mixed = None
    vector_factory = model.vector_factory
    optimized_provenance = None
    simt_helpers = None
    original_llvm = llvm
    if optimized_evidence:
        from vv_optimized_ir import verified_vectors
        from vv_ready_schedule import ReadyVectorSchedule
        llvm, optimized_provenance = verified_vectors(llvm, optimized_evidence)
        vector_factory = lambda t, a: ReadyVectorSchedule(t, a, model)
    if lowered_entry:
        if not optimized_provenance: raise backend.Unsupported('lowered entry requires optimized proof')
        from vv_lowered_entry import validate_entry
        llvm = Path(optimized_evidence['optimized_llvm']['path']).read_text()
        validate_entry(original_llvm, llvm, metadata['name'])
    kernel_factory = DependencySchedule
    if re.search(r"\bcall\b[^\n]*@pow_fp32\(", llvm):
        from vv_library_model import LibraryModel, LibrarySchedule
        library = LibraryModel(table, assumptions, vector_factory, library_evidence)
        kernel_factory = lambda config: LibrarySchedule(config, library)
    if metadata.get('parallel_mode') == 'mix_simd_simt':
        from vv_mixed_model import MixedModel, MixedSchedule
        mixed = MixedModel(model, mixed_evidence)
        if not optimized_provenance:
            vector_factory = mixed.vector_factory
        else:
            from vv_simt_helpers import VerifiedSimtHelpers
            simt_helpers = VerifiedSimtHelpers(
                Path(optimized_evidence['optimized_llvm']['path']).read_text(), assumptions, mixed.parameters,
                lowered_abi=lowered_entry)
        kernel_factory = lambda config: MixedSchedule(config, mixed, library, simt_helpers)
    if lowered_entry:
        from vv_lowered_entry import LoweredEntrySchedule
        kernel_factory = lambda config: LoweredEntrySchedule(config, mixed, library, simt_helpers)
    mixed_validator = (lambda text, ir: mixed.validate(original_llvm, ir)
                       ) if mixed and lowered_entry else mixed.validate if mixed else None
    result = backend.analyze_candidate(llvm, ttir, final_ir, metadata, contract, axis, factor, table, assumptions,
                                       kernel_factory=kernel_factory, vector_factory=vector_factory,
                                       include_traces=True, mixed_validator=mixed_validator,
                                       lowered_entry=lowered_entry)
    if result["core_count"] != rates["core_count"]:
        raise backend.Unsupported("compiled core count differs from skill INI")
    traces = result.pop("block_traces")
    if library:
        result["library_provenance"] = library.provenance
    if mixed:
        result['experimental_mixed_model'] = optimized_provenance is None
        result['mixed_provenance'] = mixed.provenance
        result['mixed_linked_variant'] = mixed.variant
        result['mixed_coverage'] = 'bounded AIV SIMD/SIMT helpers; conservative serialized cost'
    if optimized_provenance:
        result['optimized_vector_provenance'] = optimized_provenance
        result['vector_schedule'] = 'optimized LLVM; dependency-ready finite-window scheduling'
    result['scalar_entry_model'] = 'verified optimized entry' if lowered_entry else 'original scalar entry'
    if simt_helpers and simt_helpers.cache:
        result['linked_simt_helpers'] = simt_helpers.cache
    result.update(schedule(traces, result["core_count"], profile))
    result.update(
        model=VERSION, bandwidth=bandwidth, hardware_rates=rates,
        parameter_status="instruction model plus uncalibrated latency/scalar/DMA/SIMT assumptions", cost_sources={
            k.removeprefix("cost_source:"): v
            for k, v in result["counts"].items()
            if k.startswith("cost_source:")
        })
    if diagnostic:
        try:
            result["roofline"] = roofline_diagnostic(model, result, traces)
        except (model.engine.EvalError, backend.Unsupported) as exc:
            result["roofline"] = dict(status="UNSUPPORTED", reason=str(exc), used_for_selection=False)
    return result
