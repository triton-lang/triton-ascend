"""Bounded AIV SIMD/SIMT cost adapter. No device or empirical timing inputs.

The pinned O2 library is a cost reference for O0/O2 variants of the same
toolchain, not post-link ISA. SIMD helpers serialize their intrinsic work;
SIMT gather counts both mask paths, assumes misses and no warp speedup. These
are conservative modelling policies, not hardware throughput measurements.
"""
import json
from pathlib import Path
import re

import backend_cost_model as backend
from cpu_resource_gate import sha256
from vv_library_model import LibrarySchedule
from vv_skill_model import SkillVectorSchedule

DATA = Path(__file__).with_name('vv_mixed_data')
GATHER = 'indirect_load_nonvolatile_1d_bfloat16_t_int64_t'
LOAD = 'vload_NORM_int64_t_rank1'
STORE = 'masked_store_NORM_B64_int64_t_rank1'
ABI = {
    'cast_int32_t_to_int64_t': 3, 'vmuls_int64_t': 4, 'vbr_int64_t': 2, 'vadd_int64_t': 4, 'vcmp_lt_int64_t': 3,
    'vdups_int64_t': 3
}


def meta_path(ir):
    paths = re.findall(r'hivm\.aiv_bitcode = #hivm\.aiv_bitcode<"([^"]+)">', ir)
    if len(paths) != 1:
        raise backend.Unsupported('MIX needs one final-IR AIV bitcode reference')
    return Path(paths[0])


def attach_mixed_evidence(candidate):
    """Preparation only: declare the library read before installing CPU audit."""
    files = candidate['files']
    ir = Path(files['final_ir']['path']).read_text()
    if sha256(files['final_ir']['path']) != files['final_ir']['sha256']:
        raise ValueError('final IR changed before MIX preparation')
    path = meta_path(ir)
    expected_dir = Path(files['compiler']['path']).resolve().parent.parent / 'lib'
    if path.resolve().parent != expected_dir:
        raise ValueError('MIX library is outside the candidate compiler installation')
    files['aiv_meta'] = dict(path=str(path), sha256=sha256(path))


def static_extent(v, description):
    if type(v) is not int or not 0 <= v < 2**31:
        raise backend.Unsupported('unknown or invalid ' + description)
    return v


def body_lines(fn):
    return [line for lines in fn.blocks.values() for line in lines]


class MixedModel:

    def __init__(self, model, evidence, data=DATA):
        self.model = model
        self.evidence = evidence or {}
        self.provenance = json.loads((data / 'provenance.json').read_text())
        for name, digest in self.provenance['asset_sha256'].items():
            if sha256(data / name) != digest:
                raise backend.Unsupported('frozen MIX asset changed: ' + name)
        self.functions = backend.parse_module((data / 'meta-reference.ll').read_text())
        self.wrappers = backend.parse_module((data / 'wrappers.ll').read_text())
        self.parameters = self.provenance['assumptions']
        for name in ('simt_transaction_bytes', 'simt_transition_cycles', 'simt_instruction_cycles'):
            if type(self.parameters.get(name)) is not int or self.parameters[name] <= 0:
                raise backend.Unsupported('invalid MIX parameter: ' + name)
        self.variant = None

    def validate(self, llvm, final_ir):
        compiler = self.evidence.get('compiler', {})
        if compiler.get('sha256') != self.provenance['compiler_sha256']:
            raise backend.Unsupported('MIX requires the pinned compiler')
        lib = self.evidence.get('aiv_meta', {})
        path = meta_path(final_ir)
        variant = path.name
        if (not lib.get('path') or Path(lib['path']).resolve() != path.resolve()
                or variant not in self.provenance['meta_variants']
                or lib.get('sha256') != self.provenance['meta_variants'][variant]):
            raise backend.Unsupported('MIX needs matching declared AIV meta-library evidence')
        entries = [l for l in final_ir.splitlines() if 'func.func ' in l and 'hacc.entry' in l]
        if len(entries) != 1 or 'hivm.vf_mode<MIX>' not in entries[0] or 'hivm.func_core_type<AIV>' not in entries[0]:
            raise backend.Unsupported('MIX adapter supports one AIV SIMD/SIMT entry only')
        funcs = backend.parse_module(llvm)
        called = set(re.findall(r'\bcall\b[^\n]*?@([\w.$-]+)\(', llvm))
        if any(name in called and name not in funcs for name in self.wrappers):
            raise backend.Unsupported('MIX flattened wrapper definition is missing')
        for name, fn in funcs.items():
            if name.startswith('_mlir_ciface_'):
                raise backend.Unsupported('candidate shadows a MIX library definition: ' + name)
            if name in self.wrappers and fn != self.wrappers[name]:
                raise backend.Unsupported('MIX flattened wrapper changed: ' + name)
        self.variant = variant

    def vector_factory(self, table, assumptions):
        return MixedVectorSchedule(table, assumptions, self.model, self)

    def gather(self, args):
        if len(args) != 25:
            raise backend.Unsupported('unknown SIMT gather ABI')
        sizes = [static_extent(args[i], 'SIMT gather extent') for i in (8, 13, 18, 23)]
        if len(set(sizes)) != 1 or any(args[i] != 1 for i in (9, 14, 19, 24)):
            raise backend.Unsupported('SIMT gather needs equal contiguous 1D UB views')
        for i in (7, 12, 17, 22):
            static_extent(args[i], 'SIMT gather offset')
        n = sizes[0]
        body = self.functions[self.provenance['simt_body']]
        # Deliberately count both mask branches and per-element control. No
        # tensor values, cache hits, or presumed lane utilization enter ranking.
        per_element = sum(
            len(lines) for label, lines in body.blocks.items() if label not in ('entry', 'for.end', 'for.body.lr.ph'))
        fixed = sum(len(body.blocks.get(k, [])) for k in ('entry', 'for.end', 'for.body.lr.ph'))
        descriptor = len(body_lines(self.wrappers[GATHER])) + len(body_lines(self.functions['_mlir_ciface_' + GATHER]))
        return dict(elements=n, instruction_steps=fixed + n * per_element, descriptor_steps=descriptor,
                    payload_bytes_upper=n * 2, transaction_bytes_upper=n * self.parameters['simt_transaction_bytes'])


class MixedVectorSchedule(SkillVectorSchedule):

    def __init__(self, table, assumptions, model, mixed):
        super().__init__(table, assumptions, model)
        self.mixed = mixed

    def extra_call(self, name, args, rhs, dest):
        helper = name.removeprefix('_mlir_ciface_')
        descriptor = 0
        if name in (LOAD, STORE):
            if len(args) != (7 if name == LOAD else 8):
                raise backend.Unsupported('unknown int64 memory helper ABI')
            base = 1 if name == LOAD else 0
            size = static_extent(args[base + 3], 'int64 view extent')
            offset = static_extent(args[base + 5], 'int64 vector offset')
            static_extent(args[base + 2], 'int64 view offset')
            if args[base + 4] != 1 or offset > size:
                raise backend.Unsupported('int64 helper needs a contiguous bounded view')
            descriptor = len(body_lines(self.mixed.wrappers[name]))
        elif name.startswith('_mlir_ciface_') and helper in ABI:
            if len(args) != ABI[helper]:
                raise backend.Unsupported('unknown int64 arithmetic helper ABI')
        else:
            return False
        fn = self.mixed.functions['_mlir_ciface_' + helper]
        lines = body_lines(fn)
        if any(l.startswith(('br ', 'switch ')) for l in lines):
            raise backend.Unsupported('non-straight-line int64 reference helper')
        # Flush before/after each helper. This conservatively preserves memory
        # aliases passed by descriptor, which are not ordinary LLVM SSA edges.
        self.issue = max(self.issue, self.finish)
        steps, intrinsics = descriptor, 0
        for line in lines:
            call = re.search(r'\bcall\b.*?@([\w.$-]+)\(', line)
            if call:
                intrinsic = call.group(1)
                if not intrinsic.startswith('llvm.hivm.'):
                    raise backend.Unsupported('unmodelled nested MIX helper: ' + intrinsic)
                # This reference is fully serialized: use a fresh instruction
                # schedule so helpers with a different issue width do not
                # change the caller's SIMD resource layout.
                operation = SkillVectorSchedule(self.table, self.assumptions, self.model)
                operation.add(intrinsic, '', None)
                self.finish = max(self.issue, self.finish) + operation.finish
                self.counts.update(operation.counts)
                intrinsics += 1
            elif not line.startswith('ret '):
                steps += 1
        self.finish = max(self.finish, self.issue) + steps * self.assumptions['scalar_instruction_cycles']
        self.issue = self.finish
        if dest:
            self.ready[dest] = self.finish
        self.counts['mixed_vector_helper_calls:' + helper] += 1
        self.counts['cost_source:mixed_serialized_reference_intrinsics'] += intrinsics
        self.counts['cost_source:mixed_descriptor_steps_upper'] += steps
        return True


class MixedSchedule(LibrarySchedule):

    def __init__(self, assumptions, mixed, library=None, simt_helpers=None):
        super().__init__(assumptions, library)
        self.mixed = mixed
        self.simt_helpers = simt_helpers

    def extra_call(self, name, args):
        work = self.simt_helpers.profile(name, args) if self.simt_helpers else None
        if work is not None:
            ref = self.event('simt_helper', [self.scalar_ref, *self.pipe_refs.values()], work['cycles'],
                             bytes=work['transaction_bytes_upper'], startup=work['startup'])
            self.scalar_ref = ref
            self.pipe_refs = dict.fromkeys(self.pipe_refs, ref)
            finish = max(self.scalar, *self.pipes.values()) + work['cycles']
            self.scalar = finish
            self.pipes = dict.fromkeys(self.pipes, finish)
            self.counts['linked_simt_calls:' + name] += 1
            self.counts['cost_source:verified_linked_simt_warp_issue'] += work['warp_counts']['instruction_steps']
            for key, value in work['counts'].items():
                self.counts['linked_simt_' + key] += value
            self.counts['linked_simt_work_items'] += work['work_items']
            self.counts['linked_simt_warp_issued_steps'] += work['warp_counts']['instruction_steps']
            return True
        if name != GATHER:
            return super().extra_call(name, args)
        work = self.mixed.gather(args)
        p = self.mixed.parameters
        # Join prior SIMD, scalar and DMA work at the transition. The extra
        # serialization is an explicit policy until hardware overlap is known.
        startup = (2 * p['simt_transition_cycles'] +
                   work['descriptor_steps'] * self.assumptions['scalar_instruction_cycles'])
        duration = startup + work['instruction_steps'] * p['simt_instruction_cycles']
        duration += work['elements'] * (3 * self.assumptions['vector_load_latency'] +
                                        self.assumptions['vector_store_latency'])
        duration += work['transaction_bytes_upper'] / self.assumptions['dma_bytes_per_cycle']
        ref = self.event('simt_gather', [self.scalar_ref, *self.pipe_refs.values()], duration,
                         bytes=work['transaction_bytes_upper'], startup=startup)
        self.scalar_ref = ref
        self.pipe_refs = dict.fromkeys(self.pipe_refs, ref)
        finish = max(self.scalar, *self.pipes.values()) + duration
        self.scalar = finish
        self.pipes = dict.fromkeys(self.pipes, finish)
        self.counts['simt_gather_calls'] += 1
        self.counts['simt_elements_upper'] += work['elements']
        self.counts['simt_gm_payload_bytes_upper'] += work['payload_bytes_upper']
        self.counts['simt_gm_transaction_bytes_upper'] += work['transaction_bytes_upper']
        self.counts['cost_source:simt_serialized_instruction_steps_upper'] += work['instruction_steps']
        self.counts['cost_source:simt_transition_assumption'] += 2
        return True
