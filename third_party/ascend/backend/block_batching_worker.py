# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""One candidate per interpreter. The parent alone binds and publishes artifacts."""
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

_PROTOCOL = 1
_TIMEOUT = 600
_TERMINATION_GRACE = 0.5
_MAX_REPLY = 32 * 1024 * 1024
_LOG_TAIL = 16384
_MANIFEST = 'block_batching.artifacts'
_BOOTSTRAP = ('import resource; resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); '
              'import site; site.main(); from triton.backends.ascend.block_batching_worker import _worker; _worker()')


class BatchCompileError(RuntimeError):

    def __init__(self, message, *, kind='InternalError', stage='compile'):
        self.kind, self.stage = kind, stage
        super().__init__(f'block batching [{stage}/{kind}]: {message}')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _stage(directory):
    try:
        return (directory / 'stage').read_text()[:128] or 'startup'
    except FileNotFoundError:
        return 'startup'


def _stop_group(process):
    # The vendor compiler may outlive the Python worker. Check the group even
    # when its leader exited first. Reap the directly owned child on every path.
    try:
        os.killpg(process.pid, signal.SIGTERM)
        # Keep the grace period even if the group leader exits immediately;
        # surviving compiler subprocesses still need a chance to stop.
        time.sleep(_TERMINATION_GRACE)
        process.poll()
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        process.wait()


def _run_isolated(command, directory, timeout):
    """Drain diagnostics even after the tail fills, and always wait for the exit status."""
    deadline = time.monotonic() + timeout
    log = bytearray()
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    except OSError as exc:
        raise BatchCompileError(f'cannot start worker: {exc}', stage='startup') from exc
    with process as child:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise subprocess.TimeoutExpired(command, timeout)
                    chunk = os.read(child.stdout.fileno(), 65536)
                    if chunk:
                        log.extend(chunk)
                        del log[:-_LOG_TAIL]
                    else:
                        selector.unregister(child.stdout)
            code = child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            _stop_group(child)
            raise BatchCompileError(
                log.decode(errors='replace') or 'candidate deadline expired', kind='Timeout',
                stage=_stage(directory)) from exc
        except BaseException:
            _stop_group(child)
            raise
        if code:
            _stop_group(child)
            kind = 'Signal' if code < 0 else 'InternalError'
            raise BatchCompileError(f'worker exit {code}: {log.decode(errors="replace")}', kind=kind,
                                    stage=_stage(directory))
        return log.decode(errors='replace').strip()


class CompileOutcome:
    """Validate the entire operation-specific reply before interpreting its status."""

    def __init__(self, request, reply, diagnostic=''):
        if not isinstance(reply, dict) or any(
                reply.get(k) != request[k] for k in ('protocol', 'id', 'operation', 'identity')):
            raise BatchCompileError('invalid response identity', kind='ProtocolError')
        self.status, self.payload = reply.get('status'), reply.get('payload')
        if not isinstance(self.payload, dict):
            raise BatchCompileError('invalid response payload', kind='ProtocolError')
        operation = request['operation']
        if self.status == 'InternalError':
            kind = self.payload.get('kind')
            if (kind not in ('InternalError', 'InvalidArtifact', 'ProtocolError')
                    or not isinstance(self.payload.get('message'), str)
                    or not isinstance(self.payload.get('stage'), str)):
                raise BatchCompileError('invalid error response', kind='ProtocolError')
            message = self.payload['message']
            if diagnostic:
                message = f'{message}\n{diagnostic}'
            raise BatchCompileError(message, kind=kind, stage=self.payload['stage'])
        if self.status == 'NotApplicable':
            reason = self.payload.get('reason')
            if not isinstance(reason, str) or (reason not in _CAPABILITIES and reason not in _RESOURCE_REASONS):
                raise BatchCompileError(f'unrecognized refusal {reason!r}', kind='ProtocolError')
            if 'manifest' in self.payload or (operation != 'compile' and reason in _RESOURCE_REASONS):
                raise BatchCompileError('refusal does not match operation', kind='ProtocolError')
        elif self.status == 'Success':
            required = 'manifest' if operation == 'compile' else 'report'
            allowed = {'manifest', 'timings'} if operation == 'compile' else {'report'}
            if not isinstance(self.payload.get(required), dict) or self.payload.keys() - allowed:
                raise BatchCompileError(f'invalid {operation} success payload', kind='ProtocolError')
        else:
            raise BatchCompileError('unknown response status', kind='ProtocolError')


def _attempt(operation, body):
    with tempfile.TemporaryDirectory(prefix='triton-batch-worker-') as root:
        directory = Path(root)
        request = dict(protocol=_PROTOCOL, operation=operation, id=uuid4().hex, identity=_digest([operation, body]),
                       body=body)
        (directory / 'request').write_text(json.dumps(request, allow_nan=False))
        diagnostic = _run_isolated([sys.executable, '-S', '-c', _BOOTSTRAP, str(directory)], directory, _TIMEOUT)
        result = directory / 'result'
        try:
            if result.stat().st_size > _MAX_REPLY:
                raise ValueError('worker reply exceeds size limit')
            response = json.loads(result.read_bytes())
        except (OSError, UnicodeError, ValueError) as exc:
            raise BatchCompileError(f'invalid worker reply: {exc}', kind='ProtocolError',
                                    stage=_stage(directory)) from exc
        return CompileOutcome(request, response, diagnostic)


def _report_status(operation, report):
    status = report.get('compile_status')
    if status == ('ANALYZED' if operation == 'analyze' else 'REWRITTEN'):
        return 'Success', dict(report=report)
    if status == 'REJECTED' and report.get('reason') in _CAPABILITIES:
        return 'NotApplicable', dict(reason=report['reason'], report=report)
    raise BatchCompileError(f'unrecognized native {operation} result: {status!r}, {report.get("reason")!r}',
                            stage=operation)


def isolated_report(operation, ttir, factor, axis, policy='semantic_only', cv_batch=False):
    """Analyze immutable source or probe an actual rewrite on a disposable module."""
    if operation not in ('analyze', 'probe') or policy not in ('auto', 'semantic_only') or type(cv_batch) is not bool:
        raise ValueError('invalid block batching report request')
    result = _attempt(operation, dict(ttir=ttir, factor=factor, axis=axis, policy=policy, cv_batch=cv_batch))
    report = result.payload.get('report')
    expected = 'REJECTED' if result.status == 'NotApplicable' else (
        'ANALYZED' if operation == 'analyze' else 'REWRITTEN')
    if (not isinstance(report, dict) or report.get('compile_status') != expected
            or (result.status == 'NotApplicable' and report.get('reason') != result.payload['reason'])
            or report.get('requested_factor') != factor or report.get('requested_axis') != axis
            or report.get('report_version') != 1):
        raise BatchCompileError('invalid native report', kind='ProtocolError', stage=operation)
    return report


def _native_report(operation, body, directory):
    from triton._C.libtriton import ascend, buffer_ir, ir
    from .compiler import distributed
    from .block_batching import _has_dot
    context = ir.context()
    ir.load_dialects(context)
    if distributed is not None:
        distributed.ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend.ir.load_dialects(context)
    ascend.load_dialects(context)
    path = directory / 'source.ttir'
    path.write_text(body['ttir'])
    module = ir.parse_mlir_module(str(path), context)
    cv = _has_dot(module)
    if operation == 'analyze':
        report = ascend.passes.ttir.analyze_block_batching(module, body['factor'], body['axis'], cv)
    else:
        report = ascend.passes.ttir.rewrite_block_batching(module, body['factor'], body['axis'], body['policy'], cv, cv
                                                           and body['cv_batch'])
    return _report_status(operation, dict(report))


# These are refusals of a source operation, proof or supported target mode. A new
# compiler diagnostic must be classified deliberately; generated IR/ABI failures
# (e.g. INVALID_BATCH_PLAN and VALUE_MAPPING_MISSING) must propagate as errors.
_CAPABILITIES = frozenset('''
ALREADY_MAPPED ANALYSIS_BUDGET_EXCEEDED BASIC_CONTROL_FLOW_NOT_IMPLEMENTED BASIC_OP_NOT_IMPLEMENTED CROSS_LANE_REORDER
CV_BATCH_LOAD_LAYOUT_NOT_PROVEN CV_BATCH_MASK_LAYOUT_NOT_PROVEN CV_BATCH_MASK_RECTANGLE_NOT_PROVEN
CV_BATCH_QUOTIENT_GROUP_CROSSES_PERIOD CV_BATCH_VOLATILE_TRANSPOSE_NOT_VALIDATED CV_DEVICE_ADDRESS_NOT_PROVEN
CV_DOT_REQUIRED CV_FACTOR_REQUIRES_POWER_OF_TWO CV_INTEGER_BATCH_MATMUL_UNAVAILABLE CV_LOOP_BATCH_MATMUL_DISABLED
CV_MASKED_DOT_COORDINATES_NOT_PROVEN CV_MASKED_LOAD_REQUIRES_ZERO_OTHER CV_OPT_IN_REQUIRED CV_REQUIRES_RANK_TWO_DOT
CV_SHARED_ACCUMULATOR_NOT_VALIDATED CV_SPATIAL_ACCUMULATOR_AXIS_NOT_PROVEN CV_SPATIAL_BROADCAST_AXIS_NOT_PROVEN
CV_SPATIAL_COORDINATE_NOT_PROVEN CV_SPATIAL_DOT_AXIS_NOT_PROVEN CV_SPATIAL_LOAD_AXIS_NOT_PROVEN CV_SPATIAL_MIXED_AXES
CV_SPATIAL_MULTIPLE_COORDINATES CV_SPATIAL_MULTIPLE_PROGRAM_IDS CV_SPATIAL_OPERATION_NOT_SUPPORTED
CV_SPATIAL_PROGRAM_USE_NOT_PROVEN CV_SPATIAL_RECURRENCE_BUDGET_EXCEEDED CV_SPATIAL_REQUIRES_SINGLE_DOT
CV_SPATIAL_REQUIRES_ZERO_ACCUMULATOR CV_SPATIAL_SCALAR_RESULT_NOT_SUPPORTED CV_SPATIAL_STORE_NOT_CONCATENATED
CV_SPATIAL_UNIFORM_INIT_NOT_PROVEN CV_SPATIAL_UNIFORM_VALUE_NOT_PROVEN CV_SPATIAL_UNIFORM_YIELD_NOT_PROVEN
CV_SPATIAL_VARYING_CONTROL CV_STATIC_LANE_CODE_SIZE_EXCEEDED CV_UNSTRUCTURED_MEMORY_MASK CV_UNSUPPORTED_DOT_TYPE
CV_VARYING_LOOP_BOUND CV_BATCH_PADDING_NOT_PROVEN CV_BATCH_REDUCTION_MASK_NOT_PROVEN CV_MASKED_BATCH_MEMORY_LOWERING_UNSUPPORTED
CV_AUTO_BLOCKIFY_NOT_PROVEN CV_GLOBAL_SYNC_NOT_SUPPORTED EXPECTED_SINGLE_PUBLIC_ENTRY FACTOR_EXCEEDS_LANE_CAPACITY
FACTOR_OUT_OF_RANGE G0_MAPPING_CONFLICT HIDDEN_ARGUMENT_CONFLICT INACTIVE_COMBINER_UNSAFE INDEX_LAYOUT_NOT_PROVEN
NESTED_SIGNATURE_NOT_SUPPORTED CUSTOM_CACHE_NOT_SUPPORTED INVALID_AXIS_HINT INVALID_PROGRAM_AXIS INNER_TRIP_COUNT_NOT_INVARIANT LOOP_ACCESS_NOT_PROVEN LOOP_GROUP_CODE_SIZE_EXCEEDED LOOP_MASK_UNIT_MISMATCH
LOOP_NESTING_MISMATCH LOOP_ORIGIN_UNPROVEN LOOP_POINTWISE_MEMORY_VIEW_REQUIRED LOOP_VARIATION_DID_NOT_CONVERGE
LOOP_WITHOUT_MEMORY_ACCESS MASKED_LOAD_REQUIRES_OTHER MASK_NOT_INTERVAL_OR_RECTANGLE MERGED_POINTER_ROOTS
MULTI_AXIS_PROVENANCE_UNPROVEN MULTI_BLOCK_ENTRY_BODY NEW_BYTE_ENDPOINT_UNPROVEN NON_ELEMENT_ALIGNED_LAYOUT
NON_INTEGER_ADDRESS NO_MEMORY_ACCESSES NO_PROGRAM_AXIS POINTER_ROUNDTRIP_NOT_LOWERABLE REGION_DEPTH_EXCEEDED
TENSOR_SHAPE_BUDGET_EXCEEDED UNIFORM_STORE UNKNOWN_POINTER_ROOT UNNORMALIZED_BOUNDARY_CHECK UNPROVEN_LANE_SEGMENT_BOUND
UNPROVEN_LOOP_COALESCING UNPROVEN_MEMORY_NONINTERFERENCE UNPROVEN_POSITIVE_LOOP_STEP UNSUPPORTED_ADDRESS_EXPRESSION
UNSUPPORTED_BATCH_OP UNSUPPORTED_BLOCK_POINTER_FLOW UNSUPPORTED_BLOCK_POINTER_TYPE UNSUPPORTED_BYTE_OPERATION
UNSUPPORTED_CALL UNSUPPORTED_COMBINER_TYPE UNSUPPORTED_EFFECT UNSUPPORTED_ENTRY_SIGNATURE UNSUPPORTED_EXTERNAL_FUNCTION
UNSUPPORTED_INDEX_CAPTURE UNSUPPORTED_INDEX_RESHAPE UNSUPPORTED_LAYOUT_HINT UNSUPPORTED_MODULE_OPERATION
UNSUPPORTED_NAN_REDUCTION_COMBINER UNSUPPORTED_OPERATION_ATTRIBUTE UNSUPPORTED_POINTER_BITCAST UNSUPPORTED_POINTER_TYPE
UNSUPPORTED_REDUCTION UNSUPPORTED_REDUCTION_COMBINER UNSUPPORTED_REDUCTION_RETURN_CAPTURE UNSUPPORTED_REDUCTION_TYPE
UNSUPPORTED_REGION_CAPTURE UNSUPPORTED_REGION_SHAPE UNSUPPORTED_RESULT_SCHEMA UNSUPPORTED_TYPE_OR_RANK UNUSED_BATCH_AXIS
VALUE_VARIATION_UNKNOWN VARYING_CONTROL_CONDITION VARYING_INDEX_NOT_IMPLEMENTED VARYING_LOOP_BOUND
VV_AGGREGATION_ANALYSIS_BUDGET_EXCEEDED VV_AGGREGATION_DOMAIN_NOT_PROVEN VV_MULTI_AXIS_REDUCTION_NOT_SUPPORTED
VV_SCAN_DOMAIN_NOT_SUPPORTED VV_POINTWISE_MEMORY_VIEW_REQUIRED VV_INDIRECT_MEMORY_NOT_SUPPORTED UNSUPPORTED_FINAL_EXECUTION_MODE
'''.split())
_RESOURCE_REASONS = ('LAYOUT_TARGET_RESOURCE_REJECTED', 'CV_TARGET_RESOURCE_REJECTED')


@dataclass(frozen=True)
class BatchDecline:
    """A modeled stage result. Construction rejects unknown and contradictory reasons."""
    reason: str
    stage: str
    analysis: dict = None
    diagnostic: str = None
    factor: int = None
    axis: int = None

    def __post_init__(self):
        if self.reason not in _CAPABILITIES and self.reason not in _RESOURCE_REASONS:
            raise BatchCompileError(self.reason, stage=self.stage)
        if self.reason in _RESOURCE_REASONS and (self.stage != 'npubin' or not self.diagnostic):
            raise BatchCompileError('resource decline without matching evidence', stage=self.stage)

    def payload(self):
        return {key: value for key, value in vars(self).items() if value is not None}


def _classify(error, stage):
    from triton.compiler.errors import MLIRCompilationError
    cause = error
    while isinstance(cause, MLIRCompilationError) and cause.__cause__ is not None:
        cause = cause.__cause__
    evidence = []
    for stream in ('stdout', 'stderr'):
        value = getattr(cause, stream, None)
        if value:
            value = value.decode(errors='replace') if isinstance(value, bytes) else str(value)
            evidence.append(f'{stream}: {value[-_LOG_TAIL:]}')
    return 'InternalError', dict(kind=getattr(cause, 'kind', 'InternalError'), stage=getattr(cause, 'stage', stage),
                                 message='\n'.join([str(cause)[:_LOG_TAIL], *evidence]))


def _variant_key(source, backend, options, env_vars):
    from triton.runtime.cache import get_cache_key
    return hashlib.sha256(get_cache_key(source, backend, options, env_vars).encode()).hexdigest()


def _baseline_environment(source, backend, baseline_options, baseline):
    """Recover the submitted compiler environment from a cold or cached F1 result."""
    if baseline.metadata.hash != baseline.hash:
        raise BatchCompileError('baseline metadata disagrees with its cache key', kind='InvalidArtifact', stage='bind')
    # Triton flattens its uppercase cache environment into the kernel metadata.
    # Prove that projection against F1's actual key before using it in a child.
    env_vars = {key: value for key, value in baseline.metadata._asdict().items() if key.isupper()}
    normalized = backend.parse_options(dict(baseline_options, **source.parse_options()))
    if _variant_key(source, backend, normalized, env_vars) == baseline.hash:
        return env_vars
    return None


def _artifact_manifest(group, directory, variant):
    files = {}
    for name, location in group.items():
        path = Path(location)
        if not path.is_file() or path.is_symlink() or path.resolve() != directory / name or Path(name).name != name:
            raise ValueError(f'invalid candidate artifact: {name}')
        files[name] = dict(size=path.stat().st_size, digest=_file_digest(path))
    return dict(version=_PROTOCOL, variant=variant, files=files)


def _checked_group(manifest, directory, variant, name):
    """Read a complete immutable generation; the cache's get_group can omit missing files."""
    try:
        entries = manifest['files']
        if manifest['version'] != _PROTOCOL or manifest['variant'] != variant or not isinstance(entries, dict):
            raise ValueError('manifest identity')
        if not {f'{name}.json', f'{name}.npubin'}.issubset(entries):
            raise ValueError('missing metadata or binary')
        allowed = {f'{name}.{ext}' for ext in ('json', 'npubin', 'source', 'ttir', 'ttadapter')}
        if not entries.keys() <= allowed:
            raise ValueError('unexpected artifact names')
        group = {key: str(directory / key) for key in entries}
        if _artifact_manifest(group, directory, variant) != manifest:
            raise ValueError('artifact checksum or size changed')
        with (directory / f'{name}.npubin').open('rb') as stream:
            if stream.read(4) != b'\x7fELF':
                raise ValueError('missing ELF binary header')
        return group
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        raise BatchCompileError(str(error), kind='InvalidArtifact', stage='validate') from error


def _knob_overrides(group):
    # A fresh interpreter already inherits the parent's environment. An unset
    # variable and an explicitly set default can mean different things to CANN.
    return {name: value for name, value in vars(group).items() if name in group.knob_descriptors}


def _restore_knobs(knobs, body):
    # Restore the Python view without invoking setters that write to os.environ.
    vars(knobs.compilation).update(body['knobs'])
    knobs.compilation.instrumentation_mode = body['instrumentation_mode']
    vars(knobs.ascend).update(body['ascend_knobs'])
    vars(knobs.cache).update(dir=body['cache_dir'], dump_dir=body['dump_dir'])
    vars(knobs.autotuning).update(body['autotuning_knobs'])
    knobs.propagate_env = body['propagate_env']


_active_attempt = ContextVar('ascend_batch_compiler_attempt', default=None)


class _CandidateStages:
    """Child-only cache and stage binding for a single call to Triton's compiler."""

    def __init__(self, source, directory, writer):
        self.source, self.directory, self.writer = source, directory, writer
        self.index_root = directory / 'core-cache'
        self.decline = None
        self.stop = Exception('candidate stage declined')

    def bind_writer(self, source, index, options):
        from triton import knobs
        if (source is not self.source or options.block_batching_factor <= 1
                or Path(index.cache_dir).parent != self.index_root):
            raise BatchCompileError('worker stage writer identity mismatch', stage='cache')
        # The core retains its private group manager. Backend tools and dumps use
        # the original configuration for the actual compiler stages.
        vars(knobs.cache)['dir'] = self.original_cache_dir
        return self.writer

    def wrap(self, ext, stage):

        def run(module, metadata):
            (self.directory / 'stage').write_text(ext)
            result = stage(module, metadata)
            if isinstance(result, BatchDecline):
                if result.stage != ext:
                    raise BatchCompileError('decline does not match the running stage', stage=ext)
                self.decline = result
                raise self.stop
            return result

        return run

    def compile(self, compile_single, target, options, env_vars):
        from triton import knobs
        from triton.compiler.errors import MLIRCompilationError
        # The request directory is private and new, so no existing ordinary
        # group can be hit. The parent separately owns the generation directory.
        self.index_root.mkdir()
        if any(Path(self.writer.cache_dir).iterdir()):
            raise BatchCompileError('candidate generation is not empty', stage='cache')
        self.original_cache_dir = knobs.cache.dir
        token = _active_attempt.set(self)
        try:
            vars(knobs.cache)['dir'] = str(self.index_root)
            return compile_single(self.source, target, options, _env_vars=env_vars)
        except MLIRCompilationError as error:
            if error.__cause__ is self.stop and self.decline is not None:
                return self.decline
            raise
        finally:
            vars(knobs.cache)['dir'] = self.original_cache_dir
            _active_attempt.reset(token)


def _worker_input(body, directory):
    from triton import knobs
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import make_backend
    from .block_batching_source import FrozenSource
    from .reuse.snapshot import decode
    target = GPUTarget(**body['target'])
    source = FrozenSource.from_wire(body['source'], directory)
    _restore_knobs(knobs, body)
    if body.get('device') is not None:
        import torch_npu  # noqa: F401 - register the torch.npu device in this interpreter
        from triton.runtime import driver
        driver.active.set_current_device(body['device'])
    options = decode(body['options'])
    backend = make_backend(target)
    normalized = backend.parse_options(dict(options))
    if _variant_key(source, backend, normalized, body['env_vars']) != body['variant']:
        raise ValueError('worker source, backend, options or environment differ')
    return source, backend, options, normalized


def _probe_selection(body, directory):
    from triton._C.libtriton import ir
    from .compiler import make_ttir
    source, backend, _, normalized = _worker_input(body, directory)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = source.make_ir(backend.target, normalized, backend.get_codegen_implementation(normalized),
                            backend.get_module_map(), context)
    metadata = dict(vars(normalized), hash=body['variant'])
    result = make_ttir(module, metadata, normalized)
    if isinstance(result, BatchDecline):
        return 'NotApplicable', result.payload()
    return 'Success', dict(report=metadata['block_batching'])


def _compile(body, directory):
    from dataclasses import asdict
    from triton import knobs
    from triton.compiler.compiler import _compile as compile_single
    from triton.runtime.cache import FileCacheManager
    source, backend, options, _ = _worker_input(body, directory)
    target = backend.target
    cache = FileCacheManager(body['stage_key'])
    stage_dir = Path(cache.cache_dir).resolve()
    timings = {}
    if body['timing']:

        def collect(**kwargs):
            timings.update(asdict(kwargs['times']))

        knobs.compilation.listener = collect
    candidate = _CandidateStages(source, directory, cache).compile(compile_single, target, options, body['env_vars'])
    if isinstance(candidate, BatchDecline):
        return 'NotApplicable', candidate.payload()
    if candidate.hash != body['variant']:
        raise ValueError('compiled candidate identity differs')
    manifest = _artifact_manifest(candidate.metadata_group, stage_dir, candidate.hash)
    return 'Success', dict(manifest=manifest, timings=timings)


def _worker():
    # The bootstrap sets the core limit before Python processes site imports.
    directory = Path(sys.argv[1])
    request = json.loads((directory / 'request').read_bytes())
    operation, body = request['operation'], request['body']
    (directory / 'stage').write_text(operation)
    try:
        if request['protocol'] != _PROTOCOL or request['identity'] != _digest([operation, body]):
            raise ValueError('invalid request identity')
        if operation == 'compile':
            status, payload = _compile(body, directory)
        elif operation == 'selection_probe':
            status, payload = _probe_selection(body, directory)
        elif operation in ('analyze', 'probe'):
            status, payload = _native_report(operation, body, directory)
        else:
            raise ValueError('unknown worker operation')
    except Exception as error:
        status, payload = _classify(error, _stage(directory))
    reply = {key: request[key] for key in ('protocol', 'id', 'identity', 'operation')}
    reply.update(status=status, payload=payload)
    (directory / 'result').write_text(json.dumps(reply, allow_nan=False))


def _publish_generation(cache, record, pending):
    # Release rollback ownership before replacing the public index. A Python
    # interruption immediately after rename must never remove a visible generation.
    directory = Path(cache.cache_dir)
    fd, name = tempfile.mkstemp(dir=directory)
    path = Path(name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(record, stream)
        # Retain a private rollback handle for a synchronous rename failure.
        # When a signal interrupts the syscall its commit status is unknown:
        # an unreachable private directory is safer than a dangling index.
        rollback = pending.pop_all()
        try:
            path.replace(directory / _MANIFEST)
        except InterruptedError:
            raise
        except OSError:
            rollback.close()
            raise
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _worker_request(backend, source, options, env_vars, variant):
    from dataclasses import asdict
    from triton import knobs
    from .reuse.snapshot import encode
    return dict(source=source.to_wire(), target=asdict(backend.target), env_vars=env_vars,
                options=encode(dict(options)), variant=variant, cache_dir=str(Path(knobs.cache.dir).resolve()),
                dump_dir=str(Path(knobs.cache.dump_dir).resolve()), knobs=_knob_overrides(knobs.compilation),
                instrumentation_mode=knobs.compilation.instrumentation_mode, ascend_knobs=_knob_overrides(knobs.ascend),
                autotuning_knobs=_knob_overrides(knobs.autotuning), propagate_env=knobs.propagate_env)


def selection_probe(backend, frozen, options, env_vars):
    """Run the selector's normal TTIR preparation and proof in an isolated interpreter."""
    normalized = backend.parse_options(dict(options))
    variant = _variant_key(frozen, backend, normalized, env_vars)
    outcome = _attempt('selection_probe', _worker_request(backend, frozen, options, env_vars, variant))
    if outcome.status == 'NotApplicable':
        if outcome.payload.get('stage') != 'ttir':
            raise BatchCompileError('invalid selector probe refusal stage', kind='ProtocolError')
        return BatchDecline(**outcome.payload)
    report = outcome.payload['report']
    axis = report.get('selected_axis')
    if (report.get('compile_status') != 'REWRITTEN' or report.get('owner') != 'block_batching'
            or report.get('plan_id') != variant or report.get('candidate_factor') != normalized.block_batching_factor
            or type(axis) is not int or axis not in (0, 1, 2)
            or normalized.block_batching_axis >= 0 and axis != normalized.block_batching_axis):
        raise BatchCompileError('invalid selector probe identity', kind='ProtocolError')
    return report


class CandidateCompiler:
    """Own one bundle request and at most one resource-directed retry."""

    def __init__(self, backend, source, env_vars):
        self.backend, self.source, self.env_vars = backend, source, dict(env_vars)

    def _variant(self, options, pending):
        from triton import knobs
        from triton.compiler.compiler import CompileTimer
        from triton.runtime.cache import get_cache_manager
        from triton.runtime import driver

        listener = knobs.compilation.listener
        timer = CompileTimer() if listener else None
        normalized = self.backend.parse_options(dict(options))
        variant = _variant_key(self.source, self.backend, normalized, self.env_vars)
        # Old groups were published by _compile before contract binding. A distinct
        # index lets us ignore them without invalidating the independent F1 entry.
        cache = get_cache_manager(_digest(['batch_artifacts', _PROTOCOL, variant]))
        attempts = Path(cache.cache_dir).resolve() / 'attempts'
        name = self.source.name[:150]
        if Path(name).name != name or not name:
            raise BatchCompileError('invalid candidate artifact basename', kind='InvalidArtifact')
        index = cache.get_file(_MANIFEST)
        if index and not knobs.compilation.always_compile:
            try:
                stored = json.loads(Path(index).read_bytes())
                generation = stored['generation']
                if not isinstance(generation, str) or len(generation) != 32 or any(c not in '0123456789abcdef'
                                                                                   for c in generation):
                    raise ValueError('invalid cache generation')
                group = _checked_group(stored['manifest'], attempts / generation, variant, name)
            except (OSError, ValueError, TypeError, KeyError) as error:
                raise BatchCompileError(f'invalid published index: {error}', kind='InvalidArtifact',
                                        stage='cache') from error
            return None, (cache, None, group, variant, timer.end() if timer else None)
        generation = uuid4().hex
        device = None if knobs.ascend.compile_only else driver.active.get_current_device()
        body = _worker_request(self.backend, self.source, options, self.env_vars, variant)
        body.update(stage_key=f'{cache.key}/attempts/{generation}', timing=bool(listener), device=device)
        stage_dir = attempts / generation
        stage_dir.mkdir(parents=True, exist_ok=False)
        pending.callback(shutil.rmtree, stage_dir)
        outcome = _attempt('compile', body)
        if outcome.status == 'NotApplicable':
            refusal = outcome.payload
            if (refusal['reason'] == 'LAYOUT_TARGET_RESOURCE_REJECTED'
                    and (refusal.get('stage') != 'npubin' or refusal.get('factor') != normalized.block_batching_factor
                         or refusal.get('axis') != normalized.block_batching_axis
                         or normalized.block_batching_layout != 'auto')):
                raise BatchCompileError('layout refusal does not match the attempt', kind='ProtocolError')
            if refusal['reason'] in _RESOURCE_REASONS and (refusal.get('stage') != 'npubin'
                                                           or not isinstance(refusal.get('diagnostic'), str)):
                raise BatchCompileError('invalid resource refusal evidence', kind='ProtocolError')
            return refusal, None
        manifest = outcome.payload['manifest']
        group = _checked_group(manifest, attempts / generation, variant, name)
        try:
            timings = knobs.CompileTimes(**outcome.payload['timings']) if timer else None
        except (KeyError, TypeError) as error:
            raise BatchCompileError('invalid compilation timing record', kind='ProtocolError') from error
        return None, (cache, dict(generation=generation, manifest=manifest), group, variant, timings)

    def compile(self, baseline, options, request, positions):
        with ExitStack() as pending:
            return self._compile_and_bind(baseline, options, request, positions, pending)

    def _compile_and_bind(self, baseline, options, request, positions, pending):
        from triton import knobs
        from triton.compiler.compiler import CompiledKernel, _notify_compilation_listener
        from .block_batching import BatchKernel, IMPLEMENTATION_VERSION, rejected_bundle
        from .block_batching_report import validated_candidate_report
        options = dict(options)
        decline, compiled = self._variant(options, pending)
        retry = None
        if (decline and decline['reason'] == 'LAYOUT_TARGET_RESOURCE_REJECTED' and decline.get('stage') == 'npubin'
                and decline.get('factor') == request.factor and decline.get('axis') == request.axis
                and options.get('block_batching_layout', 'auto') == 'auto'):
            retry = {key: decline[key] for key in ('reason', 'stage', 'diagnostic')}
            pending.close()
            decline, compiled = self._variant(dict(options, block_batching_layout='semantic_only'), pending)
        if decline:
            return rejected_bundle(baseline, request, decline, positions, retry)
        cache, record, group, variant, timings = compiled
        try:
            candidate = CompiledKernel(self.source, group, variant)
            if (candidate.metadata.hash != variant or self.backend.target != candidate.metadata.target
                    or candidate.metadata.block_batching_backend_revision != self.backend.parse_options(
                        dict(options)).block_batching_backend_revision):
                raise ValueError('artifact metadata disagrees with candidate identity')
            report = validated_candidate_report(candidate, request, IMPLEMENTATION_VERSION)
            if retry:
                report['layout'] = dict(report['layout'], requested_policy='auto', selected_policy='semantic_only',
                                        compile_fallback=retry)
            bundle = BatchKernel(baseline, candidate, request, report, positions, publish_manifest=False)
        except Exception as exc:
            raise BatchCompileError(str(exc), kind='InvalidArtifact', stage='bind') from exc
        if record is not None:
            _publish_generation(cache, record, pending)
        bundle.publish_manifest()
        if knobs.compilation.listener:
            _notify_compilation_listener(knobs.compilation.listener, src=self.source,
                                         metadata=candidate.metadata._asdict(), metadata_group=group, times=timings,
                                         cache_hit=record is None)
        return bundle
