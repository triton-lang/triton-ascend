"""Compile one frozen CV fixture to TTAdapter using a CPU-only PR2 frontend.

Run with the pinned compiler's Python and -I -S. This invokes the normal
AST/TTIR/TTAdapter stages directly, without creating a launchable kernel or
using compile_bundle's fallback. Numerical correctness is not checked here.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    denied = []

    def audit(event, values):
        reason = None
        if event == 'import' and values[0].startswith(('torch_npu', 'acl')):
            reason = 'device runtime import: ' + values[0]
        if event == 'open' and isinstance(values[0], (str, bytes)):
            path = str(values[0])
            if any(p in path for p in ('/dev/davinci', '/dev/devmm', '/dev/nvidia', '/dev/hisi', '/dev/npu')):
                reason = 'device access: ' + path
        if reason:
            denied.append(reason)
            raise RuntimeError(reason)

    sys.addaudithook(audit)
    os.environ.update(ASCEND_RT_VISIBLE_DEVICES='', ASCEND_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='')
    os.environ['TRITON_CACHE_DIR'] = str(args.output / 'cache')
    started = time.monotonic()
    result = {
        'device_executed': False, 'numerical_correctness_verified': False, 'status': 'FRONTEND_ERROR', 'request':
        request
    }
    try:
        native = Path(request['native']).resolve()
        if digest(native) != request['native_sha256']:
            raise ValueError('compiler extension changed')
        sys.path.insert(0, str(native.parents[2]))
        import triton
        import triton.language as tl
        from triton._C.libtriton import ir
        from triton.backends.compiler import GPUTarget
        from triton.backends.ascend.compiler import (
            AscendBackend,
            _parse_linalg_metadata,
            _finalize_program_launch_policy,
        )
        from triton.backends.ascend.driver import NPUUtils
        from triton.compiler.compiler import ASTSource

        def forbidden_device_utility(*unused, **kwargs):
            denied.append('NPUUtils construction')
            raise RuntimeError('CPU frontend must not construct NPUUtils')

        NPUUtils.__new__ = staticmethod(forbidden_device_utility)
        case_path, kernel_path = Path(request['case']), Path(request['kernel'])
        if digest(case_path) != request['case_sha256'] or digest(kernel_path) != request['kernel_sha256']:
            raise ValueError('fixture or kernel changed')
        case, = json.loads(case_path.read_text())['cases']
        spec = importlib.util.spec_from_file_location('frozen_cv_kernel', kernel_path)
        module = importlib.util.module_from_spec(spec)
        module.triton, module.tl = triton, tl
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        fn = getattr(module, request['kernel_name'])
        scalars = case['scalar_arguments']
        # Match the frozen JIT specialization: explicit constexprs and unit
        # scalar strides are constants; all other strides remain arguments.
        constants = {
            name: scalars[name]
            for index, name in enumerate(fn.arg_names)
            if name in scalars and (index in fn.constexprs or scalars[name] == 1)
        }
        signature = {
            name: '*fp32' if name in case['tensors'] else 'constexpr' if name in constants else 'i32'
            for name in fn.arg_names
        }
        target = GPUTarget('npu', request['target'], 0)
        backend = AscendBackend(target)
        options = backend.parse_options(request['options'])
        source = ASTSource(fn, signature, constexprs=constants)
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        metadata = {
            'hash': digest(args.request), 'target': target, **options.__dict__, 'triton_version': triton.__version__
        }
        lowered = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                                 context)
        (args.output / 'raw.ttir').write_text(str(lowered))
        stages = {}
        backend.add_stages(stages, options, source.language)
        for stage in ('ttir', 'ttadapter'):
            lowered = stages[stage](lowered, metadata)
            (args.output / ('candidate.' + stage)).write_text(str(lowered))
        # Match the CPU metadata preparation at the start of the normal binary
        # stage. Do not enter that stage's NPUUtils/device-runtime path.
        _, metadata = _parse_linalg_metadata(str(lowered), metadata)
        _finalize_program_launch_policy(metadata, options)
        encoded = dict(metadata, target={'backend': target.backend, 'arch': target.arch, 'warp_size': target.warp_size})
        (args.output / 'candidate.json').write_text(json.dumps(encoded, indent=2) + '\n')
        result.update(status='TTADAPTER_COMPILED', metadata_path=str(args.output / 'candidate.json'),
                      source_signature=signature, constexprs=constants, native_sha256=digest(native),
                      triton_path=triton.__file__, merge_report=metadata.get('block_batching'))
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    result.update(seconds=time.monotonic() - started, forbidden_attempts=denied,
                  device_runtime_modules=[n for n in sys.modules if n.startswith(('torch_npu', 'acl'))])
    (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(
        json.dumps({k: result.get(k)
                    for k in ('status', 'error', 'seconds', 'forbidden_attempts')}, ensure_ascii=False))
    return 0 if result['status'] == 'TTADAPTER_COMPILED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
