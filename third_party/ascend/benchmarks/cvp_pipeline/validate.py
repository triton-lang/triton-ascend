"""Validate admitted hardware cases with pinned tools and a CPU reference.

The BZ controller performs reservation/device admission before invoking this
runner. Importing it or asking for help does not initialize the runtime.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

from stage_evidence import CAPTURE_FLAGS, certify_stage_plan


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bytes_value(value):
    return value if isinstance(value, bytes) else (value or '').encode()


def pinned_tools(compiler, utils, libtriton, args):
    """Separate normal compilation from the independent stage-evidence reader."""
    frontend = Path(compiler._get_npucompiler_path()[0]).resolve()
    reader = args.reader.resolve()
    core = Path(libtriton.__file__).resolve()
    assert frontend == args.frontend.resolve()
    assert sha(frontend) == args.frontend_sha256
    assert sha(reader) == args.reader_sha256
    if args.libtriton_sha256 is not None:
        assert sha(core) == args.libtriton_sha256
    normal_reader = None
    if callable(getattr(compiler, 'bc_to_linalg_by_bishengir_opt', None)):
        legacy_reader = utils._get_bishengir_opt_path()
        legacy_reader = legacy_reader[0] if isinstance(legacy_reader, tuple) else legacy_reader
        assert Path(legacy_reader).resolve() == reader
        normal_reader = {'path': str(reader), 'sha256': sha(reader)}
    provenance = {
        'frontend_input_stage': 'bcmlir' if normal_reader else 'ttadapter',
        'normal_compile_reader': normal_reader,
        'in_process_ir': {'path': str(core), 'sha256': sha(core)},
        'stage_evidence_reader': {'path': str(reader), 'sha256': sha(reader),
                                  'role': 'independent snapshot analysis and verification'},
    }
    return frontend, reader, provenance


def validate_frontend_input(asm, command, stage):
    value = asm[stage]
    expected = value if isinstance(value, bytes) else str(value).encode()
    assert Path(command['input_path']).read_bytes() == expected
    assert command['input_sha256'] == hashlib.sha256(expected).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frontend', type=Path, required=True)
    parser.add_argument('--frontend-sha256', required=True)
    parser.add_argument('--reader', type=Path, required=True,
                        help='Pinned independent CPU stage-evidence reader')
    parser.add_argument('--reader-sha256', required=True)
    parser.add_argument('--libtriton-sha256', help='Optional pin for the loaded in-process IR library')
    parser.add_argument('--repeats', type=int, default=32)
    parser.add_argument('--segments', type=int, choices=[1, 2, 3], nargs='+', default=[1, 2, 3])
    args = parser.parse_args()
    assert args.repeats >= 4
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch_npu
    import triton
    from triton._C import libtriton
    from triton.backends.ascend import compiler, utils
    from kernels import stage_chain

    frontend, reader, provenance = pinned_tools(compiler, utils, libtriton, args)
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    options = dict(enable_mixed_cv=True, multibuffer=True,
                   enable_auto_bind_sub_block=True, sync_solver=True,
                   enable_preload=True, enable_dynamic_cv_pipeline=False,
                   enable_cvp_plan_analysis=True, enable_fused_multiply_add=False,
                   disable_tightly_coupled_buffer_reuse=True)
    real_run = subprocess.run
    observations = []

    def observe(command, *positional, **kwargs):
        argv = command if isinstance(command, (tuple, list)) else []
        selected = bool(argv and Path(str(argv[0])).resolve() == args.frontend.resolve()
                        and '--help' not in argv)
        if not selected:
            return real_run(command, *positional, **kwargs)
        index = len(observations)
        folder = args.output / f'compiler-{index:03d}'
        folder.mkdir()
        captured_argv = [*argv, *CAPTURE_FLAGS]
        input_file = folder / 'input.mlir'
        input_file.write_bytes(Path(argv[1]).read_bytes())
        row = {'argv': [str(x) for x in captured_argv],
               'original_argv': [str(x) for x in argv],
               'input_path': str(input_file), 'input_sha256': sha(input_file),
               'input_stage': provenance['frontend_input_stage'], 'cwd': str(kwargs.get('cwd')),
               'frontend_sha256': sha(frontend),
               'started': datetime.now(timezone.utc).isoformat()}
        try:
            result = real_run(captured_argv, *positional, **kwargs)
        except subprocess.CalledProcessError as error:
            (folder / 'stdout.txt').write_bytes(bytes_value(error.stdout))
            (folder / 'stderr.txt').write_bytes(bytes_value(error.stderr))
            row['exit_code'] = error.returncode
            (folder / 'command.json').write_text(json.dumps(row, indent=2) + '\n')
            observations.append(row)
            raise
        (folder / 'stdout.txt').write_bytes(bytes_value(result.stdout))
        (folder / 'stderr.txt').write_bytes(bytes_value(result.stderr))
        row['exit_code'] = result.returncode
        row['diagnostics'] = str(folder)
        (folder / 'command.json').write_text(json.dumps(row, indent=2) + '\n')
        observations.append(row)
        return result

    results = []
    for segments in args.segments:
        torch.manual_seed(2026 + segments)
        left = (torch.randn((16, 16), dtype=torch.float32) * 0.125).half()
        weights = [(torch.randn((args.repeats, n, k), dtype=torch.float32) * 0.125).half()
                   for k, n in [(16, 32), (32, 64), (64, 16)]]
        # CPU FP32 accumulation, with an explicit FP16 round at every boundary.
        for tile in range(args.repeats):
            reference = left
            for stage in range(segments):
                reference = torch.exp(reference.float() @ weights[stage][tile].float().T * 0.125).half()
        device_left = left.npu()
        device_weights = [value.npu() for value in weights]
        output = torch.empty((16, [32, 64, 16][segments - 1]), device='npu', dtype=torch.float16)
        start = len(observations)
        with patch.object(subprocess, 'run', side_effect=observe):
            compiled = stage_chain[(1,)](device_left, *device_weights, output,
                                         segments, args.repeats, **options)
        torch.npu.synchronize()
        actual = output.cpu()
        torch.testing.assert_close(actual, reference, atol=0.005, rtol=0.005)
        metadata = compiled.metadata._asdict() if hasattr(compiled.metadata, '_asdict') else vars(compiled.metadata)
        for name, value in options.items():
            assert metadata[name] == value, (name, metadata.get(name), value)
        assert metadata.get('ir_override') is None
        assert metadata.get('bitcodes') in (None, (), [])
        commands = observations[start:]
        assert commands, 'use a fresh private cache for validation'
        diagnostics = ''.join((Path(row['diagnostics']) / 'stderr.txt').read_text(errors='replace')
                              + (Path(row['diagnostics']) / 'stdout.txt').read_text(errors='replace')
                              for row in commands)
        assert len(commands) == 1, 'expected one normal native compilation per case'
        validate_frontend_input(compiled.asm, commands[0], provenance['frontend_input_stage'])
        evidence = certify_stage_plan(diagnostics, Path(commands[0]['diagnostics']),
                                      reader, segments)
        native = compiled.asm['npubin']
        assert native[:4] == b'\x7fELF'
        binary = args.output / f'chain-{segments}.npubin'
        binary.write_bytes(native)
        row = {'segments': segments, 'repeats': args.repeats,
               'recognized_segments': evidence['recognized_segments'],
               'verified_stages': evidence['verified_stages'],
               'stage_evidence': evidence,
               'maximum_absolute_error': (actual.float() - reference.float()).abs().max().item(),
               'atol': 0.005, 'rtol': 0.005, 'options': options,
               'compiler_commands': commands,
               'frontend_input_stage': provenance['frontend_input_stage'],
               'native_sha256': sha(binary), 'metadata': metadata}
        results.append(row)
        (args.output / f'chain-{segments}.json').write_text(json.dumps(row, indent=2, default=str) + '\n')
    report = {'passed': True, 'cases': results, 'frontend_sha256': sha(frontend),
              'compiler_provenance': provenance,
              'reader_sha256': sha(reader), 'kernel_source_sha256': sha(Path(__file__).with_name('kernels.py')),
              'validator_source_sha256': sha(Path(__file__)),
              'stage_evidence_source_sha256': sha(Path(__file__).with_name('stage_evidence.py')),
              'triton_file': triton.__file__, 'torch_version': torch.__version__,
              'torch_npu_version': torch_npu.__version__,
              'timing': 'not measured by this correctness runner'}
    (args.output / 'SUMMARY.json').write_text(json.dumps(report, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
