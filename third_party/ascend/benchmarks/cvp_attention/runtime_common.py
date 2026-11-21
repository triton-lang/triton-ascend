"""Normal original-kernel JIT execution for this fork's compiler recipes.

No historical binary hash is accepted as qualification for this wheel. Each
first launch preserves its actual commands, options, IR stages, and native ELF.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
import sys
import types

KERNEL_SHA = '33faa144cbdde5bad8ec8a05bffab664130b9a285bec8985d5635f0cf38f0ac1'
ATOL = RTOL = 0.005
CHUNK = 4 * 1024 * 1024
FIXED_HOST_OPTIONS = {
    'enable_mixed_cv': True, 'multibuffer': True, 'enable_auto_bind_sub_block': True, 'sync_solver': True
}
FORBIDDEN_OPTIONS = ('enable_cv_split_scheduling', 'enable_cvp_async_ownership', 'enable_physical_memory_footprints',
                     'enable_cvp_physical_memory_footprints')


def sha_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')
    temporary.replace(path)


def compiler_tools(compiler, utils, libtriton, config):
    """Pin the tools actually used by this package's normal compiler route."""
    site = Path(config['wheel_site']).resolve()
    tools = {}
    for name, filename, key in (('frontend', compiler._get_npucompiler_path()[0], 'frontend_sha256'),
                                ('libtriton', libtriton.__file__, 'libtriton_sha256')):
        path = Path(filename).resolve()
        assert path.is_relative_to(site), path
        digest = sha_file(path)
        assert digest == config[key], path
        tools[name] = {'path': str(path), 'sha256': digest}
    # Historical packages used a standalone bytecode-to-text bridge. Keep its
    # real identity for those packages, without inventing it for direct text.
    if callable(getattr(compiler, 'bc_to_linalg_by_bishengir_opt', None)):
        reader = utils._get_bishengir_opt_path()
        reader = reader[0] if isinstance(reader, tuple) else reader
        path = Path(reader).resolve()
        assert path.is_relative_to(site), path
        digest = sha_file(path)
        assert digest == config['reader_sha256'], path
        tools['reader'] = {'path': str(path), 'sha256': digest}
    return tools


def validate_frontend_inputs(asm, commands, stage):
    """Bind retained compiler input bytes to the stage from this JIT result."""
    value = asm[stage]
    expected = value if isinstance(value, bytes) else str(value).encode()
    expected_sha = hashlib.sha256(expected).hexdigest()
    assert commands, 'First compile must invoke the pinned frontend'
    for command in commands:
        path = Path(command['input_path'])
        assert path.read_bytes() == expected, 'Frontend input differs from the actual JIT stage'
        assert command['input_sha256'] == expected_sha


def launch_options(preset):
    """Avoid duplicate kwargs already authored in the original host function."""
    options = dict(preset)
    assert options.get('enable_dynamic_cv_pipeline') is False
    assert options.get('cvp_persistent_pipeline_mode') in (None, 'skew'), 'query Unroll was removed'
    lanes = options.get('cvp_inner_unroll')
    assert lanes is None or (type(lanes) is int and lanes == 1), 'query Unroll was removed'
    assert options.get('enable_preload') is True
    assert options.get('enable_cvp_plan_analysis') is True
    assert all(name not in options for name in FORBIDDEN_OPTIONS)
    assert all('async_ownership' not in name and 'physical_memory_footprint' not in name for name in options)
    for name, value in FIXED_HOST_OPTIONS.items():
        if name in options:
            assert type(options[name]) is type(value) and options[name] == value, name
            options.pop(name)
    return options


def validate_native_route_metadata(metadata):
    """Allow normal empty defaults, while rejecting injected IR or bitcodes."""
    if metadata['ir_override'] is not None:
        raise ValueError('IR overrides are outside the normal compiler route')
    bitcodes = metadata.get('bitcodes')
    if bitcodes is not None and not (type(bitcodes) in (list, tuple) and len(bitcodes) == 0):
        raise ValueError('Nonempty or malformed bitcode overrides are outside the normal compiler route')
    if (metadata.get('enable_cvp_persistent_pipeline') is True
            and metadata.get('disable_tightly_coupled_buffer_reuse') is not True):
        raise ValueError('Native query metadata must contain the effective protected buffer policy')


class Runtime:

    def __init__(self, kernel_path, artifact_root, config, recipes, guard):
        import torch
        import torch_npu
        import triton
        import triton.language as tl
        from triton._C import libtriton
        from triton.backends.ascend import compiler, utils, driver

        self.torch, self.torch_npu, self.triton = torch, torch_npu, triton
        self.config, self.recipes, self.guard = config, recipes, guard
        self.root = Path(artifact_root)
        self.root.mkdir(exist_ok=False)
        self.compiles, self.forward_counts, self.commands = {}, {}, []
        self.last_compiled = None
        self.crosscheck_cpu = False
        torch.set_num_threads(4)
        guard()
        torch.npu.set_device(0)
        site = Path(config['wheel_site']).resolve()
        assert Path(triton.__file__).resolve().is_relative_to(site)
        for path, digest in ((compiler.__file__, config['compiler_python_sha256']), (driver.__file__,
                                                                                     config['driver_python_sha256'])):
            assert sha_file(path) == digest, path
        assert not any(token in Path(compiler.__file__).read_text()
                       for token in ('enable_cvp_async_ownership', 'enable_physical_memory_footprints',
                                     'enable_cvp_physical_memory_footprints'))
        assert '_cvp_async_launch_contract' not in Path(driver.__file__).read_text()
        self.tools = compiler_tools(compiler, utils, libtriton, config)
        self.frontend_input_stage = 'bcmlir' if 'reader' in self.tools else 'ttadapter'
        downstream = Path(config['downstream_bisheng_path']).resolve()
        assert downstream.is_relative_to(Path(config['cann_root']).resolve())
        assert sha_file(downstream) == config['downstream_bisheng_sha256']
        self.tools['downstream_bisheng'] = {'path': str(downstream), 'sha256': sha_file(downstream)}
        assert Path(config['cann_root']).resolve() == Path('/data/pri/Ascend/9.2.0.B060/cann-9.2.0')
        assert sha_file(kernel_path) == KERNEL_SHA == recipes['kernel_sha256']
        properties = triton.runtime.driver.active.utils.get_device_properties(0)
        self.cores = int(properties['num_aicore'])
        assert self.cores == config['expected_aicore_num']
        tree = ast.parse(Path(kernel_path).read_text())
        names = {'_attn_fwd_inner', '_attn_fwd', '_attention', 'attention'}
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert len(nodes) == len(names)
        module = types.ModuleType('cvp_without_stage6_original_kernel')
        module.__file__ = str(kernel_path)
        module.__dict__.update(torch=torch, torch_npu=torch_npu, triton=triton, tl=tl, DEVICE='npu',
                               AICORE_NUM=self.cores, COMPILER_KWARGS_CONFIGS={})
        sys.modules[module.__name__] = module
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(kernel_path), 'exec'), module.__dict__)
        self.module, self.raw_kernel = module, module._attn_fwd
        owner = self

        class LaunchRecorder:

            def __getitem__(self, grid):
                launch = owner.raw_kernel[grid]

                def run(*args, **kwargs):
                    result = launch(*args, **kwargs)
                    owner.last_compiled = result
                    return result

                return run

        module._attn_fwd = LaunchRecorder()
        observed_tools = {row['path'] for name, row in self.tools.items() if name != 'libtriton'}

        def observe(event, args):
            if event != 'subprocess.Popen':
                return
            executable, argv, cwd, _environment = args
            if not isinstance(argv, (tuple, list)) or not argv:
                return
            if str(Path(os.fsdecode(executable)).resolve()) in observed_tools:
                row = {
                    'at': datetime.now(timezone.utc).isoformat(), 'argv': [os.fsdecode(x) for x in argv], 'cwd':
                    str(cwd)
                }
                if (str(Path(os.fsdecode(executable)).resolve()) == self.tools['frontend']['path']
                        and '--help' not in row['argv']):
                    original_input = Path(row['argv'][1])
                    if not original_input.is_absolute() and cwd is not None:
                        original_input = Path(os.fsdecode(cwd)) / original_input
                    retained = self.root / f'frontend-input-{len(self.commands):03d}.mlir'
                    retained.write_bytes(original_input.read_bytes())
                    row.update(input_path=str(retained), input_sha256=sha_file(retained),
                               input_stage=self.frontend_input_stage)
                self.commands.append(row)

        sys.addaudithook(observe)
        self.source_identity = {
            'kernel_sha256': KERNEL_SHA, 'selected_ast_sha256':
            hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[])).encode()).hexdigest(),
            'original_host_forward_preserved': True, 'normal_jit_launch': True, 'source_stage6_excluded': True, 'tools':
            self.tools, 'cores': self.cores, 'frontend_input_stage': self.frontend_input_stage, 'normal_compile_reader':
            self.tools.get('reader'), 'triton_file': triton.__file__, 'torch_version': torch.__version__,
            'torch_npu_version': torch_npu.__version__, 'physical_visibility': os.environ['ASCEND_RT_VISIBLE_DEVICES'],
            'device_name': torch.npu.get_device_name(0), 'initial_libraries': self.loaded_libraries()
        }
        save_json(self.root / 'SOURCE_IDENTITY.json', self.source_identity)

    def ours(self, q, k, v, preset_name, block_n):
        self.guard()
        assert preset_name in self.recipes['presets'] and block_n in (128, 256)
        preset = self.recipes['presets'][preset_name]
        options = launch_options(preset)
        self.module.COMPILER_KWARGS_CONFIGS['integration'] = options

        class Context:

            def save_for_backward(self, *values):
                self.saved = values

        context = Context()
        key = (tuple(q.shape), preset_name, block_n)
        command_start = len(self.commands)
        self.last_compiled = None
        output = self.module._attention.forward(context, q, k, v, None, False, 0.5, 128, block_n, 'integration')
        m = context.saved[-1]
        kernel = self.last_compiled
        assert kernel is not None
        if key not in self.compiles:
            metadata = kernel.metadata._asdict() if hasattr(kernel.metadata, '_asdict') else vars(kernel.metadata)
            for name, value in {**FIXED_HOST_OPTIONS, **preset}.items():
                assert type(metadata[name]) is type(value) and metadata[name] == value, (name, value,
                                                                                         metadata.get(name))
            assert all(metadata.get(name) in (None, False) for name in FORBIDDEN_OPTIONS)
            assert not metadata.get('enable_dynamic_cv_pipeline')
            assert metadata['parallel_mode'] == 'simd' and metadata['mix_mode'] == 'mix'
            assert metadata['compile_on_910_95'] is True
            validate_native_route_metadata(metadata)
            native = kernel.asm['npubin']
            assert native[:4] == b'\x7fELF'
            commands = self.commands[command_start:]
            compiles = [
                row for row in commands
                if str(Path(row['argv'][0]).resolve()) == self.tools['frontend']['path'] and '--help' not in row['argv']
            ]
            assert compiles, 'First compile must use a fresh private cache and invoke the pinned frontend'
            validate_frontend_inputs(kernel.asm, compiles, self.frontend_input_stage)
            flags = {argument.lower() for row in compiles for argument in row['argv']}
            assert '--enable-preload=' + str(preset['enable_preload']).lower() in flags
            assert '--enable-mixed-cv=true' in flags
            assert '--enable-cvp-plan-analysis=' + str(preset['enable_cvp_plan_analysis']).lower() in flags
            if preset.get('enable_cvp_persistent_pipeline'):
                assert '--disable-tightly-coupled-buffer-reuse' in flags
                assert '--enable-cvp-persistent-pipeline=true' in flags
                assert '--cvp-persistent-pipeline-mode=' + preset['cvp_persistent_pipeline_mode'] in flags
                assert '--cvp-inner-unroll=' + str(preset['cvp_inner_unroll']) in flags
            assert not any(
                'async-ownership' in argument or 'physical-memory-footprint' in argument or 'cv-split' in argument
                for argument in flags)
            folder = self.root / ('b%d-h%d-n%d-d%d-bn%d-%s' % (*q.shape, block_n, preset_name))
            folder.mkdir(exist_ok=False)
            (folder / 'kernel.npubin').write_bytes(native)
            for stage in ('ttir', 'ttadapter', 'bcmlir', 'mlirbc', 'mlir', 'linalg'):
                if stage in kernel.asm:
                    value = kernel.asm[stage]
                    (folder /
                     ('kernel.' + stage)).write_bytes(value if isinstance(value, bytes) else str(value).encode())
            record = {
                'shape': list(q.shape), 'preset': preset_name, 'block_m': 128, 'block_n': block_n, 'options': preset,
                'host_fixed_options': FIXED_HOST_OPTIONS, 'metadata': metadata, 'compiler_commands': commands,
                'native_sha256': hashlib.sha256(native).hexdigest(), 'frontend_input_stage': self.frontend_input_stage,
                'native_bytes': len(native), 'artifact_dir': str(folder), 'hardware_correctness': 'pending',
                'performance': 'not measured'
            }
            save_json(folder / 'COMPILE.json', record)
            self.compiles[key] = record
        self.forward_counts[preset_name] = self.forward_counts.get(preset_name, 0) + 1
        return output, m

    def ascendc(self, q, k, v):
        self.guard()
        result = self.torch_npu.npu_fusion_attention(q, k, v, q.shape[1], padding_mask=None, atten_mask=None, scale=0.5,
                                                     keep_prob=1.0, input_layout='BNSD', pre_tockens=65535,
                                                     next_tockens=65535, sparse_mode=0)
        self.forward_counts['ascendc'] = self.forward_counts.get('ascendc', 0) + 1
        return result

    def lse_from_ascendc(self, result, shape):
        maximum, total = result[1], result[2]
        expected = tuple(shape[:3]) + (8, )
        assert tuple(maximum.shape) == tuple(total.shape) == expected
        assert maximum.dtype == total.dtype == self.torch.float32
        return maximum[..., 0] + self.torch.log(total[..., 0])

    def inputs(self, shape, seed, kind='normal'):
        torch = self.torch
        self.guard()
        torch.manual_seed(seed)
        if kind == 'normal':
            result = [torch.empty(shape, device='npu', dtype=torch.float16).normal_(0.0, 0.5) for _ in range(3)]
        elif kind == 'zeros':
            result = [torch.zeros(shape, device='npu', dtype=torch.float16) for _ in range(3)]
        elif kind == 'repeated':
            result = [torch.full(shape, 0.25, device='npu', dtype=torch.float16) for _ in range(3)]
        elif kind == 'extreme':
            assert shape[2] <= 512
            q = torch.full(shape, 8.0, device='npu', dtype=torch.float16)
            k = torch.full(shape, 8.0, device='npu', dtype=torch.float16)
            q[:, :, shape[2] // 2:, :].fill_(-8.0)
            for start in range(128, shape[2], 256):
                k[:, :, start:start + 128, :].fill_(-8.0)
            v = torch.empty(shape, device='npu', dtype=torch.float16).uniform_(-4.0, 4.0)
            result = [q, k, v]
        else:
            raise ValueError(kind)
        torch.npu.synchronize()
        return result

    def compare_cpu(self, actual, reference, label):
        torch = self.torch
        assert tuple(actual.shape) == tuple(reference.shape)
        a, b = actual.detach().reshape(-1), reference.detach().reshape(-1)
        finite, bad, maximum, checked = True, 0, 0.0, 0
        for start in range(0, a.numel(), CHUNK):
            self.guard()
            x, y = a[start:start + CHUNK].cpu().float(), b[start:start + CHUNK].cpu().float()
            finite &= bool(torch.isfinite(x).all().item() and torch.isfinite(y).all().item())
            delta = (x - y).abs()
            maximum = max(maximum, float(delta.max().item()))
            bad += int((delta > ATOL + RTOL * y.abs()).sum().item())
            checked += x.numel()
        return {
            'label': label, 'passed': finite and bad == 0, 'finite': finite, 'bad_elements': bad, 'elements': checked,
            'max_abs_error': maximum, 'atol': ATOL, 'rtol': RTOL
        }

    def fingerprint_cpu(self, tensor):
        torch = self.torch
        flat, digest, finite = tensor.detach().reshape(-1), hashlib.sha256(), True
        for start in range(0, flat.numel(), CHUNK):
            self.guard()
            part = flat[start:start + CHUNK].cpu()
            digest.update(memoryview(part.numpy()).cast('B'))
            finite &= bool(torch.isfinite(part).all().item())
        return {
            'sha256': digest.hexdigest(), 'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'strides':
            list(tensor.stride()), 'finite': finite
        }

    def compare(self, actual, reference, label):
        import device_checks
        if actual.device.type != 'npu' or reference.device.type != 'npu':
            return self.compare_cpu(actual, reference, label)
        result = device_checks.compare(self.torch, actual, reference, label, self.guard)
        if self.crosscheck_cpu:
            device_checks.assert_equivalent(result, self.compare_cpu(actual, reference, label))
        return result

    def fingerprint(self, tensor):
        import device_checks
        result = device_checks.fingerprint(self.torch, tensor, self.guard)
        if self.crosscheck_cpu:
            assert result == self.fingerprint_cpu(tensor)
        return result

    def validate_checkers(self):
        import device_checks
        return device_checks.validate(self, self.compare_cpu, self.fingerprint_cpu, self.guard)

    def mathematical(self, q, k, v):
        torch = self.torch
        assert q.shape[2] <= 512, 'Dense CPU reference is restricted to small mathematical probes'
        qc, kc, vc = [x.cpu().float() for x in (q, k, v)]
        score = (qc @ kc.transpose(-2, -1)) * 0.5
        return torch.softmax(score, dim=-1) @ vc, torch.logsumexp(score, dim=-1)

    def loaded_libraries(self):
        paths = sorted({
            line.split()[-1]
            for line in Path('/proc/self/maps').read_text().splitlines()
            if '/' in line and any(x in line
                                   for x in ('libascendcl', 'libopapi', 'libnnopbase', 'libprofapi', 'liboptiling'))
        })
        rows = [{'path': p, 'resolved': str(Path(p).resolve()), 'sha256': sha_file(p)}
                for p in paths
                if Path(p).is_file()]
        assert not any('B111' in row['path'] or 'B111' in row['resolved'] for row in rows)
        assert any('libascendcl' in row['path'] and '9.2.0.B060' in row['resolved'] for row in rows)
        pins = self.config['runtime_library_pins']
        assert pins, 'Resolve runtime library pins from the B060 preflight before execution'
        for row in rows:
            assert pins.get(row['resolved']) == row['sha256'], row
        return rows
