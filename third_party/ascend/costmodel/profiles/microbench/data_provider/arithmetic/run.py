#!/usr/bin/env python3
"""Compile/run one fixed CCE arithmetic auxiliary test; never change a profile."""
import argparse, json, os, subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--template-root', type=Path, required=True)
p.add_argument('--toolkit', type=Path, default=Path('/usr/local/Ascend/cann-9.1.0'))
p.add_argument('--route', choices=('simd', 'simt'), required=True)
operations = ('div', 'exp', 'log', 'srem', 'urem', 'add')
p.add_argument('--op', choices=operations, required=True)
p.add_argument('--device', required=True, help='Explicit idle physical NPU ID')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--compile-only', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parent
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=False)
flags = [f'-DARITH_ROUTE={int(a.route=="simt")}', f'-DARITH_OP={operations.index(a.op)}']
cc = [
    str(a.toolkit / 'bin/ccec'), '-c', '-std=c++17', '-O2', '-ffp-contract=off', '--cce-aicore-only',
    '--cce-aicore-arch=dav-c310', *flags, f'-I{a.template_root}/include', f'-I{a.template_root}/lib',
    str(root / 'arithmetic.cce'), '-o',
    str(a.output / 'arithmetic.o')
]
host = [
    'g++', '-std=c++17', '-O2', '-ffp-contract=off', *flags,
    str(root / 'arithmetic_host.cpp'), f'-I{a.toolkit}/include', f'-I{a.toolkit}/x86_64-linux/pkg_inc',
    f'-L{a.toolkit}/lib64', '-lruntime', '-lascendcl', '-o',
    str(a.output / 'arithmetic_host')
]
(a.output / 'commands.json').write_text(
    json.dumps(
        dict(route=a.route, op=a.op, physical_device=a.device, compile_only=a.compile_only, device_compile=cc,
             host_compile=host), indent=2) + '\n')
with (a.output / 'build.log').open('w') as log:
    subprocess.run(cc, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(host, stdout=log, stderr=subprocess.STDOUT, check=True)
if not a.compile_only:
    env = dict(
        os.environ, ASCEND_RT_VISIBLE_DEVICES=a.device,
        LD_LIBRARY_PATH=f'{a.toolkit}/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common'
    )
    with (a.output / 'physical.log').open('w') as log:
        subprocess.run([str(a.output / 'arithmetic_host')], cwd=a.output, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
