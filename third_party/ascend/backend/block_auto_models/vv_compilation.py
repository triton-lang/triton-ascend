"""Obtain executable-matched optimized LLVM once per JIT cache identity."""
from .native_model import reject
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

from .native_model import Unsupported
from .cpu_resource_gate import sha256
from .native_model import meta_path


def executable_sections(path):
    data = Path(path).read_bytes()
    reject(len(data) < 64 or data[:6] != b'\x7fELF\x02\x01', 'expected a little-endian ELF64 binary', ValueError)
    offset = struct.unpack_from('<Q', data, 40)[0]
    size, count, strings = struct.unpack_from('<HHH', data, 58)
    reject(size != 64 or not 0 < count < 65535 or strings >= count or (offset + count * size > len(data)),
           'invalid ELF section table', ValueError)
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * size) for i in range(count)]
    names_section = sections[strings]
    names = data[names_section[4]:names_section[4] + names_section[5]]
    result = {
        'elf_machine': struct.unpack_from('<H', data, 18)[0], 'elf_flags': struct.unpack_from('<I', data, 48)[0],
        'sections': {}
    }
    for section in sections:
        start, kind, flags, address, at, length, link, info, align, entsize = section
        if not (flags & 2 or kind in (4, 9)):
            continue
        reject(start >= len(names) or (kind != 8 and at + length > len(data)), 'invalid executable section range',
               ValueError)
        name = names[start:].split(b'\0', 1)[0].decode()
        reject(name in result['sections'], 'duplicate executable section', ValueError)
        payload = b'' if kind == 8 else data[at:at + length]
        result['sections'][name] = dict(kind=kind, flags=flags, address=address, size=length, alignment=align,
                                        sha256=hashlib.sha256(payload).hexdigest())
    reject('.text' not in result['sections'], 'missing executable text', ValueError)
    return result


def optimized_llvm(item):
    from triton.runtime.cache import get_cache_manager
    from ..utils import _get_bisheng_path
    files = item['files']
    compiler = Path(_get_bisheng_path()).resolve()
    link = compiler.with_name('llvm-link')
    if not link.is_file():
        link = Path(shutil.which('llvm-link') or '')
    reject(not compiler.is_file() or not link.is_file(), 'installed A5 optimizer/LLVM decoder is unavailable')
    meta = meta_path(item['final_ir'])
    library = Path(files['compiler']['path']).parent.parent.parent / 'lib/libdevice.10.bc'
    inputs = {Path(files[role]['path']): files[role]['sha256'] for role in ('llvm', 'binary', 'compiler')}
    inputs.update({path: sha256(path) for path in (compiler, link, meta, library)})
    identity = dict(files={str(path): digest for path, digest in inputs.items()})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = get_cache_manager(key)
    # Cache publication follows executable matching; the content digest detects
    # a damaged cached IR without retaining/revalidating offline proof bundles.
    cached = cache.get_file('optimized.json')
    if cached:
        record = json.loads(Path(cached).read_text())
        reject(Path(record['name']).name != record['name'], 'invalid optimized LLVM cache name')
        path = cache.get_file(record['name'])
        if path and sha256(path) == record['sha256']:
            return Path(path).read_text()
        raise Unsupported('cached optimized LLVM changed')
    directory = getattr(cache, 'cache_dir', None)
    reject(not directory, 'optimized evidence requires a local compiler cache')
    with tempfile.TemporaryDirectory(prefix='block-auto-', dir=directory) as temporary:
        output = Path(temporary)
        args = [
            str(compiler), '-c', '-x', 'ir', files['llvm']['path'], '-o',
            str(output / 'kernel.o'), '--cce-aicore-arch=dav-c310-vec', '--cce-aicore-only', '-O2',
            '-cce-bitcode-is-aicore', '-Wno-override-module', '-mllvm', '--cce-vf-auto-sync=global', '-mllvm',
            '-cce-aicore-dcci-insert-for-scalar=false', '-mllvm', '-enable-module-inliner', '-mllvm',
            '-replexitval=unusedindvarinloop', '-mllvm', '-cce-vf-aa-between-iters=true', '-mllvm',
            '-cce-vf-enable-loop-fusion=false', '--cce-disable-asc-reserved-ubuf', '-mllvm',
            '-cce-vf-stack-size=0x2000', '--cce-long-scbz=true', '-cce-link-aicore-ll-module',
            str(meta), '-cce-link-aicore-ll-module',
            str(library)
        ]
        env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES='', ASCEND_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='')

        def run(command):
            subprocess.run(command, capture_output=True, env=env, check=True, timeout=240)

        run(args)
        reject(
            executable_sections(files['binary']['path']) != executable_sections(output / 'kernel.o'),
            'CPU reproduction differs from the candidate executable sections', ValueError)
        args.insert(2, '-emit-llvm')
        args[args.index('-o') + 1] = str(output / 'kernel.bc')
        run(args)
        run([str(link), '-S', str(output / 'kernel.bc'), '-o', str(output / 'optimized.ll')])
        reject(any((sha256(path) != expected for path, expected in inputs.items())),
               'input changed during optimization', ValueError)
        text = (output / 'optimized.ll').read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    name = 'optimized-' + digest + '.ll'
    cache.put(text, name)
    cache.put(json.dumps(dict(name=name, sha256=digest)), 'optimized.json')
    return text
