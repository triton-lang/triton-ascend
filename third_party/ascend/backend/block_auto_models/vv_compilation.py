"""Obtain optimized LLVM in the candidate's ordinary compilation cache."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .native_model import meta_path, reject


def optimized_llvm(item):
    from triton.runtime.cache import get_cache_manager
    from ..utils import _get_bisheng_path
    cache = get_cache_manager(item['metadata']['hash'])
    name = 'block_auto_optimized.ll'
    cached = cache.get_file(name)
    if cached:
        return Path(cached).read_text()
    files = item['files']
    compiler = Path(_get_bisheng_path()).resolve()
    link = compiler.with_name('llvm-link')
    if not link.is_file():
        link = Path(shutil.which('llvm-link') or '')
    reject(not compiler.is_file() or not link.is_file(), 'installed A5 optimizer/LLVM decoder is unavailable')
    meta = meta_path(item['final_ir'])
    library = Path(files['compiler']['path']).parent.parent.parent / 'lib/libdevice.10.bc'
    directory = getattr(cache, 'cache_dir', None)
    reject(not directory, 'optimized LLVM requires a local compiler cache')
    with tempfile.TemporaryDirectory(prefix='block-auto-', dir=directory) as temporary:
        output = Path(temporary)
        args = [
            str(compiler), '-c', '-emit-llvm', '-x', 'ir', files['llvm']['path'], '-o',
            str(output / 'kernel.bc'), '--cce-aicore-arch=dav-c310-vec', '--cce-aicore-only', '-O2',
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
        run([str(link), '-S', str(output / 'kernel.bc'), '-o', str(output / 'optimized.ll')])
        text = (output / 'optimized.ll').read_text()
    cache.put(text, name)
    return text
