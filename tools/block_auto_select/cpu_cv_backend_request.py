"""Export current TA backend options on CPU for a specified hardware profile.

The normal binary-stage option assembly is executed up to its launch command.
Device discovery is replaced by the experiment's frozen target core counts;
neither compilation nor any runtime/device call is performed in this process.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(sys.argv[2]).resolve()
front = Path(sys.argv[1]).resolve()
denied = []


def audit(event, args):
    if event == 'import' and args[0].startswith(('torch', 'torch_npu', 'acl')):
        denied.append(str(args[0]))
        raise RuntimeError('Device libraries are not allowed in the CPU option exporter')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        if any(x in str(args[0]) for x in ('/dev/davinci', '/dev/devmm', '/dev/nvidia', '/dev/hisi')):
            denied.append(str(args[0]))
            raise RuntimeError('Device access is not allowed')


sys.addaudithook(audit)
os.environ.update(ASCEND_RT_VISIBLE_DEVICES='', ASCEND_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='')
sys.path.insert(0, str(ROOT / 'site-packages'))
import triton.backends.ascend.compiler as compiler
from triton.backends.compiler import GPUTarget


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


result = json.loads((front / 'result.json').read_text())
request = result['request']
metadata = json.loads((front / 'candidate.json').read_text())
metadata['target'] = GPUTarget('npu', request['target'], 0)
options = compiler.AscendBackend(metadata['target']).parse_options(request['options'])
assert request['target'] == 'Ascend950PR_9579'


class TargetProfile:

    def get_device_limit(self):
        return False, 28, 56


source = Path(compiler.__file__)
module = ast.parse(source.read_text())
function = next(n for n in module.body
                if isinstance(n, ast.FunctionDef) and n.name == 'linalg_to_bin_enable_npu_compile_910_95')
with_block = next(n for n in function.body if isinstance(n, ast.With))
command_index = next(
    i for i, n in enumerate(with_block.body)
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'cmd_list' for t in n.targets))
with_block.body = with_block.body[:command_index] + [ast.Return(ast.Name('_compile_option_list', ast.Load()))]
function.body = function.body[:function.body.index(with_block)] + [with_block]
function.decorator_list = []
function.name = 'export_options'
code = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
namespace = dict(compiler.__dict__, NPUUtils=TargetProfile)
exec(compile(code, str(source), 'exec'), namespace)
assembled = namespace['export_options']((front / 'candidate.ttadapter').read_text(), metadata, options)
assert not denied
compiler_path = ROOT / 'site-packages/triton/backends/ascend/bishengir/bin/bishengir-compile'
value = dict(compiler=str(compiler_path), compiler_sha256=sha(compiler_path), ir=str(front / 'candidate.ttadapter'),
             ir_sha256=sha(front / 'candidate.ttadapter'), target=request['target'],
             options=[v for v in assembled
                      if not v.startswith('--target=')], candidate=front.name, device_executed=False,
             options_source=str(source), options_source_sha256=sha(source), target_profile=dict(aic=28, aiv=56),
             upstream_target_option=[v for v in assembled if v.startswith('--target=')], forbidden_attempts=denied)
# The normal option assembler also finalizes masked-batch memory metadata.
# Save it as a separate artifact: the frontend's original metadata remains intact.
encoded = dict(
    metadata, target={
        'backend': metadata['target'].backend, 'arch': metadata['target'].arch, 'warp_size':
        metadata['target'].warp_size
    })
(front / 'backend-metadata.json').write_text(json.dumps(encoded, indent=2) + '\n')
value['metadata'] = dict(path=str(front / 'backend-metadata.json'), sha256=sha(front / 'backend-metadata.json'))
value['exporter_sha256'] = sha(__file__)
(front / 'backend-request-finalized.json').write_text(json.dumps(value, indent=2) + '\n')
print(json.dumps(value, indent=2))
