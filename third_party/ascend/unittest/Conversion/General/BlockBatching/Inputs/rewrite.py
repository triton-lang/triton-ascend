# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Exercise the production block-batching binding for FileCheck tests."""
import argparse

from triton._C.libtriton import ascend, ir

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('--factor', type=int, default=2)
parser.add_argument('--axis', type=int, default=0)
parser.add_argument('--layout', default='auto')
parser.add_argument('--cv', action='store_true')
parser.add_argument('--batch-matmul', action='store_true')
args = parser.parse_args()
context = ir.context()
ir.load_dialects(context)
ascend.ir.load_dialects(context)
module = ir.parse_mlir_module(args.source, context)
module.context = context
assert module.verify()
before = str(module)
report = ascend.passes.ttir.rewrite_block_batching(module, args.factor, args.axis, args.layout, args.cv,
                                                   args.batch_matmul)
assert module.verify()
for key in ('compile_status', 'reason', 'selected_axis', 'cv_lowering', 'vv_lowering'):
    if key in report:
        print(f'// {key}: {report[key]}')
if report['compile_status'] == 'REJECTED':
    assert str(module) == before, 'A rejected rewrite changed the original module'
    print('// unchanged: true')
print(module)
