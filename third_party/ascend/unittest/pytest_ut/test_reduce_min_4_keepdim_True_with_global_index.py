# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import pytest
import torch
import torch_npu

import test_common
from test_reduce_min_4_keepdim_True_with_index_op import triton_min_5d_dim4_keepdim

# Reduce over dim 4 with an index operand that carries each element's global
# linear index instead of its position along dim 4. The winning element is the
# same as for the position index, but the returned index must be the global
# value, not the local position. Only dims larger than 1 are used: for a unit
# reduction dim AscendNPU-IR folds the reduce to a squeeze and currently fills
# the index result with zeros (see HFusion FoldUnitDims rewriteNoOpReduce).
testlist = [
    (2, 2, 2, 2, 8),
    (9, 3, 2, 4, 17),
    (129, 1, 5, 1, 4),
    (3, 11, 1, 3, 42),
    (2, 91, 4, 2, 4),
    (5, 2, 4, 1, 26),
    (1, 1, 3, 2, 600),
]

typelist = ['int32', 'int64', 'float32', 'bool']

ids = ["{}-{}".format(triton_min_5d_dim4_keepdim.__name__, "-".join(map(str, shape))) for shape in testlist]


@pytest.mark.parametrize('shape', testlist, ids=ids)
@pytest.mark.parametrize('sigtype', typelist)
def test_min_dim4_global_index(sigtype, shape):
    dtype = eval('torch.' + sigtype)
    x0 = torch.randn(shape).to(dtype).npu()

    x1 = torch.arange(x0.numel()).view(x0.shape).npu().to(torch.int32)
    if 'int' in sigtype:
        ans, ans1 = torch.min(x0.to(torch.int64), 4)
        ans = ans.to(dtype)
    else:
        ans, ans1 = torch.min(x0, 4)
    # x1 = global linear index, so the winner's index is the base of its row
    # (x1[..., 0]) plus its local position along dim 4.
    base = torch.arange(x0.numel(), device=x0.device).view(shape)[..., 0]
    ans1 = (ans1 + base).to(torch.int32)
    output = torch.zeros(shape[0:4], dtype=dtype).npu()
    output1 = torch.zeros(shape[0:4], dtype=torch.int32).npu()
    triton_min_5d_dim4_keepdim[(1, )](x0, x1, output, output1, *shape)
    test_common.validate_cmp(sigtype, output, ans)
    test_common.validate_cmp('int32', output1, ans1)
