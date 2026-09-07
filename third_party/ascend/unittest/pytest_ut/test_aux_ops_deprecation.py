# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from triton.language import core as tl_core
from triton.language.extra.cann.extension import aux_ops

pytestmark = pytest.mark.backend("none")


@pytest.mark.parametrize(
    ("fn_name", "args", "expected_builder_args"),
    [
        pytest.param(
            "sync_block_all",
            ("all", 0),
            ("sync_block_all", "all", 0),
            id="all",
        ),
        pytest.param(
            "sync_block_set",
            ("cube", "vector", 1),
            ("sync_block_set", "cube", 1),
            id="set",
        ),
        pytest.param(
            "sync_block_wait",
            ("cube", "vector", 1),
            ("sync_block_wait", "cube", 1),
            id="wait",
        ),
    ],
)
def test_legacy_sync_block_deprecation(fn_name, args, expected_builder_args):
    emit_sync_op = Mock()
    semantic = SimpleNamespace(builder=SimpleNamespace(create_custom_op_for_inter_core_sync=emit_sync_op, ), )
    fn = getattr(aux_ops, fn_name)
    expected_message = (f"triton.language.{fn_name} is deprecated and will be removed in the next release; "
                        f"use triton.language.extra.cann.extension.{fn_name} instead.")

    assert tl_core.is_builtin(fn)
    with pytest.warns(FutureWarning, match=fn_name) as caught:
        fn(*args, _semantic=semantic)

    assert len(caught) == 1
    assert str(caught[0].message) == expected_message
    emit_sync_op.assert_called_once_with(*expected_builder_args)
