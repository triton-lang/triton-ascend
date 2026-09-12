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
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from triton.backends.ascend.compiler import (
    _append_pure_simt_auto_blockify_options,
    _build_costmodel_analysis_ttir,
    _can_materialize_scope_superblock,
    _publish_route_transform_capability,
    _resolve_auto_blockify_v1_policy,
    _selected_npuir_superblock_factor,
)

pytestmark = pytest.mark.backend("cpu")

SAFE_TTIR = "module { tt.func public @safe() { tt.return } }"
ATOMIC_TTIR = "module { tt.func public @atomic() { %0 = tt.atomic_rmw add } }"
CACHE_MODIFIER_TTIR = (
    'module { tt.func public @cached() { tt.store %ptr, %value {cacheModifier = 1 : i32} : tensor<8x!tt.ptr<f16>> } }')


@pytest.mark.parametrize(
    "env_enabled,option,expected,source",
    [
        (True, None, True, "TRITON_ALL_BLOCKS_PARALLEL"),
        (False, None, False, "TRITON_ALL_BLOCKS_PARALLEL"),
        (True, False, False, "option"),
        (False, True, True, "option"),
    ],
)
def test_auto_blockify_v1_enable_policy(env_enabled, option, expected, source):
    metadata = {}
    opt = SimpleNamespace(enable_auto_blockify=option)
    with patch("triton.backends.ascend.compiler._is_auto_map_parallel_blocks_enabled", return_value=env_enabled):
        assert _resolve_auto_blockify_v1_policy(SAFE_TTIR, metadata, opt) is expected
    assert metadata["auto_blockify_v1_enabled"] is expected
    assert metadata["auto_blockify_v1_selection_source"] == source


def test_auto_blockify_v1_blacklist_disables_both_implementations():
    metadata = {}
    opt = SimpleNamespace(enable_auto_blockify=None)
    with patch("triton.backends.ascend.compiler._is_auto_map_parallel_blocks_enabled", return_value=True), \
         patch("triton.backends.ascend.compiler._warn_auto_blockify_disabled") as warn:
        assert not _resolve_auto_blockify_v1_policy(ATOMIC_TTIR, metadata, opt)
    assert metadata["has_auto_blockify_blacklist_op"]
    warn.assert_called_once()


def test_auto_blockify_v1_explicit_blacklist_override_is_preserved():
    metadata = {"has_auto_blockify_blacklist_op": False}
    opt = SimpleNamespace(enable_auto_blockify=True)
    with patch("triton.backends.ascend.compiler._is_auto_map_parallel_blocks_enabled", return_value=False):
        assert _resolve_auto_blockify_v1_policy(ATOMIC_TTIR, metadata, opt)


def test_cache_modifier_does_not_disable_auto_blockify_v1():
    metadata = {}
    opt = SimpleNamespace(enable_auto_blockify=True)
    with patch("triton.backends.ascend.compiler._is_auto_map_parallel_blocks_enabled", return_value=False):
        assert _resolve_auto_blockify_v1_policy(CACHE_MODIFIER_TTIR, metadata, opt)
    assert not metadata["has_auto_blockify_blacklist_op"]


def test_route_transform_capability_is_single_resolved_fact():
    metadata = {
        "ttir_layout_merge_applied": True,
        "ttir_layout_coalesce_factor": 8,
        "ttir_layout_coalesce_axis": 0,
        "auto_blockify_v1_requested": True,
        "auto_blockify_v1_enabled": True,
        "auto_blockify_v1_disable_reasons": [],
    }
    opt = SimpleNamespace(compile_on_910_95=True, num_warps=4, logical_program_count_hint=9)
    capability = __import__("json").loads(_publish_route_transform_capability(metadata, opt))
    assert capability["layout_coalescing_applied"]
    assert capability["layout_coalescing_factor"] == 8
    assert capability["auto_blockify_v1_materializable"]
    assert capability["modeled_superblock_factors"] == [1, 2, 4, 8, 16, 32]
    assert capability["whole_kernel_superblock_factors"] == [1, 2, 4, 8, 16]
    assert capability["scope_superblock_factors"] == [1, 2, 4]
    assert capability["source_logical_program_count_hint"] == 9
    assert capability["logical_program_count_hint"] == 2
    assert capability["superblock_runtime_groups"]["4"] == {
        "full_group_count": 0,
        "tail_count": 2,
    }
    assert capability["superblock_runtime_groups"]["16"] == {
        "full_group_count": 0,
        "tail_count": 2,
    }


@pytest.mark.parametrize(
    "num_warps,expected",
    [
        (1, [1, 2, 4, 8, 16, 32]),
        (2, [1, 2, 4, 8, 16, 32]),
        (4, [1, 2, 4, 8, 16]),
        (8, [1, 2, 4, 8]),
        (16, [1, 2, 4]),
        (32, [1, 2]),
        (64, [1]),
    ],
)
def test_route_transform_capability_respects_warp_limit(num_warps, expected):
    metadata = {
        "auto_blockify_v1_enabled": True,
        "auto_blockify_v1_disable_reasons": [],
    }
    opt = SimpleNamespace(
        compile_on_910_95=True,
        num_warps=num_warps,
        logical_program_count_hint=0,
    )
    capability = __import__("json").loads(_publish_route_transform_capability(metadata, opt))
    assert capability["whole_kernel_superblock_factors"] == expected
    assert capability["scope_superblock_factors"] == [factor for factor in expected if factor <= 4]


@pytest.mark.parametrize(
    "compile_on_910_95,num_warps,v1_materializable,expected",
    [
        (True, 1, True, True),
        (True, 4, True, True),
        (True, 4, False, False),
        (False, 4, True, False),
        (True, 0, True, False),
    ],
)
def test_scope_superblock_uses_npuir_abi_v2(compile_on_910_95, num_warps, v1_materializable, expected):
    opt = SimpleNamespace(compile_on_910_95=compile_on_910_95, num_warps=num_warps)
    assert _can_materialize_scope_superblock({}, opt, v1_materializable) is expected


@pytest.mark.parametrize(
    "metadata,option_factor,expected",
    [
        ({}, 2, 2),
        ({"auto_simt_scope_superblock_factor": 1}, 4, 4),
        ({
            "auto_simt_effective_kind": "mixed_simd_simt",
            "auto_simt_scope_superblock_factor": 2,
        }, 4, 1),
        ({
            "compile_mode": "simd_simt",
            "auto_simt_scope_superblock_factor": 4,
        }, 4, 1),
        ({
            "auto_simt_effective_kind": "all_simt_only",
            "auto_simt_superblock_factor": 4,
        }, 2, 4),
    ],
)
def test_selected_npuir_superblock_factor_respects_route_owner(metadata, option_factor, expected):
    opt = SimpleNamespace(superblock_factor=option_factor)
    assert _selected_npuir_superblock_factor(metadata, opt) == expected


@pytest.mark.parametrize(
    "ta_materializes,runtime_cap,expected",
    [
        (False, True, ["--enable-auto-blockify-loop", "--super-block-factor=32"]),
        (True, True, ["--super-block-factor=32"]),
        (True, False, []),
    ],
)
def test_pure_simt_passes_selected_factor_to_bishengir(ta_materializes, runtime_cap, expected):
    options = []
    metadata = {
        "auto_blockify_v1_enabled": True,
        "auto_blockify_v1_runtime_cap": runtime_cap,
        "auto_simt_effective_kind": "all_simt_only",
        "auto_simt_superblock_factor": 32,
    }
    opt = SimpleNamespace(
        enable_ta_auto_blockify_v1=ta_materializes,
        superblock_factor=1,
    )
    _append_pure_simt_auto_blockify_options(options, metadata, opt)
    assert options == expected


def test_costmodel_analysis_view_materializes_v1_only_on_clone():
    metadata = {"auto_blockify_v1_enabled": True}
    original = SimpleNamespace(context=object())
    analysis = SimpleNamespace()
    with patch("triton.backends.ascend.compiler._parse_ttir_text", return_value=analysis) as parse, \
         patch("triton.backends.ascend.compiler._run_ta_simt_auto_blockify_v1", return_value=True) as run_v1:
        result = _build_costmodel_analysis_ttir(original, metadata, SimpleNamespace())
    parse.assert_called_once()
    run_v1.assert_called_once()
    assert run_v1.call_args.args[0] is analysis
    assert run_v1.call_args.kwargs["super_block_factor"] == 1
    assert metadata["auto_simt_costmodel_analysis_v1_materialized"]
    assert metadata["auto_simt_costmodel_analysis_ir"] == "post_auto_blockify_v1_f1_ttir"
    assert result == str(analysis)
