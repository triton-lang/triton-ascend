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

"""Host-only policy/merge tests; run with --confcutdir pointing at this directory.

Load the policy and config-generation method from source so these tests need
neither libtriton nor an NPU. Compiler option declarations retain their real
field defaults/metadata; compiler initialization and device execution are omitted.
"""
import ast
from dataclasses import InitVar, dataclass, field, fields
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"


def _execute(nodes, namespace, filename):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(filename), "exec"), namespace)


@pytest.fixture
def policy(monkeypatch):
    namespace = {
        "__name__": __name__, "dataclass": dataclass, "field": field, "fields": fields, "InitVar": InitVar,
        "DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK": 0, "get_libdevice": lambda: "",
        "_multibuffer_mode_to_tuple": lambda mode: mode,
        "get_byte_per_numel": lambda dtype: 4,
    }
    compiler_path = BACKEND / "compiler.py"
    compiler = ast.parse(compiler_path.read_text())
    options = next(node for node in compiler.body if isinstance(node, ast.ClassDef) and node.name == "NPUOptions")
    options.body = [node for node in options.body if isinstance(node, ast.AnnAssign)]
    _execute([options], namespace, compiler_path)
    module = ModuleType("triton.backends.ascend.compiler")
    module.NPUOptions = namespace["NPUOptions"]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    utils_path = BACKEND / "utils.py"
    utils = ast.parse(utils_path.read_text())
    constants = {"_DEPRECATED_NPU_OPTIONS", "_RESERVED_NPU_OPTION_NAMES"}
    _execute([
        node for node in utils.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)
    ], namespace, utils_path)

    autotuner_path = BACKEND / "runtime" / "autotuner.py"
    autotuner = ast.parse(autotuner_path.read_text())
    names = {"_resolve_auto_gen_config", "_normalize_config_hints", "_expand_configs_with_hints"}
    constants = {"_COMMON_COMPILE_OPTIONS", "_DEFAULT_COMPILE_MODE"}
    nodes = [
        node for node in autotuner.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets))
    ]
    tuner = next(node for node in autotuner.body if isinstance(node, ast.ClassDef) and node.name == "AutoTilingTuner")
    nodes.extend(node for node in tuner.body if isinstance(node, ast.FunctionDef) and node.name == "generate_key_and_configs")
    _execute(nodes, namespace, autotuner_path)
    return SimpleNamespace(**namespace)


def config(**kwargs):
    return SimpleNamespace(kwargs=kwargs, num_stages=3, num_warps=4, num_ctas=1, maxnreg=None)


@pytest.mark.parametrize("configs", [None, []])
@pytest.mark.parametrize("hints", [{}, {"auto_gen_config": False}, {"auto_gen_config": True}])
def test_empty_configs_keep_automatic_generation(policy, configs, hints):
    assert policy._resolve_auto_gen_config(configs, hints) is True


@pytest.mark.parametrize("kwargs", [
    {}, {"BLOCK_SIZE": 128}, {"BLOCK_SIZE": 128, "num_stages": 2},
    {"num_warps": 8, "num_ctas": 1, "maxnreg": 64}, {"debug": False, "enable_fp_fusion": True},
])
def test_kernel_and_common_options_enable_generation(policy, kwargs):
    assert policy._resolve_auto_gen_config([config(**kwargs)], {}) is True


@pytest.mark.parametrize("name,value", [
    ("multibuffer", True), ("multibuffer", False), ("multibuffer", None),
    ("multibuffer_mode", (("ub", 2),)), ("vf_fusion_mode", "ub-aware-op"),
    ("enable_ubuf_saving", False), ("unit_flag", False), ("set_workspace_multibuffer", 0),
    ("compile_mode", "simt_only"), ("simt_optimization_mode", 0),
    ("enable_hivm_batch_matmul", False), ("buf_slot_num_of_veccore", 2),
])
def test_any_explicit_ascend_option_disables_generation(policy, name, value):
    configs = [config(BLOCK_SIZE=128), config(BLOCK_SIZE=256, **{name: value})]
    assert policy._resolve_auto_gen_config(configs, {}) is False
    assert policy._resolve_auto_gen_config(list(reversed(configs)), {}) is False


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("kwargs", [{"BLOCK_SIZE": 128}, {"multibuffer": False}])
def test_explicit_hint_overrides_nonempty_config_default(policy, enabled, kwargs):
    assert policy._resolve_auto_gen_config([config(**kwargs)], {"auto_gen_config": enabled}) is enabled


def test_schema_detection_does_not_require_constructing_options(policy):
    # Construction would normally install compiler patches and require the
    # native extension. An option added to the schema must also be detected.
    from dataclasses import make_dataclass
    module = sys.modules["triton.backends.ascend.compiler"]
    module.NPUOptions = make_dataclass("NewOptions", [("new_ascend_option", bool, False)])
    assert policy._resolve_auto_gen_config([config(new_ascend_option=False)], {}) is False


@pytest.mark.parametrize("kwargs,hints,expected_generated", [
    ({"BLOCK_SIZE": 128}, {}, True),
    ({"BLOCK_SIZE": 128, "multibuffer": False}, {}, False),
    ({"BLOCK_SIZE": 128}, {"auto_gen_config": False}, False),
    ({"BLOCK_SIZE": 128, "multibuffer": False}, {"auto_gen_config": True}, True),
])
def test_generation_combines_candidates_and_preserves_user_configs(policy, kwargs, hints, expected_generated):
    user_config = config(**kwargs)
    generated_config = config(BLOCK_SIZE=512)
    parse_calls = []
    tuner = SimpleNamespace(
        arg_names=["x", "n"], keys=["n"], cache={}, fn=None, _raw_config_hints={},
        user_configs=[user_config], gen_configs=[], axis_arg_names={"x": "n"},
        auto_gen_config=policy._resolve_auto_gen_config([user_config], hints),
    )
    tuner._autoparse_axis_params = lambda args: parse_calls.append(args)
    tuner._gen_tile_configs = lambda *args: tuner.gen_configs.append(generated_config)
    key = policy.generate_key_and_configs(tuner, SimpleNamespace(dtype="float32"), 1024)
    assert key == (1024, "float32", ("compile_mode", "simd_simt_template"))
    assert bool(parse_calls) is expected_generated
    assert tuner.configs == ([generated_config, user_config] if expected_generated else [user_config])
    assert tuner.user_configs == [user_config]
    assert user_config.kwargs == kwargs
    # A tuning-cache hit must not regenerate candidates.
    tuner.cache[key] = user_config
    policy.generate_key_and_configs(tuner, SimpleNamespace(dtype="float32"), 1024)
    assert len(parse_calls) == int(expected_generated)
