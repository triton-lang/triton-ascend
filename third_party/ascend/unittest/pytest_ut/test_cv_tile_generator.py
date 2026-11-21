# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the MIT License.
"""Compare checkout CV generation with exhaustive, ordered leaf filtering."""

import ast
import importlib
import itertools
import math
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.backend("none")


@pytest.fixture
def cv():
    # Load this checkout without native backend/device initialization. Config,
    # schema, normalization, hardware rules and pruning still execute real code.
    root = Path(__file__).resolve().parents[4]
    runtime = root / "third_party/ascend/backend/runtime"
    prefix = "_cv_tile_test"
    with pytest.MonkeyPatch.context() as patch:

        def package(name, path=None):
            module = types.ModuleType(name)
            module.__path__ = [] if path is None else [str(path)]
            patch.setitem(sys.modules, name, module)
            return module

        for name in ("triton", "triton.runtime", "triton.backends", "triton.backends.ascend"):
            package(name)
        config_module = types.ModuleType("triton.runtime.autotuner")
        config_path = root / "python/triton/runtime/autotuner.py"
        config_node = next(node for node in ast.parse(config_path.read_text()).body
                           if isinstance(node, ast.ClassDef) and node.name == "Config")
        exec(compile(ast.Module(body=[config_node], type_ignores=[]), str(config_path), "exec"), config_module.__dict__)
        patch.setitem(sys.modules, config_module.__name__, config_module)
        utils = types.ModuleType("triton.backends.ascend.utils")
        utils.is_register_based_arch = lambda arch: False
        patch.setitem(sys.modules, utils.__name__, utils)
        package(prefix, runtime)
        runtime_utils = types.ModuleType(prefix + ".utils")
        runtime_utils.target = types.SimpleNamespace(arch="Ascend910B")
        patch.setitem(sys.modules, runtime_utils.__name__, runtime_utils)
        for suffix in ("cv_autotune", "cv_autotune.generators", "cv_autotune.pruning", "cv_autotune.normalization",
                       "dsl_analysis"):
            package(prefix + "." + suffix, runtime.joinpath(*suffix.split(".")))
        try:
            generator = importlib.import_module(prefix + ".cv_autotune.generators.cv_tile_generator")
            schema = importlib.import_module(prefix + ".dsl_analysis.schema")
            hardware = importlib.import_module(prefix + ".cv_autotune.hardware_consts")
            yield types.SimpleNamespace(generator=generator, schema=schema, hardware=hardware,
                                        Config=config_module.Config)
        finally:
            # importlib also registers leaf modules not installed through patch.
            for name in list(sys.modules):
                if name.startswith(prefix + "."):
                    sys.modules.pop(name, None)


def _axis(cv, name=None, maximum=128, const=None, state=None):
    return cv.schema.CvAxisLengthInfo(name,
                                      str(maximum) if name else None, state
                                      or ("tunable" if name else "fixed_compile_time"), name, const)


def _site(cv, index, names=("M", "N", "K"), maximum=128):
    return cv.schema.CvDotSiteInfo(index, "a", "b", *(_axis(cv, name, maximum) for name in names))


def _parsed(cv, sites, tunables=()):
    return cv.schema.CvParseResult("cv-parse-v1", "mix", "ok", [], list(sites), list(tunables), [], [])


def _exhaustive(generator, Config):
    """Independent old leaf-filter semantics; no early dependency scheduling."""
    if not generator.params:
        return [], 0
    configs, pruned = [], 0
    for values in itertools.product(*(param.get_values() for param in generator.params)):
        kwargs = dict(zip((param.name for param in generator.params), values))
        for site in generator.cv_parse_result.dot_sites:
            full = kwargs.copy()
            for role, info in (("M", site.m), ("N", site.n), ("K", site.k)):
                key = "standard_block_" + role.lower()
                if key in full:
                    continue
                if info.tunable_param and info.tunable_param in kwargs:
                    full[key] = kwargs[info.tunable_param]
                elif info.const_value is not None:
                    full[key] = info.const_value
                else:
                    full[key] = 64 if role == "K" else 128
            if generator.pruner.should_prune(full, generator.dtype)[0]:
                pruned += 1
                break
        else:
            configs.append(Config(kwargs=kwargs, num_warps=1, num_stages=1))
    return configs, pruned


def _assert_matches(cv, generator):
    expected, pruned = _exhaustive(generator, cv.Config)
    before = generator.stats.copy()
    actual = generator.generate_configs()
    assert actual == expected
    assert [list(config.kwargs.items()) for config in actual] == [list(config.kwargs.items()) for config in expected]
    assert [vars(config) for config in actual] == [vars(config) for config in expected]
    assert generator.stats == {**before, "pruned": before["pruned"] + pruned}
    if generator.params:
        assert pruned + len(actual) == math.prod(param.count_values() for param in generator.params)
    return actual


@pytest.mark.parametrize("maximum", [16, 64, 128, 256])
@pytest.mark.parametrize("shared", [False, True])
def test_multiple_dots_ordered(cv, maximum, shared):
    sites = [_site(cv, 0, ("M0", "N0", "K0"), maximum), _site(cv, 1, ("M0" if shared else "M1", "N1", "K1"), maximum)]
    generator = cv.generator.CVTileGenerator(_parsed(cv, sites))
    result = _assert_matches(cv, generator)
    if maximum == 256 and not shared:
        assert len(result) == 256036
        assert generator.stats["pruned"] == 743964


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int8])
@pytest.mark.parametrize("pattern", ["{}", "BLOCK_{}", "block_{}", "standard_block_{}"])
def test_dtype_and_dimension_names(cv, dtype, pattern):
    roles = "MNK" if pattern in ("{}", "BLOCK_{}") else "mnk"
    site = _site(cv, 0, tuple(pattern.format(role) for role in roles))
    _assert_matches(cv, cv.generator.CVTileGenerator(_parsed(cv, [site]), dtype=dtype))


@pytest.mark.parametrize("role", [0, 1, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_cross_dot_alias_precedence(cv, role, reverse):
    names0, names1 = ["M0", "N0", "K0"], ["M1", "N1", "K1"]
    aliases = ["standard_block_" + "mnk"[role], "BLOCK_" + "MNK"[role]]
    names0[role], names1[role] = aliases[::-1] if reverse else aliases
    generator = cv.generator.CVTileGenerator(_parsed(cv, [_site(cv, 0, names0, 256), _site(cv, 1, names1, 256)]))
    for param in generator.params:
        param.values = [16, 256]
    result = _assert_matches(cv, generator)
    if role == 0 and not reverse:
        # A later BLOCK_M=16 rescues the first site's otherwise invalid prefix.
        assert any(list(config.kwargs.values()) == [256, 16, 256, 16, 16, 16] for config in result)


@pytest.mark.parametrize("kind", ["fixed", "fallback", "inferred", "runtime", "shared_upper", "all_fixed", "no_dot"])
def test_dimension_resolution(cv, kind):
    fixed = _axis(cv, const=64)
    site = replace(_site(cv, 0), n=fixed, k=fixed)
    all_args = {}
    if kind == "fallback":
        site = replace(site, k=_axis(cv, state="runtime_non_tunable"))
    elif kind == "inferred":
        site = replace(site, m=replace(site.m, tunable_param=None))
    elif kind == "runtime":
        site = replace(site, m=replace(site.m, length_expr="runtime_m"))
        all_args = {"runtime_m": 96}
    elif kind == "all_fixed":
        site = replace(site, m=fixed)
    sites = [site]
    tunables = []
    if kind == "shared_upper":
        sites.append(replace(site, site_id=1, m=replace(site.m, length_expr="192")))
    elif kind == "no_dot":
        sites = []
        tunables = [cv.schema.CvTunableParamInfo("BLOCK_K", "K", "signature")]
    _assert_matches(cv, cv.generator.CVTileGenerator(_parsed(cv, sites, tunables), all_args=all_args))


def test_mixed_parameter_types_keep_value_order(cv):
    generator = cv.generator.CVTileGenerator(_parsed(cv, []))
    generator.params = [
        cv.generator.ParamSpace("X", "NUM", min=16, max=48, stride=16),
        cv.generator.ParamSpace("Y", "ENUM", values=[128, 16]),
        cv.generator.ParamSpace("FLAG", "BOOL")
    ]
    result = _assert_matches(cv, generator)
    assert len(result) == 12
    assert list(result[0].kwargs.values()) == [16, 128, True]
    assert list(result[-1].kwargs.values()) == [48, 16, False]


def test_pruned_subtrees_count_unequal_domains_once(cv, monkeypatch):
    fixed16, fixed256 = _axis(cv, const=16), _axis(cv, const=256)
    first = replace(_site(cv, 0), n=fixed16, k=fixed256)
    second = replace(first, site_id=1, n=_axis(cv, "N"))
    generator = cv.generator.CVTileGenerator(_parsed(cv, [first, second]))
    generator.params = [
        cv.generator.ParamSpace("M", "NUM", min=16, max=256, stride=240),
        cv.generator.ParamSpace("N", "ENUM", values=[16, 128, 32]),
        cv.generator.ParamSpace("FLAG", "BOOL")
    ]
    expected, pruned = _exhaustive(generator, cv.Config)
    calls = []
    original = generator.pruner.should_prune

    def record(config, dtype):
        calls.append(config.copy())
        return original(config, dtype)

    monkeypatch.setattr(generator.pruner, "should_prune", record)
    assert generator.generate_configs() == expected
    assert len(expected) == 4
    assert generator.stats["pruned"] == pruned == 8
    assert len(calls) == 5  # First site: 2 prefixes; second: 3, only under M=16.
    assert all("FLAG" not in config for config in calls)
    assert sum(config["M"] == 256 for config in calls) == 1


def test_fixed_root_rejection_does_not_visit_descendants(cv, monkeypatch):
    fixed = _axis(cv, const=4096)
    first = replace(_site(cv, 0), m=fixed, n=fixed, k=fixed)
    sites = [first, replace(first, site_id=1), _site(cv, 2, maximum=64)]
    generator = cv.generator.CVTileGenerator(_parsed(cv, sites))
    expected, pruned = _exhaustive(generator, cv.Config)
    calls = []
    original = generator.pruner.should_prune

    def record(config, dtype):
        calls.append(config.copy())
        return original(config, dtype)

    monkeypatch.setattr(generator.pruner, "should_prune", record)
    assert generator.generate_configs() == expected == []
    assert generator.stats["pruned"] == pruned == 64
    assert len(calls) == 1
    assert set(calls[0]) == {"standard_block_m", "standard_block_n", "standard_block_k"}


def test_repeated_generation_preserves_stats_and_output_ownership(cv):
    generator = cv.generator.CVTileGenerator(_parsed(cv, [_site(cv, 0)]), dtype=torch.float32)
    first = _assert_matches(cv, generator)
    pruned = generator.stats["pruned"]
    second = _assert_matches(cv, generator)
    assert first == second
    assert generator.stats["pruned"] == 2 * pruned > 0
    first[0].kwargs["M"] = -1
    assert second[0].kwargs["M"] == 16
    assert first[1].kwargs["M"] == 16


def test_custom_hardware_constraints(cv):
    hardware = cv.hardware.HardwareConstraints(l0a_size=16 * 1024, l0b_size=16 * 1024, l0c_size=64 * 1024,
                                               max_l1_size=64 * 1024)
    generator = cv.generator.CVTileGenerator(_parsed(cv, [_site(cv, 0)]), hardware_constraints=hardware)
    result = _assert_matches(cv, generator)
    assert result
    assert generator.stats["pruned"] > 0
