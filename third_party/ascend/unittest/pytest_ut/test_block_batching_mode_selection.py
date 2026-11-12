# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Select VV/CV from native IR through the public, unified batching options."""
import hashlib
from collections import namedtuple
from types import SimpleNamespace

import pytest
from triton import knobs
from triton._C.libtriton import ascend
from triton.backends.ascend import block_batching as batching
from triton.backends.ascend.block_batching_report import CV_VALIDATION_POLICY, VALIDATION_POLICY
from triton.backends.ascend.compiler import NPUOptions, make_ttir
from test_block_batching_cv import dot_module
from test_block_batching_cv_loop_fusion import spatial_loop
from test_block_batching_cv_spatial import spatial_module
from test_block_batching_rewrite import parse, pointwise

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


@pytest.mark.parametrize("text,cv", [
    (pointwise(0), False),
    (dot_module(), True),
    (spatial_loop(branch=True), True),
    (pointwise(0).replace("module {", 'module attributes {test.note = "tt.dot"} {', 1), False),
])
def test_classification_reads_operations_without_mutation(tmp_path, text, cv):
    module = parse(tmp_path, text)
    before = str(module)
    assert batching._has_dot(module) is cv
    assert str(module) == before


@pytest.mark.parametrize("text,factor,mode", [
    (pointwise(0), 3, "disabled"),
    (dot_module(), 2, "independent_batch_v1"),
    (spatial_module("m"), 2, "spatial_concat_v1"),
    (spatial_module("n"), 2, "spatial_concat_v1"),
    (spatial_loop(branch=True), 2, "spatial_concat_v1"),
])
@pytest.mark.parametrize("axis", [-1, 0])
def test_public_factor_selects_native_vv_or_cv(tmp_path, text, factor, mode, axis):
    module = parse(tmp_path, text)
    options = NPUOptions(arch="Ascend950", block_batching_factor=factor, block_batching_axis=axis,
                         enable_graph_optimize=False)
    metadata = {}
    make_ttir(module, metadata, options)
    report = metadata["block_batching"]
    assert report["compile_status"] == "REWRITTEN"
    assert report["cv_mode"] == mode
    assert report["selected_axis"] == 0
    cv = mode != "disabled"
    assert report["tail_policy"] == ("full_groups_only" if cv else "masked")
    assert report["layout"]["selected_policy"] == ("semantic_only" if cv else "auto")
    assert module.verify()


def test_cv_preflight_uses_the_same_automatic_classification(tmp_path):
    module = parse(tmp_path, dot_module())
    report = batching.analyze_request(module, batching.BatchRequest(2, 0), ascend.passes.ttir)
    assert report["compile_status"] == "ANALYZED", report


def test_cv_restrictions_still_reject_unsupported_factors(tmp_path):
    module = parse(tmp_path, dot_module())
    options = NPUOptions(arch="Ascend950", block_batching_factor=3, block_batching_axis=0, enable_graph_optimize=False)
    with pytest.raises(batching.BatchingRejected, match="CV_FACTOR_REQUIRES_POWER_OF_TWO"):
        make_ttir(module, {}, options)


def test_factor_one_preserves_the_unbatched_cv_path(tmp_path):
    module = parse(tmp_path, spatial_module())
    metadata = {}
    make_ttir(module, metadata, NPUOptions(arch="Ascend950", enable_graph_optimize=False))
    assert "block_batching" not in metadata
    assert "__hacc_original_grid" not in str(module)
    assert module.verify()


def test_cv_is_not_a_public_compiler_option():
    assert "block_batching_cv" not in NPUOptions.__dataclass_fields__
    with pytest.raises(TypeError, match="block_batching_cv"):
        NPUOptions(arch="Ascend950", block_batching_cv=True)


@pytest.mark.parametrize("change,reason", [
    ({}, None),
    ({"validation_policy": VALIDATION_POLICY}, "CANDIDATE_VALIDATION_MISSING"),
    ({"cv_mode": "unknown"}, "CANDIDATE_VALIDATION_MISSING"),
    ({"tail_policy": "masked"}, "CV_CONTRACT_MISMATCH"),
])
def test_bundle_validates_the_selected_cv_contract(tmp_path, change, reason):
    module = parse(tmp_path, spatial_module())
    options = dict(block_batching_factor=2, block_batching_axis=0)
    metadata = {}
    make_ttir(module, metadata, NPUOptions(arch="Ascend950", enable_graph_optimize=False, **options))
    report = metadata["block_batching"]
    report.update(compile_status="COMPILED_GUARDED", validation_policy=CV_VALIDATION_POLICY)
    report.update(change)
    binary = b"mock backend artifact"
    fields = dict(block_batching=report, block_batching_binary_sha256=hashlib.sha256(binary).hexdigest(),
                  block_batching_plan_sha256=report["guard"]["plan_fingerprint"])
    candidate = SimpleNamespace(hash=hashlib.sha256(binary).hexdigest(), kernel=binary,
                                metadata=namedtuple("Metadata", fields)(**fields))
    baseline = SimpleNamespace(hash="baseline", metadata=namedtuple("BaselineMetadata", [])())
    backend = SimpleNamespace(
        target=SimpleNamespace(arch="Ascend950"), parse_options=lambda opts: SimpleNamespace(
            is_pure_simt=False, compile_mode="simd", block_batching_backend_revision="test"))
    src = SimpleNamespace(fn=SimpleNamespace(arg_names=["x", "y", "z", "limit"]),
                          signature={"x": "*fp32", "y": "*fp32", "z": "*fp32", "limit": "i32"}, constants={})
    with knobs.cache.scope():
        knobs.cache.dir = str(tmp_path / "cache")
        bundle = batching.compile_bundle(
            backend, src, options, lambda opts, frozen: baseline if opts["block_batching_factor"] == 1 else candidate)
    if reason is None:
        assert bundle.candidate is candidate
    else:
        assert bundle.candidate is None
        assert bundle.block_batching["reason"] == reason
