"""Load operation isolation, static-mask visibility and frozen formula limits."""
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[4]
COST = ROOT / "third_party/ascend/costmodel"
sys.path.insert(0, str(COST / "python"))
from indirect_load_selected import predict

MODELS = {m: json.loads((COST / f"profiles/indirect_load_selected/{m}_load.json").read_text())["model"]
          for m in ("SIMD", "SIMT")}
CONTRACT = dict(stage_ownership=True, matched_increment=True)


def case(mode="SIMD", mask="none"):
    return dict(op="load", mode=mode, F=1, W=1 if mode == "SIMD" else 4,
                shape=[4, 8], random=[False, True], lo=[0, 0], hi=[4, 8],
                E=32, K=32, bytes=4, payload_dtype="float32", visibility="known",
                coefficients=[8, 1], mask=mask)


@pytest.mark.parametrize("mode", ["SIMD", "SIMT"])
@pytest.mark.parametrize("masked", [False, True])
def test_static_Load_paths(mode, masked):
    c = case(mode, "box" if masked else "none")
    if masked:
        c.update(hi=[4, 4], K=16)
    result = predict(MODELS[mode], c, **CONTRACT)
    assert result["eligible"] and result["ticks"] > 0


def test_operation_and_dynamic_mask_do_not_reuse_static_K():
    c = case()
    c["op"] = "store"
    with pytest.raises(ValueError, match="ordinary Load"):
        predict(MODELS["SIMD"], c, **CONTRACT)
    c.update(op="load", mask="runtime_index_valid", K_visible=False)
    with pytest.raises(ValueError, match="Host contract"):
        predict(MODELS["SIMD"], c, **CONTRACT)


def test_explicit_disable_ownership_and_factor():
    c = case()
    assert not predict(MODELS["SIMD"], c)["eligible"]
    assert not predict(MODELS["SIMD"], c, enabled=False, **CONTRACT)["eligible"]
    c["F"] = 8
    with pytest.raises(ValueError, match="F>1"):
        predict(MODELS["SIMD"], c, **CONTRACT)


def test_Device_hidden_strides_not_inspected():
    class Hidden:
        def __len__(self):
            raise AssertionError("hidden coefficient read")
        def __getitem__(self, item):
            raise AssertionError("hidden coefficient read")
    c = case("SIMT")
    c.update(visibility="device", coefficients=Hidden())
    assert predict(MODELS["SIMT"], c, Hidden(), **CONTRACT)["eligible"]


def test_Host_stride_values_and_formula_complexity():
    c = case("SIMT")
    c["visibility"] = "host"
    with pytest.raises(ValueError, match="explicit"):
        predict(MODELS["SIMT"], c, **CONTRACT)
    assert predict(MODELS["SIMT"], c, [8, 1], **CONTRACT)["eligible"]
    for model in MODELS.values():
        for formula in [*model["models"].values(), model["fallback"]]:
            assert 1 <= len(formula["features"]) == len(formula["coefficients"]) <= 8


@pytest.mark.parametrize("dtype", ["int1", "float8e4nv", "float8e5"])
def test_logical_payload_width_must_match(dtype):
    c = case()
    c["payload_dtype"] = dtype
    with pytest.raises(ValueError, match="payload type"):
        predict(MODELS["SIMD"], c, **CONTRACT)
