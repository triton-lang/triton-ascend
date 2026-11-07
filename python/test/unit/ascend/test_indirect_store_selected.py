"""CPU arithmetic and disclosure guards; this file never imports an NPU runtime."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
COSTMODEL = ROOT / "third_party/ascend/costmodel"
sys.path.insert(0, str(COSTMODEL / "python"))
from indirect_store_selected.predictor import predict  # noqa: E402
from indirect_store_selected.visible_features import visible_features  # noqa: E402


def load_json(path):
    return json.loads(path.read_text())


FIXTURES = load_json(COSTMODEL / "tests/indirect_store_selected_metadata.json")["cases"]
MODELS = {
    mode: load_json(COSTMODEL / f"profiles/indirect_store_selected/{mode}_store.json")
    for mode in ("SIMD", "SIMT")
}
CONTRACT = dict(enabled=True, stage_ownership=True, first_store=True, matched_increment=True)


@pytest.mark.parametrize("row", FIXTURES, ids=lambda row: row["id"])
def test_selected_reference_arithmetic(row):
    c = row["metadata"]
    result = predict(MODELS[c["mode"]], c, row["explicit_host_coefficients"], **CONTRACT)
    assert result["eligible"] == row["expected"]["eligible"]
    if result["eligible"]:
        assert result["ticks"] == pytest.approx(row["expected"]["ticks"], rel=1e-12, abs=1e-9)
    assert not result["accepted"]


def test_explicit_disable_and_ownership_are_required():
    c = FIXTURES[0]["metadata"]
    assert not predict(MODELS[c["mode"]], c, enabled=False)["eligible"]
    for missing in ("stage_ownership", "first_store", "matched_increment"):
        contract = dict(CONTRACT, **{missing: False})
        assert not predict(MODELS[c["mode"]], c, **contract)["eligible"]


class UnreadableCoefficients:

    def __getitem__(self, index):
        raise AssertionError("Device-hidden coefficients were accessed")

    def __len__(self):
        raise AssertionError("Device-hidden coefficients were inspected")


def test_device_hidden_coefficients_are_never_read():
    c = copy.deepcopy(next(row["metadata"] for row in FIXTURES if row["metadata"]["visibility"] == "device"))
    before = visible_features(c)
    c["coefficients"] = UnreadableCoefficients()
    assert visible_features(c, UnreadableCoefficients()) == before


def test_host_coefficients_must_be_explicit():
    c = copy.deepcopy(
        next(row["metadata"] for row in FIXTURES if row["metadata"]["visibility"] == "host" and row["metadata"]["K"]))
    with pytest.raises(ValueError, match="coefficients"):
        predict(MODELS[c["mode"]], c, **CONTRACT)


def test_final_short_formulas_have_at_most_eight_terms():
    for model in MODELS.values():
        assert model["strategy"] == "width_mask_size"
        for leaf in [*model["models"].values(), model["fallback"]]:
            assert 1 <= len(leaf["features"]) <= 8
            assert len(leaf["features"]) == len(leaf["coefficients"])
