import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def load_info(monkeypatch):
    spec = importlib.util.spec_from_file_location("llvm_metadata_checker", REPO / "scripts/check_bom_consistency.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def load(root):
        monkeypatch.setattr(module, "REPO_ROOT", root)
        monkeypatch.setattr(module, "DEPS_DIR", root / "cmake")
        return module._load_llvm_info()

    return load


@pytest.fixture
def inputs(tmp_path):
    metadata = tmp_path / "cmake" / "llvm-info.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes((REPO / "cmake/llvm-info.json").read_bytes())
    data = json.loads(metadata.read_text())
    data["llvm_hash"] = "a" * 40
    metadata.write_text(json.dumps(data))
    patches = tmp_path / "third_party" / "ascend" / "patch"
    patches.mkdir(parents=True)
    for name in data["patches"]:
        (patches / name).write_text("fixture patch")
    return tmp_path, metadata, metadata


def test_llvm_revision_has_one_authority(load_info, inputs):
    root, metadata, pin = inputs
    assert "llvm_hash_file" not in json.loads(metadata.read_text())
    assert load_info(root)["llvm_hash"] == "a" * 40
    data = json.loads(pin.read_text())
    data["llvm_hash"] = "b" * 40
    pin.write_text(json.dumps(data))
    assert load_info(root)["llvm_hash"] == "b" * 40


@pytest.mark.parametrize("value", ["", "main", "a" * 8, "x" * 40])
def test_invalid_canonical_revision_is_rejected(load_info, inputs, value):
    root, _, pin = inputs
    data = json.loads(pin.read_text())
    data["llvm_hash"] = value
    pin.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="40 hexadecimal"):
        load_info(root)


@pytest.mark.parametrize("change",
                         ["missing", "traversal", "duplicate", "extra", "schema", "copied-pin", "newline-name"])
def test_invalid_metadata_is_rejected(load_info, inputs, change):
    root, metadata, _ = inputs
    data = json.loads(metadata.read_text())
    if change == "missing":
        data.pop("llvm_hash", None)
    elif change == "traversal":
        data["llvm_hash_file"] = "../llvm-hash.txt"
    elif change == "duplicate":
        data["patches"] += data["patches"]
    elif change == "extra":
        (root / "third_party/ascend/patch/llvm_patch_extra.patch").touch()
    elif change == "schema":
        data["schema_version"] = True
    elif change == "newline-name":
        data["patches"] = ["llvm_patch_bad\nname.patch"]
    else:
        data["llvm_hash_file"] = "legacy-pin.txt"
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_info(root)


def test_duplicate_json_keys_are_rejected(load_info, inputs):
    root, metadata, _ = inputs
    metadata.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="duplicate"):
        load_info(root)
