from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_dependency_metadata_uses_upstream_layout_without_duplicate_pins():
    assert not (REPO / "third_party/ascend/deps").exists()
    for name in ("llvm-info.json", "json-version.txt", "bisheng-info.json"):
        assert (REPO / "cmake" / name).is_file()
    for name in ("llvm-hash.txt", "json-info.json", "bisheng-version.txt", "dependencies.py",
                 "TritonDependencies.cmake"):
        assert not (REPO / "cmake" / name).exists()
