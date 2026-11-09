import pytest


@pytest.mark.parametrize("revision", ["a" * 40, "b" * 40])
def test_llvm_revision_has_one_text_authority(checker, revision):
    (checker.DEPS_DIR / "llvm-hash.txt").write_text(revision + "\n")
    assert checker._load_llvm_info()["llvm_hash"] == revision


@pytest.mark.parametrize("value", ["", "main", "a" * 8, "x" * 40, "a" * 40 + "\n" + "b" * 40])
def test_invalid_canonical_revision_is_rejected(checker, value):
    (checker.DEPS_DIR / "llvm-hash.txt").write_text(value)
    with pytest.raises(ValueError, match="40 hexadecimal"):
        checker._load_llvm_info()


def test_llvm_patch_discovery_uses_sorted_local_files(checker):
    patches = checker.REPO_ROOT / "third_party/ascend/patch"
    patches.mkdir(parents=True)
    (patches / "llvm_patch_z.patch").touch()
    (patches / "llvm_patch_A.patch").touch()
    (patches / "llvm_patch_directory.patch").mkdir()
    (patches / "unrelated.patch").touch()
    assert checker._load_llvm_info()["patches"] == ["llvm_patch_A.patch", "llvm_patch_z.patch"]
