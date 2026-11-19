"""CPU-only content-identity contracts for the source backend."""
import importlib.util
import os
from pathlib import Path
import pytest

pytestmark = pytest.mark.backend("none")


@pytest.fixture
def toolchain_module():
    path = Path(__file__).resolve().parents[2] / "backend" / "toolchain.py"
    spec = importlib.util.spec_from_file_location("ascend_toolchain_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def toolchain(tmp_path):
    paths = {}
    for role, relative in {
        "compiler": "frontend/bin/bishengir-compile", "reader": "frontend/bin/bishengir-opt",
        "emitter": "triton/triton-mlir-opt", "optimizer": "triton/triton-opt",
        "libdevice": "triton/lib/libdevice.bc", "template": "frontend/lib/cube.bc",
        "downstream": "downstream/bisheng",
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((role + "-version-one").encode())
        path.chmod(0o755)
        paths[role] = path
    paths["env"] = {"PATH": str(tmp_path / "downstream")}
    return paths


def fingerprint(module, chain):
    return module.toolchain_fingerprint(chain["compiler"], chain["reader"],
                                        (chain["emitter"], chain["optimizer"]), chain["libdevice"], chain["env"])


def test_unchanged_toolchain_is_stable(toolchain_module, toolchain):
    assert fingerprint(toolchain_module, toolchain) == fingerprint(toolchain_module, toolchain)


def test_direct_frontend_uses_only_its_selected_tools(toolchain_module, toolchain):
    def direct_key():
        return toolchain_module.toolchain_fingerprint(toolchain["compiler"], None, (),
                                                      toolchain["libdevice"], toolchain["env"])

    original = direct_key()
    for role in ("reader", "emitter", "optimizer"):
        toolchain[role].unlink()
    assert direct_key() == original
    toolchain["libdevice"].write_bytes(b"updated target libdevice")
    assert direct_key() != original


@pytest.mark.parametrize("role", ["compiler", "reader", "emitter", "optimizer", "libdevice", "template", "downstream"])
def test_changed_component_invalidates_key(toolchain_module, toolchain, role):
    before = fingerprint(toolchain_module, toolchain)
    path = toolchain[role]
    path.write_bytes(path.read_bytes() + b"changed")
    assert fingerprint(toolchain_module, toolchain) != before


def test_same_size_replacement_with_preserved_mtime_invalidates(toolchain_module, toolchain):
    before = fingerprint(toolchain_module, toolchain)
    path = toolchain["compiler"]
    original = path.stat()
    replacement = path.with_name("replacement")
    replacement.write_bytes(b"x" * original.st_size)
    replacement.chmod(0o755)
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    replacement.replace(path)
    assert fingerprint(toolchain_module, toolchain) != before


def test_touch_without_content_change_keeps_key(toolchain_module, toolchain):
    before = fingerprint(toolchain_module, toolchain)
    toolchain["compiler"].touch()
    assert fingerprint(toolchain_module, toolchain) == before


def test_template_addition_and_removal_are_observed(toolchain_module, toolchain):
    before = fingerprint(toolchain_module, toolchain)
    extra = toolchain["template"].with_name("new-template.bc")
    extra.write_bytes(b"new template")
    assert fingerprint(toolchain_module, toolchain) != before
    extra.unlink()
    assert fingerprint(toolchain_module, toolchain) == before


def test_frontend_symlink_retarget_is_observed(toolchain_module, toolchain):
    original = toolchain["compiler"]
    link = original.with_name("selected-frontend")
    link.symlink_to(original)
    toolchain["compiler"] = link
    before = fingerprint(toolchain_module, toolchain)
    replacement = original.with_name("different-frontend")
    replacement.write_bytes(b"different compiler")
    link.unlink()
    link.symlink_to(replacement)
    assert fingerprint(toolchain_module, toolchain) != before


def test_capture_tracks_underlying_frontend_and_templates(toolchain_module, toolchain, tmp_path):
    real = toolchain["compiler"]
    wrapper = tmp_path / "capture" / "bishengir-compile"
    wrapper.parent.mkdir()
    wrapper.write_bytes(b"capture wrapper")
    toolchain["compiler"] = wrapper
    toolchain["env"].update(BISHENG_CAPTURE_WRAPPER=str(wrapper), TRITON_REAL_BISHENGIR_COMPILE=str(real))
    before = fingerprint(toolchain_module, toolchain)
    toolchain["template"].write_bytes(b"changed real template")
    assert fingerprint(toolchain_module, toolchain) != before
    before = fingerprint(toolchain_module, toolchain)
    real.write_bytes(b"changed real frontend")
    assert fingerprint(toolchain_module, toolchain) != before


def test_unselected_capture_environment_does_not_replace_frontend(toolchain_module, toolchain, tmp_path):
    before = fingerprint(toolchain_module, toolchain)
    toolchain["env"].update(BISHENG_CAPTURE_WRAPPER=str(tmp_path / "unused-wrapper"),
                            TRITON_REAL_BISHENGIR_COMPILE=str(tmp_path / "unused-frontend"))
    assert fingerprint(toolchain_module, toolchain) == before


def test_missing_required_tool_is_not_silently_ignored(toolchain_module, toolchain):
    toolchain["reader"].unlink()
    with pytest.raises(FileNotFoundError):
        fingerprint(toolchain_module, toolchain)


def test_explicit_downstream_directory_is_fingerprinted(toolchain_module, toolchain, tmp_path):
    directory = tmp_path / "explicit-downstream"
    directory.mkdir()
    binary = directory / "ccec"
    binary.write_bytes(b"explicit downstream one")
    toolchain["env"]["BISHENG_INSTALL_PATH"] = str(directory)
    before = fingerprint(toolchain_module, toolchain)
    binary.write_bytes(b"explicit downstream two")
    assert fingerprint(toolchain_module, toolchain) != before



@pytest.mark.parametrize("role", ["compiler", "reader", "emitter", "optimizer", "libdevice", "template",
                                  "downstream", "explicit-downstream"])
def test_same_stat_same_size_rewrite_invalidates_key(toolchain_module, toolchain, tmp_path, monkeypatch, role):
    if role == "explicit-downstream":
        directory = tmp_path / "explicit-downstream"
        directory.mkdir()
        path = directory / "ccec"
        path.write_bytes(b"explicit downstream one")
        toolchain["env"]["BISHENG_INSTALL_PATH"] = str(directory)
    else:
        path = toolchain[role]

    # Simulate a filesystem where both writes have the same observable stat key.
    # The bytes still change on disk; no sleep or timestamp advance is required.
    def unchanged_timestamps(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, 0, 0

    monkeypatch.setattr(toolchain_module, "_stat_identity", unchanged_timestamps)
    original = path.read_bytes()
    replacement = bytes([original[0] ^ 1]) + original[1:]
    identity = toolchain_module._stat_identity(path.stat())
    before = fingerprint(toolchain_module, toolchain)
    path.write_bytes(replacement)
    assert path.read_bytes() != original and len(replacement) == len(original)
    assert toolchain_module._stat_identity(path.stat()) == identity
    assert fingerprint(toolchain_module, toolchain) != before
    path.write_bytes(original)
    assert toolchain_module._stat_identity(path.stat()) == identity
    assert fingerprint(toolchain_module, toolchain) == before
