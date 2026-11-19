import pytest


def test_unknown_checkout_prefix_stays_invalid(checker):
    assert not checker._glob_exists("wrong/requirements.txt")
    assert checker._glob_exists("pr/requirements.txt")


@pytest.mark.parametrize("mutation", ["duplicate", "invalid-checksum", "unsafe-file", "missing-arch"])
def test_toolchain_rejects_invalid_inputs(checker, mutation):
    path = checker.DEPS_DIR / "bisheng-version.txt"
    text = path.read_text()
    if mutation == "duplicate":
        text += text.splitlines()[0] + "\n"
    elif mutation == "invalid-checksum":
        text = "\n".join(line + "invalid" if "_sha256:" in line else line for line in text.splitlines())
    elif mutation == "unsafe-file":
        text = text.replace("x86_64: ", "x86_64: ../")
    else:
        text = "\n".join(line for line in text.splitlines() if not line.startswith("aarch64"))
    path.write_text(text)
    checker.check_toolchain()
    assert checker.errors


@pytest.mark.parametrize("version", ["v3.11.3", "main", "v3.11.3\nv3.12.0"])
def test_audit_validates_the_upstream_json_text_pin(checker, version):
    (checker.DEPS_DIR / "json-version.txt").write_text(version)
    checker.check_toolchain()
    assert bool(checker.errors) is (version != "v3.11.3")


@pytest.mark.parametrize("suffix", ["run", "tar.gz"])
def test_bisheng_metadata_retains_supported_package_formats(checker, suffix):
    path = checker.DEPS_DIR / "bisheng-version.txt"
    text = path.read_text().replace("ascendnpu-ir_2.0.0_linux-x86.run", "fixture." + suffix)
    path.write_text(text)
    checker.check_toolchain()
    assert not checker.errors
