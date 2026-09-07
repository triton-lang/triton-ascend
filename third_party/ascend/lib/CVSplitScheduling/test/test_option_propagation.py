from pathlib import Path
import re


TEST_DIR = Path(__file__).resolve().parent
CVSPLIT_LIB = TEST_DIR.parent
ASCEND_ROOT = TEST_DIR.parents[2]
PASSES_TD = ASCEND_ROOT / "include" / "CVSplitScheduling" / "Passes.td"
PASS_CPP = CVSPLIT_LIB / "CVSplitScheduling.cpp"


def test_programmatic_constructor_copies_every_declared_option():
    declared = set(
        re.findall(r'Option<"([A-Za-z0-9_]+)"', PASSES_TD.read_text())
    )

    cpp = PASS_CPP.read_text()
    constructor = cpp.split(
        "explicit CVSplitSchedulingPass(const CVSplitSchedulingOptions &options) {",
        1,
    )[1].split("  void runOnOperation() override", 1)[0]
    assignments = re.findall(
        r"this->([A-Za-z0-9_]+)\s*=\s*options[.]([A-Za-z0-9_]+);",
        constructor,
    )

    mismatched = [(target, source) for target, source in assignments if target != source]
    copied = {target for target, _ in assignments}

    assert not mismatched
    assert copied == declared, (
        f"constructor option mismatch: missing={sorted(declared - copied)}, "
        f"extra={sorted(copied - declared)}"
    )


if __name__ == "__main__":
    test_programmatic_constructor_copies_every_declared_option()
