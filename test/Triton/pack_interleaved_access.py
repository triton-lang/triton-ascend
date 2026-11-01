"""Safety regressions for interleaved-field packing; invoked by the MLIR test."""
import pathlib
import subprocess
import sys
import tempfile

tool, fixture = sys.argv[1:]
source = pathlib.Path(fixture).read_text()
loads = [line for line in source.splitlines() if " = tt.load " in line]
assert len(loads) == 9
volatile = source.replace(loads[1], loads[1].replace(" : tensor", " {isVolatile = true} : tensor"))
mask_line = loads[3]
parts = mask_line.split(", ")
parts[1] = "%different_mask"
different = source.replace(mask_line, ", ".join(parts))
different = different.replace(loads[0],
                              "    %different_mask = arith.constant dense<true> : tensor<1024xi1>\n" + loads[0])
tests = {
    "insufficient_alignment": (source.replace("tt.divisibility = 16", "tt.divisibility = 2"), 7, 5),
    "missing_alignment": (source.replace(" {tt.divisibility = 16 : i32}", ""), 7, 5),
    "row_misalignment": (source.replace("constant 24576 : i64", "constant 24577 : i64"), 7, 5),
    "conflicting_field": (source.replace("tt.store %20, %30", "tt.store %19, %30"), 7, 5),
    "memory_consumer": (source.replace("tt.store %20, %30", "tt.store %20, %25"), 7, 5),
    "safe": (source, 4, 5),
    "noalias": (source.replace("tt.divisibility = 16 : i32}", "tt.divisibility = 16 : i32, tt.noalias}"), 4, 5),
    "false_noalias": (source.replace("tt.divisibility = 16 : i32}",
                                     "tt.divisibility = 16 : i32, tt.noalias = false}"), 4, 5),
    "volatile": (volatile, 6, 5),
    "different_mask": (different, 6, 5),
    "non_power_stride": (source.replace("dense<4> : tensor<1024xi32>", "dense<3> : tensor<1024xi32>"), 9, 5),
}
with tempfile.TemporaryDirectory() as tmp:
    for name, (text, load_count, store_count) in tests.items():
        path = pathlib.Path(tmp) / (name + ".mlir")
        path.write_text(text)
        command = [tool, str(path), "--triton-pack-interleaved-access", "--canonicalize", "--cse", "--verify-each"]
        output = subprocess.check_output(command, text=True)
        assert output.count("tt.load ") == load_count, (name, output.count("tt.load "), load_count)
        assert output.count("tt.store ") == store_count, (name, output.count("tt.store "), store_count)
        assert output.count("tt.func public") == 1, name
        assert "scf.if" not in output and "tt.ptr_to_int" not in output, name
        if name in ("safe", "noalias", "false_noalias"):
            assert "arith.cmpi ne" in output and "arith.andi" in output, name
        if name == "volatile": assert "isVolatile = true" in output
        path.write_text(output)
        assert subprocess.check_output(command, text=True) == output, name
        print("PASS", name)
