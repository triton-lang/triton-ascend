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


def fp32_fields(n, width, *, alignment=True, conflict=False, mismatch=False, volatile=False, consumer=False,
                scalar="f32"):
    """Interleaved reads separated by stores to another aligned row base.

    No noalias contract: pointer bases may alias. The field residues, not
    pointer identity, must justify moving each read to the next one.
    """
    attr = " {tt.divisibility = 32 : i32}" if alignment else ""
    ty = f"tensor<{n}x{scalar}>"
    ptr = f"tensor<{n}x!tt.ptr<{scalar}>>"
    idx = f"tensor<{n}xi32>"
    mask = f"tensor<{n}xi1>"
    lines = [
        f"module {{ tt.func public @fields(%p: !tt.ptr<{scalar}>{attr}, "
        f"%q: !tt.ptr<{scalar}>{attr}, %out: !tt.ptr<{scalar}>) {{",
        f"%i = tt.make_range {{start = 0 : i32, end = {n} : i32}} : {idx}",
        f"%stride = arith.constant dense<{width}> : {idx}", f"%end = arith.constant dense<{n - 3}> : {idx}",
        f"%mask = arith.cmpi slt, %i, %end : {idx}", f"%other_mask = arith.constant dense<true> : {mask}",
        f"%zero = arith.constant dense<0.0> : {ty}", f"%offset = arith.muli %i, %stride : {idx}",
        f"%ps = tt.splat %p : !tt.ptr<{scalar}> -> {ptr}", f"%qs = tt.splat %q : !tt.ptr<{scalar}> -> {ptr}",
        f"%pr = tt.addptr %ps, %offset : {ptr}, {idx}", f"%qr = tt.addptr %qs, %offset : {ptr}, {idx}"
    ]
    for j in range(width):
        lines += [
            f"%c{j} = arith.constant dense<{j}> : {idx}", f"%p{j} = tt.addptr %pr, %c{j} : {ptr}, {idx}",
            f"%q{j} = tt.addptr %qr, %c{j} : {ptr}, {idx}"
        ]
    for j in range(width):
        if j:
            field = 0 if conflict and j == 1 else j
            value = "%v0" if consumer and j == 1 else "%zero"
            lines.append(f"tt.store %q{field}, {value}, %mask : {ptr}")
        m = "%other_mask" if mismatch and j == 1 else "%mask"
        attrs = " {isVolatile = true}" if volatile and j == 1 else ""
        lines.append(f"%v{j} = tt.load %p{j}, {m}, %zero{attrs} : {ptr}")
        acc = "%zero" if j == 0 else f"%a{j - 1}"
        lines.append(f"%a{j} = arith.addf {acc}, %v{j} : {ty}")
    lines += [
        f"%os = tt.splat %out : !tt.ptr<{scalar}> -> {ptr}", f"%op = tt.addptr %os, %i : {ptr}, {idx}",
        f"tt.store %op, %a{width - 1}, %mask : {ptr}", "tt.return } }"
    ]
    return "\n".join(lines)


pair_tests = {f"fp32_n{n}_w{w}": (fp32_fields(n, w), w // 2, w) for n in (32, 64, 128) for w in (2, 4, 8)}
pair_tests.update({
    "fp32_missing_alignment": (fp32_fields(32, 4, alignment=False), 4, 4),
    "fp32_conflicting_store": (fp32_fields(32, 4, conflict=True), 3, 4),
    "fp32_different_mask": (fp32_fields(32, 4, mismatch=True), 3, 4),
    "fp32_volatile": (fp32_fields(32, 4, volatile=True), 3, 4),
    "fp32_memory_consumer": (fp32_fields(32, 4, consumer=True), 3, 4),
    "fp32_non_power_stride": (fp32_fields(32, 3), 3, 3),
    "unsupported_f64": (fp32_fields(32, 4, scalar="f64"), 4, 4),
})
with tempfile.TemporaryDirectory() as tmp:
    for name, (text, loads, stores) in pair_tests.items():
        path = pathlib.Path(tmp) / (name + ".mlir")
        path.write_text(text)
        command = [tool, str(path), "--triton-pack-interleaved-access", "--canonicalize", "--cse", "--verify-each"]
        output = subprocess.check_output(command, text=True)
        assert output.count("tt.load ") == loads, (name, output)
        assert output.count("tt.store ") == stores, (name, output)
        assert "x4xf32>" not in output and "x8xf32>" not in output, name
        assert "tt.noalias" not in output and "scf.if" not in output, name
        if name == "fp32_volatile":
            assert "isVolatile = true" in output
        path.write_text(output)
        assert subprocess.check_output(command, text=True) == output, name
        print("PASS", name)
