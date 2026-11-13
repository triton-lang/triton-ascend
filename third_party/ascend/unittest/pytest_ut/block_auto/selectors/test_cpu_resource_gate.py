from pathlib import Path
import tempfile
import unittest

from cpu_resource_gate import PASS_MARKER, assess, compile_candidate, inspect_final_ub

TARGET = "Ascend950PR_9579"


def static_ir(offset=32, shape="16xf32", limit=2031616):
    return f'''module attributes {{hacc.target = #hacc.target<"{TARGET}">,
  dlti.target_system_spec = #dlti.target_system_spec<"UB_SIZE", {limit} : i32>}} {{
  func.func @kernel() {{
    %0 = llvm.mlir.constant({offset} : i64) : i64
    %1 = hivm.hir.pointer_cast(%0) : memref<{shape}, #hivm.address_space<ub>>
  }}
}}'''


class CpuResourceGateTests(unittest.TestCase):

    def test_final_offsets_include_multibuffer_placement(self):
        usage = inspect_final_ub(static_ir(offset=196608, shape="4096xf32"), TARGET)
        self.assertEqual(usage["address_extent_bits"], 212992 * 8)
        self.assertEqual(usage["capacity_bits"], 248 * 1024 * 8)

    def test_unknown_shapes_and_zero_placeholder_are_not_evidence(self):
        for ir in (static_ir(shape="?xf32"), 'required_ub_bits = 0',
                   static_ir().replace('llvm.mlir.constant(32 : i64)', 'arith.addi %arg0, %arg1')):
            with self.subTest(ir=ir), self.assertRaises(ValueError):
                inspect_final_ub(ir, TARGET)

    def test_helper_function_cannot_supply_kernel_offset(self):
        ir = static_ir().replace('%1 = hivm.hir.pointer_cast',
                                 '}\n  func.func @another() {\n    %1 = hivm.hir.pointer_cast')
        with self.assertRaises(ValueError):
            inspect_final_ub(ir, TARGET)

    def test_wrong_target_rejected(self):
        with self.assertRaises(ValueError):
            inspect_final_ub(static_ir(), "Ascend910B")

    def test_zero_ub_requires_pure_aic_not_merely_missing_allocations(self):
        ir = f'''module attributes {{hacc.target = #hacc.target<"{TARGET}">,
  dlti.target_system_spec = #dlti.target_system_spec<"UB_SIZE", 2031616 : i32>}} {{
  func.func private @helper() attributes {{hivm.func_core_type = #hivm.func_core_type<AIC>}}
  func.func @kernel() attributes {{hivm.func_core_type = #hivm.func_core_type<AIC>}} {{
    return
  }}
}}'''
        self.assertEqual(inspect_final_ub(ir, TARGET)['address_extent_bits'], 0)
        for changed in (ir.replace('<AIC>', '<AIV>',
                                   1), ir.replace('hivm.func_core_type = #hivm.func_core_type<AIC>', '', 1),
                        ir.replace('return', '%x = unknown.op : memref<8xf32, #hivm.address_space<ub>>')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                inspect_final_ub(changed, TARGET)

    def test_compile_error_cannot_accept_existing_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "kernel.o"
            artifact.write_bytes(b"\x7fELFfake-test-only")
            result = assess(1, "", "ub overflow, requires 2359296 bits while 2031616 bits available!", artifact,
                            Path(directory) / "missing.mlir", TARGET)
            self.assertEqual(result["status"], "UB_REJECTED")
            self.assertFalse(result["resource_checked"])
            self.assertEqual(result["overflow"][0]["required_bits"], 2359296)

    def test_artifact_allocation_pass_and_final_ir_are_all_required(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "kernel.o"
            final_ir = Path(directory) / "final.mlir"
            artifact.write_bytes(b"\x7fELFfake-test-only")
            final_ir.write_text(static_ir())
            valid = assess(0, "", PASS_MARKER, artifact, final_ir, TARGET)
            self.assertTrue(valid["resource_checked"])
            self.assertFalse(valid["numerical_correctness_verified"])
            self.assertFalse(valid["performance_verified"])
            self.assertFalse(assess(0, "", "", artifact, final_ir, TARGET)["resource_checked"])
            final_ir.unlink()
            self.assertFalse(assess(0, "", PASS_MARKER, artifact, final_ir, TARGET)["resource_checked"])
            final_ir.write_text(static_ir())
            artifact.unlink()
            self.assertFalse(assess(0, "", PASS_MARKER, artifact, final_ir, TARGET)["resource_checked"])

    def test_address_extent_larger_than_capacity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "kernel.o"
            final_ir = Path(directory) / "final.mlir"
            artifact.write_bytes(b"\x7fELFfake-test-only")
            final_ir.write_text(static_ir(offset=253952, shape="1xf32"))
            result = assess(0, "", PASS_MARKER, artifact, final_ir, TARGET)
            self.assertEqual(result["status"], "UB_REJECTED")
            self.assertFalse(result["resource_checked"])

    def test_changed_compiler_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiler = root / "compiler"
            compiler.write_text("this file is not executable")
            ir = root / "candidate.ttadapter"
            ir.write_text(static_ir())
            with self.assertRaisesRegex(ValueError, "compiler_sha256 changed"):
                compile_candidate(
                    {
                        "compiler": str(compiler), "ir": str(ir), "target": TARGET, "compiler_sha256": "old", "options":
                        []
                    }, root / "result")
            self.assertFalse((root / "result/kernel.o").exists())


if __name__ == "__main__":
    unittest.main()
