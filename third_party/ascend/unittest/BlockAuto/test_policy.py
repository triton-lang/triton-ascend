# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CPU checks for the production C++ resource gate and CV/VV selection."""
import copy
import unittest

from triton.backends.ascend.block_auto import _native, candidate_plan
from test_analysis import candidate as vector_body


def contract(grid=(224, 1, 1)):
    return dict(grid=list(grid), arguments={"x": dict(kind="tensor", dtype="fp32")})


def item(factor=1, *, mode="simd", dtype="16xf32", offset=0):
    metadata = dict(target={"arch": "Ascend950PR_9579"}, mix_mode="aiv", parallel_mode=mode, name="kernel",
                    auto_blockify_enabled=True, shared_mem_dynamic_size=216 * 1024,
                    block_auto_evidence=dict(command=["--simt-vf-dynamic-size=216"]))
    ir = '''module attributes {hacc.target = #hacc.target<"Ascend950">,
dlti.target_system_spec = #dlti.target_system_spec<"NPU" = #hacc.target_device_spec<
#dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>,
#dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 524288 : i32>,
#dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>>>} {
func.func @kernel(%sync: !llvm.ptr<1>, %workspace: !llvm.ptr<1>, %x: !llvm.ptr<1>, %gx: i32, %gy: i32, %gz: i32) {
 %offset = llvm.mlir.constant(OFFSET : i64) : i64
 %buffer = hivm.hir.pointer_cast(%offset) : memref<DTYPE, #hivm.address_space<ub>>
 return
}
}
'''.replace("OFFSET", str(offset)).replace("DTYPE", dtype)
    source = vector_body().replace("@kernel()", "@kernel(ptr %sync, ptr %workspace, ptr %x, i32 %gx, i32 %gy, i32 %gz)")
    return dict(factor=factor, axis=-1 if factor == 1 else 0, metadata=metadata, final_ir=ir, llvm=source,
                optimized_llvm=source, files={})


def prepare(items, launch=None):
    launch = launch or contract()
    states = _native("prepare", launch, candidates=items)["candidates"]
    return [dict(value, **state) for value, state in zip(items, states)]


class TestNativeResources(unittest.TestCase):

    def test_aligned_extent_and_shared_offsets(self):
        value = item(dtype="2xi32", offset=253920)
        value["final_ir"] = value["final_ir"].replace(
            " return", " %alias = hivm.hir.pointer_cast(%offset) : memref<2xi32, #hivm.address_space<ub>>\n return")
        state = prepare([value])[0]
        self.assertTrue(state["resource_checked"])
        # Aliases do not double-count, but the final small buffer still rounds
        # up to 32 bytes. Moving it one byte past the boundary must fail.
        value["final_ir"] = value["final_ir"].replace("constant(253920 :", "constant(253921 :")
        self.assertFalse(prepare([value])[0]["resource_checked"])

    def test_ub_capacity_boundary(self):
        for offset, accepted in ((253920, True), (253952, False)):
            state = prepare([item(dtype="1xf32", offset=offset)])[0]
            self.assertEqual(state["resource_checked"], accepted)

    def test_dynamic_layout_is_not_zero_memory(self):
        state = prepare([item(dtype="?xf32")])[0]
        self.assertFalse(state["resource_checked"])
        self.assertEqual(state["status"], "RESOURCE_UNCONFIRMED")

    def test_i1_excess_remains_unconfirmed(self):
        state = prepare([item(dtype="262144xi1")])[0]
        self.assertEqual(state["status"], "RESOURCE_UNCONFIRMED")
        self.assertIn("actual overflow unconfirmed", state["error"])

    def test_mix_uses_effective_runtime_capacity(self):
        value = item(mode="mix_simd_simt", dtype="8xf32", offset=216 * 1024 - 32)
        self.assertTrue(prepare([value])[0]["resource_checked"])
        value["metadata"]["shared_mem_dynamic_size"] = 208 * 1024
        state = prepare([value])[0]
        self.assertFalse(state["resource_checked"])
        self.assertIn("exceeds capacity", state["error"])
        value["final_ir"] = value["final_ir"].replace("constant(221152 :", "constant(212960 :")
        self.assertTrue(prepare([value])[0]["resource_checked"])
        value["metadata"]["shared_mem_dynamic_size"] = 248 * 1024
        self.assertFalse(prepare([value])[0]["resource_checked"])

    def test_mix_does_not_require_compiler_help_or_command_reconstruction(self):
        value = item(mode="mix_simd_simt")
        del value["metadata"]["block_auto_evidence"]
        self.assertTrue(prepare([value])[0]["resource_checked"])
        del value["metadata"]["shared_mem_dynamic_size"]
        self.assertFalse(prepare([value])[0]["resource_checked"])

    def test_wrong_architecture_falls_back_before_optimizer(self):
        value = item()
        value["metadata"]["target"]["arch"] = "Ascend910"
        states = prepare([value, item(2)])
        self.assertFalse(any(s["needs_llvm"] for s in states))
        result = _native("rank", contract(), candidates=states)
        self.assertEqual(result["factor"], 1)
        self.assertEqual(result["reason"], "BASELINE_RESOURCE_UNCONFIRMED")


class TestNativeSelection(unittest.TestCase):

    def test_invalid_merge_parameters_are_rejected_before_grid_arithmetic(self):
        for mix_mode in ("aiv", "mix"):
            for factor, axis in ((0, 0), (3, 0), (2, -1), (4, 3)):
                with self.subTest(mix_mode=mix_mode, factor=factor, axis=axis):
                    value = item(factor)
                    value["axis"] = axis
                    value["metadata"]["mix_mode"] = mix_mode
                    state = prepare([value])[0]
                    self.assertTrue(state["resource_checked"])
                    self.assertFalse(state["needs_llvm"])
                    self.assertNotIn("config", state)
                    self.assertIn("invalid merge factor or axis", state["error"])

    def test_argument_positions_follow_compiler_abi_not_ssa_names(self):
        value = item()
        value["metadata"]["program_grid_transforms"] = dict(hidden_extent_axes=[0])
        value["final_ir"] = value["final_ir"].replace("%x: !llvm.ptr<1>",
                                                      "%renamed0: i32, %renamed1: !llvm.ptr<1>, %renamed2: i32")
        launch = dict(grid=[224, 1, 1],
                      arguments={"z": dict(kind="scalar", value=7), "a": dict(kind="tensor", dtype="fp32")})
        state = prepare([value], launch)[0]
        self.assertTrue(state["resource_checked"])
        self.assertEqual(state["config"]["arguments"], [None, None, 7, None, 224, 224, 1, 1])
        launch["argument_order"] = ["z", "z"]
        self.assertFalse(prepare([value], launch)[0]["resource_checked"])

    def test_two_spatial_axes_and_small_grid(self):

        def probe(axis, factor):
            return dict(compile_status="REWRITTEN", selected_axis=axis, requested_factor=factor,
                        cv_mode="spatial_concat_v1", cv_lowering="spatial_m_v1" if axis == 0 else "spatial_n_v1")

        result = candidate_plan(contract((8, 8, 1)), probe=probe)
        self.assertEqual(result["reason"], "CV_M4_N4")
        self.assertEqual([(v["axis"], v["factor"]) for v in result["candidates"]], [(0, 4), (1, 4)])
        self.assertEqual(candidate_plan(contract((1, 1, 1)), probe=probe)["candidates"], [])

    def test_inconsistent_axis_roles_cannot_enter_shortlist(self):

        def probe(axis, factor):
            return dict(compile_status="REWRITTEN", selected_axis=axis, requested_factor=factor,
                        cv_mode="spatial_concat_v1", cv_lowering="spatial_m_v1" if factor == 2 else "spatial_n_v1")

        result = candidate_plan(contract((8, 1, 1)), probe=probe)
        self.assertEqual(result["reason"], "CV_AXIS_ROLE_DEPENDS_ON_FACTOR")
        self.assertFalse(result["candidates"])

    def test_compiler_sequential_lowering_preserves_grid(self):
        value = item(2)
        value["metadata"]["block_batching"] = dict(vv_lowering="sequential_iterations_v1", lane_width=2)
        launch = contract((64, 1, 1))
        states = prepare([item(), value], launch)
        self.assertTrue(states[1]["needs_llvm"])
        self.assertEqual(states[1]["config"]["grid"], [64, 1, 1])
        self.assertEqual(prepare([item(), item(2)], launch)[1]["status"], "RUNTIME_REJECTED")
        value["metadata"]["row_coalescing_applied"] = True
        state = prepare([item(), value], launch)[1]
        self.assertFalse(state["needs_llvm"])
        self.assertIn("additional grid transform", state["error"])

    def test_incomplete_candidate_does_not_silently_enter_ranking(self):
        states = prepare([item(), item(2)])
        del states[1]["optimized_llvm"]
        result = _native("rank", contract(), candidates=states)
        self.assertFalse(result["model_complete"])
        self.assertEqual(result["factor"], 1)

    def test_terminal_exclusion_keeps_valid_baseline_scored(self):
        states = prepare([item(), dict(factor=2, axis=0, status="COMPILE_REJECTED")])
        result = _native("rank", contract(), candidates=states)
        self.assertTrue(result["model_complete"], result)
        self.assertEqual(result["factor"], 1)

    def test_deterministic_ties_prefer_baseline(self):
        launch = contract((224, 2, 1))
        states = prepare([item(), item(2)], launch)
        result = _native("rank", launch, candidates=states)
        self.assertTrue(result["model_complete"], result)
        self.assertEqual(result["factor"], 1)
        reverse = _native("rank", launch, candidates=list(reversed(copy.deepcopy(states))))
        self.assertEqual(result, reverse)


if __name__ == "__main__":
    unittest.main()
