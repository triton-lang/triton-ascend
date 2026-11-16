# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Candidate LLVM analysis and CV/VV resource ranking; CPU-only."""
import json
import unittest
import copy
import pytest
from triton._C.libtriton import ascend, ir
from triton.backends.ascend.block_auto import _native, candidate_plan
from block_batching_test_utils import assign_npu  # noqa: F401 - pytest fixtures

pytestmark = pytest.mark.backend('native')

# Candidate code analysis


def analyze(source, *, mixed=False):
    request = dict(mode="vv", modules={"aiv": source}, name="kernel", arguments=[], grid=[1, 1, 1], cores=56, blocks=1,
                   auto_blockify=True)
    if mixed:
        request["mixed"] = dict(simt_transaction_bytes=64, simt_transition_cycles=32, simt_instruction_cycles=1)
    item = dict(factor=1, axis=-1, vv=True, resource_checked=True, needs_llvm=True, config=request,
                optimized_llvm=source)
    result = json.loads(ascend.select_block_candidates(json.dumps(dict(action="rank", contract={}, candidates=[item]))))
    if not result["model_complete"]:
        raise RuntimeError(result["candidates"]["unmerged"]["error"])
    return result["candidates"]["unmerged"]


def vector_body(helper="pow_fp32", operations=1, iterations=1):
    body = "\n".join(f"%v{i} = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %v{i-1}, "
                     f"<64 x float> %v{i-1}, <256 x i1> %mask)" for i in range(1, operations + 1))
    return f"""
declare <256 x i1> @llvm.hivm.pset.b32(i32)
declare <64 x float> @llvm.hivm.vdups.z.v64f32(float, <256 x i1>, i32)
declare <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)
define void @kernel() {{
  %trip = alloca i32
  store i32 {iterations}, ptr %trip
  call void @{helper}(ptr %trip)
  ret void
}}
define internal void @{helper}(ptr %trip) {{
  call ptc_simdvf void @vector_body(ptr %trip)
  ret void
}}
define internal ptc_simdvf void @vector_body(ptr %trip) {{
entry:
  %count = load i32, ptr %trip
  %mask = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %v0 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.0, <256 x i1> %mask, i32 1)
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  {body}
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, %count
  br i1 %more, label %loop, label %exit
exit:
  ret void
}}
"""


class TestCandidateHelpers(unittest.TestCase):

    def test_helper_cost_comes_from_the_candidate_body(self):
        short = analyze(vector_body(operations=1))["score"]
        long = analyze(vector_body(operations=16))["score"]
        self.assertGreater(long, short)

    def test_helper_names_do_not_select_a_special_cost_model(self):
        self.assertEqual(analyze(vector_body()), analyze(vector_body(helper="arbitrary_helper")))

    def test_nested_helper_preserves_scalar_loop_inputs(self):
        short = analyze(vector_body(operations=8, iterations=1))["score"]
        long = analyze(vector_body(operations=8, iterations=4))["score"]
        self.assertGreater(long, short)

    def test_imported_constants_do_not_add_issued_instructions(self):
        source = vector_body(operations=8, iterations=4).replace(
            "store i32 4, ptr %trip",
            "%count = extractvalue { i32, i32 } { i32 4, i32 0 }, 0\n  store i32 %count, ptr %trip")
        # The original extractvalue costs one step; imported aggregate builders do not.
        self.assertEqual(analyze(source)["score"], 73.5)

    def test_freeze_preserves_the_scalar_loop_bound(self):
        source = vector_body(operations=8,
                             iterations=4).replace("%count = load i32, ptr %trip",
                                                   "%loaded = load i32, ptr %trip\n  %count = freeze i32 %loaded")
        self.assertEqual(analyze(source)["score"], 73)

    def test_loop_carried_vector_values_keep_their_dependencies(self):
        source = vector_body(operations=8, iterations=4)
        self.assertEqual(analyze(source)["score"], 72.5)
        source = source.replace("%v1 = call", "%carry = phi <64 x float> [%v0, %entry], [%v8, %loop]\n  %v1 = call")
        source = source.replace("(<64 x float> %v0, <64 x float> %v0,", "(<64 x float> %carry, <64 x float> %carry,")
        self.assertEqual(analyze(source)["score"], 180)

    def test_missing_helper_definition_is_not_assigned_zero_cost(self):
        source = "declare void @pow_fp32()\ndefine void @kernel() { call void @pow_fp32()\nret void }"
        with self.assertRaisesRegex(RuntimeError, "unmodelled call: pow_fp32"):
            analyze(source)

    def test_recursive_helpers_remain_bounded(self):
        source = "define void @kernel() { call void @helper()\nret void }"
        source += "\ndefine void @helper() { call void @helper()\nret void }"
        with self.assertRaisesRegex(RuntimeError, "helper recursion limit"):
            analyze(source)

    def test_large_launch_is_rejected_before_tracing_each_block(self):
        request = dict(mode="vv", modules={"aiv": vector_body()}, name="kernel", arguments=[], grid=[10001, 1, 1],
                       cores=56, blocks=10001, auto_blockify=False)
        item = dict(factor=1, axis=-1, vv=True, resource_checked=True, needs_llvm=True, config=request,
                    optimized_llvm=vector_body())
        result = json.loads(
            ascend.select_block_candidates(json.dumps(dict(action="rank", contract={}, candidates=[item]))))
        self.assertFalse(result["model_complete"])
        self.assertIn("launch outside bounded analysis", result["candidates"]["unmerged"]["error"])


def simt_candidate():
    return '''
declare void @llvm.hivm.store.vfsimt.info(i64)
declare i32 @llvm.hivm.get.TID.X()
define void @kernel() {
  call void @cumsum_1d_float_dim0(i32 32)
  ret void
}
define void @cumsum_1d_float_dim0(i32 %n) {
  call void @llvm.hivm.store.vfsimt.info(i64 4295032864)
  call simt_entry void @worker(i32 %n)
  ret void
}
define simt_entry void @worker(i32 %n) {
  %tid = call i32 @llvm.hivm.get.TID.X()
  %address = inttoptr i64 0 to ptr addrspace(6)
  store i32 %tid, ptr addrspace(6) %address
  ret void
}
'''


class TestSimtWrapper(unittest.TestCase):

    def test_dispatch_and_thread_work_are_both_counted(self):
        source = simt_candidate()
        score = analyze(source, mixed=True)["score"]
        doubled = source.replace("4295032864", "4295032896")
        self.assertGreater(analyze(doubled, mixed=True)["score"], score)

    def test_multiple_launches_remain_unsupported(self):
        source = simt_candidate().replace(
            "call simt_entry void @worker(i32 %n)",
            "call simt_entry void @worker(i32 %n)\n  call simt_entry void @worker(i32 %n)")
        with self.assertRaisesRegex(RuntimeError, "one static SIMT launch"):
            analyze(source, mixed=True)

    def test_wrapper_control_and_memory_are_not_silently_skipped(self):
        for prefix in ("br label %launch\nlaunch:", "store i32 1, ptr null"):
            with self.subTest(prefix=prefix):
                source = simt_candidate().replace("call void @llvm.hivm.store.vfsimt.info",
                                                  prefix + "\n  call void @llvm.hivm.store.vfsimt.info")
                with self.assertRaisesRegex(RuntimeError, "unresolved control or memory"):
                    analyze(source, mixed=True)

    def test_launch_size_still_requires_a_static_descriptor(self):
        source = simt_candidate().replace("i32 %n) {\n  call void",
                                          "i32 %n) {\n  %dynamic = zext i32 %n to i64\n  call void")
        source = source.replace("i64 4295032864", "i64 %dynamic")
        with self.assertRaisesRegex(RuntimeError, "dynamic SIMT dimensions"):
            analyze(source, mixed=True)


# Resource gates and ranking


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

    def test_empty_helper_scratch_does_not_hide_real_ub_overflow(self):
        for offset, accepted in ((253920, True), (253952, False)):
            value = item(dtype="1xf32", offset=offset)
            value["final_ir"] = value["final_ir"].replace(
                " return", " %zero = llvm.mlir.constant(0 : i64) : i64\n"
                " %scratch = hivm.hir.pointer_cast(%zero) : memref<0xf32, #hivm.address_space<ub>>\n return")
            self.assertEqual(prepare([value])[0]["resource_checked"], accepted)
        self.assertFalse(prepare([item(dtype="0xf32", offset=-1)])[0]["resource_checked"])

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

    def test_a3_uses_compiled_resources_and_core_count(self):
        value = item()
        value["metadata"]["target"]["arch"] = "Ascend910_9391"
        value["final_ir"] = value["final_ir"].replace("Ascend950", "Ascend910_9391").replace("56 : i32",
                                                                                             "48 : i32").replace(
                                                                                                 "28 : i32", "24 : i32")
        state = prepare([value])[0]
        self.assertTrue(state["resource_checked"])
        self.assertEqual(state["config"]["cores"], 48)
        self.assertEqual(state["config"]["target_arch"], "Ascend910_9391")
        prefix = '%ffts: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, '
        value['final_ir'] = value['final_ir'].replace('@kernel(', '@kernel(' + prefix)
        state = prepare([value])[0]
        self.assertTrue(state['resource_checked'])
        self.assertEqual(state['config']['arguments'], [None, None, None, None, 224, 1, 1])
        value['final_ir'] = value['final_ir'].replace(prefix, '%unproved: i64, ')
        self.assertFalse(prepare([value])[0]['resource_checked'])


class TestNativeSelection(unittest.TestCase):

    def test_a3_selection_preserves_full_core_occupancy_and_checks_ub(self):

        def a3(factor, **kwargs):
            value = item(factor, **kwargs)
            value['metadata']['target']['arch'] = 'Ascend910_9391'
            value['final_ir'] = value['final_ir'].replace('Ascend950', 'Ascend910_9391')
            value.pop('llvm')
            value.pop('optimized_llvm')
            return value

        for grid, offset, expected in ((224, 0, 2), (64, 0, 1), (225, 0, 1), (224, 253952, 1)):
            launch = contract((grid, 1, 1))
            states = prepare([a3(1), a3(2, offset=offset)], launch)
            self.assertFalse(any(s['needs_llvm'] for s in states))
            result = _native('rank', launch, candidates=states)
            self.assertEqual(result['model'], 'jit-a3-resource-v1')
            self.assertEqual(result['factor'], expected)

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
