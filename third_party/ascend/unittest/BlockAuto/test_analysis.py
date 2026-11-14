# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CPU-only tests for analysis of helpers in candidate LLVM."""
import json
import unittest

from triton._C.libtriton import ascend


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


def candidate(helper="pow_fp32", operations=1, iterations=1):
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
        short = analyze(candidate(operations=1))["score"]
        long = analyze(candidate(operations=16))["score"]
        self.assertGreater(long, short)

    def test_helper_names_do_not_select_a_special_cost_model(self):
        self.assertEqual(analyze(candidate()), analyze(candidate(helper="arbitrary_helper")))

    def test_nested_helper_preserves_scalar_loop_inputs(self):
        short = analyze(candidate(operations=8, iterations=1))["score"]
        long = analyze(candidate(operations=8, iterations=4))["score"]
        self.assertGreater(long, short)

    def test_imported_constants_do_not_add_issued_instructions(self):
        source = candidate(operations=8, iterations=4).replace(
            "store i32 4, ptr %trip",
            "%count = extractvalue { i32, i32 } { i32 4, i32 0 }, 0\n  store i32 %count, ptr %trip")
        # The original extractvalue costs one step; imported aggregate builders do not.
        self.assertEqual(analyze(source)["score"], 73.5)

    def test_freeze_preserves_the_scalar_loop_bound(self):
        source = candidate(operations=8,
                           iterations=4).replace("%count = load i32, ptr %trip",
                                                 "%loaded = load i32, ptr %trip\n  %count = freeze i32 %loaded")
        self.assertEqual(analyze(source)["score"], 73)

    def test_loop_carried_vector_values_keep_their_dependencies(self):
        source = candidate(operations=8, iterations=4)
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


if __name__ == "__main__":
    unittest.main()
