# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CPU-only tests for analysis of helpers in candidate LLVM."""
import json
import unittest

from triton._C.libtriton import ascend


def analyze(source):
    request = dict(mode="vv", modules={"aiv": source}, raw_llvm=source, name="kernel", arguments=[], grid=[1, 1, 1],
                   cores=56, blocks=1, auto_blockify=True)
    return json.loads(ascend.analyze_block_candidate(json.dumps(request)))


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

    def test_missing_helper_definition_is_not_assigned_zero_cost(self):
        source = "declare void @pow_fp32()\ndefine void @kernel() { call void @pow_fp32()\nret void }"
        with self.assertRaisesRegex(RuntimeError, "unmodelled call: pow_fp32"):
            analyze(source)

    def test_recursive_helpers_remain_bounded(self):
        source = "define void @kernel() { call void @helper()\nret void }"
        source += "\ndefine void @helper() { call void @helper()\nret void }"
        with self.assertRaisesRegex(RuntimeError, "helper recursion limit"):
            analyze(source)


if __name__ == "__main__":
    unittest.main()
