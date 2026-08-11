// RUN: not triton-opt --triton-to-linalg="named-ops=True" %s 2>&1 | FileCheck %s

// CHECK: different-width pointer bitcast requires byte-addressable scalar integer or floating-point pointee types
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @reject_non_byte_addressable_pointee(
      %src: !tt.ptr<i4> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<i16> {tt.divisibility = 16 : i32})
      attributes {noinline = false} {
    %wide_ptr = tt.bitcast %src : !tt.ptr<i4> -> !tt.ptr<i16>
    %value = tt.load %wide_ptr : !tt.ptr<i16>
    tt.store %dst, %value : !tt.ptr<i16>
    tt.return
  }
}
