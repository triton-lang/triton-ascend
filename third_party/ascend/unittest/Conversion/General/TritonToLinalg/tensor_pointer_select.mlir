// RUN: triton-opt --triton-to-linalg %s -verify-each 2>&1 | FileCheck %s

// tl.load(...).to(tl.pi32_t) + tl.where + tl.store to an i64 pointer builds an
// elementwise int_to_ptr -> select -> ptr_to_int chain over tensor-of-pointer
// values (the "*int32" variant of test_where).  The memory conversion stage
// has no tensor-of-pointer legalization, so the pre-memory canonicalization
// must fold the bit-identity chain back to a plain i64 select.  Before that
// fold this module failed with:
//   error: failed to legalize operation 'tt.int_to_ptr' that was explicitly
//   marked illegal

// CHECK-NOT: failed to legalize
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @tensor_pointer_select(%cond_ptr: !tt.ptr<i1>, %a_ptr: !tt.ptr<i64>, %b_ptr: !tt.ptr<i64>, %output_ptr: !tt.ptr<i64>, %n_elements: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<1024xi64>
    %cst_0 = arith.constant dense<0> : tensor<1024xi8>
    %c1024_i32 = arith.constant 1024 : i32
    %offsets = tt.get_program_id x : i32
    %offsets_1 = arith.muli %offsets, %c1024_i32 : i32
    %offsets_2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %offsets_3 = tt.splat %offsets_1 : i32 -> tensor<1024xi32>
    %offsets_4 = arith.addi %offsets_3, %offsets_2 : tensor<1024xi32>
    %mask = tt.splat %n_elements : i32 -> tensor<1024xi32>
    %mask_5 = arith.cmpi slt, %offsets_4, %mask : tensor<1024xi32>
    %decide = tt.splat %cond_ptr : !tt.ptr<i1> -> tensor<1024x!tt.ptr<i1>>
    %decide_6 = tt.addptr %decide, %offsets_4 : tensor<1024x!tt.ptr<i1>>, tensor<1024xi32>
    %decide_7 = tt.bitcast %decide_6 : tensor<1024x!tt.ptr<i1>> -> tensor<1024x!tt.ptr<i8>>
    %decide_8 = tt.load %decide_7, %mask_5, %cst_0 {was_bool_to_int8 = true} : tensor<1024x!tt.ptr<i8>>
    %a = tt.splat %a_ptr : !tt.ptr<i64> -> tensor<1024x!tt.ptr<i64>>
    %a_9 = tt.addptr %a, %offsets_4 : tensor<1024x!tt.ptr<i64>>, tensor<1024xi32>
    %a_10 = tt.load %a_9, %mask_5, %cst : tensor<1024x!tt.ptr<i64>>
    %a_11 = tt.int_to_ptr %a_10 : tensor<1024xi64> -> tensor<1024x!tt.ptr<i32>>
    %b = tt.splat %b_ptr : !tt.ptr<i64> -> tensor<1024x!tt.ptr<i64>>
    %b_12 = tt.addptr %b, %offsets_4 : tensor<1024x!tt.ptr<i64>>, tensor<1024xi32>
    %b_13 = tt.load %b_12, %mask_5, %cst : tensor<1024x!tt.ptr<i64>>
    %b_14 = tt.int_to_ptr %b_13 : tensor<1024xi64> -> tensor<1024x!tt.ptr<i32>>
    %output = arith.cmpi ne, %decide_8, %cst_0 : tensor<1024xi8>
    %output_15 = arith.select %output, %a_11, %b_14 : tensor<1024xi1>, tensor<1024x!tt.ptr<i32>>
    %0 = tt.splat %output_ptr : !tt.ptr<i64> -> tensor<1024x!tt.ptr<i64>>
    %1 = tt.addptr %0, %offsets_4 : tensor<1024x!tt.ptr<i64>>, tensor<1024xi32>
    %2 = tt.ptr_to_int %output_15 : tensor<1024x!tt.ptr<i32>> -> tensor<1024xi64>
    tt.store %1, %2, %mask_5 : tensor<1024x!tt.ptr<i64>>
    tt.return
  }
}

// The pointer-cast chain folds to a plain i64 select that survives lowering.
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<1024xi1>, tensor<1024xi64>
// CHECK-NOT: tt.int_to_ptr
// CHECK-NOT: tt.ptr_to_int
// CHECK-NOT: error:
