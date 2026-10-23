// RUN: triton-opt --triton-to-linalg="global-kernel=false named-ops=true" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @neg_index_from_arg
// CHECK-DAG: %[[CNEG24:.*]] = arith.constant -24 : index
// CHECK: %[[BASE:.*]] = memref.extract_aligned_pointer_as_index %[[ARG:[^ ]+]] : memref<?xf32> -> index
// CHECK: %[[ADVANCED:.*]] = arith.addi %[[BASE]], %[[CNEG24]] : index
// CHECK: %[[ADDR:.*]] = arith.index_cast %[[ADVANCED]] : index to i64
// CHECK: %[[PTR:.*]] = hivm.hir.pointer_cast(%{{.*}}) {{.*}}: memref<?xf32>
// CHECK: memref.reinterpret_cast %[[PTR]] to offset: [0], sizes: [12], strides: [1]
// CHECK-NOT: memref.reinterpret_cast {{.*}} offset: [-{{[0-9]+}}]
// CHECK: memref.subview {{.*}}[6] [6] [1]
module {
  tt.func public @neg_index_from_arg(
      %in_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst_other = arith.constant dense<0.000000e+00> : tensor<12xf32>
    %cst_neg = arith.constant dense<-6> : tensor<12xi32>
    %cst_six = arith.constant dense<6> : tensor<12xi32>
    %range = tt.make_range {end = 12 : i32, start = 0 : i32} : tensor<12xi32>
    %mask = arith.cmpi sge, %range, %cst_six : tensor<12xi32>
    %offs = arith.addi %range, %cst_neg : tensor<12xi32>
    %in_splat = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<12x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_splat, %offs : tensor<12x!tt.ptr<f32>>, tensor<12xi32>
    %loaded = tt.load %in_ptrs, %mask, %cst_other : tensor<12x!tt.ptr<f32>>
    %out_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<12x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_splat, %range : tensor<12x!tt.ptr<f32>>, tensor<12xi32>
    tt.store %out_ptrs, %loaded : tensor<12x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @neg_index_from_int_to_ptr
// CHECK-DAG: %[[CNEG24:.*]] = arith.constant -24 : i64
// CHECK: %[[ADVANCED:.*]] = arith.addi %[[ADDR:[^ ]+]], %[[CNEG24]] : i64
// CHECK: %[[PTR:.*]] = hivm.hir.pointer_cast(%{{.*}}) {{.*}}: memref<?xf32>
// CHECK: memref.reinterpret_cast %[[PTR]] to offset: [0], sizes: [12], strides: [1]
// CHECK-NOT: memref.extract_aligned_pointer_as_index
// CHECK-NOT: memref.reinterpret_cast {{.*}} offset: [-{{[0-9]+}}]
// CHECK: memref.subview {{.*}}[6] [6] [1]
module {
  tt.func public @neg_index_from_int_to_ptr(
      %addr: i64,
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst_other = arith.constant dense<0.000000e+00> : tensor<12xf32>
    %cst_neg = arith.constant dense<-6> : tensor<12xi32>
    %cst_six = arith.constant dense<6> : tensor<12xi32>
    %range = tt.make_range {end = 12 : i32, start = 0 : i32} : tensor<12xi32>
    %mask = arith.cmpi sge, %range, %cst_six : tensor<12xi32>
    %offs = arith.addi %range, %cst_neg : tensor<12xi32>
    %in_ptr = tt.int_to_ptr %addr : i64 -> !tt.ptr<f32>
    %in_splat = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<12x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_splat, %offs : tensor<12x!tt.ptr<f32>>, tensor<12xi32>
    %loaded = tt.load %in_ptrs, %mask, %cst_other : tensor<12x!tt.ptr<f32>>
    %out_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<12x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_splat, %range : tensor<12x!tt.ptr<f32>>, tensor<12xi32>
    tt.store %out_ptrs, %loaded : tensor<12x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @neg_index_zero_offset
// CHECK: memref.reinterpret_cast %[[IN:[^ ]+]] to offset: [0], sizes: [12], strides: [1] {{.*}} memref<?xf32> to memref<12xf32
// CHECK-NOT: memref.extract_aligned_pointer_as_index
module {
  tt.func public @neg_index_zero_offset(
      %in_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst_other = arith.constant dense<0.000000e+00> : tensor<12xf32>
    %range = tt.make_range {end = 12 : i32, start = 0 : i32} : tensor<12xi32>
    %true = arith.constant dense<true> : tensor<12xi1>
    %in_splat = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<12x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_splat, %range : tensor<12x!tt.ptr<f32>>, tensor<12xi32>
    %loaded = tt.load %in_ptrs, %true, %cst_other : tensor<12x!tt.ptr<f32>>
    %out_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<12x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_splat, %range : tensor<12x!tt.ptr<f32>>, tensor<12xi32>
    tt.store %out_ptrs, %loaded : tensor<12x!tt.ptr<f32>>
    tt.return
  }
}
