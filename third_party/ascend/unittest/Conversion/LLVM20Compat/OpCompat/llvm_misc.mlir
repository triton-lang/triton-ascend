// RUN: triton-opt %s | FileCheck %s

// CHECK-LABEL: func.func @llvm_misc() -> i32 {
// CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[UNDEFV:.*]] = llvm.mlir.undef : vector<2xi32>
// CHECK-NEXT: %[[V0:.*]] = llvm.insertelement %[[C0]], %[[UNDEFV]][%[[C0]] : i32] : vector<2xi32>
// CHECK-NEXT: %[[V1:.*]] = llvm.insertelement %[[C1]], %[[V0]][%[[C1]] : i32] : vector<2xi32>
// CHECK-NEXT: %[[E0:.*]] = llvm.extractelement %[[V1]][%[[C0]] : i32] : vector<2xi32>
// CHECK-NEXT: %[[UNDEFS:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32)>
// CHECK-NEXT: %[[SV0:.*]] = llvm.insertvalue %[[E0]], %[[UNDEFS]][0] : !llvm.struct<(i32, i32)>
// CHECK-NEXT: %[[SV1:.*]] = llvm.insertvalue %[[C1]], %[[SV0]][1] : !llvm.struct<(i32, i32)>
// CHECK-NEXT: %[[OUT:.*]] = llvm.extractvalue %[[SV1]][1] : !llvm.struct<(i32, i32)>
// CHECK-NEXT: return %[[OUT]] : i32
func.func @llvm_misc() -> i32 {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %undef_vec = llvm.mlir.undef : vector<2xi32>
  %vec0 = llvm.insertelement %c0, %undef_vec[%c0 : i32] : vector<2xi32>
  %vec1 = llvm.insertelement %c1, %vec0[%c1 : i32] : vector<2xi32>
  %elt0 = llvm.extractelement %vec1[%c0 : i32] : vector<2xi32>
  %undef_struct = llvm.mlir.undef : !llvm.struct<(i32, i32)>
  %sv0 = llvm.insertvalue %elt0, %undef_struct[0] : !llvm.struct<(i32, i32)>
  %sv1 = llvm.insertvalue %c1, %sv0[1] : !llvm.struct<(i32, i32)>
  %out = llvm.extractvalue %sv1[1] : !llvm.struct<(i32, i32)>
  return %out : i32
}

// CHECK-LABEL: func.func @builtin_unrealized_cast(%arg0: i32) {
// CHECK-NEXT: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : i32 to index
// CHECK-NEXT: return
func.func @builtin_unrealized_cast(%arg0: i32) {
  %0 = builtin.unrealized_conversion_cast %arg0 : i32 to index
  return
}
