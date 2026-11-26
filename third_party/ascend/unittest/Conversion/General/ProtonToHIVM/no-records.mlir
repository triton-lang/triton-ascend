// REQUIRES: proton
// RUN: triton-opt %s --assign-ascend-proton-scope-ids --lower-ascend-proton-to-hivm | FileCheck %s

module {
  func.func @unprofiled(%value: i32) {
    return
  }
}

// CHECK: module {
// CHECK-NEXT: func.func @unprofiled(%{{.*}}: i32) {
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
