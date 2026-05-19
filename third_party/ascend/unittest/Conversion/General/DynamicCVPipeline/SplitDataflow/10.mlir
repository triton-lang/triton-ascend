// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

// CHECK-LABEL: func.func @tc10_empty_func
// CHECK-NEXT: return
module {
  func.func @tc10_empty_func() {
    return
  }
}



