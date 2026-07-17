// RUN: ! triton-opt %s --pre-check-dynamic-cv-pipeline-available

module {
  func.func @pre_check_available_rejects_existing_scope(
      %arg0: memref<16x16xf16>,
      %arg1: memref<16x16xf16>,
      %arg2: memref<16x16xf32>) {
    scope.scope : () -> () {
      linalg.matmul
        ins(%arg0, %arg1 : memref<16x16xf16>, memref<16x16xf16>)
        outs(%arg2 : memref<16x16xf32>)
      scope.return
    }
    return
  }
}
