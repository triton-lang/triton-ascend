// RUN: ! triton-opt %s --pre-check-dynamic-cv-pipeline-available

module {
  func.func @pre_check_available_rejects_without_matmul(
      %arg0: tensor<16xf32>,
      %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}
