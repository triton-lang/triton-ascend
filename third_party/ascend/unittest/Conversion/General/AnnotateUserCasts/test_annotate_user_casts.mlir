// RUN: triton-opt --annotate-user-casts %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @test_fptofp_fp32_to_fp16
  func.func @test_fptofp_fp32_to_fp16(%arg0: tensor<8xf32>) {
    // CHECK: arith.fptofp {{.*}} {cast.source = "user"} : tensor<8xf32> -> tensor<8xf16>
    %0 = arith.fptofp %arg0 : tensor<8xf32> to tensor<8xf16>
    func.return
  }

  // CHECK-LABEL: func.func @test_fptofp_fp16_to_fp32
  func.func @test_fptofp_fp16_to_fp32(%arg0: tensor<8xf16>) {
    // CHECK: arith.fptofp {{.*}} {cast.source = "user"} : tensor<8xf16> -> tensor<8xf32>
    %0 = arith.fptofp %arg0 : tensor<8xf16> to tensor<8xf32>
    func.return
  }

  // CHECK-LABEL: func.func @test_fptofp_fp32_to_bf16
  func.func @test_fptofp_fp32_to_bf16(%arg0: tensor<8xf32>) {
    // CHECK: arith.fptofp {{.*}} {cast.source = "user"} : tensor<8xf32> -> tensor<8xbf16>
    %0 = arith.fptofp %arg0 : tensor<8xf32> to tensor<8xbf16>
    func.return
  }

  // CHECK-LABEL: func.func @test_fptofp_bf16_to_fp32
  func.func @test_fptofp_bf16_to_fp32(%arg0: tensor<8xbf16>) {
    // CHECK: arith.fptofp {{.*}} {cast.source = "user"} : tensor<8xbf16> -> tensor<8xf32>
    %0 = arith.fptofp %arg0 : tensor<8xbf16> to tensor<8xf32>
    func.return
  }

  // CHECK-LABEL: func.func @test_fptofp_fp16_to_bf16
  func.func @test_fptofp_fp16_to_bf16(%arg0: tensor<8xf16>) {
    // CHECK: arith.fptofp {{.*}} {cast.source = "user"} : tensor<8xf16> -> tensor<8xbf16>
    %0 = arith.fptofp %arg0 : tensor<8xf16> to tensor<8xbf16>
    func.return
  }

  // CHECK-LABEL: func.func @test_fptofp_bf16_to_fp16
  func.func @test_fptofp_bf16_to_fp16(%arg0: tensor<8xbf16>) {
    // CHECK: arith.fptofp {{.*}} {cast.source = "user"} : tensor<8xbf16> -> tensor<8xf16>
    %0 = arith.fptofp %arg0 : tensor<8xbf16> to tensor<8xf16>
    func.return
  }

  // CHECK-LABEL: func.func @test_existing_annotation_preserved
  func.func @test_existing_annotation_preserved(%arg0: tensor<8xf32>) {
    // CHECK: arith.fptofp {{.*}} {cast.source = "compiler"} : tensor<8xf32> -> tensor<8xf16>
    %0 = arith.fptofp %arg0 {cast.source = "compiler"} : tensor<8xf32> to tensor<8xf16>
    func.return
  }

  // CHECK-LABEL: func.func @test_non_cast_op_not_annotated
  func.func @test_non_cast_op_not_annotated(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) {
    // CHECK-NOT: {cast.source}
    // CHECK: arith.addf
    %0 = arith.addf %arg0, %arg1 : tensor<8xf32>
    func.return
  }

  // CHECK-LABEL: func.func @test_chain_of_casts
  func.func @test_chain_of_casts(%arg0: tensor<8xf32>) {
    // CHECK: arith.fptofp {{.*}} {cast.source = "user"} : tensor<8xf32> -> tensor<8xbf16>
    // CHECK: arith.fptofp {{.*}} {cast.source = "user"} : tensor<8xbf16> -> tensor<8xf16>
    %0 = arith.fptofp %arg0 : tensor<8xf32> to tensor<8xbf16>
    %1 = arith.fptofp %0 : tensor<8xbf16> to tensor<8xf16>
    func.return
  }
}