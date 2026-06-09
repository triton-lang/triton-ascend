module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_cube_control_flow_input_chain(%arg0: i64, %arg1: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c0_i64 = arith.constant 0 : i64
    %c-1_i64 = arith.constant -1 : i64
    %cst_0 = arith.constant 0.000000e+00 : f32
    scope.scope : () -> () {
      %cond_i1 = arith.cmpi sge, %arg0, %arg1 {ssbuffer.block_id = 20 : i32} : i64
      %empty_i1 = tensor.empty() {ssbuffer.block_id = 20 : i32} : tensor<1xi1>
      %inserted_cond = tensor.insert %cond_i1 into %empty_i1[%c0] {ssbuffer.block_id = 20 : i32} : tensor<1xi1>
      %empty_i64_t = tensor.empty() {ssbuffer.block_id = 20 : i32} : tensor<1xi64>
      %inserted_true = tensor.insert %arg0 into %empty_i64_t[%c0] {ssbuffer.block_id = 20 : i32} : tensor<1xi64>
      %empty_i64_f = tensor.empty() {ssbuffer.block_id = 20 : i32} : tensor<1xi64>
      %inserted_false = tensor.insert %c-1_i64 into %empty_i64_f[%c0] {ssbuffer.block_id = 20 : i32} : tensor<1xi64>
      %select = arith.select %inserted_cond, %inserted_true, %inserted_false {ssbuffer.block_id = 20 : i32} : tensor<1xi1>, tensor<1xi64>
      %extracted = tensor.extract %select[%c0] {ssbuffer.block_id = 7 : i32} : tensor<1xi64>
      %cmp = arith.cmpi ne, %c0_i64, %extracted {ssbuffer.block_id = 7 : i32} : i64
      scf.if %cmp{
      } {ssbuffer.block_id = 23 : i32} 
      scf.for %iv = %c0 to %c10 step %c1 iter_args(%idx = %c0) -> (index) {
        %200 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<32x32xf32>
        %300 = linalg.fill {ssbuffer.block_id = 1 : i32} ins(%cst_0 : f32) outs(%200 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %3 = math.exp %300 {ssbuffer.block_id = 1 : i32} : tensor<32x32xf32>
        scf.yield %idx : index
      } {ssbuffer.block_id = 24 : i32, ssbuffer.main_loop = 0 : i32}

      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %cond_i1 = arith.cmpi sge, %arg0, %arg1 {ssbuffer.block_id = 20 : i32} : i64
      %empty_i1 = tensor.empty() {ssbuffer.block_id = 20 : i32} : tensor<1xi1>
      %inserted_cond = tensor.insert %cond_i1 into %empty_i1[%c0] {ssbuffer.block_id = 20 : i32} : tensor<1xi1>
      %empty_i64_t = tensor.empty() {ssbuffer.block_id = 20 : i32} : tensor<1xi64>
      %inserted_true = tensor.insert %arg0 into %empty_i64_t[%c0] {ssbuffer.block_id = 20 : i32} : tensor<1xi64>
      %empty_i64_f = tensor.empty() {ssbuffer.block_id = 20 : i32} : tensor<1xi64>
      %inserted_false = tensor.insert %c-1_i64 into %empty_i64_f[%c0] {ssbuffer.block_id = 20 : i32} : tensor<1xi64>
      %select = arith.select %inserted_cond, %inserted_true, %inserted_false {ssbuffer.block_id = 20 : i32} : tensor<1xi1>, tensor<1xi64>
      %extracted = tensor.extract %select[%c0] {ssbuffer.block_id = 7 : i32} : tensor<1xi64>
      %cmp = arith.cmpi ne, %c0_i64, %extracted {ssbuffer.block_id = 7 : i32} : i64
      scf.if %cmp{
      } {ssbuffer.block_id = 23 : i32} 

      scf.for %iv = %c0 to %c10 step %c1 iter_args(%idx = %c0) -> (index) {
        scf.yield %idx : index
      } {ssbuffer.block_id = 24 : i32, ssbuffer.main_loop = 0 : i32}

      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}

    return
  }
}