// RUN: triton-opt %s --bubble-up-operation='enable-aggressive-mode=true' --verify-each | FileCheck %s

// ImplicitPermute's 1x32 -> 32x1 offset reshape must not hide the original
// indices from downstream L1 row-gather recognition.
// CHECK-LABEL: tt.func @singleton_offset_reshape(
// CHECK-SAME: %[[IDS:.*]]: tensor<32xi32>, %[[ROW:.*]]: index
// CHECK: %[[ID:.*]] = tensor.extract %[[IDS]][%[[ROW]]] {{.*}} : tensor<32xi32>
// CHECK: %[[OFFSET:.*]] = arith.muli %[[ID]], {{.*}} : i32
// CHECK: tt.return %[[OFFSET]] : i32
tt.func @singleton_offset_reshape(%ids: tensor<32xi32>, %row: index) -> i32 {
  %zero = arith.constant 0 : index
  %stride = arith.constant dense<256> : tensor<1x32xi32>
  %expanded = tt.expand_dims %ids {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %scaled = arith.muli %expanded, %stride : tensor<1x32xi32>
  %rows = tt.reshape %scaled allow_reorder : tensor<1x32xi32> -> tensor<32x1xi32>
  %offset = tensor.extract %rows[%row, %zero] {DiscreteMemAccess} : tensor<32x1xi32>
  tt.return %offset : i32
}

// Test the extract_slice/broadcast path used to build each contiguous K row.
// CHECK-LABEL: tt.func @singleton_offset_row_slice(
// CHECK-SAME: %[[IDS:.*]]: tensor<32xi32>, %[[ROW:.*]]: index
// CHECK: %[[ID:.*]] = tensor.extract %[[IDS]][{{.*}}] {{.*}} : tensor<32xi32>
// CHECK: %[[OFFSET:.*]] = arith.muli %[[ID]], {{.*}} : i32
// CHECK: %[[SPLAT:.*]] = tt.splat %[[OFFSET]] : i32 -> tensor<1x128xi32>
// CHECK: tt.return %[[SPLAT]] : tensor<1x128xi32>
tt.func @singleton_offset_row_slice(%ids: tensor<32xi32>, %row: index) -> tensor<1x128xi32> {
  %stride = arith.constant dense<256> : tensor<1x32xi32>
  %expanded = tt.expand_dims %ids {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %scaled = arith.muli %expanded, %stride : tensor<1x32xi32>
  %rows = tt.reshape %scaled : tensor<1x32xi32> -> tensor<32x1xi32>
  %tile = tt.broadcast %rows : tensor<32x1xi32> -> tensor<32x128xi32>
  %slice = tensor.extract_slice %tile[%row, 0] [1, 128] [1, 1] {DiscreteMemAccess} : tensor<32x128xi32> to tensor<1x128xi32>
  tt.return %slice : tensor<1x128xi32>
}

// Multiple data axes require a real coordinate conversion: keep the reshape.
// CHECK-LABEL: tt.func @keep_general_reshape(
// CHECK: %[[RESHAPED:.*]] = tt.reshape
// CHECK: tensor.extract %[[RESHAPED]]
tt.func @keep_general_reshape(%ids: tensor<2x16xi32>, %row: index) -> i32 {
  %zero = arith.constant 0 : index
  %rows = tt.reshape %ids : tensor<2x16xi32> -> tensor<32x1xi32>
  %offset = tensor.extract %rows[%row, %zero] {DiscreteMemAccess} : tensor<32x1xi32>
  tt.return %offset : i32
}

// A reshape outside discrete address extraction is unaffected.
// CHECK-LABEL: tt.func @keep_non_discrete_reshape(
// CHECK: %[[RESHAPED:.*]] = tt.reshape
// CHECK: tensor.extract %[[RESHAPED]]
tt.func @keep_non_discrete_reshape(%ids: tensor<1x32xi32>, %row: index) -> i32 {
  %zero = arith.constant 0 : index
  %rows = tt.reshape %ids : tensor<1x32xi32> -> tensor<32x1xi32>
  %offset = tensor.extract %rows[%row, %zero] : tensor<32x1xi32>
  tt.return %offset : i32
}
