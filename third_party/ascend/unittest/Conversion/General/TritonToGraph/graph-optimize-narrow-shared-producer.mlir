// RUN: triton-opt %s -graph-optimize='compile-on-910-95=true' --verify-each | FileCheck %s
// RUN: triton-opt %s -graph-optimize='compile-on-910-95=false' --verify-each | FileCheck %s --check-prefix=OTHER
// RUN: triton-opt %s -graph-optimize='compile-on-910-95=true' --verify-each | triton-opt -graph-optimize='compile-on-910-95=true' --verify-each | FileCheck %s

// Reuse a shared wide producer instead of duplicating its XOR at i32.
// CHECK-LABEL: tt.func @shared_wide_xor(
// CHECK: %[[SHARED:.*]] = arith.xori {{.*}} : tensor<128xi64>
// CHECK-NOT: arith.xori
// CHECK: %[[LOW:.*]] = arith.trunci %[[SHARED]] : tensor<128xi64> to tensor<128xi32>
// CHECK-NOT: arith.xori
// CHECK: arith.andi %[[LOW]], {{.*}} : tensor<128xi32>
// CHECK-NOT: arith.xori
// CHECK: tt.return %[[SHARED]],
// OTHER-LABEL: tt.func @shared_wide_xor(
// OTHER: arith.xori {{.*}} : tensor<128xi64>
// OTHER: arith.andi {{.*}} : tensor<128xi64>
tt.func @shared_wide_xor(%a: tensor<128xi64>, %b: tensor<128xi64>) -> (tensor<128xi64>, tensor<128xi64>) {
  %mask = arith.constant dense<255> : tensor<128xi64>
  %shared = arith.xori %a, %b : tensor<128xi64>
  %masked = arith.andi %shared, %mask : tensor<128xi64>
  tt.return %shared, %masked : tensor<128xi64>, tensor<128xi64>
}

// Recurse through the private XOR, then stop at its shared upstream producer.
// CHECK-LABEL: tt.func @deep_shared_xor(
// CHECK: %[[DEEP_SHARED:.*]] = arith.xori {{.*}} : tensor<128xi64>
// CHECK-NOT: arith.xori
// CHECK: %[[LOW_SHARED:.*]] = arith.trunci %[[DEEP_SHARED]] : tensor<128xi64> to tensor<128xi32>
// CHECK: %[[LOW_C:.*]] = arith.trunci {{.*}} : tensor<128xi64> to tensor<128xi32>
// CHECK: %[[PRIVATE:.*]] = arith.xori %[[LOW_SHARED]], %[[LOW_C]] : tensor<128xi32>
// CHECK-NOT: arith.xori
// CHECK: arith.andi %[[PRIVATE]], {{.*}} : tensor<128xi32>
// CHECK-NOT: arith.xori
// CHECK: tt.return %[[DEEP_SHARED]],
// OTHER-LABEL: tt.func @deep_shared_xor(
// OTHER-COUNT-2: arith.xori {{.*}} : tensor<128xi64>
// OTHER: arith.andi {{.*}} : tensor<128xi64>
tt.func @deep_shared_xor(%a: tensor<128xi64>, %b: tensor<128xi64>, %c: tensor<128xi64>) -> (tensor<128xi64>, tensor<128xi64>) {
  %mask = arith.constant dense<255> : tensor<128xi64>
  %shared = arith.xori %a, %b : tensor<128xi64>
  %private = arith.xori %shared, %c : tensor<128xi64>
  %masked = arith.andi %private, %mask : tensor<128xi64>
  tt.return %shared, %masked : tensor<128xi64>, tensor<128xi64>
}

// The replacement root itself may have multiple uses. Truncating it instead
// would create a cycle when its uses are replaced with the new extension.
// CHECK-LABEL: tt.func @multi_use_root(
// CHECK-NOT: arith.andi {{.*}} : tensor<128xi64>
// CHECK: %[[ROOT:.*]] = arith.andi {{.*}} : tensor<128xi32>
// CHECK: %[[EXTENDED_ROOT:.*]] = arith.extui %[[ROOT]] : tensor<128xi32> to tensor<128xi64>
// CHECK: tt.return %[[EXTENDED_ROOT]], %[[EXTENDED_ROOT]]
// OTHER-LABEL: tt.func @multi_use_root(
// OTHER: %[[WIDE_ROOT:.*]] = arith.andi {{.*}} : tensor<128xi64>
// OTHER: tt.return %[[WIDE_ROOT]], %[[WIDE_ROOT]]
tt.func @multi_use_root(%a: tensor<128xi64>) -> (tensor<128xi64>, tensor<128xi64>) {
  %mask = arith.constant dense<255> : tensor<128xi64>
  %masked = arith.andi %a, %mask : tensor<128xi64>
  tt.return %masked, %masked : tensor<128xi64>, tensor<128xi64>
}

// Keep narrowing useful when the original wide producer can be removed.
// CHECK-LABEL: tt.func @private_xor(
// CHECK-NOT: arith.xori {{.*}} : tensor<128xi64>
// CHECK: %[[PRIVATE_LOW:.*]] = arith.xori {{.*}} : tensor<128xi32>
// CHECK-NOT: arith.xori {{.*}} : tensor<128xi64>
// CHECK: arith.andi %[[PRIVATE_LOW]], {{.*}} : tensor<128xi32>
// CHECK-NOT: arith.xori {{.*}} : tensor<128xi64>
// CHECK: tt.return
// OTHER-LABEL: tt.func @private_xor(
// OTHER: arith.xori {{.*}} : tensor<128xi64>
// OTHER: arith.andi {{.*}} : tensor<128xi64>
tt.func @private_xor(%a: tensor<128xi64>, %b: tensor<128xi64>) -> tensor<128xi64> {
  %mask = arith.constant dense<255> : tensor<128xi64>
  %private = arith.xori %a, %b : tensor<128xi64>
  %masked = arith.andi %private, %mask : tensor<128xi64>
  tt.return %masked : tensor<128xi64>
}

// A depth-zero input is not necessarily the root being replaced. Reuse the
// shared comparison RHS while preserving the scalar upper-bit guard.
// CHECK-LABEL: tt.func @shared_comparison_inputs(
// CHECK: %[[LIMIT:.*]] = tt.splat {{.*}} : i64 -> tensor<128xi64>
// CHECK: arith.cmpi ule, {{.*}} : i64
// CHECK: %[[LOW_LIMIT:.*]] = arith.trunci %[[LIMIT]] : tensor<128xi64> to tensor<128xi32>
// CHECK: arith.cmpi eq, {{.*}}, %[[LOW_LIMIT]] : tensor<128xi32>
// CHECK: arith.andi {{.*}} : tensor<128xi1>
// CHECK: tt.return
// OTHER-LABEL: tt.func @shared_comparison_inputs(
// OTHER: arith.cmpi eq, {{.*}} : tensor<128xi64>
tt.func @shared_comparison_inputs(%keys: tensor<128xi32>, %threshold: i64) -> (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) {
  %wide_keys = arith.extui %keys : tensor<128xi32> to tensor<128xi64>
  %wide_limit = tt.splat %threshold : i64 -> tensor<128xi64>
  %equal = arith.cmpi eq, %wide_keys, %wide_limit : tensor<128xi64>
  tt.return %equal, %wide_keys, %wide_limit : tensor<128xi1>, tensor<128xi64>, tensor<128xi64>
}

// Bypassing an extension remains safe when the extension has other wide users.
// CHECK-LABEL: tt.func @shared_select_extensions(
// CHECK-NOT: arith.trunci
// CHECK: %[[SELECTED:.*]] = arith.select {{.*}} : tensor<128xi1>, tensor<128xi32>
// CHECK-NOT: arith.trunci
// CHECK: arith.extui %[[SELECTED]] : tensor<128xi32> to tensor<128xi64>
// CHECK-NOT: arith.trunci
// CHECK: tt.return
// OTHER-LABEL: tt.func @shared_select_extensions(
// OTHER: arith.select {{.*}} : tensor<128xi1>, tensor<128xi64>
tt.func @shared_select_extensions(%condition: tensor<128xi1>, %a: tensor<128xi32>, %b: tensor<128xi32>) -> (tensor<128xi64>, tensor<128xi64>, tensor<128xi64>) {
  %wide_a = arith.extui %a : tensor<128xi32> to tensor<128xi64>
  %wide_b = arith.extui %b : tensor<128xi32> to tensor<128xi64>
  %selected = arith.select %condition, %wide_a, %wide_b : tensor<128xi1>, tensor<128xi64>
  tt.return %selected, %wide_a, %wide_b : tensor<128xi64>, tensor<128xi64>, tensor<128xi64>
}

// The original radix byte extraction still narrows across constants/extensions.
// CHECK-LABEL: tt.func @radix_byte_extract(
// CHECK-NOT: arith.shrui {{.*}} : tensor<128xi64>
// CHECK: %[[SHIFTED:.*]] = arith.shrui {{.*}} : tensor<128xi32>
// CHECK: arith.andi %[[SHIFTED]], {{.*}} : tensor<128xi32>
// CHECK: tt.return
// OTHER-LABEL: tt.func @radix_byte_extract(
// OTHER: arith.shrui {{.*}} : tensor<128xi64>
tt.func @radix_byte_extract(%keys: tensor<128xi64>) -> tensor<128xi32> {
  %u32mask = arith.constant dense<4294967295> : tensor<128xi64>
  %shift = arith.constant dense<24> : tensor<128xi64>
  %byte_mask = arith.constant dense<255> : tensor<128xi64>
  %bounded = arith.andi %keys, %u32mask : tensor<128xi64>
  %shifted = arith.shrui %bounded, %shift : tensor<128xi64>
  %byte = arith.andi %shifted, %byte_mask : tensor<128xi64>
  %indices = arith.trunci %byte : tensor<128xi64> to tensor<128xi32>
  tt.return %indices : tensor<128xi32>
}
