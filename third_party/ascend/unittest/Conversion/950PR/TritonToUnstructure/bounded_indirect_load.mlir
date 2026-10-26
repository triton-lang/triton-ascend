// RUN: triton-opt %s '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' --mlir-print-op-generic | FileCheck %s --check-prefix=VOLATILE
// RUN: triton-opt %s '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' | FileCheck %s --check-prefix=CHUNK
// RUN: triton-opt %s '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' --bubble-up-operation --canonicalize --cse | FileCheck %s --check-prefix=BUBBLE

// A dynamic modulo in the columns must retain its exact address semantics.
// Split only the generated indirect load, keeping the original result tile.
// CHUNK-LABEL: tt.func @large_modulo_load
// CHUNK: scf.for
// CHUNK: tensor.extract_slice {{.*}}[{{.*}}, 0] [32, 256] [1, 1]
// CHUNK: ascend.indirect_load {{.*}}-> tensor<32x256xi8>
// CHUNK: tensor.insert_slice {{.*}}[{{.*}}, 0] [32, 256] [1, 1]
// CHUNK: } {ExtractedLoadOrStore, hivm.parallel_loop}
// CHUNK: tt.return {{.*}} : tensor<128x256xi8>
// BUBBLE-LABEL: tt.func @large_modulo_load
// BUBBLE-NOT: tensor<128x256xi64>
// BUBBLE: scf.for
// BUBBLE-NOT: tensor<128x256xi64>
// BUBBLE: ascend.indirect_load {{.*}}-> tensor<32x256xi8>
// BUBBLE-NOT: tensor<128x256xi64>
// BUBBLE: } {ExtractedLoadOrStore, hivm.parallel_loop}
// BUBBLE: tt.return
tt.func @large_modulo_load(%src: !tt.ptr<i8>, %n: i32, %stride: i32) -> tensor<128x256xi8> {
  %rows = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %cols = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %nvec = tt.splat %n : i32 -> tensor<256xi32>
  %wrapped = arith.remsi %cols, %nvec : tensor<256xi32>
  %row2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  %strides = tt.splat %stride : i32 -> tensor<128x1xi32>
  %scaled = arith.muli %row2d, %strides : tensor<128x1xi32>
  %col2d = tt.expand_dims %wrapped {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
  %brows = tt.broadcast %scaled : tensor<128x1xi32> -> tensor<128x256xi32>
  %bcols = tt.broadcast %col2d : tensor<1x256xi32> -> tensor<128x256xi32>
  %offset = arith.addi %brows, %bcols : tensor<128x256xi32>
  %base = tt.splat %src : !tt.ptr<i8> -> tensor<128x256x!tt.ptr<i8>>
  %ptr = tt.addptr %base, %offset : tensor<128x256x!tt.ptr<i8>>, tensor<128x256xi32>
  %value = tt.load %ptr : tensor<128x256x!tt.ptr<i8>>
  tt.return %value : tensor<128x256xi8>
}

// CHUNK-LABEL: tt.func @threshold_modulo_load
// CHUNK: scf.for
// CHUNK: tensor.extract_slice {{.*}}[{{.*}}, 0] [32, 256] [1, 1]
// CHUNK: ascend.indirect_load {{.*}}-> tensor<32x256xi8>
// CHUNK: tensor.insert_slice {{.*}}[{{.*}}, 0] [32, 256] [1, 1]
// CHUNK: } {ExtractedLoadOrStore, hivm.parallel_loop}
// CHUNK: tt.return {{.*}} : tensor<64x256xi8>
// BUBBLE-LABEL: tt.func @threshold_modulo_load
// BUBBLE-NOT: tensor<64x256xi64>
// BUBBLE: scf.for
// BUBBLE-NOT: tensor<64x256xi64>
// BUBBLE: ascend.indirect_load {{.*}}-> tensor<32x256xi8>
// BUBBLE-NOT: tensor<64x256xi64>
// BUBBLE: } {ExtractedLoadOrStore, hivm.parallel_loop}
// BUBBLE: tt.return
tt.func @threshold_modulo_load(%src: !tt.ptr<i8>, %n: i32, %stride: i32) -> tensor<64x256xi8> {
  %rows = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %cols = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %nvec = tt.splat %n : i32 -> tensor<256xi32>
  %wrapped = arith.remsi %cols, %nvec : tensor<256xi32>
  %row2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %strides = tt.splat %stride : i32 -> tensor<64x1xi32>
  %scaled = arith.muli %row2d, %strides : tensor<64x1xi32>
  %col2d = tt.expand_dims %wrapped {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
  %brows = tt.broadcast %scaled : tensor<64x1xi32> -> tensor<64x256xi32>
  %bcols = tt.broadcast %col2d : tensor<1x256xi32> -> tensor<64x256xi32>
  %offset = arith.addi %brows, %bcols : tensor<64x256xi32>
  %base = tt.splat %src : !tt.ptr<i8> -> tensor<64x256x!tt.ptr<i8>>
  %ptr = tt.addptr %base, %offset : tensor<64x256x!tt.ptr<i8>>, tensor<64x256xi32>
  %value = tt.load %ptr : tensor<64x256x!tt.ptr<i8>>
  tt.return %value : tensor<64x256xi8>
}

// CHUNK-LABEL: tt.func @masked_modulo_load
// CHUNK: %[[OTHER:.*]] = arith.constant dense<-7> : tensor<32x256xi8>
// CHUNK: scf.for
// CHUNK: tensor.extract_slice {{.*}}[{{.*}}, 0] [32, 256] [1, 1]
// CHUNK: %[[MASK:.*]] = tensor.extract_slice {{.*}} : tensor<128x256xi1> to tensor<32x256xi1>
// CHUNK: ascend.indirect_load {{.*}}, %[[MASK]] : tensor<32x256xi1>, %[[OTHER]] : tensor<32x256xi8> {isVolatile = false} -> tensor<32x256xi8>
// CHUNK: tensor.insert_slice {{.*}}[{{.*}}, 0] [32, 256] [1, 1]
// CHUNK: } {ExtractedLoadOrStore, hivm.parallel_loop}
// CHUNK: tt.return {{.*}} : tensor<128x256xi8>
// BUBBLE-LABEL: tt.func @masked_modulo_load
// BUBBLE-NOT: tensor<128x256xi64>
// BUBBLE: scf.for
// BUBBLE-NOT: tensor<128x256xi64>
// BUBBLE: ascend.indirect_load {{.*}}-> tensor<32x256xi8>
// BUBBLE-NOT: tensor<128x256xi64>
// BUBBLE: } {ExtractedLoadOrStore, hivm.parallel_loop}
// BUBBLE: tt.return
tt.func @masked_modulo_load(%src: !tt.ptr<i8>, %n: i32, %stride: i32) -> tensor<128x256xi8> {
  %rows = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %cols = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %nvec = tt.splat %n : i32 -> tensor<256xi32>
  %wrapped = arith.remsi %cols, %nvec : tensor<256xi32>
  %row2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  %strides = tt.splat %stride : i32 -> tensor<128x1xi32>
  %scaled = arith.muli %row2d, %strides : tensor<128x1xi32>
  %col2d = tt.expand_dims %wrapped {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
  %brows = tt.broadcast %scaled : tensor<128x1xi32> -> tensor<128x256xi32>
  %bcols = tt.broadcast %col2d : tensor<1x256xi32> -> tensor<128x256xi32>
  %offset = arith.addi %brows, %bcols : tensor<128x256xi32>
  %base = tt.splat %src : !tt.ptr<i8> -> tensor<128x256x!tt.ptr<i8>>
  %ptr = tt.addptr %base, %offset : tensor<128x256x!tt.ptr<i8>>, tensor<128x256xi32>
  %limit = tt.splat %n : i32 -> tensor<128x256xi32>
  %mask = arith.cmpi slt, %offset, %limit : tensor<128x256xi32>
  %other = arith.constant dense<-7> : tensor<128x256xi8>
  %value = tt.load %ptr, %mask, %other : tensor<128x256x!tt.ptr<i8>>
  tt.return %value : tensor<128x256xi8>
}

// CHUNK-LABEL: tt.func @small_modulo_load
// CHUNK-NOT: scf.for
// CHUNK: ascend.indirect_load {{.*}}-> tensor<32x256xi8>
// CHUNK-NOT: scf.for
// CHUNK: tt.return
// BUBBLE-LABEL: tt.func @small_modulo_load
// BUBBLE-NOT: scf.for
// BUBBLE: ascend.indirect_load {{.*}}-> tensor<32x256xi8>
// BUBBLE-NOT: scf.for
// BUBBLE: tt.return
tt.func @small_modulo_load(%src: !tt.ptr<i8>, %n: i32, %stride: i32) -> tensor<32x256xi8> {
  %rows = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %cols = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %nvec = tt.splat %n : i32 -> tensor<256xi32>
  %wrapped = arith.remsi %cols, %nvec : tensor<256xi32>
  %row2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %strides = tt.splat %stride : i32 -> tensor<32x1xi32>
  %scaled = arith.muli %row2d, %strides : tensor<32x1xi32>
  %col2d = tt.expand_dims %wrapped {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
  %brows = tt.broadcast %scaled : tensor<32x1xi32> -> tensor<32x256xi32>
  %bcols = tt.broadcast %col2d : tensor<1x256xi32> -> tensor<32x256xi32>
  %offset = arith.addi %brows, %bcols : tensor<32x256xi32>
  %base = tt.splat %src : !tt.ptr<i8> -> tensor<32x256x!tt.ptr<i8>>
  %ptr = tt.addptr %base, %offset : tensor<32x256x!tt.ptr<i8>>, tensor<32x256xi32>
  %value = tt.load %ptr : tensor<32x256x!tt.ptr<i8>>
  tt.return %value : tensor<32x256xi8>
}

// VOLATILE-LABEL: sym_name = "volatile_modulo_load"
// VOLATILE-NOT: "scf.for"
// VOLATILE: "ascend.indirect_load"{{.*}}isVolatile = true
// CHUNK-LABEL: tt.func @volatile_modulo_load
// CHUNK-NOT: scf.for
// CHUNK: ascend.indirect_load %{{[^ ]+}} : <i8>, %{{[^ ]+}} : tensor<128x256xi64> -> tensor<128x256xi8>
// CHUNK-NOT: scf.for
// CHUNK: tt.return
// BUBBLE-LABEL: tt.func @volatile_modulo_load
// BUBBLE-NOT: scf.for
// BUBBLE: ascend.indirect_load %{{[^ ]+}} : <i8>, %{{[^ ]+}} : tensor<128x256xi64> -> tensor<128x256xi8>
// BUBBLE-NOT: scf.for
// BUBBLE: tt.return
tt.func @volatile_modulo_load(%src: !tt.ptr<i8>, %n: i32, %stride: i32) -> tensor<128x256xi8> {
  %rows = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %cols = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %nvec = tt.splat %n : i32 -> tensor<256xi32>
  %wrapped = arith.remsi %cols, %nvec : tensor<256xi32>
  %row2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  %strides = tt.splat %stride : i32 -> tensor<128x1xi32>
  %scaled = arith.muli %row2d, %strides : tensor<128x1xi32>
  %col2d = tt.expand_dims %wrapped {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
  %brows = tt.broadcast %scaled : tensor<128x1xi32> -> tensor<128x256xi32>
  %bcols = tt.broadcast %col2d : tensor<1x256xi32> -> tensor<128x256xi32>
  %offset = arith.addi %brows, %bcols : tensor<128x256xi32>
  %base = tt.splat %src : !tt.ptr<i8> -> tensor<128x256x!tt.ptr<i8>>
  %ptr = tt.addptr %base, %offset : tensor<128x256x!tt.ptr<i8>>, tensor<128x256xi32>
  %value = tt.load %ptr {isVolatile = true} : tensor<128x256x!tt.ptr<i8>>
  tt.return %value : tensor<128x256xi8>
}
