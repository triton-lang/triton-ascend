// RUN: triton-opt %s --vector-address-lookahead --verify-each | FileCheck %s
// RUN: triton-opt %s --vector-address-lookahead --vector-address-lookahead --verify-each | FileCheck %s
// CHECK-LABEL: func.func @positive
// CHECK: scf.if
// CHECK: arith.mulf {{.*}} : tensor<1xf32>
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (f32, index, index, f32, f32)
// CHECK: memref.copy
// CHECK: memref.copy
// CHECK: scf.if
// CHECK: arith.mulf {{.*}} : tensor<1xf32>
// CHECK: } else {
// CHECK: arith.mulf {{.*}} : f32
// CHECK: scf.yield {{.*}} : f32, index, index, f32, f32
// CHECK-LABEL: func.func @address_uses_accumulator
// CHECK-NOT: scf.if
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (f32)
// CHECK-LABEL: func.func @intervening_write
// CHECK: scf.if
// CHECK: scf.for
// CHECK: memref.copy
// CHECK: scf.if
// CHECK: memref.store
// CHECK: memref.copy
// CHECK-LABEL: func.func @simt_is_untouched
// CHECK-NOT: scf.if
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (f32)
// CHECK-LABEL: func.func @scalar_only
// CHECK-NOT: scf.if
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (f32)
// CHECK-LABEL: func.func @indirect_address
// CHECK-NOT: scf.if
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (f32)
// CHECK-LABEL: func.func @mixed_loads_and_views
// CHECK: arith.cmpi slt, %[[LOWER:[^, ]+]],
// CHECK: scf.if
// CHECK: math.absf {{.*}} : tensor<1xf32>
// CHECK: scf.for {{.*}} = %[[LOWER]] to
// CHECK: memref.load
// CHECK: memref.copy
// CHECK: memref.subview
// CHECK: memref.subview
// CHECK: memref.copy
// CHECK: scf.if
// CHECK: math.absf {{.*}} : tensor<1xf32>
// CHECK: tensor.extract
// CHECK-LABEL: func.func @invariant_vector_math
// CHECK-NOT: scf.if
// CHECK: scf.for
// CHECK: math.absf {{.*}} : tensor<1xf32>
// CHECK-NOT: scf.if
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @positive(%x: memref<?xf32>, %n: i32, %y0: f32) -> f32 attributes {parallel_mode = "simd"} {

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %idx0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32
    %scale = arith.constant dense<0.625> : tensor<1xf32>
    %z = arith.constant dense<0.0> : tensor<1xf32>
    %one = arith.constant dense<1.0> : tensor<1xf32>
    %init = arith.constant 0.0 : f32
    %result = scf.for %j = %c0 to %n step %c1 iter_args(%T = %init) -> (f32) : i32 {
      %jf = arith.sitofp %j : i32 to f32
      %jt = tensor.from_elements %jf : tensor<1xf32>
      %y0t = tensor.from_elements %y0 : tensor<1xf32>
      %a = arith.mulf %jt, %scale : tensor<1xf32>
      %b = arith.addf %y0t, %a : tensor<1xf32>
      %yc = arith.maxnumf %b, %z : tensor<1xf32>
      %y = tensor.extract %yc[%idx0] : tensor<1xf32>
      %yi = arith.fptosi %y : f32 to i32
      %yi1 = arith.addi %yi, %c1 : i32
      %lo = arith.index_cast %yi : i32 to index
      %hi = arith.index_cast %yi1 : i32 to index
      %yf = arith.sitofp %yi : i32 to f32
      %yft = tensor.from_elements %yf : tensor<1xf32>
      %lyt = arith.subf %yc, %yft : tensor<1xf32>
      %hyt = arith.subf %one, %lyt : tensor<1xf32>
      %ly = tensor.extract %lyt[%idx0] : tensor<1xf32>
      %hy = tensor.extract %hyt[%idx0] : tensor<1xf32>
      %sl = memref.reinterpret_cast %x to offset: [%lo], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %bl = memref.alloc() : memref<1xf32>
      memref.copy %sl, %bl : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
      %vl = bufferization.to_tensor %bl restrict writable : memref<1xf32> to tensor<1xf32>
      %sh = memref.reinterpret_cast %x to offset: [%hi], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %bh = memref.alloc() : memref<1xf32>
      memref.copy %sh, %bh : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
      %vh = bufferization.to_tensor %bh restrict writable : memref<1xf32> to tensor<1xf32>
      %l = tensor.extract %vl[%idx0] : tensor<1xf32>
      %h = tensor.extract %vh[%idx0] : tensor<1xf32>
      %wl = arith.mulf %hy, %l : f32
      %t1 = arith.addf %T, %wl : f32
      %wh = arith.mulf %ly, %h : f32
      %t2 = arith.addf %t1, %wh : f32
      scf.yield %t2 : f32
    }
    return %result : f32
  }
  func.func @address_uses_accumulator(%x: memref<?xf32>, %n: i32, %y0: f32) -> f32 attributes {parallel_mode = "simd"} {

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %idx0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32
    %scale = arith.constant dense<0.625> : tensor<1xf32>
    %z = arith.constant dense<0.0> : tensor<1xf32>
    %one = arith.constant dense<1.0> : tensor<1xf32>
    %init = arith.constant 0.0 : f32
    %result = scf.for %j = %c0 to %n step %c1 iter_args(%T = %init) -> (f32) : i32 {
      %jf = arith.addf %T, %y0 : f32
      %jt = tensor.from_elements %jf : tensor<1xf32>
      %y0t = tensor.from_elements %y0 : tensor<1xf32>
      %a = arith.mulf %jt, %scale : tensor<1xf32>
      %b = arith.addf %y0t, %a : tensor<1xf32>
      %yc = arith.maxnumf %b, %z : tensor<1xf32>
      %y = tensor.extract %yc[%idx0] : tensor<1xf32>
      %yi = arith.fptosi %y : f32 to i32
      %yi1 = arith.addi %yi, %c1 : i32
      %lo = arith.index_cast %yi : i32 to index
      %hi = arith.index_cast %yi1 : i32 to index
      %yf = arith.sitofp %yi : i32 to f32
      %yft = tensor.from_elements %yf : tensor<1xf32>
      %lyt = arith.subf %yc, %yft : tensor<1xf32>
      %hyt = arith.subf %one, %lyt : tensor<1xf32>
      %ly = tensor.extract %lyt[%idx0] : tensor<1xf32>
      %hy = tensor.extract %hyt[%idx0] : tensor<1xf32>
      %sl = memref.reinterpret_cast %x to offset: [%lo], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %bl = memref.alloc() : memref<1xf32>
      memref.copy %sl, %bl : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
      %vl = bufferization.to_tensor %bl restrict writable : memref<1xf32> to tensor<1xf32>
      %sh = memref.reinterpret_cast %x to offset: [%hi], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %bh = memref.alloc() : memref<1xf32>
      memref.copy %sh, %bh : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
      %vh = bufferization.to_tensor %bh restrict writable : memref<1xf32> to tensor<1xf32>
      %l = tensor.extract %vl[%idx0] : tensor<1xf32>
      %h = tensor.extract %vh[%idx0] : tensor<1xf32>
      %wl = arith.mulf %hy, %l : f32
      %t1 = arith.addf %T, %wl : f32
      %wh = arith.mulf %ly, %h : f32
      %t2 = arith.addf %t1, %wh : f32
      scf.yield %t2 : f32
    }
    return %result : f32
  }
  func.func @intervening_write(%x: memref<?xf32>, %n: i32, %y0: f32) -> f32 attributes {parallel_mode = "simd"} {

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %idx0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32
    %scale = arith.constant dense<0.625> : tensor<1xf32>
    %z = arith.constant dense<0.0> : tensor<1xf32>
    %one = arith.constant dense<1.0> : tensor<1xf32>
    %init = arith.constant 0.0 : f32
    %result = scf.for %j = %c0 to %n step %c1 iter_args(%T = %init) -> (f32) : i32 {
      %jf = arith.sitofp %j : i32 to f32
      %jt = tensor.from_elements %jf : tensor<1xf32>
      %y0t = tensor.from_elements %y0 : tensor<1xf32>
      %a = arith.mulf %jt, %scale : tensor<1xf32>
      %b = arith.addf %y0t, %a : tensor<1xf32>
      %yc = arith.maxnumf %b, %z : tensor<1xf32>
      %y = tensor.extract %yc[%idx0] : tensor<1xf32>
      %yi = arith.fptosi %y : f32 to i32
      %yi1 = arith.addi %yi, %c1 : i32
      %lo = arith.index_cast %yi : i32 to index
      %hi = arith.index_cast %yi1 : i32 to index
      %yf = arith.sitofp %yi : i32 to f32
      %yft = tensor.from_elements %yf : tensor<1xf32>
      %lyt = arith.subf %yc, %yft : tensor<1xf32>
      %hyt = arith.subf %one, %lyt : tensor<1xf32>
      %ly = tensor.extract %lyt[%idx0] : tensor<1xf32>
      %hy = tensor.extract %hyt[%idx0] : tensor<1xf32>
      %sl = memref.reinterpret_cast %x to offset: [%lo], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %bl = memref.alloc() : memref<1xf32>
      memref.copy %sl, %bl : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
      %vl = bufferization.to_tensor %bl restrict writable : memref<1xf32> to tensor<1xf32>
      memref.store %f0, %x[%idx0] : memref<?xf32>
      %sh = memref.reinterpret_cast %x to offset: [%hi], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %bh = memref.alloc() : memref<1xf32>
      memref.copy %sh, %bh : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
      %vh = bufferization.to_tensor %bh restrict writable : memref<1xf32> to tensor<1xf32>
      %l = tensor.extract %vl[%idx0] : tensor<1xf32>
      %h = tensor.extract %vh[%idx0] : tensor<1xf32>
      %wl = arith.mulf %hy, %l : f32
      %t1 = arith.addf %T, %wl : f32
      %wh = arith.mulf %ly, %h : f32
      %t2 = arith.addf %t1, %wh : f32
      scf.yield %t2 : f32
    }
    return %result : f32
  }
  func.func @simt_is_untouched(%x: memref<?xf32>, %n: i32, %y0: f32) -> f32 attributes {parallel_mode = "simt"} {

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %idx0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32
    %scale = arith.constant dense<0.625> : tensor<1xf32>
    %z = arith.constant dense<0.0> : tensor<1xf32>
    %one = arith.constant dense<1.0> : tensor<1xf32>
    %init = arith.constant 0.0 : f32
    %result = scf.for %j = %c0 to %n step %c1 iter_args(%T = %init) -> (f32) : i32 {
      %jf = arith.sitofp %j : i32 to f32
      %jt = tensor.from_elements %jf : tensor<1xf32>
      %y0t = tensor.from_elements %y0 : tensor<1xf32>
      %a = arith.mulf %jt, %scale : tensor<1xf32>
      %b = arith.addf %y0t, %a : tensor<1xf32>
      %yc = arith.maxnumf %b, %z : tensor<1xf32>
      %y = tensor.extract %yc[%idx0] : tensor<1xf32>
      %yi = arith.fptosi %y : f32 to i32
      %yi1 = arith.addi %yi, %c1 : i32
      %lo = arith.index_cast %yi : i32 to index
      %hi = arith.index_cast %yi1 : i32 to index
      %yf = arith.sitofp %yi : i32 to f32
      %yft = tensor.from_elements %yf : tensor<1xf32>
      %lyt = arith.subf %yc, %yft : tensor<1xf32>
      %hyt = arith.subf %one, %lyt : tensor<1xf32>
      %ly = tensor.extract %lyt[%idx0] : tensor<1xf32>
      %hy = tensor.extract %hyt[%idx0] : tensor<1xf32>
      %sl = memref.reinterpret_cast %x to offset: [%lo], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %bl = memref.alloc() : memref<1xf32>
      memref.copy %sl, %bl : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
      %vl = bufferization.to_tensor %bl restrict writable : memref<1xf32> to tensor<1xf32>
      %sh = memref.reinterpret_cast %x to offset: [%hi], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %bh = memref.alloc() : memref<1xf32>
      memref.copy %sh, %bh : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
      %vh = bufferization.to_tensor %bh restrict writable : memref<1xf32> to tensor<1xf32>
      %l = tensor.extract %vl[%idx0] : tensor<1xf32>
      %h = tensor.extract %vh[%idx0] : tensor<1xf32>
      %wl = arith.mulf %hy, %l : f32
      %t1 = arith.addf %T, %wl : f32
      %wh = arith.mulf %ly, %h : f32
      %t2 = arith.addf %t1, %wh : f32
      scf.yield %t2 : f32
    }
    return %result : f32
  }
  func.func @scalar_only(%x: memref<?xf32>, %n: i32, %y0: f32) -> f32 attributes {parallel_mode = "simd"} {

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %idx0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32
    %scale = arith.constant dense<0.625> : tensor<1xf32>
    %z = arith.constant dense<0.0> : tensor<1xf32>
    %one = arith.constant dense<1.0> : tensor<1xf32>
    %init = arith.constant 0.0 : f32
    %result = scf.for %j = %c0 to %n step %c1 iter_args(%T = %init) -> (f32) : i32 {
      %jf = arith.sitofp %j : i32 to f32
      %jt = tensor.from_elements %jf : tensor<1xf32>
      %y0t = tensor.from_elements %y0 : tensor<1xf32>
      %s = arith.constant 0.625 : f32
      %a = arith.mulf %jf, %s : f32
      %b = arith.addf %y0, %a : f32
      %y = arith.maxnumf %b, %f0 : f32
      %yc = tensor.from_elements %y : tensor<1xf32>
      %yi = arith.fptosi %y : f32 to i32
      %yi1 = arith.addi %yi, %c1 : i32
      %lo = arith.index_cast %yi : i32 to index
      %hi = arith.index_cast %yi1 : i32 to index
      %yf = arith.sitofp %yi : i32 to f32
      %yft = tensor.from_elements %yf : tensor<1xf32>
      %ly = arith.subf %y, %yf : f32
      %hy = arith.subf %f1, %ly : f32
      %sl = memref.reinterpret_cast %x to offset: [%lo], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %bl = memref.alloc() : memref<1xf32>
      memref.copy %sl, %bl : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
      %vl = bufferization.to_tensor %bl restrict writable : memref<1xf32> to tensor<1xf32>
      %sh = memref.reinterpret_cast %x to offset: [%hi], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %bh = memref.alloc() : memref<1xf32>
      memref.copy %sh, %bh : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
      %vh = bufferization.to_tensor %bh restrict writable : memref<1xf32> to tensor<1xf32>
      %l = tensor.extract %vl[%idx0] : tensor<1xf32>
      %h = tensor.extract %vh[%idx0] : tensor<1xf32>
      %wl = arith.mulf %hy, %l : f32
      %t1 = arith.addf %T, %wl : f32
      %wh = arith.mulf %ly, %h : f32
      %t2 = arith.addf %t1, %wh : f32
      scf.yield %t2 : f32
    }
    return %result : f32
  }
  func.func @indirect_address(%x: memref<?xf32>, %n: i32, %y0: f32) -> f32 attributes {parallel_mode = "simd"} {

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %idx0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32
    %scale = arith.constant dense<0.625> : tensor<1xf32>
    %z = arith.constant dense<0.0> : tensor<1xf32>
    %one = arith.constant dense<1.0> : tensor<1xf32>
    %init = arith.constant 0.0 : f32
    %result = scf.for %j = %c0 to %n step %c1 iter_args(%T = %init) -> (f32) : i32 {
      %jf = memref.load %x[%idx0] : memref<?xf32>
      %jt = tensor.from_elements %jf : tensor<1xf32>
      %y0t = tensor.from_elements %y0 : tensor<1xf32>
      %a = arith.mulf %jt, %scale : tensor<1xf32>
      %b = arith.addf %y0t, %a : tensor<1xf32>
      %yc = arith.maxnumf %b, %z : tensor<1xf32>
      %y = tensor.extract %yc[%idx0] : tensor<1xf32>
      %yi = arith.fptosi %y : f32 to i32
      %yi1 = arith.addi %yi, %c1 : i32
      %lo = arith.index_cast %yi : i32 to index
      %hi = arith.index_cast %yi1 : i32 to index
      %yf = arith.sitofp %yi : i32 to f32
      %yft = tensor.from_elements %yf : tensor<1xf32>
      %lyt = arith.subf %yc, %yft : tensor<1xf32>
      %hyt = arith.subf %one, %lyt : tensor<1xf32>
      %ly = tensor.extract %lyt[%idx0] : tensor<1xf32>
      %hy = tensor.extract %hyt[%idx0] : tensor<1xf32>
      %sl = memref.reinterpret_cast %x to offset: [%lo], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %bl = memref.alloc() : memref<1xf32>
      memref.copy %sl, %bl : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
      %vl = bufferization.to_tensor %bl restrict writable : memref<1xf32> to tensor<1xf32>
      %sh = memref.reinterpret_cast %x to offset: [%hi], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %bh = memref.alloc() : memref<1xf32>
      memref.copy %sh, %bh : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
      %vh = bufferization.to_tensor %bh restrict writable : memref<1xf32> to tensor<1xf32>
      %l = tensor.extract %vl[%idx0] : tensor<1xf32>
      %h = tensor.extract %vh[%idx0] : tensor<1xf32>
      %wl = arith.mulf %hy, %l : f32
      %t1 = arith.addf %T, %wl : f32
      %wh = arith.mulf %ly, %h : f32
      %t2 = arith.addf %t1, %wh : f32
      scf.yield %t2 : f32
    }
    return %result : f32
  }

  func.func @mixed_loads_and_views(%x: memref<?xf32>, %indices: memref<?xindex>,
                                  %lb: i32, %ub: i32, %shift: i32) -> f32 attributes {parallel_mode = "simd"} {
    %one = arith.constant 1 : i32
    %zero = arith.constant 0 : index
    %init = arith.constant 0.0 : f32
    %result:2 = scf.for %i = %lb to %ub step %one iter_args(%sum = %init, %s = %shift) -> (f32, i32) : i32 {
      // This copy reads an indirect offset. It must stay in place and must
      // not disqualify the later copy.
      %iv = arith.index_cast %i : i32 to index
      %indirect = memref.load %indices[%iv] : memref<?xindex>
      %src0 = memref.reinterpret_cast %x to offset: [%indirect], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %dst0 = memref.alloc() : memref<1xf32>
      memref.copy %src0, %dst0 : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
      %tile0 = bufferization.to_tensor %dst0 restrict : memref<1xf32> to tensor<1xf32>

      %shifted = arith.addi %i, %s : i32
      %f = arith.sitofp %shifted : i32 to f32
      %vec = tensor.from_elements %f : tensor<1xf32>
      %abs = math.absf %vec : tensor<1xf32>
      %scalar = tensor.extract %abs[%zero] : tensor<1xf32>
      %offset = arith.fptosi %scalar : f32 to i32
      %idx = arith.index_cast %offset : i32 to index
      %src1 = memref.subview %x[%idx] [1] [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %base = memref.alloc() : memref<2xf32>
      %dst1 = memref.subview %base[0] [1] [1] : memref<2xf32> to memref<1xf32, strided<[1]>>
      memref.copy %src1, %dst1 : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1]>>
      %tile1 = bufferization.to_tensor %dst1 restrict : memref<1xf32, strided<[1]>> to tensor<1xf32>
      %a = tensor.extract %tile0[%zero] : tensor<1xf32>
      %b = tensor.extract %tile1[%zero] : tensor<1xf32>
      %next0 = arith.addf %sum, %a : f32
      %next1 = arith.addf %next0, %b : f32
      scf.yield %next1, %s : f32, i32
    }
    return %result#0 : f32
  }

  func.func @invariant_vector_math(%x: memref<?xf32>, %ub: i32,
                                   %start: tensor<1xf32>) -> f32 attributes {parallel_mode = "simd"} {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %idx0 = arith.constant 0 : index
    %init = arith.constant 0.0 : f32
    %result = scf.for %i = %zero to %ub step %one iter_args(%sum = %init) -> (f32) : i32 {
      // The address depends on the IV, but the Vector operation doesn't.
      %abs = math.absf %start : tensor<1xf32>
      %scalar = tensor.extract %abs[%idx0] : tensor<1xf32>
      %base = arith.fptosi %scalar : f32 to i32
      %offset = arith.addi %base, %i : i32
      %idx = arith.index_cast %offset : i32 to index
      %src = memref.subview %x[%idx] [1] [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %dst = memref.alloc() : memref<1xf32>
      memref.copy %src, %dst : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
      %tile = bufferization.to_tensor %dst restrict : memref<1xf32> to tensor<1xf32>
      %v = tensor.extract %tile[%idx0] : tensor<1xf32>
      %next = arith.addf %sum, %v : f32
      scf.yield %next : f32
    }
    return %result : f32
  }
}
