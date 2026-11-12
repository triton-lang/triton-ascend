# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
KINDS = ("nd_reused_range", "ptr_roundtrip", "ptr_bitcast", "integer64", "index_cast", "index_castui",
         "masked_scalar_like")


def memory_signature(kind, *, mixed_width=False):
    return {
        "x":
        "*i64" if kind == "integer64" else "*fp32", "y":
        "*i64" if kind == "integer64" or
        (kind == "ptr_bitcast" and mixed_width) else "*i32" if kind == "ptr_bitcast" else "*fp32", "n":
        "i32", "k":
        "i32"
    }


def memory_ir(kind, layout=True, *, mixed_width=False):
    assert kind in KINDS
    x_type = "i64" if kind == "integer64" else "f32"
    y_type = "i64" if kind == "integer64" or (kind == "ptr_bitcast"
                                              and mixed_width) else "i32" if kind == "ptr_bitcast" else "f32"
    attr = ' attributes {llvm.data_layout = "e-p:64:64:64:64"}' if kind.startswith("index_") and layout else ""
    prefix = f'''module{attr} {{
  tt.func public @block_batching_memory(%x: !tt.ptr<{x_type}>, %y: !tt.ptr<{y_type}>, %n: i32, %k: i32) {{
    %p = tt.get_program_id x : i32
    %one = arith.constant 1.0 : f32
'''
    if kind == "nd_reused_range":
        body = '''    %four = arith.constant 4 : i32
    %base = arith.muli %p, %four : i32
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %r = tt.expand_dims %range {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %c = tt.expand_dims %range {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %bv = tt.splat %base : i32 -> tensor<4x1xi32>
    %row = arith.addi %r, %bv : tensor<4x1xi32>
    %rows = tt.broadcast %row : tensor<4x1xi32> -> tensor<4x4xi32>
    %cols = tt.broadcast %c : tensor<1x4xi32> -> tensor<4x4xi32>
    %nv = tt.splat %n : i32 -> tensor<4x4xi32>
    %kv = tt.splat %k : i32 -> tensor<4x4xi32>
    %r_mask = arith.cmpi slt, %rows, %nv : tensor<4x4xi32>
    %c_mask = arith.cmpi slt, %cols, %kv : tensor<4x4xi32>
    %mask = arith.andi %r_mask, %c_mask : tensor<4x4xi1>
    %row_offset = arith.muli %rows, %kv : tensor<4x4xi32>
    %offset = arith.addi %row_offset, %cols : tensor<4x4xi32>
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %xa = tt.addptr %xp, %offset : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %ya = tt.addptr %yp, %offset : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %zero = arith.constant dense<0.0> : tensor<4x4xf32>
    %ones = arith.constant dense<1.0> : tensor<4x4xf32>
    %value = tt.load %xa, %mask, %zero : tensor<4x4x!tt.ptr<f32>>
    %result = arith.addf %value, %ones : tensor<4x4xf32>
    tt.store %ya, %result, %mask : tensor<4x4x!tt.ptr<f32>>
'''
    elif kind == "masked_scalar_like":
        body = '''    %zero_i = arith.constant 0 : i32
    %huge = arith.constant 1099511627776 : i64
    %invalid = tt.addptr %x, %huge : !tt.ptr<f32>, i64
    %xp = tt.splat %invalid : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %enabled = arith.cmpi slt, %k, %zero_i : i32
    %mask = tt.splat %enabled : i1 -> tensor<4xi1>
    %other = arith.constant dense<7.0> : tensor<4xf32>
    %value = tt.load %xp, %mask, %other : tensor<4x!tt.ptr<f32>>
    %four = arith.constant 4 : i32
    %base = arith.muli %p, %four : i32
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %bv = tt.splat %base : i32 -> tensor<4xi32>
    %offset = arith.addi %bv, %range : tensor<4xi32>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ya = tt.addptr %yp, %offset : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %ya, %value : tensor<4x!tt.ptr<f32>>
'''
    elif kind == "ptr_roundtrip":
        body = '''    %wide = arith.extsi %p : i32 to i64
    %four = arith.constant 4 : i64
    %bytes = arith.muli %wide, %four : i64
    %bits = tt.ptr_to_int %x : !tt.ptr<f32> -> i64
    %address = arith.addi %bits, %bytes : i64
    %xp = tt.int_to_ptr %address : i64 -> !tt.ptr<f32>
    %yp = tt.addptr %y, %p : !tt.ptr<f32>, i32
    %value = tt.load %xp : !tt.ptr<f32>
    %result = arith.addf %value, %one : f32
    tt.store %yp, %result : !tt.ptr<f32>
'''
    elif kind == "ptr_bitcast":
        body = '''    %two = arith.constant 2 : i32
    %offset = arith.muli %p, %two : i32
    %xp = tt.addptr %x, %offset : !tt.ptr<f32>, i32
    %view = tt.bitcast %xp : !tt.ptr<f32> -> !tt.ptr<i64>
    %yp = tt.addptr %y, %p : !tt.ptr<i64>, i32
    %value = tt.load %view : !tt.ptr<i64>
    tt.store %yp, %value : !tt.ptr<i64>
'''
    elif kind == "integer64":
        body = '''    %xp = tt.addptr %x, %p : !tt.ptr<i64>, i32
    %yp = tt.addptr %y, %p : !tt.ptr<i64>, i32
    %value = tt.load %xp : !tt.ptr<i64>
    tt.store %yp, %value : !tt.ptr<i64>
'''
    else:
        cast = "arith.index_castui" if kind == "index_castui" else "arith.index_cast"
        body = f'''    %index = {cast} %k : i32 to index
    %index_one = arith.constant 1 : index
    %next = arith.addi %index, %index_one : index
    %wide = arith.index_cast %next : index to i64
    %scalar = arith.sitofp %wide : i64 to f32
    %four = arith.constant 4 : i32
    %base = arith.muli %p, %four : i32
    %range = tt.make_range {{start = 0 : i32, end = 4 : i32}} : tensor<4xi32>
    %bv = tt.splat %base : i32 -> tensor<4xi32>
    %idx = arith.addi %bv, %range : tensor<4xi32>
    %nv = tt.splat %n : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %idx, %nv : tensor<4xi32>
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %xa = tt.addptr %xp, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %ya = tt.addptr %yp, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %zeros = arith.constant dense<0.0> : tensor<4xf32>
    %bias = tt.splat %scalar : f32 -> tensor<4xf32>
    %value = tt.load %xa, %mask, %zeros : tensor<4x!tt.ptr<f32>>
    %result = arith.addf %value, %bias : tensor<4xf32>
    tt.store %ya, %result, %mask : tensor<4x!tt.ptr<f32>>
'''
    if kind == "ptr_bitcast" and not mixed_width:
        # The stock TA converter supports equal-pointee-width pointer casts.
        # mixed_width=True preserves the original f32*->i64* reproducer.
        body = body.replace("%two = arith.constant 2",
                            "%two = arith.constant 1").replace("!tt.ptr<i64>", "!tt.ptr<i32>")
    return prefix + body + "    tt.return\n  }\n}"
