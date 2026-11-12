# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Small original TTIR fixtures for byte reads and known external functions."""


def case_ir(kind="bool", symbol=None, pure=True, libpath=""):
    ctype = "i8" if kind == "byte" else "i1"
    f = "tensor<8xf32>"
    if kind in ("bool", "byte"):
        cast = "%cp8 = tt.bitcast %cp : tensor<8x!tt.ptr<i1>> -> tensor<8x!tt.ptr<i8>>" if kind == "bool" else ""
        pointer = "%cp8" if kind == "bool" else "%cp"
        marker = " {was_bool_to_int8 = true}" if kind == "bool" else ""
        calculation = f"""
    %bytezero = arith.constant dense<0> : tensor<8xi8>
    %cs = tt.splat %c : !tt.ptr<{ctype}> -> tensor<8x!tt.ptr<{ctype}>>
    %cp = tt.addptr %cs, %idx : tensor<8x!tt.ptr<{ctype}>>, tensor<8xi32>
    {cast}
    %bytes = tt.load {pointer}, %mask, %bytezero{marker} : tensor<8x!tt.ptr<i8>>
    %condition = arith.cmpi ne, %bytes, %bytezero : tensor<8xi8>
    %result = arith.select %condition, %xv, %yv : tensor<8xi1>, {f}
"""
    else:
        symbol = symbol or ("__hmf_pow_fp32" if kind == "pow" else "__hmf_tanhf")
        operands, types = ("%xv, %yv", f"{f}, {f}") if kind == "pow" else ("%xv", f)
        calculation = f'''%result = tt.extern_elementwise {operands} {{libname = "", libpath = "{libpath}", pure = {str(pure).lower()}, symbol = "{symbol}"}} : ({types}) -> {f}'''
    return f'''module {{
  tt.func public @block_batching_byte_extern(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %c: !tt.ptr<{ctype}>, %out: !tt.ptr<f32>, %n: i32) {{
    %eight = arith.constant 8 : i32
    %zero = arith.constant dense<0.0> : {f}
    %pid = tt.get_program_id x : i32
    %base = arith.muli %pid, %eight : i32
    %bs = tt.splat %base : i32 -> tensor<8xi32>
    %range = tt.make_range {{start = 0 : i32, end = 8 : i32}} : tensor<8xi32>
    %idx = arith.addi %bs, %range : tensor<8xi32>
    %ns = tt.splat %n : i32 -> tensor<8xi32>
    %mask = arith.cmpi slt, %idx, %ns : tensor<8xi32>
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ys = tt.splat %y : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %xp = tt.addptr %xs, %idx : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %yp = tt.addptr %ys, %idx : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %xv = tt.load %xp, %mask, %zero : tensor<8x!tt.ptr<f32>>
    %yv = tt.load %yp, %mask, %zero : tensor<8x!tt.ptr<f32>>
    {calculation}
    %os = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %op = tt.addptr %os, %idx : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %op, %result, %mask : tensor<8x!tt.ptr<f32>>
    tt.return
  }}
}}'''
