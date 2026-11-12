# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Small native IR fixture with independent dynamic input/output row strides."""


def symbolic_columns_ir():
    source = symbolic_memory_ir()
    return source.replace(
        "%cw = arith.extsi %cc : tensor<4xi32> to tensor<4xi64>", """%column_p = tt.get_program_id y : i32
    %column_pw = arith.extsi %column_p : i32 to i64
    %four = arith.constant 4 : i64
    %column_base = arith.muli %column_pw, %four : i64
    %column_bv = tt.splat %column_base : i64 -> tensor<4xi64>
    %cw_local = arith.extsi %cc : tensor<4xi32> to tensor<4xi64>
    %cw = arith.addi %cw_local, %column_bv : tensor<4xi64>""")


def symbolic_memory_ir(axis=0, mode="plain", origin=0):
    stride = "%sx"
    prelude = ""
    if mode == "affine_parameter":
        prelude = "%sx2 = arith.addi %sx, %sx : i32"
        stride = "%sx2"
    elif mode == "parameter_product":
        prelude = "%sx2 = arith.muli %sx, %sy : i32"
        stride = "%sx2"
    row_offset = "%xm = arith.muli %rows, %sxv : tensor<2x4xi64>"
    if mode == "mismatch":
        row_offset = """%pv = tt.splat %base : i64 -> tensor<2x4xi64>
    %localv = tt.broadcast %r : tensor<2x1xi64> -> tensor<2x4xi64>
    %pm = arith.muli %pv, %sxv : tensor<2x4xi64>
    %rm = arith.muli %localv, %syv : tensor<2x4xi64>
    %xm = arith.addi %pm, %rm : tensor<2x4xi64>"""
    return f'''module {{
  tt.func public @symbolic_rows(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %nr: i32, %nc: i32, %sx: i32, %sy: i32) {{
    %p = tt.get_program_id {"xyz"[axis]} : i32
    %pw = arith.extsi %p : i32 to i64
    %two = arith.constant 2 : i64
    %base = arith.muli %pw, %two : i64
    %rr = tt.make_range {{start = {origin} : i32, end = {origin+2} : i32}} : tensor<2xi32>
    %cc = tt.make_range {{start = 0 : i32, end = 4 : i32}} : tensor<4xi32>
    %rw = arith.extsi %rr : tensor<2xi32> to tensor<2xi64>
    %cw = arith.extsi %cc : tensor<4xi32> to tensor<4xi64>
    %r = tt.expand_dims %rw {{axis = 1 : i32}} : tensor<2xi64> -> tensor<2x1xi64>
    %c = tt.expand_dims %cw {{axis = 0 : i32}} : tensor<4xi64> -> tensor<1x4xi64>
    %bv = tt.splat %base : i64 -> tensor<2x1xi64>
    %row = arith.addi %bv, %r : tensor<2x1xi64>
    %rows = tt.broadcast %row : tensor<2x1xi64> -> tensor<2x4xi64>
    %cols = tt.broadcast %c : tensor<1x4xi64> -> tensor<2x4xi64>
    %nrw = arith.extsi %nr : i32 to i64
    %ncw = arith.extsi %nc : i32 to i64
    %nrv = tt.splat %nrw : i64 -> tensor<2x4xi64>
    %ncv = tt.splat %ncw : i64 -> tensor<2x4xi64>
    %mr = arith.cmpi slt, %rows, %nrv : tensor<2x4xi64>
    %mc = arith.cmpi slt, %cols, %ncv : tensor<2x4xi64>
    %mask = arith.andi %mr, %mc : tensor<2x4xi1>
    {prelude}
    %sxw = arith.extsi {stride} : i32 to i64
    %syw = arith.extsi %sy : i32 to i64
    %sxv = tt.splat %sxw : i64 -> tensor<2x4xi64>
    %syv = tt.splat %syw : i64 -> tensor<2x4xi64>
    {row_offset}
    %ym = arith.muli %rows, %syv : tensor<2x4xi64>
    %xo = arith.addi %xm, %cols : tensor<2x4xi64>
    %yo = arith.addi %ym, %cols : tensor<2x4xi64>
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %xa = tt.addptr %xp, %xo : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi64>
    %ya = tt.addptr %yp, %yo : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi64>
    %other = arith.constant dense<7.0> : tensor<2x4xf32>
    %one = arith.constant dense<1.0> : tensor<2x4xf32>
    %v = tt.load %xa, %mask, %other : tensor<2x4x!tt.ptr<f32>>
    %out = arith.addf %v, %one : tensor<2x4xf32>
    tt.store %ya, %out, %mask : tensor<2x4x!tt.ptr<f32>>
    tt.return
  }}
}}'''
