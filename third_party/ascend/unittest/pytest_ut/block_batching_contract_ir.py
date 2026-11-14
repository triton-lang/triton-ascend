# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""TTIR fixtures for the VV source/layout contract regressions."""
import json
from triton._C.libtriton import ascend, ir


def parse(tmp_path, text):
    context = ir.context()
    ir.load_dialects(context)
    ascend.ir.load_dialects(context)
    path = tmp_path / 'batch.ttir'
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    return module


def pointwise(axis):
    return '''module {
  tt.func public @arbitrary_name(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32) {
    %p = tt.get_program_id AXIS : i32
    %g = tt.get_num_programs AXIS : i32
    %gf = arith.sitofp %g : i32 to f32
    %gv = tt.splat %gf : f32 -> tensor<4xf32>
    %b = arith.constant 4 : i32
    %base = arith.muli %p, %b : i32
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %basev = tt.splat %base : i32 -> tensor<4xi32>
    %idx = arith.addi %basev, %range : tensor<4xi32>
    %nv = tt.splat %n : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %idx, %nv : tensor<4xi32>
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %xa = tt.addptr %xp, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %ya = tt.addptr %yp, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %zero = arith.constant dense<0.0> : tensor<4xf32>
    %v = tt.load %xa, %mask, %zero : tensor<4x!tt.ptr<f32>>
    %result = arith.addf %v, %gv : tensor<4xf32>
    tt.store %ya, %result, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}'''.replace('AXIS', 'xyz'[axis])


def analyze(tmp_path, text, factor=3, axis=0):
    module = parse(tmp_path, text)
    before = str(module)
    result = ascend.passes.ttir.analyze_block_batching(module, factor, axis)
    assert str(module) == before
    assert module.verify()
    assert result["compile_status"] == "ANALYZED", result
    layout = json.loads(result["layout_json"])
    assert layout["planner_version"] == 2
    assert not any(row["rewritten"] for row in layout["accesses"])
    return layout["accesses"]


REDUCTION_KINDS = ('sum', 'max', 'maxnum', 'maxnum_capture', 'argmax', 'argmax_index', 'scan', 'reverse_scan',
                   'tuple_scan')


def reduction_ir(kind):
    assert kind in REDUCTION_KINDS
    text = """module {
  tt.func public @block_batching_elementwise(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32) {
    %p = tt.get_program_id x : i32
    %g = tt.get_num_programs x : i32
    %b = arith.constant 4 : i32
    %base = arith.muli %p, %b : i32
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %basev = tt.splat %base : i32 -> tensor<4xi32>
    %idx = arith.addi %basev, %range : tensor<4xi32>
    %nv = tt.splat %n : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %idx, %nv : tensor<4xi32>
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %xa = tt.addptr %xp, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %ya = tt.addptr %yp, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %onef = arith.constant dense<1.0> : tensor<4xf32>
    %twof = arith.constant dense<2.0> : tensor<4xf32>
    %onei = arith.constant dense<1> : tensor<4xi32>
    %twoi = arith.constant dense<2> : tensor<4xi32>
    %fouri = arith.constant dense<4> : tensor<4xi32>
    %thirtytwo = arith.constant dense<32> : tensor<4xi32>
    %big = arith.constant dense<1073741824> : tensor<4xi32>
    %v = tt.load %xa, %mask, %onef : tensor<4x!tt.ptr<f32>>
    %pv = tt.splat %p : i32 -> tensor<4xi32>
    %gv = tt.splat %g : i32 -> tensor<4xi32>
    %iv = arith.addi %pv, %onei : tensor<4xi32>
    %signed = arith.subi %iv, %fouri : tensor<4xi32>
    %denom = arith.subi %gv, %pv : tensor<4xi32>
    %denomf = arith.sitofp %denom : tensor<4xi32> to tensor<4xf32>
    %shift = arith.subi %thirtytwo, %denom : tensor<4xi32>
    %op_out = "arith.maximumf"(%v, %denomf) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    tt.store %ya, %op_out, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}"""
    old = '%op_out = "arith.maximumf"(%v, %denomf) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>'
    scan = kind in ("scan", "reverse_scan", "tuple_scan")
    tuple_ = kind in ("argmax", "argmax_index", "tuple_scan", "maxnum_capture")
    op, ret = ("tt.scan", "tt.scan.return") if scan else ("tt.reduce", "tt.reduce.return")
    args = "%v, %range" if tuple_ else "%v"
    inputs = "tensor<4xf32>, tensor<4xi32>" if tuple_ else "tensor<4xf32>"
    result_types = ("tensor<4xf32>, tensor<4xi32>" if tuple_ else "tensor<4xf32>") if scan else (
        "f32, i32" if tuple_ else "f32")
    result_names = "%reduced, %index" if tuple_ else "%reduced"
    attributes = "axis = 0 : i32"
    if scan:
        attributes += ", reverse = " + ("true" if kind == "reverse_scan" else "false")
    parameters = "%lhs_value: f32, %lhs_index: i32, %rhs_value: f32, %rhs_index: i32" if tuple_ else "%lhs_value: f32, %rhs_value: f32"
    if kind in ("argmax", "argmax_index"):
        combiner = '''%gt = arith.cmpf ogt, %lhs_value, %rhs_value : f32
        %eq_values = arith.cmpf oeq, %lhs_value, %rhs_value : f32
        %lhs_nan = arith.cmpf uno, %lhs_value, %lhs_value : f32
        %rhs_nan = arith.cmpf uno, %rhs_value, %rhs_value : f32
        %both_nan = arith.andi %lhs_nan, %rhs_nan : i1
        %eq = arith.ori %eq_values, %both_nan : i1
        %true = arith.constant true
        %rhs_number = arith.xori %rhs_nan, %true : i1
        %lhs_nan_first = arith.andi %lhs_nan, %rhs_number : i1
        %greater = arith.ori %gt, %lhs_nan_first : i1
        %smaller = arith.cmpi slt, %lhs_index, %rhs_index : i32
        %tie = arith.andi %eq, %smaller : i1
        %pick = arith.ori %greater, %tie : i1
        %value = arith.select %pick, %lhs_value, %rhs_value : f32
        %which = arith.select %pick, %lhs_index, %rhs_index : i32
        tt.reduce.return %value, %which : f32, i32'''
    elif kind == "maxnum_capture":
        # The runtime scalar is used only by the terminator, never by an
        # ordinary combiner operation. Both tuple results remain observable.
        combiner = """%value = arith.maxnumf %lhs_value, %rhs_value : f32
        tt.reduce.return %value, %n : f32, i32"""
    elif kind == "tuple_scan":
        combiner = '''%value = arith.addf %lhs_value, %rhs_value : f32
        %which = arith.addi %lhs_index, %rhs_index : i32
        tt.scan.return %value, %which : f32, i32'''
    else:
        function = "arith.maximumf" if kind == "max" else "arith.maxnumf" if kind == "maxnum" else "arith.addf"
        combiner = f'%value = {function} %lhs_value, %rhs_value : f32\n        {ret} %value : f32'
    body = f'''{result_names} = "{op}"({args}) <{{{attributes}}}> ({{
      ^bb0({parameters}):
        {combiner}
    }}) : ({inputs}) -> ({result_types})
'''
    if tuple_:
        input_type = "tensor<4xi32>" if scan else "i32"
        output_type = "tensor<4xf32>" if scan else "f32"
        body += f'    %float_index = arith.sitofp %index : {input_type} to {output_type}\n'
        body += f'    %combined = arith.addf %reduced, %float_index : {output_type}\n'
        result = "%float_index" if kind == "argmax_index" else "%combined"
    else:
        result = "%reduced"
    body += f'    %op_out = arith.addf {result}, %onef : tensor<4xf32>' if scan else f'    %op_out = tt.splat {result} : f32 -> tensor<4xf32>'
    return text.replace(old, body)


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


def recurrent_ir():
    return '''module {
  tt.func public @recurrence(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %p = tt.get_program_id x : i32
    %g = tt.get_num_programs x : i32
    %c = arith.constant 4 : i32
    %lo = arith.muli %p, %c : i32
    %step = arith.muli %g, %c : i32
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %nv = tt.splat %n : i32 -> tensor<4xi32>
    %z = arith.constant dense<0.0> : tensor<4xf32>
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %sum = scf.for %iv = %lo to %n step %step iter_args(%acc = %z) -> (tensor<4xf32>) : i32 {
      %ivv = tt.splat %iv : i32 -> tensor<4xi32>
      %ix = arith.addi %ivv, %r : tensor<4xi32>
      %mask = arith.cmpi slt, %ix, %nv : tensor<4xi32>
      %xp = tt.addptr %xs, %ix : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %v = tt.load %xp, %mask, %z : tensor<4x!tt.ptr<f32>>
      %next = arith.addf %acc, %v : tensor<4xf32>
      scf.yield %next : tensor<4xf32>
    }
    %pv = tt.splat %lo : i32 -> tensor<4xi32>
    %oi = arith.addi %pv, %r : tensor<4xi32>
    %os = tt.splat %o : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %op = tt.addptr %os, %oi : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %op, %sum : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}'''


def constant_recurrence(lower, upper, step, bits):
    return f'''module {{
  tt.func public @static_recurrence(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i{bits}) {{
    %p = tt.get_program_id x : i32
    %xp = tt.addptr %x, %p : !tt.ptr<f32>, i32
    %op = tt.addptr %o, %p : !tt.ptr<f32>, i32
    %lo = arith.constant {lower} : i{bits}
    %hi = arith.constant {upper if upper is not None else 16} : i{bits}
    %step = arith.constant {step} : i{bits}
    %zero = arith.constant 0.0 : f32
    %sum = scf.for %iv = %lo to {'%hi' if upper is not None else '%n'} step %step
        iter_args(%acc = %zero) -> (f32) : i{bits} {{
      %v = tt.load %xp : !tt.ptr<f32>
      %next = arith.addf %acc, %v : f32
      scf.yield %next : f32
    }}
    tt.store %op, %sum : !tt.ptr<f32>
    tt.return
  }}
}}'''
