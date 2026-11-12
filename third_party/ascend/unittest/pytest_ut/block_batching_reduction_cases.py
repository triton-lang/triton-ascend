# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
from block_batching_op_cases import case_ir

REDUCTION_KINDS = ("sum", "max", "maxnum", "maxnum_capture", "argmax", "argmax_index", "scan", "reverse_scan",
                   "tuple_scan")


def reduction_ir(kind):
    assert kind in REDUCTION_KINDS
    text = case_ir("arith.maximumf")
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


def reduction_reference(kind, x, blocks):
    import torch
    padded = torch.cat((x, torch.ones(blocks * 4 - x.numel()))).reshape(blocks, 4)
    if kind == "sum":
        result = padded.sum(1).repeat_interleave(4)
    elif kind == "max":
        result = padded.max(1).values.repeat_interleave(4)
    elif kind in ("argmax", "argmax_index"):
        value, index = padded.max(1)
        result = (index.float() if kind == "argmax_index" else value + index.float()).repeat_interleave(4)
    elif kind in ("maxnum", "maxnum_capture"):
        value = padded[:, 0]
        for column in range(1, 4):
            value = torch.fmax(value, padded[:, column])
        if kind == "maxnum_capture":
            value = value + x.numel()
        result = value.repeat_interleave(4)
    elif kind == "reverse_scan":
        result = padded.flip(1).cumsum(1).flip(1).reshape(-1) + 1
    else:
        result = padded.cumsum(1)
        if kind == "tuple_scan":
            result += torch.arange(4).cumsum(0)
        result = result.reshape(-1) + 1
    return result[:x.numel()]
