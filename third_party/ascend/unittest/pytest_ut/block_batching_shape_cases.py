# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
from block_batching_op_cases import case_ir

SHAPE_OPS = ("tt.unsplat", "tt.expand_dims", "tt.broadcast", "tt.reshape", "tt.trans", "tt.join", "tt.split")


def shape_ir(op, *, literal_weights=False):
    assert op in SHAPE_OPS
    prefix = case_ir("arith.maximumf")
    old = '%op_out = "arith.maximumf"(%v, %denomf) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>'
    assert old in prefix
    if op == "tt.unsplat":
        body = '''%xp1 = tt.splat %x : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
    %pv1 = tt.splat %p : i32 -> tensor<1xi32>
    %address1 = tt.addptr %xp1, %pv1 : tensor<1x!tt.ptr<f32>>, tensor<1xi32>
    %value1 = tt.load %address1 : tensor<1x!tt.ptr<f32>>
    %scalar = tt.unsplat %value1 : tensor<1xf32>
    %op_out = tt.splat %scalar : f32 -> tensor<4xf32>'''
    elif op in ("tt.expand_dims", "tt.broadcast"):
        body = '''%expanded = tt.expand_dims %v {axis = 0 : i32} : tensor<4xf32> -> tensor<1x4xf32>
    %wide = tt.broadcast %expanded : tensor<1x4xf32> -> tensor<2x4xf32>
    %weights = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
    %weighted = arith.mulf %wide, %weights : tensor<2x4xf32>
    %op_out = "tt.reduce"(%weighted) <{axis = 0 : i32}> ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        tt.reduce.return %sum : f32
    }) : (tensor<2x4xf32>) -> tensor<4xf32>'''
    elif op == "tt.reshape":
        body = '''%matrix = tt.reshape %v : tensor<4xf32> -> tensor<2x2xf32>
    %weights = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %weighted = arith.mulf %matrix, %weights : tensor<2x2xf32>
    %op_out = tt.reshape %weighted : tensor<2x2xf32> -> tensor<4xf32>'''
    elif op == "tt.trans":
        body = '''%matrix = tt.reshape %v : tensor<4xf32> -> tensor<2x2xf32>
    %transposed = tt.trans %matrix {order = array<i32: 1, 0>} : tensor<2x2xf32> -> tensor<2x2xf32>
    %op_out = tt.reshape %transposed : tensor<2x2xf32> -> tensor<4xf32>'''
    elif op == "tt.join":
        body = '''%joined = tt.join %v, %onef : tensor<4xf32> -> tensor<4x2xf32>
    %op_out = "tt.reduce"(%joined) <{axis = 1 : i32}> ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        tt.reduce.return %sum : f32
    }) : (tensor<4x2xf32>) -> tensor<4xf32>'''
    else:
        body = '''%matrix = tt.reshape %v : tensor<4xf32> -> tensor<2x2xf32>
    %lhs, %rhs = tt.split %matrix : tensor<2x2xf32> -> tensor<2xf32>
    %sum = arith.addf %lhs, %rhs : tensor<2xf32>
    %joined = tt.join %sum, %rhs : tensor<2xf32> -> tensor<2x2xf32>
    %op_out = tt.reshape %joined : tensor<2x2xf32> -> tensor<4xf32>'''
    if not literal_weights and op in ("tt.expand_dims", "tt.broadcast"):
        body = body.replace(
            "%weights = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>",
            "%plus = arith.addf %v, %onef : tensor<4xf32>\n    "
            "%pair = tt.join %v, %plus : tensor<4xf32> -> tensor<4x2xf32>\n    "
            "%weights = tt.trans %pair {order = array<i32: 1, 0>} : tensor<4x2xf32> -> tensor<2x4xf32>")
    if not literal_weights and op == "tt.reshape":
        body = body.replace(
            "%weights = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>",
            "%ones_mat = arith.constant dense<1.0> : tensor<2x2xf32>\n    "
            "%weights = arith.addf %matrix, %ones_mat : tensor<2x2xf32>")
    return prefix.replace(old, body)


def shape_reference(op, x, blocks):
    import torch
    if op == "tt.unsplat":
        return x[torch.arange(x.numel()) // 4]
    if op in ("tt.expand_dims", "tt.broadcast"):
        return x * x + x * (x + 1)
    if op == "tt.reshape":
        return x * (x + 1)
    padded = torch.cat((x, torch.ones(blocks * 4 - x.numel())))
    if op == "tt.trans":
        return padded.reshape(blocks, 2, 2).transpose(1, 2).reshape(-1)[:x.numel()]
    if op == "tt.join":
        return x + 1
    matrix = padded.reshape(-1, 2)
    return torch.stack((matrix[:, 0] + matrix[:, 1], matrix[:, 1]), dim=-1).reshape(-1)[:x.numel()]
