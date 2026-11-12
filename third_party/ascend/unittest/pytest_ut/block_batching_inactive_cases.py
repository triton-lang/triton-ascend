# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
IEEE_OPS = ('arith.divf', 'arith.remf', 'tt.precise_divf', 'tt.precise_sqrt', 'math.exp', 'math.exp2', 'math.log',
            'math.log2', 'math.sqrt', 'math.rsqrt')
KINDS = ('div_nan_tail', 'log_to_int', 'exp_to_int', 'sibling_lifts')


def inactive_ir(kind):
    assert kind in KINDS or kind in IEEE_OPS
    head = '''module {
  tt.func public @block_batching_memory(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32, %k: i32) {
    %p = tt.get_program_id x : i32
    %g = tt.get_num_programs x : i32
    %four = arith.constant 4 : i32
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %base = arith.muli %p, %four : i32
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %bv = tt.splat %base : i32 -> tensor<4xi32>
    %offset = arith.addi %bv, %range : tensor<4xi32>
    %nv = tt.splat %n : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %offset, %nv : tensor<4xi32>
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %xa = tt.addptr %xp, %offset : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %ya = tt.addptr %yp, %offset : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %zeros = arith.constant dense<0.0> : tensor<4xf32>
    %ones = arith.constant dense<1.0> : tensor<4xf32>
    %v = tt.load %xa, %mask, OTHER : tensor<4x!tt.ptr<f32>>
'''
    head = head.replace('OTHER', '%zeros' if kind in ('div_nan_tail', 'sibling_lifts') else '%ones')
    if kind == 'div_nan_tail':
        body = '    %result = arith.divf %v, %v : tensor<4xf32>\n'
    elif kind in ('log_to_int', 'exp_to_int'):
        if kind == 'log_to_int':
            body = '''    %remaining = arith.subi %g, %p : i32
    %rf = arith.sitofp %remaining : i32 to f32
    %rv = tt.splat %rf : f32 -> tensor<4xf32>
    %term = arith.mulf %rv, %v : tensor<4xf32>
    %fp = math.log %term : tensor<4xf32>
'''
        else:
            body = '''    %distance = arith.subi %p, %g : i32
    %next = arith.addi %distance, %one : i32
    %rf = arith.sitofp %next : i32 to f32
    %rv = tt.splat %rf : f32 -> tensor<4xf32>
    %large = arith.constant dense<1000.0> : tensor<4xf32>
    %scaled = arith.mulf %rv, %large : tensor<4xf32>
    %term = arith.mulf %scaled, %v : tensor<4xf32>
    %fp = math.exp %term : tensor<4xf32>
'''
        body += '''    %integer = arith.fptosi %fp : tensor<4xf32> to tensor<4xi32>
    %result = arith.sitofp %integer : tensor<4xi32> to tensor<4xf32>
'''
    elif kind == 'sibling_lifts':
        body = '''    %condition = arith.cmpi sgt, %k, %zero : i32
    %joined = scf.if %condition -> (tensor<4xf32>) {
      %a = arith.addf %v, %ones : tensor<4xf32>
      %b = arith.mulf %a, %ones : tensor<4xf32>
      scf.yield %b : tensor<4xf32>
    } else {
      %c = arith.subf %v, %ones : tensor<4xf32>
      %d = arith.mulf %c, %ones : tensor<4xf32>
      scf.yield %d : tensor<4xf32>
    }
    %result = arith.addf %joined, %ones : tensor<4xf32>
'''
    else:
        inputs = '%v, %v' if kind in ('arith.divf', 'arith.remf', 'tt.precise_divf') else '%v'
        types = 'tensor<4xf32>, tensor<4xf32>' if ', ' in inputs else 'tensor<4xf32>'
        body = f'    %result = "{kind}"({inputs}) : ({types}) -> tensor<4xf32>\n'
    return head + body + '''    tt.store %ya, %result, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}'''
