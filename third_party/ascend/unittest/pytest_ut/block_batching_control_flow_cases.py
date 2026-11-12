# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
CONTROL_KINDS = ("if_join", "if_effects", "if_untaken", "for_accumulate", "for_fixed_point", "if_for", "for_if",
                 "bounded_read_loop", "uniform_load_condition")


def control_ir(kind):
    assert kind in CONTROL_KINDS
    pre = '''module {
  tt.func public @block_batching_control(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32, %k: i32) {
    %p = tt.get_program_id x : i32
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %zero_f = arith.constant 0.0 : f32
    %one_f = arith.constant 1.0 : f32
    %pf = arith.sitofp %p : i32 to f32
    %kf = arith.sitofp %k : i32 to f32
    %condition = arith.cmpi sgt, %k, %zero : i32
    %out = tt.addptr %y, %p : !tt.ptr<f32>, i32
'''
    if kind == "if_join":
        body = '''    %result = scf.if %condition -> (f32) {
      scf.yield %pf : f32
    } else {
      scf.yield %kf : f32
    }
'''
    elif kind == "if_effects":
        body = '''    scf.if %condition {
      %in = tt.addptr %x, %p : !tt.ptr<f32>, i32
      %value = tt.load %in : !tt.ptr<f32>
      tt.store %out, %value : !tt.ptr<f32>
    }
    tt.return
  }
}'''
        return pre + body
    elif kind == "if_untaken":
        body = '''    %result = scf.if %condition -> (f32) {
      scf.yield %pf : f32
    } else {
      %huge = arith.constant 1099511627776 : i64
      %invalid = tt.addptr %x, %huge : !tt.ptr<f32>, i64
      %value = tt.load %invalid : !tt.ptr<f32>
      scf.yield %value : f32
    }
'''
    elif kind == "uniform_load_condition":
        body = '''    %control = tt.load %x : !tt.ptr<f32>
    %loaded_condition = arith.cmpf ogt, %control, %zero_f : f32
    %result = scf.if %loaded_condition -> (f32) {
      scf.yield %pf : f32
    } else {
      scf.yield %kf : f32
    }
'''
    elif kind == "for_fixed_point":
        body = '''    %a, %b = scf.for %iv = %zero to %k step %one iter_args(%left = %zero_f, %right = %one_f) -> (f32, f32) : i32 {
      scf.yield %right, %pf : f32, f32
    }
    %result = arith.addf %a, %b : f32
'''
    elif kind in ("for_accumulate", "for_if", "bounded_read_loop"):
        increment = "%pf"
        extra = ""
        if kind == "for_if":
            extra = '''      %first = arith.cmpi slt, %iv, %one : i32
      %increment = scf.if %first -> (f32) {
        scf.yield %pf : f32
      } else {
        scf.yield %one_f : f32
      }
'''
            increment = "%increment"
        elif kind == "bounded_read_loop":
            extra = '''      %eight = arith.constant 8 : i32
      %base = arith.muli %p, %eight : i32
      %offset = arith.addi %base, %iv : i32
      %in = tt.addptr %x, %offset : !tt.ptr<f32>, i32
      %within = arith.cmpi slt, %iv, %eight : i32
      %value = tt.load %in, %within, %zero_f : !tt.ptr<f32>
'''
            increment = "%value"
        body = f'''    %result = scf.for %iv = %zero to %k step %one iter_args(%sum = %zero_f) -> (f32) : i32 {{
{extra}      %updated = arith.addf %sum, {increment} : f32
      scf.yield %updated : f32
    }}
'''
    else:
        body = '''    %result = scf.if %condition -> (f32) {
      %sum = scf.for %iv = %zero to %k step %one iter_args(%acc = %zero_f) -> (f32) : i32 {
        %ivf = arith.sitofp %iv : i32 to f32
        %increment = arith.addf %pf, %ivf : f32
        %updated = arith.addf %acc, %increment : f32
        scf.yield %updated : f32
      }
      scf.yield %sum : f32
    } else {
      scf.yield %kf : f32
    }
'''
    return pre + body + '''    tt.store %out, %result : !tt.ptr<f32>
    tt.return
  }
}'''


def control_reference(kind, x, blocks, k):
    import torch
    p = torch.arange(blocks, dtype=torch.float32)
    if kind == "if_join": return p if k > 0 else torch.full_like(p, float(k))
    if kind == "if_effects": return x[:blocks] if k > 0 else torch.full_like(p, -37.)
    if kind == "if_untaken": return p
    if kind == "uniform_load_condition": return p if x[0] > 0 else torch.full_like(p, float(k))
    if kind == "for_accumulate": return max(k, 0) * p
    if kind == "for_fixed_point": return torch.ones_like(p) if k <= 0 else 1 + p if k == 1 else 2 * p
    if kind == "if_for": return k * p + k * (k - 1) / 2 if k > 0 else torch.full_like(p, float(k))
    if kind == "for_if": return torch.zeros_like(p) if k <= 0 else p + (k - 1)
    return x.reshape(blocks, 8)[:, :max(0, min(k, 8))].sum(1)
