# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Observable TTIR fixtures for the W1/W2 elementwise models.

Each fixture stores the selected operation's result, so the operation cannot
be counted merely because a dead instruction appeared in the input text.
"""
W1_OPS = (
    "tt.bitcast",
    "tt.clampf",
    "tt.mulhiui",
    "arith.xori",
    "arith.maxsi",
    "arith.maxui",
    "arith.minsi",
    "arith.minui",
    "arith.remf",
    "arith.maximumf",
    "arith.minimumf",
    "arith.maxnumf",
    "arith.minnumf",
    "arith.cmpf",
    "arith.uitofp",
    "arith.bitcast",
    "math.absf",
    "math.absi",
    "math.ceil",
    "math.floor",
    "math.log",
    "math.log2",
    "math.sin",
    "math.cos",
    "math.erf",
)
W2_OPS = (
    "arith.divsi",
    "arith.divui",
    "arith.ceildivsi",
    "arith.ceildivui",
    "arith.floordivsi",
    "arith.remsi",
    "arith.remui",
    "arith.shli",
    "arith.shrsi",
    "arith.shrui",
    "arith.fptosi",
    "arith.fptoui",
    "ascend.mod",
)
ELEMENTWISE_OPS = W1_OPS + W2_OPS
BASIC_COMPLETION_OPS = ("arith.andi", "arith.extui", "arith.negf", "arith.ori", "arith.trunci", "math.exp2", "math.fma",
                        "math.rsqrt", "tt.fp_to_fp", "tt.precise_divf", "tt.precise_sqrt")


def case_ir(op, variant="VV"):
    """variant controls the second operand for U/V and V/V cases."""
    assert op in ELEMENTWISE_OPS + BASIC_COMPLETION_OPS and variant in ("UV", "VV", "SCALAR_RHS")
    f = "tensor<4xf32>"
    i = "tensor<4xi32>"
    scalar_rhs = "%twof" if variant == "UV" else "%denomf"
    if op in ("arith.remf", "ascend.mod") and variant == "VV":
        scalar_rhs = "%element_rhs"
    integer_rhs = "%twoi" if variant == "UV" else "%denom"
    lines = []
    if op in ("arith.remf", "ascend.mod") and variant == "VV":
        # Keep both operands element-varying. The standard CANN 9.1 compiler
        # currently rejects the scalar-broadcast RHS in fmod's vector-thread
        # ABI; SCALAR_RHS retains that original backend reproducer.
        lines += [
            f"%half = arith.constant dense<0.5> : {f}", f"%half_v = arith.mulf %v, %half : {f}",
            f"%element_rhs = arith.subf %twof, %half_v : {f}"
        ]
    output_type = f
    if op in ("arith.andi", "arith.ori"):
        lines.append(f'%op_out = "{op}"(%iv, %twoi) : ({i}, {i}) -> {i}')
        output_type = i
    elif op == "arith.extui":
        lines += [
            f"%boolean = arith.cmpf olt, %v, %twof : {f}", f"%op_out = arith.extui %boolean : tensor<4xi1> to {i}"
        ]
        output_type = i
    elif op == "arith.trunci":
        lines += [
            f"%wide = arith.extsi %iv : {i} to tensor<4xi64>",
            "%high = arith.constant dense<4294967296> : tensor<4xi64>",
            "%sum = arith.addi %wide, %high : tensor<4xi64>", f"%op_out = arith.trunci %sum : tensor<4xi64> to {i}"
        ]
        output_type = i
    elif op == "arith.negf":
        lines.append(f"%op_out = arith.negf %v : {f}")
    elif op == "math.fma":
        lines.append(f"%op_out = math.fma %v, %twof, %onef : {f}")
    elif op == "tt.fp_to_fp":
        lines += [
            f"%half = tt.fp_to_fp %v, rounding = rtne : {f} -> tensor<4xf16>",
            f"%op_out = tt.fp_to_fp %half : tensor<4xf16> -> {f}"
        ]
    elif op == "tt.precise_divf":
        lines.append(f"%op_out = tt.precise_divf %v, %denomf : {f}")
    elif op == "tt.precise_sqrt":
        lines.append(f"%op_out = tt.precise_sqrt %v : {f}")
    elif op in ("tt.bitcast", "arith.bitcast"):
        lines.append(f'%op_out = "{op}"(%v) : ({f}) -> {i}')
        output_type = i
    elif op == "tt.clampf":
        lines.append(f"%op_out = tt.clampf %v, %onef, %twof, propagateNan = none : {f}")
    elif op == "tt.mulhiui":
        lines.append(f'%op_out = "tt.mulhiui"(%iv, %big) : ({i}, {i}) -> {i}')
        output_type = i
    elif op in ("arith.xori", "arith.maxsi", "arith.maxui", "arith.minsi", "arith.minui"):
        lines.append(f'%op_out = "{op}"(%signed, {integer_rhs}) : ({i}, {i}) -> {i}')
        output_type = i
    elif op in ("arith.remf", "arith.maximumf", "arith.minimumf", "arith.maxnumf", "arith.minnumf", "ascend.mod"):
        lines.append(f'%op_out = "{op}"(%v, {scalar_rhs}) : ({f}, {f}) -> {f}')
    elif op == "arith.cmpf":
        lines += [
            f"%predicate = arith.cmpf olt, %v, {scalar_rhs} : {f}",
            f"%op_out = arith.select %predicate, %v, %onef : tensor<4xi1>, {f}"
        ]
    elif op == "arith.uitofp":
        lines.append(f"%op_out = arith.uitofp %signed : {i} to {f}")
    elif op == "math.absi":
        lines.append(f"%op_out = math.absi %signed : {i}")
        output_type = i
    elif op.startswith("math."):
        lines.append(f'%op_out = "{op}"(%v) : ({f}) -> {f}')
    elif op in ("arith.divsi", "arith.divui", "arith.ceildivsi", "arith.ceildivui", "arith.floordivsi", "arith.remsi",
                "arith.remui"):
        lines.append(f'%op_out = "{op}"(%signed, {integer_rhs}) : ({i}, {i}) -> {i}')
        output_type = i
    elif op in ("arith.shli", "arith.shrsi", "arith.shrui"):
        lines.append(f'%op_out = "{op}"(%signed, %shift) : ({i}, {i}) -> {i}')
        output_type = i
    elif op in ("arith.fptosi", "arith.fptoui"):
        lines.append(f"%op_out = {op} %v : {f} to {i}")
        output_type = i
    else:
        raise AssertionError(op)
    result = "%op_out"
    if output_type == i:
        lines.append(f"%as_float = arith.sitofp %op_out : {i} to {f}")
        result = "%as_float"
    body = "\n    ".join(lines)
    return f'''module {{
  tt.func public @block_batching_elementwise(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32) {{
    %p = tt.get_program_id x : i32
    %g = tt.get_num_programs x : i32
    %b = arith.constant 4 : i32
    %base = arith.muli %p, %b : i32
    %range = tt.make_range {{start = 0 : i32, end = 4 : i32}} : {i}
    %basev = tt.splat %base : i32 -> {i}
    %idx = arith.addi %basev, %range : {i}
    %nv = tt.splat %n : i32 -> {i}
    %mask = arith.cmpi slt, %idx, %nv : {i}
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %xa = tt.addptr %xp, %idx : tensor<4x!tt.ptr<f32>>, {i}
    %ya = tt.addptr %yp, %idx : tensor<4x!tt.ptr<f32>>, {i}
    %onef = arith.constant dense<1.0> : {f}
    %twof = arith.constant dense<2.0> : {f}
    %onei = arith.constant dense<1> : {i}
    %twoi = arith.constant dense<2> : {i}
    %fouri = arith.constant dense<4> : {i}
    %thirtytwo = arith.constant dense<32> : {i}
    %big = arith.constant dense<1073741824> : {i}
    %v = tt.load %xa, %mask, %onef : tensor<4x!tt.ptr<f32>>
    %pv = tt.splat %p : i32 -> {i}
    %gv = tt.splat %g : i32 -> {i}
    %iv = arith.addi %pv, %onei : {i}
    %signed = arith.subi %iv, %fouri : {i}
    %denom = arith.subi %gv, %pv : {i}
    %denomf = arith.sitofp %denom : {i} to {f}
    %shift = arith.subi %thirtytwo, %denom : {i}
    {body}
    tt.store %ya, {result}, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }}
}}'''


def reference(op, x, blocks, variant="VV"):
    import torch
    p = torch.arange(x.numel(), device=x.device, dtype=torch.int64) // 4
    signed = p - 3
    unsigned = signed & 0xffffffff
    rhs_i = torch.full_like(p, 2) if variant == "UV" else blocks - p
    rhs_f = rhs_i.to(torch.float32)
    if op == "arith.andi": return ((p + 1) & 2).float()
    if op == "arith.ori": return ((p + 1) | 2).float()
    if op == "arith.extui": return (x < 2).float()
    if op == "arith.trunci": return (p + 1).float()
    if op == "arith.negf": return -x
    if op == "math.fma": return x * 2 + 1
    if op == "tt.fp_to_fp": return x.half().float()
    if op == "tt.precise_divf": return x / rhs_f
    if op == "tt.precise_sqrt": return x.sqrt()
    if op in ("tt.bitcast", "arith.bitcast"):
        return x.view(torch.int32).float()
    if op == "tt.clampf":
        return x.clamp(1, 2)
    if op == "tt.mulhiui":
        out = ((p + 1) * (1 << 30)) >> 32
    elif op == "arith.xori":
        out = signed ^ rhs_i
    elif op == "arith.maxsi":
        out = torch.maximum(signed, rhs_i)
    elif op == "arith.minsi":
        out = torch.minimum(signed, rhs_i)
    elif op == "arith.maxui":
        out = torch.maximum(unsigned, rhs_i)
    elif op == "arith.minui":
        out = torch.minimum(unsigned, rhs_i)
    elif op in ("arith.remf", "ascend.mod"):
        return torch.fmod(x, 2 - .5 * x if variant == "VV" else rhs_f)
    elif op in ("arith.maximumf", "arith.maxnumf"):
        return torch.maximum(x, rhs_f)
    elif op in ("arith.minimumf", "arith.minnumf"):
        return torch.minimum(x, rhs_f)
    elif op == "arith.cmpf":
        return torch.where(x < rhs_f, x, torch.ones_like(x))
    elif op == "arith.uitofp":
        return unsigned.float()
    elif op == "math.absi":
        out = signed.abs()
    elif op.startswith("math."):
        fn = {
            "math.absf": torch.abs, "math.ceil": torch.ceil, "math.floor": torch.floor, "math.log": torch.log,
            "math.log2": torch.log2, "math.sin": torch.sin, "math.cos": torch.cos, "math.erf": torch.erf, "math.exp2":
            torch.exp2, "math.rsqrt": torch.rsqrt
        }[op]
        return fn(x)
    elif op == "arith.divsi":
        out = torch.div(signed, rhs_i, rounding_mode="trunc")
    elif op == "arith.divui":
        out = unsigned // rhs_i
    elif op == "arith.ceildivsi":
        out = -torch.div(-signed, rhs_i, rounding_mode="floor")
    elif op == "arith.ceildivui":
        out = (unsigned + rhs_i - 1) // rhs_i
    elif op == "arith.floordivsi":
        out = signed // rhs_i
    elif op == "arith.remsi":
        out = signed - torch.div(signed, rhs_i, rounding_mode="trunc") * rhs_i
    elif op == "arith.remui":
        out = unsigned % rhs_i
    elif op == "arith.shli":
        out = signed << (32 - (blocks - p))
    elif op == "arith.shrsi":
        out = signed >> (32 - (blocks - p))
    elif op == "arith.shrui":
        out = unsigned >> (32 - (blocks - p))
    elif op in ("arith.fptosi", "arith.fptoui"):
        return x.trunc().float()
    else:
        raise AssertionError(op)
    return out.to(torch.int32).float()
