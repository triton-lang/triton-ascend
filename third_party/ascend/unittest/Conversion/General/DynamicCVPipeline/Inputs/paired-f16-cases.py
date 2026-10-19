# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the MIT license.

from pathlib import Path
import re
import sys


source = Path(sys.argv[1]).read_text()
prefix, function = source.split("  func.func ", 1)
function = "  func.func " + function.rsplit("\n}", 1)[0]
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)


def change(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def case(name, *replacements):
    text = change(function, "@_attn_fwd_paired_f16(", "@" + name + "(")
    for old, new in replacements:
        text = change(text, old, new)
    return text


def write(name, functions, module_prefix=prefix):
    (output / (name + ".mlir")).write_text(module_prefix + "\n".join(functions) + "\n}\n")


state = "iter_args(%arg18 = %9, %arg19 = %11, %arg20 = %13) -> (tensor<128xf16>, tensor<128xf32>, tensor<128x128xf32>)"
yield_state = "scf.yield %78#1, %84, %96 : tensor<128xf16>, tensor<128xf32>, tensor<128x128xf32>"
custom_inputs = "ins(%77, %arg18, %23 : tensor<32x8x16x16xf16>, tensor<128xf16>, tensor<256xi8>)"
causal_inputs = "ins(%77, %arg18, %23, %limits : tensor<32x8x16x16xf16>, tensor<128xf16>, tensor<256xi8>, tensor<128xi32>)"

causal = case("causal", ("    %c32768 =", "    %limits = arith.constant dense<511> : tensor<128xi32>\n    %c32768 ="),
              (custom_inputs, causal_inputs))
causal_if = change(causal, "@causal(", "@causal_if(")
causal_if = change(causal_if, "    %c32768 =", "    %mask = arith.constant false\n    %c32768 =")
causal_if = change(causal_if, causal_inputs, causal_inputs.replace("%limits :", "%limits, %mask :").replace("tensor<128xi32>)", "tensor<128xi32>, i1)"))

reordered = case("reordered", (state, "iter_args(%arg20 = %13, %arg18 = %9, %arg19 = %11) -> (tensor<128x128xf32>, tensor<128xf16>, tensor<128xf32>)"),
                 (yield_state, "scf.yield %96, %78#1, %84 : tensor<128x128xf32>, tensor<128xf16>, tensor<128xf32>"))
reordered = re.sub(r"%43#([012])", lambda m: "%43#" + {"0": "1", "1": "2", "2": "0"}[m[1]], reordered)
extra_state = case("extra_state", ("%43:3 =", "%43:4 ="),
                   (state, state.replace("%arg20 = %13)", "%arg20 = %13, %aux = %c0_i32)").replace("tensor<128x128xf32>)", "tensor<128x128xf32>, i32)")),
                   (yield_state, yield_state.replace("%96 :", "%96, %aux :") + ", i32"))
commuted = case("commuted", ("arith.mulf %arg19, %82", "arith.mulf %82, %arg19"),
                ("arith.addf %83, %78#2", "arith.addf %78#2, %83"),
                ("arith.mulf %87, %93", "arith.mulf %93, %87"),
                ("arith.mulf %arg20, %broadcasted_22", "arith.mulf %broadcasted_22, %arg20"),
                ("arith.addf %95, %94", "arith.addf %94, %95"))
head64 = case("head64", ("%arg8 to offset: [%36], sizes: [128, 128]", "%arg8 to offset: [%36], sizes: [128, 64]"),
              ("sizes: [512, 128]", "sizes: [512, 64]"))
for old, new in [("128x128xf32", "128x64xf32"), ("512x128xf8E4M3FN", "512x64xf8E4M3FN")]:
    head64 = head64.replace(old, new)
unrelated = case("unrelated_work", ("        %52 =", "        %unrelated = memref.load %arg0[%c0] : memref<?xi8>\n        %unrelated_tensor = arith.addf %13, %13 : tensor<128x128xf32>\n        %52 ="))
two_loops = case("two_loops")
start = two_loops.index("    scf.for %arg16")
end = two_loops.rindex("    return")
two_loops = two_loops[:end] + two_loops[start:end] + two_loops[end:]
write("positive", [case("plain"), causal, causal_if, reordered, extra_state, commuted, head64, unrelated, two_loops])
write("fallback", [case("preexisting_fallback")],
      change(prefix, "module attributes {", "module attributes {triton_ascend.dynamic_cv_pipeline.rc = 2 : i32, "))

# Unmatched preparation adds no changes beyond the existing generic rewrites.
negative = [
    case("bad_old_m", (custom_inputs, custom_inputs.replace("%arg18", "%9"))),
    case("bad_m_yield", (yield_state, yield_state.replace("%78#1", "%arg18"))),
    case("bad_l_yield", (yield_state, yield_state.replace("%84", "%arg19"))),
    case("bad_acc_yield", (yield_state, yield_state.replace("%96", "%arg20"))),
    case("bad_l_addend", ("arith.addf %83, %78#2", "arith.addf %83, %arg19")),
    case("bad_old_max", ("arith.extf %arg18", "arith.extf %78#1")),
    case("bad_acc_addend", ("arith.addf %95, %94", "arith.addf %95, %95")),
    case("bad_normalize", ("linalg.broadcast ins(%43#1 :", "linalg.broadcast ins(%11 :")),
    case("bad_alpha_broadcast", ("dimensions = [1]\n        %95", "dimensions = [0]\n        %95")),
    case("bad_l_broadcast", ("dimensions = [1]\n      %51", "dimensions = [0]\n      %51")),
    case("extra_old_m", ("        %79 =", "        %observed = tensor.extract %arg18[%c0] : tensor<128xf16>\n        %79 =")),
    case("extra_old_l", ("        %83 =", "        %observed = tensor.extract %arg19[%c0] : tensor<128xf32>\n        %83 =")),
    case("extra_acc", ("        %95 =", "        %observed = tensor.extract %arg20[%c0, %c0] : tensor<128x128xf32>\n        %95 =")),
    case("extra_p", ("        %79 =", "        %observed = tensor.extract %78#0[%c0, %c0, %c0, %c0] : tensor<16x8x16x32xf8E4M3FN>\n        %79 =")),
    case("extra_alpha", ("        %83 =", "        %observed = tensor.extract %82[%c0] : tensor<128xf32>\n        %83 =")),
    case("extra_normalize", ("      bufferization.materialize_in_destination %51", "      %observed = tensor.extract %51[%c0, %c0] : tensor<128x128xf32>\n      bufferization.materialize_in_destination %51")),
    case("uninitialized_pv", ("outs(%13 : tensor<128x128xf32>)", "outs(%12 : tensor<128x128xf32>)")),
    case("unrelated_p", ("        %87 =", "        %other_p = tensor.empty() : tensor<128x512xf8E4M3FN>\n        %87 ="),
         ("ins(%86, %85 :", "ins(%other_p, %85 :")),
    case("ordinary_custom", ('symbol = "__builtin_online_softmax_nz"', 'symbol = "ordinary_custom"'),
         ('"__builtin_online_softmax_nz" ins(', '"ordinary_custom" ins(')),
]
duplicate = case("two_customs")
line = next(line for line in duplicate.splitlines(keepends=True) if "%78:3 = hivm.hir.custom" in line)
duplicate = change(duplicate, line, line + line.replace("%78:3", "%duplicate:3"))
negative.append(duplicate)
write("negative", negative)
write("mixed", [case("matched"), negative[0]])
