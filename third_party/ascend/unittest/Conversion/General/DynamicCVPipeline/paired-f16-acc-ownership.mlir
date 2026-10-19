// RUN: %PYTHON %S/Inputs/paired-f16-cases.py %S/Inputs/paired-f16-recurrence.mlir %t
// RUN: triton-opt %t/positive.mlir --ssbuf-standardize-op --verify-each -o %t/positive.once
// RUN: FileCheck %s --check-prefix=NO-DEST --implicit-check-not=ascend.destination_ --implicit-check-not=ascend.online_softmax_score_candidate < %t/positive.once
// RUN: FileCheck %s < %t/positive.once
// RUN: triton-opt %t/positive.mlir --ssbuf-standardize-op --verify-each --emit-bytecode -o %t/positive.bc
// RUN: triton-opt %t/positive.bc --ssbuf-standardize-op --verify-each -o %t/positive.twice
// RUN: diff %t/positive.once %t/positive.twice
// RUN: triton-opt %t/negative.mlir --ssbuf-standardize-op-pattern-match --verify-each -o %t/negative.base
// RUN: triton-opt %t/negative.mlir --ssbuf-standardize-op --verify-each -o %t/negative.pass
// RUN: diff %t/negative.base %t/negative.pass
// RUN: triton-opt %t/mixed.mlir --ssbuf-standardize-op --verify-each | FileCheck %s --check-prefix=MIXED --implicit-check-not=ascend.destination_ --implicit-check-not=ascend.online_softmax_score_candidate
// RUN: triton-opt %t/fallback.mlir -o %t/fallback.base
// RUN: triton-opt %t/fallback.mlir --ssbuf-standardize-op --verify-each -o %t/fallback.pass
// RUN: diff %t/fallback.base %t/fallback.pass
// RUN: triton-opt %t/positive.bc '--add_dynamic_cv_pipeline=compile-on-910-95=False' --verify-each | FileCheck %s --check-prefix=NO-DEST --implicit-check-not=ascend.online_softmax_old_max_read

// NO-DEST: module
// CHECK-LABEL: func.func @plain(
// CHECK: arith.extf {{.*}}ascend.online_softmax_old_max_read
// CHECK: hivm.hir.custom
// CHECK: arith.mulf {{.*}}
// CHECK: arith.addf {{.*}}
// CHECK: arith.divf {{.*}}
// CHECK-LABEL: func.func @causal(
// CHECK: arith.extf {{.*}}ascend.online_softmax_old_max_read
// CHECK: hivm.hir.custom
// CHECK: arith.mulf {{.*}}
// CHECK: arith.addf {{.*}}
// CHECK: arith.divf {{.*}}
// CHECK-LABEL: func.func @causal_if(
// CHECK: arith.extf {{.*}}ascend.online_softmax_old_max_read
// CHECK: hivm.hir.custom
// CHECK: arith.mulf {{.*}}
// CHECK: arith.addf {{.*}}
// CHECK: arith.divf {{.*}}
// CHECK-LABEL: func.func @reordered(
// CHECK: arith.extf {{.*}}ascend.online_softmax_old_max_read
// CHECK: hivm.hir.custom
// CHECK: arith.mulf {{.*}}
// CHECK: arith.addf {{.*}}
// CHECK: arith.divf {{.*}}
// CHECK-LABEL: func.func @extra_state(
// CHECK: arith.extf {{.*}}ascend.online_softmax_old_max_read
// CHECK: hivm.hir.custom
// CHECK: arith.mulf {{.*}}
// CHECK: arith.addf {{.*}}
// CHECK: arith.divf {{.*}}
// CHECK-LABEL: func.func @commuted(
// CHECK: arith.extf {{.*}}ascend.online_softmax_old_max_read
// CHECK: hivm.hir.custom
// CHECK: arith.mulf {{.*}}
// CHECK: arith.addf {{.*}}
// CHECK: arith.divf {{.*}}
// CHECK-LABEL: func.func @head64(
// CHECK: arith.extf {{.*}}ascend.online_softmax_old_max_read
// CHECK: hivm.hir.custom
// CHECK: arith.mulf {{.*}} : tensor<128x64xf32>
// CHECK: arith.addf {{.*}} : tensor<128x64xf32>
// CHECK: arith.divf {{.*}} : tensor<128x64xf32>
// CHECK-LABEL: func.func @unrelated_work(
// CHECK: memref.load
// CHECK: arith.extf {{.*}}ascend.online_softmax_old_max_read
// CHECK: hivm.hir.custom
// CHECK: arith.mulf {{.*}}
// CHECK: arith.addf {{.*}}
// CHECK: arith.divf {{.*}}
// CHECK-LABEL: func.func @two_loops(
// CHECK: arith.extf {{.*}}ascend.online_softmax_old_max_read
// CHECK: hivm.hir.custom
// CHECK: arith.mulf {{.*}}
// CHECK: arith.addf {{.*}}
// CHECK: arith.divf {{.*}}
// CHECK: arith.extf {{.*}}ascend.online_softmax_old_max_read
// CHECK: hivm.hir.custom
// CHECK: arith.mulf {{.*}}
// CHECK: arith.addf {{.*}}
// CHECK: arith.divf {{.*}}

// MIXED-LABEL: func.func @matched(
// MIXED: ascend.online_softmax_old_max_read
// MIXED: hivm.hir.custom
// MIXED-LABEL: func.func @bad_old_m(
// MIXED-NOT: ascend.online_softmax_
// MIXED: return
