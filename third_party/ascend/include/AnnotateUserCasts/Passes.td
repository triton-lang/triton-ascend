/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef ANNOTATE_USER_CASTS_CONVERSION_PASSES
#define ANNOTATE_USER_CASTS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AnnotateUserCasts : Pass<"annotate-user-casts", "mlir::ModuleOp"> {
  let summary = "Mark cast operations written by the user with an annotation to enable backend optimization of cast chains";
  let constructor = "triton::createAnnotateUserCastsPass()";
  let dependentDialects = [
    "annotation::AnnotationDialect",
    "triton::TritonDialect",
    "arith::ArithDialect"
  ];
}

#endif // ANNOTATE_USER_CASTS_CONVERSION_PASSES