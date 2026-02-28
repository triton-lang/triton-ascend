#ifndef TRITON_TO_STRUCTURED_PASSES
#define TRITON_TO_STRUCTURED_PASSES

include "mlir/Pass/PassBase.td"

def TritonToStructured : Pass<"triton-to-structured", "mlir::ModuleOp"> {
    let summary = "remove reminder/divider and reproduce addptr/mask expression ";
    let constructor = "triton::createTritonToStructuredPass()";
    let options = [
        Option<"enableMaskFallbackConversion", "enable-mask-fallback-conversion",
            "bool", /*default*/"false",
            "If enabled, select will perform a fallback conversion when mask matching fails.">,
        Option<"optimizeDynamicOffset", "optimize-dynamic-offset",
            "bool", /*default*/"false",
            "Enable dynamic offset feature">,
        Option<"compileOn91095", "compile-on-910-95",
            "bool", /*default*/"false",
            "compile on 910_95">
    ];
}

#endif // TRITON_TO_STRUCTURED_PASSES
