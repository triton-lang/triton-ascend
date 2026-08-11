#ifndef TRITON_TO_LINALG_CONVERSION_PASSES
#define TRITON_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToLinalg : Pass<"triton-to-linalg", "mlir::ModuleOp"> {
    let summary = "Convert Triton to Linalg dialect";
    let constructor = "triton::createTritonToLinalgPass()";
    let options = [
        Option<"globalKernel", "global-kernel",
            "bool", /*default*/"true",
            "generate a global kernel">,
        Option<"namedOps", "named-ops",
            "bool", /*default*/"false",
            "use linalg named ops instead of linalg.generic">,
        Option<"enableNd2nzOnVector", "enable-nd2nz-on-vector",
            "bool", /*default*/"false",
            "enable nd2nz on vector">,
        Option<"enableSelectAnalysis", "enable-select-analysis",
            "bool", /*default*/"true",
            "enable select analysis">,
        Option<"compileOn91095", "compile-on-910-95",
            "bool", /*default*/"false",
            "compile on 910_95">
    ];
}

def MarkTensorKind : Pass<"mark-tensor-kind", "mlir::ModuleOp"> {
    let summary = "Mark tensor kind (INPUT/OUTPUT/INPUT_OUTPUT) on Triton function arguments";
    let constructor = "triton::createMarkTensorKindPass()";
}

def ConvertDescriptorOps : Pass<"convert-descriptor-ops", "mlir::ModuleOp"> {
    let summary = "Lower Triton tensor descriptor operations to plain Triton IR";
    let description = [{
        Converts `tt.descriptor_load`, `tt.descriptor_store`, `tt.descriptor_gather`,
        `tt.descriptor_scatter`, and `tt.descriptor_reduce` operations (which operate
        on `!tt.tensordesc` values produced by `tt.make_tensor_descriptor`) into
        equivalent Triton block-pointer, `tt.load`/`tt.store`, and `scf.for` based
        IR that no longer references `!tt.tensordesc`.
    }];
    let constructor = "triton::createConvertDescriptorOpsPass()";
}
#endif // TRITON_TO_LINALG_CONVERSION_PASSES
