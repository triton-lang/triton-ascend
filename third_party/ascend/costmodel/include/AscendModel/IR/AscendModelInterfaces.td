//===- AscendModelInterfaces.td - AscendModel Op Interfaces ----*- tablegen -*-===//
//
// Defines operation interfaces for the AscendModel dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MODEL_INTERFACES_TD
#define ASCEND_MODEL_INTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// EstimateCyclesOpInterface
//===----------------------------------------------------------------------===//

def EstimateCyclesOpInterface : OpInterface<"EstimateCyclesOpInterface"> {
  let description = [{
    Interface for operations that can estimate their execution cycles
    on Ascend NPU hardware.
    
    Operations implementing this interface provide:
    - Cycle estimation based on hardware configuration
    - Hardware unit assignment (Cube, Vector, MTE, etc.)
    - Transfer bytes (for memory operations)
    - FLOP count (for compute operations)
    
    Example implementation:
    ```cpp
    int64_t MyOp::estimateCycles(const HardwareConfig &config) {
      int64_t numElements = getNumElements();
      return utils::estimateVectorCycles(numElements, 1, 16, config.getVectorStartupLatency());
    }
    ```
  }];
  
  let cppNamespace = "::mlir::ascend";
  
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Estimate the execution cycles for this operation given hardware config.
        Returns 0 if cycles cannot be estimated.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"estimateCycles",
      /*args=*/(ins "const HardwareConfig &":$config)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the hardware unit this operation executes on.
      }],
      /*retTy=*/"HWUnit",
      /*methodName=*/"getHWUnit",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the number of bytes transferred (for memory operations).
        Returns 0 for non-memory operations.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getTransferBytes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return 0;"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the number of floating-point operations.
        Returns 0 for non-compute operations.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getFlops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return 0;"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the cycles per vector operation (for vector ops).
        1 for simple ops (add, mul), 2-4 for complex ops (exp, log).
      }],
      /*retTy=*/"int",
      /*methodName=*/"getCyclesPerVectorOp",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return 1;"
    >
  ];
  
  let extraClassDeclaration = [{
    /// Helper to get element bit width from first operand.
    int getElementBits();
    
    /// Helper to get number of elements from first operand.
    int64_t getNumElements();
  }];
}

#endif // ASCEND_MODEL_INTERFACES_TD
