//===- AscendModelBase.td - Base definitions for AscendModel dialect ------===//
//
// Defines the AscendModel dialect for Ascend 910B hardware modeling.
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MODEL_BASE_TD
#define ASCEND_MODEL_BASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// AscendModel Dialect Definition
//===----------------------------------------------------------------------===//

def AscendModel_Dialect : Dialect {
  let name = "ascend";
  let summary = "Ascend 910B Hardware Modeling Dialect";
  let description = [{
    This dialect provides operations for modeling Ascend 910B NPU hardware
    behavior and performance characteristics. It supports:

    - Cube Core operations (matrix multiply)
    - Vector Core operations (elementwise, reduction, etc.)
    - Memory transfer operations (MTE2, MTE3, FixPipe)
    - Pipeline modeling and scheduling

    The dialect is designed for performance analysis and roofline modeling.
  }];

  let cppNamespace = "::mlir::ascend";
}

//===----------------------------------------------------------------------===//
// Hardware Unit Enum
//===----------------------------------------------------------------------===//

def HWUnit_Cube      : I32EnumAttrCase<"Cube", 0, "cube">;
def HWUnit_CubeMTE2  : I32EnumAttrCase<"CubeMTE2", 1, "cube_mte2">;
def HWUnit_FixPipe   : I32EnumAttrCase<"FixPipe", 2, "fixpipe">;
def HWUnit_Vector    : I32EnumAttrCase<"Vector", 3, "vector">;
def HWUnit_VecMTE2   : I32EnumAttrCase<"VecMTE2", 4, "vec_mte2">;
def HWUnit_MTE3      : I32EnumAttrCase<"MTE3", 5, "mte3">;
def HWUnit_Scalar    : I32EnumAttrCase<"Scalar", 6, "scalar">;
// This target drains the Cube accumulator straight into UB instead of going out to
// HBM and back. It is a separate engine from the HBM write-back, so it gets
// its own unit; on a part where the two share one pipe, declaring them a mutex
// clique in the hardware profile serialises them without a code change.
// Appended rather than inserted so the existing values stay stable.
def HWUnit_FixPipeUB : I32EnumAttrCase<"FixPipeUB", 7, "fixpipe_ub">;
// On-chip staging between UB and L1, which this target can do without a round
// trip through HBM or L2. These are movers in their own right, not a mode of
// the off-chip ones: UB can be draining into L1 at the same time as it streams
// out to L2, so they overlap that traffic rather than queue behind it. Keeping
// them separate also lets a report tell on-chip staging apart from real
// off-chip traffic, which is the difference that matters when reading where a
// kernel spends its time. A part where they do share a pipe is expressed by a
// mutex clique in the hardware profile, without a code change.
def HWUnit_MTE3ToL1  : I32EnumAttrCase<"MTE3ToL1", 8, "mte3_ub_l1">;
def HWUnit_MTE1ToUB  : I32EnumAttrCase<"MTE1ToUB", 9, "mte1_l1_ub">;
// Staging the Cube's operands from L1 into L0A/L0B. Distinct from CubeMTE2,
// which brings them into L1 from off chip: the profiler reports the two
// separately as aic_mte1_time and aic_mte2_time, and on a matmul kernel the
// first was a third of the Cube's busy time while nothing charged it at all.
def HWUnit_CubeMTE1  : I32EnumAttrCase<"CubeMTE1", 10, "cube_mte1">;
// Scalar issue on the Cube core. Scalar work does not belong to one core --
// each has its own scalar unit and the profiler reports aic_scalar_time and
// aiv_scalar_time apart -- so a single Scalar unit would put both cores'
// bookkeeping on one pipe and serialise blocks that in fact run at the same
// time. Scalar keeps its meaning as the Vector core's; this is the Cube's.
def HWUnit_ScalarCube : I32EnumAttrCase<"ScalarCube", 11, "scalar_cube">;

def HWUnitAttr : I32EnumAttr<"HWUnit", "Hardware execution unit", [
  HWUnit_Cube, HWUnit_CubeMTE2, HWUnit_FixPipe,
  HWUnit_Vector, HWUnit_VecMTE2, HWUnit_MTE3, HWUnit_Scalar,
  HWUnit_FixPipeUB, HWUnit_MTE3ToL1, HWUnit_MTE1ToUB,
  HWUnit_CubeMTE1, HWUnit_ScalarCube
]> {
  let cppNamespace = "::mlir::ascend";
}

//===----------------------------------------------------------------------===//
// Memory Space Enum
//===----------------------------------------------------------------------===//

def MemSpace_HBM : I32EnumAttrCase<"HBM", 0, "hbm">;
def MemSpace_L2  : I32EnumAttrCase<"L2", 1, "l2">;
def MemSpace_L1  : I32EnumAttrCase<"L1", 2, "l1">;
def MemSpace_UB  : I32EnumAttrCase<"UB", 3, "ub">;  // Unified Buffer

def MemSpaceAttr : I32EnumAttr<"MemSpace", "Memory space", [
  MemSpace_HBM, MemSpace_L2, MemSpace_L1, MemSpace_UB
]> {
  let cppNamespace = "::mlir::ascend";
}

//===----------------------------------------------------------------------===//
// Data Type for Operations
//===----------------------------------------------------------------------===//

def DataType_FP32 : I32EnumAttrCase<"FP32", 0, "fp32">;
def DataType_FP16 : I32EnumAttrCase<"FP16", 1, "fp16">;
def DataType_BF16 : I32EnumAttrCase<"BF16", 2, "bf16">;
def DataType_INT8 : I32EnumAttrCase<"INT8", 3, "int8">;
def DataType_INT32 : I32EnumAttrCase<"INT32", 4, "int32">;

def DataTypeAttr : I32EnumAttr<"DataType", "Data type for operations", [
  DataType_FP32, DataType_FP16, DataType_BF16, DataType_INT8, DataType_INT32
]> {
  let cppNamespace = "::mlir::ascend";
}

//===----------------------------------------------------------------------===//
// Vector Operation Kind
//===----------------------------------------------------------------------===//

def VecOpKind_Add    : I32EnumAttrCase<"Add", 0, "add">;
def VecOpKind_Sub    : I32EnumAttrCase<"Sub", 1, "sub">;
def VecOpKind_Mul    : I32EnumAttrCase<"Mul", 2, "mul">;
def VecOpKind_Div    : I32EnumAttrCase<"Div", 3, "div">;
def VecOpKind_Exp    : I32EnumAttrCase<"Exp", 4, "exp">;
def VecOpKind_Log    : I32EnumAttrCase<"Log", 5, "log">;
def VecOpKind_Max    : I32EnumAttrCase<"Max", 6, "max">;
def VecOpKind_Min    : I32EnumAttrCase<"Min", 7, "min">;
def VecOpKind_Select : I32EnumAttrCase<"Select", 8, "select">;
def VecOpKind_Cast   : I32EnumAttrCase<"Cast", 9, "cast">;

def VecOpKindAttr : I32EnumAttr<"VecOpKind", "Vector operation kind", [
  VecOpKind_Add, VecOpKind_Sub, VecOpKind_Mul, VecOpKind_Div,
  VecOpKind_Exp, VecOpKind_Log, VecOpKind_Max, VecOpKind_Min,
  VecOpKind_Select, VecOpKind_Cast
]> {
  let cppNamespace = "::mlir::ascend";
}

//===----------------------------------------------------------------------===//
// Reduction Kind
//===----------------------------------------------------------------------===//

def ReduceKind_Sum : I32EnumAttrCase<"Sum", 0, "sum">;
def ReduceKind_Max : I32EnumAttrCase<"Max", 1, "max">;
def ReduceKind_Min : I32EnumAttrCase<"Min", 2, "min">;
def ReduceKind_Prod : I32EnumAttrCase<"Prod", 3, "prod">;

def ReduceKindAttr : I32EnumAttr<"ReduceKind", "Reduction kind", [
  ReduceKind_Sum, ReduceKind_Max, ReduceKind_Min, ReduceKind_Prod
]> {
  let cppNamespace = "::mlir::ascend";
}

//===----------------------------------------------------------------------===//
// Base Operation Class
//===----------------------------------------------------------------------===//

class AscendModel_Op<string mnemonic, list<Trait> traits = []> :
    Op<AscendModel_Dialect, mnemonic, traits>;

#endif // ASCEND_MODEL_BASE_TD
