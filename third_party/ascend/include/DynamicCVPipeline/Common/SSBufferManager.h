/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SSBUFFER_MANAGER_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SSBUFFER_MANAGER_H

#include <optional>
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace triton {

// SSBuffer Manager for managing SSBuffer address allocation and type tracking
// Purpose: Globally manage SSBuffer addresses across the entire pass pipeline
// This class maintains a single mapping table: Value -> address (int64_t)
// Type information is retrieved from the Value itself
class SSBufferManager {
public:
  // SSBuffer address space and constants
  static constexpr int SSBUF_ADDR_SPACE = 11;
  static constexpr int ADDR_INT_TYPE = 64;
  static constexpr int SSBUF_BASE_ADDR = 2048;      // Base address for SSBuffer
  static constexpr int SSBUF_ADDR_OFFSET = 8;       // Address offset for each allocation
  static constexpr int SSBUF_ADDR_MAX = 6072;       // Maximum allowed address

  // Constructor
  SSBufferManager() = default;

  // Allocate SSBuffer address for a value
  // This function manages SSBuffer address allocation:
  // - Checks if the value's type is a scalar type (IntegerType or FloatType)
  // - If the value already has an allocated SSBuffer address, reuse it
  // - Otherwise, allocate a new SSBuffer address based on map size
  //   Address formula: base_addr + map_size * offset (2048 + size * 8)
  // - Returns std::nullopt if:
  //   1. The value's type is not a scalar type
  //   2. Address exceeds maximum limit (6072)
  std::optional<int64_t> allocateAddr(Value value);

  // Find the Value and its type for a given address
  // This function searches the mapping table to find the Value corresponding to the address
  // - Returns std::nullopt if the address is not found
  // - Returns the Value and its type if found
  std::optional<std::pair<Value, Type>> findValueByAddr(int64_t addr);

  // Write a value to SSBuffer and return the SSBuffer address (int64_t)
  // This function uses allocateAddr to get the address and then writes the value
  // - The value must be a scalar type (IntegerType or FloatType)
  // - Returns std::nullopt if:
  //   1. The value's type is not a scalar type
  //   2. Address allocation fails (out of range)
  // - createdOps: A vector to store all operations created during this function call
  //   (including ConstantOp, IntToPtrOp, StoreOp)
  // - Note: The actual write operation is performed, but only the address value is returned
  std::optional<int64_t> writeToSSBuffer(Value value, OpBuilder &builder, 
                                         SmallVectorImpl<Operation *> &createdOps);

  // Read a value from SSBuffer based on the given address (int64_t)
  // This function loads the value from the specified SSBuffer address
  // - The address must be a valid SSBuffer address that was previously allocated
  // - The data type is automatically retrieved by searching the mapping table
  // - Returns std::nullopt if:
  //   1. The address is not found in the mapping table
  //   2. The address is invalid
  // - createdOps: A vector to store all operations created during this function call
  //   (including ConstantOp, IntToPtrOp, LoadOp)
  std::optional<Value> readFromSSBuffer(int64_t addr, OpBuilder &builder,
                                        SmallVectorImpl<Operation *> &createdOps);

  // Get the number of allocated addresses
  size_t getAllocatedCount() const { return valueToAddrMap.size(); }

  // Clear all mappings (for testing or reset)
  void clear() { valueToAddrMap.clear(); }

private:
  // Helper function to check if a type is a scalar type
  // Scalar types include: IntegerType (i1, i8, i16, i32, i64, etc.) 
  // and FloatType (f16, f32, f64, bf16, f8, etc.)
  static bool isScalarType(Type type);

  // Memory management for SSBuffer addresses
  // Maps original Value to its allocated SSBuffer address (int64_t)
  // Used for address allocation, reuse, and type retrieval during read operations
  // Address is calculated based on map size: base_addr + size * offset
  llvm::DenseMap<Value, int64_t> valueToAddrMap;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SSBUFFER_MANAGER_H