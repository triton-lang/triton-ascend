// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// SPDX-License-Identifier: MIT
#include "launcher_args.h"

#include <cstdlib>
#include <iostream>

// Header-only probe for issue #2797: workspace sizing must use the full 64-bit
// program count, while the block count handed to CANN stays 32-bit.
//
// usage: workspace_probe <flags> <physical_blocks> <gridX> <gridY> <gridZ>
// <workspace_size>
int main(int argc, char **argv) {
  if (argc != 7)
    return 2;
  TritonNpuLaunchSpecV1 spec{};
  spec.version = 1;
  spec.struct_size = sizeof(spec);
  spec.flags = std::strtoull(argv[1], nullptr, 10);
  spec.physical_blocks = std::strtoul(argv[2], nullptr, 10);
  spec.participant_factor = spec.task_type = 1;
  spec.coalesce_factor = 1;
  spec.coalesce_axis = -1;
  const uint64_t workspaceSize = std::strtoull(argv[6], nullptr, 10);
  int32_t input[]{std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5])};
  triton::ascend::validateSpec(spec);
  try {
    auto grid = triton::ascend::prepareGrid(spec, input);
    const uint64_t total =
        workspaceSize
            ? triton::ascend::checkedMultiply(workspaceSize, grid.logicalBlocks)
            : 0;
    std::cout << grid.logicalBlocks << ' ' << grid.physicalBlocks << ' '
              << total << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cout << "overflow " << error.what() << '\n';
    return 3;
  }
}
