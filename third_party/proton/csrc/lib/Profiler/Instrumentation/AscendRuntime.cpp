#include "Profiler/Instrumentation/AscendRuntime.h"

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <algorithm>
#include <stdexcept>

namespace proton {

void AscendRuntime::allocateHostBuffer(uint8_t **buffer, size_t size) {
  aclError ret = aclrtMallocHost(reinterpret_cast<void **>(buffer), size);
  if (ret != ACL_ERROR_NONE) {
    throw std::runtime_error("Failed to allocate host buffer: " +
                             std::to_string(ret));
  }
}

void AscendRuntime::freeHostBuffer(uint8_t *buffer) {
  aclError ret = aclrtFreeHost(buffer);
  if (ret != ACL_ERROR_NONE) {
    throw std::runtime_error("Failed to free host buffer: " +
                             std::to_string(ret));
  }
}

uint64_t AscendRuntime::getDevice() {
  int32_t deviceId;
  aclError ret = aclrtGetDevice(&deviceId);
  if (ret != ACL_ERROR_NONE) {
    throw std::runtime_error("Failed to get device: " + std::to_string(ret));
  }
  return static_cast<uint64_t>(deviceId);
}

void *AscendRuntime::getPriorityStream() {
  aclrtStream stream;
  aclError ret =
      aclrtCreateStreamWithConfig(&stream, 0, ACL_STREAM_FAST_LAUNCH);
  if (ret != ACL_ERROR_NONE) {
    throw std::runtime_error("Failed to create stream: " + std::to_string(ret));
  }
  return reinterpret_cast<void *>(stream);
}

void AscendRuntime::synchronizeStream(void *stream) {
  aclError ret = aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream));
  if (ret != ACL_ERROR_NONE) {
    throw std::runtime_error("Failed to synchronize stream: " +
                             std::to_string(ret));
  }
}

void AscendRuntime::processHostBuffer(
    uint8_t *hostBuffer, size_t hostBufferSize, uint8_t *deviceBuffer,
    size_t deviceBufferSize, void *stream,
    std::function<void(uint8_t *, size_t)> callback) {
  int64_t chunkSize = std::min(hostBufferSize, deviceBufferSize);
  int64_t sizeLeftOnDevice = deviceBufferSize;

  while (chunkSize > 0) {
    aclError ret = aclrtMemcpyAsync(
        reinterpret_cast<void *>(hostBuffer), chunkSize,
        reinterpret_cast<void *>(deviceBuffer), chunkSize,
        ACL_MEMCPY_DEVICE_TO_HOST, reinterpret_cast<aclrtStream>(stream));

    if (ret != ACL_ERROR_NONE) {
      throw std::runtime_error("Failed to copy memory from device to host: " +
                               std::to_string(ret));
    }

    // For simplicity, only copy the buffer after the kernel has finished.
    ret = aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream));
    if (ret != ACL_ERROR_NONE) {
      throw std::runtime_error(
          "Failed to synchronize stream during buffer processing: " +
          std::to_string(ret));
    }

    callback(hostBuffer, chunkSize);

    sizeLeftOnDevice -= chunkSize;
    deviceBuffer += chunkSize;
    chunkSize =
        std::min(static_cast<int64_t>(hostBufferSize), sizeLeftOnDevice);
  }
}

} // namespace proton
