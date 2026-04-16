#include "Profiler/Instrumentation/AscendRuntime.h"

#include "Driver/GPU/AscendclApi.h"
#include "Driver/GPU/AscendrtApi.h"
#include <stdexcept>
namespace proton {

void AscendRuntime::allocateHostBuffer(uint8_t **buffer, size_t size) {
  ascend::memAllocHost<true>(reinterpret_cast<void **>(buffer), size);
}

void AscendRuntime::freeHostBuffer(uint8_t *buffer) {
  ascend::memFreeHost<true>(buffer);
}

uint64_t AscendRuntime::getDevice() {
  int32_t device;
  ascend::ctxGetDevice<true>(&device);
  return static_cast<uint64_t>(device);
}

void *AscendRuntime::getPriorityStream() {
  rtStream_t stream;
  // TODO: Change priority
  int32_t lowestPriority, highestPriority;
  ascend::ctxGetStreamPriorityRange<true>(&lowestPriority, &highestPriority);
  ascend::streamCreateWithPriority<true>(&stream, highestPriority);
  return reinterpret_cast<void *>(stream);
}

void AscendRuntime::synchronizeStream(void *stream) {
  ascend::streamSynchronize<true>(reinterpret_cast<rtStream_t>(stream));
}

void AscendRuntime::processHostBuffer(
    uint8_t *hostBuffer, size_t hostBufferSize, uint8_t *deviceBuffer,
    size_t deviceBufferSize, void *stream,
    std::function<void(uint8_t *, size_t)> callback) {
  // Mirrors CudaRuntime::processHostBuffer / HipRuntime::processHostBuffer:
  // the source pointer is intentionally not advanced because the current
  // profile_buffer_size is pinned at 1 (see instrumentation.py FIXME), so the
  // loop executes exactly once in practice.
  // Fix when proper device-to-host streaming lands.
  int64_t chunkSize = std::min(hostBufferSize, deviceBufferSize);
  int64_t sizeLeftOnDevice = deviceBufferSize;
  while (chunkSize > 0) {
    ascend::memcpyDToHAsync<true>(reinterpret_cast<void *>(hostBuffer),
                                  chunkSize,
                                  reinterpret_cast<void*>(deviceBuffer),
                                  chunkSize, 
                                  rtMemcpyKind_t::RT_MEMCPY_DEVICE_TO_HOST,
                                  reinterpret_cast<rtStream_t>(stream));
    // We should not use synchronization here in general if we want to copy
    // buffer while the kernel is running. But for the sake of simplicity, we
    // only copy the buffer after the kernel is finished for now.
    ascend::streamSynchronize<true>(reinterpret_cast<rtStream_t>(stream));
    callback(hostBuffer, chunkSize);
    sizeLeftOnDevice -= chunkSize;
    chunkSize =
        std::min(static_cast<int64_t>(hostBufferSize), sizeLeftOnDevice);
  }
}

} // namespace proton
