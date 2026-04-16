#ifndef PROTON_PROFILER_INSTRUMENTATION_ASCEND_RUNTIME_H
#define PROTON_PROFILER_INSTRUMENTATION_ASCEND_RUNTIME_H

#include "Runtime.h"

namespace proton {

class AscendRuntime : public Runtime {
public:
  AscendRuntime() : Runtime(DeviceType::ASCEND) {}
  ~AscendRuntime() = default;

  void allocateHostBuffer(uint8_t **buffer, size_t size) override;
  void freeHostBuffer(uint8_t *buffer) override;
  uint64_t getDevice() override;
  void *getPriorityStream() override;
  void synchronizeStream(void *stream) override;
  void
  processHostBuffer(uint8_t *hostBuffer, size_t hostBufferSize,
                    uint8_t *deviceBuffer, size_t deviceBufferSize,
                    void *stream,
                    std::function<void(uint8_t *, size_t)> callback) override;
};

} // namespace proton

#endif // PROTON_PROFILER_INSTRUMENTATION_ASCEND_RUNTIME_H
