#ifndef PROTON_PROFILER_INSTRUMENTATION_ASCEND_RUNTIME_H_
#define PROTON_PROFILER_INSTRUMENTATION_ASCEND_RUNTIME_H_

#include "Runtime.h"

namespace proton {

class AscendRuntime : public Runtime {
public:
  AscendRuntime() : Runtime(DeviceType::NPU) {}
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
  ParserConfig::ErrorPolicy getParserErrorPolicy() const override {
    return ParserConfig::ErrorPolicy::BestEffort;
  }
  bool allowRepeatedFunctionMetadata() const override { return true; }
  double getCycleCounterFreqMhz() const override { return 50.0; }
  std::optional<int64_t> getTimeShiftCost() const override;
};

} // namespace proton

#endif // PROTON_PROFILER_INSTRUMENTATION_ASCEND_RUNTIME_H
