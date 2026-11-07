#ifndef PROTON_PROFILER_INSTRUMENTATION_RUNTIME_H_
#define PROTON_PROFILER_INSTRUMENTATION_RUNTIME_H_

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <optional>

#include "Device.h"
#include "TraceDataIO/Parser.h"

namespace proton {
class Runtime {
public:
  Runtime(DeviceType deviceType) : deviceType(deviceType) {}
  virtual ~Runtime() = default;

  virtual void allocateHostBuffer(uint8_t **buffer, size_t size) = 0;

  virtual void freeHostBuffer(uint8_t *buffer) = 0;

  virtual uint64_t getDevice() = 0;

  virtual void *getPriorityStream() = 0;

  virtual void synchronizeStream(void *stream) = 0;

  virtual void
  processHostBuffer(uint8_t *hostBuffer, size_t hostBufferSize,
                    uint8_t *deviceBuffer, size_t deviceBufferSize,
                    void *stream,
                    std::function<void(uint8_t *, size_t)> callback) = 0;

  // Preserve the upstream policies unless a runtime explicitly opts in.
  virtual ParserConfig::ErrorPolicy getParserErrorPolicy() const {
    return ParserConfig::ErrorPolicy::Strict;
  }
  virtual bool allowRepeatedFunctionMetadata() const { return false; }
  virtual double getCycleCounterFreqMhz() const { return 1000.0; }
  virtual std::optional<int64_t> getTimeShiftCost() const {
    return std::nullopt;
  }

  DeviceType getDeviceType() const { return deviceType; }

protected:
  DeviceType deviceType;
};
} // namespace proton

#endif // PROTON_PROFILER_INSTRUMENTATION_RUNTIME_H
