#include "Data/Metric.h"
#include "Profiler/Instrumentation/Runtime.h"
#include <gtest/gtest.h>

using namespace proton;

namespace {
class TestRuntime : public Runtime {
public:
  explicit TestRuntime(DeviceType type) : Runtime(type) {}
  void allocateHostBuffer(uint8_t **, size_t) override {}
  void freeHostBuffer(uint8_t *) override {}
  uint64_t getDevice() override { return 0; }
  void *getPriorityStream() override { return nullptr; }
  void synchronizeStream(void *) override {}
  void processHostBuffer(uint8_t *, size_t, uint8_t *, size_t, void *,
                         std::function<void(uint8_t *, size_t)>) override {}
};
} // namespace

TEST(BackendPolicyTest, DefaultsPreserveCommunityBehavior) {
  for (auto device : {DeviceType::CUDA, DeviceType::HIP, DeviceType::NPU}) {
    TestRuntime runtime(device);
    EXPECT_EQ(runtime.getParserErrorPolicy(),
              ParserConfig::ErrorPolicy::Strict);
    EXPECT_FALSE(runtime.allowRepeatedFunctionMetadata());
    EXPECT_DOUBLE_EQ(runtime.getCycleCounterFreqMhz(), 1000.0);
    EXPECT_FALSE(runtime.getTimeShiftCost().has_value());
  }
}

TEST(BackendPolicyTest, CycleMetricFrequencyIsNotInferredFromDevice) {
  CycleMetric defaultMetric;
  EXPECT_DOUBLE_EQ(defaultMetric.getCycleCounterFreqMhz(), 1000.0);
  CycleMetric npuMetric(10, 20, 10, 10.0, 1, "kernel", 0, 0, 0, 0,
                        static_cast<uint64_t>(DeviceType::NPU), 0, 0, 0, 0,
                        50.0);
  EXPECT_DOUBLE_EQ(npuMetric.getCycleCounterFreqMhz(), 50.0);
}
