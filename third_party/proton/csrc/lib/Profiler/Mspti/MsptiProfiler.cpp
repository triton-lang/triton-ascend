#include "Profiler/Mspti/MsptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Device.h"
#include "Driver/GPU/MsptiApi.h"

#include <atomic>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace proton {

template <>
thread_local GPUProfiler<MsptiProfiler>::ThreadState
    GPUProfiler<MsptiProfiler>::threadState(MsptiProfiler::instance());

template <>
thread_local std::deque<size_t>
    GPUProfiler<MsptiProfiler>::Correlation::externIdQueue{};

namespace {

thread_local std::vector<bool> callbackExternalCorrelationPushed;
// Triton launch hooks and MSPTI runtime callbacks can run on different
// threads. Preserve launch order until the callback can install the external
// correlation ID on the thread observed by MSPTI.
std::deque<uint64_t> pendingExternalIds;
std::mutex pendingExternalIdsMutex;
std::atomic<size_t> pendingKernelActivities{0};

std::shared_ptr<Metric>
convertActivityToMetric(const msptiActivityKernel *kernel) {
  if (kernel->start == 0 && kernel->end == 0)
    return nullptr;
  if (kernel->start >= kernel->end)
    return nullptr;
  return std::make_shared<KernelMetric>(
      static_cast<uint64_t>(kernel->start), static_cast<uint64_t>(kernel->end),
      /*invocations=*/1, static_cast<uint64_t>(kernel->ds.deviceId),
      static_cast<uint64_t>(DeviceType::NPU),
      static_cast<uint64_t>(kernel->ds.streamId));
}

void processActivityKernel(MsptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
                           MsptiProfiler::ApiExternIdSet &apiExternIds,
                           std::set<Data *> &dataSet,
                           const msptiActivityKernel *kernel,
                           uint64_t &outMaxCorrId) {

  auto corrId = kernel->correlationId;
  if (corrId > outMaxCorrId)
    outMaxCorrId = corrId;

  if (!corrIdToExternId.contain(corrId))
    return;

  auto [parentId, numInstances] = corrIdToExternId.at(corrId);

  for (auto *data : dataSet) {
    auto metric = convertActivityToMetric(kernel);
    if (metric)
      data->addMetric(parentId, metric);
  }

  apiExternIds.erase(parentId);
  --numInstances;
  if (numInstances == 0)
    corrIdToExternId.erase(corrId);
  else
    corrIdToExternId[corrId].second = numInstances;

  auto pending = pendingKernelActivities.load(std::memory_order_relaxed);
  while (pending != 0 && !pendingKernelActivities.compare_exchange_weak(
                             pending, pending - 1, std::memory_order_relaxed)) {
  }
}

bool isDriverAPILaunch(msptiCallbackId cbId) {
  return cbId == MSPTI_CBID_RUNTIME_LAUNCH ||
         cbId == MSPTI_CBID_RUNTIME_AICPU_LAUNCH ||
         cbId == MSPTI_CBID_RUNTIME_AIV_LAUNCH ||
         cbId == MSPTI_CBID_RUNTIME_FFTS_LAUNCH ||
         cbId == MSPTI_CBID_RUNTIME_CPU_LAUNCH;
}

void setRuntimeCallbacks(msptiSubscriberHandle subscriber, bool enable) {
#define CALLBACK_ENABLE(id)                                                    \
  mspti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,       \
                              MSPTI_CB_DOMAIN_RUNTIME, (id))

  CALLBACK_ENABLE(MSPTI_CBID_RUNTIME_LAUNCH);
  CALLBACK_ENABLE(MSPTI_CBID_RUNTIME_AICPU_LAUNCH);
  CALLBACK_ENABLE(MSPTI_CBID_RUNTIME_AIV_LAUNCH);
  CALLBACK_ENABLE(MSPTI_CBID_RUNTIME_FFTS_LAUNCH);
  CALLBACK_ENABLE(MSPTI_CBID_RUNTIME_CPU_LAUNCH);
#undef CALLBACK_ENABLE
}

} // namespace

struct MsptiProfiler::MsptiProfilerPimpl
    : public GPUProfiler<MsptiProfiler>::GPUProfilerPimplInterface {

  MsptiProfilerPimpl(MsptiProfiler &profiler)
      : GPUProfiler<MsptiProfiler>::GPUProfilerPimplInterface(profiler) {}
  virtual ~MsptiProfilerPimpl() = default;

  void doStart() override;
  void doFlush() override;
  void doStop() override;

  static void allocBuffer(uint8_t **buffer, size_t *bufferSize,
                          size_t *maxNumRecords);
  static void completeBuffer(uint8_t *buffer, size_t size, size_t validSize);
  static void callbackFn(void *userData, msptiCallbackDomain domain,
                         msptiCallbackId cbId, const msptiCallbackData *cbData);

  static constexpr size_t BufferSize = 64 * 1024 * 1024;

  msptiSubscriberHandle subscriber{};
};

void MsptiProfiler::MsptiProfilerPimpl::allocBuffer(uint8_t **buffer,
                                                    size_t *bufferSize,
                                                    size_t *maxNumRecords) {
  *buffer = static_cast<uint8_t *>(std::malloc(BufferSize));
  if (!*buffer)
    throw std::runtime_error("[proton/mspti] allocBuffer: malloc failed");
  *bufferSize = BufferSize;
  *maxNumRecords = 0;
}

void MsptiProfiler::MsptiProfilerPimpl::completeBuffer(uint8_t *buffer,
                                                       size_t /*size*/,
                                                       size_t validSize) {
  MsptiProfiler &profiler = threadState.profiler;
  auto &dataSet = profiler.dataSet;
  if (validSize == 0 || buffer == nullptr) {
    std::free(buffer);
    return;
  }

  uint64_t maxCorrId = 0;
  msptiActivity *record = nullptr;
  std::vector<const msptiActivityKernel *> kernelRecords;
  do {
    auto status =
        mspti::activityGetNextRecord<false>(buffer, validSize, &record);
    if (status == MSPTI_SUCCESS) {
      if (record->kind == MSPTI_ACTIVITY_KIND_KERNEL) {
        auto *kernel = reinterpret_cast<const msptiActivityKernel *>(record);
        kernelRecords.push_back(kernel);
      } else if (record->kind == MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
        auto *external =
            reinterpret_cast<const msptiActivityExternalCorrelation *>(record);
        if (external->externalKind == MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0) {
          profiler.correlation.corrIdToExternId[external->correlationId] = {
              static_cast<size_t>(external->externalId), 1};
        }
      }
    } else if (status == MSPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      throw std::runtime_error("mspti::activityGetNextRecord failed");
    }
  } while (true);

  // MSPTI does not guarantee that external-correlation records precede kernel
  // records in a completed buffer. Build all mappings before consuming kernels.
  for (const auto *kernel : kernelRecords) {
    processActivityKernel(profiler.correlation.corrIdToExternId,
                          profiler.correlation.apiExternIds, dataSet, kernel,
                          maxCorrId);
  }

  std::free(buffer);
  profiler.correlation.complete(maxCorrId);
}

void MsptiProfiler::MsptiProfilerPimpl::callbackFn(
    void * /*userData*/, msptiCallbackDomain domain, msptiCallbackId cbId,
    const msptiCallbackData *cbData) {
  if (domain != MSPTI_CB_DOMAIN_RUNTIME)
    return;
  if (!isDriverAPILaunch(cbId))
    return;

  if (cbData->callbackSite == MSPTI_API_ENTER) {
    bool pushedExternalCorrelation = false;
    {
      std::lock_guard<std::mutex> lock(pendingExternalIdsMutex);
      if (!pendingExternalIds.empty()) {
        const auto externalId = pendingExternalIds.front();
        pendingExternalIds.pop_front();
        mspti::activityPushExternalCorrelationId<true>(
            MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, externalId);
        pushedExternalCorrelation = true;
      }
    }
    callbackExternalCorrelationPushed.push_back(pushedExternalCorrelation);
  } else if (cbData->callbackSite == MSPTI_API_EXIT) {
    if (!callbackExternalCorrelationPushed.empty()) {
      if (callbackExternalCorrelationPushed.back()) {
        uint64_t externalId = 0;
        mspti::activityPopExternalCorrelationId<true>(
            MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &externalId);
      }
      callbackExternalCorrelationPushed.pop_back();
    }
  }
}

void MsptiProfiler::MsptiProfilerPimpl::doStart() {
  {
    std::lock_guard<std::mutex> lock(pendingExternalIdsMutex);
    pendingExternalIds.clear();
  }
  profiler.correlation.corrIdToExternId.clear();
  profiler.correlation.apiExternIds.clear();
  pendingKernelActivities.store(0, std::memory_order_relaxed);
  mspti::subscribe<true>(&subscriber, callbackFn, nullptr);
  mspti::activityRegisterCallbacks<true>(allocBuffer, completeBuffer);
  mspti::activityEnable<true>(MSPTI_ACTIVITY_KIND_KERNEL);
  mspti::activityEnable<true>(MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
  setRuntimeCallbacks(subscriber, /*enable=*/true);
}

void MsptiProfiler::MsptiProfilerPimpl::doFlush() {
  mspti::activityFlushAll<true>(/*flag=*/0);
  if (pendingKernelActivities.load(std::memory_order_relaxed) != 0) {
    // MSPTI 26.1 flushes its host ActivityBuffer but does not flush the
    // device profiling channel. Cycling only KERNEL forces those SOC records
    // through KernelParser while the Proton Data objects are still registered.
    mspti::activityDisable<true>(MSPTI_ACTIVITY_KIND_KERNEL);
    mspti::activityFlushAll<true>(/*flag=*/0);
    mspti::activityEnable<true>(MSPTI_ACTIVITY_KIND_KERNEL);
  }
  mspti::activityFlushAll<true>(/*flag=*/1);
}

void MsptiProfiler::MsptiProfilerPimpl::doStop() {
  mspti::activityDisable<true>(MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
  mspti::activityDisable<true>(MSPTI_ACTIVITY_KIND_KERNEL);
  setRuntimeCallbacks(subscriber, /*enable=*/false);
  mspti::unsubscribe<true>(subscriber);
  subscriber = nullptr;
  profiler.correlation.corrIdToExternId.clear();
  profiler.correlation.apiExternIds.clear();
  pendingKernelActivities.store(0, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(pendingExternalIdsMutex);
  pendingExternalIds.clear();
}

MsptiProfiler::MsptiProfiler() {
  pImpl = std::make_unique<MsptiProfilerPimpl>(*this);
}

MsptiProfiler::~MsptiProfiler() = default;

void MsptiProfiler::startOp(const Scope &scope) {
  GPUProfiler<MsptiProfiler>::startOp(scope);
  correlation.apiExternIds.insert(scope.scopeId);
  pendingKernelActivities.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(pendingExternalIdsMutex);
  pendingExternalIds.push_back(static_cast<uint64_t>(scope.scopeId));
}

void MsptiProfiler::stopOp(const Scope &scope) {
  GPUProfiler<MsptiProfiler>::stopOp(scope);
}

void MsptiProfiler::doSetMode(const std::vector<std::string> &modeAndOptions) {
  auto mode = modeAndOptions[0];
  if (!mode.empty()) {
    throw std::invalid_argument("[PROTON] MsptiProfiler: unsupported mode: " +
                                mode);
  }
}

} // namespace proton
