// Temporary CI diagnostic: observe raw MSPTI activity without Proton.
#include <mspti.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <thread>

namespace {
std::atomic<bool> measuring{false};
std::atomic<unsigned> enters{0}, exits{0}, callbackErrors{0};
std::mutex mutex;
unsigned buffers = 0, kernels = 0, valid = 0, external = 0;
std::set<uint64_t> validIds, measuredIds;
msptiSubscriberHandle subscriber{};

void allocate(uint8_t **buffer, size_t *size, size_t *maxRecords) {
  *size = 8 * 1024 * 1024;
  *buffer = static_cast<uint8_t *>(std::malloc(*size));
  if (!*buffer)
    *size = 0;
  *maxRecords = 0;
}

void complete(uint8_t *buffer, size_t, size_t bytes) {
  std::lock_guard<std::mutex> lock(mutex);
  ++buffers;
  msptiActivity *record = nullptr;
  while (bytes && buffer) {
    auto rc = msptiActivityGetNextRecord(buffer, bytes, &record);
    if (rc == MSPTI_ERROR_MAX_LIMIT_REACHED)
      break;
    if (rc != MSPTI_SUCCESS || !record) {
      std::fprintf(stderr, "MSPTI-PROBE next_record=%d\n", int(rc));
      break;
    }
    if (record->kind == MSPTI_ACTIVITY_KIND_KERNEL) {
      auto *k = reinterpret_cast<const msptiActivityKernel *>(record);
      ++kernels;
      if (k->end > k->start) {
        ++valid;
        validIds.insert(k->correlationId);
      }
    } else if (record->kind == MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
      auto *e =
          reinterpret_cast<const msptiActivityExternalCorrelation *>(record);
      ++external;
      if (e->externalKind == MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0 &&
          e->externalId == 2)
        measuredIds.insert(e->correlationId);
    }
  }
  std::free(buffer);
}

void callback(void *, msptiCallbackDomain, msptiCallbackId,
              const msptiCallbackData *data) {
  if (data->callbackSite == MSPTI_API_ENTER) {
    bool active = measuring.load();
    if (active)
      ++enters;
    if (msptiActivityPushExternalCorrelationId(
            MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, active ? 2 : 1) !=
        MSPTI_SUCCESS)
      ++callbackErrors;
  } else if (data->callbackSite == MSPTI_API_EXIT) {
    uint64_t id = 0;
    if (msptiActivityPopExternalCorrelationId(
            MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &id) != MSPTI_SUCCESS)
      ++callbackErrors;
    if (id == 2)
      ++exits;
  }
}

void snapshot(const char *phase) {
  std::lock_guard<std::mutex> lock(mutex);
  size_t matched = 0;
  for (auto id : measuredIds)
    matched += validIds.count(id);
  std::fprintf(
      stderr,
      "MSPTI-PROBE phase=%s enter=%u exit=%u cb_errors=%u buffers=%u kernel=%u "
      "valid=%u external=%u measured_ids=%zu matched_valid=%zu\n",
      phase, enters.load(), exits.load(), callbackErrors.load(), buffers,
      kernels, valid, external, measuredIds.size(), matched);
}
} // namespace

#define CHECK(call)                                                            \
  do {                                                                         \
    auto rc = (call);                                                          \
    if (rc != MSPTI_SUCCESS) {                                                 \
      std::fprintf(stderr, "MSPTI-PROBE %s rc=%d\n", #call, int(rc));          \
      return int(rc);                                                          \
    }                                                                          \
  } while (false)

extern "C" int probe_start() {
  CHECK(msptiSubscribe(&subscriber, callback, nullptr));
  CHECK(msptiActivityRegisterCallbacks(allocate, complete));
  CHECK(msptiActivityEnable(MSPTI_ACTIVITY_KIND_KERNEL));
  CHECK(msptiActivityEnable(MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  for (msptiCallbackId id :
       {MSPTI_CBID_RUNTIME_LAUNCH, MSPTI_CBID_RUNTIME_AICPU_LAUNCH,
        MSPTI_CBID_RUNTIME_AIV_LAUNCH, MSPTI_CBID_RUNTIME_FFTS_LAUNCH,
        MSPTI_CBID_RUNTIME_CPU_LAUNCH})
    CHECK(msptiEnableCallback(1, subscriber, MSPTI_CB_DOMAIN_RUNTIME, id));
  return 0;
}

extern "C" void probe_mark(int on) { measuring.store(on != 0); }

extern "C" int probe_wait() {
  CHECK(msptiActivityFlushAll(0));
  snapshot("immediate");
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  CHECK(msptiActivityFlushAll(0));
  snapshot("after_500ms");
  return 0;
}

extern "C" int probe_stop() {
  snapshot("before_flush");
  CHECK(msptiActivityFlushAll(0));
  snapshot("flush");
  CHECK(msptiActivityDisable(MSPTI_ACTIVITY_KIND_KERNEL));
  CHECK(msptiActivityFlushAll(0));
  snapshot("disable_flush");
  CHECK(msptiActivityFlushAll(1));
  CHECK(msptiActivityDisable(MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  CHECK(msptiUnsubscribe(subscriber));
  snapshot("stop");
  return 0;
}
