#pragma once

// Clang >= 18 predefines the __MEMORY_SCOPE_* macros (InitPreprocessor.cpp)
// and declares the __scoped_atomic_* builtins (Builtins.td); older clang
// does neither. Some Ascend CI containers additionally ship clang builds
// that declare the scoped atomics but not the scope macros, so detect both
// independently instead of keying on the version.
#if defined(__clang__)
// Scope macros; values match clang's AtomicScopeModel:
// SYSTEM=0, DEVICE=1, WRKGRP=2, WVFRNT=3, SINGLE=4.
#ifndef __MEMORY_SCOPE_WRKGRP
#define __MEMORY_SCOPE_SYSTEM 0
#define __MEMORY_SCOPE_DEVICE 1
#define __MEMORY_SCOPE_WRKGRP 2
#define __MEMORY_SCOPE_WVFRNT 3
#define __MEMORY_SCOPE_SINGLE 4
#endif
// Scoped atomics; __is_identifier is 1 only when the name is not a builtin.
// Fall back to the plain __atomic_* builtins (available in all clang builds
// and in CUDA device mode), ignoring the memory-scope argument: gsan.ll is
// only used by the NVIDIA backend for debugging, so the weaker scope is
// acceptable for this compatibility path.
#if __is_identifier(__scoped_atomic_fetch_add)
#define __scoped_atomic_fetch_add(p, v, o, s) __atomic_fetch_add(p, v, o)
#define __scoped_atomic_load_n(p, o, s) __atomic_load_n(p, o)
#define __scoped_atomic_fetch_sub(p, v, o, s) __atomic_fetch_sub(p, v, o)
#define __scoped_atomic_fetch_and(p, v, o, s) __atomic_fetch_and(p, v, o)
#define __scoped_atomic_store_n(p, v, o, s) __atomic_store_n(p, v, o)
#define __scoped_atomic_thread_fence(o, s) __atomic_thread_fence(o)
#define __scoped_atomic_compare_exchange_n(p, e, d, w, o, f, s) \
  __atomic_compare_exchange_n(p, e, d, w, o, f)
#endif
#endif

#if defined(__CUDA__) && defined(__clang__)
#define GSAN_DEVICE __attribute__((device))
#define GSAN_HOST_DEVICE __attribute__((host)) __attribute__((device))
#elif defined(__CUDACC__)
#define GSAN_DEVICE __device__
#define GSAN_HOST_DEVICE __host__ __device__
#else
#define GSAN_DEVICE
#define GSAN_HOST_DEVICE
#endif

namespace gsan {

using size_t = __SIZE_TYPE__;
using uint8_t = __UINT8_TYPE__;
using uint16_t = __UINT16_TYPE__;
using uint32_t = __UINT32_TYPE__;
using uintptr_t = __UINTPTR_TYPE__;

// Reserve 1 PiB, should be big enough for a while :)
static constexpr size_t kReserveSize = 1ull << 40;
static constexpr int kShadowMemGranularityBytes = 4;
static_assert((kReserveSize & (kReserveSize - 1)) == 0,
              "kReserveSize must be a power of 2");

using thread_id_t = uint16_t;

enum class AtomicScope : uint8_t {
  NonAtomic,
  CTA,
  GPU,
  System,
  MAX_VALUE = System,
};

using epoch_t = uint16_t;

struct alignas(4) ScalarClock {
  epoch_t epoch;
  thread_id_t threadId : 12; // Supports 4096 threads
  AtomicScope scope : 2;
  // For a release write, the epoch is actually an index into the thread's
  // circular clock buffer where the full vector clock is stored.
  bool isRelease : 1;
};
static constexpr int kMaxThreads = 1 << 12;
static_assert(sizeof(ScalarClock) == 4);
static_assert(static_cast<int>(AtomicScope::MAX_VALUE) == 3);

// TODO: Change to struct-of-array for better coalescing?
struct alignas(4) ShadowCell {
  static constexpr int kReadClockSize = 4;
  ScalarClock readClocks[kReadClockSize];
  ScalarClock writeClock;
  uint16_t numReads;
  uint16_t lock;
};
static_assert(sizeof(ShadowCell) == 24);
static_assert(alignof(ShadowCell) == 4);

struct GlobalState {
  // Base address of gsan managed memory
  uintptr_t reserveBase;
  uintptr_t globalsBase;

  uint32_t rngSeed;

  thread_id_t numSms;
  thread_id_t numDevices;
  // numThreads = numSms * numDevices
  thread_id_t numThreads;

  uint16_t clockBufferSize;
};

struct ThreadState {
  GlobalState *globals;
  uintptr_t reserveBase;

  // monotonic counter, used for stochastic read clock updates
  uint32_t numReads;

  // Index to head of the circular clock buffer, plus a bit to mark if the
  // vector clock has changed since the last clock buffer write (to allow reuse)
  uint32_t clockBufferDirty : 1;
  uint32_t clockBufferHead : 31;

  // Reader-writer lock controlling access to the vector clock and clock buffer
  uint32_t lock;

  thread_id_t threadId;

  // Local vector clock, shape [numThreads]
  // Followed by the clock buffer, shape [clockBufferSize, numThreads]
  epoch_t vectorClock[];
};

static constexpr int kMaxAtomicShadowCells = 3;

struct AtomicEventState {
  ThreadState *threadState;
  ShadowCell *cells[kMaxAtomicShadowCells];
  uint8_t numCells;
};

// Place the thread state for each device at a fixed stride for ease of
// address calculation.
static constexpr uintptr_t kPerDeviceStateStride = 1ull << 30;
static constexpr uintptr_t kMaxGPUs = 32;
static constexpr uintptr_t kGlobalsReserveSize =
    kPerDeviceStateStride * kMaxGPUs;

inline GSAN_HOST_DEVICE GlobalState *getGlobalState(ThreadState *threadState) {
  auto threadAddr = (uintptr_t)threadState;
  return (GlobalState *)(threadAddr & ~(kPerDeviceStateStride - 1));
}

inline GSAN_HOST_DEVICE uintptr_t getRealBaseAddress(uintptr_t reserveBase) {
  return reserveBase + kReserveSize / 2;
}

inline GSAN_HOST_DEVICE uintptr_t getReserveBaseFromAddress(uintptr_t addr) {
  return addr & ~(kReserveSize - 1);
}

// Assumes address is in gsan-managed memory
inline GSAN_HOST_DEVICE uintptr_t getShadowAddress(uintptr_t virtualAddress) {
  auto reserveBase = getReserveBaseFromAddress(virtualAddress);
  auto realBase = getRealBaseAddress(reserveBase);
  auto byteOffset = virtualAddress - realBase;
  auto wordOffset = byteOffset / kShadowMemGranularityBytes;
  return reserveBase + sizeof(ShadowCell) * wordOffset;
}

inline GSAN_HOST_DEVICE bool isGsanManaged(uintptr_t addr,
                                           uintptr_t reserveBase) {
  return getReserveBaseFromAddress(addr) == reserveBase;
}

inline GSAN_HOST_DEVICE bool isAtomicScope(AtomicScope scope) {
  return scope != AtomicScope::NonAtomic;
}

} // namespace gsan
