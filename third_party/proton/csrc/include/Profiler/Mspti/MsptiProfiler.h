#ifndef PROTON_PROFILER_MSPTI_PROFILER_H_
#define PROTON_PROFILER_MSPTI_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class MsptiProfiler : public GPUProfiler<MsptiProfiler> {
public:
  MsptiProfiler();
  virtual ~MsptiProfiler();

  static size_t captureExternId();
  static void restoreExternId(size_t id);

protected:
  void startOp(const Scope &scope) override;
  void stopOp(const Scope &scope) override;

private:
  struct MsptiProfilerPimpl;
  virtual void
  doSetMode(const std::vector<std::string> &modeAndOptions) override;
};

} // namespace proton

#endif // PROTON_PROFILER_MSPTI_PROFILER_H_
