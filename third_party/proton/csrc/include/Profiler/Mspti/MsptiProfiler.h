#ifndef PROTON_PROFILER_MSPTI_PROFILER_H_
#define PROTON_PROFILER_MSPTI_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class MsptiProfiler : public GPUProfiler<MsptiProfiler>, public StateInterface {
public:
  MsptiProfiler();
  virtual ~MsptiProfiler();

  void setState(std::optional<Context> state) override;

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
