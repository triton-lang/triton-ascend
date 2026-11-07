#ifndef PROTON_PROFILER_INSTRUMENTATION_METADATA_H_
#define PROTON_PROFILER_INSTRUMENTATION_METADATA_H_

#include <vector>

namespace proton {

class InstrumentationMetadata {

public:
  InstrumentationMetadata(const std::string &metadataPath)
      : metadataPath(metadataPath) {
    parse();
  }

  size_t getScratchMemorySize() const { return scratchMemorySize; }

  size_t getNumWarps() const { return numWarps; }

  size_t getNumUnits() const { return numUnits; }

  bool operator==(const InstrumentationMetadata &other) const {
    return metadataPath == other.metadataPath &&
           scratchMemorySize == other.scratchMemorySize &&
           numWarps == other.numWarps && numUnits == other.numUnits;
  }

private:
  void parse();

  const std::string metadataPath;
  size_t scratchMemorySize{};
  size_t numWarps{};
  size_t numUnits{};
};

} // namespace proton

#endif // PROTON_PROFILER_INSTRUMENTATION_METADATA_H_
