#include "Context/Context.h"

namespace proton {

/*static*/ thread_local std::optional<Context> ContextSource::state =
    std::nullopt;
/*static*/ std::mutex ContextSource::activeStatesMutex;
/*static*/ std::map<std::thread::id, Context> ContextSource::activeStates;

void ContextSource::setState(std::optional<Context> newState) {
  state = newState;
  std::lock_guard<std::mutex> lock(activeStatesMutex);
  const auto threadId = std::this_thread::get_id();
  if (newState.has_value())
    activeStates[threadId] = std::move(newState.value());
  else
    activeStates.erase(threadId);
}

std::optional<Context> ContextSource::getActiveState() {
  if (state.has_value())
    return state;

  std::lock_guard<std::mutex> lock(activeStatesMutex);
  if (activeStates.empty())
    return std::nullopt;

  // A callback without an originating thread can only use an unambiguous
  // state shared by all active launch threads.
  const auto &activeState = activeStates.begin()->second;
  for (const auto &entry : activeStates) {
    if (entry.second != activeState)
      return std::nullopt;
  }
  return activeState;
}

std::atomic<size_t> Scope::scopeIdCounter{1};

/*static*/ thread_local std::map<OpInterface *, bool> OpInterface::opInProgress;

} // namespace proton
