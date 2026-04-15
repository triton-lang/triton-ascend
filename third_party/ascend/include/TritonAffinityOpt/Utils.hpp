#ifndef TRITON_AFFINITY_UTILS_HPP
#define TRITON_AFFINITY_UTILS_HPP

#include <functional>
#include <type_traits>

namespace mlir::AffinityDAG {

template <typename T, typename F> constexpr inline T enumOp(F &&func, T lhs, T rhs)
{
    static_assert(std::is_enum_v<T>, "T must be an enum type");

    using U = std::underlying_type_t<T>;

    return static_cast<T>(std::invoke(std::forward<F>(func), static_cast<U>(lhs), static_cast<U>(rhs)));
}

} // namespace mlir::AffinityDAG

#endif
