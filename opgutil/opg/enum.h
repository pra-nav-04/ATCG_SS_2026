#pragma once

#include <type_traits>

namespace opg {

template <typename E>
constexpr typename std::underlying_type<E>::type to_underlying(E e) noexcept
{
    return static_cast<typename std::underlying_type<E>::type>(e);
}

} // namespace opg

#define OPG_DECLARE_ENUM_OPERATIONS(E) \
    constexpr typename std::underlying_type<E>::type operator|(E a, E b) { return opg::to_underlying<E>(a) | opg::to_underlying<E>(b); }      \
    constexpr typename std::underlying_type<E>::type operator&(E a, E b) { return opg::to_underlying<E>(a) & opg::to_underlying<E>(b); }      \
    constexpr typename std::underlying_type<E>::type operator|(std::underlying_type<E>::type a, E b) { return a | opg::to_underlying<E>(b); } \
    constexpr typename std::underlying_type<E>::type operator&(std::underlying_type<E>::type a, E b) { return a & opg::to_underlying<E>(b); } \
    constexpr typename std::underlying_type<E>::type operator~(E e) { return ~opg::to_underlying<E>(e); }                                     \
    constexpr typename std::underlying_type<E>::type operator+(E e) { return opg::to_underlying<E>(e); }                                      \
    constexpr bool has_flag(std::underlying_type<E>::type a, E b) { return (a & b) != 0; }                                                    \
    constexpr void set_flag(std::underlying_type<E>::type &a, E b) { a |= +b; }                                                               \
    constexpr void reset_flag(std::underlying_type<E>::type &a, E b) { a &= ~b; }
