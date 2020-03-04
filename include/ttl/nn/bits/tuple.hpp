#pragma once
#include <array>
#include <tuple>
#include <vector>

template <size_t r, typename T, size_t... I>
std::array<T, r> index_array(const std::vector<T> &v, std::index_sequence<I...>)
{
    return std::array<T, r>({v[I]...});
}

template <size_t n, typename T>
std::array<T, n> vec2arr(const std::vector<T> &v)
{
    if (v.size() != n) { throw std::logic_error("invalid arity"); }
    return index_array<n>(v, std::make_index_sequence<n>());
}

template <typename T, typename Tuple, size_t... I>
std::array<T, std::tuple_size<Tuple>::value> tup2arr(const Tuple &t,
                                                     std::index_sequence<I...>)
{
    using Array = std::array<T, std::tuple_size<Tuple>::value>;
    return Array({static_cast<T>(std::get<I>(t))...});
}

template <typename T, typename Tuple>
std::array<T, std::tuple_size<Tuple>::value> tup2arr(const Tuple &t)
{
    return tup2arr<T>(
        t, std::make_index_sequence<std::tuple_size<Tuple>::value>());
}

template <typename F, typename T, size_t... I>
auto tuple_map(const F &f, const T &t, std::index_sequence<I...>)
{
    return std::make_tuple(f(std::get<I>(t))...);
}

template <typename F, typename T>
auto tuple_map(const F &f, const T &t)
{
    return tuple_map(f, t,
                     std::make_index_sequence<std::tuple_size<T>::value>());
}
