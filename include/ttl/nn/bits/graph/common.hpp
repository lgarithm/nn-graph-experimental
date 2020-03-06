#pragma once
#include <climits>
#include <sstream>
#include <string>

#include <ttl/bits/std_reflect.hpp>
#include <ttl/debug>
#include <ttl/experimental/type>
#include <ttl/tensor>

namespace ttl::nn::graph
{
using arity_t = uint8_t;

namespace internal
{
using ttl::internal::demangled_type_info_name;

template <typename T, typename P>
T *down_cast(P *parent)
{
    T *p = dynamic_cast<T *>(parent);
    if (p == nullptr) {
        throw std::logic_error("invalid down_cast from " +
                               demangled_type_info_name<P>() + " to " +
                               demangled_type_info_name<T>());
    }
    return p;
}
template <typename T, typename P>
const T *down_cast(const P *parent)
{
    const T *p = dynamic_cast<const T *>(parent);
    if (p == nullptr) {
        throw std::logic_error("invalid down_cast from " +
                               demangled_type_info_name<P>() + " to " +
                               demangled_type_info_name<T>());
    }
    return p;
}

}  // namespace internal
}  // namespace ttl::nn::graph

namespace ttl
{
namespace internal
{
template <typename R, typename S, typename D, typename A>
basic_tensor_type<R, S> type_of(const basic_tensor<R, S, D, A> &x)
{
    return basic_tensor_type<R, S>(x.dims());
}
}  // namespace internal

using internal::type_of;
}  // namespace ttl
