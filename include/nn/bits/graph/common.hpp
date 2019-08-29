#pragma once
#include <climits>
#include <sstream>
#include <string>

#include <ttl/debug>
#include <ttl/tensor>

#include <experimental/reflect>

namespace nn::graph
{

using arity_t = uint8_t;

struct type_size {
    template <typename R> auto operator()() const { return sizeof(R); }
};

struct scalar_type_name {
    template <typename R> std::string prefix() const
    {
        static_assert(std::is_floating_point<R>::value ||
                      std::is_integral<R>::value);

        if (std::is_floating_point<R>::value) {
            return "f";
        } else if (std::is_integral<R>::value) {
            return std::is_signed<R>::value ? "i" : "u";
        } else {
            // return "s";
            return "";
        }
    }

    template <typename R> std::string operator()() const
    {
        return prefix<R>() + std::to_string(sizeof(R) * CHAR_BIT);
    }
};

namespace internal
{
template <typename T, typename P> T *down_cast(P *parent)
{
    T *p = dynamic_cast<T *>(parent);
    if (p == nullptr) {
        throw std::logic_error("invalid down_cast from " +
                               demangled_type_info_name(typeid(P)) + " to " +
                               demangled_type_info_name(typeid(T)));
    }
    return p;
}

template <typename T, typename P> const T *down_cast(const P *parent)
{
    const T *p = dynamic_cast<const T *>(parent);
    if (p == nullptr) {
        throw std::logic_error("invalid down_cast from " +
                               demangled_type_info_name(typeid(P)) + " to " +
                               demangled_type_info_name(typeid(T)));
    }
    return p;
}

}  // namespace internal
}  // namespace nn::graph

namespace ttl
{
template <typename T> std::string tensor_type_name(const T &t)
{
    return nn::graph::scalar_type_name()
               .template operator()<typename T::value_type>() +
           ttl::to_string(t.shape());
}
}  // namespace ttl
