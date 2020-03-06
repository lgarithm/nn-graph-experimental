#pragma once
#define NN_GRAPG_TRACE 1

#include <experimental/iterator>
#include <experimental/reflect>
#include <functional>
#include <iostream>
#include <sstream>

#include <ttl/nn/bits/graph/common.hpp>
#include <ttl/nn/bits/graph/device.hpp>

#if NN_GRAPG_TRACE
#    include <ttl/nn/bits/graph/trace.hpp>
#endif

namespace ttl::nn::graph::internal
{
template <typename Tuple, size_t... I>
std::string signature(const Tuple &args, std::index_sequence<I...>)
{
    std::array<std::string, sizeof...(I)> names(
        {ttl::type_of(std::get<I>(args)).name()...});
    std::string name;
    for (auto n : names) {
        if (!name.empty()) { name += ","; }
        name += n;
    }
    return name;
}

template <typename F, typename... Args>
std::string apply_name(const F &f, const std::tuple<Args...> &args,
                       bool detail = false)
{
    const auto f_name = demangled_type_info_name(typeid(F));
    if (detail) {
        static constexpr auto arity = sizeof...(Args);
        return f_name + "(" +
               signature(args, std::make_index_sequence<arity>()) + ")";
    } else {
        return f_name;
    }
}

struct traced_apply {
    template <typename F, typename... Args>
    void operator()(const F &f, const std::tuple<Args...> &args) const
    {
        TRACE_SCOPE(apply_name(f, args, true));
        std::apply(f, args);
    }
};

template <typename, bool>
struct maybe_apply;

template <typename D>
struct maybe_apply<D, false> {
    template <typename F, typename... Args>
    void operator()(const F &f, const std::tuple<Args...> &args) const
    {
        static constexpr auto arity = sizeof...(Args);
        std::array<std::string, arity> names(
            {demangled_type_info_name(typeid(Args))...});

        std::stringstream ss;
        ss << "(\n\t";
        std::copy(names.begin(), names.end(),
                  std::experimental::make_ostream_joiner(ss, "\n\t"));
        ss << "\n)";

        std::cerr << "can't apply " << demangled_type_info_name(typeid(F))
                  << " to " << ss.str() << std::endl;
    }
};

template <typename D>
struct maybe_apply<D, true> {
    template <typename F, typename... Args>
    void operator()(const F &f, const std::tuple<Args...> &args) const
    {
#if NN_GRAPG_TRACE
        traced_apply()(f, args);
#else
        std::apply(f, args);
#endif
    }
};

template <typename D, typename F, typename... Args>
void apply_if(const F &f, const std::tuple<Args...> &args)
{
    maybe_apply<D, std::is_invocable<F, Args...>::value>()(f, args);
}

template <typename D, typename F, typename... Args>
void invoke(const D &, const F &f, const Args &... args)
{
    f(args...);
}

#ifndef NN_GRAPH_ENABLE_CUDA
template <typename F, typename... Args>
void invoke(const nvidia_gpu &, const F &f, const Args &... args)
{
    std::fprintf(stderr, "CUDA not enabled\n");
}
#endif
}  // namespace ttl::nn::graph::internal
