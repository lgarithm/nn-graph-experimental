#pragma once
#include <experimental/reflect>
#include <iomanip>
#include <sstream>

#include <nn/bits/graph/apply.hpp>
#include <nn/bits/graph/cuda_ops.hpp>
#include <ttl/algorithm>
#include <ttl/cuda_tensor>
#include <ttl/nn/ops>
#include <ttl/range>

#include "trace.hpp"

template <typename R, ttl::rank_t r>
class tt;

template <typename R, ttl::rank_t r>
class tt
{
    struct view;
    struct ref;
    const ttl::shape<r> shape_;

  public:
    using value_type = R;
    static constexpr auto rank = r;

    template <typename... D>
    explicit tt(const D &... d) : shape_(d...)
    {
    }

    auto shape() const { return shape_; }
};

struct cpu {
};
struct gpu {
};

template <typename>
struct make_tensor;

template <>
struct make_tensor<cpu> {
    template <typename TT>
    auto operator()(const TT &tt) const
    {
        using R = typename TT::value_type;
        ttl::tensor<R, TT::rank> t(tt.shape());
        ttl::fill(ref(t), static_cast<R>(1));
        return t;
    }
};

#ifdef NN_GRAPH_ENABLE_CUDA

template <>
struct make_tensor<gpu> {
    template <typename TT>
    auto operator()(const TT &tt) const
    {
        using R = typename TT::value_type;
        ttl::cuda_tensor<R, TT::rank> t(tt.shape());
        ttl::nn::ops::constant<R>(0)(ref(t));
        return t;
    }
};
#endif

class benchmark
{
    const std::string prefix_;

    template <typename Op, typename Y, typename XS, size_t... I>
    void call_op(const Op &f, const Y &y, const XS &xs,
                 std::index_sequence<I...>) const
    {
        const auto args = std::make_tuple(ref(y), view(std::get<I>(xs))...);
        auto name = ttl::nn::graph::internal::apply_name(f, args, true);
        if (!prefix_.empty()) { name = prefix_ + name; }
        TRACE_SCOPE(name);
        std::apply(f, args);
    }

  public:
    benchmark(const std::string &prefix = "") : prefix_(prefix) {}

    template <typename D, typename Op, typename YT, typename... XTs>
    void operator()(const D &, const Op &f, const YT &yt, XTs &... xts) const
    {
        const auto y = make_tensor<D>()(yt);
        const auto xs = std::make_tuple(make_tensor<D>()(xts)...);
        constexpr auto arity = sizeof...(XTs);
        for (auto i [[gnu::unused]] : ttl::range(60)) {
            call_op(Op(), y, xs, std::make_index_sequence<arity>());
        }
    }
};
