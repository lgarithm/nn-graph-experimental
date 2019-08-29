#pragma once
#include <nn/bits/graph/apply.hpp>
#include <nn/bits/graph/cuda_ops.hpp>
#include <nn/bits/graph/runtime.hpp>
#include <nn/bits/ops/axpy.hpp>

namespace nn::graph::internal
{
template <typename R, ttl::rank_t r> class var_node;
class base_var_node;

template <typename D> class gradient_descent
{
    template <typename R, ttl::rank_t r>
    using tensor = typename D::template tensor_type<R, r>;

    using RT = basic_runtime<D>;

    using axpy = typename nn::for_device<nn::ops::axpy, D>::type;

  public:
    template <typename R, ttl::rank_t r>
    auto operator()(const var_node<R, r> *v,
                    const std::vector<const base_var_node *> &gs,
                    const var_node<R, 0> *lr) const
    {
        return [=](const RT &rt) {
            for (const auto g : gs) {
                apply_if<D>(axpy(),
                            std::make_tuple(rt.template get_ref<R, r>(v),
                                            rt.template get_view<R, 0>(lr),
                                            rt.template get_view<R, r>(g),
                                            rt.template get_view<R, r>(v)));
            }
        };
    }
};
}  // namespace nn::graph::internal