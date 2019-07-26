#pragma once
#include <ttl/shape>

#include <ttl/cuda_tensor>
#include <ttl/tensor>

#include <nn/bits/graph/cuda_ops.hpp>
#include <nn/bits/graph/runtime.hpp>
#include <nn/bits/ops/axpy.hpp>
#include <nn/bits/ops/init.hpp>

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
    auto operator()(float eta, const var_node<R, r> *v,
                    const std::vector<const base_var_node *> &gs) const
    {
        return [=](const RT &rt) {
            tensor<R, 0> a;
            using fill_op =
                typename nn::for_device<nn::ops::constant<R>, D>::type;
            (fill_op(static_cast<R>(-eta)))(ref(a));
            for (const auto g : gs) {
                axpy()(rt.template get_ref<R, r>(v), view(a),
                       rt.template get_view<R, r>(g),
                       rt.template get_view<R, r>(v));
            }
        };
    }
};

}  // namespace nn::graph::internal
