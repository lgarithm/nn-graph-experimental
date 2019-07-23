#pragma once
#include <ttl/shape>

#include <ttl/cuda_tensor>
#include <ttl/tensor>

#include <nn/bits/graph/runtime.hpp>
#include <nn/bits/ops/axpy.hpp>
#include <nn/bits/ops/cuda_ops.hpp>

namespace nn::graph::internal
{

template <typename R, ttl::rank_t r> class var_node;
class base_var_node;

template <typename> struct make_scalar;

template <> struct make_scalar<cpu> {
    template <typename R> ttl::tensor<R, 0> operator()(const R x) const
    {
        ttl::tensor<R, 0> a;
        a.data()[0] = x;
        return a;
    }
};

template <> struct make_scalar<nvidia_gpu> {
    template <typename R> ttl::cuda_tensor<R, 0> operator()(const R x) const
    {
        ttl::cuda_tensor<R, 0> a;
        // a.data()[0] = x;
        std::cerr << "TODO: make scalar cuda tensor" << std::endl;
        return a;
    }
};

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
            auto a = make_scalar<D>()(static_cast<R>(-eta));
            for (const auto g : gs) {
                axpy()(rt.template get_ref<R, r>(v), view(a),
                       rt.template get_view<R, r>(g),
                       rt.template get_view<R, r>(v));
            }
        };
    }
};

}  // namespace nn::graph::internal
