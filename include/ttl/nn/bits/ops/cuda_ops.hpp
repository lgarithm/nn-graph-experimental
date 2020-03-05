#pragma once
#include <experimental/reflect>

#include <ttl/nn/bits/graph/device.hpp>

namespace ttl::nn
{
namespace ops
{
template <typename F>
class not_implemented_for_cuda
{
    mutable int cnt_;

  public:
    template <typename... T>
    not_implemented_for_cuda(const T &...) : cnt_(0)
    {
    }

    template <typename... T>
    void operator()(const T &...) const
    {
        if (cnt_ == 0) {
            std::cerr << demangled_type_info_name(typeid(F))
                      << " not implemented for cuda!" << std::endl;
        }
        ++cnt_;
    }
};
}  // namespace ops

template <typename F>
struct cuda_op;

template <typename F, typename D>
struct for_device;

template <typename F>
struct for_device<F, nn::graph::internal::cpu> {
    using type = F;
};

template <typename F>
struct for_device<F, nn::graph::internal::nvidia_gpu> {
    using type = typename cuda_op<F>::type;
};

template <typename F>
struct cuda_op {
    using type = ops::not_implemented_for_cuda<F>;
};
}  // namespace ttl::nn

#ifdef NN_GRAPH_ENABLE_CUDA
#    include <ttl/nn/bits/ops/gradients/activation.hpp>
#    include <ttl/nn/bits/ops/gradients/bias.hpp>
#    include <ttl/nn/bits/ops/gradients/conv2d.hpp>
#    include <ttl/nn/bits/ops/gradients/matmul.hpp>
#    include <ttl/nn/bits/ops/gradients/mul.hpp>
#    include <ttl/nn/bits/ops/gradients/reshape.hpp>
#    include <ttl/nn/bits/ops/gradients/softmax.hpp>
#    include <ttl/nn/bits/ops/gradients/xentropy.hpp>
#    include <ttl/nn/ops>

#    include <ttl/nn/cuda_gradients>
#    include <ttl/nn/cuda_ops>
#    include <ttl/nn/kernels/cuda>

namespace ttl::nn
{
template <>
struct cuda_op<ttl::nn::ops::onehot> {
    using type = ttl::nn::ops::onehot;
};

template <>
struct cuda_op<ttl::nn::ops::zeros> {
    using type = ttl::nn::ops::zeros;
};

template <>
struct cuda_op<ttl::nn::ops::ones> {
    using type = ttl::nn::ops::ones;
};

template <typename R>
struct cuda_op<ttl::nn::ops::constant<R>> {
    using type = ttl::nn::ops::constant<R>;
};

template <>
struct cuda_op<ttl::nn::ops::uniform_constant> {
    using type = ttl::nn::ops::uniform_constant;
};

template <rank_t... rs>
struct cuda_op<ttl::nn::ops::copy_flatten<rs...>> {
    using type = ttl::nn::cuda::ops::copy_flatten<rs...>;
};

template <rank_t... rs>
struct cuda_op<ttl::nn::ops::grad::copy_flatten<rs...>> {
    using type = ttl::nn::cuda::ops::grad::copy_flatten<rs...>;
};

template <>
struct cuda_op<ttl::nn::ops::relu> {
    using type = ttl::nn::ops::relu;
};

template <>
struct cuda_op<ttl::nn::ops::add> {
    using type = ttl::nn::ops::add;
};

template <>
struct cuda_op<ttl::nn::ops::mul> {
    using type = ttl::nn::ops::mul;
};

template <>
struct cuda_op<ttl::nn::ops::axpy> {
    using type = ttl::nn::ops::axpy;
};

template <arity_t p>
struct cuda_op<ttl::nn::ops::grad::mul<p>> {
    using type = ttl::nn::cuda::ops::grad::mul<p>;
};

template <>
struct cuda_op<ttl::nn::ops::matmul> {
    using type = ttl::nn::ops::matmul;
};

template <typename image_order, typename filter_order>
struct cuda_op<ttl::nn::ops::conv<image_order, filter_order>> {
    using type = ttl::nn::ops::conv<image_order, filter_order>;
};

template <typename image_order>
struct cuda_op<ttl::nn::ops::add_bias<image_order>> {
    using type = ttl::nn::ops::add_bias<image_order>;
};

template <>
struct cuda_op<ttl::nn::ops::softmax> {
    using type = ttl::nn::ops::softmax;
};

template <>
struct cuda_op<ttl::nn::ops::argmax> {
    using type = ttl::nn::ops::argmax;
};

template <>
struct cuda_op<ttl::nn::ops::similarity> {
    using type = ttl::nn::ops::similarity;
};

template <>
struct cuda_op<ttl::nn::ops::xentropy> {
    using type = ttl::nn::ops::xentropy;
};

template <arity_t p>
struct cuda_op<ttl::nn::ops::grad::add_bias<ttl::nn::ops::hw, p>> {
    using type = ttl::nn::ops::grad::add_bias<ttl::nn::ops::hw, p>;
};

template <arity_t p>
struct cuda_op<ttl::nn::ops::grad::add_bias<ttl::nn::ops::nhwc, p>> {
    using type = ttl::nn::ops::grad::add_bias<ttl::nn::ops::nhwc, p>;
};

template <arity_t p>
struct cuda_op<ttl::nn::ops::grad::matmul<p>> {
    using type = ttl::nn::ops::grad::matmul<p>;
};

/*
template <arity_t p>
struct cuda_op<
    ttl::nn::ops::grad::conv<ttl::nn::traits::nhwc, ttl::nn::traits::rscd, p>> {
    using type = ttl::nn::ops::grad::conv<ttl::nn::traits::nhwc,
                                          ttl::nn::traits::rscd, p>;
};
*/

template <>
struct cuda_op<ttl::nn::ops::grad::relu<0>> {
    using type = ttl::nn::cuda::ops::grad::relu<0>;
};

template <>
struct cuda_op<ttl::nn::ops::grad::softmax<0>> {
    using type = ttl::nn::cuda::ops::grad::softmax<0>;
};

template <>
struct cuda_op<ttl::nn::ops::grad::xentropy<1>> {
    using type = ttl::nn::cuda::ops::grad::xentropy<1>;
};

}  // namespace ttl::nn
#endif
