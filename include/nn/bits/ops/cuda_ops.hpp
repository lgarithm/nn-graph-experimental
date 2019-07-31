#pragma once
#include <experimental/reflect>

#include <nn/bits/graph/device.hpp>

namespace nn
{
namespace ops
{
template <typename F> class not_implemented_for_cuda
{
    mutable int cnt_;

  public:
    template <typename... T> not_implemented_for_cuda(const T &...) : cnt_(0) {}

    template <typename... T> void operator()(const T &...) const
    {
        if (cnt_ == 0) {
            std::cerr << demangled_type_info_name(typeid(F))
                      << " not implemented for cuda!" << std::endl;
        }
        ++cnt_;
    }
};
}  // namespace ops

template <typename F> struct cuda_op;

template <typename F, typename D> struct for_device;

template <typename F> struct for_device<F, nn::graph::internal::cpu> {
    using type = F;
};

template <typename F> struct for_device<F, nn::graph::internal::nvidia_gpu> {
    using type = typename cuda_op<F>::type;
};

template <typename F> struct cuda_op {
    using type = ops::not_implemented_for_cuda<F>;
};
}  // namespace nn

#ifdef NN_GRAPH_ENABLE_CUDA
#include <nn/bits/ops/gradients/bias.hpp>
#include <nn/bits/ops/gradients/matmul.hpp>
#include <nn/bits/ops/gradients/mul.hpp>
#include <nn/bits/ops/gradients/softmax.hpp>
#include <nn/bits/ops/gradients/xentropy.hpp>
#include <nn/ops>

#include <nn/cuda/gradients>
#include <nn/cuda/ops>

namespace nn
{

template <> struct cuda_op<ops::zeros> {
    using type = cuda::ops::zeros;
};

template <> struct cuda_op<ops::ones> {
    using type = cuda::ops::ones;
};

template <typename R> struct cuda_op<ops::constant<R>> {
    using type = cuda::ops::constant<R>;
};

template <> struct cuda_op<ops::uniform_distribution> {
    using type = cuda::ops::uniform_distribution;
};

template <> struct cuda_op<ops::add> {
    using type = cuda::ops::add;
};

template <> struct cuda_op<ops::mul> {
    using type = cuda::ops::mul;
};

template <> struct cuda_op<ops::axpy> {
    using type = cuda::ops::axpy;
};

template <int p> struct cuda_op<ops::grad::mul<p>> {
    using type = cuda::ops::grad::mul<p>;
};

template <typename E> struct cuda_op<ops::matmul_<E>> {
    using type = cuda::ops::matmul;
};

// template <> struct cuda_op<ops::conv<ops::nhwc, ops::rscd>> {
//     using type = cuda::ops::conv<nhwc, rscd>;
// };

template <typename image_order> struct cuda_op<ops::add_bias<image_order>> {
    using type = cuda::ops::add_bias<image_order>;
};

template <> struct cuda_op<ops::softmax> {
    using type = cuda::ops::softmax;
};

template <> struct cuda_op<ops::argmax> {
    using type = cuda::ops::argmax;
};

template <> struct cuda_op<ops::similarity> {
    using type = cuda::ops::similarity;
};

template <> struct cuda_op<ops::xentropy> {
    using type = cuda::ops::xentropy;
};

template <int p> struct cuda_op<ops::grad::add_bias<ops::hw, p>> {
    using type = cuda::ops::grad::add_bias<hw, p>;
};

template <int p> struct cuda_op<ops::grad::add_bias<ops::nhwc, p>> {
    using type = cuda::ops::grad::add_bias<nhwc, p>;
};

template <int p, typename E> struct cuda_op<ops::grad::matmul<p, E>> {
    using type = cuda::ops::grad::matmul<p>;
};

template <> struct cuda_op<ops::grad::softmax<0>> {
    using type = cuda::ops::grad::softmax<0>;
};

template <> struct cuda_op<ops::grad::xentropy<1>> {
    using type = cuda::ops::grad::xentropy<1>;
};

}  // namespace nn
#endif
