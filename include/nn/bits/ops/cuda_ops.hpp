#pragma once
#include <experimental/reflect>

#include <nn/bits/graph/device.hpp>

namespace nn
{
namespace ops
{
template <typename F> class not_implemented_for_cuda
{
  public:
    template <typename... T> not_implemented_for_cuda(const T &...) {}

    template <typename... T> void operator()(const T &...) const
    {
        std::cerr << demangled_type_info_name(typeid(F))
                  << " not implemented for cuda!" << std::endl;
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

#ifdef ENABLE_CUDA
#include <nn/experimental/bits/ops/grad/bias.hpp>
#include <nn/experimental/bits/ops/grad/matmul.hpp>
#include <nn/experimental/bits/ops/grad/mul.hpp>
#include <nn/experimental/bits/ops/grad/softmax.hpp>
#include <nn/experimental/bits/ops/grad/xentropy.hpp>
#include <nn/experimental/bits/ops/utility.hpp>
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

template <int p> struct cuda_op<experimental::ops::grad::mul<p>> {
    using type = cuda::ops::grad::mul<p>;
};

template <typename E> struct cuda_op<ops::matmul_<E>> {
    using type = cuda::ops::matmul;
};

// template <typename image_order> struct cuda_op<ops::add_bias<image_order>> {
//     using type = cuda::ops::add_bias<image_order>;
// };

// FIXME: unify traits
template <> struct cuda_op<ops::add_bias<ops::hw>> {
    using type = cuda::ops::add_bias<hw>;
};

template <> struct cuda_op<ops::softmax> {
    using type = cuda::ops::softmax;
};

template <> struct cuda_op<experimental::ops::argmax> {
    using type = cuda::ops::argmax;
};

template <> struct cuda_op<experimental::ops::similarity> {
    using type = cuda::ops::similarity;
};

template <> struct cuda_op<ops::xentropy> {
    using type = cuda::ops::xentropy;
};

template <int p> struct cuda_op<experimental::ops::grad::add_bias<ops::hw, p>> {
    using type = cuda::ops::grad::add_bias<hw, p>;
};

template <int p, typename E>
struct cuda_op<experimental::ops::grad::matmul<p, E>> {
    using type = cuda::ops::grad::matmul<p>;
};

template <> struct cuda_op<experimental::ops::grad::softmax<0>> {
    using type = cuda::ops::grad::softmax<0>;
};

template <> struct cuda_op<experimental::ops::grad::xentropy<1>> {
    using type = cuda::ops::grad::xentropy<1>;
};

}  // namespace nn

#endif
