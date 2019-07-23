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
