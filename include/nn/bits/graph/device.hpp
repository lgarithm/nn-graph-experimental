#pragma once
#include <ttl/bits/std_device.hpp>

namespace nn::graph::internal
{
struct cpu;
struct nvidia_gpu;

template <typename D> struct ttl_device;

template <> struct ttl_device<cpu> {
    using type = ttl::internal::host_memory;
};

template <> struct ttl_device<nvidia_gpu> {
    using type = ttl::internal::cuda_memory;
};
}  // namespace nn::graph::internal
