#pragma once
#include <ttl/bits/std_device.hpp>

namespace ttl::internal
{
struct host_memory {
};
struct cuda_memory {
};
}  // namespace ttl::internal

namespace ttl::nn::graph::internal
{
using cpu = ttl::internal::host_memory;
using nvidia_gpu = ttl::internal::cuda_memory;

static constexpr cpu cpu0;
static constexpr nvidia_gpu nvidia0;

template <typename D>
struct default_device;

template <>
struct default_device<cpu> {
    static constexpr cpu value = cpu0;
};

template <>
struct default_device<nvidia_gpu> {
    static constexpr nvidia_gpu value = nvidia0;
};
}  // namespace ttl::nn::graph::internal
