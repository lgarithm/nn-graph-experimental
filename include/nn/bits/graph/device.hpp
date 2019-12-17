#pragma once
#include <ttl/bits/std_device.hpp>

namespace nn::graph::internal
{
using cpu = ttl::internal::host_memory;
using nvidia_gpu = ttl::internal::cuda_memory;
}  // namespace nn::graph::internal
