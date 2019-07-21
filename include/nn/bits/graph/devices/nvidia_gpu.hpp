#pragma once
#include <ttl/cuda_tensor>

namespace nn::graph::internal
{
struct nvidia_gpu {
    template <typename R, ttl::rank_t r>
    using tensor_type = ttl::cuda_tensor<R, r>;

    template <typename R, ttl::rank_t r>
    using reference_type = ttl::cuda_tensor_ref<R, r>;

    template <typename R, ttl::rank_t r>
    using view_type = ttl::cuda_tensor_view<R, r>;
};
};  // namespace nn::graph::internal
