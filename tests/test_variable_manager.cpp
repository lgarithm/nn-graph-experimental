#include "testing.hpp"

#include <ttl/device>
#include <ttl/nn/bits/graph/variable_manager.hpp>

TEST(variable_manager_test, test1)
{
    ttl::nn::graph::internal::variable_manager<ttl::host_memory> vm;
    using TT = ttl::experimental::raw_type<ttl::experimental::idx_encoder>;

    using ttl::nn::graph::internal::flat_shape;
    using ttl::nn::graph::internal::raw_tensor;
    TT type(TT::type<int>(), flat_shape(1));

    raw_tensor<ttl::host_memory> t(type.value_type(), type.shape());
    vm.create_tensor(type);
}
