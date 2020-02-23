#include "testing.hpp"

#include <ttl/device>
#include <ttl/nn/bits/graph/variable_manager.hpp>

TEST(variable_manager_test, test1)
{
    ttl::nn::graph::internal::variable_manager<ttl::host_memory> vm;
    auto x = vm.create_tensor<float>(ttl::make_shape(2, 3));
    ASSERT_EQ(x->data_size(), static_cast<size_t>(24));
}
