#include "testing.hpp"

#include <ttl/device>
#include <ttl/nn/bits/graph/variable_manager.hpp>

TEST(variable_manager_test, test1)
{
    ttl::nn::graph::internal::variable_manager<ttl::host_memory> vm;

    using ttl::nn::graph::internal::flat_shape;
    using ttl::nn::graph::internal::raw_tensor;
    using ttl::nn::graph::internal::tensor_symbol;
    tensor_symbol sym(tensor_symbol::type<int>(), flat_shape(1));
    raw_tensor<ttl::host_memory> t(sym.value_type(), sym.shape());

    vm.create_tensor(sym);
}
