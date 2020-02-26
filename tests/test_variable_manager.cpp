#include "testing.hpp"

#include <ttl/device>
#include <ttl/nn/bits/graph/variable_manager.hpp>

TEST(variable_manager_test, test1)
{
    ttl::nn::graph::internal::variable_manager<ttl::host_memory> vm;
    auto x = vm.create_tensor<float>(ttl::make_shape(2, 3));
    ASSERT_EQ(x->data_size(), static_cast<size_t>(24));

    using ttl::nn::graph::internal::flat_shape;
    using ttl::nn::graph::internal::raw_tensor;
    using ttl::nn::graph::internal::tensor_symbol;
    tensor_symbol sym(tensor_symbol::type<int>(), flat_shape(1));
    raw_tensor<ttl::host_memory> t(sym.value_type(), sym.shape());

    using ttl::nn::graph::internal::raw_tensor_variable;
    raw_tensor_variable<ttl::host_memory> v0(sym.value_type(), sym.shape());
    raw_tensor_variable<ttl::host_memory> v1(sym);
    vm.create_tensor(sym);
}
