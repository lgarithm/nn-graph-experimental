#include "testing.hpp"

#include <ttl/nn/bits/graph/symbol_manager.hpp>

TEST(symbol_test, test1)
{
    using ttl::nn::graph::internal::symbol_manager;
    symbol_manager sm;
    sm.create<float>(ttl::shape<3>(1, 2, 3));
    sm.create<float>(1, 2, 3);
}
