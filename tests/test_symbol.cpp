#include "testing.hpp"

#include <nn/bits/graph/symbol_manager.hpp>

TEST(symbol_test, test1)
{
    using nn::graph::internal::symbol_manager;
    symbol_manager sm;
    sm.create<float>(ttl::shape<3>(1, 2, 3));
    sm.create<float>(1, 2, 3);
}
