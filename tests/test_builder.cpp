#include "testing.hpp"

#include <nn/graph>

TEST(builder_test, test1)
{
    ttl::nn::graph::builder b;
    b.var<float>("w", ttl::make_shape(100, 10));
    b.var<float>("b", ttl::make_shape(10));
}

TEST(builder_test, test_gradients)
{
    ttl::nn::graph::builder b;
    auto x = b.var<float>("x", ttl::make_shape());
    auto y = b.var<float>("y", ttl::make_shape());
    auto z = b.invoke<float>("z", ttl::nn::ops::add(), x, y);
    b.gradients(z);
}
