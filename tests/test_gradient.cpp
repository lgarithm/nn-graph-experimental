#include "testing.hpp"

#include <nn/graph>

TEST(gradient_test, test1)
{
    nn::graph::builder b;
    auto x = b.covar<float>("x", ttl::make_shape());
    auto y = b.covar<float>("y", ttl::make_shape());
    auto z = b.invoke<float>("z", nn::ops::add(), x, y);

    auto gvs = b.gradients(z);

    ASSERT_EQ(gvs.size(), static_cast<size_t>(2));
    ASSERT_EQ(gvs[0].second, x);
    ASSERT_EQ(gvs[1].second, y);
}

TEST(gradient_test, test2)
{
    nn::graph::builder b;

    const int n = 10;

    auto x = b.covar<float>("x", ttl::make_shape(n));
    auto y = b.covar<float>("y", ttl::make_shape(n));
    auto z = b.covar<float>("z", ttl::make_shape(n));

    auto u = b.invoke<float>("u", nn::ops::add(), x, y);
    auto v = b.invoke<float>("v", nn::ops::add(), y, z);
    auto w = b.invoke<float>("w", nn::ops::add(), u, v);

    auto gvs = b.gradients(w);

    ASSERT_EQ(gvs.size(), static_cast<size_t>(4));
}
