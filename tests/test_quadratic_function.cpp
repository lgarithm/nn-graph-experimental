#include "testing.hpp"

#include <nn/graph>

TEST(quad_test, test1)
{
    nn::graph::builder b;

    auto x = b.covar<float>("x", ttl::make_shape(), nn::ops::ones());
    auto y = b.invoke<float>("y", nn::ops::mul(), x, x);

    nn::graph::optimizer opt;
    auto f = opt.minimize(b, y);

    nn::graph::runtime rt;
    b.build(rt);
    b.init(rt);

    float e = 1;
    for (int i = 0; i < 10; ++i) {
        e *= 0.8;
        b.run(rt, f);
        auto v = x->get_view(rt);
        ASSERT_FLOAT_EQ(v.data()[0], e);
    }
}
