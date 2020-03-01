#include "testing.hpp"

#include <stdml/experimental/control>
#include <ttl/nn/computation_graph>

template <typename T1, typename T2>
std::vector<T1> firsts(const std::vector<std::pair<T1, T2>> &pairs)
{
    std::vector<T1> v(pairs.size());
    std::transform(pairs.begin(), pairs.end(), v.begin(),
                   [](auto p) { return p.first; });
    return v;
}

TEST(quad_test, test1)
{
    ttl::nn::graph::builder b;

    auto x = b.covar<float>("x", ttl::make_shape(), ttl::nn::ops::ones());
    auto y = b.invoke<float>("y", ttl::nn::ops::mul(), x, x);

    auto gvs = b.gradients(y);
    ASSERT_EQ(static_cast<int>(gvs.size()), 2);  // FIXME: merge gradients
    auto gs = firsts(gvs);

    ttl::nn::graph::runtime rt;
    b.build(rt);
    b.init(rt);

    float e = 1;
    for (int i = 0; i < 10; ++i) {
        e *= 0.8;
        b.run(rt, gs);
        stdml::internal::learn_all<float>(gvs, rt, 0.1);
        auto v = rt.view(x);
        ASSERT_FLOAT_EQ(v.data()[0], e);
    }
}
