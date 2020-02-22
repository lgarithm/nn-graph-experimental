#include "testing.hpp"

#include <nn/graph>

template <typename T1, typename T2>
std::vector<T1> firsts(const std::vector<std::pair<T1, T2>> &pairs)
{
    std::vector<T1> v(pairs.size());
    std::transform(pairs.begin(), pairs.end(), v.begin(),
                   [](auto p) { return p.first; });
    return v;
}

template <typename R, ttl::rank_t r>
void learn(const ttl::tensor_ref<R, r> &x, const ttl::tensor_view<R, r> &g,
           const R lr)
{
    ttl::tensor<R, 0> a;
    ttl::ref(a) = -lr;
    ttl::nn::ops::axpy()(x, ttl::view(a), g, ttl::view(x));
}

TEST(quad_test, test1)
{
    ttl::nn::graph::builder b;

    auto x = b.covar<float>("x", ttl::make_shape(), ttl::nn::ops::ones());
    auto y = b.invoke<float>("y", ttl::nn::ops::mul(), x, x);

    auto gvs = b.gradients(y);
    ASSERT_EQ(static_cast<int>(gvs.size()), 2);  // FIXME: merge gradients
    auto gs = firsts(gvs);
    // auto gx = gvs.at(0).first;

    ttl::nn::graph::runtime rt;
    b.build(rt);
    b.init(rt);

    float e = 1;
    for (int i = 0; i < 10; ++i) {
        e *= 0.8;
        b.run(rt, gs);

        for (auto &[g, v] : gvs) {
            learn<float>(
                ttl::flatten(rt.get_raw_ref(v).template typed<float>()),
                ttl::flatten(rt.get_raw_view(g).template typed<float>()), 0.1);
        }

        auto v = x->get_view(rt);
        ASSERT_FLOAT_EQ(v.data()[0], e);
    }
}
