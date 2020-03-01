#include <ttl/nn/computation_graph>
#include <ttl/nn/ops>
#include <ttl/tensor>

#include "utils.hpp"

DEFINE_TRACE_CONTEXTS;

int main()
{
    ttl::nn::graph::builder b;

    auto x = b.covar<float>("x", ttl::make_shape());
    auto y = b.covar<float>("y", ttl::make_shape());
    auto z = b.invoke<float>("z", ttl::nn::ops::add(), x, y);

    auto gvs = b.gradients(z);
    auto gs = firsts(gvs);

    ttl::nn::graph::runtime rt;
    b.build(rt);
    {
        ttl::fill(rt.ref(x), static_cast<float>(1.0));
        ttl::fill(rt.ref(y), static_cast<float>(2.0));
    }

    b.run(rt, gs);
    {
        auto gx = rt.view<float, 0>(gs.at(0));
        std::cerr << "gx[0] = " << gx.data()[0] << std::endl;
        auto gy = rt.view<float, 0>(gs.at(1));
        std::cerr << "gy[0] = " << gy.data()[0] << std::endl;
    }

    return 0;
}
