/*
    u = x + y
    v = y + z
    w = u + v
 */

#include <ttl/nn/computation_graph>
#include <ttl/nn/ops>
#include <ttl/tensor>

DEFINE_TRACE_CONTEXTS;

auto simple_auto(ttl::nn::graph::builder &b)
{
    const int n = 10;

    auto x = b.var<float>("x", ttl::make_shape(n));
    auto y = b.var<float>("y", ttl::make_shape(n));
    auto z = b.var<float>("z", ttl::make_shape(n));

    auto u = b.invoke<float>("u", ttl::nn::ops::add(), x, y);
    auto v = b.invoke<float>("v", ttl::nn::ops::add(), y, z);
    auto w = b.invoke<float>("w", ttl::nn::ops::add(), u, v);

    auto gvs = b.gradients(w);
    for (auto [g, v] : gvs) {
        std::cerr << "(" << g->name() << ", " << v->name() << ")" << std::endl;
    }
    return w;
}

void example()
{
    ttl::nn::graph::builder b;
    simple_auto(b);
    ttl::nn::graph::runtime rt;
    b.build(rt);
}

int main()
{
    example();
    return 0;
}
