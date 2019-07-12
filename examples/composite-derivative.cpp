/*
    u = x + y
    v = y + z
    w = u + v
 */

#include <nn/graph>
#include <nn/ops>
#include <ttl/tensor>

auto simple_auto(nn::graph::builder &b)
{
    const int n = 10;

    auto x = b.var<float>("x", ttl::make_shape(n));
    auto y = b.var<float>("y", ttl::make_shape(n));
    auto z = b.var<float>("z", ttl::make_shape(n));

    auto u = b.invoke<float>("u", nn::ops::add(), x, y);
    auto v = b.invoke<float>("v", nn::ops::add(), y, z);
    auto w = b.invoke<float>("w", nn::ops::add(), u, v);

    auto [gw, gvs] = b.gradients(w);
    for (auto [g, v] : gvs) {
        std::cerr << "(" << g->name() << ", " << v->name() << ")" << std::endl;
    }
    return w;
}

void example()
{
    nn::graph::builder b;
    simple_auto(b);
    b.debug();
    nn::graph::runtime rt;
    b.build(rt);
}

int main()
{
    example();
    return 0;
}
