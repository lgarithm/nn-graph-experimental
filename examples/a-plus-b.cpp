#include <ttl/nn/computation_graph>
#include <ttl/nn/ops>
#include <ttl/tensor>

DEFINE_TRACE_CONTEXTS;

int main()
{
    ttl::nn::graph::builder b;

    auto x = b.covar<int>(ttl::make_shape());
    auto y = b.covar<int>(ttl::make_shape());
    auto z = b.invoke(ttl::nn::ops::add(), x, y);

    ttl::nn::graph::runtime rt;
    b.build(rt);
    {
        ttl::fill(rt.ref(x), 1);
        ttl::fill(rt.ref(y), 2);
    }
    b.run(rt, z);
    {
        auto v = rt.view(z);
        std::cerr << "z[0] = " << v.data()[0] << std::endl;
    }
    return 0;
}
