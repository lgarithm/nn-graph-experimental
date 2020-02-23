#include "utils.hpp"
#include <stdml/experimental/control>
#include <ttl/nn/computation_graph>
#include <ttl/nn/ops>
#include <ttl/tensor>

DEFINE_TRACE_CONTEXTS;

void example1()
{
    ttl::nn::graph::builder b;

    auto x = b.covar<float>(ttl::make_shape(), ttl::nn::ops::ones());
    auto y = b.invoke(ttl::nn::ops::mul(), x, x);

    auto gvs = b.gradients(y);
    auto gs = firsts(gvs);

    ttl::nn::graph::runtime rt;
    b.build(rt);
    b.init(rt);

    float e = 1;
    for (int i = 0; i < 10; ++i) {
        e *= 0.8;
        std::cerr << "step = " << i << ", 0.8 ^ " << i + 1 << " = " << e
                  << std::endl;
        b.run(rt, gs);
        stdml::internal::learn_all<float>(gvs, rt, 0.1);
        {
            auto v = y->get_view(rt);
            std::cerr << "y = " << v.data()[0] << std::endl;
        }
        {
            auto v = x->get_view(rt);
            std::cerr << "x = " << v.data()[0] << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    example1();
    return 0;
}
