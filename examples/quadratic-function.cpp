#include <nn/graph>
#include <nn/ops>
#include <ttl/tensor>

void example1()
{
    nn::graph::builder b;

    auto x = b.covar<float>(ttl::make_shape(), nn::ops::ones());
    auto y = b.invoke(nn::ops::mul(), x, x);

    nn::graph::optimizer opt;
    auto train_step = opt.minimize(b, y);

    nn::graph::runtime rt;
    b.build(rt);
    b.init(rt);

    float e = 1;
    for (int i = 0; i < 10; ++i) {
        e *= 0.8;
        std::cerr << "step = " << i << ", 0.8 ^ " << i + 1 << " = " << e
                  << std::endl;
        b.run(rt, train_step);
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
