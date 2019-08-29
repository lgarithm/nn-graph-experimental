#include <nn/graph>
#include <nn/ops>
#include <ttl/tensor>

int main()
{
    nn::graph::builder b;

    auto x = b.covar<int>(ttl::make_shape());
    auto y = b.covar<int>(ttl::make_shape());
    auto z = b.invoke(nn::ops::add(), x, y);

    nn::graph::runtime rt;
    b.build(rt);
    {
        ttl::fill(x->get_ref(rt), 1);
        ttl::fill(y->get_ref(rt), 2);
    }
    b.run(rt, z);
    {
        auto v = z->get_view(rt);
        std::cerr << "z[0] = " << v.data()[0] << std::endl;
    }
    return 0;
}
