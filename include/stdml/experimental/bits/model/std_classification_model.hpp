#pragma once
#include <stdml/experimental/bits/model/std_supervised_model.hpp>
#include <ttl/nn/computation_graph>

namespace stdml::internal
{
template <typename Rx, typename ttl::rank_t r, typename N,
          typename D = ttl::host_memory>
class basic_classification_model : public basic_supervised_model<Rx, r, N, 0, D>
{
    using P = basic_supervised_model<Rx, r, N, 0, D>;

  protected:
    const N n_categories;

    using P::b;
    using P::gvs;
    using P::rt;

  public:
    basic_classification_model(const ttl::shape<r> &input, const N n_categories)
        : P(input, ttl::make_shape()), n_categories(n_categories)
    {
    }

    template <typename F>
    void init(const F &create_model, const int batch_size)
    {
        std::tie(this->samples_, this->labels_, this->outputs_, this->loss_) =
            create_model(b, this->sx, n_categories, batch_size);
        gvs = b.gradients(this->loss_->template typed<float, 1>());
        for (const auto &[g, v] : gvs) {
            printf("%s is the gradient of %s\n", g->name().c_str(),
                   v->name().c_str());
            this->train_step_ops.push_back(g);
        }
        b.build(rt);
        b.init(rt);
    }
};
}  // namespace stdml::internal
