#pragma once
#include <nn/bits/ops/chunk.hpp>
#include <ttl/range>

class simple_data_iterator
{
    const int batch_size_;

  public:
    simple_data_iterator(int batch_size) : batch_size_(batch_size) {}

    template <typename Images, typename Labels, typename F>
    void operator()(const Images &images, const Labels &labels,
                    const F &f) const
    {
        const auto image_batches = ttl::chunk(images, batch_size_);
        const auto label_batches = ttl::chunk(labels, batch_size_);
        for (const auto idx : ttl::range<0>(image_batches)) {
            f(idx, image_batches[idx], label_batches[idx]);
        }
    }
};

class simple_trainer
{
    const std::string name_;

    const int epoches_;
    const int batch_size_;

    using TestFunc = std::function<void(int epoch, int step)>;

  public:
    simple_trainer(const std::string &name, int epoches, int batch_size)
        : name_(name), epoches_(epoches), batch_size_(batch_size)
    {
    }

    template <typename Images,
              typename Labels,  //
              typename TrainFunc>
    void operator()(const Images &images,
                    const Labels &labels,  //
                    const TrainFunc &train, const TestFunc &test) const
    {
        std::cerr << "batch size :: " << batch_size_ << std::endl;
        simple_data_iterator train_epoch(batch_size_);
        for (auto epoch : ttl::range(epoches_)) {
            TRACE_SCOPE(name_ + "::epoch");
            train_epoch(images, labels, [&](int idx, auto xs, auto y_s) {
                TRACE_SCOPE(name_ + "::step");
                train(idx, xs, y_s);
                bool do_test = true;
                if (do_test) {
                    TRACE_SCOPE(name_ + "::test");
                    test(epoch, idx);
                }
            });
        }
    }
};
