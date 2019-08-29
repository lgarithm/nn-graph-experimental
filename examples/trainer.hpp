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
    const int epoches_;
    const int batch_size_;
    const bool do_test_;

    using TestFunc = std::function<void(int epoch, int step)>;

  public:
    simple_trainer(int epoches, int batch_size, bool do_test = true)
        : epoches_(epoches), batch_size_(batch_size), do_test_(do_test)
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
            train_epoch(images, labels, [&](int idx, auto xs, auto y_s) {
                train(idx, xs, y_s);
                if (do_test_) { test(epoch, idx); }
            });
        }
    }
};
