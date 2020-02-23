#pragma once
#include <stdml/control>
#include <ttl/range>
#include <ttl/tensor>

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
        int step = 0;
        for (auto epoch : ttl::range(epoches_)) {
            int idx = 0;
            stdml::batch_invoke(batch_size_,
                                [&](auto xs, auto y_s) {
                                    ++step;
                                    ++idx;
                                    printf("step: %4d:\n", step);
                                    train(idx, xs, y_s);
                                    if (do_test_) { test(epoch, idx); }
                                },
                                images, labels);
        }
        test(epoches_, 0);
    }
};
