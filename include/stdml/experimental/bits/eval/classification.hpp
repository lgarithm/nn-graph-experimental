#pragma once
#include <ttl/algorithm>
#include <ttl/tensor>

#include <stdml/experimental/bits/eval/binary_result.hpp>

namespace stdml::internal
{
template <typename Int = uint32_t>
class basic_classification_result
{
    basic_binary_results<Int> result;

  public:
    template <typename N>
    void add(const ttl::tensor_view<N, 1> &y, const ttl::tensor_view<N, 1> &y_)
    {
        Int failed = ttl::hamming_distance(y, y_);
        result.add(false, failed);
        result.add(true, std::get<0>(y.dims()) - failed);
    }

    Int positives() const { return result.positives(); }

    Int negatives() const { return result.negatives(); }

    template <typename R = float>
    R positive_percent() const
    {
        return result.template positive_percent<R>();
    }
};
}  // namespace stdml::internal
