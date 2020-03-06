#pragma once
#include <array>

namespace stdml::internal
{
template <typename N = uint32_t>
class basic_binary_results
{
    std::array<N, 2> counter_;

    template <typename R>
    static R ratio(N a, N b)
    {
        if (b == 0) { return static_cast<R>(0); }
        return static_cast<R>(a) / static_cast<R>(b);
    }

    template <typename R, typename X, typename Y>
    static R percent(X a, Y b)
    {
        if (b == 0) { return static_cast<R>(0); }
        return static_cast<R>(100) * static_cast<R>(a) / static_cast<R>(b);
    }

  public:
    basic_binary_results() : basic_binary_results(0, 0) {}

    basic_binary_results(N failed, N succ) : counter_({failed, succ}) {}

    void add(bool ok, N n = 1) { counter_[ok] += n; }

    template <typename R = float>
    R accuracy() const
    {
        return ratio<R>(std::get<1>(counter_),
                        std::get<0>(counter_) + std::get<1>(counter_));
    }

    N total() const { return std::get<0>(counter_) + std::get<1>(counter_); }

    N positives() const { return std::get<1>(counter_); }

    N negatives() const { return std::get<0>(counter_); }

    template <typename R = float>
    R positive_percent() const
    {
        return percent<R>(positives(), total());
    }

    template <typename R = float>
    R negative_percent() const
    {
        return percent<R>(negatives(), total());
    }
};
}  // namespace stdml::internal
