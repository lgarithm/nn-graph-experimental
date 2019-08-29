#pragma once
// #pragma message("std::experimental::zip is error-prone, use with care!")

#include <array>
#include <functional>
#include <numeric>
#include <tuple>

namespace std::experimental
{
namespace
{
template <typename... Ts> class zipper
{
    static constexpr auto arity = sizeof...(Ts);

    const tuple<const Ts &...> ranges_;

    template <typename... Iters> class iterator
    {
        tuple<Iters...> is_;

        template <size_t... Is> auto operator*(index_sequence<Is...>)
        {
            return make_tuple(*get<Is>(is_)...);
        }

        template <typename... P> static void noop(const P &...) {}

        template <typename Iter> int incr(Iter &i)
        {
            ++i;
            return 0;
        }

        template <size_t... Is> void _advance(index_sequence<Is...>)
        {
            noop(incr(get<Is>(is_))...);
        }

        template <size_t... Is>
        bool neq(index_sequence<Is...>, const iterator &p) const
        {
            // TODO: expand the expression
            array<bool, arity> neqs({(get<Is>(is_) != get<Is>(p.is_))...});
            return accumulate(neqs.begin(), neqs.end(), false,
                              logical_or<bool>());
        }

      public:
        iterator(const Iters &... i) : is_(i...) {}

        bool operator!=(const iterator &p) const
        {
            // return get<0>(is_) != get<0>(p.is_) || get<1>(is_) !=
            // get<1>(p.is_);
            return neq(make_index_sequence<arity>(), p);
        }

        void operator++()
        {
            _advance(make_index_sequence<arity>());
            // ++get<0>(is_), ++get<1>(is_);
        }

        auto operator*() { return (operator*)(make_index_sequence<arity>()); }
    };

    template <typename... Iters>
    static iterator<Iters...> make_iterator(const Iters &... is)
    {
        return iterator<Iters...>(is...);
    }

    template <size_t... Is> auto begin(index_sequence<Is...>) const
    {
        return make_iterator(get<Is>(ranges_).begin()...);
    }

    template <size_t... Is> auto end(index_sequence<Is...>) const
    {
        return make_iterator(get<Is>(ranges_).end()...);
    }

  public:
    zipper(const Ts &... ranges) : ranges_(ranges...) {}

    auto begin() const { return begin(make_index_sequence<arity>()); }

    auto end() const { return end(make_index_sequence<arity>()); }
};
}  // namespace

template <typename... Ts> zipper<Ts...> zip(const Ts &... ranges)
{
    return zipper<Ts...>(ranges...);
}
}  // namespace std::experimental
