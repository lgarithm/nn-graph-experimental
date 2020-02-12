#pragma once
#include <iomanip>
#include <iostream>

// fatal error: function 'operator<<' is
//   not needed and will not be emitted [-Wunneeded-internal-declaration]
namespace _internal
{
class show_size_t
{
    size_t value_;

  public:
    show_size_t(size_t value) : value_(value) {}

    operator size_t() const { return value_; }
};

std::ostream &operator<<(std::ostream &os, const show_size_t &s)
{
    static constexpr size_t Ki = 1 << 10;
    static constexpr size_t Mi = 1 << 20;
    static constexpr size_t Gi = 1 << 30;
    const size_t value = static_cast<size_t>(s);
    if (s >= Gi) {
        os << value / Gi << "Gi";
    } else if (s >= Mi) {
        os << value / Mi << "Mi";
    } else if (s >= Ki) {
        os << value / Ki << "Ki";
    } else {
        os << value;
    }
    return os;
}
}  // namespace _internal

template <typename N>
_internal::show_size_t show_size(N n)
{
    return _internal::show_size_t(n);
}
