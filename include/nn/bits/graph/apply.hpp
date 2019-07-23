#pragma once
#include <experimental/iterator>
#include <experimental/reflect>
#include <functional>
#include <sstream>

template <bool> struct maybe_apply;

template <> struct maybe_apply<false> {

    template <typename F, typename... Args>
    void operator()(const F &f, const std::tuple<Args...> &args) const
    {
        static constexpr auto arity = sizeof...(Args);
        std::array<std::string, arity> names(
            {demangled_type_info_name(typeid(Args))...});

        std::stringstream ss;
        ss << "(\n\t";
        std::copy(names.begin(), names.end(),
                  std::experimental::make_ostream_joiner(ss, "\n\t"));
        ss << "\n)";

        std::cerr << "can't apply " << demangled_type_info_name(typeid(F))
                  << " to " << ss.str() << std::endl;
    }
};

template <> struct maybe_apply<true> {
    template <typename F, typename... Args>
    void operator()(const F &f, const std::tuple<Args...> &args) const
    {
        std::apply(f, args);
    }
};

template <typename F, typename... Args>
void apply_if(const F &f, const std::tuple<Args...> &args)
{
    maybe_apply<std::is_invocable<F, Args...>::value>()(f, args);
}
